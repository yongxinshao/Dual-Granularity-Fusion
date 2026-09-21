# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
from mmcv.cnn.bricks import build_norm_layer
from mmdet.models.utils import multi_apply
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import nn as nn
from mmdet3d.structures.bbox_3d import (BaseInstance3DBoxes,
                                        DepthInstance3DBoxes,
                                        LiDARInstance3DBoxes)
from mmdet3d.registry import MODELS
from mmdet3d.utils import InstanceList
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple
from torch import Tensor


@MODELS.register_module()
class ForegroundSegmentation(BaseModule):
    """Foreground segmentation head.

    Args:
        in_channels (int): The number of input channel.
        mlp_channels (tuple[int]): Specify of mlp channels. Defaults
            to (256, 256).
        extra_width (float): Boxes enlarge width. Default used 0.1.
        norm_cfg (dict): Type of normalization method. Defaults to
            dict(type='BN1d', eps=1e-5, momentum=0.1).
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
        loss_seg (dict): Config of segmentation loss. Defaults to
            dict(type='mmdet.FocalLoss')
    """

    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            mlp_channels: Tuple[int] = (128, 128),
            extra_width: float = 0.1,
            norm_cfg: dict = dict(type='BN1d', eps=1e-5, momentum=0.1),
            init_cfg: Optional[dict] = None,
    ) -> None:
        super(ForegroundSegmentation, self).__init__(init_cfg=init_cfg)
        self.extra_width = extra_width
        self.num_classes = num_classes

        self.in_channels = in_channels

        out_channels = num_classes

        mlps_layers = []
        cin = in_channels
        for mlp in mlp_channels:
            mlps_layers.extend([
                nn.Linear(cin, mlp, bias=False),
                build_norm_layer(norm_cfg, mlp)[1],
                nn.ReLU()
            ])
            cin = mlp
        mlps_layers.append(nn.Linear(cin, out_channels, bias=True))

        self.seg_cls_layer = nn.Sequential(*mlps_layers)

    def forward(self, feats: torch.Tensor) -> dict:
        """Forward head.

        Args:
            feats (torch.Tensor): Point-wise features.

        Returns:
            dict: Segment predictions.
        """
        seg_preds = self.seg_cls_layer(feats)
        return dict(seg_preds=seg_preds)


@MODELS.register_module()
class seg_process(BaseModule):
    def __init__(self, extra_width=1.0, loss_seg: dict = dict(
        type='mmdet.FocalLoss',
        use_sigmoid=True,
        reduction='sum',
        gamma=2.0,
        alpha=0.25,
        activated=True,
        loss_weight=1.0), cls_loss: dict = None, num_classes=3):
        super(seg_process, self).__init__()
        self.extra_width = extra_width
        self.num_classes = num_classes
        self.use_sigmoid_cls = loss_seg.get('use_sigmoid', False)
        self.loss_seg = MODELS.build(loss_seg)
        self.cls_loss = MODELS.build(cls_loss)
        self.enlarge_width = 0.1
        # cls_loss=dict(
        #     type='mmdet.FocalLoss',
        #     use_sigmoid=True,
        #     reduction='sum',
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=1.0),

    def _get_targets_single(self, point_xyz: torch.Tensor,
                            gt_bboxes_3d: InstanceData,
                            gt_labels_3d: torch.Tensor) -> torch.Tensor:
        """generate segmentation targets for a single sample.

        Args:
            point_xyz (torch.Tensor): Coordinate of points.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth boxes in
                shape (box_num, 7).
            gt_labels_3d (torch.Tensor): Class labels of ground truths in
                shape (box_num).

        Returns:
            torch.Tensor: Points class labels.
        """
        point_cls_labels_single = point_xyz.new_zeros(
            point_xyz.shape[0]).long()
        enlarged_gt_boxes = gt_bboxes_3d.enlarged_box(self.extra_width)

        box_idxs_of_pts = gt_bboxes_3d.points_in_boxes_part(point_xyz).long()
        extend_box_idxs_of_pts = enlarged_gt_boxes.points_in_boxes_part(
            point_xyz).long()
        box_fg_flag = box_idxs_of_pts >= 0
        fg_flag = box_fg_flag.clone()
        ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
        point_cls_labels_single[ignore_flag] = -1
        gt_box_of_fg_points = gt_labels_3d[box_idxs_of_pts[fg_flag]]
        point_cls_labels_single[
            fg_flag] = 1 if self.num_classes == 1 else \
            gt_box_of_fg_points.long()
        return point_cls_labels_single,

    def get_targets(self, points_bxyz: torch.Tensor,
                    batch_gt_instances_3d: InstanceList) -> dict:
        """Generate segmentation targets.

        Args:
            points_bxyz (torch.Tensor): The coordinates of point in shape
                (B, num_points, 3).
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.

        Returns:
            dict: Prediction targets
                - seg_targets (torch.Tensor): Segmentation targets.
        """
        batch_size = len(batch_gt_instances_3d)
        points_xyz_list = []
        gt_bboxes_3d = []
        gt_labels_3d = []
        points_bxyz = points_bxyz['points']
        for idx in range(batch_size):
            gt_bboxes_3d.append(batch_gt_instances_3d[idx].bboxes_3d)
            gt_labels_3d.append(batch_gt_instances_3d[idx].labels_3d)
        seg_targets, = multi_apply(self._get_targets_single, points_bxyz,
                                   gt_bboxes_3d, gt_labels_3d)
        seg_targets = torch.cat(seg_targets, dim=0)
        return dict(seg_targets=seg_targets)

    def loss(self, semantic_results: dict,
             semantic_targets: dict) -> Dict[str, torch.Tensor]:
        """Calculate point-wise segmentation losses.

        Args:
            semantic_results (dict): Results from semantic head.
            semantic_targets (dict): Targets of semantic results.

        Returns:
            dict: Loss of segmentation.

            - loss_semantic (torch.Tensor): Segmentation prediction loss.
        """
        seg_preds = semantic_results['seg_preds']
        seg_targets = semantic_targets['seg_targets']

        positives = (seg_targets > 0)

        negative_cls_weights = (seg_targets == 0).float()
        seg_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        seg_weights /= torch.clamp(pos_normalizer, min=1.0)

        seg_preds = torch.sigmoid(seg_preds)
        loss_seg = self.loss_seg(seg_preds, (~positives).long(), seg_weights)
        return dict(loss_semantic=loss_seg)

    def get_targets_multi(self, points: List[Tensor],
                          batch_gt_instances_3d: InstanceList) -> Tuple[Tensor]:
        """Generate targets of PointRCNN RPN head.

        Args:
            points (list[torch.Tensor]): Points in one batch.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances_3d. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.

        Returns:
            tuple[torch.Tensor]: Targets of PointRCNN RPN head.
        """
        gt_labels_3d = [
            instances.labels_3d for instances in batch_gt_instances_3d
        ]
        gt_bboxes_3d = [
            instances.bboxes_3d for instances in batch_gt_instances_3d
        ]

        (mask_targets, positive_mask, negative_mask,
         ) = multi_apply(self.get_targets_single, points,
                         gt_bboxes_3d, gt_labels_3d)

        mask_targets = torch.cat(mask_targets, 0)
        positive_mask = torch.cat(positive_mask, 0)
        negative_mask = torch.cat(negative_mask, 0)
        return (mask_targets, positive_mask, negative_mask)

    def loss_by_feat(
            self,
            cls_preds: List[Tensor],
            points: List[Tensor],
            batch_gt_instances_3d: InstanceList,
            batch_input_metas: Optional[List[dict]] = None,
            batch_gt_instances_ignore: Optional[InstanceList] = None) -> Dict:
        """Compute loss.

        Args:
            bbox_preds (list[torch.Tensor]): Predictions from forward of
                PointRCNN RPN_Head.
            cls_preds N*classes.
            points (list[torch.Tensor]): Input points.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances_3d. It usually includes ``bboxes_3d`` and
                ``labels_3d`` attributes.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: Losses of PointRCNN RPN module.
        """
        targets = self.get_targets_multi(points, batch_gt_instances_3d)
        (mask_targets, positive_mask, negative_mask) = targets

        semantic_points = cls_preds
        semantic_targets = mask_targets
        semantic_targets[negative_mask] = self.num_classes
        semantic_points_label = semantic_targets
        # for ignore, but now we do not have ignored label
        semantic_loss_weight = negative_mask.float() + positive_mask.float()
        semantic_loss = self.cls_loss(semantic_points,
                                      semantic_points_label,
                                      semantic_loss_weight)
        semantic_loss /= positive_mask.float().sum()
        losses = dict(semantic_loss=semantic_loss)

        return losses

    def get_targets_single(self, points: Tensor,
                           gt_bboxes_3d: BaseInstance3DBoxes,
                           gt_labels_3d: Tensor) -> Tuple[Tensor]:
        """Generate targets of PointRCNN RPN head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)

        valid_gt = gt_labels_3d != -1
        gt_bboxes_3d = gt_bboxes_3d[valid_gt]
        gt_labels_3d = gt_labels_3d[valid_gt]

        # transform the bbox coordinate to the point cloud coordinate
        gt_bboxes_3d_tensor = gt_bboxes_3d.tensor.clone()
        gt_bboxes_3d_tensor[..., 2] += gt_bboxes_3d_tensor[..., 5] / 2

        points_mask, assignment = self._assign_targets_by_points_inside(
            gt_bboxes_3d, points)
        mask_targets = gt_labels_3d[assignment]

        positive_mask = (points_mask.max(1)[0] > 0)
        # add ignore_mask
        extend_gt_bboxes_3d = gt_bboxes_3d.enlarged_box(self.enlarge_width)
        points_mask, _ = self._assign_targets_by_points_inside(
            extend_gt_bboxes_3d, points)
        negative_mask = (points_mask.max(1)[0] == 0)
        return (mask_targets, positive_mask, negative_mask)

    def _assign_targets_by_points_inside(self, bboxes_3d: BaseInstance3DBoxes,
                                         points: Tensor) -> Tuple[Tensor]:
        """Compute assignment by checking whether point is inside bbox.

        Args:
            bboxes_3d (:obj:`BaseInstance3DBoxes`): Instance of bounding boxes.
            points (torch.Tensor): Points of a batch.

        Returns:
            tuple[torch.Tensor]: Flags indicating whether each point is
                inside bbox and the index of box where each point are in.
        """
        # TODO: align points_in_boxes function in each box_structures
        num_bbox = bboxes_3d.tensor.shape[0]
        if isinstance(bboxes_3d, LiDARInstance3DBoxes):
            assignment = bboxes_3d.points_in_boxes(points[:, 0:3]).long()
            points_mask = assignment.new_zeros(
                [assignment.shape[0], num_bbox + 1])
            assignment[assignment == -1] = num_bbox
            points_mask.scatter_(1, assignment.unsqueeze(1), 1)
            points_mask = points_mask[:, :-1]
            assignment[assignment == num_bbox] = num_bbox - 1
        elif isinstance(bboxes_3d, DepthInstance3DBoxes):
            points_mask = bboxes_3d.points_in_boxes(points)
            assignment = points_mask.argmax(dim=-1)
        else:
            raise NotImplementedError('Unsupported bbox type!')

        return points_mask, assignment

