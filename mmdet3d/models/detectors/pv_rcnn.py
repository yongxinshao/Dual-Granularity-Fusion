# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import InstanceList
from .two_stage import TwoStage3DDetector
from typing import Dict, List, Optional, Sequence
from torch import Tensor
from mmdet3d.structures.bbox_3d import (get_proj_mat_by_coord_type, points_cam2img, points_img2cam)
from torch import Tensor
from mmdet3d.models.layers.fusion_layers.coord_transform import apply_3d_transformation
import torch


@MODELS.register_module()
class PointVoxelRCNN(TwoStage3DDetector):
    r"""PointVoxelRCNN detector.

    Please refer to the `PointVoxelRCNN <https://arxiv.org/abs/1912.13192>`_.

    Args:
        voxel_encoder (dict): Point voxelization encoder layer.
        middle_encoder (dict): Middle encoder layer
            of points cloud modality.
        backbone (dict): Backbone of extracting points features.
        neck (dict, optional): Neck of extracting points features.
            Defaults to None.
        rpn_head (dict, optional): Config of RPN head. Defaults to None.
        points_encoder (dict, optional): Points encoder to extract point-wise
            features. Defaults to None.
        roi_head (dict, optional): Config of ROI head. Defaults to None.
        train_cfg (dict, optional): Train config of model.
            Defaults to None.
        test_cfg (dict, optional): Train config of model.
            Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`Det3DDataPreprocessor`. Defaults to None.
    """

    def __init__(self,
                 img_backbone: dict,
                 img_decodeHead: dict,
                 pts_voxel_encoder: dict,
                 voxel_encoder: dict,
                 middle_encoder: dict,
                 pts_backbone: dict,
                 neck: Optional[dict] = None,
                 rpn_head: Optional[dict] = None,
                 points_encoder: Optional[dict] = None,
                 roi_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None) -> None:
        super().__init__(
            backbone=pts_backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        # self.voxel_encoder = MODELS.build(voxel_encoder)
        self.coord = 'LIDAR'
        self.middle_encoder = MODELS.build(middle_encoder)
        self.pts_backbone = MODELS.build(pts_backbone)
        self.points_encoder = MODELS.build(points_encoder)
        if img_backbone:
            self.backbone = MODELS.build(img_backbone)
        if img_decodeHead is not None:
            self.decode_head = MODELS.build(img_decodeHead)

        if pts_voxel_encoder:
            self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self,
                       'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'backbone') and self.backbone is not None

    @property
    def with_img_decoder(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'decode_head') and self.decode_head is not None

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'voxels' keys.

                    - points (list[torch.Tensor]): Point cloud of each sample.
                    - voxels (dict[torch.Tensor]): Voxels of the batch sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats_dict,key_points = self.extract_img_feat(batch_inputs_dict, batch_input_metas)
        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(feats_dict,
                                                     batch_data_samples)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        batch_inputs_dict['key_points'] = key_points[0][:,:3].unsqueeze(0)
        # extrack points feats by points_encoder
        points_feats_dict = self.extract_points_feat(batch_inputs_dict,
                                                     feats_dict,
                                                     rpn_results_list)
        # visualize_point_cloud(points_feats_dict['keypoints'].cpu().numpy()[:,1:])
        # from vis_points.points_vis import visualize_point_cloud
        results_list_3d,score = self.roi_head.predict(points_feats_dict,
                                                rpn_results_list,
                                                batch_data_samples)

        # connvert to Det3DDataSample
        results_list = self.add_pred_to_datasample(batch_data_samples,
                                                   results_list_3d)

        return results_list

    # def extract_feat(self, batch_inputs_dict: dict) -> dict:
    #     """Extract features from the input voxels.
    #
    #     Args:
    #         batch_inputs_dict (dict): The model input dict which include
    #             'points', 'voxels' keys.
    #
    #             - points (list[torch.Tensor]): Point cloud of each sample.
    #             - voxels (dict[torch.Tensor]): Voxels of the batch sample.
    #
    #     Returns:
    #         dict: We typically obtain a dict of features from the backbone +
    #             neck, it includes:
    #
    #             - spatial_feats (torch.Tensor): Spatial feats from middle
    #                 encoder.
    #             - multi_scale_3d_feats (list[torch.Tensor]): Multi scale
    #                 middle feats from middle encoder.
    #             - neck_feats (torch.Tensor): Neck feats from neck.
    #     """
    #     feats_dict = dict()
    #     voxel_dict = batch_inputs_dict['voxels']
    #     voxel_features = self.voxel_encoder(voxel_dict['voxels'],
    #                                         voxel_dict['num_points'],
    #                                         voxel_dict['coors'])
    #     batch_size = voxel_dict['coors'][-1, 0].item() + 1
    #     feats_dict['spatial_feats'], feats_dict[
    #         'multi_scale_3d_feats'] = self.middle_encoder(
    #             voxel_features, voxel_dict['coors'], batch_size)
    #     x = self.backbone(feats_dict['spatial_feats'])
    #     if self.with_neck:
    #         neck_feats = self.neck(x)
    #         feats_dict['neck_feats'] = neck_feats
    #     return feats_dict
    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        feats_dict = dict()
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        feats_dict['spatial_feats'], feats_dict[
            'multi_scale_3d_feats'] = self.middle_encoder(
            voxel_features, voxel_dict['coors'], batch_size)
        x = self.pts_backbone(feats_dict['spatial_feats'])
        if self.with_neck:
            neck_feats = self.neck(x)
            feats_dict['neck_feats'] = neck_feats
        return feats_dict

    def extract_pts_feat(
            self,
            voxel_dict: Dict[str, Tensor],
            points: Optional[List[Tensor]] = None,
            img_feats: Optional[Sequence[Tensor]] = None,
            batch_input_metas: Optional[List[dict]] = None,
            images: Optional[Sequence[Tensor]] = None,
    ) -> Sequence[Tensor]:
        org_points = []
        pts_2ds = []
        feats_dict = dict()
        for i in range(len(batch_input_metas)):
            proj_mat = get_proj_mat_by_coord_type(batch_input_metas[i], self.coord)
            org_point = apply_3d_transformation(points[i][:, :3], self.coord, batch_input_metas[i], reverse=True)
            pts_2d = points_cam2img(org_point, org_point.new_tensor(proj_mat))
            org_points.append(org_point)
            pts_2ds.append(pts_2d)
        # 铻嶅悎杩囧悗鐨勯潪绌轰綋绱犵壒寰?voxel_dict['voxels']涓巔oints銆?銆戜竴鏍?self.pts_voxel_encoder鐢ㄤ簬铻嶅悎浜岃€呯壒寰?        voxel_features, feature_coors,key_points = self.pts_voxel_encoder(
            voxel_dict['voxels'], voxel_dict['coors'], points, img_feats,
            batch_input_metas, images, org_points, pts_2ds)
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        feats_dict['spatial_feats'], feats_dict[
            'multi_scale_3d_feats'] = self.middle_encoder(
            voxel_features, feature_coors, batch_size)
        x = self.pts_backbone(feats_dict['spatial_feats'])
        if self.with_neck:
            neck_feats = self.neck(x)
            feats_dict['neck_feats'] = neck_feats
        return feats_dict,key_points

    def extract_img_feat(self, batch_inputs_dict: dict,
                         batch_input_metas: List[dict]) -> tuple:
        voxel_dict = batch_inputs_dict.get('voxels', None)
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        org_imgs = batch_inputs_dict.get('paintimgs', None)
        # 鍥惧儚鐨凢PN鐗瑰緛
        # 璇箟鍒嗗壊鐓х墖

        img_feats = self._extract_img_feat(imgs, batch_input_metas)
        # 鏈€鍚庣殑鐗瑰緛  鍘熷鐐逛簯points锛屼互鍙?        # import numpy as np
        # yu_img = img_feats['output']
        # yu = np.uint8(torch.max(yu_img, dim=1)[1].permute(1, 2, 0)[...,0].cpu())

        pts_feats,key_points = self.extract_pts_feat(
            voxel_dict,
            points=points,
            img_feats=img_feats,
            batch_input_metas=batch_input_metas,
            images=org_imgs)
        return pts_feats,key_points

    def _extract_img_feat(self, img: Tensor, input_metas: List[dict]) -> dict:
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in input_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)  # 璁粌 416 1344 544 1760 楠岃瘉 384 1280

            # import numpy as np
            # import mmcv
            # mmcv.imshow(np.array(img[0].permute(1, 2, 0).cpu(), np.uint8))
            with torch.no_grad():
                self.backbone.eval()
                img_feats = self.backbone(img)
        else:
            return None
        if self.with_img_decoder:
            with torch.no_grad():
                self.decode_head.eval()
                img_feats_seg = self.decode_head(img_feats)
            # F.interpolate(img_feats, size, scale_factor, mode, align_corners)# 5涓?56*h*w鐨?self.img_neck鎶婃瘡涓€灞傜殑閫氶亾鏁版崲鎴?56
        return img_feats_seg

    def extract_points_feat(self, batch_inputs_dict: dict, feats_dict: dict,
                            rpn_results_list: InstanceList) -> dict:
        """Extract point-wise features from the raw points and voxel features.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'voxels' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - voxels (dict[torch.Tensor]): Voxels of the batch sample.
            feats_dict (dict): Contains features from the first stage.
            rpn_results_list (List[:obj:`InstanceData`]): Detection results
                of rpn head.

        Returns:
            dict: Contain Point-wise features, include:
                - keypoints (torch.Tensor): Sampled key points.
                - keypoint_features (torch.Tensor): Gather key points features
                    from multi input.
                - fusion_keypoint_features (torch.Tensor): Fusion
                    keypoint_features by point_feature_fusion_layer.
        """
        return self.points_encoder(batch_inputs_dict, feats_dict,
                                   rpn_results_list)

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'voxels' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - voxels (dict[torch.Tensor]): Voxels of the batch sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            dict: A dictionary of loss components.
        """
        ###鏀瑰姩

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats_dict,key_points = self.extract_img_feat(batch_inputs_dict, batch_input_metas)
        ############
        # feats_dict = self.extract_feat(batch_inputs_dict)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                feats_dict,
                rpn_data_samples,
                proposal_cfg=proposal_cfg,
                **kwargs)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in keys:
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            # TODO: Not support currently, should have a check at Fast R-CNN
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        batch_inputs_dict['key_points']=key_points
        points_feats_dict = self.extract_points_feat(batch_inputs_dict,
                                                     feats_dict,
                                                     rpn_results_list)

        roi_losses = self.roi_head.loss(points_feats_dict, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses

