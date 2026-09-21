# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence
import torch
from mmdet3d.structures.bbox_3d import (get_proj_mat_by_coord_type, points_cam2img, points_img2cam)
from torch import Tensor
from mmdet3d.models.layers.fusion_layers.coord_transform import apply_3d_transformation
from mmdet3d.registry import MODELS
from .pointpainting import Painting


@MODELS.register_module()
class PaintingFasterRCNN(Painting):
    """Multi-modality VoxelNet using Faster R-CNN."""

    def __init__(self, **kwargs):
        super(PaintingFasterRCNN, self).__init__(**kwargs)


@MODELS.register_module()
class DynamicPaintingRCNN(Painting):
    """Multi-modality VoxelNet using Faster R-CNN and dynamic voxelization."""

    def __init__(self, **kwargs):
        super(DynamicPaintingRCNN, self).__init__(**kwargs)
        self.coord = 'LIDAR'

    def extract_pts_feat(
            self,
            voxel_dict: Dict[str, Tensor],
            points: Optional[List[Tensor]] = None,
            img_feats: Optional[Sequence[Tensor]] = None,
            batch_input_metas: Optional[List[dict]] = None,
            images: Optional[Sequence[Tensor]] = None,
    ) -> Sequence[Tensor]:
        """Extract features of points.
        Args:
            voxel_dict(Dict[str, Tensor]): Dict of voxelization infos.
            points (List[tensor], optional):  Point cloud of multiple inputs.
            img_feats (list[Tensor], tuple[tensor], optional): Features from
                image backbone.
            batch_input_metas (list[dict], optional): The meta information
                of multiple samples. Defaults to True.

        Returns:
            Sequence[tensor]: points features of multiple inputs
            from backbone or neck.
        """
        if not self.with_pts_bbox:
            return None
        org_points = []
        pts_2ds = []
        for i in range(len(batch_input_metas)):
            proj_mat = get_proj_mat_by_coord_type(batch_input_metas[i], self.coord)
            org_point = apply_3d_transformation(points[i][:, :3], self.coord, batch_input_metas[i], reverse=True)
            pts_2d = points_cam2img(org_point, org_point.new_tensor(proj_mat))
            org_points.append(org_point)
            pts_2ds.append(pts_2d)
        # 铻嶅悎杩囧悗鐨勯潪绌轰綋绱犵壒寰?voxel_dict['voxels']涓巔oints銆?銆戜竴鏍?self.pts_voxel_encoder鐢ㄤ簬铻嶅悎浜岃€呯壒寰?        voxel_features, feature_coors, key_points = self.pts_voxel_encoder(
            voxel_dict['voxels'], voxel_dict['coors'], points, img_feats,
            batch_input_metas, images, org_points, pts_2ds)
        batch_size = voxel_dict['coors'][-1, 0] + 1
        if self.pts_middle_encoder.voxel_sa_layers is not None or self.pts_middle_encoder.bev_fusion is not None:
            x, sample_features = self.pts_middle_encoder(voxel_features, feature_coors, batch_size,
                                                         key_points['key_points'], key_points['seg_result'])
            key_points = dict(key_points=key_points, seg_result=sample_features)
        else:
            x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x, key_points

