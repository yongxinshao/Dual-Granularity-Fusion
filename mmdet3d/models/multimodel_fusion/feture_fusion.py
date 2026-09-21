from typing import List, Tuple, Union
import mmcv
import numpy as np
import torch
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
from mmcv.ops import DynamicScatter
from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d import (get_proj_mat_by_coord_type,
                                        points_cam2img, points_img2cam)
from mmdet3d.utils import OptConfigType, OptMultiConfig
from mmdet3d.models.layers.fusion_layers.coord_transform import apply_3d_transformation
from mmdet3d.models.layers.fusion_layers.vis import Visualizer


def vis_point(image_mete, pts_2d, points, point):
    import mmcv
    image = mmcv.imread(image_mete['img_path'])
    image = mmcv.bgr2rgb(image)
    mmcv.imshow(image)
    pts_2d = np.array(pts_2d.cpu(), np.uint16)
    bgr = image[pts_2d[:, 1], pts_2d[:, 0], :]
    vis = Visualizer()
    vis.visuallize_pointcloud(np.array(points.cpu()), bgr)
    print(1)


@MODELS.register_module()
class FeatureFusion(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 add_out,
                 cat_out,
                 is_add: bool = True,
                 init_cfg: OptMultiConfig = None,
                 use_foreground_features: bool = True,
                 multi_model_fusion: dict = None,
                 voxel_size: Tuple[float] = (0.2, 0.2, 4),
                 point_cloud_range: Tuple[float] = (0, -40, -3, 70.4, 40, 1),
                 ) -> None:
        super(FeatureFusion, self).__init__(init_cfg=init_cfg)
        self.point_cloud_range = point_cloud_range
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            3,
            padding=1,
            inplace=False)
        self.is_add = is_add
        self.foreground_features = use_foreground_features
        if self.is_add:
            self.line1 = nn.Sequential(  # self.pts_transform lines 64---銆?28
                nn.Linear(out_channels, add_out),
                nn.BatchNorm1d(add_out, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            )
        else:
            self.line1 = nn.Sequential(  # self.pts_transform lines 64---銆?28
                nn.Linear(out_channels * 2, cat_out),
                nn.BatchNorm1d(cat_out, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),

            )
        self.coord = 'LIDAR'
        self.multi_model_fusion = None
        self.cluster_scatter_mean = DynamicScatter(voxel_size, point_cloud_range, average_points=True)
        if multi_model_fusion is not None:
            self.multi_model_fusion = MODELS.build(multi_model_fusion)

    def fusion(self, img, point, img_meta, proj_mat):
        points = apply_3d_transformation(
            point, self.coord, img_meta, reverse=True)
        # vis = Visualizer()
        # vis.visuallize_pointcloud(np.array(points.cpu()))
        # image = img_meta[0]['img_path']

        pts_2d = points_cam2img(points, proj_mat)
        img_coors = pts_2d[:, 0:2]
        # vis_point(img_meta, pts_2d, points, point)

        coor_x, coor_y = torch.split(img_coors, 1, dim=1)
        norm_coor_y = coor_y / 384 * 2 - 1
        norm_coor_x = coor_x / 1280 * 2 - 1
        grid = torch.cat([norm_coor_x, norm_coor_y], dim=1).unsqueeze(0).unsqueeze(0)
        mode = 'bilinear'
        point_features = F.grid_sample(
            img,
            grid,
            mode=mode,
            align_corners=True)
        return point_features.squeeze().t()

    def forward(self, img_feats: List[Tensor], point_feats: Tensor, coors: Tensor, pts: List[Tensor],
                img_metas: List[dict], mask: List) -> Tensor:

        mask = torch.cat(mask, dim=0)
        out = self.conv1(img_feats)
        img_feats_per_point = []
        # TODO: image transformation also extracted
        # Sample multi-level features
        for i in range(len(img_metas)):
            proj_mat = get_proj_mat_by_coord_type(img_metas[i], self.coord)
            img_feats_per_point.append(
                self.fusion(out[i].unsqueeze(0), pts[i][:, :3], img_metas[i], pts[i].new_tensor(proj_mat)))

        img_pts = torch.cat(img_feats_per_point, dim=0)

        if self.foreground_features:
            img_pts[~mask] = 0
        if self.is_add:
            fusion_features = point_feats + img_pts
            fusion_features = self.line1(fusion_features)
        else:
            fusion_features = self.multi_model_fusion(point_feats, img_pts)
        return fusion_features

    def map_voxel_center_to_point(self, pts_coors: Tensor, voxel_mean: Tensor,
                                  voxel_coors: Tensor) -> Tensor:
        canvas_z = int(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        canvas_y = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        # canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        # Only include non-empty pillars
        indices = (
                voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
                voxel_coors[:, 1] * canvas_y * canvas_x +
                voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device)

        # Step 2: get voxel mean for each point
        voxel_index = (
                pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
                pts_coors[:, 1] * canvas_y * canvas_x +
                pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point

