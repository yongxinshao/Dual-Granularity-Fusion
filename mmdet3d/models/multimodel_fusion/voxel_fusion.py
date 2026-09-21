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
class feature_voxel_fusion(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 init_cfg: OptMultiConfig = None,
                 multi_model_fusion: dict = None,
                 voxel_size: Tuple[float] = (0.2, 0.2, 4),
                 point_cloud_range: Tuple[float] = (0, -40, -3, 70.4, 40, 1),
                 pool_out: list = [32, 128],
                 ) -> None:
        super(feature_voxel_fusion, self).__init__(init_cfg=init_cfg)
        self.point_cloud_range = point_cloud_range
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]

        self.conv_first = nn.Sequential(
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d', eps=1e-5, momentum=0.1),
                bias='auto'))

        self.line1 = nn.Sequential(  # self.pts_transform lines 64---銆?28
            nn.Linear(out_channels, pool_out[0]),
            nn.BatchNorm1d(pool_out[0], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.line2 = nn.Sequential(  # self.pts_transform lines 64---銆?28
            nn.Linear(out_channels, pool_out[1]),
            nn.BatchNorm1d(pool_out[1], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),

        )

        self.line3 = nn.Sequential(  # self.pts_transform lines 64---銆?28
            nn.Linear(out_channels, pool_out[0]),
            nn.BatchNorm1d(pool_out[0], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.line4 = nn.Sequential(  # self.pts_transform lines 64---銆?28
            nn.Linear(out_channels, pool_out[1]),
            nn.BatchNorm1d(pool_out[1], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),

        )

        self.cluster_scatter_mean = DynamicScatter(voxel_size, point_cloud_range, average_points=True)

    def forward(self, img_feats: Tensor, coors: Tensor, pts_2d, point_feats=None, mask=None):
        out = self.conv_first(img_feats['low_cat'])
        scores = img_feats['output']
        sf = torch.nn.Softmax(dim=1)
        scores = sf(scores)[:, 0:1, ...]
        offsets = torch.tensor([
            [1, 1], [1, 0], [1, -1],
            [0, 1], [0, -1], [0, 0],
            [-1, 1], [-1, 0], [-1, -1]
        ]).to(out.device)
        img_feats_per_point = []
        max_pools = []
        for i in range(out.shape[0]):
            neighborhood = pts_2d[i].unsqueeze(1) + offsets.unsqueeze(0)
            cat_img = [out[i]] * 9
            cat_score = [scores[i]] * 9
            result_img = torch.stack(cat_img, dim=0)
            result_score = torch.stack(cat_score, dim=0)
            img_feature, n_img_feature, score = bilinear_fusion(result_img, neighborhood.permute(1, 0, 2), result_score)
            # max_pool = torch.max(n_img_feature, 1)[0]
            avg_pool = torch.mean(n_img_feature, dim=1)
            img_feats_per_point.append(img_feature)
            max_pools.append(avg_pool)
        img_pts = torch.cat(img_feats_per_point, dim=0)
        img_pool_fea = torch.cat(max_pools, dim=0)
        layer1=None
        layer2 = None
        # voxel_mean, mean_coors = self.cluster_scatter_mean(img_pts, coors)
        # feas = torch.cat((img_pts, img_pool_fea), dim=1)
        # layer1 = self.line1(voxel_mean)
        # layer2 = self.line2(voxel_mean)

        layer3 = self.line3(img_pts)
        layer4 = self.line4(img_pool_fea)
        return [layer1, layer2], [layer3, layer4]

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


def fusion(img, pts_2d):
    img_coors = pts_2d[:, 0:2]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)
    norm_coor_y = coor_y / 384 * 2 - 1
    norm_coor_x = coor_x / 1280 * 2 - 1
    grid = torch.cat([norm_coor_x, norm_coor_y], dim=1).unsqueeze(0).unsqueeze(0)
    mode = 'bilinear'
    point_features = F.grid_sample(
        img.unsqueeze(0),
        grid,
        mode=mode,
        align_corners=True)
    return point_features.squeeze().t()


def bilinear_fusion(img, pts_2d, cat_score):
    img_coors = pts_2d[:, :, 0:2]
    coor_x, coor_y = torch.split(img_coors, 1, dim=2)
    norm_coor_y = coor_y / 384 * 2 - 1
    norm_coor_x = coor_x / 1280 * 2 - 1
    grid = torch.cat([norm_coor_x, norm_coor_y], dim=2).unsqueeze(1)
    mode = 'bilinear'
    point_features = F.grid_sample(
        img,
        grid,
        mode=mode,
        align_corners=True)
    point_score = F.grid_sample(
        cat_score,
        grid,
        mode=mode,
        align_corners=True)
    n_points_fes = point_features.squeeze(2).permute(2, 0, 1)
    points_fes = n_points_fes[:, 5, :]
    scores_fes = 1 - point_score.squeeze(2).permute(2, 0, 1)
    return points_fes, n_points_fes * scores_fes, scores_fes

