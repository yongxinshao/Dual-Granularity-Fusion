# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union
from datetime import datetime
import mmcv
import torch
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from mmdet3d.registry import MODELS
from mmdet3d.structures.bbox_3d import (get_proj_mat_by_coord_type, points_cam2img, points_img2cam)
from mmdet3d.utils import OptConfigType, OptMultiConfig
from . import apply_3d_transformation
from mmdet3d.models.layers.fusion_layers.vis import Visualizer
from PIL import Image


def pointpainting_sample(img_meta: dict,
                         img_features: Tensor,
                         points: Tensor,
                         proj_mat: Tensor,
                         coord_type: str,
                         images: Tensor,
                         img_scale_factor: Tensor,
                         img_crop_offset: Tensor,
                         img_flip: bool,
                         img_pad_shape: Tuple[int],
                         img_shape: Tuple[int],
                         aligned: bool = True,
                         padding_mode: str = 'zeros',
                         align_corners: bool = True,
                         valid_flag: bool = False,
                         ) -> Tensor:
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        img_features (Tensor): 1 x C x H x W image features.
        points (Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (Tensor): Scale factor with shape of
            (w_scale, h_scale).
        img_crop_offset (Tensor): Crop offset used to crop image during
            data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (Tuple[int]): Int tuple indicates the h & w after
            padding. This is necessary to obtain features in feature map.
        img_shape (Tuple[int]): Int tuple indicates the h & w before padding
            after scaling. This is necessary for flipping coordinates.
        aligned (bool): Whether to use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool): Whether to align corners when
            sampling image features for each point. Defaults to True.
        valid_flag (bool): Whether to filter out the points that outside
            the image and with depth smaller than 0. Defaults to False.

    Returns:
        Tensor: NxC image features sampled by point coordinates.
    """

    # apply transformation based on info in img_meta
    # 杩樺師鍘熷鐐逛簯锛燂紵锛燂紵
    points = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True)

    # project points to image coordinate 寰楀埌鍘熷鐐逛簯瀵瑰簲鐨勫師濮嬪浘鍍忕殑浜岀淮鍧愭爣
    if valid_flag:
        proj_pts = points_cam2img(points, proj_mat, with_depth=True)
        pts_2d = proj_pts[..., :2]
        depths = proj_pts[..., 2]
    else:
        pts_2d = points_cam2img(points, proj_mat)

    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1
    # 缁忚繃缈昏浆鍚庡搴斿潗鏍囧彉涓簒 = w - x
    # if img_flip:
    #     # by default we take it as horizontal flip
    #     # use img_shape before padding for flip
    #     ori_h, ori_w = img_shape
    #     coor_x = ori_w - coor_x
    # 锛燂紵锛燂紵锛焎oor_y / h * 2 鎶婂潗鏍囧彉涓猴紙2锛?锛夌殑鑼冨洿鍚屾椂-1.鍒帮紙-1锛?锛?org-闅忔満灏哄-pad鍒?4鐨勫€嶆暟
    h, w = img_pad_shape
    norm_coor_y = coor_y / h * 2 - 1
    norm_coor_x = coor_x / w * 2 - 1
    grid = torch.cat([norm_coor_x, norm_coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    point_features = F.grid_sample(
        img_features.unsqueeze(0),
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCx1xN feats

    if valid_flag:
        # (N, )
        valid = (coor_x.squeeze() < w) & (coor_x.squeeze() > 0) & (
                coor_y.squeeze() < h) & (coor_y.squeeze() > 0) & (
                        depths > 0)
        valid_features = point_features.squeeze().t()
        valid_features[~valid] = 0
        return valid_features, valid  # (N, C), (N,)

    return point_features.squeeze().t()


@MODELS.register_module()
class PaintingFusion(BaseModule):

    def __init__(self,

                 img_channels: Union[List[int], int],
                 pts_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 img_levels: Union[List[int], int] = 3,
                 coord_type: str = 'LIDAR',
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 activate_out: bool = True,
                 fuse_out: bool = False,
                 dropout_ratio: Union[int, float] = 0,
                 aligned: bool = True,
                 align_corners: bool = True,
                 padding_mode: str = 'zeros',
                 lateral_conv: bool = True,
                 use_rgb: bool = False,
                 use_score: bool = False,
                 use_key_points=False) -> None:
        super(PaintingFusion, self).__init__(init_cfg=init_cfg)
        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)

        self.use_rgb = use_rgb
        self.use_score = use_score
        self.use_key_points = use_key_points

        self.img_levels = img_levels
        self.coord_type = coord_type
        self.act_cfg = act_cfg
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.dropout_ratio = dropout_ratio
        self.img_channels = img_channels
        self.aligned = aligned
        self.align_corners = align_corners
        self.padding_mode = padding_mode
        self.now_data = datetime.now()
        self.lateral_convs = None
        if lateral_conv:
            self.lateral_convs = nn.ModuleList()
            for i in range(len(img_channels)):
                l_conv = ConvModule(
                    img_channels[i],
                    mid_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                self.lateral_convs.append(l_conv)  # 5涓?56--->128鐨勫嵎绉?  鐒跺悗self.img_transform--lines5*128--->128
            self.img_transform = nn.Sequential(
                nn.Linear(mid_channels * len(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        else:
            self.img_transform = nn.Sequential(
                nn.Linear(sum(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        self.pts_transform = nn.Sequential(  # self.pts_transform lines 64---銆?28
            nn.Linear(pts_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        )

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Linear(mid_channels, out_channels),
                # For pts the BN is initialized differently by default
                # TODO: check whether this is necessary
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=False))

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                dict(type='Xavier', layer='Linear', distribution='uniform')
            ]

    def forward(self, img_feats: List[Tensor], pts: List[Tensor],
                pts_feats: Tensor, img_metas: List[dict], images: List[Tensor]) -> Tensor:
        # 鑾峰緱鐐逛簯涓庡浘鍍忓尮閰嶅悗锛屽浘鍍忕殑鐗瑰緛 N锛?*128
        if self.use_key_points or self.use_score or self.use_rgb:
            img_pts = self.obtain_mlvl_feats(img_feats, pts, img_metas, images)
        # 5*128--->128
            fusion_feature = torch.cat((pts_feats, img_pts), dim=1)

            return fusion_feature
        else:
            return None

    def obtain_mlvl_feats(self, img_feats: [str, Tensor], pts: List[Tensor],
                          img_metas: List[dict], images: List[Tensor]) -> Tensor:
        # if self.lateral_convs is not None:
        #     img_ins = [
        #         lateral_conv(img_feats[i])
        #         for i, lateral_conv in zip(self.img_levels, self.lateral_convs)  # FPN 5灞?56--->5灞?28
        #     ]
        # else:
        #     img_ins = img_feats
        img_feats_per_point = []
        # Sample multi-level features
        if len(img_feats['output'].size()) == 2:
            img_feats = img_feats['output'].unsqueeze(0)
        for i in range(len(img_metas)):
            # 鐐规墍瀵瑰簲鐨勬姇褰辩壒寰? 姣忎竴灞傚搴?28涓壒寰?            mlvl_img_feats = []
            mlvl_img_feats.append(
                self.sample_single(img_feats['seg_feature'][i], img_feats['low_cat'][i], img_feats['output'][i],
                                   pts[i][:, :3], img_metas[i], images[i]))
            mlvl_img_feats = torch.cat(mlvl_img_feats, dim=-1)
            img_feats_per_point.append(mlvl_img_feats)

        img_pts = torch.cat(img_feats_per_point, dim=0)
        return img_pts

    def sample_single(self, seg_fests: Tensor, cat_feats: Tensor, img_feats: Tensor, pts: Tensor,
                      img_meta: dict, images: Tensor) -> Tensor:
        # TODO: image transformation also extracted
        img_scale_factor = (
            pts.new_tensor(img_meta['scale_factor'][:2])
            if 'scale_factor' in img_meta.keys() else 1)
        img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
        img_crop_offset = (
            pts.new_tensor(img_meta['img_crop_offset'])
            if 'img_crop_offset' in img_meta.keys() else 0)
        proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
        seg_fests = seg_fests
        cat_feats = cat_feats
        img_pts = self.point_sample(
            fes_seg=seg_fests,
            fes_cat=cat_feats,
            img_meta=img_meta,
            img_features=img_feats,
            points=pts,
            proj_mat=pts.new_tensor(proj_mat),
            coord_type=self.coord_type,
            img_scale_factor=img_scale_factor,
            images=images,
        )
        return img_pts

    def point_sample(self,
                     fes_seg: Tensor,
                     fes_cat: Tensor,
                     img_meta: dict,
                     img_features: Tensor,
                     points: Tensor,
                     proj_mat: Tensor,
                     coord_type: str,
                     images: Tensor,
                     img_scale_factor: Tensor,
                     valid_flag: bool = False,
                     ) -> Tensor:
        # vis = Visualizer()
        # vis.visuallize_pointcloud(np.array(points.cpu()))
        # 杩樺師鍘熷鐐逛簯锛燂紵锛燂紵
        points1 = points
        points = apply_3d_transformation(points, coord_type, img_meta, reverse=True)
        # project points to image coordinate 寰楀埌鍘熷鐐逛簯瀵瑰簲鐨勫師濮嬪浘鍍忕殑浜岀淮鍧愭爣
        if valid_flag:
            proj_pts = points_cam2img(points, proj_mat, with_depth=True)
            pts_2d = proj_pts[..., :2]
            depths = proj_pts[..., 2]
        else:
            pts_2d = points_cam2img(points, proj_mat)
        img_coors = pts_2d[:, 0:2] * img_scale_factor
        coor_x, coor_y = torch.split(img_coors, 1, dim=1)
        pts_2d = torch.cat((coor_x, coor_y), dim=1)
        # image_seg = torch.tensor(img_features)  0 backbone 1 person 2 car 3 bicycle
        pts_2d = pts_2d.to(torch.int32)
        filter = 0
        # mask to remove any point that has x or y > image coord
        x_mask = pts_2d[:, 0] > 1280 - 1
        y_mask = pts_2d[:, 1] > 384 - 1
        # mask to remove neg values
        neg_mask = pts_2d < 0
        neg_mask = torch.any(neg_mask, dim=1)
        pts_2d[x_mask] = filter
        pts_2d[y_mask] = filter
        pts_2d[neg_mask] = filter
        # 澶氬崱鍔犺浇鍒癎PU
        aug = []
        if self.use_score or self.use_key_points:
            score = pointed_score(img_features, pts_2d)
            if self.use_score:
                aug.append(score)
            if self.use_key_points:
                key_points = (score[:, 0] < 0.3).to(torch.float32).to(score.device).unsqueeze(1)
                aug.append(key_points)
        if self.use_rgb:
            image_rgb = pointed_rgb(img_features, images, pts_2d)
        aug.append(image_rgb)

        # vis_points(score, points, images, pts_2d)
        aug = torch.cat(aug, dim=1)
        # Image.fromarray(np.array(image_seg.cpu()*50, np.uint8)).show()
        # image_rgb = image_bgr[:, [2, 1, 0]]
        # vis.visuallize_pointcloud(np.array(points[:, :3].cpu()), np.array(image_rgb.cpu()))
        # painted_pointcloud = torch.hstack((points[:, :3], image_rgb))  # (N, 4)  (final - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
        current_time = datetime.now()
        # 鑾峰彇褰撳墠鏃堕棿
        if (current_time - self.now_data).total_seconds() / 20 >= 1:
            print('淇濆瓨鍥剧墖鎴愬姛')
            self.now_data = current_time
            file = current_time.strftime('%Y%m%d_%H%M%S')
            file_name = img_meta['img_path'].split('/')[-1].split('.png')[0]
            filename = f'{file_name}_{file}.pt'
            image_rgb[:, 0] = image_rgb[:, 0] * 58.395 + 123.675
            image_rgb[:, 1] = image_rgb[:, 1] * 57.12 + 116.28
            image_rgb[:, 2] = image_rgb[:, 2] * 57.375 + 103.53
            torch.save(torch.hstack((points[:, :3], image_rgb)), filename)
        return aug


def pointed_rgb(img_features, org_img, coords_2d):
    image_seg = img_features.argmax(dim=0).squeeze().detach()
    semantic_channel = image_seg[coords_2d[:, 1], coords_2d[:, 0]].reshape(-1, 1)
    image_rgb = org_img[:, coords_2d[:, 1], coords_2d[:, 0]].T
    mask = (semantic_channel == 0).squeeze(1)
    image_rgb = image_rgb.to(torch.float32)
    image_rgb.to(torch.float32)[mask] = torch.tensor([-123.675 / 58.395, -116.28 / 57.12, -103.53 / 57.375]).to(
        image_rgb.device)
    return image_rgb


def vis_points(score, points, org_img, coords_2d):
    vis = Visualizer()
    mask = score[:, 0] < 0.3
    mask = mask.cpu().numpy()
    image_rgb = org_img[:, coords_2d[:, 1], coords_2d[:, 0]].T
    image_rgb = image_rgb.cpu().numpy()
    image_rgb[mask] = [255, 0, 0]
    image_rgb[~mask] = [0, 0, 0]
    vis.visuallize_pointcloud(points.cpu().numpy(), image_rgb)


def pointed_score(img_features, coords_2d):
    from PIL import Image
    output_permute = torch.tensor(img_features).permute(1, 2, 0)
    sf = torch.nn.Softmax(dim=2)
    output_reassign_softmax = sf(output_permute)
    # data =  np.array(output_reassign_softmax[:, :, 2].cpu())
    # data1 = np.array(output_reassign_softmax[:, :, 0].cpu())
    # data2 = np.array(output_reassign_softmax[:, :, 1].cpu())
    # data3 = np.array(output_reassign_softmax[:, :, 3].cpu())
    # Image.fromarray(data).show()

    score = output_reassign_softmax[coords_2d[:, 1], coords_2d[:, 0]]
    return score


def voxel_sample(voxel_features: Tensor,
                 voxel_range: List[float],
                 voxel_size: List[float],
                 depth_samples: Tensor,
                 proj_mat: Tensor,
                 downsample_factor: int,
                 img_scale_factor: Tensor,
                 img_crop_offset: Tensor,
                 img_flip: bool,
                 img_pad_shape: Tuple[int],
                 img_shape: Tuple[int],
                 aligned: bool = True,
                 padding_mode: str = 'zeros',
                 align_corners: bool = True) -> Tensor:
    # construct frustum grid
    device = voxel_features.device
    h, w = img_pad_shape
    h_out = round(h / downsample_factor)
    w_out = round(w / downsample_factor)
    ws = (torch.linspace(0, w_out - 1, w_out) * downsample_factor).to(device)
    hs = (torch.linspace(0, h_out - 1, h_out) * downsample_factor).to(device)
    depths = depth_samples[::downsample_factor]
    num_depths = len(depths)
    ds_3d, ys_3d, xs_3d = torch.meshgrid(depths, hs, ws)
    # grid: (D, H_out, W_out, 3) -> (D*H_out*W_out, 3)
    grid = torch.stack([xs_3d, ys_3d, ds_3d], dim=-1).view(-1, 3)
    # recover the coordinates in the canonical space
    # reverse order of augmentations: flip -> crop -> scale
    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        ori_h, ori_w = img_shape
        grid[:, 0] = ori_w - grid[:, 0]
    grid[:, :2] += img_crop_offset
    grid[:, :2] /= img_scale_factor
    # grid3d: (D*H_out*W_out, 3) in LiDAR coordinate system
    grid3d = points_img2cam(grid, proj_mat)
    # convert the 3D point coordinates to voxel coordinates
    voxel_range = torch.tensor(voxel_range).to(device).view(1, 6)
    voxel_size = torch.tensor(voxel_size).to(device).view(1, 3)
    # suppose the voxel grid is generated with AlignedAnchorGenerator
    # -0.5 given each grid is located at the center of the grid
    # TODO: study whether here needs -0.5
    grid3d = (grid3d - voxel_range[:, :3]) / voxel_size - 0.5
    grid_size = (voxel_range[:, 3:] - voxel_range[:, :3]) / voxel_size
    # normalize grid3d to (-1, 1)
    grid3d = grid3d / grid_size * 2 - 1
    # (x, y, z) -> (z, y, x) for grid_sampling
    grid3d = grid3d.view(1, num_depths, h_out, w_out, 3)[..., [2, 1, 0]]
    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    frustum_features = F.grid_sample(
        voxel_features,
        grid3d,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCxDxHxW feats

    return frustum_features

