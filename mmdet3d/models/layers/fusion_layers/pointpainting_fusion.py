# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union, Any, Optional
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
    """Fuse image features from multi-scale features.

    Args:
        img_channels (List[int] or int): Channels of image features.
            It could be a list if the input is multi-scale image features.
        pts_channels (int): Channels of point features
        mid_channels (int): Channels of middle layers
        out_channels (int): Channels of output fused features
        img_levels (List[int] or int): Number of image levels. Defaults to 3.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'. Defaults to 'LIDAR'.
        conv_cfg (:obj:`ConfigDict` or dict): Config dict for convolution
            layers of middle layers. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layers of middle layers. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or List[:obj:`Contigdict` or dict],
            optional): Initialization config dict. Defaults to None.
        activate_out (bool): Whether to apply relu activation to output
            features. Defaults to True.
        fuse_out (bool): Whether to apply conv layer to the fused features.
            Defaults to False.
        dropout_ratio (int or float): Dropout ratio of image features to
            prevent overfitting. Defaults to 0.
        aligned (bool): Whether to apply aligned feature fusion.
            Defaults to True.
        align_corners (bool): Whether to align corner when sampling features
            according to points. Defaults to True.
        padding_mode (str): Mode used to pad the features of points that do not
            have corresponding image features. Defaults to 'zeros'.
        lateral_conv (bool): Whether to apply lateral convs to image features.
            Defaults to True.
    """

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
                ) -> None:
        super(PaintingFusion, self).__init__(init_cfg=init_cfg)
        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)

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

    def forward(self, img_feats: List[Tensor], images: List[Tensor],
                coors_2d: List[Tensor], use_rgb: bool = False, use_score: bool = False,
                use_key_points: bool = False):
        seg_result = img_feats['output']
        mlvl_img_feats = []
        mlvl_img_mask = []
        mlvl_img_score = []
        for i in range(len(images)):
            feature, mask, no_soft_score = self.point_sample(
                img_features=seg_result[i],
                images=images[i],
                coors_2d=coors_2d[i],
                use_rgb=use_rgb,
                use_score=use_score,
                use_key_points=use_key_points
            )
            mlvl_img_feats.append(feature)
            mlvl_img_mask.append(mask)
            mlvl_img_score.append(no_soft_score)
        if use_rgb or use_score or use_key_points:
            return torch.cat(mlvl_img_feats, dim=0), mlvl_img_mask, mlvl_img_score
        else:
            return mlvl_img_feats, mlvl_img_mask, mlvl_img_score

    def point_sample(self,
                     img_features: Tensor,
                     images: Tensor,
                     coors_2d: Tensor,
                     use_rgb: bool = False,
                     use_score: bool = False,
                     use_key_points: bool = False
                     ):

        pts_2d = coors_2d
        img_coors = pts_2d[:, 0:2]
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
        aug = []
        score, no_soft_score = pointed_score(img_features, pts_2d)
        foreground_mask = score[:, 0] < 0.3
        if use_score or use_key_points:
            if use_score:
                aug.append(score)
            if use_key_points:
                key_points = (score[:, 0] < 0.3).to(torch.float32).to(score.device).unsqueeze(1)
                aug.append(key_points)
        if use_rgb:
            image_rgb = pointed_rgb(foreground_mask, images, pts_2d)
            aug.append(image_rgb)
        if use_rgb or use_score or use_key_points:
            aug = torch.cat(aug, dim=1)
            return aug, foreground_mask, no_soft_score
        return aug, foreground_mask, no_soft_score


def pointed_rgb(mask, org_img, coords_2d):
    image_rgb = org_img[:, coords_2d[:, 1], coords_2d[:, 0]].T
    image_rgb = image_rgb.to(torch.float32)
    image_rgb.to(torch.float32)[~mask] = torch.tensor([-123.675 / 58.395, -116.28 / 57.12, -103.53 / 57.375]).to(
        image_rgb.device)
    return image_rgb


def pointed_score(img_features, coords_2d):
    from PIL import Image
    output_permute = torch.tensor(img_features).permute(1, 2, 0)
    sf = torch.nn.Softmax(dim=1)
    # output_reassign_softmax = sf(output_permute)
    # data =  np.array(output_reassign_softmax[:, :, 2].cpu())
    # data1 = np.array(output_reassign_softmax[:, :, 0].cpu())
    # data2 = np.array(output_reassign_softmax[:, :, 1].cpu())
    # data3 = np.array(output_reassign_softmax[:, :, 3].cpu())
    # Image.fromarray(data).show()

    no_soft_score = output_permute[coords_2d[:, 1], coords_2d[:, 0]]
    score = sf(no_soft_score)
    return score, no_soft_score


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
    """Obtain image features using points.

    Args:
        voxel_features (Tensor): 1 x C x Nx x Ny x Nz voxel features.
        voxel_range (List[float]): The range of voxel features.
        voxel_size (List[float]): The voxel size of voxel features.
        depth_samples (Tensor): N depth samples in LiDAR coordinates.
        proj_mat (Tensor): ORIGINAL LiDAR2img projection matrix for N views.
        downsample_factor (int): The downsample factor in rescaling.
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

    Returns:
        Tensor: 1xCxDxHxW frustum features sampled from voxel features.
    """
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

