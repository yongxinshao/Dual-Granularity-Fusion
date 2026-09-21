# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple

import torch
from mmcv.cnn import build_norm_layer
from mmcv.ops import DynamicScatter
from torch import Tensor
from mmdet3d.registry import MODELS
from mmdet3d.tools.fps import DFPSSampler, sample_points
from mmcv.ops.furthest_point_sample import furthest_point_sample
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


@MODELS.register_module()
class HardSimpleVFE(nn.Module):
    """Simple voxel feature encoder used in SECOND.

    It simply averages the values of points in a voxel.

    Args:
        num_features (int, optional): Number of features to use. Default: 4.
    """

    def __init__(self, num_features: int = 4) -> None:
        super(HardSimpleVFE, self).__init__()
        self.num_features = num_features

    def forward(self, features: Tensor, num_points: Tensor, coors: Tensor,
                *args, **kwargs) -> Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features in shape
                (N, M, 3(4)). N is the number of voxels and M is the maximum
                number of points inside a single voxel.
            num_points (torch.Tensor): Number of points in each voxel,
                 shape (N, ).
            coors (torch.Tensor): Coordinates of voxels.

        Returns:
            torch.Tensor: Mean of points inside each voxel in shape (N, 3(4))
            瀵规瘡涓€涓綋绱犲唴鐨勭偣姹傚拰锛岀劧鍚庡啀闄や互姣忎竴涓綋绱犲唴鐨勭偣鏁般€傚緱鍒版瘡涓€涓綋绱犵殑骞冲潎鐗瑰緛  16000*5*4--->16000*4 銆傛瘡涓€涓綋绱犲惈鏈変簲涓偣
        """
        points_mean = features[:, :, :self.num_features].sum(dim=1, keepdim=False) / num_points.type_as(features).view(
            -1, 1)
        return points_mean.contiguous()


@MODELS.register_module()
class DynamicSimpleVFE(nn.Module):
    """Simple dynamic voxel feature encoder used in DV-SECOND.

    It simply averages the values of points in a voxel.
    But the number of points in a voxel is dynamic and varies.

    Args:
        voxel_size (tupe[float]): Size of a single voxel
        point_cloud_range (tuple[float]): Range of the point cloud and voxels
    """

    def __init__(self,
                 voxel_size: Tuple[float] = (0.2, 0.2, 4),
                 point_cloud_range: Tuple[float] = (0, -40, -3, 70.4, 40, 1)):
        super(DynamicSimpleVFE, self).__init__()
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

    @torch.no_grad()
    def forward(self, features: Tensor, coors: Tensor, *args,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features in shape
                (N, 3(4)). N is the number of points.
            coors (torch.Tensor): Coordinates of voxels.

        Returns:
            torch.Tensor: Mean of points inside each voxel in shape (M, 3(4)).
                M is the number of voxels.
        """
        # This function is used from the start of the voxelnet
        # num_points: [concated_num_points]
        features, features_coors = self.scatter(features, coors)
        return features, features_coors


@MODELS.register_module()
class PaintingDynamicVFE(nn.Module):
    """Dynamic Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    Args:
        in_channels (int, optional): Input channels of VFE. Defaults to 4.
        feat_channels (list(int), optional): Channels of features in VFE.
        with_distance (bool, optional): Whether to use the L2 distance of
            points to the origin point. Defaults to False.
        with_cluster_center (bool, optional): Whether to use the distance
            to cluster center of points inside a voxel. Defaults to False.
        with_voxel_center (bool, optional): Whether to use the distance
            to center of voxel for each points inside a voxel.
            Defaults to False.
        voxel_size (tuple[float], optional): Size of a single voxel.
            Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): The range of points
            or voxels. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict, optional): Config dict of normalization layers.
        mode (str, optional): The mode when pooling features of points
            inside a voxel. Available options include 'max' and 'avg'.
            Defaults to 'max'.
        fusion_layer (dict, optional): The config dict of fusion
            layer used in multi-modal detectors. Defaults to None.
        return_point_feats (bool, optional): Whether to return the features
            of each points. Defaults to False.
    """

    def __init__(self,
                 feature_fusion_layer: dict = None,
                 in_channels: int = 4,
                 feat_channels: list = [],
                 sample_points: list = [2048, 1024],
                 with_distance: bool = False,
                 with_cluster_center: bool = False,
                 with_voxel_center: bool = False,
                 with_rgb: bool = False,
                 with_score: bool = False,
                 with_key_points: bool = False,
                 voxel_size: Tuple[float] = (0.2, 0.2, 4),
                 point_cloud_range: Tuple[float] = (0, -40, -3, 70.4, 40, 1),
                 norm_cfg: dict = dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode: str = 'max',
                 fusion_layer: dict = None,
                 fnn_layer: dict = None,
                 return_point_feats: bool = False,
                 cro_attention_list: [dict] = None,
                 semantic_head: [dict] = None,
                 seg_encoder: [dict] = None,
                 neck: [dict] = None, ):
        super(PaintingDynamicVFE, self).__init__()
        assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        self.use_rgb = with_rgb
        self.use_score = with_score
        self.use_key_points = with_key_points
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        if with_rgb:
            in_channels += 3
        if with_score:
            in_channels += 4
        if with_key_points:
            in_channels += 1

        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.num_keypoints = sample_points
        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        # 17---->32---->
        feat_channels = [self.in_channels] + list(feat_channels)
        # feat_channels = [17,32,128]# 4---->10---->64----->64 缁忚繃for鍚庯紙10-->64锛?128-->64) [16,
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            vfe_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True)))  #
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
        self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range, (mode != 'max'))
        self.cluster_scatter = DynamicScatter(voxel_size, point_cloud_range, average_points=True)
        self.fusion_layer = None
        if cro_attention_list is not None:
            CrossAttention_layers = []
            for layer in cro_attention_list:
                CrossAttention_layers.append(MODELS.build(layer))
            self.CrossAttention_layers = nn.ModuleList(CrossAttention_layers)
        if neck is not None:
            self.neck = MODELS.build(neck)
        if fnn_layer is not None:
            self.fnn_layer = MODELS.build(fnn_layer)
        if seg_encoder is not None:
            self.seg_encoder = MODELS.build(seg_encoder)
        if fusion_layer is not None:
            self.fusion_layer = MODELS.build(fusion_layer)
        if feature_fusion_layer is not None:
            self.feature_fusion_layer = MODELS.build(feature_fusion_layer)
        self.cross_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        if semantic_head is not None:
            self.semantic_head = MODELS.build(semantic_head)
        # self.CrossAttention_layer = nn.MultiheadAttention(64, 4, batch_first=True)

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

    def score_cross_attention(self, iters, coors, features, score, points, s=0.75):
        batch_features = []
        attention_features = []
        batch_feature = []
        sample_feature = []
        key_points = []
        for j in range(iters):
            inds = coors[:, 0] == j
            batch_fe = features[inds]
            batch_features.append(batch_fe)
            foreground_mask = ((1 - nn.Softmax(dim=1)(score[j])[:, 0:1]) > s)[:, 0]
            # foreground_mask = (torch.rand(foreground_mask.size()) > 0.7).cuda(0)
            foreground_numbers = foreground_mask.sum()
            batch_feature.append(batch_fe)
            if foreground_numbers < 2:
                attention_features.append(batch_fe.squeeze(0))
                sample_pts, sample_pts_feature = sample_key_points(2048, points[j], batch_fe.unsqueeze(0))
                sample_feature.append(sample_pts_feature.squeeze(0))
                key_points.append(sample_pts)
                # if i == 1:
                #     foreground_points.append(points[j][foreground_mask])
            elif foreground_numbers <= 2048:
                pts = points[j][foreground_mask]
                pts_feature = batch_fe[foreground_mask]
                sample_pt, sample_pts_feature = sample_key_points(2048, points[j], batch_fe.unsqueeze(0))
                sample_pts_feature = sample_pts_feature.squeeze(0)
                # 纭繚鐐逛簯鍜岀壒寰佺殑鎷兼帴椤哄簭涓€鑷?                combined_pts = torch.cat((pts, sample_pt), dim=0)
                sample_pts, inverse_indices = torch.unique(combined_pts, dim=0, return_inverse=True)
                features_A = pts_feature
                features_B = sample_pts_feature
                # combined_fea = torch.cat((pts_feature, sample_pts_feature), dim=0)
                features_union = torch.cat((features_A, features_B), dim=0)
                # 鍘婚噸鍚庣殑鐗瑰緛
                final_features = features_union[torch.unique(inverse_indices, return_inverse=False)]
                # 浣跨敤姝ｇ‘鐨勬嫾鎺ュ悗鐨勭壒寰佸紶閲?
                sample_feature.append(final_features)
                key_points.append(sample_pts)
            else:
                pts = points[j][foreground_mask]
                sample_pts_img, sample_pts_feature_img = sample_key_points(2048, pts,
                                                                           batch_fe[foreground_mask].unsqueeze(0))
                sample_pts, sample_pts_feature = sample_key_points(2048, points[j], batch_fe.unsqueeze(0))
                combined_pts = torch.cat((sample_pts_img, sample_pts), 0)
                sample_pts, inverse_indices = torch.unique(combined_pts, dim=0, return_inverse=True)
                features_A = sample_pts_feature_img.squeeze(0)
                features_B = sample_pts_feature.squeeze(0)
                # combined_fea = torch.cat((pts_feature, sample_pts_feature), dim=0)
                features_union = torch.cat((features_A, features_B), dim=0)
                # 鍘婚噸鍚庣殑鐗瑰緛
                final_features = features_union[torch.unique(inverse_indices, return_inverse=False)]
                # 浣跨敤姝ｇ‘鐨勬嫾鎺ュ悗鐨勭壒寰佸紶閲?
                sample_feature.append(final_features)
                key_points.append(sample_pts)
        return sample_feature, batch_features, key_points

    def forward(self,
                features: Tensor,
                coors: Tensor,
                points: Optional[Sequence[Tensor]] = None,
                img_feats: Optional[Sequence[Tensor]] = None,
                img_metas: Optional[dict] = None,
                images: Optional[Sequence[Tensor]] = None,
                org_points: list = [],
                coors_2d: list = [],
                ) -> tuple:
        features_ls = [features]  # 浣撶礌鐗瑰緛--->鐩墠鍜岀偣浜戞暟鎹竴鏍?        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            # coors姣忎釜鐐规墍鍦ㄤ綋绱犵殑浣嶇疆  姣斿18370涓偣 浜х敓浜?5653涓潪绌轰綋绱?涓€涓綋绱犳湁澶氫釜鐐癸紝姹傚彇鍧囧€间綔涓簐oxel_mean
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            # 浜х敓闈炵┖浣撶礌鐨勫钩鍧囩壒寰併€備竴涓潪绌轰綋绱犱骇鐢熶竴涓钩鍧囩壒寰侊紱涓€涓潪绌轰綋绱犲搴斿涓偣锛沵ean_coors锛氶潪绌轰綋绱犲潗鏍囥€傜敤浜庝笌coors瀵瑰簲鏉ヤ骇鐢焢oints_mean
            # points_mean锛氭瘮濡?8370涓偣鍖归厤15653鐗瑰緛 鍚屼竴涓綋绱犲唴鐨勭偣璧嬩簣鐩稿悓鐨剉oxel_mean鐗瑰緛銆備綋绱犱笌鐐逛簯鐗瑰緛鍖归厤銆?            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)
        # 鏄负浜嗗緱鍒癧x-x(mean),y-y(mean),z-z(mean)]鐗瑰緛
        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)
        # 杩欐浠ｇ爜鐨勪富瑕佺洰鐨勬槸璁＄畻姣忎釜鐐圭浉瀵逛簬鍏舵墍鍦ㄤ綋绱犱腑蹇冪殑浣嶇疆鍋忕Щ閲忥紝骞跺皢杩欎簺鍋忕Щ閲忎綔涓烘柊鐨勭壒寰佹坊鍔犲埌鐗瑰緛鍒楄〃涓?        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)
        aug, _, score = self.fusion_layer(img_feats, images, coors_2d, self.use_rgb, self.use_score,
                                          self.use_key_points)
        # background_scores = torch.cat(score, dim=0)
        # f_score = 1 - nn.Softmax(dim=1)(background_scores)[:, 0:1]
        # seg_mask = f_score > 0.7
        # f_score = 1 - background_scores.sigmoid().max(dim=-1, keepdim=True).values
        # key_feature = torch.cat((features[:, :3], background_scores), dim=1)
        # points_score_no = self.semantic_head(key_feature)
        # points_score = points_score_no['seg_preds'].sigmoid().max(dim=-1, keepdim=True).values
        batch_size = len(img_metas)
        if not isinstance(aug, list):
            features = torch.cat((features, aug), dim=1)  # 10-16-32
        img_features, pool_features = self.feature_fusion_layer(img_feats, coors, coors_2d, points)
        image_featuress = pool_features[0] + pool_features[1]
        # images_feature = torch.cat(pool_features, dim=1)
        for i, vfe in enumerate(self.vfe_layers):
            # Vfe 10-->(64-->(鎷兼帴64锛?->128)-->614358555.
            point_feats = vfe(features)
            voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors)
            if i != len(self.vfe_layers) - 1:
                feat_per_point = self.map_voxel_center_to_point(coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)
        key_points = None
        seg_result = None
        return voxel_feats, voxel_coors, dict(key_points=key_points, seg_result=seg_result)


def sample_key_points(numbers, points: Tensor, features: Tensor
                      ):
    pts = points[:, :3]
    cur_pt_idxs = furthest_point_sample(
        pts.unsqueeze(dim=0).contiguous(),
        numbers).long()[0]
    key_feature = features[:, cur_pt_idxs]
    keypoints = points[cur_pt_idxs]
    return keypoints, key_feature


def batch_points(points, sample_points):
    query_lengths = [q.shape[0] for q in points]  # [100, 150]
    # key_lengths = [k.shape[0] for k in keys]  # [80, 120]

    padded_queries = pad_sequence(points, batch_first=True, padding_value=0)  # (2, 150, 64)
    padded_keys = pad_sequence(sample_points, batch_first=True, padding_value=0)  # (2, 120, 64)
    mask_kv = (padded_keys.sum(dim=-1) == 0)
    return padded_queries, padded_keys, mask_kv, query_lengths


# 1. 鍑嗗鍙橀暱鏁版嵁
def attention(points, sample_points, cross_attn_layers):
    padded_queries, padded_keys, mask_kv, query_lengths = batch_points(points, sample_points)
    padded_output, _ = cross_attn_layers(
        query=padded_queries,
        key=padded_keys,
        value=padded_keys,
        key_padding_mask=mask_kv
    )  # padded_output.shape = (2, 150, 64)

    output_list = [
        padded_output[i, :query_lengths[i], :]
        for i in range(len(query_lengths))
    ]
    return output_list


def cross_attention(iters, coors, features, score, points, num_keypoints, cross_layer, s=0.75):
    foreground_points = []
    attention_features = []

    for j in range(iters):
        inds = coors[:, 0] == j
        batch_fe = features[inds].unsqueeze(0)
        foreground_mask = ((1 - nn.Softmax(dim=1)(score[j])[:, 0:1]) > s)[:, 0]
        foreground_numbers = foreground_mask.sum()
        if foreground_numbers < 2:
            attention_features.append(batch_fe.squeeze(0))
            # if i == 1:
            #     foreground_points.append(points[j][foreground_mask])
            continue
        else:
            pts = points[j][foreground_mask]
            sample_pts, sample_pts_feature = sample_key_points(num_keypoints, pts, batch_fe[:, foreground_mask])
        attention_feature = cross_layer(batch_fe, sample_pts_feature)
        attention_features.append(attention_feature)
    features = torch.cat(attention_features, dim=0)
    return features


"""

            for j in range(len(img_metas)):
                inds = coors[:, 0] == j
                batch_fe = features[inds].unsqueeze(0)
                foreground_mask = mask[j]
                foreground_numbers = foreground_mask.sum()
                if foreground_numbers < 2:
                    attention_features.append(batch_fe.squeeze(0))
                    if i == 1:
                        foreground_points.append(points[j][foreground_mask])
                    continue
                elif foreground_numbers > self.num_keypoints[i]:
                    pts = points[j][foreground_mask]
                    sample_pts, sample_pts_feature = sample_key_points(self.num_keypoints[i], pts,
                                                                       batch_fe[:, foreground_mask])
                    if i == 1:
                        foreground_points.append(sample_pts)
                else:
                    if i == 1:
                        foreground_points.append(points[j][foreground_mask])
                    sample_pts_feature = batch_fe[:, foreground_mask]
                if self.CrossAttention_layers is not None:
                    attention_feature = self.CrossAttention_layers[i](batch_fe, sample_pts_feature).squeeze(0)
                    attention_features.append(attention_feature)
            if self.CrossAttention_layers and attention_features:
                features = torch.cat(attention_features, dim=0)


                            # def score_cross_attention(self, iters, coors, features, score, points, num_keypoints, cross_layer, s=0.75):
    #     foreground_points = []
    #     attention_features = []
    #     batch_feature = []
    #     sample_feature = []
    #     key_points = []
    #     sample_coors_ = []
    #     key_points_4096 = []
    #     for j in range(iters):
    #         inds = coors[:, 0] == j
    #
    #         batch_fe = features[inds]
    #         batch_coors = coors[inds]
    #         feature_fe_coor = torch.cat((batch_coors,batch_fe),dim=-1)
    #         foreground_mask = ((1 - nn.Softmax(dim=1)(score[j])[:, 0:1]) > s)[:, 0]
    #         # foreground_mask = (torch.rand(foreground_mask.size()) > 0.7).cuda(0)
    #         foreground_numbers = foreground_mask.sum()
    #         batch_feature.append(batch_fe)
    #         if foreground_numbers < 2:
    #             attention_features.append(batch_fe.squeeze(0))
    #             sample_pts, sample_pts_feature_coors = sample_key_points(2048, points[j], feature_fe_coor.unsqueeze(0))
    #             sample_pts_feature = sample_pts_feature_coors[...,4:]
    #             sample_coors = sample_pts_feature_coors[...,:4]
    #             sample_coors_.append(sample_coors.squeeze(0))
    #             sample_feature.append(sample_pts_feature.squeeze(0))
    #             key_points.append(sample_pts)
    #             # if i == 1:
    #             #     foreground_points.append(points[j][foreground_mask])
    #         else:
    #             pts = points[j][foreground_mask]
    #             pts_feature = batch_fe[foreground_mask]
    #             pts_coors = batch_coors[foreground_mask]
    #             sample_pt, sample_pts_feature_coors = sample_key_points(6400, points[j], feature_fe_coor.unsqueeze(0))
    #             # sample_pts_feature = sample_pts_feature_coors[:,4:].squeeze(0)
    #             # sample_pts_coors = sample_pts_feature_coors[:,:4].squeeze(0)
    #             # 纭繚鐐逛簯鍜岀壒寰佺殑鎷兼帴椤哄簭涓€鑷?    #             combined_pts = torch.cat((pts, sample_pt), dim=0)
    #             sample_pts, inverse_indices = torch.unique(combined_pts, dim=0, return_inverse=True)
    #             features_A = torch.cat((pts_coors,pts_feature),-1)
    #             features_B = sample_pts_feature_coors.squeeze(0)
    #             # combined_fea = torch.cat((pts_feature, sample_pts_feature), dim=0)
    #             features_union = torch.cat((features_A, features_B), dim=0)
    #             # 鍘婚噸鍚庣殑鐗瑰緛
    #             final_features_coors = features_union[torch.unique(inverse_indices, return_inverse=False)]
    #             # 浣跨敤姝ｇ‘鐨勬嫾鎺ュ悗鐨勭壒寰佸紶閲?    #             key_points_4096.append(sample_pts)
    #             sample_pt, sample_pts_feature_coors = sample_key_points(2048, sample_pts, final_features_coors.unsqueeze(0))
    #             sample_pts_feature = sample_pts_feature_coors[..., 4:]
    #             sample_coors = sample_pts_feature_coors[..., :4]
    #             sample_coors_.append(sample_coors.squeeze(0))
    #             sample_feature.append(sample_pts_feature.squeeze(0))
    #             key_points.append(sample_pt)
            # else:
            #     pts = points[j][foreground_mask]
            #     sample_pts_img, sample_pts_feature_img = sample_key_points(4096, pts,
            #                                                                batch_fe[foreground_mask].unsqueeze(0))
            #     sample_pts, sample_pts_feature = sample_key_points(4096, points[j], batch_fe.unsqueeze(0))
            #     combined_pts = torch.cat((sample_pts_img, sample_pts), 0)
            #     sample_pts, inverse_indices = torch.unique(combined_pts, dim=0, return_inverse=True)
            #     features_A = sample_pts_feature_img.squeeze(0)
            #     features_B = sample_pts_feature.squeeze(0)
            #     # combined_fea = torch.cat((pts_feature, sample_pts_feature), dim=0)
            #     features_union = torch.cat((features_A, features_B), dim=0)
            #     # 鍘婚噸鍚庣殑鐗瑰緛
            #     final_features = features_union[torch.unique(inverse_indices, return_inverse=False)]
            #     # 浣跨敤姝ｇ‘鐨勬嫾鎺ュ悗鐨勭壒寰佸紶閲?            #     key_points_4096.append(sample_pts)
            #     sample_pt, sample_pts_feature = sample_key_points(2048, sample_pts, final_features.unsqueeze(0))
            #     sample_feature.append(sample_pts_feature)
            #     key_points.append(sample_pt)
            # for i in range(iters):
            #     if i == 0:
            #         points_score_no = None
            points_score_no=None
                # points_score = points_score_no['seg_preds'].sigmoid().max(dim=-1, keepdim=True).values[
                #                i * 1024:(i + 1) * 1024]
                # attention_feature = cross_layer(batch_feature[i], sample_feature[i] * points_score, sample_feature[i])[0]
            #     attention_features.append(batch_feature[i].squeeze(0))
            # features = torch.cat(attention_features, dim=0)
        return torch.cat(sample_feature,0), torch.cat(sample_coors_,0), key_points
"""

