# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmcv.ops import points_in_boxes_all, three_interpolate, three_nn
from mmdet.models.losses import sigmoid_focal_loss, smooth_l1_loss
from mmengine.runner import amp
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.registry import MODELS
from mmdet3d.structures import BaseInstance3DBoxes
from mmcv.cnn import ConvModule

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential

TwoTupleIntType = Tuple[Tuple[int]]


@MODELS.register_module()
class Seg_SparseEncoder(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (tuple[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
        return_middle_feats (bool): Whether output middle features.
            Default to False.
    """

    def __init__(
            self,
            voxel_size: List[float],
            point_cloud_range: List[float],
            in_channels: int,
            sparse_shape: List[int],
            semantic_head: Optional[dict] = None,
            order: Optional[Tuple[str]] = ('conv', 'norm', 'act'),
            norm_cfg: Optional[dict] = dict(
                type='BN1d', eps=1e-3, momentum=0.01),
            base_channels: Optional[int] = 16,
            output_channels: Optional[int] = 128,
            voxel_sa_cfgs_list: Optional[list] = None,
            encoder_channels: Optional[TwoTupleIntType] = ((16,), (32, 32,
                                                                   32),
                                                           (64, 64,
                                                            64), (64, 64, 64)),
            encoder_paddings: Optional[TwoTupleIntType] = ((1,), (1, 1, 1),
                                                           (1, 1, 1),
                                                           ((0, 1, 1), 1, 1)),
            block_type: Optional[str] = 'conv_module',
            return_middle_feats: Optional[bool] = False,
            bias: str = 'auto'
    ):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.return_middle_feats = return_middle_feats
        # Spconv init all weight on its own
        self.voxel_sa_layers = None
        self.bev_fusion = None
        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}
        gathered_channel = 0
        if semantic_head is not None:
            self.semantic_head = MODELS.build(semantic_head)
        if voxel_sa_cfgs_list is not None:
            self.voxel_sa_configs_list = voxel_sa_cfgs_list
            self.voxel_sa_layers = nn.ModuleList()
            for voxel_sa_config in voxel_sa_cfgs_list:
                cur_layer = MODELS.build(voxel_sa_config)
                self.voxel_sa_layers.append(cur_layer)
                gathered_channel += sum(
                    [x[-1] for x in voxel_sa_config.mlp_channels])

        self.point_feature_fusion_layer = nn.Sequential(
            ConvModule(
                gathered_channel,
                64,
                kernel_size=(1, 1),
                stride=(1, 1),
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d', eps=1e-5, momentum=0.1),
                bias=bias))
        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv',))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type)

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')

    def get_voxel_centers(self, coors: torch.Tensor,
                          scale_factor: float) -> torch.Tensor:
        """Get voxel centers coordinate.

        Args:
            coors (torch.Tensor): Coordinates of voxels shape is Nx(1+NDim),
                where 1 represents the batch index.
            scale_factor (float): Scale factor.

        Returns:
            torch.Tensor: Voxel centers coordinate with shape (N, 3).
        """
        assert coors.shape[1] == 4
        voxel_centers = coors[:, [3, 2, 1]].float()  # (xyz)
        voxel_size = torch.tensor(
            self.voxel_size,
            device=voxel_centers.device).float() * scale_factor
        pc_range = torch.tensor(
            self.point_cloud_range[0:3], device=voxel_centers.device).float()
        voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
        return voxel_centers

    # @amp.autocast(enabled=False)
    # def forward1(self, voxel_features: Tensor, coors: Tensor,
    #             batch_size: int, keys_xyz, point_fea, points) -> Union[Tensor, Tuple[Tensor, list]]:
    #     """Forward of SparseEncoder.
    #
    #     Args:
    #         voxel_features (torch.Tensor): Voxel features in shape (N, C).
    #         coors (torch.Tensor): Coordinates in shape (N, 4),
    #             the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
    #         batch_size (int): Batch size.
    #         keys_xyz (N1+N2)*3
    #         point_fea N*3
    #         points N*3
    #     Returns:
    #         torch.Tensor | tuple[torch.Tensor, list]: Return spatial features
    #             include:
    #
    #         - spatial_features (torch.Tensor): Spatial features are out from
    #             the last layer.
    #         - encode_features (List[SparseConvTensor], optional): Middle layer
    #             output features. When self.return_middle_feats is True, the
    #             module returns middle features.
    #     """
    #
    #     with torch.no_grad():
    #         coors = coors.int()
    #         input_sp_tensor = SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
    #         key_xyz_batch = [len(p) for p in keys_xyz]
    #         key_xyz_batch_cnt = voxel_features.new_tensor(key_xyz_batch, dtype=torch.int32)
    #         key_xyz = torch.cat(keys_xyz, 0)[:, :3]
    #         x = self.conv_input(input_sp_tensor)
    #         # seconda self.conv_input  16000*4--->conv_input--->16000*16
    #         # second  [16000*4--绗竴灞俿ub-->16000*4,29644*32,20869*64,10017*64
    #         # n*16-->encoder1-->n1*16-->encoder2-->n2*32-->encoder3-->n3*64-->encoder4-->n4*64
    #         # [41,1600,1408]-->encoder1-->[41,1600,1408]-->encoder2-->[21,800,704]-->encoder3-->[11,400,352]-->encoder3-->[5,200,176]
    #         encode_features = []
    #         k = 0
    #         for encoder_layer in self.encoder_layers:
    #             k = k + 1
    #             x = encoder_layer(x)
    #             encode_features.append(x)
    #         # if k == 1:
    #         #     break
    #     # for detection head
    #     # [200, 176, 5] -> [200, 176, 2]  64*[5,200,176]--->128*[2,200,176]--->256*200*176
    #     point_features_list = []
    #     if self.voxel_sa_layers is not None:
    #         for k, voxel_sa_layer in enumerate(self.voxel_sa_layers):
    #             if k > 1:
    #                 break
    #             cur_coords = encode_features[k].indices
    #             xyz = self.get_voxel_centers(
    #                 coors=cur_coords,
    #                 scale_factor=self.voxel_sa_configs_list[k].scale_factor
    #             ).contiguous()
    #             xyz_batch_cnt = xyz.new_zeros(batch_size).int()  # 鐐逛簯鏁伴噺
    #             for bs_idx in range(batch_size):
    #                 xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
    #
    #             pooled_points, pooled_features = voxel_sa_layer(
    #                 xyz=xyz.contiguous(),  # N*3
    #                 xyz_batch_cnt=xyz_batch_cnt,  # [N1.N2]
    #                 new_xyz=key_xyz.contiguous(),  # N*3
    #                 new_xyz_batch_cnt=key_xyz_batch_cnt,  # [n1,n2]
    #                 features=encode_features[k].features.contiguous(),  # N*C
    #             )
    #             point_features_list.append(pooled_features.contiguous())  # 4096*640*1
    #     sample_feature = torch.cat(point_fea, 0)
    #     points_feature = torch.cat((torch.cat(point_features_list, -1), sample_feature), -1)
    #     # fusion_feature = self.point_feature_fusion_layer(
    #     #     points_feature.unsqueeze(dim=-1).unsqueeze(-1)).squeeze(dim=-1).squeeze(dim=-1)
    #     # with torch.no_grad():
    #     # v_features = encode_features[-1].features.contiguous()
    #     # v_index = encode_features[-1].indices
    #     #
    #     # batch_features = []
    #     # batch_sample_features = []
    #     # semantic_results = self.semantic_head(points_feature)
    #     # score = semantic_results['seg_preds'].sigmoid().max(dim=-1, keepdim=True).values
    #     # score_seg = []
    #     # current_index = 0
    #     # for i in key_xyz_batch_cnt:
    #     #     score_seg.append(score[current_index:current_index+i])
    #     #     current_index = current_index + i
    #     # out = self.conv_out(encode_features[-1])
    #     # spatial_features = out.dense()
    #     #
    #     # N, C, D, H, W = spatial_features.shape
    #     # spatial_features = spatial_features.view(N, C * D, H, W)
    #     return points_feature

    @amp.autocast(enabled=False)
    def forward(self, points_features: Tensor, coors: Tensor,
                batch_size: int, keys_xyz, point_fea, points):
        point_features_list = []
        xyz_batch = [len(p) for p in points]
        key_xyz_batch = [len(p) for p in keys_xyz]
        key_xyz_batch_cnt = points_features.new_tensor(key_xyz_batch, dtype=torch.int32)
        xyz_batch = points_features.new_tensor(xyz_batch, dtype=torch.int32)
        key_xyz = torch.cat(keys_xyz, 0)[:, :3]
        xyz = torch.cat(points, 0)[:, :3]
        if self.voxel_sa_layers is not None:
            for k, voxel_sa_layer in enumerate(self.voxel_sa_layers):
                if k > 1:
                    break
                pooled_points, pooled_features = voxel_sa_layer(
                    xyz=xyz.contiguous(),  # N*3
                    xyz_batch_cnt=xyz_batch,  # [N1.N2]
                    new_xyz=key_xyz.contiguous(),  # N*3
                    new_xyz_batch_cnt=key_xyz_batch_cnt,  # [n1,n2]
                    features=points_features.contiguous(),  # N*C
                )
                point_features_list.append(pooled_features.contiguous())  # 4096*640*1
        sample_feature = torch.cat(point_fea, 0)
        sample_feature_ = torch.cat(point_features_list, -1)
        points_feature = torch.cat((sample_feature_, sample_feature), -1)
        fusion_feature = self.point_feature_fusion_layer(sample_feature_.unsqueeze(-1).unsqueeze(-1)).squeeze(
            -1).squeeze(-1)
        con = 0
        sample = []
        for i in key_xyz_batch:
            sample.append(fusion_feature[con:con + i])
            con = i
        return dict(seg_feature=points_feature, fusion_feature=sample)

    def make_encoder_layers(
            self,
            make_block: nn.Module,
            norm_cfg: Dict,
            in_channels: int,
            block_type: Optional[str] = 'conv_module',
            conv_cfg: Optional[dict] = dict(type='SubMConv3d')
    ) -> int:
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = SparseSequential()
        for i, blocks in enumerate(self.encoder_channels):
            if i > 0:
                break
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv3d'))
                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels

