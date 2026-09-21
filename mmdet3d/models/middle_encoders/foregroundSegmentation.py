# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
from mmcv.cnn.bricks import build_norm_layer
from mmdet.models.utils import multi_apply
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import InstanceList


@MODELS.register_module()
class Foreground_seg(BaseModule):
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
        mlp_channels: Tuple[int] = (256, 256),
        extra_width: float = 0.1,
        norm_cfg: dict = dict(type='BN1d', eps=1e-5, momentum=0.1),
        init_cfg: Optional[dict] = None,
        loss_seg: dict = dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            reduction='sum',
            gamma=2.0,
            alpha=0.25,
            activated=True,
            loss_weight=1.0)
    ) -> None:
        super(Foreground_seg, self).__init__(init_cfg=init_cfg)
        self.extra_width = extra_width
        self.num_classes = 3

        self.in_channels = in_channels
        self.use_sigmoid_cls = loss_seg.get('use_sigmoid', False)

        out_channels = self.num_classes
        if self.use_sigmoid_cls:
            self.out_channels = out_channels
        else:
            self.out_channels = out_channels + 1

        mlps_layers = []
        cin = in_channels
        for mlp in mlp_channels:
            mlps_layers.extend([
                nn.Linear(cin, mlp, bias=False),
                build_norm_layer(norm_cfg, mlp)[1],
                nn.ReLU()
            ])
            cin = mlp
        mlps_layers.append(nn.Linear(cin, self.out_channels, bias=True))

        self.seg_cls_layer = nn.Sequential(*mlps_layers)

        # self.loss_seg = MODELS.build(loss_seg)

    def forward(self, feats: torch.Tensor) -> dict:
        """Forward head.

        Args:
            feats (torch.Tensor): Point-wise features.

        Returns:
            dict: Segment predictions.
        """
        seg_preds = self.seg_cls_layer(feats)
        return dict(seg_preds=seg_preds)
