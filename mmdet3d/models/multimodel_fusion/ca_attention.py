import torch.nn as nn
import torch
from mmdet3d.registry import MODELS


@MODELS.register_module()
class Multimodal_feature_fusion(nn.Module):
    def __init__(self, dim=128, reduction=4):
        super(Multimodal_feature_fusion, self).__init__()
        self.lines_layers = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, pts_features, img_features):
        initial = pts_features + img_features
        mean_features = (initial.mean(0) + initial.max(0).values).unsqueeze(0)
        pat = self.lines_layers(mean_features)
        x = pat*pts_features+(1-pat)*img_features+initial
        fusion_features = self.fusion_layer(x)
        return fusion_features
