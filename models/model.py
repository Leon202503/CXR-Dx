"""多标签分类模型：timm backbone + 全局池化 + C 路 sigmoid 分类头。

- 训练时 pretrained=True 自动下载 ImageNet 权重；
- 提交时 pretrained=False，从本地 .pth 加载，保证离线可跑。
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class CXRClassifier(nn.Module):
    def __init__(self, backbone: str = "densenet121", num_classes: int = 14,
                 pretrained: bool = True, drop_rate: float = 0.2):
        super().__init__()
        self.backbone_name = backbone
        # in_chans=3：X光灰度图在数据层复制为三通道，直接复用预训练权重
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,            # 去掉自带分类头，返回池化特征
            in_chans=3,
            drop_rate=drop_rate,
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


def build_model(cfg: dict, pretrained: bool | None = None) -> CXRClassifier:
    mcfg = cfg["model"]
    return CXRClassifier(
        backbone=mcfg["backbone"],
        num_classes=len(cfg["class_names"]),
        pretrained=mcfg["pretrained"] if pretrained is None else pretrained,
        drop_rate=mcfg.get("drop_rate", 0.2),
    )
