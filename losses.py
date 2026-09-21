"""多标签长尾分类损失。

- bce:           标准 BCEWithLogits
- weighted_bce:  每类 pos_weight = neg/pos（带上限），缓解长尾
- focal:         Focal Loss，聚焦难样本
- asl:           Asymmetric Loss（Ben-Baruch, ICCV 2021），对负样本施加更大抑制，
                 是多标签长尾任务的标配，ChestX-ray14 上通常优于加权 BCE
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCEWithLogits(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = logits.sigmoid()
        # 概率截断：把容易负样本的概率压到 clip 以下后直接"扔掉"
        if self.clip is not None and self.clip > 0:
            probs = torch.clamp(probs, min=self.clip, max=1.0)
        log_pos = torch.log(probs.clamp(min=self.eps))
        log_neg = torch.log((1 - probs).clamp(min=self.eps))
        loss_pos = targets * ((1 - probs) ** self.gamma_pos) * log_pos
        loss_neg = (1 - targets) * (probs ** self.gamma_neg) * log_neg
        return -(loss_pos + loss_neg).mean()


def compute_pos_weights(labels: np.ndarray, cap: float = 10.0) -> torch.Tensor:
    """根据训练集多热标签统计每类 pos_weight = neg/pos（带上限）。"""
    pos = labels.sum(axis=0)
    neg = len(labels) - pos
    pos = np.maximum(pos, 1.0)
    w = np.minimum(neg / pos, cap)
    return torch.tensor(w, dtype=torch.float32)


def build_loss(name: str, tcfg: dict, pos_weight: torch.Tensor | None = None) -> nn.Module:
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "focal":
        return FocalBCEWithLogits(gamma=tcfg.get("focal_gamma", 2.0))
    if name == "asl":
        return AsymmetricLoss(
            gamma_neg=tcfg.get("asl_gamma_neg", 4.0),
            gamma_pos=tcfg.get("asl_gamma_pos", 0.0),
            clip=tcfg.get("asl_clip", 0.05),
        )
    raise ValueError(f"未知损失：{name}")
