"""胸部 X 光多标签分类的数据增强。

医学影像增强原则：保守、不破坏解剖结构。
- 只做水平翻转（左右肺近似对称）、小角度旋转、轻微微缩放与亮度/对比度抖动；
- 不做垂直翻转、大角度旋转、强颜色抖动（X光为灰度图，色调无意义）。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


from predict.runtime import CLAHE, eval_transform


class MaybeCLAHE:
    def __init__(self, enable: bool, clip_limit: float = 2.0):
        self.enable = enable
        self.op = CLAHE(clip_limit) if enable else None

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.op(img) if self.op else img


def build_train_transforms(img_size: int = 224, aug_cfg: dict | None = None) -> transforms.Compose:
    aug_cfg = aug_cfg or {}
    ops = [MaybeCLAHE(aug_cfg.get("clahe", False), aug_cfg.get("clahe_clip", 2.0))]
    ops.append(transforms.Resize((img_size, img_size)))
    if aug_cfg.get("hflip", True):
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops.append(transforms.RandomApply(
        [transforms.RandomAffine(
            degrees=aug_cfg.get("rotate_deg", 7),
            scale=(1.0 - aug_cfg.get("scale_ratio", 0.1), 1.0 + aug_cfg.get("scale_ratio", 0.1)),
            translate=(0.03, 0.03),
        )], p=0.7))
    ops.append(transforms.RandomApply(
        [transforms.ColorJitter(brightness=aug_cfg.get("brightness", 0.1),
                                contrast=aug_cfg.get("contrast", 0.1))], p=0.5))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def build_eval_transforms(img_size: int = 224, clahe: bool = False, clip_limit: float = 2.0):
    return eval_transform(img_size, clahe, clip_limit)


def build_tta_transforms(img_size: int = 224, clahe: bool = False, clip_limit: float = 2.0):
    """TTA：[原图, 水平翻转]，推理时概率平均。"""
    base = build_eval_transforms(img_size, clahe, clip_limit)
    hflip = transforms.Compose([
        MaybeCLAHE(clahe, clip_limit),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return [base, hflip]
