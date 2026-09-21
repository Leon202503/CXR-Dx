# -*- coding: utf-8 -*-
"""
比赛提交入口（模板）。
====================================================================
重要：本文件按往届（2024 第六届）"固定区域不可修改"的形式预留了接口。
     2026 官方数据发布时会同时给 submit_example.zip（含官方 run.py 与
     结果格式 demo）。拿到后：
       1) 保留官方 run.py 的【固定区域】原样不动；
       2) 只把本文件【选手实现区】的推理逻辑搬进官方固定区域调用的函数；
       3) 严格按官方要求的 csv 列名/字段输出（概率 or 二值标签）。
====================================================================
打包目录（zip 根目录即下列文件，不要多套一层文件夹）：
    run.py
    requirements.txt
    weights/best.pth          （可放多个权重，自动平均集成；总体积 ≤ 2GB）
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import cv2
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 选手实现区（开始）
# ============================================================
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
USE_TTA = True            # 水平翻转 TTA，推理时间换 0.2~0.5 个点
BATCH_SIZE = 32
NUM_WORKERS = 4


class CXRClassifier(nn.Module):
    """与训练端 models/model.py 结构完全一致，保证 state_dict 可直接加载。"""

    def __init__(self, backbone: str, num_classes: int, drop_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0, in_chans=3)
        self.head = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(self.backbone.num_features, num_classes))

    def forward(self, x):
        return self.head(self.backbone(x))


def load_image(path: str) -> Image.Image:
    """鲁棒读图：灰度/16 位/三通道统一转 RGB，损坏图兜底黑图。"""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.fromarray(np.zeros((224, 224), dtype=np.uint8)).convert("RGB")


class ApplyCLAHE:
    """与训练端 data/transforms.py 的 CLAHE 保持一致。"""

    def __init__(self, clip_limit: float = 2.0):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = self.clahe.apply(np.array(img.convert("L")))
        return Image.fromarray(arr).convert("RGB")


class Inference:
    def __init__(self, weight_dir: str = os.path.join(HERE, "weights")):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weight_files = sorted(glob.glob(os.path.join(weight_dir, "*.pth")))
        assert weight_files, f"未在 {weight_dir} 找到 .pth 权重"
        self.bundles = []
        for wf in weight_files:
            ckpt = torch.load(wf, map_location=self.device, weights_only=False)
            model = CXRClassifier(ckpt["backbone"], len(ckpt["class_names"])).to(self.device).eval()
            model.load_state_dict(ckpt["state_dict"])
            self.bundles.append((model, ckpt))
        self.class_names = list(self.bundles[0][1]["class_names"])
        print(f"[run.py] 加载 {len(weight_files)} 个权重，类别={self.class_names}", flush=True)

    @torch.no_grad()
    def _forward(self, model, ckpt, paths):
        size = ckpt["img_size"]
        pre = [ApplyCLAHE(ckpt.get("clahe_clip", 2.0))] if ckpt.get("clahe", False) else []
        base = transforms.Compose(pre + [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        hflip = transforms.Compose(pre + [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        ts_list = [base, hflip] if USE_TTA else [base]
        probs_runs = []
        for ts in ts_list:
            batch = torch.stack([ts(load_image(p)) for p in paths]).to(self.device)
            probs_runs.append(torch.sigmoid(model(batch).float()).cpu().numpy())
        return np.mean(probs_runs, axis=0)

    @torch.no_grad()
    def predict_paths(self, paths):
        probs = np.mean([
            self._forward(model, ckpt, paths) for model, ckpt in self.bundles
        ], axis=0)
        return probs  # [N, C]，每个类别的患病概率

# ============================================================
# 选手实现区（结束）
# ============================================================


# ============================================================
# 以下为【固定区域】示例。官方 submit_example.zip 发布后，
# 请以官方版本为准，保持固定区域代码不变，只让它调用上面的 Inference。
# 常见形式：predict(test_path, save_path)
#   test_path : 测试集目录（内含测试 csv 与图片）或测试 csv 路径
#   save_path : 结果 csv 保存路径
# ============================================================
IMG_COLS = ["image_path", "image", "img", "Image Index", "file", "filename"]


def _resolve_csv(test_path: str) -> str:
    if os.path.isfile(test_path):
        return test_path
    csvs = glob.glob(os.path.join(test_path, "**", "*.csv"), recursive=True)
    assert csvs, f"测试目录 {test_path} 下未找到 csv"
    return csvs[0]


def predict(test_path: str, save_path: str):
    csv_path = _resolve_csv(test_path)
    df = pd.read_csv(csv_path)
    img_col = next((c for c in IMG_COLS if c in df.columns), None)
    assert img_col, f"测试 csv 缺图片路径列，实际列：{list(df.columns)}"
    image_root = test_path if os.path.isdir(test_path) else os.path.dirname(csv_path)

    def full(p):
        return p if os.path.isabs(p) else os.path.join(image_root, p)

    paths = [full(str(p)) for p in df[img_col]]
    engine = Inference()

    probs = np.zeros((len(paths), len(engine.class_names)), dtype=np.float32)
    for i in range(0, len(paths), BATCH_SIZE):
        probs[i:i + BATCH_SIZE] = engine.predict_paths(paths[i:i + BATCH_SIZE])

    # 输出：id 列（无 id 则用文件名）+ 每类概率。
    # 若官方要求输出 0/1 标签，用 ckpt["thresholds"] 二值化（见下注释）。
    id_col = "id" if "id" in df.columns else ("ID" if "ID" in df.columns else None)
    ids = df[id_col].astype(str) if id_col else \
        [os.path.splitext(os.path.basename(p))[0] for p in paths]
    out = pd.DataFrame(probs, columns=engine.class_names)
    out.insert(0, "id", ids)

    # 需要二值标签时取消注释（阈值随权重保存）：
    # thresholds = np.mean([np.array(c["thresholds"]) for _, c in engine.bundles], axis=0)
    # out[engine.class_names] = (probs >= thresholds).astype(int)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[run.py] 结果已保存：{save_path}，共 {len(out)} 行", flush=True)


if __name__ == "__main__":
    # 本地断网自测：python run.py <测试目录或csv> <结果csv>
    assert len(sys.argv) == 3, "用法：python run.py <test_path> <save_path>"
    predict(sys.argv[1], sys.argv[2])
