"""多标签胸部 X 光数据集。

兼容三种标注格式（自动识别）：
1. 本工程统一格式 csv：image_path,labels,patient_id
   - labels: 与 class_names 等长的多热字符串，如 "0|1|0|0|..."
2. NIH ChestX-ray14 原生 Data_Entry_2017.csv：
   - 列含 "Image Index"、"Finding Labels"（病变名以 "|" 分隔，正常为 "No Finding"）、"Patient ID"
3. 宽表 csv：一列图片路径 + 每个类别一列 0/1（官方常见给法），图片列名可为 image/image_path/Image Index 等
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 容忍截断图片，提升鲁棒性

IMG_COL_CANDIDATES = ["image_path", "image", "img", "Image Index", "Image_Index", "file", "filename"]
LABEL_COL_CANDIDATES = ["labels", "label", "Finding Labels", "Finding_Labels", "Target"]
PID_COL_CANDIDATES = ["patient_id", "Patient ID", "Patient_ID", "pid"]


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_labels(df: pd.DataFrame, class_names: List[str], include_no_finding: bool = False):
    """返回 (多热标签数组 [N,C], patient_id 数组或 None)。"""
    n = len(df)
    nc = len(class_names)
    labels = np.zeros((n, nc), dtype=np.float32)

    label_col = _find_column(df, LABEL_COL_CANDIDATES)
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    if label_col is not None:
        raw = df[label_col].astype(str).tolist()
        # 形如 "0|1|0" 或 "0,1,0" 的多热字符串
        sample = raw[0].replace("|", ",").replace(" ", "")
        is_onehot_str = all(ch in "01," for ch in sample) and "," in sample
        if is_onehot_str and len(sample.split(",")) == nc:
            for i, v in enumerate(raw):
                vec = [float(x) for x in v.replace("|", ",").replace(" ", "").split(",")]
                labels[i] = vec
        else:
            # 病变名以 "|" 分隔，如 "Atelectasis|Effusion"
            for i, v in enumerate(raw):
                for finding in v.split("|"):
                    finding = finding.strip()
                    if finding in name_to_idx:
                        labels[i, name_to_idx[finding]] = 1.0
    else:
        # 宽表：每个类别一列
        for j, name in enumerate(class_names):
            if name in df.columns:
                labels[:, j] = df[name].astype(float).values

    pid_col = _find_column(df, PID_COL_CANDIDATES)
    patient_ids = df[pid_col].astype(str).values if pid_col else None
    return labels, patient_ids


def robust_load_image(path: str) -> Image.Image:
    """鲁棒读图：统一转 RGB；兼容 16 位灰度、单通道、损坏图兜底为黑图。"""
    try:
        img = Image.open(path)
        img = img.convert("RGB")
    except Exception:
        # 损坏/缺失图返回黑图，保证推理批处理不中断（训练时可在 csv 阶段清洗）
        img = Image.fromarray(np.zeros((224, 224), dtype=np.uint8)).convert("RGB")
    return img


class CXRDataset(Dataset):
    def __init__(self, csv_path: str, class_names: List[str], image_root: str = "",
                 transform=None, include_no_finding: bool = False):
        self.df = pd.read_csv(csv_path)
        self.class_names = class_names
        self.image_root = image_root
        self.transform = transform
        self._post_init(include_no_finding)

    @classmethod
    def from_df(cls, df: pd.DataFrame, class_names: List[str], image_root: str = "",
                transform=None, include_no_finding: bool = False):
        """从已切分好的 DataFrame（统一格式：image_path,labels,patient_id）构造。"""
        obj = cls.__new__(cls)
        obj.df = df.reset_index(drop=True)
        obj.class_names = class_names
        obj.image_root = image_root
        obj.transform = transform
        obj._post_init(include_no_finding)
        return obj

    def _post_init(self, include_no_finding: bool = False):
        self.img_col = _find_column(self.df, IMG_COL_CANDIDATES)
        if self.img_col is None:
            raise ValueError(f"数据中找不到图片路径列，候选名：{IMG_COL_CANDIDATES}，实际列：{list(self.df.columns)}")
        if "labels" in self.df.columns:
            self.labels = np.array(
                [[float(x) for x in str(v).split("|")] for v in self.df["labels"].tolist()],
                dtype=np.float32)
        else:
            self.labels, _ = load_labels(self.df, self.class_names, include_no_finding)
        if self.labels.shape[1] != len(self.class_names):
            raise ValueError(f"标签维度 {self.labels.shape[1]} 与类别数 {len(self.class_names)} 不一致")
        self.patient_ids = self.df["patient_id"].astype(str).values if "patient_id" in self.df.columns else None

    def __len__(self):
        return len(self.df)

    def _full_path(self, rel: str) -> str:
        return rel if os.path.isabs(rel) else os.path.join(self.image_root, rel)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self._full_path(str(row[self.img_col]))
        img = robust_load_image(path)
        if self.transform is not None:
            img = self.transform(img)
        target = torch.from_numpy(self.labels[idx]).float()
        return img, target, idx


class CXRTestDataset(Dataset):
    """无标签测试集，返回 (图像, 图片标识)。标识优先用 id 列，否则用文件名。"""

    def __init__(self, csv_path: str, image_root: str = "", transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform
        self.img_col = _find_column(self.df, IMG_COL_CANDIDATES)
        if self.img_col is None:
            raise ValueError(f"{csv_path} 中找不到图片路径列，实际列：{list(self.df.columns)}")
        self.id_col = "id" if "id" in self.df.columns else ("ID" if "ID" in self.df.columns else None)

    def __len__(self):
        return len(self.df)

    def _full_path(self, rel: str) -> str:
        return rel if os.path.isabs(rel) else os.path.join(self.image_root, rel)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel = str(row[self.img_col])
        path = self._full_path(rel)
        img = robust_load_image(path)
        if self.transform is not None:
            img = self.transform(img)
        ident = str(row[self.id_col]) if self.id_col else os.path.splitext(os.path.basename(rel))[0]
        return img, ident
