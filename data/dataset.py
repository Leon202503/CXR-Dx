"""严格解析多标签标注，支持统一多热表、NIH 病变名和宽表。"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from predict.runtime import load_image as robust_load_image

IMG_COL_CANDIDATES = ["image_path", "image", "img", "Image Index", "Image_Index", "file", "filename"]
LABEL_COL_CANDIDATES = ["labels", "label", "Finding Labels", "Finding_Labels", "Target"]
PID_COL_CANDIDATES = ["patient_id", "Patient ID", "Patient_ID", "pid"]


def _find_column(df, candidates):
    return next((c for c in candidates if c in df.columns), None)


def load_labels(df, class_names, include_no_finding=False):
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("类别清单为空或有重复")
    if include_no_finding and "No Finding" not in class_names:
        raise ValueError("输出 No Finding 时必须将其加入 class_names")
    if df.empty:
        raise ValueError("标注表为空")
    nc = len(class_names)
    labels = np.zeros((len(df), nc), dtype=np.float32)
    col = _find_column(df, LABEL_COL_CANDIDATES)
    if col:
        for i, value in enumerate(df[col]):
            if pd.isna(value) or not str(value).strip():
                raise ValueError(f"第 {i} 行标签缺失")
            tokens = [x.strip() for x in str(value).replace(",", "|").split("|")]
            try:
                numeric = [float(x) for x in tokens]
            except ValueError:
                numeric = None
            if numeric is not None:
                if len(numeric) != nc:
                    raise ValueError(f"第 {i} 行标签维度 {len(numeric)} != {nc}")
                labels[i] = numeric
            else:
                unknown = set(tokens) - set(class_names) - {"No Finding"}
                if unknown:
                    raise ValueError(f"第 {i} 行存在未知类别: {unknown}")
                if "No Finding" in tokens and len(tokens) > 1:
                    raise ValueError(f"第 {i} 行 No Finding 与病变标签冲突")
                for token in tokens:
                    if token in class_names:
                        labels[i, class_names.index(token)] = 1
    else:
        missing = set(class_names) - set(df.columns)
        if missing:
            raise ValueError(f"宽表缺少类别列: {missing}")
        labels = df[class_names].to_numpy(dtype=np.float32)
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("标签必须是有限的 0/1；未知或不确定标签需显式清洗")
    pid = _find_column(df, PID_COL_CANDIDATES)
    return labels, df[pid].astype(str).values if pid else None


class CXRDataset(Dataset):
    def __init__(self, csv_path, class_names, image_root="", transform=None, include_no_finding=False):
        self.df = pd.read_csv(csv_path, dtype=str)
        self.class_names, self.image_root, self.transform = class_names, image_root, transform
        self._post_init(include_no_finding)

    @classmethod
    def from_df(cls, df, class_names, image_root="", transform=None, include_no_finding=False):
        obj = cls.__new__(cls)
        obj.df = df.reset_index(drop=True)
        obj.class_names, obj.image_root, obj.transform = class_names, image_root, transform
        obj._post_init(include_no_finding)
        return obj

    def _post_init(self, include_no_finding=False):
        self.img_col = _find_column(self.df, IMG_COL_CANDIDATES)
        if self.img_col is None or self.df[self.img_col].isna().any():
            raise ValueError("缺少有效图片路径列")
        self.labels, self.patient_ids = load_labels(self.df, self.class_names, include_no_finding)

    def __len__(self):
        return len(self.df)

    def _full_path(self, rel):
        return rel if os.path.isabs(rel) else os.path.join(self.image_root, rel)

    def __getitem__(self, idx):
        image = robust_load_image(self._full_path(str(self.df.iloc[idx][self.img_col])))
        if self.transform:
            image = self.transform(image)
        return image, torch.from_numpy(self.labels[idx]).float(), idx


class CXRTestDataset(Dataset):
    def __init__(self, csv_path, image_root="", transform=None):
        self.df = pd.read_csv(csv_path, dtype=str)
        self.image_root, self.transform = image_root, transform
        self.img_col = _find_column(self.df, IMG_COL_CANDIDATES)
        if self.img_col is None or self.df[self.img_col].isna().any():
            raise ValueError("缺少有效图片路径列")
        self.id_col = _find_column(self.df, ["id", "ID"])

    def __len__(self):
        return len(self.df)

    def _full_path(self, rel):
        return rel if os.path.isabs(rel) else os.path.join(self.image_root, rel)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self._full_path(str(row[self.img_col]))
        image = robust_load_image(path)
        if self.transform:
            image = self.transform(image)
        ident = str(row[self.id_col]) if self.id_col else os.path.splitext(os.path.basename(path))[0]
        return image, ident
