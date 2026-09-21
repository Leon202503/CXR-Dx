"""共享推理器：训练评估与比赛提交 run.py 复用同一套逻辑，避免预处理不一致。

能力：
- 从一个或多个 .pth 加载模型（异构/同构 backbone 概率平均集成）；
- 权重内自带 class_names / img_size / clahe / thresholds，提交时无需额外 yaml；
- 支持水平翻转 TTA；
- pretrained=False，完全离线加载（timm 只用到网络结构定义）。
"""
from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset import CXRDataset, CXRTestDataset
from data.transforms import build_eval_transforms, build_tta_transforms
from models.model import CXRClassifier


class Predictor:
    def __init__(self, ckpt_paths: Sequence[str], device: str = "", tta: bool = True,
                 batch_size: int = 32, num_workers: int = 4):
        if isinstance(ckpt_paths, str):
            ckpt_paths = [ckpt_paths]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tta = tta
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.models: List[torch.nn.Module] = []
        self.metas = []
        for p in ckpt_paths:
            ckpt = torch.load(p, map_location=self.device, weights_only=False)
            model = CXRClassifier(
                backbone=ckpt["backbone"],
                num_classes=len(ckpt["class_names"]),
                pretrained=False,
            ).to(self.device).eval()
            model.load_state_dict(ckpt["state_dict"])
            self.models.append(model)
            self.metas.append(ckpt)
        # 以第一个权重为准（集成时要求各模型类别顺序一致）
        self.class_names = self.metas[0]["class_names"]
        for m in self.metas[1:]:
            if m["class_names"] != self.class_names:
                raise ValueError("集成模型的类别顺序不一致，请先对齐：%s" % p)

    def _loader_from_transform(self, ds, transform):
        ds.transform = transform
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    @torch.no_grad()
    def _forward(self, model, loader) -> np.ndarray:
        outs = []
        for batch in loader:
            imgs = batch[0].to(self.device)
            logits = model(imgs)
            outs.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(outs)

    def predict_labeled(self, df: pd.DataFrame, image_root: str = "") -> tuple:
        """对统一格式 DataFrame 推理，返回 (probs[N,C], labels[N,C])。"""
        probs_per_model = []
        ds_ref = None
        for model, meta in zip(self.models, self.metas):
            ds = CXRDataset.from_df(df, self.class_names, image_root, transform=None)
            ds_ref = ds
            if self.tta:
                transforms_list = build_tta_transforms(
                    meta["img_size"], meta.get("clahe", False), meta.get("clahe_clip", 2.0))
                p = np.mean([self._forward(model, self._loader_from_transform(ds, t)) for t in transforms_list], axis=0)
            else:
                tf = build_eval_transforms(meta["img_size"], meta.get("clahe", False), meta.get("clahe_clip", 2.0))
                p = self._forward(model, self._loader_from_transform(ds, tf))
            probs_per_model.append(p)
        probs = np.mean(probs_per_model, axis=0)
        return probs, ds_ref.labels

    @torch.no_grad()
    def predict_test_csv(self, test_csv: str, image_root: str = "") -> tuple:
        """对无标签测试 csv 推理，返回 (id 列表, probs[N,C])。"""
        probs_per_model, ids = [], None
        for model, meta in zip(self.models, self.metas):
            if self.tta:
                transforms_list = build_tta_transforms(
                    meta["img_size"], meta.get("clahe", False), meta.get("clahe_clip", 2.0))
                ps = []
                for t in transforms_list:
                    ds = CXRTestDataset(test_csv, image_root, transform=t)
                    ids = [self._ident(ds, i) for i in range(len(ds))]
                    ps.append(self._forward(model, DataLoader(
                        ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)))
                p = np.mean(ps, axis=0)
            else:
                tf = build_eval_transforms(meta["img_size"], meta.get("clahe", False), meta.get("clahe_clip", 2.0))
                ds = CXRTestDataset(test_csv, image_root, transform=tf)
                ids = [self._ident(ds, i) for i in range(len(ds))]
                p = self._forward(model, DataLoader(
                    ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers))
            probs_per_model.append(p)
        return ids, np.mean(probs_per_model, axis=0)

    @staticmethod
    def _ident(ds, i):
        row = ds.df.iloc[i]
        if ds.id_col:
            return str(row[ds.id_col])
        import os
        return os.path.splitext(os.path.basename(str(row[ds.img_col])))[0]

    @property
    def thresholds(self) -> np.ndarray:
        # 集成时各模型阈值取平均
        return np.mean([np.array(m["thresholds"]) for m in self.metas], axis=0)
