"""多标签评测指标：每类 AUC / AP、macro mAUC / mAP、macro-F1，以及阈值搜索。

- AUC、AP 与阈值无关，是多标签排序能力的主指标（ChestX-ray14 官方用 mAUC）；
- F1 依赖阈值，提交二值结果时用验证集搜索每类最优阈值。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def per_class_auc(probs: np.ndarray, labels: np.ndarray):
    c = labels.shape[1]
    aucs = np.full(c, np.nan)
    for j in range(c):
        if labels[:, j].min() != labels[:, j].max():  # 该类正负样本都存在
            aucs[j] = roc_auc_score(labels[:, j], probs[:, j])
    return aucs


def per_class_ap(probs: np.ndarray, labels: np.ndarray):
    c = labels.shape[1]
    aps = np.full(c, np.nan)
    for j in range(c):
        if labels[:, j].min() != labels[:, j].max():
            aps[j] = average_precision_score(labels[:, j], probs[:, j])
    return aps


def macro_mean(arr: np.ndarray) -> float:
    return float(np.nanmean(arr))


def search_per_class_thresholds(probs: np.ndarray, labels: np.ndarray,
                                grid=None) -> np.ndarray:
    """为每个类别搜索使该类 F1 最大的阈值。"""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    c = labels.shape[1]
    best = np.full(c, 0.5)
    for j in range(c):
        if labels[:, j].min() == labels[:, j].max():
            continue
        scores = [f1_score(labels[:, j], (probs[:, j] >= t).astype(int), zero_division=0) for t in grid]
        best[j] = grid[int(np.argmax(scores))]
    return best


def evaluate_all(probs: np.ndarray, labels: np.ndarray, class_names, thresholds: np.ndarray | None = None):
    """返回汇总 dict 与每类明细 dict。"""
    aucs = per_class_auc(probs, labels)
    aps = per_class_ap(probs, labels)
    if thresholds is None:
        thresholds = search_per_class_thresholds(probs, labels)
    preds = (probs >= thresholds).astype(int)
    f1s = np.full(labels.shape[1], np.nan)
    for j in range(labels.shape[1]):
        if labels[:, j].min() != labels[:, j].max():
            f1s[j] = f1_score(labels[:, j], preds[:, j], zero_division=0)
    summary = {
        "mAUC": macro_mean(aucs),
        "mAP": macro_mean(aps),
        "macro_F1": macro_mean(f1s),
        "micro_F1@thr": float(f1_score(labels, preds, average="micro", zero_division=0)),
    }
    detail = {
        "class": list(class_names),
        "AUC": [round(float(x), 4) if not np.isnan(x) else None for x in aucs],
        "AP": [round(float(x), 4) if not np.isnan(x) else None for x in aps],
        "F1": [round(float(x), 4) if not np.isnan(x) else None for x in f1s],
        "threshold": [round(float(x), 3) for x in thresholds],
        "pos_rate": [round(float(x), 4) for x in labels.mean(axis=0)],
    }
    return summary, detail
