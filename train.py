"""训练入口。

示例：
    python train.py --config configs/config.yaml
    python train.py --backbone convnext_tiny --img_size 320 --loss asl --fold 0
    python train.py --debug                      # 少量样本快速跑通流程

产物（runs/<backbone>_f<fold>/）：
    best.pth / last.pth / thresholds.json / config.yaml / metrics_history.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from contextlib import nullcontext
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import CXRDataset
from data.transforms import build_train_transforms, build_eval_transforms
from models.model import build_model
from losses import build_loss, compute_pos_weights
from metrics import evaluate_all, search_per_class_thresholds

# 兼容 torch 1.x / 2.x 的混合精度 API（无 CUDA 时安全降级为普通上下文）
try:
    from torch.amp import GradScaler as _GradScaler, autocast as _autocast
    def amp_scaler(enabled): return _GradScaler("cuda", enabled=enabled)
    def amp_autocast(enabled):
        return _autocast("cuda", enabled=True) if enabled else nullcontext()
except ImportError:  # 旧版 torch
    from torch.cuda.amp import GradScaler as _GradScaler, autocast as _autocast
    def amp_scaler(enabled): return _GradScaler(enabled=enabled)
    def amp_autocast(enabled):
        return _autocast(enabled=True) if enabled else nullcontext()


# ---------------- EMA（权重滑动平均，几乎稳赚的提分点） ----------------
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9997):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                s.copy_(m)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_splits(cfg: dict, fold: int):
    """返回 train_df / val_df（统一格式 DataFrame）。优先用独立 val.csv，否则按患者五折。"""
    dcfg = cfg["data"]
    train_df = pd.read_csv(dcfg["train_csv"])
    val_csv = dcfg.get("val_csv", "")
    if val_csv and os.path.exists(val_csv):
        return train_df, pd.read_csv(val_csv)

    n_folds = cfg["split"]["n_folds"]
    groups = train_df["patient_id"].astype(str).values if "patient_id" in train_df.columns \
        else np.arange(len(train_df)).astype(str)
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(train_df, groups=groups))
    tr_idx, va_idx = splits[fold % n_folds]
    print(f"[split] 按患者 GroupKFold：fold={fold}，train={len(tr_idx)}，val={len(va_idx)}，"
          f"患者无交叉={len(set(groups[tr_idx]) & set(groups[va_idx])) == 0}")
    return train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[va_idx].reset_index(drop=True)


@torch.no_grad()
def run_eval(model, loader, device, num_classes, use_amp):
    model.eval()
    probs_all, labels_all = [], []
    for imgs, targets, _ in loader:
        imgs = imgs.to(device)
        with amp_autocast(use_amp and device.type == "cuda"):
            logits = model(imgs)
        probs_all.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels_all.append(targets.numpy())
    return np.concatenate(probs_all), np.concatenate(labels_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--loss", default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--debug", action="store_true", help="200 张样本、2 epoch 跑通流程")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.backbone: cfg["model"]["backbone"] = args.backbone
    if args.img_size: cfg["train"]["img_size"] = args.img_size
    if args.batch_size: cfg["train"]["batch_size"] = args.batch_size
    if args.epochs: cfg["train"]["epochs"] = args.epochs
    if args.loss: cfg["train"]["loss"] = args.loss
    fold = args.fold if args.fold is not None else cfg["split"]["fold"]
    set_seed(cfg["split"]["seed"] + fold)

    tcfg = cfg["train"]
    img_size = tcfg["img_size"]
    class_names = cfg["class_names"]
    nc = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}, backbone={cfg['model']['backbone']}, img_size={img_size}, loss={tcfg['loss']}")

    # ---------- 数据 ----------
    train_df, val_df = make_splits(cfg, fold)
    if args.debug:
        train_df, val_df = train_df.iloc[:200].reset_index(drop=True), val_df.iloc[:64].reset_index(drop=True)
    aug_cfg = cfg.get("aug", {})
    train_ds = CXRDataset.from_df(
        train_df, class_names, cfg["data"]["image_root"],
        build_train_transforms(img_size, aug_cfg), cfg.get("include_no_finding", False))
    val_ds = CXRDataset.from_df(
        val_df, class_names, cfg["data"]["image_root"],
        build_eval_transforms(img_size, aug_cfg.get("clahe", False), aug_cfg.get("clahe_clip", 2.0)),
        cfg.get("include_no_finding", False))

    nw = 0 if args.debug else cfg["data"].get("num_workers", 8)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                            num_workers=nw, pin_memory=True)

    # ---------- 模型 / 损失 / 优化器 ----------
    model = build_model(cfg).to(device)
    train_labels = train_ds.labels
    pos_weight = compute_pos_weights(train_labels, tcfg.get("pos_weight_cap", 10.0)).to(device)
    criterion = build_loss(tcfg["loss"], tcfg, pos_weight)
    if tcfg["loss"] == "weighted_bce":
        print("[loss] 每类 pos_weight =", np.round(pos_weight.cpu().numpy(), 2))

    epochs = 2 if args.debug else tcfg["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    warmup = tcfg.get("warmup_epochs", 2)
    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = amp_scaler(tcfg.get("amp", True) and device.type == "cuda")
    ema = EMA(model, tcfg.get("ema_decay", 0.9997)) if tcfg.get("ema", True) else None
    ls = tcfg.get("label_smoothing", 0.0)

    out_dir = os.path.join(cfg["output"]["dir"], f"{cfg['model']['backbone']}_f{fold}")
    os.makedirs(out_dir, exist_ok=True)
    monitor = cfg["eval"].get("monitor", "mAUC")
    best_score, history, bad = -1.0, [], 0

    def save_ckpt(path, eval_model, thresholds=None):
        torch.save({
            "state_dict": eval_model.state_dict(),
            "backbone": cfg["model"]["backbone"],
            "class_names": class_names,
            "img_size": img_size,
            "clahe": cfg.get("aug", {}).get("clahe", False),
            "clahe_clip": cfg.get("aug", {}).get("clahe_clip", 2.0),
            "thresholds": thresholds.tolist() if thresholds is not None else [0.5] * nc,
        }, path)

    for ep in range(epochs):
        model.train()
        losses = []
        for imgs, targets, _ in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            if ls > 0:  # 双向标签平滑，抗 NLP 噪声标注
                targets = targets * (1 - ls) + 0.5 * ls
            opt.zero_grad()
            with amp_autocast(tcfg.get("amp", True) and device.type == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema: ema.update(model)
            losses.append(loss.item())
        scheduler.step()

        eval_model = ema.shadow if ema else model
        probs, labels = run_eval(eval_model, val_loader, device, nc, tcfg.get("amp", True))
        thresholds = search_per_class_thresholds(probs, labels) if cfg["eval"].get("search_threshold", True) \
            else np.full(nc, 0.5)
        summary, detail = evaluate_all(probs, labels, class_names, thresholds)
        summary["train_loss"] = float(np.mean(losses))
        history.append({"epoch": ep, **summary})
        print(f"[ep {ep+1}/{epochs}] loss={summary['train_loss']:.4f} "
              f"mAUC={summary['mAUC']:.4f} mAP={summary['mAP']:.4f} macroF1={summary['macro_F1']:.4f}")

        score = summary[monitor]
        if score > best_score:
            best_score, bad = score, 0
            save_ckpt(os.path.join(out_dir, "best.pth"), eval_model, thresholds)
            with open(os.path.join(out_dir, "thresholds.json"), "w", encoding="utf-8") as f:
                json.dump({"class_names": class_names, "thresholds": thresholds.tolist(),
                           "detail": detail, "summary": summary}, f, ensure_ascii=False, indent=2)
            print(f"  -> 新最优 {monitor}={score:.4f}，已保存 best.pth")
        else:
            bad += 1
            if bad >= tcfg.get("early_stop_patience", 6):
                print(f"[early stop] {monitor} 连续 {bad} 轮未提升")
                break

    save_ckpt(os.path.join(out_dir, "last.pth"), ema.shadow if ema else model)
    with open(os.path.join(out_dir, "metrics_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    print(f"[done] 最优 {monitor}={best_score:.4f}，产物目录：{out_dir}")


if __name__ == "__main__":
    main()
