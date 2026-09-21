"""离线评估 / 集成 / TTA / 生成提交 csv。

示例：
    # 单模型在验证集评估
    python evaluate.py --config configs/config.yaml --ckpt runs/densenet121_f0/best.pth --tta
    # 多模型集成（5 折或异构 backbone）
    python evaluate.py --ckpt runs/convnext_tiny_f0/best.pth runs/swin_tiny_f0/best.pth
    # 对官方测试 csv 产出结果
    python evaluate.py --ckpt runs/densenet121_f0/best.pth --test_csv data_csv/test.csv --out submit
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from infer import Predictor
from metrics import evaluate_all, search_per_class_thresholds
from train import load_cfg, make_splits, set_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", nargs="+", required=True, help="一个或多个权重，概率平均集成")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--tta", action="store_true", default=True)
    ap.add_argument("--no_tta", dest="tta", action="store_false")
    ap.add_argument("--retrain_threshold", action="store_true",
                    help="在当前验证集重新搜索阈值（默认用权重内保存的阈值平均）")
    ap.add_argument("--test_csv", default="", help="无标签测试集 csv，给出后产出提交 csv")
    ap.add_argument("--image_root", default="")
    ap.add_argument("--out", default="runs/eval")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg["split"]["seed"])
    os.makedirs(args.out, exist_ok=True)
    predictor = Predictor(args.ckpt, tta=args.tta,
                          batch_size=cfg["train"]["batch_size"],
                          num_workers=cfg["data"].get("num_workers", 8))
    print(f"[model] 集成 {len(args.ckpt)} 个权重，TTA={args.tta}")

    # ---------- 验证集评估 ----------
    val_csv = cfg["data"].get("val_csv", "")
    if val_csv and os.path.exists(val_csv):
        val_df = pd.read_csv(val_csv)
    else:
        fold = args.fold if args.fold is not None else cfg["split"]["fold"]
        _, val_df = make_splits(cfg, fold)

    probs, labels = predictor.predict_labeled(val_df, cfg["data"]["image_root"])
    thresholds = search_per_class_thresholds(probs, labels) if args.retrain_threshold else predictor.thresholds
    summary, detail = evaluate_all(probs, labels, predictor.class_names, thresholds)
    print("=" * 60)
    print(f"mAUC={summary['mAUC']:.4f}  mAP={summary['mAP']:.4f}  "
          f"macroF1={summary['macro_F1']:.4f}  microF1={summary['micro_F1@thr']:.4f}")
    detail_df = pd.DataFrame(detail)
    print(detail_df.to_string(index=False))
    detail_df.to_csv(os.path.join(args.out, "per_class_metrics.csv"), index=False, encoding="utf-8-sig")
    print("每类指标已保存：", os.path.join(args.out, "per_class_metrics.csv"))

    # ---------- 测试集推理，产出提交 csv ----------
    if args.test_csv:
        ids, test_probs = predictor.predict_test_csv(
            args.test_csv, args.image_root or cfg["data"]["image_root"])
        prob_df = pd.DataFrame(test_probs, columns=predictor.class_names)
        prob_df.insert(0, "id", ids)
        prob_path = os.path.join(args.out, "submission_prob.csv")
        prob_df.to_csv(prob_path, index=False, encoding="utf-8-sig")

        bin_df = prob_df.copy()
        bin_df[predictor.class_names] = (test_probs >= thresholds).astype(int)
        bin_path = os.path.join(args.out, "submission_binary.csv")
        bin_df.to_csv(bin_path, index=False, encoding="utf-8-sig")
        print(f"概率结果：{prob_path}\n二值结果：{bin_path}")
        print("注意：最终以官方 run.py 固定接口/提交格式为准，这里仅作离线核对。")


if __name__ == "__main__":
    main()
