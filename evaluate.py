"""无泄漏评估、独立阈值校准与无标签测试推理。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from infer import Predictor
from metrics import evaluate_all, search_per_class_thresholds
from train import load_cfg
from data.splits import make_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--val_csv", help="所有模型均未见过的评估 CSV")
    ap.add_argument("--calibration_csv", help="独立阈值校准集，必须与评估集患者隔离")
    ap.add_argument("--tta", action="store_true", default=None)
    ap.add_argument("--no_tta", dest="tta", action="store_false")
    ap.add_argument("--test_csv")
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--image_root")
    ap.add_argument("--out", default="runs/eval")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    root = args.image_root or cfg["data"]["image_root"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tta = cfg["eval"].get("tta", False) if args.tta is None else args.tta
    predictor = Predictor(args.ckpt, tta=tta, batch_size=cfg["train"]["batch_size"])
    thresholds = predictor.thresholds
    if args.calibration_csv:
        cal = pd.read_csv(args.calibration_csv, dtype=str)
        cp, cy = predictor.predict_labeled(cal, root)
        thresholds = search_per_class_thresholds(cp, cy)
        payload = {"class_names": predictor.class_names, "thresholds": thresholds.tolist(),
                   "tta": bool(tta), "model_count": len(args.ckpt), "model_hashes": predictor.model_hashes}
        (out / "thresholds.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif tta or len(args.ckpt) > 1:
        thresholds = np.full(len(predictor.class_names), 0.5)
        print("未提供校准集：TTA/集成使用 0.5 阈值；F1 仅反映该阈值设置")
    if not args.test_only:
        if args.val_csv:
            val = pd.read_csv(args.val_csv, dtype=str)
        else:
            _, val = make_splits(cfg, args.fold if args.fold is not None else cfg["split"]["fold"])
        if args.calibration_csv:
            from data.splits import assert_disjoint
            assert_disjoint(cal, val)
        probs, labels = predictor.predict_labeled(val, root)
        summary, detail = evaluate_all(probs, labels, predictor.class_names, thresholds)
        summary.update({"tta": bool(tta), "model_count": len(args.ckpt),
                        "threshold_source": "independent_calibration" if args.calibration_csv else "checkpoint_or_0.5"})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        pd.DataFrame(detail).to_csv(out / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(out / "predictions.npz", probs=probs, labels=labels,
                            class_names=np.asarray(predictor.class_names))
    if args.test_csv:
        ids, probs = predictor.predict_test_csv(args.test_csv, root)
        table = pd.DataFrame(probs, columns=predictor.class_names)
        table.insert(0, "id", ids)
        table.to_csv(out / "submission_prob.csv", index=False, encoding="utf-8-sig")
        table[predictor.class_names] = (probs >= thresholds).astype(int)
        table.to_csv(out / "submission_binary.csv", index=False, encoding="utf-8-sig")
    elif args.test_only:
        ap.error("--test_only 需要 --test_csv")


if __name__ == "__main__":
    main()
