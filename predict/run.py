"""提交入口模板：拿到官方接口后，仅调整输入输出适配层。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from .runtime import InferenceEngine
except ImportError:
    from runtime import InferenceEngine

HERE = Path(__file__).resolve().parent
IMG_COLS = ["image_path", "image", "img", "Image Index", "Image_Index", "file", "filename"]


def predict(test_path, save_path, weight_dir=None, image_root=None, tta=False,
            batch_size=32, binary=False, thresholds_file=None, error_policy="raise", device=""):
    test_path = Path(test_path)
    if test_path.is_dir():
        csvs = sorted(test_path.rglob("*.csv"))
        if len(csvs) != 1:
            raise ValueError("测试目录必须仅有一个 CSV，或请明确指定 CSV 路径")
        csv_path = csvs[0]
    else:
        csv_path = test_path
    root = Path(image_root) if image_root else csv_path.parent
    df = pd.read_csv(csv_path, dtype=str)
    col = next((c for c in IMG_COLS if c in df.columns), None)
    if col is None or df[col].isna().any():
        raise ValueError("测试 CSV 缺少有效图片路径")
    paths = [str(Path(p) if Path(p).is_absolute() else root / p) for p in df[col]]
    weights = sorted((Path(weight_dir) if weight_dir else HERE / "weights").glob("*.pth"))
    engine = InferenceEngine(weights, device=device, tta=tta, batch_size=batch_size, error_policy=error_policy)
    probs = engine.predict_paths(paths)
    if binary:
        thresholds = engine.thresholds
        if thresholds_file:
            payload = json.loads(Path(thresholds_file).read_text(encoding="utf-8"))
            if payload["class_names"] != engine.class_names:
                raise ValueError("校准阈值的类别顺序不一致")
            if payload.get("tta") != bool(tta) or payload.get("model_count") != len(weights):
                raise ValueError("阈值的 TTA/模型数量与推理设置不一致")
            if payload.get("model_hashes") != engine.model_hashes:
                raise ValueError("校准阈值与当前权重文件不匹配")
            thresholds = np.asarray(payload["thresholds"])
        elif tta or len(weights) > 1:
            raise ValueError("TTA 或集成的二值输出需要独立校准阈值文件")
        if thresholds.shape != (len(engine.class_names),) or not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
            raise ValueError("阈值格式无效")
        probs = (probs >= thresholds).astype(int)
    id_col = next((c for c in ("id", "ID") if c in df), None)
    ids = df[id_col].tolist() if id_col else [Path(p).stem for p in paths]
    if any(pd.isna(x) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("输出 id 缺失或重复")
    output = pd.DataFrame(probs, columns=engine.class_names)
    output.insert(0, "id", ids)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(save_path, index=False, encoding="utf-8-sig")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_path")
    parser.add_argument("save_path")
    parser.add_argument("--weight_dir")
    parser.add_argument("--image_root")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--thresholds_file")
    parser.add_argument("--error_policy", choices=["raise", "black"], default="raise")
    parser.add_argument("--device", default="")
    predict(**vars(parser.parse_args()))
