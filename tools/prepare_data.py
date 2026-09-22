"""把官方数据 / NIH ChestX-ray14 原始数据转成本工程统一的 csv。

统一 csv 三列：
    image_path,labels,patient_id
其中 labels 是与 config.class_names 等长、以 "|" 分隔的多热字符串，如 0|1|0|...|0

用法 A（ChestX-ray14 原始目录，内含 Data_Entry_2017.csv / train_val_list.txt / test_list.txt）：
    python tools/prepare_data.py --data_root D:/data/ChestX-ray14 --out data_csv

用法 B（任意官方标注 csv，脚本自动识别图片列/标签列/宽表）：
    python tools/prepare_data.py --csv path/to/train.csv --image_root path/to/images --out data_csv --out_name train
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.splits import patient_ids
from data.dataset import load_labels, _find_column  # noqa: E402

def build_index(root: str) -> dict:
    """递归建立 文件名 -> 相对路径 的索引（ChestX-ray14 图片分散在 images_xxx/images/ 下）。"""
    index = {}
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                if f in index:
                    raise ValueError(f"图片文件名重复，无法唯一索引: {f}")
                index[f] = os.path.relpath(os.path.join(dirpath, f), root)
    return index


def convert(df: pd.DataFrame, class_names, image_root: str = "", file_index: dict | None = None) -> pd.DataFrame:
    labels, _ = load_labels(df, class_names)
    pids = patient_ids(df)
    img_col = _find_column(df, ["image_path", "image", "img", "Image Index", "Image_Index", "file", "filename"])
    if img_col is None:
        raise ValueError(f"找不到图片列，实际列为 {list(df.columns)}")

    paths = []
    for name in df[img_col].astype(str):
        base = os.path.basename(name)
        if file_index is not None:
            rel = file_index.get(base, name)
        else:
            rel = name
        if image_root and not (Path(image_root) / rel).is_file():
            raise FileNotFoundError(str(Path(image_root) / rel))
        paths.append(rel)

    out = pd.DataFrame({
        "image_path": paths,
        "labels": ["|".join(str(int(x)) for x in row) for row in labels],
        "patient_id": pids,
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--data_root", default="", help="ChestX-ray14 根目录")
    ap.add_argument("--csv", default="", help="或直接指定单个标注 csv")
    ap.add_argument("--image_root", default="", help="csv 模式下图片根目录")
    ap.add_argument("--out", default="data_csv")
    ap.add_argument("--out_name", default="train")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.config, encoding="utf-8") as f:
        classes = yaml.safe_load(f)["class_names"]
    for name in (["train", "test"] if args.data_root else [args.out_name]):
        if (Path(args.out) / (name + ".csv")).exists() and not args.overwrite:
            raise FileExistsError("输出已存在；使用不同目录或显式 --overwrite")

    if args.data_root:
        root = args.data_root
        entry = None
        for dirpath, _, files in os.walk(root):
            if "Data_Entry_2017.csv" in files:
                entry = os.path.join(dirpath, "Data_Entry_2017.csv")
                break
        if entry is None:
            raise FileNotFoundError("未找到 Data_Entry_2017.csv")
        entry_dir = os.path.dirname(entry)
        df_all = pd.read_csv(entry, dtype=str)
        file_index = build_index(root)
        print(f"索引到 {len(file_index)} 张图片")

        def read_list(fn):
            p = os.path.join(entry_dir, fn)
            if not os.path.exists(p):
                return None
            with open(p) as f:
                return set(x.strip() for x in f if x.strip())

        train_val = read_list("train_val_list.txt")
        test_list = read_list("test_list.txt")
        img_col = "Image Index"

        if train_val is not None:
            df_tv = df_all[df_all[img_col].isin(train_val)].reset_index(drop=True)
            convert(df_tv, classes, root, file_index).to_csv(
                os.path.join(args.out, "train.csv"), index=False)
            print(f"train.csv: {len(df_tv)} 行")
        if test_list is not None:
            df_te = df_all[df_all[img_col].isin(test_list)].reset_index(drop=True)
            convert(df_te, classes, root, file_index).to_csv(
                os.path.join(args.out, "test.csv"), index=False)  # 保持独立测试集，不用于调参
            print(f"test.csv(独立留出集): {len(df_te)} 行")
        if train_val is None and test_list is None:
            convert(df_all, classes, root, file_index).to_csv(
                os.path.join(args.out, "train.csv"), index=False)
        print("类别顺序：", classes)
    else:
        if not args.csv:
            ap.error("必须提供 --data_root 或 --csv")
        df = pd.read_csv(args.csv, dtype=str)
        out = convert(df, classes, args.image_root or None)
        out_path = os.path.join(args.out, f"{args.out_name}.csv")
        out.to_csv(out_path, index=False)
        print(f"已写出 {out_path}: {len(out)} 行")


if __name__ == "__main__":
    main()
