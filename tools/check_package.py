"""提交前自检：目录结构、2GB 体积限制、权重完整性，并可断网冒烟测试 run.py。

用法：
    python tools/check_package.py --model_dir predict
    python tools/check_package.py --model_dir predict --smoke   # 生成假图实际跑一遍推理
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

LIMIT_BYTES = 2 * 1024 ** 3


def dir_size(root: str) -> int:
    total = 0
    for dp, _, fns in os.walk(root):
        for f in fns:
            total += os.path.getsize(os.path.join(dp, f))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="predict")
    ap.add_argument("--smoke", action="store_true", help="用随机假图断网跑一遍 run.predict")
    args = ap.parse_args()
    root = args.model_dir
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and cond

    check(os.path.isfile(os.path.join(root, "run.py")), "存在 run.py")
    check(os.path.isfile(os.path.join(root, "requirements.txt")), "存在 requirements.txt")
    weights = [os.path.join(dp, f) for dp, _, fns in os.walk(root) for f in fns if f.endswith((".pth", ".pt"))]
    check(len(weights) > 0, f"至少 1 个权重文件（当前 {len(weights)} 个）")

    size = dir_size(root)
    print(f"[INFO] 总体积：{size / 1024 ** 2:.1f} MB（限制 2048 MB）")
    check(size <= LIMIT_BYTES, "总体积 ≤ 2GB")

    # 权重内容检查
    try:
        import torch
        for w in weights:
            ckpt = torch.load(w, map_location="cpu", weights_only=False)
            need = ["state_dict", "backbone", "class_names", "img_size", "thresholds"]
            miss = [k for k in need if k not in ckpt]
            check(not miss, f"{os.path.basename(w)} 字段完整（缺：{miss}）")
    except ImportError:
        print("[WARN] 本地未装 torch，跳过权重内容检查")

    # 断网冒烟测试
    if args.smoke:
        try:
            import numpy as np
            import pandas as pd
            from PIL import Image
            sys.path.insert(0, os.path.abspath(root))
            import run as submission
            with tempfile.TemporaryDirectory() as td:
                img_dir = os.path.join(td, "images")
                os.makedirs(img_dir)
                rows = []
                for i in range(6):
                    p = os.path.join(img_dir, f"{i}.png")
                    Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8)).convert("RGB").save(p)
                    rows.append({"id": i, "image": f"images/{i}.png"})
                csv_path = os.path.join(td, "test.csv")
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                out_path = os.path.join(td, "result.csv")
                submission.predict(td, out_path)
                res = pd.read_csv(out_path)
                check(len(res) == 6, f"冒烟测试：输出 6 行（实际 {len(res)} 行）")
                check("id" in res.columns, "冒烟测试：输出含 id 列")
        except Exception as e:
            check(False, f"冒烟测试失败：{e}")

    print("\n" + ("全部检查通过，可以打包 zip。" if ok else "存在未通过项，修复后再提交。"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
