"""提交包结构、元数据、体积和禁止 Python socket 联网的独立进程冒烟检查。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

SMOKE = r'''
import socket
def blocked(*args, **kwargs):
    raise RuntimeError("离线检查禁止网络访问")
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
torch.set_num_threads(2)
import run
root = Path(sys.argv[1])
rows = []
for i in range(3):
    path = root / (str(i) + ".png")
    array = np.arange(4096, dtype=np.uint16).reshape(64,64) if i == 0 else np.full((64,64), i * 80, dtype=np.uint8)
    Image.fromarray(array).save(path)
    rows.append({"id": "00" + str(i), "image": path.name})
pd.DataFrame(rows).to_csv(root / "test.csv", index=False)
out = run.predict(root / "test.csv", root / "result.csv", batch_size=2, device="cpu")
assert out["id"].tolist() == ["000","001","002"]
weights = sorted((Path(run.__file__).parent / "weights").glob("*.pth"))
meta = torch.load(weights[0], map_location="cpu", weights_only=True)
assert out.columns.tolist() == ["id"] + meta["class_names"]
values = out.iloc[:,1:].to_numpy()
assert values.shape == (3, len(meta["class_names"]))
assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
print("独立进程离线推理通过：RGB/灰度16位输入、ID顺序、输出范围和形状")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="predict")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_mb", type=float, default=2048, help="工程默认预算，正式限制以官方为准")
    ap.add_argument("--allow_debug", action="store_true", help="仅开发自测，允许调试权重")
    args = ap.parse_args()
    root = Path(args.model_dir).resolve()
    for name in ("run.py", "runtime.py", "requirements.txt"):
        if not (root / name).is_file():
            raise FileNotFoundError(name)
    weights = sorted((root / "weights").glob("*.pth"))
    if not weights:
        raise ValueError("weights/ 下没有 .pth 权重")
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if size > args.max_mb * 1024 ** 2:
        raise ValueError("超过配置的提交体积预算")
    import torch
    names = None
    for path in weights:
        meta = torch.load(path, map_location="cpu", weights_only=True)
        required = {"state_dict", "backbone", "class_names", "img_size", "thresholds", "preprocess_version"}
        if required - meta.keys():
            raise ValueError(f"{path}: 缺少字段 {required - meta.keys()}")
        if meta.get("debug") and not args.allow_debug:
            raise ValueError("调试权重禁止作为正式提交，请完成正式训练")
        if names is not None and names != meta["class_names"]:
            raise ValueError("集成类别顺序不一致")
        names = meta["class_names"]
    if args.smoke:
        with tempfile.TemporaryDirectory() as temp:
            env = os.environ.copy()
            env["HF_HUB_OFFLINE"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            subprocess.run([sys.executable, "-c", SMOKE, temp], cwd=root, env=env,
                           check=True, timeout=300)
    print(json.dumps({"weights": len(weights), "size_mb": round(size / 1024**2, 2),
                      "smoke": args.smoke, "status": "passed"}))


if __name__ == "__main__":
    main()
