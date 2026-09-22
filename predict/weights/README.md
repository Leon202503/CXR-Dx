# 模型权重

正式训练完成后使用工具导出自包含提交包：

```powershell
python tools/build_package.py --ckpt runs/densenet121_f0/best.pth --out artifacts/submission
python tools/check_package.py --model_dir artifacts/submission --smoke
```

提交包包含 run.py、runtime.py、requirements.txt 和 weights/*.pth。多个模型必须使用完全一致的类别顺序及预处理版本。只放需要参与推理的权重，避免 best/last 重复集成。

旧版工程权重缺少预处理版本与患者记录，需要重新训练。调试权重会被正式打包检查拒绝。
