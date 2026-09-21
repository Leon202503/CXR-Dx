# 权重放置目录

把训练导出的权重放到这里，例如：

```
predict/
  run.py
  requirements.txt
  weights/
    densenet121_f0_best.pth
    convnext_tiny_f0_best.pth    # 放多个权重会自动概率平均集成
```

注意：
- `run.py` 启动时会自动加载本目录下全部 `*.pth`；
- 权重 + 代码**总大小不得超过 2GB**；
- 权重由 `train.py` 保存，内含 backbone 名、类别顺序、输入尺寸、CLAHE 开关和每类阈值，
  因此提交包里不需要 yaml，也不会联网下载任何预训练权重。
