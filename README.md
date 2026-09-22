# 胸部 X 光多标签病变分类

面向“高精度、高鲁棒性的多标签胸部病变分类”比赛要求，提供数据检查、患者分组训练、折外评估、受控退化测试与离线提交工具。

当前输出是**图像级各病变概率**，没有边界框定位或分割。用户提供的比赛说明没有确定类别数、评分公式、运行镜像和提交接口；14 类 NIH 标签是当前开发配置，不应视为官方最终规范。旧版技术路线图仅为规划参考，实际功能以本 README 和代码为准。

## 当前状态

- 本地开发数据：5,606 张图片、4,230 名患者；14 类。图片路径保留在 configs/config.yaml。
- 本轮已完成逐图解码和五折患者隔离检查。
- 已用 CPU、随机初始化、16 张训练/16 张验证、64 像素输入、1 epoch 跑通训练和推理；这是流程验证，不能作为精度结论。
- 未完成正式 GPU 训练、全量五折实验或官方提交。不存在已经达到某个 AUC 的承诺。
- 技术总结：docs/胸部X光多标签分类项目技术总结.docx。

## 环境

推荐使用独立 Python 3.11 环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r docs/requirements.txt
```

本机本轮创建的 .venv 复用了已有的 CPU PyTorch，并在项目环境中补齐缺失依赖。下文命令在该环境激活后执行，或者将 python 替换为 .\.venv\Scripts\python.exe。GPU 训练前应确认 torch.cuda.is_available() 为 True，选择与服务器匹配的 PyTorch CUDA 构建。正式比赛镜像尚未确定。

## 数据准备与检查

统一 CSV：

```csv
image_path,labels,patient_id
example.png,0|1|0|0|0|0|0|0|0|0|0|0|0|0,patient_001
```

同时支持 NIH 病变名字符串和每类一列的宽表。所有类别必须显式提供；缺失值、未知类别、非 0/1 标签都报错。患者 ID 必须真实存在，不会用图片序号伪造。

```powershell
python tools/prepare_data.py --config configs/config.yaml --csv source.csv --image_root D:/images --out data_csv/new
python tools/check_data.py --config configs/config.yaml --decode
```

转换读取 config.class_names 的顺序。输出已有文件时默认拒绝覆盖，需要显式 --overwrite。NIH 原始目录模式 --data_root 会将官方 train_val_list 转为 train.csv，将 test_list 保留为 test.csv；独立测试集不应参与日常调参。

默认 GroupKFold 按患者五折；指定 val_csv 时检查训练和验证患者/图片是否重叠。患者 ID 别名包含 patient_id、Patient ID、Patient_ID、pid。患者分组不保证多标签分层，工具会报告每折各类阳性数。

## 训练

```powershell
# 小规模离线调试，无需下载预训练权重
python train.py --debug --no_pretrained --epochs 1 --img_size 64 --batch_size 4 --run_name debug_example

# 正式单折基线；首次预训练权重加载可能联网
python train.py --config configs/config.yaml --run_name densenet121_f0

# 更换 backbone / 分辨率，必须用固定划分对比
python train.py --backbone convnext_tiny --img_size 320 --loss asl --fold 0 --run_name convnext_tiny_f0

# PowerShell 五折示例
0..4 | ForEach-Object { python train.py --fold $_ --run_name "densenet121_f$_" }
```

训练采用 logits + BCE/加权 BCE/Focal/ASL；ASL 仅对负概率分支截断，避免原实现低概率阳性梯度为零。默认 AdamW、warmup + cosine、AMP（GPU）、EMA、梯度裁剪与早停。标签平滑默认关闭，单独消融后再启用。

默认增强包含小角度旋转、缩放、亮度/对比度与水平翻转；CLAHE 可选。涉及左右侧标签时应关闭水平翻转和翻转 TTA。图像级分类没有病灶定位标注时，不宣称实现定位检测。

每次训练输出：
- best.pth / last.pth：可推理权重，含类别、尺寸、预处理版本、训练患者哈希、fold、epoch、调试标记。
- train_split.csv / val_split.csv：实际使用的数据划分。
- val_predictions.npz：最优 epoch 的验证概率与标签。
- thresholds.json / metrics_history.json / config_snapshot.yaml：阈值、指标和配置。

同目录已有权重时拒绝覆盖，用 --run_name 创建新实验。当前权重不包含优化器和 scaler 状态，不支持精确断点续训。训练中验证集同时用于选 epoch 和阈值，因此这些分数是开发指标。

## 评估、校准与 OOF

```powershell
python evaluate.py --ckpt runs/densenet121_f0/best.pth --val_csv runs/densenet121_f0/val_split.csv --no_tta
python evaluate.py --ckpt runs/densenet121_f0/best.pth --val_csv data_csv/holdout.csv --calibration_csv data_csv/calibration.csv --tta --out runs/calibrated_eval
python tools/evaluate_oof.py --runs runs/densenet121_f0 runs/densenet121_f1 runs/densenet121_f2 runs/densenet121_f3 runs/densenet121_f4 --image_root D:/data/ChestX-ray14-sample/sample/images
```

评估前核对每个权重的训练患者记录，发现重叠即拒绝。不能将五折模型一起预测某一折验证集，再把结果称为独立验证。OOF 工具只让对应折模型预测该折未见患者，并拒绝折间重复患者。

mAUC 只平均同时有正负样本的类别，并报告有效类别数；mAP 只平均有阳性样本的类别；macro-F1 覆盖全部类别、零分母按 0。正式评分口径公布后需对齐。训练阈值网格为 0.01～0.99；TTA/集成阈值不能直接沿用单模型阈值，应使用与评估患者独立的校准集。导出的校准文件绑定模型 SHA256、类别、TTA 和模型数量。

无标签推理不依赖验证集：

```powershell
python evaluate.py --ckpt runs/densenet121_f0/best.pth --test_only --test_csv data_csv/test_unlabeled.csv --image_root D:/test/images --out runs/predictions
```

## 鲁棒性评估

```powershell
python tools/evaluate_robustness.py --ckpt runs/densenet121_f0/best.pth --val_csv runs/densenet121_f0/val_split.csv --image_root D:/data/ChestX-ray14-sample/sample/images --limit 200
```

固定随机种子抽样，比较原图与亮度 0.7、对比度 0.7、高斯模糊半径 1、JPEG 质量 40，输出各条件指标和 mAUC 变化。--limit 0 使用全量。该检查测量受控图像退化表现，不能替代跨医院、跨设备或临床验证。

8 位图像转 RGB；16 位/浮点灰度按单图 min-max 映射到 8 位，避免直接转换饱和。此映射可能受离群像素影响，需要在真实数据上验证。损坏图片默认报错；提交可显式 --error_policy black 并输出警告，不静默掩盖错误。未实现 DICOM 窗宽窗位处理。

## 打包与离线检查

```powershell
python tools/build_package.py --ckpt runs/densenet121_f0/best.pth --out artifacts/submission
python tools/check_package.py --model_dir artifacts/submission --smoke
python artifacts/submission/run.py test.csv result.csv --image_root D:/test/images
```

包中只有 run.py、runtime.py、requirements.txt、weights/*.pth。模型与预处理均来自 predict/runtime.py，避免训练/提交双份实现漂移。保存 CSV 时保持输入顺序，字符串 ID 保留前导零。

默认输出概率。二值输出使用 --binary；TTA/集成还必须提供 --thresholds_file runs/calibrated_eval/thresholds.json。--tta 和 --batch_size 可选。路径默认相对测试 CSV 所在目录解析，也可指定 --image_root。

自检 --smoke 在独立进程中阻断常见 Python socket 连接，并检查输出类别、ID、形状、有限数值与概率范围；不是操作系统级网络隔离。默认体积预算 2048 MB 可通过 --max_mb 修改，不能视为官方限制。调试权重默认拒绝正式自检；--allow_debug 仅用于开发流程验证。自检通过仍需对齐官方 run.py 固定接口与实际镜像。

## 测试与文档

```powershell
python -m unittest discover -s tests -v
python tools/generate_technical_report.py
```

自动化测试覆盖 ASL 数值/梯度、标签格式、患者隔离、16 位读图、异常策略、CLAHE 多进程序列化、指标口径、推理一致性、集成类别检查和阈值校准约束。Word 文档由脚本生成，实验记录位于 artifacts/verification.json 和 artifacts/data_audit.json。

后续正式实验顺序：可信单模型基线 → 损失/增强消融 → 更高分辨率与不同 backbone → 五折 OOF → 独立校准与鲁棒性评估 → 官方镜像提交验证。所有精度增益均需实测。

ASL 原始实现参考：https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
