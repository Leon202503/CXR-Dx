# 全国高校计算机能力挑战赛 · 人工智能挑战赛（2026 第八届）
## 胸部 X 光片多标签病变分类 —— 参赛方案与 Baseline 工程

> 赛题：开发**高精度、高鲁棒性的多标签分类算法**，对一张胸部 X 光片（Chest X-ray）同时检测/分类多种胸部病变，服务于计算机辅助诊断（CAD）。
> 官网：http://www.ncccu.org.cn/index/Paper/case1.html
> 数据集开放：**2026-10-15 15:00**；区域赛截止：**2026-11-13 23:59**；国赛：11-19 ~ 11-24。

---

## 一、赛题本质：这是一个什么任务

| 关键点 | 说明 |
| --- | --- |
| 任务类型 | **多标签图像分类**（multi-label classification）。一张片子可能同时患有多种病变，14（或 N）个标签互相独立，每个输出一个 0~1 概率，**不是单分类 softmax** |
| 输出 | 对每张图输出每个病变类别的概率；最终二值化阈值通常按验证集搜索（默认 0.5） |
| 最可能的数据来源 | 该赛题是医学影像经典基准，几乎可以确定基于 **NIH ChestX-ray14**（112,120 张正位片、14 类病变、标签由报告 NLP 抽取、含噪声、长尾）或其裁剪版；也可能混入 CheXpert / PadChest。**类别清单与指标以 10-15 官方发布为准**，本工程已全部配置化 |
| 主要难点 | ① 类别极不均衡（长尾，Hernia 阳性率约 0.2%，Infiltration 约 17%）；② 标签有噪声；③ 多设备/多医院域差异（鲁棒性要求）；④ 病变区域小（结节）、纹理细（纤维化） |

ChestX-ray14 的 14 个标签（标准顺序，配置文件里可改）：
```
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis,
Pleural_Thickening, Hernia
```

## 二、往届规则（2024 第六届人工智能挑战赛，今年大概率沿用）

- **提交物是"模型 + 代码"的 zip 包**，不是只交结果 csv；测试集不开放，在官方服务器后台跑推理，**推理环境无网络**。
- 目录结构固定：
  ```
  model/
    ├── requirements.txt     # 所有依赖及版本
    ├── run.py               # 推理入口，固定区域不允许修改
    ├── *.pth                # 模型权重
    └── 其他代码文件
  ```
- 服务器：Python 3.8 + CUDA 11.3/12.1（以官方上线为准）；**模型权重 + 代码总体积 ≤ 2GB**。
- 禁止调用任何需要联网/API 的大模型；允许用预训练权重，但**必须有自己的工作量**（继续训练、结构改进、新模块等），并在技术报告中说明。
- 每天提交次数有限（去年每天 1 次），**务必本地先把离线验证流程做扎实**。
- 去年二分类指标是 F1；今年多标签任务的主流指标是 **macro mean AUC（每类 AUC 再平均）**，也可能用 mAP / macro-F1。本工程三个指标全部计算，以官方公布为准。

## 三、技术路线（三档投入，按时间/算力选择）

| 档位 | 方案 | 预期 mAUC（ChestX-ray14 口径） | 适用 |
| --- | --- | --- | --- |
| **Baseline（必做）** | DenseNet121 / ResNet50，ImageNet 预训练，BCEWithLogits，224 输入 | 0.80~0.83 | 保底、跑通全流程、高职组/省奖 |
| **主力（推荐）** | ConvNeXt-Tiny / EfficientNetV2-S / Swin-T，320 输入，ASL 非对称损失 + 阳性率权重，EMA + 每类阈值 + TTA，5 折集成 2~3 个异构 backbone | 0.84~0.87 | 冲省一/国奖 |
| **进阶（有余力）** | 医学预训练权重（RadImageNet / BioViL / MedKLIP）、分辨率 384 微调、知识蒸馏、报告一致性约束、Style/域随机化增强提升鲁棒性 | 0.87+ | 冲前三 |

> 参考坐标：CheXNet（DenseNet121，2017）官方测试集 mAUC ≈ 0.835；现代 timm backbone + 上述技巧普遍 0.85+。**具体数值取决于官方数据划分，不要写死预期。**

提分优先级（按性价比排序）：
1. **严格按患者划分训练/验证**（ChestX-ray14 自带 train_val_list.txt / test_list.txt，杜绝患者泄漏）；
2. 解决长尾：**ASL 非对称损失 或 按阳性率倒数的加权 BCE**；
3. **每类阈值搜索**（在验证集上为每个类选最优阈值，macro-F1 可涨 2~5 个点；AUC 不受阈值影响）；
4. **EMA 权重滑动平均 + TTA（水平翻转）**；
5. **异构模型集成**（CNN + Transformer 概率平均，通常 +0.5~1 个点）；
6. 输入分辨率 224 → 320/384 微调（小结节类收益明显）；
7. 医学领域预训练权重替换 ImageNet 权重。

## 四、工程结构

```
AIChallengeCompetition/
├── configs/config.yaml        # 全部超参与路径、类别清单（数据发布后主要改这里）
├── data/
│   ├── dataset.py             # 多标签 Dataset，兼容两种标注格式
│   └── transforms.py          # 训练/测试增强（含 CLAHE、TTA）
├── models/model.py            # timm backbone + 多标签分类头
├── losses.py                  # BCE / 加权BCE / Focal / ASL 非对称损失
├── metrics.py                 # 每类AUC、mAUC、mAP、macro-F1、阈值搜索
├── train.py                   # 训练入口：AMP/EMA/cosine/按患者划分/按mAUC存最优
├── evaluate.py                # 离线评估 + 阈值导出 + TTA
├── tools/
│   ├── prepare_data.py        # 把官方/ChestX-ray14 数据转成统一 csv
│   └── check_package.py       # 提交前自检：目录结构、体积(≤2GB)、离线可跑
├── predict/                   # 提交包（zip 这个目录里的内容）
│   ├── run.py                 # 推理入口模板（等官方 submit_example 后对齐固定接口）
│   ├── requirements.txt
│   └── (训练导出的 *.pth 放这里)
└── docs/技术路线图.html
```

## 五、环境搭建（重要：不要用系统自带的 Python 3.14）

PyTorch 对最新 Python 版本支持滞后，建议用 conda（Miniconda 即可）建独立环境：

```powershell
conda create -n cxr python=3.10 -y
conda activate cxr
# 有 NVIDIA 显卡（CUDA12.1）：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 无显卡（仅调试，训练需 GPU，可白嫖 Colab/Kaggle/AutoDL 租卡）：
pip install torch torchvision
pip install -r requirements.txt
```

显卡显存参考：224 分辨率 + DenseNet121 约 6GB；320 + ConvNeXt-T 约 8~10GB；384 + Swin-B 约 16GB。
**ChestX-ray14 全量训练（11 万张）单卡 3090/4090 约 6~10 小时/模型，5 折 ×3 模型约 2~4 天，提前规划。**

## 六、分步操作

### 第 0 步（现在 ~10-15，数据未发布）：用公开 ChestX-ray14 预研
1. 下载 NIH ChestX-ray14（约 42GB，Kaggle 搜 "Chest X-Ray Images (Pneumonia)" 不对，应搜 **"ChestX-ray14" / data.gov 的 ChestXray-NIHCC**；含 `Data_Entry_2017.csv`、`train_val_list.txt`、`test_list.txt`）。
   - 磁盘紧张可先下 224 分辨率重压缩版（Kaggle 上有社区版本）做流程调试。
2. 生成统一标注：
   ```powershell
   python tools/prepare_data.py --data_root D:\data\ChestX-ray14 --out data_csv
   ```
3. 跑通 baseline：
   ```powershell
   python train.py --config configs/config.yaml --backbone densenet121 --epochs 10 --debug
   ```

### 第 1 步（10-15 官方数据发布后）
- 下载官方数据，**先读官方 README/样例提交**，核对：类别清单与顺序、训练/测试目录结构、标注 csv 字段、`run.py` 固定接口、评价指标。
- 改 `configs/config.yaml`：`num_classes`、`class_names`、数据路径、指标口径；用 `tools/prepare_data.py` 转格式（如官方格式已兼容可跳过）。

### 第 2 步：训练与验证
```powershell
# 单折训练（主力模型）
python train.py --config configs/config.yaml --backbone convnext_tiny --img_size 320 --loss asl --fold 0
# 5 折（每折一个 fold，脚本会按患者 id 分组）
for /L %f in (0,1,4) do python train.py --config configs/config.yaml --fold %f
```
- 脚本自动：按 mAUC 保存最优权重、导出每类最优阈值 `thresholds.json`、画每类 AUC 表。

### 第 3 步：离线评估 + 集成 + TTA
```powershell
python evaluate.py --config configs/config.yaml --ckpt runs/convnext_tiny_f0/best.pth --tta
```

### 第 4 步：打包提交
1. 把权重和 `thresholds.json` 放进 `predict/`；按官方 `submit_example.zip` **对齐 `run.py` 固定区域**（读测试路径、输出 csv 的字段名）。
2. 自检：
   ```powershell
   python tools/check_package.py --model_dir predict
   ```
3. 压缩 `predict/` 内全部文件为 zip 提交；**先用少量图片在本地断网模拟跑一遍 run.py**。

## 七、时间规划（倒排）

| 时间 | 任务 |
| --- | --- |
| 现在 ~ 10-14 | 搭环境、用公开 ChestX-ray14 跑通 baseline、读完 2~3 篇参考方案（CheXNet、ASL 论文） |
| 10-15 ~ 10-21 | 对齐官方数据格式，完成 baseline 并拿到首个离线 mAUC；建立"按患者划分"的可靠验证集 |
| 10-22 ~ 11-02 | 主力模型：换 backbone、ASL/加权损失、EMA、阈值搜索、TTA；做错误分析 |
| 11-03 ~ 11-09 | 5 折 + 异构集成、分辨率微调、鲁棒性增强；开始写技术报告 |
| 11-10 ~ 11-13 | 冻结模型，打包、断网实测 run.py、留 buffer 提前 1 天提交 |
| 晋级后 11-19 ~ 11-24 | 国赛通常换隐藏测试集/限时重训，保证代码可复现、一键训练 |

## 八、避坑清单

1. **不要随机划分图片**：同一患者的正侧位/多次拍片必须整体进同一折，否则指标虚高、上线崩。
2. **不要用 softmax + 交叉熵**：多标签必须 sigmoid + BCE 系损失。
3. **验证集评估要输出"每类 AUC"**：长尾类（Hernia、Pneumonia、Fibrosis）是拉开差距的关键。
4. 官方标签若是 NLP 自动标注（带噪声），不要在训练集上过拟合，增强和 EMA 比硬背标签更有效；**严禁用测试集任何信息**（作弊取消资格）。
5. 提交包在**无网络、仅 CPU 也可能被调度**的环境下要能跑：timm 等库的预训练权重下载缓存要去掉，权重全部本地加载（run.py 里 `pretrained=False` 再 load 本地权重）。
6. 读图用 `cv2.IMREAD_UNCHANGED` / PIL 转 RGB 统一处理 8/16 位、单通道/三通道、png/jpg；损坏图要有 try/except 兜底（鲁棒性评分点）。
7. 依赖版本锁定且与官方 Python 3.8/CUDA 镜像兼容；避免只支持高版本 GLibc 的库。
8. 每天只有 1 次提交机会，**提交次数留给验证过的版本**，不要拿官方平台当调试器。

## 九、参考资料（训练前精读）

- Rajpurkar et al., *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*, 2017（DenseNet121 baseline 源头）
- Wang et al., *ChestX-ray8 / ChestX-ray14*, 2017（数据集与评测协议）
- Ben-Baruch et al., *Asymmetric Loss For Multi-Label Classification (ASL)*, ICCV 2021（长尾多标签标配损失）
- Ridnik et al., *TResNet* / 后续 ChestX-ray14 高分方案（高分辨率、增强策略）
- timm 文档：https://huggingface.co/docs/timm （backbone 选型）
