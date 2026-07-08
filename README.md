# Fatigue-Contrast: 跨对象疲劳检测对比实验框架

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=200&text=Fatigue%20Detection%20Baselines&fontSize=36&fontAlign=50&fontColor=28F2E6&color=0:0B2447,50:5B2B8A,100:00A586&desc=Cross-Subject%20Fatigue%20Detection%20Contrast%20Experiments&descAlign=50&descAlignY=78&descSize=16&descColor=B1FBE4&animation=twinkling"/>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch"></a>
  <a href="https://github.com/state-spaces/mamba"><img src="https://img.shields.io/badge/Mamba--SSM-2.0+-green.svg" alt="Mamba"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

---

## 📖 项目简介

本项目是**跨对象（Cross-Subject）疲劳检测**的对比实验框架，基于 PyTorch 构建。实现了多种经典基线方法，用于与本文提出的 SelfNet 方法进行性能对比。

### 支持的基线方法

| 类别 | 模型 | 说明 |
|:---:|:---:|:---|
| **时序分类** | LSTM | 长短时记忆网络，经典的时序建模方法 |
| | Transformer | 基于自注意力的时序编码器 |
| | Mamba | 基于选择性状态空间模型（S4/S6），线性复杂度的时序建模 |
| | TimesNet | 时序 2D 变化建模，FFT 多周期发现 + Inception 2D 卷积，ICLR 2023 |
| **小样本学习** | ProtoNet | 原型网络，基于欧氏距离的度量学习 |
| | RelationNet | 关系网络，基于可学习关系模块的度量学习 |
| **域适应** | MLDA | 多级域适应，Wasserstein 域间对齐 + ICD 类条件对比 |
| | DAEEGViT | 域适应 ViT + MBConv，CLS token MMD 对齐 |
| | LA-MSDA | 多源标签域适应，LLMMD 标签条件对齐 + 多分类器共识 |
| | DANN | 域对抗训练，梯度反转层 (GRL) + 域分类器对抗 |
| | DeepCORAL | 协方差对齐，二阶统计量 (CORAL) 分布匹配 |
| **域泛化** | InterpretableCNN | 可解释 CNN，空间-时间可分离卷积，纯 ERM 跨被试泛化 |
| | AFM-CIR | 因果域泛化，自适应傅里叶 Mixup + 因果启发回归，TPAMI 2026 |

---

## 📂 项目结构

```text
Fatigue-Contrast-exp/
├── configs/
│   ├── config.py                         # 原始模板配置
│   ├── experiments_object.py             # 原始模板实验字典
│   ├── fatigue_temporal_baselines.py     # 时序基线配置 (LSTM/Transformer/Mamba)
│   ├── fatigue_fewshot_baselines.py      # 小样本基线配置 (ProtoNet/RelationNet)
│   ├── fatigue_domain_adapt_baselines.py # 域适应基线配置 (MLDA/DAEEGViT/LA-MSDA/DANN/DeepCORAL)
│   └── fatigue_domain_generalization.py  # 域泛化基线配置 (InterpretableCNN/AFM-CIR)
├── data/
│   ├── dataset.py                        # 原始图像数据集
│   └── fatigue_dataset.py                # 疲劳检测数据集 (JSONL 滑动窗口)
├── models/
│   ├── get_model.py                      # 模型工厂（统一入口）
│   ├── lstm.py                           # LSTM 分类器
│   ├── transformer_encoder.py            # Transformer Encoder 分类器
│   ├── mamba_model.py                    # Mamba 分类器 (基于 mamba-ssm)
│   ├── timesnet_model.py                 # TimesNet 分类器 (FFT 周期发现 + 2D Inception)
│   ├── protonet.py                       # Prototypical Networks
│   ├── relationnet.py                    # Relation Network
│   ├── mlda_model.py                     # MLDA 域适应模型 (Encoder+Classifier+U/V)
│   ├── daeevit_model.py                  # DAEEGViT 域适应 ViT+MBConv 模型
│   ├── lamsda_model.py                   # LA-MSDA 多源域适应模型 (SharedNet+DSCNN×N)
│   ├── dann_model.py                     # DANN 域对抗模型 (Encoder+Classifier+GRL+DomainCls)
│   ├── deepcoral_model.py                # DeepCORAL 域适应模型 (Encoder+Classifier+CORAL损失)
│   ├── interpcnn_model.py                # InterpretableCNN 域泛化模型 (可分离卷积)
│   ├── afmcir_model.py                   # AFM-CIR 因果域泛化模型 (AFM增强+CIR训练)
│   └── ...                               # 原始模板模型 (ResNet/EfficientNet 等)
├── utils/
│   ├── basic.py                          # 优化器/学习率调度器
│   ├── loss_function.py                  # 损失函数 (FocalLoss 等)
│   ├── mlda_loss.py                      # MLDA 域适应损失 (ICD + Wasserstein)
│   ├── observer.py                       # 指标监控/日志/早停
│   ├── metrics_recorder.py               # CSV 指标记录器
│   ├── reproducibility.py                # 可复现性工具
│   └── swanlab_logger.py                 # SwanLab 实验跟踪
├── MLDA/                                 # MLDA 原始参考实现 (EEG, 独立运行)
│   ├── main.py                           # 原始训练脚本
│   ├── network.py                        # 原始网络定义
│   ├── idcd.py                           # 原始 ICD 损失
│   └── load_data2.py                     # 原始 .mat 数据加载
├── DAEEGViT/                             # DAEEGViT 原始参考实现 (EEG, 独立运行)
│   ├── DAEEGViT.py                       # 原始模型代码
│   └── *.pdf                             # 论文 PDF
├── LA_MSDA/                              # LA-MSDA 原始参考实现 (EEG, 独立运行)
│   ├── main_mutil_sources_LAMSDA.py      # 原始训练脚本
│   ├── DataLoader/                       # 数据加载模块
│   ├── LAMSDA_Modules/                   # 模型和损失模块
│   └── *.pdf                             # 论文 PDF
├── EEG-based-Cross-Subject-Driver-.../   # InterpretableCNN 原始参考实现 (EEG, 独立运行)
│   ├── InterpretableCNN.py               # 原始模型 (2D Conv, 30ch EEG)
│   ├── LeaveOneOut_acc.py                # 原始 LOSO 训练脚本
│   ├── VisTechnique.py                   # 可解释性可视化
│   └── *.pdf                             # 论文 PDF
├── tests/
│   ├── test_dann_deepcoral.py            # DANN/DeepCORAL 单元测试
│   └── test_interpcnn.py                 # InterpretableCNN 单元测试
├── main_fatigue.py                       # 疲劳检测训练入口 ⭐
├── main.py                               # 原始模板训练入口
├── requirements.txt                      # 依赖清单
└── README.md                             # 本文档
```

---

## 🛠️ 环境安装

### 基础环境

```bash
# 创建虚拟环境
conda create -n fatigue python=3.10
conda activate fatigue

# 安装 PyTorch (根据你的 CUDA 版本选择)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装核心依赖
pip install torchmetrics numpy scikit-learn scipy Pillow tqdm tensorboard matplotlib seaborn PyYAML
```

### 安装 Mamba (Linux + CUDA)

Mamba 需要 CUDA 环境编译，**仅支持 Linux**：

```bash
# 安装 causal-conv1d
pip install causal-conv1d

# 安装 mamba-ssm
pip install mamba-ssm
```

> ⚠️ **注意**: `mamba-ssm` 需要 CUDA toolkit 和对应版本的 PyTorch。如果安装失败，请参考 [mamba-ssm 官方文档](https://github.com/state-spaces/mamba)。

### 可选依赖

```bash
pip install ptflops    # 模型 FLOPs 计算
pip install swanlab    # 实验跟踪
pip install timm       # PoolFormer 等模型
pip install -r requirements.txt   # 一键安装全部依赖
```

---

## 📊 数据格式

数据为 JSONL 文件，每个文件代表一个受试者的一次实验：

```
[id]_[easy|hard]_[alert|sleepy].jsonl
```

示例：
```
001_easy_alert.jsonl
002_hard_sleepy.jsonl
```

每行是一帧数据，包含 `deviation_px_before_calibrate` 等特征：

```json
{
  "timestamp": 4.16,
  "frame_idx": 100,
  "deviation_px_before_calibrate": 25.3,
  "deviation_px_after_calibrate": 13.19,
  ...
}
```

数据目录结构：
```
data_dir/
├── 001_easy_alert.jsonl
├── 001_easy_sleepy.jsonl
├── 001_hard_alert.jsonl
├── 001_hard_sleepy.jsonl
├── 002_easy_alert.jsonl
└── ...
```

---

## 🚀 快速开始

### 1️⃣ 修改配置

打开对应的配置文件，修改以下关键参数：

```python
# configs/fatigue_temporal_baselines.py
# configs/fatigue_fewshot_baselines.py
# configs/fatigue_domain_adapt_baselines.py
# configs/fatigue_domain_generalization.py

# 数据目录（修改为你服务器上的实际路径）
"data_dir": "/your/path/to/jsonl/data",

# 任务难度过滤
"difficulty": "easy",   # "easy" / "hard" / None(全部)

# 测试集受试者ID（可选，None 表示无独立测试集）
"test_ids": ["010", "011"],

# 验证策略
"val_strategy": "kfold",  # "kfold"（手动指定）/ "loso"（自动留一）

# K-Fold 配置（每个 fold 指定验证集受试者ID，loso 模式下不需要）
"folds": {
    1: {"val_ids": ["001", "002"]},
    2: {"val_ids": ["003", "004"]},
    3: {"val_ids": ["005", "006"]},
},

# GPU 设备
"device": "cuda:0",
```

### 2️⃣ 训练

```bash
# ---- 时序基线 ----
python main_fatigue.py --exp_name Fatigue_LSTM_baseline
python main_fatigue.py --exp_name Fatigue_Transformer_baseline
python main_fatigue.py --exp_name Fatigue_Mamba_baseline
python main_fatigue.py --exp_name Fatigue_TimesNet_baseline

# ---- 小样本基线 ----
python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline
python main_fatigue.py --exp_name Fatigue_RelationNet_baseline

# ---- 域适应基线 ----
python main_fatigue.py --exp_name Fatigue_MLDA_baseline
python main_fatigue.py --exp_name Fatigue_DAEEGViT_baseline
python main_fatigue.py --exp_name Fatigue_LA_MSDA_baseline
python main_fatigue.py --exp_name Fatigue_DANN_baseline
python main_fatigue.py --exp_name Fatigue_DeepCORAL_baseline

# ---- 域泛化基线 ----
python main_fatigue.py --exp_name Fatigue_InterpCNN_baseline
python main_fatigue.py --exp_name Fatigue_AFM_CIR_baseline
```

### 3️⃣ 输出

每次训练会在 `output_dir` 下生成带时间戳的目录：

```
result_20260630_143000_Fatigue_LSTM_baseline/
├── config.yaml                    # 当前实验的完整配置（自动保存）
├── best_model_fold1.pth           # Fold 1 最佳模型
├── best_model_fold2.pth           # Fold 2 最佳模型
├── best_model_fold3.pth           # Fold 3 最佳模型
├── best_results.csv               # ⭐ 各 fold 最佳轮 train+val 指标（含 mean/std）
├── history_fold01.csv             # ⭐ Fold 1 每轮 train/val/test 指标（实时追加）
├── history_fold02.csv             # ⭐ Fold 2 每轮 train/val/test 指标（实时追加）
├── history_fold03.csv             # ⭐ Fold 3 每轮 train/val/test 指标（实时追加）
├── summary/                       # TensorBoard 日志
└── log.txt                        # 训练日志
```

如果配置了 `test_ids`，训练完成后会自动加载每个 fold 的模型进行批量测试评估，并输出汇总结果。

#### 📄 指标 CSV 文件说明

| 文件 | 内容 |
|:---|:---|
| `history_fold{N}.csv` | 每个 fold / LOSO **单独一个文件**，**每训练一轮实时追加** train + val 指标到文件末尾。长表格式，`split` 列区分 `train` / `val` / `test`，`is_best` 列标记该轮是否为当前 fold 的最佳轮；混淆矩阵展平为 `cm_00 / cm_01 / cm_10 / cm_11` 四个独立列。即使训练中途中断，已完成的轮次指标也不会丢失。 |
| `best_results.csv` | 每个 fold / LOSO **最佳轮**的训练集 + 验证集全部指标（宽表，`train_*` / `val_*` 并排），每折完成时实时追加一行；全部 fold 结束后追加 `mean` / `std` 汇总行。 |

---

## ⚙️ 配置说明

### 通用配置（所有模型共用）

| 参数 | 说明 | 默认值 |
|:---|:---|:---:|
| `data_dir` | JSONL 数据目录 | 必填 |
| `difficulty` | 任务难度过滤 | `None`（全部） |
| `test_ids` | 测试集受试者ID列表 | `None` |
| `val_strategy` | 验证策略 | `"kfold"` |
| `folds` | K-Fold 验证集配置（loso 模式下忽略） | `{}` |
| `feature_name` | 使用的特征字段 | `"deviation_px_before_calibrate"` |
| `window_size` | 滑动窗口大小（帧数） | 256 |
| `stride` | 滑动窗口步长 | 64 |
| `use_adf` | 是否使用 ADF 三通道特征 | `True` |
| `batch_size` | 批大小 | 128 |
| `epochs` | 最大训练轮数 | 1000 |
| `patience` | 早停耐心值 | 1000 |
| `lr` | 学习率 | 1e-3 |
| `lr_policy` | 学习率调度策略 | `"onecycle"` |
| `device` | GPU 设备 | `"cuda:0"` |
| `seed` | 随机种子 | 42 |

### 时序基线特有参数 (LSTM / Transformer / Mamba)

| 参数 | 说明 | 默认值 |
|:---|:---|:---:|
| `hidden_size` | LSTM 隐藏层维度 | 64 |
| `num_layers` | LSTM/Transformer 层数 | 2 |
| `d_model` | Transformer/Mamba 模型维度 | 64 |
| `nhead` | Transformer 注意力头数 | 4 |
| `d_state` | Mamba SSM 状态维度 | 16 |

### 时序基线特有参数 (TimesNet)

| 参数 | 说明 | 默认值 |
|:---|:---|:---:|
| `d_model` | 模型维度 (论文: min(max(2·ceil(log2(C)),32),64)) | 32 |
| `d_ff` | Inception 瓶颈层维度 | 64 |
| `num_kernels` | Inception 核数量 (核大小 1,3,5,7,9,11) | 6 |
| `top_k` | FFT top-k 周期数 (论文分类默认 3) | 3 |
| `e_layers` | TimesBlock 层数 (论文分类默认 2) | 2 |
| `dropout` | Dropout 率 | 0.1 |

### 小样本基线特有参数 (ProtoNet / RelationNet)

| 参数 | 说明 | 默认值 |
|:---|:---|:---:|
| `hidden_size` | 编码器隐藏层维度 | 64 |
| `embedding_size` | 嵌入空间维度 | 32 |
| `n_way` | 每个 episode 的类别数 | 2 |
| `k_shot` | 支持集样本数/类别 | 5 |
| `n_query` | 查询集样本数/类别 | 10 |
| `episodes_per_epoch` | 每个 epoch 的 episode 数 | 100 |

### 域适应基线特有参数 (MLDA)

| 参数 | 说明 | 默认值 |
|:---|:---|:---:|
| `feat_dim` | 编码器输出特征维度 | 32 |
| `proj_dim` | U/V 投影网络维度 | 32 |
| `dropout` | 编码器 Dropout 率 | 0.05 |
| `optimizer_name` | 优化器 (原论文使用 SGD) | `"SGD"` |
| `lr` | 学习率 | 5e-3 |
| `weight_decay` | 权重衰减 | 5e-4 |
| `batch_size` | 批大小 (源域=目标域) | 64 |
| `mlda_loss_weight` | 域间/域内损失平衡权重 λ | 0.5 |
| `mlda_lambda_center` | sigmoid 调度中心 epoch | 100 |
| `label_smoothing` | 标签平滑 (建议关闭) | 0.0 |

### 域适应基线特有参数 (DAEEGViT)

| 参数 | 说明 | 默认值 |
|:---|:---|:---:|
| `embed_dim` | ViT 嵌入维度 D | 64 |
| `depth` | Transformer 层数 | 4 |
| `num_heads` | 注意力头数 | 4 |
| `patch_size` | Patch 大小 (需整除 window_size) | 32 |
| `mlp_ratio` | MLP 隐藏层比率 | 4.0 |
| `dropout` | Dropout 率 | 0.1 |
| `drop_path_ratio` | Stochastic Depth 率 | 0.1 |
| `mbconv_expand_ratio` | MBConv 扩展比率 | 4 |
| `mbconv_se_ratio` | MBConv SE 比率 | 0.25 |
| `mmd_weight` | MMD 损失权重 | 1.0 |

### 域适应基线特有参数 (LA-MSDA)

| 参数 | 说明 | 默认值 |
|:---|:---|:---:|
| `num_sources` | 源域分支数量上限 | 5 |
| `feature_dim` | 共享特征提取器输出维度 | 64 |
| `ds_hidden_dim` | 域特定网络隐藏维度 | 256 |
| `optimizer_name` | 优化器 (原论文使用 SGD) | `"SGD"` |
| `lr` | 学习率 | 1e-3 |
| `epochs` | 训练轮数 | 500 |
| `da_warmup_scale` | sigmoid 预热陡峭度 | 10.0 |

### 域适应基线特有参数 (DANN)

| 参数 | 说明 | 默认值 |
|:---|:---|:---:|
| `feat_dim` | 编码器输出特征维度 | 32 |
| `dropout` | 编码器 Dropout 率 | 0.05 |
| `domain_hidden` | 域分类器隐藏层维度 | 1024 |
| `dann_gamma` | GRL sigmoid 调度陡峭度 (论文 Eq.9) | 10.0 |
| `optimizer_name` | 优化器 (原论文使用 SGD) | `"SGD"` |
| `lr` | 学习率 | 5e-3 |
| `weight_decay` | 权重衰减 | 5e-4 |
| `batch_size` | 批大小 (源域=目标域) | 64 |

### 域适应基线特有参数 (DeepCORAL)

| 参数 | 说明 | 默认值 |
|:---|:---|:---:|
| `feat_dim` | 编码器输出特征维度 | 32 |
| `dropout` | 编码器 Dropout 率 | 0.05 |
| `coral_weight` | CORAL 损失权重 (论文 Eq.2) | 1.0 |
| `optimizer_name` | 优化器 (原论文使用 SGD) | `"SGD"` |
| `lr` | 学习率 | 5e-3 |
| `weight_decay` | 权重衰减 | 5e-4 |
| `batch_size` | 批大小 (源域=目标域) | 64 |

### 域泛化基线特有参数 (InterpretableCNN)

| 参数 | 说明 | 默认值 |
|:---|:---|:---:|
| `n_filters` | 通道混合滤波器数 N1 | 16 |
| `depth_multiplier` | 深度可分离乘数 d | 2 |
| `kernel_size` | 时间卷积核大小 | 64 |
| `dropout` | 分类器前 Dropout 率 | 0.0 |
| `optimizer_name` | 优化器 (原论文使用 Adam) | `"AdamW"` |
| `lr` | 学习率 | 1e-3 |
| `weight_decay` | 权重衰减 | 1e-2 |

### 域泛化基线特有参数 (AFM-CIR)

AFM-CIR 是因果域泛化方法，三阶段训练：引导编码器预训练 → AFM 频域增强 → 因果启发训练。目标域数据完全不参与训练。

| 参数 | 说明 | 默认值 |
|:---|:---|:---:|
| `feat_dim` | 骨干网络特征维度 N | 64 |
| `dropout` | 骨干网络 Dropout 率 | 0.1 |
| `adv_hidden` | 对抗掩码器隐藏层维度 | 64 |
| `kappa` | 优势维度比例 (论文 Fig.15) | 0.8 |
| `guidance_embed_dim` | 引导编码器嵌入维度 | 32 |
| `guidance_epochs` | 引导编码器预训练轮数 | 50 |
| `guidance_lr` | 引导编码器学习率 | 1e-3 |
| `guidance_alpha_adv` | 域对抗损失权重 | 1.0 |
| `guidance_alpha_rnc` | RNC 对比损失权重 | 1.0 |
| `afm_gamma_A` | 振幅混合系数下界 (论文 Fig.14) | 0.5 |
| `afm_gamma_P` | 相位扰动系数下界 (论文 Fig.14) | 0.9 |
| `afm_eta` | 相位邻居相似度阈值 (论文 Fig.15) | 0.8 |
| `cir_tau_fac` | FAC 关联因子化损失权重 (论文 Fig.15) | 2.0 |
| `cir_adv_weight` | 对抗掩码损失权重 | 0.5 |
| `optimizer_name` | 优化器 | `"AdamW"` |
| `lr` | 主模型学习率 | 1e-3 |
| `weight_decay` | 权重衰减 | 1e-2 |

### 验证策略

通过 `val_strategy` 参数切换 K-Fold 和 LOSO：

**K-Fold（手动指定每折验证集）：**
```python
"val_strategy": "kfold",
"test_ids": ["010", "011"],
"folds": {
    1: {"val_ids": ["001", "002"]},
    2: {"val_ids": ["003", "004"]},
    3: {"val_ids": ["005", "006"]},
}
# 训练集 = 全部受试者 - test_ids - 当前fold的val_ids
```

**LOSO（自动留一法）：**
```python
"val_strategy": "loso",
"test_ids": ["010", "011"],
# 不需要写 folds，代码自动扫描数据目录，每个受试者单独做一折验证
# 20 个受试者（去掉 test）→ 自动生成 18 折
```

> 💡 **域适应说明**: 对于所有域适应方法 (MLDA/DAEEGViT/LA-MSDA/DANN/DeepCORAL)，`val_ids` 中的受试者同时作为**目标域**。训练时目标域标签不可见（仅特征参与域适应损失），评估在目标域上进行。如果需要与论文完全对齐（LOSO，每折仅 1 个目标受试者），可使用 `"val_strategy": "loso"`。

---

## 🧠 模型架构

### LSTM

```
Input (B, W, C) → Linear(C→H) → LSTM(H, n_layers) → last_hidden
              → Dropout → Linear(H→H) → ReLU → Dropout → Linear(H→C) → Logits
```

### Transformer

```
Input (B, W, C) → Linear(C→D) → PosEnc → TransformerEncoder(D, nhead, n_layers)
              → GlobalAvgPool → Dropout → Linear(D→D) → ReLU → Dropout → Linear(D→C) → Logits
```

### Mamba

```
Input (B, W, C) → Linear(C→D) → [MambaBlock × n_layer] → LayerNorm
              → GlobalAvgPool → Dropout → Linear(D→D) → ReLU → Dropout → Linear(D→C) → Logits
```

MambaBlock 内部（来自 `mamba-ssm` 包）：
```
x → LayerNorm → [Linear→SiLU→Conv1d→SSM] ⊙ SiLU(gate) → Linear → Dropout + x → y
```

### TimesNet

TimesNet 通过 FFT 自动发现多周期性，将 1D 序列重塑为 2D 张量，使用 2D Inception 卷积同时捕获周期间和周期内变化模式。

```
Input (B, W, C) → DataEmbedding (Conv1d+PosEnc) → (B, W, d_model)
       ↓
   [TimesBlock + LayerNorm] × e_layers
       ↓
   GELU → Dropout → GlobalAvgPool → Linear → Logits
```

**TimesBlock** (核心模块):
```
x (B, T, d_model)
  → FFT → top-k 频率 → 周期长度 p = T / freq
  → 对每个周期:
      1D → 填充至 p 整数倍 → 重塑 2D (B, d_model, T//p, p)
      → Inception Block (d_model→d_ff) → GELU
      → Inception Block (d_ff→d_model)
      → 重塑回 1D (B, T, d_model)
  → Softmax(振幅) 加权聚合 k 个周期结果
  → + x (残差)
```

**Inception Block** (2D 多尺度卷积):
```
x (B, C, H, W) → [Conv2d(k=1,3,5,7,9,11)] → Stack → Mean → (B, C', H, W)
```

**关键设计**:
- FFT 周期发现: rfft → 振幅均值 → 去直流 → TopK → period = T/freq
- 共享 2D 卷积: 同一 Inception 块应用于所有 k 个周期 (参数高效)
- 自适应聚合: FFT 振幅经 Softmax 作为权重, 加权合并多周期结果
- 分类头: GlobalAvgPool + FC (比原论文 Flatten+FC 更轻量)

### ProtoNet

```
Support/Query → MLP Encoder → Embedding
Prototype_c = mean(Embedding[class_c])
Dist(q, c) = ||Embedding(q) - Prototype_c||₂
Logits = -Dist → Softmax
```

### RelationNet

```
Support/Query → MLP Encoder → Embedding
Prototype_c = mean(Embedding[class_c])
Relation(q, c) = MLP([Embedding(q) || Prototype_c])
Logits = Relation → Softmax
```

### MLDA (Multi-Level Domain Adaptation)

MLDA 是域适应方法，训练时需要源域（有标签）和目标域（无标签）双流数据。

```
源域数据 ──┐                              ┌── 分类损失 L_CE (源域标签)
           ├→ Encoder (共享) → features ──┤
目标域数据 ─┘                              ├── 域间损失 L_inter (Wasserstein)
                     ↓                     │   ├─ src_feat → U → wasserstein → scalar
                     Classifier            │   └─ tar_feat → V → wasserstein → scalar
                     ↓                     │
                  predictions              └── 域内损失 L_intra (ICD 对比)
                                               ├─ 源域: 真实标签
                                               └─ 目标域: 伪标签 (argmax)
```

**Encoder** (共享特征提取器):
```
Input (B, W*C) → Linear(→512) → BN → ReLU → Dropout
               → Linear(→128) → BN → ReLU → Dropout
               → Linear(→128) → BN → ReLU → Dropout
               → Linear(→32)  → BN → ReLU → Dropout → features (B, 32)
```

**损失函数**:
```
L = L_CE + 2 × [(1 - λ) × L_inter + λ × L_intra]

- L_CE:    源域交叉熵 (标准监督信号)
- L_inter: Wasserstein 距离 (U(src_feat) vs V(tar_feat), 1D EMD 逐维平均)
- L_intra: ICD 对比损失 (类条件 MMD, 最小化类内域差异 - 最大化类间域差异)
- λ:       sigmoid 调度 1/(1+e^(epoch-center)), 从偏重域间逐渐过渡到偏重域内
```

**训练特点**:
- 三组独立优化器分别更新 Encoder+Classifier / U / V
- 源域+目标域 batch 配对训练 (`zip(src_loader, tar_loader)`)
- 目标域标签在训练时不可见，使用伪标签参与 ICD 损失计算

### DAEEGViT (Domain Adaptive EEG Vision Transformer)

DAEEGViT 是 1D Vision Transformer 结合 MBConv 的域适应方法，在 CLS token 特征上使用 MMD 进行跨域对齐。

```
Input (B, C, W) → Conv1d PatchEmbed → [CLS] + Patches + PosEmbed
                 ↓
            ┌─ Block × depth ─────────────────────────────┐
            │  Attention (全局特征) → residual             │
            │  MBConv  (局部特征, token 维度卷积) → residual │
            │  MLP     (特征变换) → residual               │
            └──────────────────────────────────────────────┘
                 ↓
            CLS token → Classifier → logits
                      → MMD(source_cls, target_cls)
```

**Block** (三子模块，区别于标准 ViT 的两子模块):
```
x → LN → MultiHeadAttention → residual → MBConv → residual → LN → MLP → residual
```

MBConv 在 token 序列维度 (N+1) 上做 1D 卷积，捕获局部模式:
```
expand(1x1) → DW_Conv(3x3) → SqueezeExcite → project(1x1) + residual
```

**损失函数**:
```
L = L_cls + L_mmd

- L_cls: 源域交叉熵 (对 logits)
- L_mmd: CLS token 特征上的 MK-MMD (多核高斯核)
```

**训练特点**:
- 单模型 + 单优化器 (比 MLDA 更简洁)
- 源域+目标域 batch 配对训练 (`zip(src_loader, tar_loader)`)
- MMD 无额外调度参数，直接等权加到分类损失上

### LA-MSDA (Label-based Alignment Multi-Source Domain Adaptation)

LA-MSDA 是多源域适应方法，每个源域拥有独立的域特定网络和分类器，共享底层特征提取器。推理时集成所有源域分支的预测。

```
Input (B, C, W)
       ↓
   SharedNet (1D CNN: Conv1d→BN→ELU→Pool × 4 层 → AdaptiveAvgPool)
       ↓ shared features (B, feature_dim)
       ↓
  ┌────┼─────────────┐
  ↓    ↓             ↓
DSCNN₁ DSCNN₂  ... DSCNN_N   ← 每个源域一个 (Linear+BN+ReLU × 3)
  ↓    ↓             ↓
Cls₁  Cls₂   ...  Cls_N     ← 每个源域一个分类头
```

**损失函数 (每个源域分支独立计算)**:
```
L = L_cls + μ·L_llmmd + γ·L_global

- L_cls:    当前源域分支的交叉熵
- L_llmmd:  标签条件 MMD (共享特征上，源域用真实标签，目标域用 softmax 伪标签加权核矩阵)
- L_global: 所有分支在目标域上预测的 L1 分歧 (排序加权)
- μ, γ:     sigmoid 预热 2/(1+e^(-10·t/T))-1，从 0 渐增到 1
```

**训练特点**:
- 多源域: 每个 epoch 轮流训练各源域分支
- 标签条件对齐: LLMMD 通过类别信息加权核矩阵，避免负迁移
- 分类器共识: 鼓励所有源域分类器在目标域上达成一致
- 集成推理: 测试时所有分支 softmax 预测取平均后 argmax

### DANN (Domain-Adversarial Training of Neural Networks)

DANN 是经典的域对抗训练方法，通过梯度反转层 (GRL) 使特征提取器学习域不变表示，同时域分类器试图区分源域和目标域。

```
源域数据 ──┐                              ┌── Classifier → 分类损失 L_CE (源域标签)
           ├→ Encoder (共享) → features ──┤
目标域数据 ─┘                              └── GRL → DomainClassifier → 域损失 L_d
                                               ├─ 源域: 标签=0
                                               └─ 目标域: 标签=1
```

**GRL (梯度反转层)**:
```
前向传播: 恒等映射 (output = input)
反向传播: grad_output × (-α)
```

**DomainClassifier** (域分类器):
```
features → Linear(→1024) → BN → ReLU → Dropout
         → Linear(→1024) → BN → ReLU → Dropout
         → Linear(→1) → domain_logit
```

**损失函数**:
```
L = L_y + λ × L_d

- L_y: 源域交叉熵 (标准监督信号)
- L_d: 域二分类 BCE (源=0, 目标=1, 通过 GRL 对抗训练)
- λ:   sigmoid 调度 2/(1+exp(-γ·p))-1，p=step/total_steps
       从 ~0 (初始仅分类) 渐增到 ~1 (完整对抗)
```

**训练特点**:
- 梯度反转: 域分类器的梯度经 GRL 反转后回传到编码器，使编码器"对抗"域分类器
- λ 调度: 训练初期 λ≈0 (特征先学好分类)，后期 λ→1 (全面对抗)
- 单优化器统一更新全部参数 (比 MLDA 更简洁)
- 推理时域分类器不参与，仅使用 Encoder + Classifier

### DeepCORAL (Deep CORrelation ALignment)

DeepCORAL 是最简洁的域适应方法之一，通过对齐源域和目标域特征的二阶统计量 (协方差矩阵) 来减少域偏移。无需对抗训练、无需域分类器。

```
源域数据 ──┐                              ┌── Classifier → 分类损失 L_CE
           ├→ Encoder (共享) → features ──┤
目标域数据 ─┘                              └── CORAL(src_feat, tar_feat) → L_CORAL
```

**CORAL 损失**:
```
L_CORAL = (1 / 4d²) × ||C_s - C_t||_F²

- C_s: 源域特征协方差矩阵 (d×d)
- C_t: 目标域特征协方差矩阵 (d×d)
- d:   特征维度
```

**总损失函数**:
```
L = L_cls + λ × L_CORAL

- L_cls:  源域交叉熵
- L_CORAL: 协方差对齐损失 (Frobenius 范数平方)
- λ:      CORAL 权重 (可配置，无调度)
```

**训练特点**:
- 最简洁的域适应方法: 无对抗训练、无伪标签、无调度参数
- 单模型 + 单优化器
- CORAL 损失直接可微，计算开销极低
- 推理时仅使用 Encoder + Classifier

### InterpretableCNN (EEG-Based Cross-Subject Driver Drowsiness Recognition)

InterpretableCNN 是一种轻量级可分离卷积网络，原论文用于 EEG 跨被试疲劳识别。本实现将 2D 卷积分离架构适配为 1D 卷积，处理偏差序列。

**域泛化 (DG) 说明**: 这是纯 ERM (Empirical Risk Minimization) 基线，不含任何域适应/域泛化特有机制。跨被试泛化能力来自 LOSO / K-Fold 评估协议：训练仅在源域被试上进行，测试在未见过的目标域被试上进行，目标域数据完全不参与训练。

```
Input (B, C, W)
       ↓
   Pointwise Conv1d(C → N1, kernel=1)    ← 通道混合 (等效原论文空间滤波)
       ↓
   Depthwise Conv1d(N1 → N1×d, kernel=K, groups=N1)  ← 时间滤波 (分组卷积)
       ↓
   ReLU → BatchNorm1d (track_running_stats=False)
       ↓
   Global Average Pooling (时间维度均值)
       ↓
   FC(N1×d → num_classes) → LogSoftmax
```

**关键设计细节**:
- `track_running_stats=False`: 评估模式下 BatchNorm 仍使用 batch 统计量 (匹配原论文)
- 输出为 log-probabilities，配合 NLLLoss 使用
- 超轻量: 仅 ~2K 参数 (3通道版本)

**损失函数**:
```
L = NLLLoss(log_probs, labels)
```

**训练特点**:
- 纯监督学习，无域适应损失
- 目标域数据完全不参与训练 (与 DA 方法本质区别)
- Adam 优化器 (原论文) / AdamW (本项目默认)

### AFM-CIR (Causality-Preserving Domain Generalization via Adaptive Fourier Mixup)

AFM-CIR 是一种因果域泛化方法，核心思想是在频域中通过语义相似度引导的自适应增强来丰富训练分布，同时通过因果启发损失确保增强不破坏标签语义。原论文面向 RUL 回归任务，本实现将其适配为二分类疲劳检测。

**三阶段训练流程**:
```
Phase 1: 预训练引导编码器 (每个 fold 独立训练)
  Input (B, C, W) → EncBlock×3 → AdaptiveAvgPool → Linear → z (嵌入)
                                         ↓ (冻结)
Phase 2: AFM 频域增强 (每个 batch)
  x → FFT → (A, P) 振幅/相位
         ↓ 引导嵌入相似度 → 自适应混合系数 (λ_A, λ_P)
  A_aug = λ_A·A + (1-λ_A)·A_ref       ← 跨域振幅混合
  P_aug = P - ΔΘ·(1-λ_P)              ← 有界最短角相位扰动
  x_aug = IFFT(A_aug · exp(j·P_aug))   ← 重建增强信号

Phase 3: 因果启发训练
  x, x_aug → Backbone → features, features_aug
                            ↓
              ┌─────────────┼────────────────┐
              ↓             ↓                ↓
         Classifier     FAC Loss       Adversarial Masker
         (NLLLoss)   (关联因子化)     (Gumbel-Softmax)
              ↓             ↓                ↓
          L_sup        L_fac = ½||C-I||²  L_inf (min-max)
```

**Backbone** (特征提取器 g_hat):
```
Input (B, C, W)
       ↓
   Conv1d(C→32, k=7) → BN → ReLU
   Conv1d(32→64, k=5) → BN → ReLU
   Conv1d(64→N, k=3)  → BN → ReLU
   AdaptiveAvgPool1d → Linear(N→N) → features (B, N)
```

**引导编码器** (Phase 1, 预训练后冻结):
```
Encoder:  Conv1d(C→16) → Conv1d(16→32) → Conv1d(32→64) → AdaptiveAvgPool → Linear(64→D)
Decoder:  Linear(D→64·W/8) → ConvTranspose1d×3 → x_recon
Domain Classifier: Linear(D→64) → ReLU → Linear(64→num_domains)
Loss = L_recon + α_adv·L_adv(GRL) + α_rnc·L_rnc
```

**损失函数**:
```
L = L_sup + L_aug + τ·L_FAC + w·L_inf

- L_sup:  原始样本 NLLLoss (分类损失)
- L_aug:  增强样本 NLLLoss (AFM 增强后的分类损失)
- L_FAC:  关联因子化损失 (Common Cause + ICM Principle)
           C_ij = cos(r_i, r^a_j), L = ½||C - I||²_F
           对角线→1 (增强不变性), 非对角线→0 (维度独立性)
- L_inf:  对抗掩码损失 (因果充分性, min-max 博弈)
           masker 选 Top-κ 优势维度 → superior classifier
           剩余劣势维度 → inferior classifier
           adversary: max L_inf (暴露劣势维度)
           backbone:  min L_inf (强制所有维度携带信息)
- τ, w:   损失权重 (τ=2.0, w=0.5)
```

**AFM 增强细节**:
```
振幅混合: λ_A = 1 - (1-σ_A)^γ_A ∈ (γ_A, 1)
  σ_A: 引导嵌入相似度, γ_A: 下界控制 (0.5)
  相似度越高 → λ_A 越小 → 混合越强

相位扰动: λ_P = 1 - (1-σ_P)^γ_P ∈ (γ_P, 1)
  σ_P: 邻居相似度 (需 ≥ η), γ_P: 下界控制 (0.9)
  沿最短角路径微调, |ΔΘ| ≤ π, 保持因果语义

理论保证:
  Theorem 1: ACE ≤ 2C·√(2·I(T;Y))  [互信息上界]
  Theorem 2: ACE ≤ L_f·C_n·||ΔΦ||_∞·E[||A||²]  [Lipschitz-谱范数界]
```

**训练特点**:
- 纯 DG 方法: 目标域数据完全不参与训练
- 三阶段: Phase 1 预训练 (50 epochs) → Phase 2+3 联合训练
- 两个独立优化器: 主模型 + 对抗掩码器 (min-max 博弈)
- 推理时仅使用 Backbone + Classifier (无引导编码器、无掩码器开销)

---

## 📋 模型参数量与计算量

> 使用 `calflops` 库实测，输入 batch_size=1, C=3 (ADF三通道), W=256。
> 运行 `python stats_model_flops.py` 可重新生成 `model_stats.csv`。

| 模型 | 输入格式 | Params | MACs | FLOPs |
|:---|:---:|---:|---:|---:|
| LSTM | (B,W,C) | 71.11 K | 53.38 KMACs | 33.99 MFLOPS |
| Transformer | (B,W,C) | 71.49 K | 16.83 MMACs | 33.99 MFLOPS |
| Mamba | (B,W,C) | ~37K* | — | — |
| TimesNet | (B,W,C) | 2.34 M | 1.82 GMACs | 3.63 GFLOPS |
| ProtoNet | (B×20,768) | 55.71 K | 1.66 MMACs | 3.33 MFLOPS |
| RelationNet | (B×20,768) | 57.1 K | 1.71 MMACs | 3.44 MFLOPS |
| MLDA | (B,768) | 481.7 K | 479.3 KMACs | 960.99 KFLOPS |
| DAEEGViT | (B,C,W) | 211.57 K | 2.01 MMACs | 4.12 MFLOPS |
| LA-MSDA | (B,C,W) | 768.27 K | 375.3 KMACs | 772.86 KFLOPS |
| DANN | (B,768) | 1.57 M | 958.59 KMACs | 1.92 MFLOPS |
| DeepCORAL | (B,768) | 481.7 K | 958.59 KMACs | 1.92 MFLOPS |
| InterpretableCNN | (B,C,W) | 2.27 K | 815.23 KMACs | 1.73 MFLOPS |
| AFM-CIR | (B,C,W) | 36.29 K | 11.89 MMACs | 24.13 MFLOPS |

> *Mamba 需要 CUDA 环境 (`mamba-ssm`, `causal-conv1d`)，标注 * 的数值为估算值。

---

## 🔧 扩展指南

### 添加新模型

1. 在 `models/` 下创建 `your_model.py`
2. 在 `models/get_model.py` 中注册
3. 在 `models/__init__.py` 中添加 import
4. 在 `configs/` 下创建配置字典
5. 在 `main_fatigue.py` 的 `main()` 中合并配置并添加训练路由
6. 在 `tests/` 下编写单元测试
7. 运行: `python main_fatigue.py --exp_name Your_Exp_Name`

### 添加新的训练范式

项目当前支持以下 `training_type` 路由：

| training_type | 训练函数 | 适用方法 |
|:---|:---|:---|
| _(默认)_ | `run_temporal_fold()` | LSTM, Transformer, Mamba |
| _(fewshot)_ | `run_fewshot_fold()` | ProtoNet, RelationNet |
| `domain_adapt` | `run_mlda_fold()` | MLDA |
| `domain_adapt_vit` | `run_daeevit_fold()` | DAEEGViT |
| `multi_source_da` | `run_lamsda_fold()` | LA-MSDA |
| `dann` | `run_dann_fold()` | DANN |
| `deepcoral` | `run_deepcoral_fold()` | DeepCORAL |
| `dg_interpcnn` | `run_interpcnn_fold()` | InterpretableCNN |
| `dg_afmcir` | `run_afmcir_fold()` | AFM-CIR |

如需添加新的域适应方法，参考 `run_mlda_fold()` (复杂域适应) 或 `run_deepcoral_fold()` (简洁域适应) 的双流数据 + 域适应损失模式。如需添加新的域泛化方法，参考 `run_interpcnn_fold()` (纯 ERM 基线) 或 `run_afmcir_fold()` (含数据增强的因果 DG)。

### 自定义数据

如果数据格式不同，修改 `data/fatigue_dataset.py` 中的：
- `_load_data()`: 数据加载逻辑
- `__getitem__()`: 返回格式

---

## 🧪 测试

```bash
# 运行 DANN/DeepCORAL 单元测试 (35 个测试用例)
python -m pytest tests/test_dann_deepcoral.py -v

# 运行 InterpretableCNN 单元测试 (21 个测试用例)
python -m pytest tests/test_interpcnn.py -v

# 运行全部测试
python -m pytest tests/ -v

# 仅运行特定测试类
python -m pytest tests/test_dann_deepcoral.py::TestGRL -v
python -m pytest tests/test_interpcnn.py::TestArchitecture -v
```

测试覆盖: GRL 梯度反转、DANN/DeepCORAL 前向/反向传播、CORAL 损失正确性、InterpretableCNN 可分离卷积架构、BatchNorm 行为、配置加载、端到端训练步骤、模型参数量统计、多轮训练收敛性。

---

## 📚 参考文献

- **LSTM**: Hochreiter & Schmidhuber. *Long Short-term Memory*. Neural Computation, 1997.
- **Transformer**: Vaswani et al. *Attention Is All You Need*. NeurIPS, 2017.
- **Mamba**: Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. 2023.
- **TimesNet**: Wu et al. *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*. ICLR, 2023.
- **ProtoNet**: Snell et al. *Prototypical Networks for Few-shot Learning*. NeurIPS, 2017.
- **RelationNet**: Sung et al. *Learning to Compare: Relation Network for Few-Shot Learning*. CVPR, 2018.
- **MLDA**: Huang et al. *Multi-level domain adaptation for improved generalization in electroencephalogram-based driver fatigue detection*. Engineering Applications of Artificial Intelligence, 2025.
- **DAEEGViT**: *DAEEGViT: A domain adaptive vision transformer framework for EEG cognitive state identification*.
- **LA-MSDA**: *Label-based Alignment Multi-Source Domain Adaptation for Cross-subject EEG Fatigue Mental State Evaluation*.
- **DANN**: Ganin et al. *Domain-Adversarial Training of Neural Networks*. Journal of Machine Learning Research (JMLR), 2016.
- **DeepCORAL**: Sun & Saenko. *Deep CORAL: Correlation Alignment for Deep Domain Adaptation*. ECCV, 2016.
- **InterpretableCNN**: Cui et al. *EEG-Based Cross-Subject Driver Drowsiness Recognition With an Interpretable Convolutional Neural Network*. IEEE TNNLS, 2022.
- **AFM-CIR**: Zhu et al. *Causality-Preserving Domain Generalization via Adaptive Fourier Mixup for RUL Prediction*. IEEE TPAMI, 2026. DOI: 10.1109/TPAMI.2026.3688520.

---

## 📬 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

> **Made with ❤️ for Fatigue Detection Research**
