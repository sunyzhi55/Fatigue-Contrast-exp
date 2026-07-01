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
| **小样本学习** | ProtoNet | 原型网络，基于欧氏距离的度量学习 |
| | RelationNet | 关系网络，基于可学习关系模块的度量学习 |
| **域适应** | MLDA | 多级域适应，Wasserstein 域间对齐 + ICD 类条件对比 |
| | DAEEGViT | 域适应 ViT + MBConv，CLS token MMD 对齐 |

---

## 📂 项目结构

```text
Fatigue-Contrast-exp/
├── configs/
│   ├── config.py                         # 原始模板配置
│   ├── experiments_object.py             # 原始模板实验字典
│   ├── fatigue_temporal_baselines.py     # 时序基线配置 (LSTM/Transformer/Mamba)
│   ├── fatigue_fewshot_baselines.py      # 小样本基线配置 (ProtoNet/RelationNet)
│   └── fatigue_domain_adapt_baselines.py # 域适应基线配置 (MLDA)
├── data/
│   ├── dataset.py                        # 原始图像数据集
│   └── fatigue_dataset.py                # 疲劳检测数据集 (JSONL 滑动窗口)
├── models/
│   ├── get_model.py                      # 模型工厂（统一入口）
│   ├── lstm.py                           # LSTM 分类器
│   ├── transformer_encoder.py            # Transformer Encoder 分类器
│   ├── mamba_model.py                    # Mamba 分类器 (基于 mamba-ssm)
│   ├── protonet.py                       # Prototypical Networks
│   ├── relationnet.py                    # Relation Network
│   ├── mlda_model.py                     # MLDA 域适应模型 (Encoder+Classifier+U/V)
│   ├── daeevit_model.py                  # DAEEGViT 域适应 ViT+MBConv 模型
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

# ---- 小样本基线 ----
python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline
python main_fatigue.py --exp_name Fatigue_RelationNet_baseline

# ---- 域适应基线 ----
python main_fatigue.py --exp_name Fatigue_MLDA_baseline
python main_fatigue.py --exp_name Fatigue_DAEEGViT_baseline
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

> 💡 **MLDA 域适应说明**: 对于 MLDA，`val_ids` 中的受试者同时作为**目标域**。训练时目标域标签不可见（仅特征参与域适应损失），评估在目标域上进行。如果需要与论文完全对齐（LOSO，每折仅 1 个目标受试者），可使用 `"val_strategy": "loso"`。

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

---

## 📋 模型参数量

| 模型 | 参数量 | 说明 |
|:---|:---:|:---|
| LSTM | ~71K | hidden=64, layers=2 |
| Transformer | ~71K | d_model=64, heads=4, layers=2 |
| Mamba | ~37K | d_model=64, layers=2, d_state=16 |
| ProtoNet | ~8.5K | hidden=64, emb=32 |
| RelationNet | ~9.9K | hidden=64, emb=32, relation=16 |
| MLDA | ~482K + 2.2K (U/V) | Encoder 481K, U 1.1K, V 1.1K |
| DAEEGViT | ~212K | embed=64, depth=4, heads=4, patch=32 |

---

## 🔧 扩展指南

### 添加新模型

1. 在 `models/` 下创建 `your_model.py`
2. 在 `models/get_model.py` 中注册
3. 在 `configs/` 下创建配置字典
4. 在 `main_fatigue.py` 的 `main()` 中合并配置并添加训练路由
5. 运行: `python main_fatigue.py --exp_name Your_Exp_Name`

### 添加新的训练范式

项目当前支持三种 `training_type` 路由：

| training_type | 训练函数 | 适用方法 |
|:---|:---|:---|
| _(默认)_ | `run_temporal_fold()` | LSTM, Transformer, Mamba |
| _(fewshot)_ | `run_fewshot_fold()` | ProtoNet, RelationNet |
| `domain_adapt` | `run_mlda_fold()` | MLDA |
| `domain_adapt_vit` | `run_daeevit_fold()` | DAEEGViT |

如需添加新的域适应方法，参考 `run_mlda_fold()` 的双流数据 + 域适应损失模式。

### 自定义数据

如果数据格式不同，修改 `data/fatigue_dataset.py` 中的：
- `_load_data()`: 数据加载逻辑
- `__getitem__()`: 返回格式

---

## 📚 参考文献

- **LSTM**: Hochreiter & Schmidhuber. *Long Short-term Memory*. Neural Computation, 1997.
- **Transformer**: Vaswani et al. *Attention Is All You Need*. NeurIPS, 2017.
- **Mamba**: Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. 2023.
- **ProtoNet**: Snell et al. *Prototypical Networks for Few-shot Learning*. NeurIPS, 2017.
- **RelationNet**: Sung et al. *Learning to Compare: Relation Network for Few-Shot Learning*. CVPR, 2018.
- **MLDA**: Huang et al. *Multi-level domain adaptation for improved generalization in electroencephalogram-based driver fatigue detection*. Engineering Applications of Artificial Intelligence, 2025.
- **DAEEGViT**: *DAEEGViT: A domain adaptive vision transformer framework for EEG cognitive state identification*.

---

## 📬 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

> **Made with ❤️ for Fatigue Detection Research**
