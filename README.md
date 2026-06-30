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

---

## 📂 项目结构

```text
Fatigue-Contrast-exp/
├── configs/
│   ├── config.py                     # 原始模板配置
│   ├── experiments_object.py         # 原始模板实验字典
│   ├── fatigue_temporal_baselines.py # 时序基线配置 (LSTM/Transformer/Mamba)
│   └── fatigue_fewshot_baselines.py  # 小样本基线配置 (ProtoNet/RelationNet)
├── data/
│   ├── dataset.py                    # 原始图像数据集
│   └── fatigue_dataset.py            # 疲劳检测数据集 (JSONL 滑动窗口)
├── models/
│   ├── get_model.py                  # 模型工厂（统一入口）
│   ├── lstm.py                       # LSTM 分类器
│   ├── transformer_encoder.py        # Transformer Encoder 分类器
│   ├── mamba_model.py                # Mamba 分类器 (基于 mamba-ssm)
│   ├── protonet.py                   # Prototypical Networks
│   ├── relationnet.py                # Relation Network
│   └── ...                           # 原始模板模型 (ResNet/EfficientNet 等)
├── engine/
│   └── trainer.py                    # 原始模板训练器
├── utils/
│   ├── basic.py                      # 优化器/学习率调度器
│   ├── loss_function.py              # 损失函数
│   ├── observer.py                   # 指标监控/日志/早停
│   ├── reproducibility.py            # 可复现性工具
│   └── swanlab_logger.py             # SwanLab 实验跟踪
├── main.py                           # 原始模板训练入口
├── main_fatigue.py                   # 疲劳检测训练入口 ⭐
├── test.py                           # 模型评估脚本
├── infer.py                          # 推理脚本
├── requirements.txt                  # 依赖清单
└── README.md                         # 本文档
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
pip install torchmetrics numpy scikit-learn Pillow tqdm tensorboard matplotlib seaborn PyYAML
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
# 模型 FLOPs 计算
pip install ptflops

# 实验跟踪
pip install swanlab

# PoolFormer 等模型
pip install timm

# 一键安装全部依赖
pip install -r requirements.txt
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

每行是一帧数据，包含 `deviation_px_after_calibrate` 等特征：

```json
{
  "timestamp": 4.16,
  "frame_idx": 100,
  "deviation_px_after_calibrate": 13.19,
  ...
}
```

数据目录结构：
```
data_dir/
├── 001_easy_alert.jsonl
├── 001_hard_alert.jsonl
├── 001_easy_sleepy.jsonl
├── 002_easy_alert.jsonl
└── ...
```

---

## 🚀 快速开始

### 1️⃣ 修改配置

打开对应的配置文件，修改以下关键参数：

```python
# configs/fatigue_temporal_baselines.py 或 configs/fatigue_fewshot_baselines.py

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
# LSTM 基线
python main_fatigue.py --exp_name Fatigue_LSTM_baseline

# Transformer 基线
python main_fatigue.py --exp_name Fatigue_Transformer_baseline

# Mamba 基线
python main_fatigue.py --exp_name Fatigue_Mamba_baseline

# ProtoNet 基线
python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline

# RelationNet 基线
python main_fatigue.py --exp_name Fatigue_RelationNet_baseline
```

### 3️⃣ 输出

每次训练会在 `output_dir` 下生成带时间戳的目录：

```
result_20260629_143000_Fatigue_LSTM_baseline/
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
| `history_fold{N}.csv` | 每个 fold / LOSO **单独一个文件**，**每训练一轮实时追加** train + val 指标到文件末尾（测试评估完成后再追加 test 行）。长表格式，`split` 列区分 `train` / `val` / `test`，`is_best` 列标记该轮是否为当前 fold 的最佳轮；混淆矩阵按位置展平为 `cm_00 / cm_01 / cm_10 / cm_11` 四个独立列。即使训练中途中断，已完成的轮次指标也不会丢失。 |
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
| `feature_name` | 使用的特征字段 | `"deviation_px_after_calibrate"` |
| `window_size` | 滑动窗口大小（帧数） | 30 |
| `stride` | 滑动窗口步长 | 15 |
| `batch_size` | 批大小 | 32 |
| `epochs` | 最大训练轮数 | 100 |
| `patience` | 早停耐心值 | 20 |
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

### 任务难度过滤

数据文件名格式为 `[id]_[easy|hard]_[alert|sleepy].jsonl`，通过 `difficulty` 参数控制使用哪类数据：

```python
"difficulty": "easy",   # 仅使用 easy 任务数据
"difficulty": "hard",   # 仅使用 hard 任务数据
"difficulty": None,     # 使用全部数据（easy + hard）
```

### 验证策略

通过 `val_strategy` 参数切换 K-Fold 和 LOSO：

**K-Fold（手动指定每折验证集）：**
```python
"val_strategy": "kfold",   # 或不写，默认就是 kfold
"test_ids": ["010", "011"],
"folds": {
    1: {"val_ids": ["001", "002"]},  # Fold 1: 001,002 做验证
    2: {"val_ids": ["003", "004"]},  # Fold 2: 003,004 做验证
    3: {"val_ids": ["005", "006"]},  # Fold 3: 005,006 做验证
}
# 训练集 = 全部受试者 - test_ids - 当前fold的val_ids
```

**LOSO（自动留一法）：**
```python
"val_strategy": "loso",
"test_ids": ["010", "011"],   # 可选，不参与 LOSO 划分
# 不需要写 folds，代码自动扫描数据目录，每个受试者单独做一折验证
# 20 个受试者（去掉 test）→ 自动生成 18 折
```

---

## 🧠 模型架构

### LSTM

```
Input (B, W) → Linear(W→H) → LSTM(H, n_layers) → Dropout → Linear(H→C) → Logits
```

### Transformer

```
Input (B, W) → Linear(W→D) → PosEnc → TransformerEncoder(D, nhead, n_layers)
             → GlobalAvgPool → Dropout → Linear(D→C) → Logits
```

### Mamba

```
Input (B, W) → Linear(W→D) → [MambaBlock × n_layer] → LayerNorm
             → GlobalAvgPool → Dropout → Linear(D→C) → Logits
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

---

## 📋 模型参数量

| 模型 | 参数量 | 说明 |
|:---|:---:|:---|
| LSTM | ~71K | hidden=64, layers=2 |
| Transformer | ~71K | d_model=64, heads=4, layers=2 |
| Mamba | ~37K | d_model=64, layers=2, d_state=16 |
| ProtoNet | ~8.5K | hidden=64, emb=32 |
| RelationNet | ~9.9K | hidden=64, emb=32, relation=16 |

---

## 🔧 扩展指南

### 添加新模型

1. 在 `models/` 下创建 `your_model.py`
2. 在 `models/get_model.py` 中注册
3. 在 `configs/` 下创建配置
4. 运行: `python main_fatigue.py --exp_name Your_Exp_Name`

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

---

## 📬 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

> **Made with ❤️ for Fatigue Detection Research**
