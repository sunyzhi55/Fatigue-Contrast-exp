"""
InterpretableCNN — 可解释卷积网络
基于论文 "EEG-Based Cross-Subject Driver Drowsiness Recognition With an
Interpretable Convolutional Neural Network" (IEEE TNNLS, 2022)
Cui et al. doi: 10.1109/TNNLS.2022.3147208

适配说明:
    原论文使用 EEG 信号 (30通道 × 384时间点) 配合 2D 卷积。
    本实现将 2D 卷积分离架构适配为 1D 卷积，输入为偏差序列的 ADF
    三通道特征 (B, C, W)，其中 C=3 (空间漂移 + 一阶差分 + 滑动均值)，
    W=window_size=256。

网络架构 (空间-时间可分离卷积):
    Pointwise Conv1d: 通道混合 (C → N1)，类似原论文的 "空间滤波器"
    Depthwise Conv1d: 时间滤波 (N1 → N1×d, groups=N1)，带分组卷积
    ReLU + BatchNorm1d (track_running_stats=False)
    Global Average Pooling (全局时间均值池化)
    FC + LogSoftmax 分类头

域泛化 (Domain Generalization) 说明:
    本方法是纯 ERM (Empirical Risk Minimization) 基线，不含任何
    域适应/域泛化特有机制。"跨被试" 泛化能力来自 LOSO / K-Fold
    评估协议: 训练仅在源域被试上进行，测试在未见过的目标域被试上
    进行。训练过程中目标域数据完全不参与 (连无标签特征也不使用)。
    这与 DANN/DeepCORAL/MLDA 等 DA 方法有本质区别。

参考文献:
    Cui, H., Lan, Z., Liu, Y., et al. "EEG-Based Cross-Subject Driver
    Drowsiness Recognition With an Interpretable Convolutional Neural
    Network." IEEE Transactions on Neural Networks and Learning Systems,
    2022.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpretableCNN(nn.Module):
    """可解释卷积网络 (1D 适配版)

    原论文使用 2D 可分离卷积处理 EEG (30通道 × 384时间点)。
    本实现将 2D 架构适配为 1D 卷积，处理偏差序列 (C通道 × W时间点)。

    架构:
        Pointwise Conv1d(C→N1, kernel=1): 通道混合 (等效于原论文空间滤波)
        Depthwise Conv1d(N1→N1*d, kernel=K, groups=N1): 时间滤波
        ReLU → BatchNorm1d → GlobalAvgPool → FC → LogSoftmax

    Args:
        in_channels: 输入通道数 (ADF: 3, 单通道: 1)
        seq_len: 序列长度 (默认 256)
        num_classes: 分类类别数 (默认 2)
        n_filters: 通道混合后的滤波器数 N1 (原论文: 16)
        depth_multiplier: 深度可分离乘数 d (原论文: 2)
        kernel_size: 时间卷积核大小 (原论文: 64)
        dropout: 分类器前 Dropout 率 (原论文: 0)
    """

    def __init__(self, in_channels: int = 3, seq_len: int = 256,
                 num_classes: int = 2, n_filters: int = 16,
                 depth_multiplier: int = 2, kernel_size: int = 64,
                 dropout: float = 0.0):
        super().__init__()

        self.n_filters = n_filters
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.num_features = n_filters * depth_multiplier

        # ---- 通道混合 (Pointwise Conv1d, kernel=1) ----
        # 等效于原论文的 "空间滤波器": 将 C 个输入通道线性组合为 N1 个虚拟通道
        # 原论文: Conv2d(1, N1, (C_eeg, 1))，这里简化为 Conv1d(C, N1, 1)
        self.pointwise = nn.Conv1d(
            in_channels, n_filters, kernel_size=1
        )

        # ---- 时间滤波 (Depthwise Conv1d, groups=N1) ----
        # 每个虚拟通道独立应用 d 个时间滤波器
        # 原论文: Conv2d(N1, N1*d, (1, K), groups=N1)
        self.depthwise = nn.Conv1d(
            n_filters, n_filters * depth_multiplier,
            kernel_size=kernel_size, groups=n_filters
        )

        # ---- 激活 + 归一化 ----
        self.activation = nn.ReLU()
        # 原论文关键细节: track_running_stats=False
        # 评估模式下仍使用 batch 统计量，而非 running 统计量
        self.batchnorm = nn.BatchNorm1d(
            n_filters * depth_multiplier, track_running_stats=False
        )

        # ---- Dropout (原论文未使用，留作可调) ----
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ---- 分类器 ----
        self.fc = nn.Linear(n_filters * depth_multiplier, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """前向传播

        Args:
            x: (B, C, W) 输入张量，C=通道数，W=序列长度

        Returns:
            (B, num_classes) log-probabilities
        """
        # 通道混合: (B, C, W) → (B, N1, W)
        x = self.pointwise(x)

        # 时间滤波: (B, N1, W) → (B, N1*d, W-K+1)
        x = self.depthwise(x)

        # 激活 + 归一化
        x = self.activation(x)
        x = self.batchnorm(x)

        # 全局平均池化: (B, N1*d, T) → (B, N1*d)
        x = x.mean(dim=-1)

        # Dropout + 分类
        x = self.dropout(x)
        x = self.fc(x)
        x = self.log_softmax(x)

        return x

    def predict_proba(self, x):
        """输出概率 (exp of log-probabilities)

        Args:
            x: (B, C, W) 输入张量

        Returns:
            (B, num_classes) probabilities
        """
        log_probs = self.forward(x)
        return torch.exp(log_probs)
