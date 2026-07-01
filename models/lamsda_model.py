"""
LA-MSDA (Label-based Alignment Multi-Source Domain Adaptation) 模型
基于论文 "Label-based Alignment Multi-Source Domain Adaptation for
Cross-subject EEG Fatigue Mental State Evaluation"

适配说明:
    原论文使用 EEG 频谱特征 (B, 1, 27, 61) 输入 2D CNN。
    本实现将输入替换为偏差序列 (B, C, W)，使用 1D CNN 替代 2D CNN。

架构:
    - SharedNet: 1D CNN 共享特征提取器 (所有域共用)
    - DSCNN × num_sources: 域特定 1D CNN (每个源域独立)
    - Classifier × num_sources: 域特定分类头 (每个源域独立)

训练损失:
    L = L_cls + μ·L_llmmd + γ·L_global
    - L_cls: 源域交叉熵
    - L_llmmd: 标签条件 MMD (源=one-hot, 目标=softmax 伪标签)
    - L_global: 分类器共识损失 (目标域上预测一致性)
    - μ, γ: sigmoid 预热调度 (从 0 渐增到 1)

推理:
    集成所有源域分类器的 softmax 预测，取平均后 argmax。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================================== #
#  共享特征提取器 (1D CNN)                                                     #
# ========================================================================== #

class SharedNet(nn.Module):
    """共享 1D CNN 特征提取器

    输入: (B, in_channels, seq_len) → 输出: (B, 64)
    结构: 4 层 Conv1d + BN + ELU + MaxPool，逐步下采样后全局平均池化
    """

    def __init__(self, in_channels=3, seq_len=256):
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1: (B, C, 256) → (B, 16, 128)
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(2),
            # Layer 2: (B, 16, 64) → (B, 32, 32)
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(2),
            # Layer 3: (B, 32, 16) → (B, 64, 8)
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.25),
            # Layer 4: (B, 64, 4) → (B, 64, 4)
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.features(x)        # (B, 64, T')
        x = self.pool(x).squeeze(-1)  # (B, 64)
        return x


# ========================================================================== #
#  域特定网络 (Domain-Specific CNN)                                            #
# ========================================================================== #

class DSCNN(nn.Module):
    """域特定 1D CNN: 对共享特征做进一步域适配变换

    输入: (B, 64) → 输出: (B, 256)
    """

    def __init__(self, in_features=64, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ========================================================================== #
#  LA-MSDA 主模型                                                             #
# ========================================================================== #

class LAMSDAModel(nn.Module):
    """LA-MSDA 多源域适应模型

    Args:
        in_channels: 输入通道数 (ADF=3)
        seq_len: 序列长度 (window_size=256)
        num_classes: 分类类别数 (2: alert/sleepy)
        num_sources: 源域数量 (= 训练受试者数，上限由配置控制)
        feature_dim: 共享特征维度 (64)
        ds_hidden_dim: 域特定网络隐藏维度 (256)

    forward 行为:
        - domain_idx 指定使用哪个源域分支
        - 返回 (logits, features): 该分支的分类预测和特征
    """

    def __init__(self, in_channels=3, seq_len=256, num_classes=2,
                 num_sources=5, feature_dim=64, ds_hidden_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.num_sources = num_sources
        self.feature_dim = feature_dim

        # 共享特征提取器
        self.shared_net = SharedNet(in_channels, seq_len)

        # 域特定网络 + 分类器 (每个源域一组)
        self.specific_nets = nn.ModuleList([
            DSCNN(feature_dim, ds_hidden_dim) for _ in range(num_sources)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(ds_hidden_dim, num_classes) for _ in range(num_sources)
        ])

    def forward(self, x, domain_idx):
        """单分支前向传播

        Args:
            x: (B, C, W) 输入数据
            domain_idx: 使用哪个源域分支

        Returns:
            logits: (B, num_classes)
            features: (B, ds_hidden_dim) DSCNN 输出特征 (用于 MMD)
        """
        shared_feat = self.shared_net(x)                  # (B, feature_dim)
        ds_feat = self.specific_nets[domain_idx](shared_feat)  # (B, ds_hidden_dim)
        logits = self.classifiers[domain_idx](ds_feat)     # (B, num_classes)
        return logits, ds_feat

    def shared_features(self, x):
        """仅提取共享特征 (用于 LLMMD 计算)"""
        return self.shared_net(x)

    def ensemble_predict(self, x):
        """集成所有源域分支的 softmax 预测 (推理时使用)

        Args:
            x: (B, C, W) 输入数据

        Returns:
            avg_probs: (B, num_classes) 所有分支 softmax 平均
        """
        shared_feat = self.shared_net(x)
        probs_list = []
        for i in range(self.num_sources):
            ds_feat = self.specific_nets[i](shared_feat)
            logits = self.classifiers[i](ds_feat)
            probs_list.append(F.softmax(logits, dim=1))
        return torch.stack(probs_list, dim=0).mean(dim=0)  # (B, num_classes)
