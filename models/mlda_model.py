"""
MLDA (Multi-Level Domain Adaptation) 模型
基于论文 "Multi-level domain adaptation for improved generalization in
electroencephalogram-based driver fatigue detection" (EAAI 2025)

适配说明:
    原论文使用 EEG 差分熵 (DE) 特征作为输入 (750维)。
    本实现将输入替换为偏差序列 (deviation sequences) 的展平向量
    (window_size * num_channels 维)，用于基于注视偏差的疲劳检测。

网络架构:
    - Encoder: 4层 MLP (input_dim → 512 → 128 → 128 → feat_dim)
    - Classifier: Linear (feat_dim → num_classes)
    - U/V: 域特定投影网络，用于 Wasserstein 距离计算
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """共享特征提取器: 4层全连接网络 + BatchNorm + Dropout"""

    def __init__(self, input_dim: int, feat_dim: int = 32, dropout: float = 0.05):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.features(x)


class Classifier(nn.Module):
    """任务分类器: 单层线性映射"""

    def __init__(self, feat_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


class DomainProjection(nn.Module):
    """域特定投影网络 (U/V)，用于 Wasserstein 距离计算

    原论文使用两个独立的小网络分别投影源域和目标域特征，
    在投影后的空间上计算 1D Wasserstein 距离。
    """

    def __init__(self, feat_dim: int = 32, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MLDAModel(nn.Module):
    """MLDA 多级域适应模型

    forward 行为:
        - tar_data=None (推理模式): 返回 (src_feature, src_logits)
        - tar_data!=None (训练模式): 返回 (src_feature, tar_feature, src_logits, tar_logits)

    输入形状:
        - 推理: (B, input_dim) 展平后的偏差序列向量
        - 训练: src (B, input_dim), tar (B, input_dim)
    """

    def __init__(self, input_dim: int, num_classes: int = 2,
                 feat_dim: int = 32, dropout: float = 0.05):
        super().__init__()
        self.encoder = Encoder(input_dim, feat_dim, dropout)
        self.classifier = Classifier(feat_dim, num_classes)

    def forward(self, src_data, tar_data=None):
        src_feature = self.encoder(src_data)
        src_output_cls = self.classifier(src_feature)

        if tar_data is None:
            return src_feature, src_output_cls
        else:
            tar_feature = self.encoder(tar_data)
            tar_output_cls = self.classifier(tar_feature)
            return src_feature, tar_feature, src_output_cls, tar_output_cls
