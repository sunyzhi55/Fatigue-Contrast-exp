"""
DeepCORAL (Deep CORrelation ALignment) 损失函数及模型
基于论文 "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" (ECCV 2016)
Sun & Saenko

适配说明:
    原论文使用 VGG/AlexNet 作为特征提取器处理图像。
    本实现将特征提取器替换为 MLP 编码器，输入为偏差序列的展平向量
    (window_size * num_channels 维)，用于基于注视偏差的疲劳检测域适应。

核心思想:
    通过对齐源域和目标域特征的二阶统计量 (协方差矩阵) 来减少域偏移。
    CORAL 损失直接可微，无需对抗训练，实现简单高效。

CORAL 损失:
    L_CORAL = (1 / 4*d²) * ||C_s - C_t||_F²
    - C_s: 源域特征协方差矩阵 (d×d)
    - C_t: 目标域特征协方差矩阵 (d×d)
    - d: 特征维度

损失函数:
    L = L_cls + λ * L_CORAL
    - L_cls: 源域交叉熵
    - L_CORAL: 协方差对齐损失
    - λ: CORAL 损失权重

参考文献:
    Sun, B. and Saenko, K., 2016.
    Deep CORAL: Correlation alignment for deep domain adaptation.
    In European Conference on Computer Vision (pp. 443-450). Springer.
"""
import torch
import torch.nn as nn


def coral_loss(source, target):
    """CORAL 损失: 对齐源域和目标域特征的二阶统计量

    计算两个域特征向量的协方差矩阵差异的 Frobenius 范数平方。

    公式 (论文 Eq.1):
        C_s = (1/(n-1)) * (X_s^T X_s - 1^T X_s 的外积归一化)
        L_CORAL = (1 / 4*d²) * ||C_s - C_t||_F²

    Args:
        source: (n, d) 源域特征
        target: (m, d) 目标域特征

    Returns:
        标量损失 (可微)
    """
    n = source.size(0)
    m = target.size(0)
    d = source.size(1)

    # 去均值后的协方差: C = X_c^T X_c / (n-1)
    # 等价于论文 Eq.1 的展开形式
    src_mean = source.mean(dim=0, keepdim=True)
    tar_mean = target.mean(dim=0, keepdim=True)

    src_centered = source - src_mean
    tar_centered = target - tar_mean

    # 协方差矩阵
    cov_src = (src_centered.t() @ src_centered) / max(n - 1, 1)
    cov_tar = (tar_centered.t() @ tar_centered) / max(m - 1, 1)

    # Frobenius 范数平方
    diff = cov_src - cov_tar
    loss = (diff * diff).sum() / (4.0 * d * d)

    return loss


class DeepCORALModel(nn.Module):
    """DeepCORAL 域适应模型

    forward 行为:
        - tar_data=None (推理模式):
            返回 (src_feature, src_logits)
        - tar_data!=None (训练模式):
            返回 (src_feature, tar_feature, src_logits)
            tar_feature 用于外部计算 CORAL 损失

    输入形状:
        - 推理: src_data (B, input_dim) 展平后的偏差序列向量
        - 训练: src_data (B, input_dim), tar_data (B, input_dim)

    说明:
        CORAL 损失通过本文件中的 coral_loss() 函数计算，
        训练循环中通过 `from models.deepcoral_model import coral_loss` 引入。
        这与其他域适应方法 (如 DAEEGViT 的 MMD)
        保持一致的损失计算模式。
    """

    def __init__(self, input_dim: int, num_classes: int = 2,
                 feat_dim: int = 32, dropout: float = 0.05):
        super().__init__()

        from models.mlda_model import Encoder, Classifier

        self.encoder = Encoder(input_dim, feat_dim, dropout)
        self.classifier = Classifier(feat_dim, num_classes)

    def forward(self, src_data, tar_data=None):
        src_feature = self.encoder(src_data)
        src_logits = self.classifier(src_feature)

        if tar_data is None:
            return src_feature, src_logits
        else:
            tar_feature = self.encoder(tar_data)
            return src_feature, tar_feature, src_logits
