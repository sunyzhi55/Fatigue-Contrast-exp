"""
DANN (Domain-Adversarial Training of Neural Networks) 模型
基于论文 "Domain-Adversarial Training of Neural Networks" (JMLR 2016)
Ganin et al.

适配说明:
    原论文使用 AlexNet/ResNet 作为特征提取器处理图像。
    本实现将特征提取器替换为 MLP 编码器，输入为偏差序列的展平向量
    (window_size * num_channels 维)，用于基于注视偏差的疲劳检测域适应。

核心思想:
    通过梯度反转层 (Gradient Reversal Layer, GRL) 对抗训练域分类器，
    使特征提取器学习域不变特征。源域带标签用于分类损失，目标域无标签
    仅参与域对抗损失。

网络架构:
    - Encoder: 4层 MLP (input_dim → 512 → 128 → 128 → feat_dim)
    - Classifier: Linear (feat_dim → num_classes) — 任务分类器
    - DomainClassifier: 3层 MLP (feat_dim → 1024 → 1024 → 1) — 域分类器
    - GRL: 梯度反转层，前向传播为恒等映射，反向传播乘以 -λ

损失函数:
    L = L_y(classification) + λ * L_d(domain classification)
    - L_y: 源域交叉熵 (仅有源域有标签)
    - L_d: 域二分类 BCE (源=0, 目标=1)
    - λ: 对抗权重，通过 sigmoid 调度从 0 渐增到 1

参考文献:
    Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H.,
    Laviolette, F., Marchand, M. and Lempitsky, V., 2016.
    Domain-adversarial training of neural networks.
    Journal of Machine Learning Research, 17(1), pp.2096-2030.
"""
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """梯度反转层 (Gradient Reversal Layer, GRL)

    前向传播: 恒等映射 (输出 = 输入)
    反向传播: 将梯度乘以 -alpha (梯度反转)

    这使得域分类器的梯度会反向传播到特征提取器时变为"对抗"方向:
    特征提取器试图使域分类器犯错，从而学习域不变特征。

    alpha 通过 set_alpha() 在外部控制调度 (从 0 渐增到 1)。
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GRL(nn.Module):
    """梯度反转层模块

    Args:
        alpha: 梯度反转系数，默认为 1.0。
               训练过程中通过 set_alpha() 动态调整。
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def set_alpha(self, alpha: float):
        """设置梯度反转系数 (由外部训练循环调度)"""
        self.alpha = alpha

    def forward(self, x):
        if self.training:
            return GradientReversalFunction.apply(
                x, torch.tensor(self.alpha, device=x.device, dtype=x.dtype)
            )
        return x


class DANNClassifier(nn.Module):
    """DANN 任务分类器: Linear(feat_dim → num_classes)"""

    def __init__(self, feat_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


class DANNDomainClassifier(nn.Module):
    """DANN 域分类器: 3层 MLP (feat_dim → 1024 → 1024 → 1)

    输入特征经过 GRL 后送入域分类器。
    源域标签=0，目标域标签=1，使用 BCEWithLogitsLoss。

    结构与原论文 DANN 的域分类器一致 (两个 1024 隐藏层)。
    """

    def __init__(self, feat_dim: int = 32, hidden_dim: int = 1024):
        super().__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.domain_classifier(x)


class DANNModel(nn.Module):
    """DANN 域对抗训练网络

    forward 行为:
        - tar_data=None (推理模式):
            返回 (src_feature, src_logits)
        - tar_data!=None (训练模式):
            返回 (src_feature, src_logits, domain_logits)
            domain_logits: (src_batch + tar_batch, 1) 的域预测 logit
            前 src_batch_size 个为源域预测，后 tar_batch_size 个为目标域预测

    输入形状:
        - 推理: src_data (B, input_dim) 展平后的偏差序列向量
        - 训练: src_data (B, input_dim), tar_data (B, input_dim)
    """

    def __init__(self, input_dim: int, num_classes: int = 2,
                 feat_dim: int = 32, dropout: float = 0.05,
                 domain_hidden: int = 1024):
        super().__init__()

        from models.mlda_model import Encoder

        self.encoder = Encoder(input_dim, feat_dim, dropout)
        self.classifier = DANNClassifier(feat_dim, num_classes)
        self.grl = GRL(alpha=1.0)
        self.domain_classifier = DANNDomainClassifier(feat_dim, domain_hidden)

    def forward(self, src_data, tar_data=None):
        src_feature = self.encoder(src_data)
        src_logits = self.classifier(src_feature)

        if tar_data is None:
            # 推理模式: 不需要域分类
            return src_feature, src_logits
        else:
            # 训练模式: GRL + 域分类
            tar_feature = self.encoder(tar_data)
            combined_features = torch.cat([src_feature, tar_feature], dim=0)
            domain_logits = self.domain_classifier(self.grl(combined_features))
            return src_feature, src_logits, domain_logits
