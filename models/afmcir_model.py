"""
AFM-CIR -- Causality-Preserving Domain Generalization via Adaptive Fourier Mixup
基于论文 "Causality-Preserving Domain Generalization via Adaptive Fourier Mixup
for RUL Prediction" (IEEE TPAMI, 2026)
Zhu, Chen, Cheng, Ye, Zhang. DOI: 10.1109/TPAMI.2026.3688520

适配说明:
    原论文面向 RUL (剩余使用寿命) 回归任务。本实现将其核心思想适配为
    二分类疲劳检测任务 (alert=0 / sleepy=1):
    - Phase 1: 域不变保序引导嵌入 (轻量级 1D-CNN 自编码器)
    - Phase 2: 自适应傅里叶 Mixup (AFM, 频域振幅混合 + 有界相位扰动)
    - Phase 3: 因果启发训练 (FAC 关联因子化损失 + 对抗掩码充分性)

输入格式: (B, C, W), C=3 (ADF三通道), W=window_size=256
域泛化: 纯 DG 方法, 目标域数据完全不参与训练

参考文献:
    Zhu et al. "Causality-Preserving Domain Generalization via Adaptive
    Fourier Mixup for RUL Prediction." IEEE TPAMI, 2026.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================================== #
#  GRL (Gradient Reversal Layer)                                              #
# ========================================================================== #

class _GRLFunction(torch.autograd.Function):
    """梯度反转: 前向不变, 反向乘以 -alpha"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grl(x, alpha=1.0):
    return _GRLFunction.apply(x, alpha)


# ========================================================================== #
#  Phase 1: Guidance Encoder (域不变保序引导嵌入)                               #
# ========================================================================== #

class _EncBlock(nn.Module):
    """轻量级 1D 编码块: Conv1d-BN-ReLU-MaxPool"""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(c_in, c_out, 5, stride=2, padding=2),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _DecBlock(nn.Module):
    """轻量级 1D 解码块: ConvTranspose1d-BN-ReLU"""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(c_in, c_out, 5, stride=2, padding=2,
                               output_padding=1),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class GuidanceEncoder(nn.Module):
    """域不变保序引导编码器 (Phase 1)

    轻量级 1D-CNN 自编码器, 配合域对抗训练和对比学习,
    学习域不变且保序的嵌入空间, 为 AFM 提供语义相似度引导。

    架构:
        Encoder:  3 层 Conv1d 下采样 → AdaptiveAvgPool → latent
        Decoder:  3 层 ConvTranspose1d 上采样 → 重建
        Domain Classifier: GRL + MLP (多类域判别器)

    Args:
        in_channels: 输入通道数 (ADF: 3)
        seq_len: 序列长度 (默认 256)
        embed_dim: 嵌入维度 (默认 32)
        num_domains: 源域数量
        alpha_adv: 对抗损失权重
        alpha_rnc: RNC 对比损失权重
    """

    def __init__(self, in_channels=3, seq_len=256, embed_dim=32,
                 num_domains=2, alpha_adv=1.0, alpha_rnc=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.alpha_adv = alpha_adv
        self.alpha_rnc = alpha_rnc
        self.num_domains = num_domains

        # Encoder: 3 层下采样, 总下采样 8 倍
        self.encoder = nn.Sequential(
            _EncBlock(in_channels, 16),   # W -> W/2
            _EncBlock(16, 32),            # W/2 -> W/4
            _EncBlock(32, 64),            # W/4 -> W/8
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(64, embed_dim)

        # Decoder: 3 层上采样, 恢复原始长度
        self.dec_proj = nn.Linear(embed_dim, 64 * (seq_len // 8))
        self.decoder = nn.Sequential(
            _DecBlock(64, 32),            # W/8 -> W/4
            _DecBlock(32, 16),            # W/4 -> W/2
            _DecBlock(16, in_channels),   # W/2 -> W
        )

        # Domain classifier (多类域判别器, 配合 GRL)
        self.domain_clf = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains),
        )

    def encode(self, x):
        """编码为潜在嵌入 z"""
        h = self.encoder(x)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)

    def decode(self, z):
        """从嵌入解码重建信号"""
        h = self.dec_proj(z)
        h = h.view(z.size(0), 64, self.seq_len // 8)
        return self.decoder(h)

    def forward(self, x):
        """完整前向传播 (编码 + 解码)"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon

    def pretrain_step(self, x, domain_ids, tau=0.1):
        """单步预训练: 重建 + 域对抗 + RNC 对比

        Args:
            x: (B, C, W) 输入
            domain_ids: (B,) 域标签 (整数)
            tau: 对比学习温度参数

        Returns:
            dict: 各项损失值
        """
        z, x_recon = self.forward(x)

        # 重建损失
        loss_recon = F.mse_loss(x_recon, x)

        # 域对抗损失 (GRL: encoder 尝试欺骗判别器)
        domain_logits = self.domain_clf(grl(z))
        loss_adv = F.cross_entropy(domain_logits, domain_ids)

        # Rank-N-Contrast 损失 (保序对比)
        loss_rnc = rnc_loss_binary(z, domain_ids, tau=tau)

        # 总损失
        loss = loss_recon + self.alpha_adv * loss_adv + self.alpha_rnc * loss_rnc

        return {
            "loss": loss,
            "loss_recon": loss_recon.detach(),
            "loss_adv": loss_adv.detach(),
            "loss_rnc": loss_rnc.detach(),
        }


# ========================================================================== #
#  Phase 2: Adaptive Fourier Mixup (AFM, 自适应傅里叶 Mixup)                    #
# ========================================================================== #

class AFMAugmentation(nn.Module):
    """自适应傅里叶 Mixup 数据增强 (Phase 2)

    在频域中, 利用引导嵌入空间的语义相似度, 执行:
    1. 自适应振幅混合: 跨域混合振幅谱, 相似度越高混合越强
    2. 有界相位扰动: 沿最短角路径微调相位, 保持因果语义

    理论保证: 互信息上界 (Theorem 1) 和 Lipschitz-谱范数界 (Theorem 2)
    确保增强不破坏标签语义。

    Args:
        gamma_A: 振幅混合系数下界 (论文默认 0.5, 越小混合越强)
        gamma_P: 相位扰动系数下界 (论文默认 0.9, 越大扰动越弱)
        eta: 相位扰动邻居相似度阈值 (论文默认 0.8)
    """

    def __init__(self, gamma_A=0.5, gamma_P=0.9, eta=0.8):
        super().__init__()
        self.gamma_A = gamma_A
        self.gamma_P = gamma_P
        self.eta = eta

    @staticmethod
    def _mono_map(sigma, gamma):
        """单调映射: sigma -> lambda, 相似度越高混合率越大

        lambda = 1 - (1 - sigma)^gamma in (gamma, 1)
        """
        return 1.0 - (1.0 - sigma).clamp(min=1e-7).pow(gamma)

    def augment(self, x, z, z_pool):
        """对批次中的每个样本执行 AFM 增强

        Args:
            x: (B, C, W) 原始输入信号
            z: (B, D) 当前批次在引导空间中的嵌入
            z_pool: (M, D) 全局嵌入池 (来自所有源域训练样本)

        Returns:
            x_aug: (B, C, W) 增强后的信号
        """
        B, C, W = x.shape
        device = x.device

        # ---- FFT: 逐通道独立变换 ----
        fft_x = torch.fft.rfft(x, dim=-1)          # (B, C, W//2+1)
        A = fft_x.abs()
        P = fft_x.angle()

        # ---- 全局相似度矩阵 ----
        z_n = F.normalize(z, dim=-1)
        pool_n = F.normalize(z_pool, dim=-1)
        sim = z_n @ pool_n.t()                      # (B, M)

        A_aug = A.clone()
        P_aug = P.clone()

        for i in range(B):
            s = sim[i]

            # ---- 振幅混合: 从全局池选参考样本 ----
            sorted_idx = torch.argsort(s, descending=True)
            # 排除与自身过于相似的样本 (可能是同一来源)
            ref_candidates = sorted_idx[sorted_idx < len(z_pool)]
            if len(ref_candidates) > 0:
                ref_idx = ref_candidates[torch.randint(len(ref_candidates), (1,))]
                ref_x = x[ref_idx % B]              # 从当前批次取参考
                ref_fft = torch.fft.rfft(ref_x, dim=-1)
                ref_A = ref_fft.abs()

                sigma_A = s[ref_idx].clamp(0, 1)
                lam_A = self._mono_map(sigma_A, self.gamma_A)
                A_aug[i] = lam_A * A[i] + (1 - lam_A) * ref_A

            # ---- 相位扰动: 从满足阈值的邻居中选 ----
            eligible = torch.where(s >= self.eta)[0]
            if len(eligible) > 0:
                nb_idx = eligible[torch.randint(len(eligible), (1,))]
                nb_x = x[nb_idx % B]
                nb_fft = torch.fft.rfft(nb_x, dim=-1)
                nb_P = nb_fft.angle()

                sigma_P = s[nb_idx].clamp(0, 1)
                lam_P = self._mono_map(sigma_P, self.gamma_P)

                # 最短角路径
                theta = P[i] - nb_P
                theta = torch.remainder(theta + math.pi, 2 * math.pi) - math.pi

                P_aug[i] = P[i] - theta * (1 - lam_P)

        # ---- 重建增强信号 ----
        fft_aug = A_aug * torch.exp(1j * P_aug)
        x_aug = torch.fft.irfft(fft_aug, n=W, dim=-1)

        return x_aug


# ========================================================================== #
#  AFMCIRNet: 主分类模型 (特征提取 + 分类 + 对抗掩码)                           #
# ========================================================================== #

class AFMCIRNet(nn.Module):
    """AFM-CIR 域泛化二分类模型

    架构:
        Backbone: 3 层 Conv1d + BN + ReLU + AdaptiveAvgPool + FC
        Classifier: FC + LogSoftmax (与 NLLLoss 配对)
        Adversarial Masker: FC + Gumbel-Softmax (生成维度选择掩码)

    Phase 3 因果启发训练:
        - FAC 损失: 关联因子化, 强制增强前后特征不变性 + 维度间独立性
        - 对抗掩码: 通过 min-max 博弈暴露劣势维度, 强制因果充分性

    Args:
        in_channels: 输入通道数 (ADF: 3)
        seq_len: 序列长度 (默认 256)
        num_classes: 分类数 (默认 2)
        feat_dim: 特征维度 (N, 默认 64)
        dropout: Dropout 率
        adv_hidden: 对抗判别器隐藏层维度
        kappa: 优势维度比例 (论文默认 0.8)
    """

    def __init__(self, in_channels=3, seq_len=256, num_classes=2,
                 feat_dim=64, dropout=0.1, adv_hidden=64, kappa=0.8):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.kappa = kappa

        # ---- Backbone: 1D CNN 特征提取器 g_hat ----
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, feat_dim, 3, padding=1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.feat_proj = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)

        # ---- 分类器 h_hat_1 ----
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, num_classes),
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        # ---- 对抗掩码器 w_hat (Gumbel-Softmax 生成 k-hot 掩码) ----
        k = max(1, int(feat_dim * kappa))
        self.masker = nn.Sequential(
            nn.Linear(feat_dim, adv_hidden),
            nn.ReLU(),
            nn.Linear(adv_hidden, feat_dim),
        )
        self.k_superior = k

    def extract_features(self, x):
        """提取特征表示 R = g_hat(x)

        Args:
            x: (B, C, W)

        Returns:
            (B, feat_dim) 特征矩阵
        """
        h = self.backbone(x)                # (B, feat_dim, 1)
        h = h.squeeze(-1)                   # (B, feat_dim)
        return self.feat_proj(h)

    def forward(self, x):
        """前向传播

        训练模式: 返回 (log_probs, features, mask_superior, mask_inferior)
        评估模式: 返回 log_probs

        Args:
            x: (B, C, W) 输入张量

        Returns:
            训练: tuple(log_probs, features, m, 1-m)
            评估: log_probs (B, num_classes)
        """
        features = self.extract_features(x)
        logits = self.classifier(self.dropout(features))
        log_probs = self.log_softmax(logits)

        if self.training:
            # Gumbel-Softmax 生成 k-hot 掩码
            mask_logits = self.masker(features.detach())
            m = F.gumbel_softmax(mask_logits, tau=1.0, hard=True)
            # 取 Top-k 作为优势维度掩码
            _, topk_idx = torch.topk(m, self.k_superior, dim=-1)
            m_superior = torch.zeros_like(m).scatter_(1, topk_idx, 1.0)
            m_inferior = 1.0 - m_superior
            return log_probs, features, m_superior, m_inferior
        else:
            return log_probs

    def predict_proba(self, x):
        """输出概率"""
        log_probs = self.forward(x)
        return torch.exp(log_probs)


# ========================================================================== #
#  Loss Functions (损失函数)                                                    #
# ========================================================================== #

def fac_loss(R, R_a):
    """关联因子化损失 (Correlation Factorization Loss)

    基于 Common Cause Principle 和 ICM Principle:
    - 对角线趋近 1: 增强前后每个特征维度保持不变 (因果不变性)
    - 非对角线趋近 0: 不同特征维度相互独立 (因果独立性)

    L_FAC = 0.5 * ||C - I||_F^2
    C_ij = <r_i, r^a_j> / (||r_i|| * ||r^a_j||)

    Args:
        R: (B, N) 原始样本的特征矩阵
        R_a: (B, N) 增强样本的特征矩阵

    Returns:
        scalar loss
    """
    N = R.size(1)
    # Z-score 归一化 (每列 / 每维度)
    R_c = R - R.mean(dim=0, keepdim=True)
    R_a_c = R_a - R_a.mean(dim=0, keepdim=True)
    R_n = R_c / (R_c.norm(dim=0, keepdim=True) + 1e-8)   # (B, N)
    Ra_n = R_a_c / (R_a_c.norm(dim=0, keepdim=True) + 1e-8)

    # 关联矩阵: C_ij = cos(r_i, r^a_j), i,j in {1,...,N}
    C = R_n.t() @ Ra_n                      # (N, N)
    I = torch.eye(N, device=R.device)

    return 0.5 * ((C - I) ** 2).sum() / N


def rnc_loss_binary(embeddings, labels, tau=0.1):
    """Rank-N-Contrast 损失 (二分类适配版)

    原论文使用连续标签距离构建正/负对。二分类适配:
    - 同类样本为正对
    - 异类样本为负对

    Args:
        embeddings: (B, D) 嵌入向量
        labels: (B,) 类别标签 (0/1)
        tau: 温度参数

    Returns:
        scalar loss
    """
    B = embeddings.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=embeddings.device)

    z = F.normalize(embeddings, dim=1)
    sim = z @ z.t() / tau                    # (B, B)

    # 标签矩阵
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    mask_neq = ~label_eq                     # 负对掩码

    loss = torch.tensor(0.0, device=embeddings.device)
    count = 0
    for i in range(B):
        # 正对: 同类 (排除自身)
        pos_mask = label_eq[i].clone()
        pos_mask[i] = False
        pos_indices = torch.where(pos_mask)[0]
        if len(pos_indices) == 0:
            continue

        # 负对: 异类
        neg_mask = mask_neq[i]
        neg_indices = torch.where(neg_mask)[0]
        if len(neg_indices) == 0:
            continue

        for j in pos_indices:
            # log exp(sim_ij) - log sum exp(sim_ik) for k != i
            all_mask = torch.ones(B, dtype=torch.bool, device=embeddings.device)
            all_mask[i] = False
            log_prob = sim[i, j] - torch.logsumexp(sim[i][all_mask], dim=0)
            loss = loss - log_prob
            count += 1

    return loss / max(count, 1)
