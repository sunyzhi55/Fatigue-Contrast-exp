"""
TimesNet -- Temporal 2D-Variation Modeling for General Time Series Analysis
基于论文 "TimesNet: Temporal 2D-Variation Modeling for General Time Series
Analysis" (ICLR 2023)
Wu et al. https://openreview.net/forum?id=ju_Uqw384Oq

核心思想:
    1. 通过 FFT 自动发现多周期性 (multi-periodicity)
    2. 将 1D 时序序列按周期重塑为 2D 张量
       (行=周期间变化 inter-period, 列=周期内变化 intra-period)
    3. 使用 2D Inception 卷积同时捕获两种变化模式
    4. 基于 FFT 振幅自适应聚合多周期结果

适配说明:
    原论文支持预测、补全、分类、异常检测四种任务。
    本实现针对分类任务:
    - 输入 (B, W, C): ADF 三通道偏差序列
    - 编码器: DataEmbedding + N 层 TimesBlock + LayerNorm
    - 分类头: GlobalAvgPool + FC (比原论文的 Flatten+FC 更轻量)

参考文献:
    Wu, H., Hu, T., Liu, Y., et al. "TimesNet: Temporal 2D-Variation
    Modeling for General Time Series Analysis." ICLR, 2023.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================================== #
#  Inception Block (2D)                                                        #
# ========================================================================== #

class InceptionBlock(nn.Module):
    """Inception 多尺度 2D 卷积块

    使用多种核大小的 Conv2d 并行处理输入, 输出取均值聚合。
    默认 6 个核: 1×1, 3×3, 5×5, 7×7, 9×9, 11×11

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        num_kernels: 核数量 (核大小 = 2i+1, i=0..num_kernels-1)
    """

    def __init__(self, in_channels, out_channels, num_kernels=6):
        super().__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=2 * i + 1, padding=i, bias=True)
            for i in range(num_kernels)
        ])
        for m in self.kernels:
            nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                    nonlinearity='relu')

    def forward(self, x):
        """x: (B, C, H, W) -> (B, out_channels, H, W)"""
        outputs = [k(x) for k in self.kernels]
        return torch.stack(outputs, dim=-1).mean(dim=-1)


# ========================================================================== #
#  TimesBlock (核心模块)                                                       #
# ========================================================================== #

class TimesBlock(nn.Module):
    """TimesBlock: 1D → 2D 变换 + 2D 卷积 + 自适应聚合

    流程:
        1. FFT 发现 top-k 周期
        2. 对每个周期: 1D → 填充 → 重塑 2D → Inception 2D Conv → 重塑回 1D
        3. Softmax 加权聚合 k 个周期结果
        4. 残差连接

    Args:
        seq_len: 序列长度
        d_model: 模型维度
        d_ff: Inception 瓶颈层维度
        num_kernels: Inception 核数量
        top_k: 选取的 top-k 周期数
    """

    def __init__(self, seq_len, d_model, d_ff, num_kernels=6, top_k=3):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k

        # 2D 卷积: Inception expand → GELU → Inception project
        self.conv = nn.Sequential(
            InceptionBlock(d_model, d_ff, num_kernels),
            nn.GELU(),
            InceptionBlock(d_ff, d_model, num_kernels),
        )

    def _fft_periods(self, x):
        """FFT 发现 top-k 周期

        Args:
            x: (B, T, N)

        Returns:
            periods: (k,) 周期长度列表
            amplitudes: (B, k) 对应振幅 (用于聚合权重)
        """
        B, T, N = x.shape
        xf = torch.fft.rfft(x, dim=1)                      # (B, T//2+1, N)

        # 振幅谱: 在 batch 和 channel 维度取均值
        freq_amp = xf.abs().mean(dim=0).mean(dim=-1)       # (T//2+1,)
        freq_amp[0] = 0                                     # 去除直流分量

        # 选取 top-k 频率
        _, top_idx = torch.topk(freq_amp, min(self.top_k, len(freq_amp)))
        periods = T // top_idx.clamp(min=1)                 # 周期 = T / 频率
        periods = periods.clamp(min=1)

        # 每个样本在 top-k 频率处的振幅
        amplitudes = xf.abs().mean(dim=-1)[:, top_idx]      # (B, k)

        return periods, amplitudes

    def forward(self, x):
        """
        Args:
            x: (B, T, N)

        Returns:
            (B, T, N) 残差连接后的输出
        """
        B, T, N = x.shape
        periods, amplitudes = self._fft_periods(x)

        res_list = []
        for i, p in enumerate(periods):
            p = int(p.item())

            # 填充到 period 的整数倍
            pad_len = (p - T % p) % p
            if pad_len > 0:
                x_pad = F.pad(x, (0, 0, 0, pad_len))        # (B, T+pad, N)
            else:
                x_pad = x

            L = x_pad.shape[1]

            # 重塑为 2D: (B, L, N) → (B, L//p, p, N) → (B, N, L//p, p)
            x_2d = x_pad.reshape(B, L // p, p, N).permute(0, 3, 1, 2)

            # 2D Inception 卷积
            x_2d = self.conv(x_2d)

            # 重塑回 1D: (B, N, L//p, p) → (B, L//p, p, N) → (B, L, N)
            x_1d = x_2d.permute(0, 2, 3, 1).reshape(B, L, N)

            # 截断到原始长度
            res_list.append(x_1d[:, :T, :])

        # 自适应聚合: softmax(amplitudes) 加权求和
        stacked = torch.stack(res_list, dim=-1)              # (B, T, N, k)
        weights = F.softmax(amplitudes, dim=-1)              # (B, k)
        weights = weights.unsqueeze(1).unsqueeze(1)          # (B, 1, 1, k)
        out = (stacked * weights).sum(dim=-1)                # (B, T, N)

        # 残差连接
        return out + x


# ========================================================================== #
#  Data Embedding (Token + Positional)                                         #
# ========================================================================== #

class _TokenEmbedding(nn.Module):
    """Token 嵌入: Conv1d + 循环填充"""

    def __init__(self, c_in, d_model):
        super().__init__()
        self.conv = nn.Conv1d(c_in, d_model, kernel_size=3,
                              padding=1, padding_mode='circular',
                              bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in',
                                nonlinearity='leaky_relu')

    def forward(self, x):
        """(B, T, C) → (B, T, d_model)"""
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class _PositionalEmbedding(nn.Module):
    """固定正弦位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))          # (1, max_len, d)

    def forward(self, x):
        """x: (B, T, *) → (B, T, d_model)"""
        return self.pe[:, :x.size(1), :]


class DataEmbedding(nn.Module):
    """数据嵌入 = Token 嵌入 + 位置编码 + Dropout"""

    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.token_emb = _TokenEmbedding(c_in, d_model)
        self.pos_emb = _PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """(B, T, C) → (B, T, d_model)"""
        return self.dropout(self.token_emb(x) + self.pos_emb(x))


# ========================================================================== #
#  TimesNet 分类器                                                             #
# ========================================================================== #

class TimesNetClassifier(nn.Module):
    """TimesNet 时序分类器

    架构:
        DataEmbedding → [TimesBlock + LayerNorm] × e_layers
        → GELU → Dropout → GlobalAvgPool → FC → Logits

    Args:
        input_size: 输入通道数 C (ADF: 3, 单通道: 1)
        seq_len: 序列长度 (默认 256)
        d_model: 模型维度 (默认 32)
        d_ff: Inception 瓶颈层维度 (默认 64)
        num_kernels: Inception 核数量 (默认 6)
        top_k: FFT top-k 周期数 (论文分类默认 3)
        e_layers: TimesBlock 层数 (论文分类默认 2)
        num_classes: 分类数 (默认 2)
        dropout: Dropout 率 (默认 0.1)
    """

    def __init__(self, input_size=3, seq_len=256, d_model=32, d_ff=64,
                 num_kernels=6, top_k=3, e_layers=2,
                 num_classes=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len

        # 嵌入层
        self.embedding = DataEmbedding(input_size, d_model, dropout)

        # TimesBlock 编码器
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, d_model, d_ff, num_kernels, top_k)
            for _ in range(e_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(e_layers)
        ])

        # 分类头
        self.act = nn.GELU()
        self.dropout_cls = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """前向传播

        Args:
            x: (B, W, C) 输入时序张量

        Returns:
            (B, num_classes) logits
        """
        # 嵌入: (B, W, C) → (B, W, d_model)
        out = self.embedding(x)

        # TimesBlock 编码
        for block, norm in zip(self.blocks, self.norms):
            out = norm(block(out))

        # 激活 + Dropout
        out = self.act(out)
        out = self.dropout_cls(out)

        # 全局平均池化: (B, W, d_model) → (B, d_model)
        out = out.mean(dim=1)

        # 分类
        return self.fc(out)
