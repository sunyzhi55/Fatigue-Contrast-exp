"""
Transformer Encoder 时序分类模型
用于疲劳检测的基线方法，基于自注意力机制捕获时序依赖。
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoderClassifier(nn.Module):
    """
    Transformer Encoder 时序分类器

    输入: (batch_size, window_size) 的 deviation 特征序列
    输出: (batch_size, num_classes) 的分类 logits

    架构:
    - 输入线性映射到 d_model
    - 可学习的位置编码
    - 多层 Transformer Encoder
    - 全局平均池化
    - 全连接层分类
    """

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        max_seq_len: int = 500,
    ):
        super().__init__()
        self.d_model = d_model

        # 输入映射
        self.input_proj = nn.Linear(input_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout)

        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, window_size) — deviation 特征序列

        Returns:
            logits: (batch_size, num_classes)
        """
        # 添加特征维度: (B, W) -> (B, W, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        # 输入映射: (B, W, 1) -> (B, W, d_model)
        x = self.input_proj(x)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer Encoder: (B, W, d_model) -> (B, W, d_model)
        x = self.transformer_encoder(x)

        # 全局平均池化: (B, W, d_model) -> (B, d_model)
        x = x.mean(dim=1)

        # 分类
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
