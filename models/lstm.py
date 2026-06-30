"""
LSTM 时序分类模型
用于疲劳检测的基线方法，处理滑动窗口的 deviation 特征序列。
"""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM 时序分类器

    输入: (batch_size, window_size) 的 deviation 特征序列
    输出: (batch_size, num_classes) 的分类 logits

    架构:
    - 输入线性映射到 hidden_dim
    - 多层 LSTM 编码时序特征
    - 取最后时刻的隐藏状态
    - 全连接层分类
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 输入映射
        self.input_proj = nn.Linear(input_size, hidden_size)

        # LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        # 分类头
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
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

        # 输入映射: (B, W, 1) -> (B, W, hidden_dim)
        x = self.input_proj(x)

        # LSTM 编码: (B, W, hidden_dim) -> (B, W, hidden_dim*num_directions)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最后时刻的隐藏状态
        if self.bidirectional:
            # 拼接正向和反向的最后隐藏状态
            h_forward = h_n[-2]   # 正向最后一层
            h_backward = h_n[-1]  # 反向最后一层
            h_last = torch.cat([h_forward, h_backward], dim=-1)
        else:
            h_last = h_n[-1]  # 最后一层的最后时刻

        # 分类
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits
