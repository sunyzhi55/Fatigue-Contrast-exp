"""
Mamba 时序分类模型
基于选择性状态空间模型（Selective State Space Model）的时序分类基线。

依赖: mamba-ssm (https://github.com/state-spaces/mamba)
安装: pip install mamba-ssm causal-conv1d

参考论文: Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
"""
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba as MambaBlock
except ImportError:
    raise ImportError(
        "Mamba 模型需要 mamba-ssm 包。请安装:\n"
        "  pip install mamba-ssm causal-conv1d\n"
        "或从源码编译:\n"
        "  pip install causal-conv1d\n"
        "  pip install mamba-ssm"
    )


class MambaModel(nn.Module):
    """
    Mamba 时序分类模型

    输入: (batch_size, window_size) 的 deviation 特征序列
    输出: (batch_size, num_classes) 的分类 logits

    架构:
    - 输入线性映射到 d_model
    - 多层 Mamba 块（来自 mamba_ssm 包）
    - LayerNorm + 全局平均池化
    - 全连接层分类

    Mamba 块内部结构:
    - 输入投影 -> [因果卷积 -> SSM] -> 门控输出 -> 残差连接
    - SSM 使用选择性扫描（硬件感知并行算法），训练效率远高于 LSTM
    """

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        n_layer: int = 2,
        d_conv: int = 4,
        d_state: int = 16,
        expand: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_size: 输入特征维度（deviation 是标量，所以为 1）
            d_model: Mamba 模型维度
            n_layer: Mamba 块的层数
            d_conv: 因果卷积的卷积核宽度
            d_state: SSM 状态维度（越大表达能力越强，计算量也越大）
            expand: 内部扩展因子（控制中间层维度 = expand * d_model）
            num_classes: 分类类别数
            dropout: Dropout 概率
        """
        super().__init__()
        self.d_model = d_model

        # 输入映射: (B, L, 1) -> (B, L, d_model)
        self.input_proj = nn.Linear(input_size, d_model)

        # 多层 Mamba 块（来自 mamba_ssm 包，内部包含选择性 SSM）
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_conv=d_conv,
                d_state=d_state,
                expand=expand,
            )
            for _ in range(n_layer)
        ])

        # 层归一化
        self.norm_f = nn.LayerNorm(d_model)
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

        # Mamba 层: (B, L, d_model) -> (B, L, d_model)
        for layer in self.layers:
            x = layer(x)

        # 最终归一化
        x = self.norm_f(x)

        # 全局平均池化: (B, L, d_model) -> (B, d_model)
        x = x.mean(dim=1)

        # 分类
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
