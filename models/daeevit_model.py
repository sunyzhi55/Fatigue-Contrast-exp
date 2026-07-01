"""
DAEEGViT (Domain Adaptive EEG Vision Transformer) 模型
基于论文 "DAEEGViT: A domain adaptive vision transformer framework for
EEG cognitive state identification"

适配说明:
    原论文使用 EEG 差分熵 (DE) 特征作为输入 (C_channels × 5_bands)。
    本实现将输入替换为偏差序列 (deviation sequences)，
    输入形状 (B, in_channels, seq_len) → (B, 3, 256)。

网络架构:
    - PatchEmbed: 1D 卷积将序列切分为 patches
    - CLS token: 用于分类和域适应 MMD 对齐
    - Block: Attention + MBConv + MLP (三子模块)
    - Classifier: 单层 FC

域适应机制:
    在 CLS token 特征上计算 MMD 损失，实现跨域分布对齐。
    forward 返回 (logits, cls_features)，cls_features 用于 MMD。
"""
from functools import partial
from collections import OrderedDict
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor


# ========================================================================== #
#  基础模块                                                                    #
# ========================================================================== #

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Stochastic Depth: 训练时随机丢弃残差路径"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNAct(nn.Module):
    """Conv1d + BatchNorm1d + Activation"""

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 groups=1, norm_layer=None, activation_layer=None):
        super().__init__()
        padding = (kernel_size - 1) // 2
        norm_layer = norm_layer or nn.BatchNorm1d
        activation_layer = activation_layer or nn.GELU

        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = norm_layer(out_planes)
        self.act = activation_layer()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation for 1D (序列维度做 squeeze)"""

    def __init__(self, input_c, expand_c, se_ratio=0.25):
        super().__init__()
        squeeze_c = max(1, int(input_c * se_ratio))
        self.conv_reduce = nn.Conv1d(expand_c, squeeze_c, 1)
        self.act1 = nn.GELU()
        self.conv_expand = nn.Conv1d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        scale = x.mean(2, keepdim=True)        # (B, expand_c, 1)
        scale = self.act1(self.conv_reduce(scale))
        scale = self.act2(self.conv_expand(scale))
        return scale * x


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv (1D 版本)

    结构: 1x1 expand → DW conv → SE → 1x1 project + residual
    在 DAEEGViT 中操作 token 序列维度 (N+1 tokens as "channels")
    """

    def __init__(self, kernel_size, input_c, out_c, expand_ratio,
                 stride, se_ratio, drop_rate, norm_layer):
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f"illegal stride value: {stride}")

        self.has_shortcut = (stride == 1 and input_c == out_c)
        expanded_c = input_c * expand_ratio

        self.expand_conv = ConvBNAct(input_c, expanded_c, kernel_size=1,
                                     norm_layer=norm_layer, activation_layer=nn.GELU)
        self.dwconv = ConvBNAct(expanded_c, expanded_c, kernel_size=kernel_size,
                                stride=stride, groups=expanded_c,
                                norm_layer=norm_layer, activation_layer=nn.GELU)
        self.se = SqueezeExcite(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()
        self.project_conv = ConvBNAct(expanded_c, out_c, kernel_size=1,
                                      norm_layer=norm_layer, activation_layer=nn.Identity)
        self.out_channels = out_c
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x):
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        result = self.project_conv(result)
        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x
        return result


class Mlp(nn.Module):
    """标准 ViT MLP"""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class PatchEmbed(nn.Module):
    """1D Patch Embedding: Conv1d 将序列切分为固定大小的 patches

    输入: (B, C, L) → 输出: (B, num_patches, embed_dim)
    """

    def __init__(self, seq_len=256, patch_size=32, in_c=3,
                 embed_dim=64, norm_layer=None):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.grid_size = seq_len // patch_size
        self.num_patches = self.grid_size

        self.proj = nn.Conv1d(in_c, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        assert L == self.seq_len, \
            f"Input seq length ({L}) doesn't match model ({self.seq_len})."
        x = self.proj(x).transpose(1, 2)  # (B, embed_dim, N) → (B, N, embed_dim)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """多头自注意力"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop_ratio=0., proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ========================================================================== #
#  Transformer Block (Attention + MBConv + MLP)                               #
# ========================================================================== #

class Block(nn.Module):
    """DAEEGViT Block: 三子模块 (Attention → MBConv → MLP)

    与标准 ViT Block 的区别: 在 Attention 和 MLP 之间插入了 MBConv 模块，
    用于捕获局部特征 (MBConv 在 token 序列维度上做卷积)。
    """

    def __init__(self, dim, num_heads, num_tokens,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 mbconv_expand_ratio=4, mbconv_se_ratio=0.25):
        super().__init__()
        # Sub-module 1: Multi-Head Self-Attention
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio,
                              proj_drop_ratio=drop_ratio)

        # Sub-module 2: MBConv (在 token 序列维度上做卷积)
        # input_c = out_c = num_tokens (包括 CLS token)
        self.mbconv = MBConv(
            kernel_size=3, input_c=num_tokens, out_c=num_tokens,
            expand_ratio=mbconv_expand_ratio, stride=1,
            se_ratio=mbconv_se_ratio, drop_rate=drop_ratio,
            norm_layer=nn.BatchNorm1d,
        )

        # Sub-module 3: MLP
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        # Attention + residual
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # MBConv (操作 token 维度) + residual
        x = x + self.mbconv(x)
        # MLP + residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ========================================================================== #
#  DAEEGViT 主模型                                                            #
# ========================================================================== #

class DAEEGViTModel(nn.Module):
    """DAEEGViT: Domain Adaptive EEG Vision Transformer

    输入: (B, in_channels, seq_len)，如 (B, 3, 256) 的 ADF 三通道偏差序列
    输出: (logits, cls_features)
        - logits: (B, num_classes) 分类预测 (未激活)
        - cls_features: (B, embed_dim) CLS token 特征 (用于 MMD 域适应)
    """

    def __init__(self, seq_len=256, patch_size=32, in_channels=3,
                 num_classes=2, embed_dim=64, depth=4, num_heads=4,
                 mlp_ratio=4.0, qkv_bias=True, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0.,
                 mbconv_expand_ratio=4, mbconv_se_ratio=0.25,
                 representation_size=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            seq_len=seq_len, patch_size=patch_size, in_c=in_channels,
            embed_dim=embed_dim, norm_layer=norm_layer,
        )
        num_patches = self.patch_embed.num_patches
        num_tokens = num_patches + 1  # +1 for CLS token

        # CLS token + Position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]

        # Transformer Blocks (Attention + MBConv + MLP)
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, num_tokens=num_tokens,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                drop_path_ratio=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                mbconv_expand_ratio=mbconv_expand_ratio,
                mbconv_se_ratio=mbconv_se_ratio,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Pre-logits (optional representation layer)
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh()),
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Weight initialization
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        """提取 CLS token 特征

        Args:
            x: (B, in_channels, seq_len)

        Returns:
            cls_features: (B, embed_dim) 或 (B, representation_size)
        """
        x = self.patch_embed(x)                         # (B, N, D)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)            # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return self.pre_logits(x[:, 0])                 # (B, D) CLS token

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, seq_len)

        Returns:
            logits: (B, num_classes) 分类预测 (原始 logits，未经 sigmoid/softmax)
            cls_features: (B, embed_dim) CLS token 特征 (用于 MMD 域适应)
        """
        cls_features = self.forward_features(x)
        logits = self.head(cls_features)
        return logits, cls_features


def _init_vit_weights(m):
    """ViT 权重初始化"""
    if isinstance(m, nn.Linear):
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
