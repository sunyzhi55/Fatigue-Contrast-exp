"""
Prototypical Networks (ProtoNet)
用于疲劳检测的小样本学习基线方法。

参考论文: Prototypical Networks for Few-shot Learning (Snell et al., 2017)

核心思想:
1. 使用共享编码器将支持集和查询集映射到嵌入空间
2. 对每个类别计算原型（支持集样本嵌入的均值）
3. 查询样本根据到各类原型的距离进行分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNet(nn.Module):
    """
    Prototypical Networks

    编码器结构:
    - 3层全连接网络，每层带 BatchNorm 和 ReLU
    - 将窗口特征映射到嵌入空间

    分类方式:
    - 计算每个类别的原型（支持集嵌入的均值）
    - 查询样本分类为最近原型对应的类别
    """

    def __init__(
        self,
        input_size: int = 30,
        hidden_size: int = 64,
        embedding_size: int = 32,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size

        # 特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_size),
        )

    def forward(self, support_windows, support_labels, query_windows, query_labels):
        """
        前向传播（训练模式，计算损失和准确率）

        Args:
            support_windows: (n_way * k_shot, window_size) — 支持集
            support_labels: (n_way * k_shot,) — 支持集标签
            query_windows: (n_way * n_query, window_size) — 查询集
            query_labels: (n_way * n_query,) — 查询集标签

        Returns:
            loss: 标量损失值
            acc: 分类准确率
        """
        # 编码
        support_emb = self.encoder(support_windows)   # (n_way*k_shot, emb_dim)
        query_emb = self.encoder(query_windows)        # (n_way*n_query, emb_dim)

        # 计算各类原型
        prototypes = []
        for c in range(self.num_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                proto = support_emb[mask].mean(dim=0)
            else:
                proto = torch.zeros(self.embedding_size, device=support_emb.device)
            prototypes.append(proto)

        prototypes = torch.stack(prototypes, dim=0)  # (n_way, emb_dim)

        # 计算查询样本到各原型的欧氏距离
        # dist: (n_query_total, n_way)
        dist = torch.cdist(query_emb, prototypes, p=2)

        # 负距离作为 logits（距离越小，相似度越高）
        logits = -dist

        # 计算损失（NLLLoss）
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(log_probs, query_labels)

        # 计算准确率
        preds = torch.argmax(log_probs, dim=1)
        acc = (preds == query_labels).float().mean()

        return loss, acc

    def predict(self, support_windows, support_labels, query_windows):
        """
        预测模式（不计算损失）

        Args:
            support_windows: (n_way * k_shot, window_size) — 支持集
            support_labels: (n_way * k_shot,) — 支持集标签
            query_windows: (n_query, window_size) — 查询集

        Returns:
            probs: (n_query, n_way) — 各类别概率
            preds: (n_query,) — 预测类别
        """
        support_emb = self.encoder(support_windows)
        query_emb = self.encoder(query_windows)

        prototypes = []
        for c in range(self.num_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                proto = support_emb[mask].mean(dim=0)
            else:
                proto = torch.zeros(self.embedding_size, device=support_emb.device)
            prototypes.append(proto)

        prototypes = torch.stack(prototypes, dim=0)
        dist = torch.cdist(query_emb, prototypes, p=2)
        logits = -dist
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        return probs, preds
