"""
Relation Network (RelationNet)
用于疲劳检测的小样本学习基线方法。

参考论文: Learning to Compare: Relation Network for Few-Shot Learning (Sung et al., 2018)

核心思想:
1. 使用共享编码器将支持集和查询集映射到嵌入空间
2. 对每个类别计算原型（支持集样本嵌入的均值）
3. 将查询样本嵌入与各原型拼接，通过关系模块计算关系分数
4. 关系分数最高的类别即为预测结果
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationNet(nn.Module):
    """
    Relation Network

    与 ProtoNet 的区别:
    - ProtoNet 使用固定的距离度量（欧氏距离）
    - RelationNet 使用可学习的关系模块（MLP）来度量相似性

    结构:
    - 特征编码器: 将输入映射到嵌入空间
    - 关系模块: 计算查询样本与原型之间的关系分数
    """

    def __init__(
        self,
        input_size: int = 30,
        hidden_size: int = 64,
        embedding_size: int = 32,
        relation_size: int = 16,
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

        # 关系模块: 输入为 (query_emb || proto_emb)，维度为 2*embedding_size
        self.relation_module = nn.Sequential(
            nn.Linear(embedding_size * 2, relation_size),
            nn.BatchNorm1d(relation_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(relation_size, relation_size),
            nn.BatchNorm1d(relation_size),
            nn.ReLU(),
            nn.Linear(relation_size, 1),
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
        n_query = query_windows.size(0)

        # 编码
        support_emb = self.encoder(support_windows)   # (n_way*k_shot, emb_dim)
        query_emb = self.encoder(query_windows)        # (n_query, emb_dim)

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

        # 计算关系分数
        # 对每个查询样本，与每个原型拼接后送入关系模块
        relation_scores = []
        for c in range(self.num_classes):
            proto = prototypes[c].unsqueeze(0).expand(n_query, -1)  # (n_query, emb_dim)
            pair = torch.cat([query_emb, proto], dim=1)  # (n_query, 2*emb_dim)
            score = self.relation_module(pair)  # (n_query, 1)
            relation_scores.append(score)

        # (n_query, n_way)
        relation_scores = torch.cat(relation_scores, dim=1)

        # Softmax 归一化 -> log 概率
        log_probs = F.log_softmax(relation_scores, dim=1)

        # NLLLoss
        loss = F.nll_loss(log_probs, query_labels)

        # 准确率
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
        n_query = query_windows.size(0)

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

        relation_scores = []
        for c in range(self.num_classes):
            proto = prototypes[c].unsqueeze(0).expand(n_query, -1)
            pair = torch.cat([query_emb, proto], dim=1)
            score = self.relation_module(pair)
            relation_scores.append(score)

        relation_scores = torch.cat(relation_scores, dim=1)
        probs = F.softmax(relation_scores, dim=1)
        preds = torch.argmax(probs, dim=1)

        return probs, preds
