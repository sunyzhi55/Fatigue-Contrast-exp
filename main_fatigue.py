"""
疲劳检测对比实验训练脚本

支持的实验类型:
- 时序基线: LSTM, Transformer, Mamba, TimesNet
- 小样本学习: ProtoNet, RelationNet
- 域适应: MLDA, DAEEGViT, LA-MSDA, DANN, DeepCORAL
- 域泛化: InterpretableCNN, AFM-CIR

功能:
- 训练前自动保存配置到 YAML 文件
- 支持基于受试者ID的 K-Fold / LOSO 数据划分
- 使用 RuntimeObserver 进行完整指标监控（Acc/Pre/Rec/Spe/F1/Kappa/AUC/BalAcc）
- 训练完成后批量评估所有 fold 的模型

使用方法:
    python main_fatigue.py --exp_name Fatigue_LSTM_baseline
    python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline
    python main_fatigue.py --exp_name Fatigue_DANN_baseline
    python main_fatigue.py --exp_name Fatigue_DeepCORAL_baseline
    python main_fatigue.py --exp_name Fatigue_TimesNet_baseline
    python main_fatigue.py --exp_name Fatigue_AFM_CIR_baseline
"""
import sys
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

from configs.fatigue_temporal_baselines import fatigue_temporal_experiments
from configs.fatigue_fewshot_baselines import fatigue_fewshot_experiments
from configs.fatigue_domain_adapt_baselines import fatigue_da_experiments
from configs.fatigue_domain_generalization import fatigue_dg_experiments
from data.fatigue_dataset import (
    FatigueDataset, FewShotFatigueDataset,
    build_temporal_loader, build_fewshot_loader,
)
from data.gaipat_dataset import (
    GaipatDataset, scan_gaipat_subject_ids,
    generate_gaipat_loso_folds, generate_gaipat_kfold,
)
from models.get_model import get_model
from utils.basic import get_optimizer, get_scheduler
from utils.observer import RuntimeObserver
from utils.reproducibility import set_global_seed
from utils.metrics_recorder import MetricsRecorder


# ========================================================================== #
#  工具函数                                                                   #
# ========================================================================== #

def save_config_yaml(args, save_path: str):
    """将当前实验配置保存为 YAML 文件，便于后续查看和复现"""
    try:
        import yaml
    except ImportError:
        print("⚠️  PyYAML 未安装，跳过配置保存。请运行: pip install pyyaml")
        return

    config_dict = {}
    skip_keys = {"parser"}
    for key, value in vars(args).items():
        if key.startswith("_") or key in skip_keys:
            continue
        if callable(value):
            continue
        if isinstance(value, Path):
            config_dict[key] = str(value)
        elif isinstance(value, torch.device):
            config_dict[key] = str(value)
        elif isinstance(value, set):
            config_dict[key] = sorted(list(value))
        else:
            try:
                import json
                json.dumps(value)
                config_dict[key] = value
            except (TypeError, ValueError):
                config_dict[key] = str(value)

    yaml_path = Path(save_path) / "config.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    print(f"✅ 配置已保存到 {yaml_path}")


def build_model_with_kwargs(args, device):
    """构建模型，自动传递模型特有参数"""
    # 时序模型: 输入 (B, W, C)，C=3(ADF三通道) 或 1(单通道)；模型内部按 input_size 投影
    # 小样本模型: 输入 (B, W*C) 展平向量；input_size = window_size * C
    # MLDA域适应模型: 输入 (B, W*C) 展平向量；input_size = window_size * C
    is_fewshot = args.model_name in ("protonet", "relationnet")
    is_mlda = args.model_name == "mlda"
    is_flat_da = args.model_name in ("dann", "deepcoral")
    use_adf = getattr(args, "use_adf", True)
    num_channels = 3 if use_adf else 1
    if is_fewshot or is_mlda or is_flat_da:
        input_size = getattr(args, "window_size", 30) * num_channels
    else:
        input_size = num_channels

    model_kwargs = {
        "input_size": input_size,
        "hidden_size": getattr(args, "hidden_size", 64),
        "num_layers": getattr(args, "num_layers", 2),
        "dropout": getattr(args, "dropout", 0.3),
        "bidirectional": getattr(args, "bidirectional", False),
        "d_model": getattr(args, "d_model", 64),
        "nhead": getattr(args, "nhead", 4),
        "dim_feedforward": getattr(args, "dim_feedforward", 128),
        "max_seq_len": getattr(args, "max_seq_len", 500),
        "n_layer": getattr(args, "n_layer", 2),
        "d_conv": getattr(args, "d_conv", 4),
        "d_state": getattr(args, "d_state", 16),
        "expand": getattr(args, "expand", 2),
        "embedding_size": getattr(args, "embedding_size", 32),
        "relation_size": getattr(args, "relation_size", 16),
        "feat_dim": getattr(args, "feat_dim", 32),
        # DAEEGViT 特有参数
        "seq_len": getattr(args, "window_size", 256),
        "in_channels": num_channels,
        "patch_size": getattr(args, "patch_size", 32),
        "embed_dim": getattr(args, "embed_dim", 64),
        "depth": getattr(args, "depth", 4),
        "num_heads": getattr(args, "num_heads", 4),
        "mlp_ratio": getattr(args, "mlp_ratio", 4.0),
        "qkv_bias": getattr(args, "qkv_bias", True),
        "attn_drop_ratio": getattr(args, "attn_drop_ratio", 0.0),
        "drop_path_ratio": getattr(args, "drop_path_ratio", 0.1),
        "mbconv_expand_ratio": getattr(args, "mbconv_expand_ratio", 4),
        "mbconv_se_ratio": getattr(args, "mbconv_se_ratio", 0.25),
        "representation_size": getattr(args, "representation_size", None),
        # LA-MSDA 特有参数
        "num_sources": getattr(args, "num_sources", 5),
        # DANN 特有参数
        "domain_hidden": getattr(args, "domain_hidden", 1024),
        # InterpretableCNN 特有参数
        "n_filters": getattr(args, "n_filters", 16),
        "depth_multiplier": getattr(args, "depth_multiplier", 2),
        "kernel_size": getattr(args, "kernel_size", 64),
        # AFM-CIR 特有参数
        "feat_dim": getattr(args, "feat_dim", 64),
        "adv_hidden": getattr(args, "adv_hidden", 64),
        "kappa": getattr(args, "kappa", 0.8),
        # TimesNet 特有参数
        "d_ff": getattr(args, "d_ff", 64),
        "num_kernels": getattr(args, "num_kernels", 6),
        "top_k": getattr(args, "top_k", 3),
        "e_layers": getattr(args, "e_layers", 2),
    }
    return get_model(
        args.model_name, args.num_classes,
        args.checkpoint_path, device, **model_kwargs
    )


def scan_subject_ids(data_dir: str, difficulty: str = None):
    """
    扫描 JSONL 数据目录，提取受试者ID

    Args:
        data_dir: JSONL 数据目录
        difficulty: "easy" / "hard" / None(全部)

    Returns:
        排序后的受试者ID列表，如 ["001", "002", ..., "020"]
    """
    from pathlib import Path as P
    data_path = P(data_dir)
    subject_ids = set()
    for f in data_path.glob("*.jsonl"):
        parts = f.stem.split("_")
        if len(parts) >= 3 and parts[1] in ("easy", "hard") and parts[2] in ("alert", "sleepy"):
            if difficulty is not None and parts[1] != difficulty:
                continue
            subject_ids.add(parts[0])
    return sorted(subject_ids)


def generate_loso_folds(data_dir: str, test_ids=None, difficulty=None):
    """
    自动生成 LOSO (Leave-One-Subject-Out) 的 folds 配置

    每个 fold 留一个受试者做验证，其余做训练。
    如果指定了 test_ids，则 test_ids 不参与 LOSO 划分。

    Returns:
        folds: {1: {"val_ids": ["001"]}, 2: {"val_ids": ["002"]}, ...}
        all_subject_ids: 全部受试者ID（含 test_ids）
    """
    all_ids = scan_subject_ids(data_dir, difficulty)
    if not all_ids:
        raise RuntimeError(f"未在 {data_dir} 中找到任何受试者数据")

    test_set = set(test_ids) if test_ids else set()
    loo_ids = [s for s in all_ids if s not in test_set]

    folds = {}
    for i, sid in enumerate(loo_ids, start=1):
        folds[i] = {"val_ids": [sid]}

    print(f"[LOSO] 检测到 {len(all_ids)} 个受试者: {all_ids}")
    if test_ids:
        print(f"[LOSO] 测试集受试者: {test_ids}（不参与 LOSO 划分）")
    print(f"[LOSO] 生成 {len(folds)} 个 fold，每折留 1 个受试者做验证")

    return folds, all_ids


def build_fold_data(args, fold_config, mode="train"):
    """
    根据 fold 配置构建数据集

    Args:
        args: 配置参数
        fold_config: {"val_ids": [...], "train_ids": [...](可选)}
        mode: "train" / "val" / "test"

    Returns:
        dataset: FatigueDataset / GaipatDataset / FewShotFatigueDataset
    """
    is_fewshot = "fewshot" in args.dataset_name.lower()
    is_gaipat = getattr(args, "data_type", "fatigue") == "gaipat"
    difficulty = getattr(args, "difficulty", None)  # "easy" / "hard" / None
    use_adf = getattr(args, "use_adf", True)
    local_mean_size = getattr(args, "local_mean_size", 16)
    per_sample_norm = getattr(args, "per_sample_norm", False)

    if mode == "test":
        subject_ids = getattr(args, "test_ids", None)
    elif mode == "val":
        subject_ids = fold_config.get("val_ids", None)
    else:
        val_ids = fold_config.get("val_ids", [])
        test_ids = getattr(args, "test_ids", []) or []
        exclude_ids = set(val_ids) | set(test_ids)
        # GAIPAT 的 fold_config 可能显式提供 train_ids
        if "train_ids" in fold_config and mode == "train":
            subject_ids = fold_config["train_ids"]
        else:
            all_subject_ids = getattr(args, "all_subject_ids", None)
            if all_subject_ids is not None:
                subject_ids = [s for s in all_subject_ids if s not in exclude_ids]
            else:
                subject_ids = None

    adf_kwargs = dict(use_adf=use_adf, local_mean_size=local_mean_size)

    if is_gaipat:
        # GAIPAT 数据集: 使用 GaipatDataset（默认开启 per_sample_norm）
        dataset = GaipatDataset(
            data_dir=args.data_dir,
            window_size=args.window_size,
            feature_name=getattr(args, "feature_name", "deviation_cm"),
            subject_ids=subject_ids,
            per_sample_norm=per_sample_norm,
            **adf_kwargs,
        )
    elif is_fewshot:
        dataset = FewShotFatigueDataset(
            data_dir=args.data_dir,
            window_size=args.window_size,
            stride=args.stride,
            feature_name=args.feature_name,
            subject_ids=subject_ids,
            difficulty=difficulty,
            per_sample_norm=per_sample_norm,
            **adf_kwargs,
        )
    else:
        dataset = FatigueDataset(
            data_dir=args.data_dir,
            window_size=args.window_size,
            stride=args.stride,
            feature_name=args.feature_name,
            subject_ids=subject_ids,
            difficulty=difficulty,
            per_sample_norm=per_sample_norm,
            **adf_kwargs,
        )

    if mode == "train" and subject_ids is None and not is_fewshot and not is_gaipat:
        val_set = set(fold_config.get("val_ids", []))
        test_set = set(getattr(args, "test_ids", []) or [])
        exclude = val_set | test_set
        if exclude:
            keep_indices = [i for i in range(len(dataset))
                           if dataset.subject_ids[i] not in exclude]
            dataset = _SubsetFatigueDataset(dataset, keep_indices)

    return dataset


class _SubsetFatigueDataset(FatigueDataset):
    """FatigueDataset 的子集视图，用于训练集过滤"""

    def __init__(self, parent, indices):
        self.windows = [parent.windows[i] for i in indices]
        self.labels = [parent.labels[i] for i in indices]
        self.subject_ids = [parent.subject_ids[i] for i in indices]
        self.file_ids = [parent.file_ids[i] for i in indices]
        self.difficulties = [parent.difficulties[i] for i in indices]
        self.num_samples = len(self.windows)
        self.window_size = parent.window_size
        self.stride = parent.stride
        self.feature_name = parent.feature_name
        self.data_dir = parent.data_dir
        print(f"[_SubsetFatigueDataset] 训练集子集: {self.num_samples} 个样本")


class GaipatFewShotAdapter:
    """
    将 GaipatDataset 适配为 FewShotFatigueDataset 接口
    用于 GAIPAT 数据的小样本模型评估 (ProtoNet / RelationNet)
    """

    def __init__(self, gaipat_dataset):
        self.window_size = gaipat_dataset.window_size
        self.use_adf = gaipat_dataset.use_adf
        self.num_classes = 2
        self.class_to_samples = {0: [], 1: []}

        for i in range(len(gaipat_dataset)):
            label = gaipat_dataset.labels[i]
            self.class_to_samples[label].append({
                "window": gaipat_dataset.windows[i],
                "subject_id": gaipat_dataset.subject_ids[i],
                "file_id": gaipat_dataset.file_ids[i],
                "difficulty": None,
            })

        total = sum(len(v) for v in self.class_to_samples.values())
        print(f"[GaipatFewShotAdapter] {total} samples"
              f" (alert={len(self.class_to_samples[0])},"
              f" focused={len(self.class_to_samples[1])})")

        for c in [0, 1]:
            if len(self.class_to_samples[c]) == 0:
                raise RuntimeError(f"类别 {c} 没有样本")

    def get_class_samples_count(self):
        return {c: len(v) for c, v in self.class_to_samples.items()}

    def _flatten_windows(self, items):
        arr = np.stack([item["window"] for item in items], axis=0)
        arr = arr.reshape(arr.shape[0], -1)
        return torch.from_numpy(arr).float()

    def sample_episode(self, n_way=2, k_shot=5, n_query=10):
        import random
        available = [c for c in range(self.num_classes)
                     if len(self.class_to_samples[c]) >= k_shot + n_query]
        if len(available) < n_way:
            available = [c for c in range(self.num_classes)
                         if len(self.class_to_samples[c]) >= k_shot + 1]
            if len(available) < n_way:
                raise ValueError(
                    f"可用类别 ({len(available)}) < n_way ({n_way})"
                )

        selected_classes = random.sample(available, n_way)
        support_items, support_labels = [], []
        query_items, query_labels = [], []

        for new_label, cls in enumerate(selected_classes):
            samples = self.class_to_samples[cls]
            selected = random.sample(samples, min(k_shot + n_query, len(samples)))
            support_items.extend(selected[:k_shot])
            support_labels.extend([new_label] * k_shot)
            query_items.extend(selected[k_shot:k_shot + n_query])
            query_labels.extend([new_label] * n_query)

        return {
            "support_windows": self._flatten_windows(support_items),
            "support_labels": torch.tensor(support_labels, dtype=torch.long),
            "query_windows": self._flatten_windows(query_items),
            "query_labels": torch.tensor(query_labels, dtype=torch.long),
        }

    def __len__(self):
        total = sum(len(v) for v in self.class_to_samples.values())
        return max(total // 10, 1)

    def __getitem__(self, idx):
        return self.sample_episode()


def create_observer(args, device, log_dir):
    """创建 RuntimeObserver，统一管理指标、日志、早停、模型保存"""
    return RuntimeObserver(
        log_dir=str(log_dir),
        device=device,
        num_classes=args.num_classes,
        task="multiclass" if args.num_classes > 2 else "binary",
        average="macro" if args.num_classes > 2 else "micro",
        patience=getattr(args, "patience", 20),
        hyperparameters=vars(args),
    )


# ========================================================================== #
#  时序模型训练/验证（使用 RuntimeObserver）                                    #
# ========================================================================== #

def run_temporal_fold(args, device, fold_idx, fold_config, recorder=None):
    """训练一个 fold 的时序模型，使用 RuntimeObserver 管理全流程"""
    print(f"\n{'='*60}")
    print(f"🔥 Fold {fold_idx} / {args.k_fold}")
    print(f"{'='*60}")

    # ---- 构建数据集 ----
    train_dataset = build_fold_data(args, fold_config, mode="train")
    val_dataset = build_fold_data(args, fold_config, mode="val")

    train_loader = build_temporal_loader(train_dataset, args.batch_size, shuffle=True)
    val_loader = build_temporal_loader(val_dataset, args.batch_size, shuffle=False)

    print(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")

    # ---- 构建模型 ----
    model = build_model_with_kwargs(args, device)
    optimizer = get_optimizer(args.optimizer_name, model.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args, train_loader)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=getattr(args, "label_smoothing", 0.0)
    ).to(device)

    # ---- 创建 Observer ----
    observer = create_observer(args, device, args.save_dir)

    # ---- 训练循环 ----
    for epoch in range(1, args.epochs + 1):
        # === 训练阶段 ===
        model.train()
        observer.reset()

        for batch in train_loader:
            windows = batch["window"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(windows)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            prob = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(prob, dim=1)
            observer.train_update(loss, prob, predictions, labels)

        # === 验证阶段 ===
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                windows = batch["window"].to(device)
                labels = batch["label"].to(device)

                outputs = model(windows)
                loss = criterion(outputs, labels)

                prob = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(prob, dim=1)
                observer.eval_update(loss, prob, predictions, labels)

        # === 使用 observer 的 execute 方法：计算指标、打印、早停、保存最佳模型 ===
        should_stop = observer.execute(
            epoch, args.epochs,
            len(train_loader.dataset), len(val_loader.dataset),
            fold=fold_idx, model=model,
        )

        # 记录该轮 train + val 指标到 history.csv
        if recorder is not None:
            is_best = (observer.best_dicts.get("epoch", 0) == epoch)
            recorder.record_epoch(observer, fold_idx, epoch, is_best=is_best)

        # 学习率调度（非 OneCycleLR 的在 epoch 级别更新）
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        if should_stop:
            print("⚠️  早停触发")
            break

    # ---- 记录该 fold 最佳结果 ----
    observer.finish(fold_idx)
    best_val_acc = float(observer.best_dicts["Accuracy"])
    if recorder is not None:
        recorder.record_best(observer, fold_idx)
    print(f"Fold {fold_idx} 最佳验证: Acc={best_val_acc:.4f}")
    return best_val_acc


# ========================================================================== #
#  小样本模型训练/验证（使用 RuntimeObserver）                                  #
# ========================================================================== #

def _fewshot_train_epoch(model, loader, optimizer, scheduler, device, model_name,
                         observer=None):
    """小样本模型训练一个 epoch（episodic training）"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for episode in loader:
        support_w = episode["support_windows"].squeeze(0).to(device)
        support_l = episode["support_labels"].squeeze(0).to(device)
        query_w = episode["query_windows"].squeeze(0).to(device)
        query_l = episode["query_labels"].squeeze(0).to(device)

        optimizer.zero_grad()
        loss, acc = model(support_w, support_l, query_w, query_l)

        # 用同一份权重记录逐样本训练指标，使 observer 的 train 指标被真正填充
        # （否则 compute_result 计算 train AUROC 时会因无样本而报错）
        if observer is not None:
            with torch.no_grad():
                probs, preds = model.predict(support_w, support_l, query_w)
                probs = probs.to(device)
                preds = preds.to(device)
            observer.train_update(loss, probs, preds, query_l)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1

    return total_loss / num_batches, total_acc / num_batches


@torch.no_grad()
def _fewshot_eval_epoch(model, loader, device, observer, criterion, mode="eval"):
    """
    小样本模型逐样本验证/测试（使用 observer 记录完整指标）

    通过 model.predict() 对查询集逐样本预测，转换为标准分类格式。
    """
    model.eval()
    update_fn = observer.eval_update if mode == "eval" else observer.test_update

    for episode in loader:
        support_w = episode["support_windows"].squeeze(0).to(device)
        support_l = episode["support_labels"].squeeze(0).to(device)
        query_w = episode["query_windows"].squeeze(0).to(device)
        query_l = episode["query_labels"].squeeze(0).to(device)

        # 使用 predict 获取逐样本概率
        probs, preds = model.predict(support_w, support_l, query_w)
        probs = probs.to(device)
        preds = preds.to(device)
        query_l = query_l.to(device)

        # 用 NLLLoss 计算一个 loss 用于 observer 记录
        log_probs = torch.log(probs + 1e-8)
        loss = criterion(log_probs, query_l)

        update_fn(loss, probs, preds, query_l)


def run_fewshot_fold(args, device, fold_idx, fold_config, recorder=None):
    """训练一个 fold 的小样本模型，使用 RuntimeObserver 管理验证指标"""
    print(f"\n{'='*60}")
    print(f"🔥 Fold {fold_idx} / {args.k_fold} (Few-Shot)")
    print(f"{'='*60}")

    n_way = getattr(args, "n_way", 2)
    k_shot = getattr(args, "k_shot", 5)
    n_query = getattr(args, "n_query", 10)
    episodes_per_epoch = getattr(args, "episodes_per_epoch", 100)

    # ---- 构建数据集 ----
    train_dataset = build_fold_data(args, fold_config, mode="train")
    val_dataset = build_fold_data(args, fold_config, mode="val")

    train_loader = build_fewshot_loader(
        train_dataset, n_way, k_shot, n_query, episodes_per_epoch
    )
    val_loader = build_fewshot_loader(
        val_dataset, n_way, k_shot, n_query, episodes_per_epoch // 2
    )

    train_counts = train_dataset.get_class_samples_count()
    val_counts = val_dataset.get_class_samples_count()
    print(f"训练集: {train_counts}, 验证集: {val_counts}")

    # ---- 构建模型 ----
    model = build_model_with_kwargs(args, device)
    optimizer = get_optimizer(args.optimizer_name, model.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_policy == "onecycle":
        try:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr * 5,
                steps_per_epoch=episodes_per_epoch,
                epochs=args.epochs,
                anneal_strategy="cos",
                pct_start=0.1,
            )
        except Exception:
            scheduler = None

    # ---- Observer ----
    observer = create_observer(args, device, args.save_dir)
    criterion = nn.NLLLoss().to(device)

    # ---- 训练循环 ----
    for epoch in range(1, args.epochs + 1):
        # === 训练（episodic，同时通过 observer 记录逐样本 train 指标） ===
        train_loss, train_acc = _fewshot_train_epoch(
            model, train_loader, optimizer, scheduler, device, args.model_name,
            observer=observer,
        )

        # === 验证（逐样本，使用 observer 记录完整指标） ===
        _fewshot_eval_epoch(model, val_loader, device, observer, criterion, mode="eval")

        # 计算训练+验证指标并打印（train 指标已在训练阶段通过 train_update 填充）
        observer.compute_result(
            epoch, episodes_per_epoch, len(val_loader), fold=fold_idx
        )
        observer.print_result(epoch, args.epochs)

        # 判断是否为最佳轮（在 get_best 之前判断，用于 history 标记）
        new_best = observer.eval_balance_accuracy > observer.best_dicts["BalanceAccuracy"]

        # 记录该轮 train + val 指标到该 fold 的 history 文件
        if recorder is not None:
            recorder.record_epoch(observer, fold_idx, epoch, is_best=new_best)

        # 手动检查是否更新最佳模型 + 早停
        if new_best:
            observer.get_best(epoch)
            model_path = str(Path(args.save_dir) / f"{args.exp_name}_best_model_fold{fold_idx}.pth")
            torch.save(model.state_dict(), model_path)
            observer.log(f"✅ Best model saved to {model_path}")
        else:
            observer.early_stopping.counter += 1
            if observer.early_stopping.counter >= observer.early_stopping.patience:
                observer.log(f"⚠️  早停触发 (patience={observer.early_stopping.patience})")
                observer.reset()
                break

        # 学习率调度
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        observer.reset()

    # ---- 记录该 fold 最佳结果 ----
    observer.finish(fold_idx)
    best_val_acc = float(observer.best_dicts["Accuracy"])
    if recorder is not None:
        recorder.record_best(observer, fold_idx)
    print(f"Fold {fold_idx} 最佳验证: Acc={best_val_acc:.4f}")
    return best_val_acc


# ========================================================================== #
#  MLDA 域适应训练/验证（使用 RuntimeObserver）                               #
# ========================================================================== #

def run_mlda_fold(args, device, fold_idx, fold_config, recorder=None):
    """训练一个 fold 的 MLDA 域适应模型

    训练范式:
        - 源域 = 训练集受试者（带标签）
        - 目标域 = 验证集受试者（训练时标签不可见，使用伪标签做 ICD）
        - 评估在目标域上进行

    损失函数:
        L = L_CE + 2 * [(1-λ)*L_inter + λ*L_intra]
        - L_CE: 源域交叉熵
        - L_inter: Wasserstein 域间分布对齐
        - L_intra: ICD 类内/类间域差异对比损失
        - λ: sigmoid 调度，从偏重域间逐渐过渡到偏重域内
    """
    from models.mlda_model import DomainProjection
    from utils.mlda_loss import idcd_loss, compute_wasserstein_distance

    print(f"\n{'='*60}")
    print(f"🔥 Fold {fold_idx} / {args.k_fold} (MLDA 域适应)")
    print(f"{'='*60}")

    # ---- 构建源域 (train subjects) 和目标域 (val subjects) 数据集 ----
    source_dataset = build_fold_data(args, fold_config, mode="train")
    target_dataset = build_fold_data(args, fold_config, mode="val")

    # MLDA 需要展平的向量输入: (B, window_size, C) -> (B, window_size * C)
    src_windows = np.stack(source_dataset.windows, axis=0)  # (N, W, C) or (N, W)
    src_flat = src_windows.reshape(src_windows.shape[0], -1).astype(np.float32)
    src_labels = np.array(source_dataset.labels, dtype=np.int64)

    tar_windows = np.stack(target_dataset.windows, axis=0)
    tar_flat = tar_windows.reshape(tar_windows.shape[0], -1).astype(np.float32)
    tar_labels = np.array(target_dataset.labels, dtype=np.int64)  # 仅用于评估，训练不可见

    # 构建 TensorDataset + DataLoader
    src_dset = torch.utils.data.TensorDataset(
        torch.from_numpy(src_flat), torch.from_numpy(src_labels)
    )
    tar_dset = torch.utils.data.TensorDataset(
        torch.from_numpy(tar_flat), torch.from_numpy(tar_labels)
    )

    batch_size = args.batch_size
    src_loader = torch.utils.data.DataLoader(
        src_dset, batch_size=batch_size, shuffle=True
    )
    tar_loader = torch.utils.data.DataLoader(
        tar_dset, batch_size=batch_size, shuffle=True
    )
    # 评估用 loader（不 shuffle，不 drop_last）
    eval_loader = torch.utils.data.DataLoader(
        tar_dset, batch_size=batch_size, shuffle=False
    )

    print(f"源域 (训练): {len(src_dset)} 样本, 目标域 (验证): {len(tar_dset)} 样本")

    # ---- 构建模型 ----
    model = build_model_with_kwargs(args, device)
    feat_dim = getattr(args, "feat_dim", 32)
    proj_dropout = getattr(args, "dropout", 0.05)
    u_net = DomainProjection(feat_dim, proj_dropout).to(device)
    v_net = DomainProjection(feat_dim, proj_dropout).to(device)

    # ---- 三个独立优化器（原论文设计） ----
    lr = args.lr
    wd = args.weight_decay
    optimizer = get_optimizer(args.optimizer_name, model.parameters(), lr=lr, weight_decay=wd)
    optimizer_u = get_optimizer(args.optimizer_name, u_net.parameters(), lr=lr, weight_decay=wd)
    optimizer_v = get_optimizer(args.optimizer_name, v_net.parameters(), lr=lr, weight_decay=wd)

    # 学习率调度
    steps_per_epoch = min(len(src_loader), len(tar_loader))

    def _make_onecycle(opt, steps):
        try:
            return torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=lr * 5, steps_per_epoch=steps,
                epochs=args.epochs, anneal_strategy="cos", pct_start=0.1,
            )
        except Exception:
            return None

    scheduler = scheduler_u = scheduler_v = None
    if args.lr_policy == "onecycle":
        scheduler = _make_onecycle(optimizer, steps_per_epoch)
        scheduler_u = _make_onecycle(optimizer_u, steps_per_epoch)
        scheduler_v = _make_onecycle(optimizer_v, steps_per_epoch)

    # ---- Observer ----
    observer = create_observer(args, device, args.save_dir)

    # ---- 损失函数 ----
    criterion = nn.CrossEntropyLoss(
        label_smoothing=getattr(args, "label_smoothing", 0.0)
    ).to(device)

    # MLDA 域适应参数
    loss_weight = getattr(args, "mlda_loss_weight", 0.5)
    lambda_center = getattr(args, "mlda_lambda_center", 100)

    # ---- 训练循环 ----
    for epoch in range(1, args.epochs + 1):
        # === 训练阶段 ===
        model.train()
        u_net.train()
        v_net.train()
        observer.reset()

        # sigmoid 调度: λ 从 ~1 (偏重域间) 衰减至 ~0 (偏重域内)
        lam = 1.0 / (1.0 + np.exp(np.clip(epoch - lambda_center, -500, 500)))

        for (src_batch, tar_batch) in zip(src_loader, tar_loader):
            src_data = src_batch[0].to(device)
            src_label = src_batch[1].to(device)
            tar_data = tar_batch[0].to(device)

            # batch size 不一致时截断到较小的一方
            min_bs = min(src_data.size(0), tar_data.size(0))
            src_data = src_data[:min_bs]
            tar_data = tar_data[:min_bs]
            src_label = src_label[:min_bs]

            # 前向传播 (源域 + 目标域)
            src_feat, tar_feat, src_cls, tar_cls = model(src_data, tar_data)

            # 1) 分类损失 (仅源域监督)
            cls_loss = criterion(src_cls, src_label)

            # 2) 域间损失: Wasserstein 距离 (梯度仅通过 U/V 投影)
            src_proj = u_net(src_feat)
            tar_proj = v_net(tar_feat)
            inter_loss = compute_wasserstein_distance(src_proj, tar_proj)

            # 3) 域内损失: ICD 对比损失 (目标域使用伪标签)
            pseudo_labels = torch.argmax(tar_cls, dim=1)
            intra_loss = idcd_loss(src_feat, tar_feat, src_label, pseudo_labels)

            # 总损失: L_CE + 2 * [(1-λ)*L_inter + λ*L_intra]
            da_loss = (1.0 - lam) * inter_loss + lam * intra_loss
            total_loss = cls_loss + 2.0 * da_loss

            # 反向传播 (三个优化器)
            optimizer.zero_grad()
            optimizer_u.zero_grad()
            optimizer_v.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer_u.step()
            optimizer_v.step()

            # OneCycleLR 按 batch 步进
            if scheduler is not None:
                scheduler.step()
            if scheduler_u is not None:
                scheduler_u.step()
            if scheduler_v is not None:
                scheduler_v.step()

            # 记录训练指标（使用源域分类结果）
            prob = torch.softmax(src_cls, dim=1)
            _, preds = torch.max(prob, dim=1)
            observer.train_update(cls_loss, prob, preds, src_label)

        # === 验证阶段 (在目标域上评估) ===
        model.eval()
        with torch.no_grad():
            for batch_data, batch_label in eval_loader:
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)

                _, outputs = model(batch_data, None)  # 推理模式
                loss = criterion(outputs, batch_label)

                prob = torch.softmax(outputs, dim=1)
                _, preds = torch.max(prob, dim=1)
                observer.eval_update(loss, prob, preds, batch_label)

        # === 计算指标、打印、早停、保存最佳模型 ===
        should_stop = observer.execute(
            epoch, args.epochs,
            len(src_loader.dataset), len(eval_loader.dataset),
            fold=fold_idx, model=model,
        )

        # 记录该轮 train + val 指标到 history.csv
        if recorder is not None:
            is_best = (observer.best_dicts.get("epoch", 0) == epoch)
            recorder.record_epoch(observer, fold_idx, epoch, is_best=is_best)

        # 非 OneCycleLR 的调度器在 epoch 级别步进
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        if scheduler_u is not None and not isinstance(scheduler_u, torch.optim.lr_scheduler.OneCycleLR):
            scheduler_u.step()
        if scheduler_v is not None and not isinstance(scheduler_v, torch.optim.lr_scheduler.OneCycleLR):
            scheduler_v.step()

        if should_stop:
            print("⚠️  早停触发")
            break

    # ---- 记录该 fold 最佳结果 ----
    observer.finish(fold_idx)
    best_val_acc = float(observer.best_dicts["Accuracy"])
    if recorder is not None:
        recorder.record_best(observer, fold_idx)
    print(f"Fold {fold_idx} 最佳验证: Acc={best_val_acc:.4f}")
    return best_val_acc


# ========================================================================== #
#  DAEEGViT 域适应训练/验证（使用 RuntimeObserver）                           #
# ========================================================================== #

def run_daeevit_fold(args, device, fold_idx, fold_config, recorder=None):
    """训练一个 fold 的 DAEEGViT 域适应模型

    训练范式:
        - 源域 = 训练集受试者（带标签，用于分类损失）
        - 目标域 = 验证集受试者（无标签，仅特征参与 MMD 损失）
        - 评估在目标域上进行

    损失函数 (论文 Eq.8):
        L = L_cls + L_mmd
        - L_cls: 源域交叉熵 (对 logits)
        - L_mmd: CLS token 特征上的 MMD (源域 vs 目标域)
    """
    from utils.mlda_loss import mmd_loss

    print(f"\n{'='*60}")
    print(f"🔥 Fold {fold_idx} / {args.k_fold} (DAEEGViT 域适应)")
    print(f"{'='*60}")

    # ---- 构建源域 (train) 和目标域 (val) 数据集 ----
    source_dataset = build_fold_data(args, fold_config, mode="train")
    target_dataset = build_fold_data(args, fold_config, mode="val")

    # DAEEGViT 需要 (B, C, W) 格式: (B, window_size, channels) → (B, channels, window_size)
    use_adf = getattr(args, "use_adf", True)
    num_channels = 3 if use_adf else 1

    src_windows = np.stack(source_dataset.windows, axis=0)  # (N, W, C)
    src_bcw = src_windows.transpose(0, 2, 1).astype(np.float32)  # (N, C, W)
    src_labels = np.array(source_dataset.labels, dtype=np.int64)

    tar_windows = np.stack(target_dataset.windows, axis=0)
    tar_bcw = tar_windows.transpose(0, 2, 1).astype(np.float32)  # (N, C, W)
    tar_labels = np.array(target_dataset.labels, dtype=np.int64)

    # 构建 TensorDataset + DataLoader
    src_dset = torch.utils.data.TensorDataset(
        torch.from_numpy(src_bcw), torch.from_numpy(src_labels)
    )
    tar_dset = torch.utils.data.TensorDataset(
        torch.from_numpy(tar_bcw), torch.from_numpy(tar_labels)
    )

    batch_size = args.batch_size
    src_loader = torch.utils.data.DataLoader(src_dset, batch_size=batch_size, shuffle=True)
    tar_loader = torch.utils.data.DataLoader(tar_dset, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(tar_dset, batch_size=batch_size, shuffle=False)

    print(f"源域 (训练): {len(src_dset)} 样本, 目标域 (验证): {len(tar_dset)} 样本")

    # ---- 构建模型 ----
    model = build_model_with_kwargs(args, device)
    optimizer = get_optimizer(args.optimizer_name, model.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度
    steps_per_epoch = min(len(src_loader), len(tar_loader))
    scheduler = None
    if args.lr_policy == "onecycle":
        try:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr * 5, steps_per_epoch=steps_per_epoch,
                epochs=args.epochs, anneal_strategy="cos", pct_start=0.1,
            )
        except Exception:
            scheduler = None

    # ---- Observer ----
    observer = create_observer(args, device, args.save_dir)

    # ---- 损失函数 ----
    criterion = nn.CrossEntropyLoss(
        label_smoothing=getattr(args, "label_smoothing", 0.0)
    ).to(device)
    mmd_weight = getattr(args, "mmd_weight", 1.0)

    # ---- 训练循环 ----
    for epoch in range(1, args.epochs + 1):
        # === 训练阶段 ===
        model.train()
        observer.reset()

        for (src_batch, tar_batch) in zip(src_loader, tar_loader):
            src_data = src_batch[0].to(device)    # (B, C, W)
            src_label = src_batch[1].to(device)
            tar_data = tar_batch[0].to(device)    # (B, C, W)

            # batch size 不一致时截断到较小的一方
            min_bs = min(src_data.size(0), tar_data.size(0))
            src_data = src_data[:min_bs]
            tar_data = tar_data[:min_bs]
            src_label = src_label[:min_bs]

            # 前向传播
            src_logits, src_cls_feat = model(src_data)    # logits + CLS 特征
            tar_logits, tar_cls_feat = model(tar_data)

            # 1) 分类损失 (源域)
            cls_loss = criterion(src_logits, src_label)

            # 2) MMD 损失 (CLS token 特征对齐)
            mmd = mmd_loss(src_cls_feat, tar_cls_feat)

            # 总损失: L = L_cls + weight * L_mmd (论文 Eq.8)
            total_loss = cls_loss + mmd_weight * mmd

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            # 记录训练指标
            prob = torch.softmax(src_logits, dim=1)
            _, preds = torch.max(prob, dim=1)
            observer.train_update(cls_loss, prob, preds, src_label)

        # === 验证阶段 (目标域评估) ===
        model.eval()
        with torch.no_grad():
            for batch_data, batch_label in eval_loader:
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)

                logits, _ = model(batch_data)
                loss = criterion(logits, batch_label)

                prob = torch.softmax(logits, dim=1)
                _, preds = torch.max(prob, dim=1)
                observer.eval_update(loss, prob, preds, batch_label)

        # === 计算指标、打印、早停、保存最佳模型 ===
        should_stop = observer.execute(
            epoch, args.epochs,
            len(src_loader.dataset), len(eval_loader.dataset),
            fold=fold_idx, model=model,
        )

        if recorder is not None:
            is_best = (observer.best_dicts.get("epoch", 0) == epoch)
            recorder.record_epoch(observer, fold_idx, epoch, is_best=is_best)

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        if should_stop:
            print("⚠️  早停触发")
            break

    # ---- 记录该 fold 最佳结果 ----
    observer.finish(fold_idx)
    best_val_acc = float(observer.best_dicts["Accuracy"])
    if recorder is not None:
        recorder.record_best(observer, fold_idx)
    print(f"Fold {fold_idx} 最佳验证: Acc={best_val_acc:.4f}")
    return best_val_acc


# ========================================================================== #
#  LA-MSDA 多源域适应训练/验证（使用 RuntimeObserver）                        #
# ========================================================================== #

def run_lamsda_fold(args, device, fold_idx, fold_config, recorder=None):
    """训练一个 fold 的 LA-MSDA 多源域适应模型

    训练范式:
        - 每个训练受试者 = 一个源域 (带标签)
        - 验证受试者 = 目标域 (训练时标签不可见)
        - 每个源域有独立的 DSCNN + 分类器分支
        - 推理时集成所有分支的 softmax 平均

    损失函数:
        L = L_cls + μ·L_llmmd + γ·L_global
        - L_cls: 当前源域分支的交叉熵
        - L_llmmd: 标签条件 MMD (共享特征上)
        - L_global: 所有分支在目标域上的共识损失
        - μ, γ: sigmoid 预热 (从 0 渐增到 1)
    """
    from utils.mlda_loss import llmmd_loss, global_consensus_loss

    print(f"\n{'='*60}")
    print(f"🔥 Fold {fold_idx} / {args.k_fold} (LA-MSDA 多源域适应)")
    print(f"{'='*60}")

    # ---- 构建源域 (按受试者分组) 和目标域 ----
    source_dataset = build_fold_data(args, fold_config, mode="train")
    target_dataset = build_fold_data(args, fold_config, mode="val")

    use_adf = getattr(args, "use_adf", True)
    num_channels = 3 if use_adf else 1

    # 按受试者分组源域数据
    subject_to_indices = {}
    for i, sid in enumerate(source_dataset.subject_ids):
        subject_to_indices.setdefault(sid, []).append(i)

    # 构建每个源域的数据 → (B, C, W) 格式
    source_loaders = []
    subject_ids_list = sorted(subject_to_indices.keys())
    for sid in subject_ids_list:
        indices = subject_to_indices[sid]
        windows = np.stack([source_dataset.windows[i] for i in indices], axis=0)
        bcw = windows.transpose(0, 2, 1).astype(np.float32)  # (N, C, W)
        labels = np.array([source_dataset.labels[i] for i in indices], dtype=np.int64)
        dset = torch.utils.data.TensorDataset(
            torch.from_numpy(bcw), torch.from_numpy(labels)
        )
        loader = torch.utils.data.DataLoader(
            dset, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        source_loaders.append(loader)

    # 目标域数据
    tar_windows = np.stack(target_dataset.windows, axis=0)
    tar_bcw = tar_windows.transpose(0, 2, 1).astype(np.float32)
    tar_labels = np.array(target_dataset.labels, dtype=np.int64)
    tar_dset = torch.utils.data.TensorDataset(
        torch.from_numpy(tar_bcw), torch.from_numpy(tar_labels)
    )
    tar_loader = torch.utils.data.DataLoader(
        tar_dset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    eval_loader = torch.utils.data.DataLoader(
        tar_dset, batch_size=args.batch_size, shuffle=False
    )

    # 源域数量受配置上限控制
    num_sources = min(len(source_loaders), getattr(args, "num_sources", 5))
    source_loaders = source_loaders[:num_sources]

    print(f"源域: {num_sources} 个受试者分支, 目标域: {len(tar_dset)} 样本")

    # ---- 构建模型 ----
    # 覆盖 num_sources 参数
    args._lamsda_num_sources = num_sources
    orig_num_sources = getattr(args, "num_sources", 5)
    args.num_sources = num_sources
    model = build_model_with_kwargs(args, device)
    args.num_sources = orig_num_sources

    optimizer = get_optimizer(args.optimizer_name, model.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)

    # ---- Observer ----
    observer = create_observer(args, device, args.save_dir)

    # sigmoid 预热参数
    max_epoch = args.epochs
    warmup_scale = getattr(args, "da_warmup_scale", 10.0)

    # ---- 训练循环 ----
    total_iterations = 0
    for epoch in range(1, max_epoch + 1):
        model.train()
        observer.reset()

        # sigmoid 调度: 从 0 渐增到 1
        progress = epoch / max_epoch
        gamma = 2.0 / (1.0 + np.exp(-warmup_scale * progress)) - 1.0
        mu = gamma

        # 为每个源域创建无限迭代器
        src_iters = [iter(loader) for loader in source_loaders]

        # 目标域迭代器 (循环使用)
        tar_iter = iter(tar_loader)

        for mark in range(num_sources):
            # 获取源域 batch (自动循环)
            try:
                src_data, src_label = next(src_iters[mark])
            except StopIteration:
                src_iters[mark] = iter(source_loaders[mark])
                src_data, src_label = next(src_iters[mark])

            # 获取目标域 batch
            try:
                tar_data, _ = next(tar_iter)
            except StopIteration:
                tar_iter = iter(tar_loader)
                tar_data, _ = next(tar_iter)

            src_data = src_data.to(device)
            src_label = src_label.to(device)
            tar_data = tar_data.to(device)

            # 前向传播: 当前源域分支
            src_logits, src_feat = model(src_data, mark)

            # 1) 分类损失 (当前源域分支)
            cls_loss = criterion(src_logits, src_label)

            # 2) LLMMD 损失 (共享特征上)
            with torch.no_grad():
                src_shared = model.shared_features(src_data)
                tar_shared = model.shared_features(tar_data)
                tar_logits_mark, _ = model(tar_data, mark)
                tar_probs = F.softmax(tar_logits_mark, dim=1)

            llmmd = llmmd_loss(src_shared, tar_shared, src_label, tar_probs,
                               num_classes=args.num_classes)

            # 3) 全局共识损失 (所有分支在目标域上的预测一致性)
            if num_sources > 1:
                target_probs_list = []
                for k in range(num_sources):
                    with torch.no_grad():
                        t_logits, _ = model(tar_data, k)
                    target_probs_list.append(F.softmax(t_logits, dim=1))
                # 让当前分支有梯度
                t_logits_mark, _ = model(tar_data, mark)
                target_probs_list[mark] = F.softmax(t_logits_mark, dim=1)
                glo_loss = global_consensus_loss(target_probs_list)
            else:
                glo_loss = torch.tensor(0.0, device=device)

            # 总损失
            total_loss = cls_loss + mu * llmmd + gamma * glo_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_iterations += 1

            # 记录训练指标 (当前源域分支)
            prob = torch.softmax(src_logits, dim=1)
            _, preds = torch.max(prob, dim=1)
            observer.train_update(cls_loss, prob, preds, src_label)

        # === 验证阶段 (集成所有分支) ===
        model.eval()
        with torch.no_grad():
            for batch_data, batch_label in eval_loader:
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)

                avg_probs = model.ensemble_predict(batch_data)
                _, preds = torch.max(avg_probs, dim=1)

                # 使用 CE loss 作为 observer 记录值
                loss = criterion(torch.log(avg_probs + 1e-8), batch_label)
                observer.eval_update(loss, avg_probs, preds, batch_label)

        # === 计算指标、打印、早停、保存最佳模型 ===
        should_stop = observer.execute(
            epoch, max_epoch,
            sum(len(l.dataset) for l in source_loaders),
            len(eval_loader.dataset),
            fold=fold_idx, model=model,
        )

        if recorder is not None:
            is_best = (observer.best_dicts.get("epoch", 0) == epoch)
            recorder.record_epoch(observer, fold_idx, epoch, is_best=is_best)

        if should_stop:
            print("⚠️  早停触发")
            break

    # ---- 记录该 fold 最佳结果 ----
    observer.finish(fold_idx)
    best_val_acc = float(observer.best_dicts["Accuracy"])
    if recorder is not None:
        recorder.record_best(observer, fold_idx)
    print(f"Fold {fold_idx} 最佳验证: Acc={best_val_acc:.4f}")
    return best_val_acc


# ========================================================================== #
#  DANN 域对抗训练/验证（使用 RuntimeObserver）                                #
# ========================================================================== #

def run_dann_fold(args, device, fold_idx, fold_config, recorder=None):
    """训练一个 fold 的 DANN 域对抗模型

    训练范式:
        - 源域 = 训练集受试者（带标签，用于分类损失）
        - 目标域 = 验证集受试者（无标签，仅特征参与域对抗损失）
        - 评估在目标域上进行

    损失函数 (论文 Eq.3):
        L = L_y + λ * L_d
        - L_y: 源域交叉熵 (对 logits)
        - L_d: 域二分类 BCE (通过 GRL 对抗训练)
        - λ: 对抗权重，sigmoid 调度从 0 渐增到 1

    GRL 调度 (论文 Eq.9):
        p = current_step / total_steps
        λ = 2/(1 + exp(-10*p)) - 1
    """
    print(f"\n{'='*60}")
    print(f"🔥 Fold {fold_idx} / {args.k_fold} (DANN 域对抗)")
    print(f"{'='*60}")

    # ---- 构建源域 (train) 和目标域 (val) 数据集 ----
    source_dataset = build_fold_data(args, fold_config, mode="train")
    target_dataset = build_fold_data(args, fold_config, mode="val")

    # DANN 需要展平向量输入: (B, W, C) -> (B, W*C)
    src_windows = np.stack(source_dataset.windows, axis=0)
    src_flat = src_windows.reshape(src_windows.shape[0], -1).astype(np.float32)
    src_labels = np.array(source_dataset.labels, dtype=np.int64)

    tar_windows = np.stack(target_dataset.windows, axis=0)
    tar_flat = tar_windows.reshape(tar_windows.shape[0], -1).astype(np.float32)
    tar_labels = np.array(target_dataset.labels, dtype=np.int64)

    # 构建 TensorDataset + DataLoader
    src_dset = torch.utils.data.TensorDataset(
        torch.from_numpy(src_flat), torch.from_numpy(src_labels)
    )
    tar_dset = torch.utils.data.TensorDataset(
        torch.from_numpy(tar_flat), torch.from_numpy(tar_labels)
    )

    batch_size = args.batch_size
    src_loader = torch.utils.data.DataLoader(src_dset, batch_size=batch_size, shuffle=True)
    tar_loader = torch.utils.data.DataLoader(tar_dset, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(tar_dset, batch_size=batch_size, shuffle=False)

    print(f"源域 (训练): {len(src_dset)} 样本, 目标域 (验证): {len(tar_dset)} 样本")

    # ---- 构建模型 ----
    model = build_model_with_kwargs(args, device)
    optimizer = get_optimizer(args.optimizer_name, model.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度
    steps_per_epoch = min(len(src_loader), len(tar_loader))
    total_steps = args.epochs * steps_per_epoch

    scheduler = None
    if args.lr_policy == "onecycle":
        try:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr * 5, steps_per_epoch=steps_per_epoch,
                epochs=args.epochs, anneal_strategy="cos", pct_start=0.1,
            )
        except Exception:
            scheduler = None

    # ---- Observer ----
    observer = create_observer(args, device, args.save_dir)

    # ---- 损失函数 ----
    criterion = nn.CrossEntropyLoss(
        label_smoothing=getattr(args, "label_smoothing", 0.0)
    ).to(device)
    domain_criterion = nn.BCEWithLogitsLoss().to(device)

    # DANN 对抗权重调度参数
    dann_gamma = getattr(args, "dann_gamma", 10.0)

    # ---- 训练循环 ----
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        # === 训练阶段 ===
        model.train()
        observer.reset()

        for (src_batch, tar_batch) in zip(src_loader, tar_loader):
            src_data = src_batch[0].to(device)
            src_label = src_batch[1].to(device)
            tar_data = tar_batch[0].to(device)

            # batch size 不一致时截断到较小的一方
            min_bs = min(src_data.size(0), tar_data.size(0))
            src_data = src_data[:min_bs]
            tar_data = tar_data[:min_bs]
            src_label = src_label[:min_bs]

            # DANN GRL 调度 (论文 Eq.9): p ∈ [0, 1], λ = 2/(1+exp(-γp)) - 1
            p = global_step / max(total_steps, 1)
            lam = 2.0 / (1.0 + np.exp(-dann_gamma * p)) - 1.0
            model.grl.set_alpha(lam)

            # 前向传播 (源域 + 目标域)
            src_feat, src_logits, domain_logits = model(src_data, tar_data)

            # 1) 分类损失 (源域监督)
            cls_loss = criterion(src_logits, src_label)

            # 2) 域对抗损失 (源=0, 目标=1)
            src_size = src_data.size(0)
            tar_size = tar_data.size(0)
            domain_labels = torch.cat([
                torch.zeros(src_size, 1, device=device),  # 源域标签=0
                torch.ones(tar_size, 1, device=device),   # 目标域标签=1
            ], dim=0)
            domain_loss = domain_criterion(domain_logits, domain_labels)

            # 总损失: L = L_y + λ * L_d (论文 Eq.3)
            total_loss = cls_loss + lam * domain_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            global_step += 1

            # 记录训练指标 (使用源域分类结果)
            prob = torch.softmax(src_logits, dim=1)
            _, preds = torch.max(prob, dim=1)
            observer.train_update(cls_loss, prob, preds, src_label)

        # === 验证阶段 (在目标域上评估) ===
        model.eval()
        with torch.no_grad():
            for batch_data, batch_label in eval_loader:
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)

                _, outputs = model(batch_data, None)  # 推理模式
                loss = criterion(outputs, batch_label)

                prob = torch.softmax(outputs, dim=1)
                _, preds = torch.max(prob, dim=1)
                observer.eval_update(loss, prob, preds, batch_label)

        # === 计算指标、打印、早停、保存最佳模型 ===
        should_stop = observer.execute(
            epoch, args.epochs,
            len(src_loader.dataset), len(eval_loader.dataset),
            fold=fold_idx, model=model,
        )

        if recorder is not None:
            is_best = (observer.best_dicts.get("epoch", 0) == epoch)
            recorder.record_epoch(observer, fold_idx, epoch, is_best=is_best)

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        if should_stop:
            print("⚠️  早停触发")
            break

    # ---- 记录该 fold 最佳结果 ----
    observer.finish(fold_idx)
    best_val_acc = float(observer.best_dicts["Accuracy"])
    if recorder is not None:
        recorder.record_best(observer, fold_idx)
    print(f"Fold {fold_idx} 最佳验证: Acc={best_val_acc:.4f}")
    return best_val_acc


# ========================================================================== #
#  DeepCORAL 域适应训练/验证（使用 RuntimeObserver）                          #
# ========================================================================== #

def run_deepcoral_fold(args, device, fold_idx, fold_config, recorder=None):
    """训练一个 fold 的 DeepCORAL 域适应模型

    训练范式:
        - 源域 = 训练集受试者（带标签，用于分类损失）
        - 目标域 = 验证集受试者（无标签，仅特征参与 CORAL 损失）
        - 评估在目标域上进行

    损失函数 (论文 Eq.2):
        L = L_cls + λ * L_CORAL
        - L_cls: 源域交叉熵 (对 logits)
        - L_CORAL: 协方差对齐损失 (源域 vs 目标域特征)
        - λ: CORAL 损失权重 (可配置)
    """
    from models.deepcoral_model import coral_loss

    print(f"\n{'='*60}")
    print(f"🔥 Fold {fold_idx} / {args.k_fold} (DeepCORAL 域适应)")
    print(f"{'='*60}")

    # ---- 构建源域 (train) 和目标域 (val) 数据集 ----
    source_dataset = build_fold_data(args, fold_config, mode="train")
    target_dataset = build_fold_data(args, fold_config, mode="val")

    # DeepCORAL 需要展平向量输入: (B, W, C) -> (B, W*C)
    src_windows = np.stack(source_dataset.windows, axis=0)
    src_flat = src_windows.reshape(src_windows.shape[0], -1).astype(np.float32)
    src_labels = np.array(source_dataset.labels, dtype=np.int64)

    tar_windows = np.stack(target_dataset.windows, axis=0)
    tar_flat = tar_windows.reshape(tar_windows.shape[0], -1).astype(np.float32)
    tar_labels = np.array(target_dataset.labels, dtype=np.int64)

    # 构建 TensorDataset + DataLoader
    src_dset = torch.utils.data.TensorDataset(
        torch.from_numpy(src_flat), torch.from_numpy(src_labels)
    )
    tar_dset = torch.utils.data.TensorDataset(
        torch.from_numpy(tar_flat), torch.from_numpy(tar_labels)
    )

    batch_size = args.batch_size
    src_loader = torch.utils.data.DataLoader(src_dset, batch_size=batch_size, shuffle=True)
    tar_loader = torch.utils.data.DataLoader(tar_dset, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(tar_dset, batch_size=batch_size, shuffle=False)

    print(f"源域 (训练): {len(src_dset)} 样本, 目标域 (验证): {len(tar_dset)} 样本")

    # ---- 构建模型 ----
    model = build_model_with_kwargs(args, device)
    optimizer = get_optimizer(args.optimizer_name, model.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度
    steps_per_epoch = min(len(src_loader), len(tar_loader))
    scheduler = None
    if args.lr_policy == "onecycle":
        try:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr * 5, steps_per_epoch=steps_per_epoch,
                epochs=args.epochs, anneal_strategy="cos", pct_start=0.1,
            )
        except Exception:
            scheduler = None

    # ---- Observer ----
    observer = create_observer(args, device, args.save_dir)

    # ---- 损失函数 ----
    criterion = nn.CrossEntropyLoss(
        label_smoothing=getattr(args, "label_smoothing", 0.0)
    ).to(device)
    coral_weight = getattr(args, "coral_weight", 1.0)

    # ---- 训练循环 ----
    for epoch in range(1, args.epochs + 1):
        # === 训练阶段 ===
        model.train()
        observer.reset()

        for (src_batch, tar_batch) in zip(src_loader, tar_loader):
            src_data = src_batch[0].to(device)
            src_label = src_batch[1].to(device)
            tar_data = tar_batch[0].to(device)

            # batch size 不一致时截断到较小的一方
            min_bs = min(src_data.size(0), tar_data.size(0))
            src_data = src_data[:min_bs]
            tar_data = tar_data[:min_bs]
            src_label = src_label[:min_bs]

            # 前向传播 (源域 + 目标域)
            src_feat, tar_feat, src_logits = model(src_data, tar_data)

            # 1) 分类损失 (源域监督)
            cls_loss = criterion(src_logits, src_label)

            # 2) CORAL 损失 (协方差对齐)
            coral = coral_loss(src_feat, tar_feat)

            # 总损失: L = L_cls + λ * L_CORAL
            total_loss = cls_loss + coral_weight * coral

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            # 记录训练指标 (使用源域分类结果)
            prob = torch.softmax(src_logits, dim=1)
            _, preds = torch.max(prob, dim=1)
            observer.train_update(cls_loss, prob, preds, src_label)

        # === 验证阶段 (在目标域上评估) ===
        model.eval()
        with torch.no_grad():
            for batch_data, batch_label in eval_loader:
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)

                _, outputs = model(batch_data, None)  # 推理模式
                loss = criterion(outputs, batch_label)

                prob = torch.softmax(outputs, dim=1)
                _, preds = torch.max(prob, dim=1)
                observer.eval_update(loss, prob, preds, batch_label)

        # === 计算指标、打印、早停、保存最佳模型 ===
        should_stop = observer.execute(
            epoch, args.epochs,
            len(src_loader.dataset), len(eval_loader.dataset),
            fold=fold_idx, model=model,
        )

        if recorder is not None:
            is_best = (observer.best_dicts.get("epoch", 0) == epoch)
            recorder.record_epoch(observer, fold_idx, epoch, is_best=is_best)

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        if should_stop:
            print("⚠️  早停触发")
            break

    # ---- 记录该 fold 最佳结果 ----
    observer.finish(fold_idx)
    best_val_acc = float(observer.best_dicts["Accuracy"])
    if recorder is not None:
        recorder.record_best(observer, fold_idx)
    print(f"Fold {fold_idx} 最佳验证: Acc={best_val_acc:.4f}")
    return best_val_acc


# ========================================================================== #
#  InterpretableCNN 域泛化训练/验证（使用 RuntimeObserver）                     #
# ========================================================================== #

def run_interpcnn_fold(args, device, fold_idx, fold_config, recorder=None):
    """训练一个 fold 的 InterpretableCNN 域泛化模型

    训练范式 (Domain Generalization, 区别于 DA):
        - 源域 = 训练集受试者（带标签，用于分类损失）
        - 目标域 = 验证集受试者（完全不参与训练，仅用于评估）
        - 训练过程中目标域数据不可见（连无标签特征也不使用）
        - 这与 DANN/DeepCORAL/MLDA 等 DA 方法有本质区别

    损失函数:
        L = NLLLoss (与模型 LogSoftmax 输出配对)

    输入格式:
        (B, C, W) — 与 DAEEGViT/LA-MSDA 相同
        从数据集的 (B, W, C) 转置得到

    参考论文:
        Cui et al. "EEG-Based Cross-Subject Driver Drowsiness Recognition
        With an Interpretable Convolutional Neural Network." IEEE TNNLS, 2022.
    """
    print(f"\n{'='*60}")
    print(f"🔥 Fold {fold_idx} / {args.k_fold} (InterpretableCNN 域泛化)")
    print(f"{'='*60}")

    # ---- 构建训练集和验证集 (DG: 目标域完全不参与训练) ----
    train_dataset = build_fold_data(args, fold_config, mode="train")
    val_dataset = build_fold_data(args, fold_config, mode="val")

    # InterpretableCNN 需要 (B, C, W) 格式: (B, W, C) → (B, C, W)
    use_adf = getattr(args, "use_adf", True)

    train_windows = np.stack(train_dataset.windows, axis=0)  # (N, W, C)
    train_bcw = train_windows.transpose(0, 2, 1).astype(np.float32)  # (N, C, W)
    train_labels = np.array(train_dataset.labels, dtype=np.int64)

    val_windows = np.stack(val_dataset.windows, axis=0)
    val_bcw = val_windows.transpose(0, 2, 1).astype(np.float32)
    val_labels = np.array(val_dataset.labels, dtype=np.int64)

    # 构建 TensorDataset + DataLoader
    train_dset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_bcw), torch.from_numpy(train_labels)
    )
    val_dset = torch.utils.data.TensorDataset(
        torch.from_numpy(val_bcw), torch.from_numpy(val_labels)
    )

    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dset, batch_size=batch_size, shuffle=False
    )

    print(f"训练集: {len(train_dset)} 样本, 验证集: {len(val_dset)} 样本")

    # ---- 构建模型 ----
    model = build_model_with_kwargs(args, device)
    optimizer = get_optimizer(args.optimizer_name, model.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度
    scheduler = None
    if args.lr_policy == "onecycle":
        try:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr * 5, steps_per_epoch=len(train_loader),
                epochs=args.epochs, anneal_strategy="cos", pct_start=0.1,
            )
        except Exception:
            scheduler = None

    # ---- Observer ----
    observer = create_observer(args, device, args.save_dir)

    # ---- 损失函数: NLLLoss (与 LogSoftmax 输出配对) ----
    criterion = nn.NLLLoss().to(device)

    # ---- 训练循环 ----
    for epoch in range(1, args.epochs + 1):
        # === 训练阶段 ===
        model.train()
        observer.reset()

        for batch_data, batch_label in train_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            optimizer.zero_grad()
            log_probs = model(batch_data)
            loss = criterion(log_probs, batch_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            # 记录训练指标: log_probs → probs
            prob = torch.exp(log_probs)
            _, preds = torch.max(prob, dim=1)
            observer.train_update(loss, prob, preds, batch_label)

        # === 验证阶段 ===
        model.eval()
        with torch.no_grad():
            for batch_data, batch_label in val_loader:
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)

                log_probs = model(batch_data)
                loss = criterion(log_probs, batch_label)

                prob = torch.exp(log_probs)
                _, preds = torch.max(prob, dim=1)
                observer.eval_update(loss, prob, preds, batch_label)

        # === 计算指标、打印、早停、保存最佳模型 ===
        should_stop = observer.execute(
            epoch, args.epochs,
            len(train_loader.dataset), len(val_loader.dataset),
            fold=fold_idx, model=model,
        )

        if recorder is not None:
            is_best = (observer.best_dicts.get("epoch", 0) == epoch)
            recorder.record_epoch(observer, fold_idx, epoch, is_best=is_best)

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        if should_stop:
            print("⚠️  早停触发")
            break

    # ---- 记录该 fold 最佳结果 ----
    observer.finish(fold_idx)
    best_val_acc = float(observer.best_dicts["Accuracy"])
    if recorder is not None:
        recorder.record_best(observer, fold_idx)
    print(f"Fold {fold_idx} 最佳验证: Acc={best_val_acc:.4f}")
    return best_val_acc


def run_afmcir_fold(args, device, fold_idx, fold_config, recorder=None):
    """训练一个 fold 的 AFM-CIR 因果域泛化模型

    训练范式 (Domain Generalization, 区别于 DA):
        - 源域 = 训练集受试者 (带标签, 用于分类损失 + AFM 增强)
        - 目标域 = 验证集受试者 (完全不参与训练, 仅用于评估)
        - 训练过程中目标域数据不可见

    三阶段训练流程:
        Phase 1: 预训练引导编码器 (重建 + 域对抗 + RNC 对比)
        Phase 2: 使用冻结引导编码器计算 AFM 增强样本
        Phase 3: 因果启发训练 (分类损失 + FAC 损失 + 对抗掩码)

    损失函数:
        L = L_sup(CE) + L_aug(CE) + tau * L_FAC + adv_weight * L_inf

    输入格式: (B, C, W) — 与 InterpretableCNN/DAEEGViT 相同

    参考论文:
        Zhu et al. "Causality-Preserving Domain Generalization via
        Adaptive Fourier Mixup for RUL Prediction." IEEE TPAMI, 2026.
    """
    from models.afmcir_model import (
        GuidanceEncoder, AFMAugmentation, fac_loss, rnc_loss_binary,
    )

    print(f"\n{'='*60}")
    print(f"🔬 Fold {fold_idx} / {args.k_fold} (AFM-CIR 因果域泛化)")
    print(f"{'='*60}")

    # ---- 构建训练集和验证集 ----
    train_dataset = build_fold_data(args, fold_config, mode="train")
    val_dataset = build_fold_data(args, fold_config, mode="val")

    # 数据转换为 (B, C, W) 格式
    use_adf = getattr(args, "use_adf", True)

    train_windows = np.stack(train_dataset.windows, axis=0)   # (N, W, C)
    train_bcw = train_windows.transpose(0, 2, 1).astype(np.float32)  # (N, C, W)
    train_labels = np.array(train_dataset.labels, dtype=np.int64)

    val_windows = np.stack(val_dataset.windows, axis=0)
    val_bcw = val_windows.transpose(0, 2, 1).astype(np.float32)
    val_labels = np.array(val_dataset.labels, dtype=np.int64)

    # 构建域标签 (每个受试者一个域)
    unique_subjects = sorted(set(train_dataset.subject_ids))
    subject_to_domain = {s: i for i, s in enumerate(unique_subjects)}
    train_domain_ids = np.array(
        [subject_to_domain[s] for s in train_dataset.subject_ids],
        dtype=np.int64,
    )
    num_domains = len(unique_subjects)

    # TensorDatasets
    train_dset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_bcw),
        torch.from_numpy(train_labels),
        torch.from_numpy(train_domain_ids),
    )
    val_dset = torch.utils.data.TensorDataset(
        torch.from_numpy(val_bcw), torch.from_numpy(val_labels)
    )

    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dset, batch_size=batch_size, shuffle=False
    )

    print(f"训练集: {len(train_dset)} 样本 ({num_domains} 域), "
          f"验证集: {len(val_dset)} 样本")

    # ================================================================ #
    #  Phase 1: 预训练引导编码器                                         #
    # ================================================================ #
    guidance_epochs = getattr(args, "guidance_epochs", 50)
    guidance_lr = getattr(args, "guidance_lr", 1e-3)
    guidance_embed_dim = getattr(args, "guidance_embed_dim", 32)
    guidance_alpha_adv = getattr(args, "guidance_alpha_adv", 1.0)
    guidance_alpha_rnc = getattr(args, "guidance_alpha_rnc", 1.0)
    guidance_tau = getattr(args, "guidance_tau", 0.1)
    num_channels = 3 if use_adf else 1

    print(f"\n📐 Phase 1: 预训练引导编码器 ({guidance_epochs} epochs)")
    guidance_encoder = GuidanceEncoder(
        in_channels=num_channels,
        seq_len=args.window_size,
        embed_dim=guidance_embed_dim,
        num_domains=max(num_domains, 2),
        alpha_adv=guidance_alpha_adv,
        alpha_rnc=guidance_alpha_rnc,
    ).to(device)

    guidance_optimizer = torch.optim.AdamW(
        guidance_encoder.parameters(), lr=guidance_lr, weight_decay=1e-4,
    )

    for g_epoch in range(1, guidance_epochs + 1):
        guidance_encoder.train()
        total_recon = 0.0
        count = 0
        for batch_data, _, batch_domain in train_loader:
            batch_data = batch_data.to(device)
            batch_domain = batch_domain.to(device)

            guidance_optimizer.zero_grad()
            losses = guidance_encoder.pretrain_step(
                batch_data, batch_domain, tau=guidance_tau
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(guidance_encoder.parameters(), 1.0)
            guidance_optimizer.step()

            total_recon += losses["loss_recon"].item()
            count += 1

        if g_epoch % 10 == 0 or g_epoch == 1:
            avg_recon = total_recon / max(count, 1)
            print(f"  Guidance Epoch {g_epoch}/{guidance_epochs} — "
                  f"Recon Loss: {avg_recon:.4f}")

    # 冻结引导编码器
    guidance_encoder.eval()
    for p in guidance_encoder.parameters():
        p.requires_grad_(False)

    # 预计算所有训练样本的引导嵌入 (用于 AFM 的 z_pool)
    with torch.no_grad():
        all_x = torch.from_numpy(train_bcw).to(device)
        z_pool = guidance_encoder.encode(all_x)   # (N_train, embed_dim)

    print(f"✅ 引导编码器预训练完成, 嵌入池: {z_pool.shape}")

    # ================================================================ #
    #  Phase 2 + 3: AFM 增强 + 因果启发训练                              #
    # ================================================================ #
    print(f"\n🔥 Phase 2+3: AFM 增强 + 因果启发训练")

    # 构建主模型
    model = build_model_with_kwargs(args, device)
    optimizer = get_optimizer(args.optimizer_name, model.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay)

    # 对抗掩码器单独优化器 (min-max 博弈)
    adversary_params = list(model.masker.parameters())
    adversary_optimizer = torch.optim.AdamW(
        adversary_params, lr=args.lr, weight_decay=args.weight_decay,
    )

    # 学习率调度
    scheduler = None
    if args.lr_policy == "onecycle":
        try:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr * 5, steps_per_epoch=len(train_loader),
                epochs=args.epochs, anneal_strategy="cos", pct_start=0.1,
            )
        except Exception:
            scheduler = None

    # AFM 增强模块
    afm = AFMAugmentation(
        gamma_A=getattr(args, "afm_gamma_A", 0.5),
        gamma_P=getattr(args, "afm_gamma_P", 0.9),
        eta=getattr(args, "afm_eta", 0.8),
    )

    # 损失函数
    criterion = nn.NLLLoss().to(device)
    tau_fac = getattr(args, "cir_tau_fac", 2.0)
    adv_weight = getattr(args, "cir_adv_weight", 0.5)

    # ---- Observer ----
    observer = create_observer(args, device, args.save_dir)

    # ---- 训练循环 ----
    for epoch in range(1, args.epochs + 1):
        # === 训练阶段 ===
        model.train()
        observer.reset()

        for batch_data, batch_label, _ in train_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            B = batch_data.size(0)

            # ---- 前向传播 (原始 + 增强) ----
            log_probs, features, m_sup, m_inf = model(batch_data)

            # 主分类损失
            loss_sup = criterion(log_probs, batch_label)

            # AFM 增强 (Phase 2)
            with torch.no_grad():
                z_batch = guidance_encoder.encode(batch_data)
            x_aug = afm.augment(batch_data, z_batch, z_pool)

            # 增强样本分类
            log_probs_aug, features_aug, _, _ = model(x_aug)
            loss_aug = criterion(log_probs_aug, batch_label)

            # FAC 损失 (Phase 3: 关联因子化)
            loss_fac = fac_loss(features, features_aug)

            # ---- 对抗掩码 (因果充分性) ----
            inferior_feat = features.detach() * m_inf
            inf_logits = model.classifier(inferior_feat)
            loss_inf = F.nll_loss(F.log_softmax(inf_logits, dim=1), batch_label)

            # ---- 组合主损失 (最小化) ----
            main_loss = loss_sup + loss_aug + tau_fac * loss_fac - adv_weight * loss_inf

            optimizer.zero_grad()
            main_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # ---- 对抗器更新 (最大化 loss_inf) ----
            adversary_optimizer.zero_grad()
            adv_loss = -loss_inf
            adv_loss.backward()
            adversary_optimizer.step()

            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            # 记录训练指标
            prob = torch.exp(log_probs)
            _, preds = torch.max(prob, dim=1)
            observer.train_update(loss_sup, prob, preds, batch_label)

        # === 验证阶段 ===
        model.eval()
        with torch.no_grad():
            for batch_data, batch_label in val_loader:
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)

                log_probs = model(batch_data)
                loss = criterion(log_probs, batch_label)

                prob = torch.exp(log_probs)
                _, preds = torch.max(prob, dim=1)
                observer.eval_update(loss, prob, preds, batch_label)

        # === 计算指标、打印、早停、保存最佳模型 ===
        should_stop = observer.execute(
            epoch, args.epochs,
            len(train_loader.dataset), len(val_loader.dataset),
            fold=fold_idx, model=model,
        )

        if recorder is not None:
            is_best = (observer.best_dicts.get("epoch", 0) == epoch)
            recorder.record_epoch(observer, fold_idx, epoch, is_best=is_best)

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        if should_stop:
            print("⚠️  早停触发")
            break

    # ---- 记录该 fold 最佳结果 ----
    observer.finish(fold_idx)
    best_val_acc = float(observer.best_dicts["Accuracy"])
    if recorder is not None:
        recorder.record_best(observer, fold_idx)
    print(f"Fold {fold_idx} 最佳验证: Acc={best_val_acc:.4f}")
    return best_val_acc


# ========================================================================== #
#  批量测试评估（使用 RuntimeObserver）                                         #
# ========================================================================== #

def batch_evaluate_folds(args, device, recorder=None):
    """读取每个 fold 保存的模型，使用 RuntimeObserver 进行完整测试评估"""
    test_ids = getattr(args, "test_ids", None)
    if not test_ids:
        print("\n⚠️  未配置 test_ids，跳过批量测试评估。")
        return

    print(f"\n{'='*60}")
    print("📊 批量测试评估")
    print(f"{'='*60}")

    is_fewshot = "fewshot" in args.dataset_name.lower()
    is_mlda = args.model_name == "mlda"
    is_daeevit = args.model_name == "daeevit"
    is_lamsda = args.model_name == "lamsda"
    is_dann = args.model_name == "dann"
    is_deepcoral = args.model_name == "deepcoral"
    is_interpcnn = args.model_name == "interpcnn"
    is_afmcir = args.model_name == "afmcir"
    is_flat_da = is_mlda or is_dann or is_deepcoral
    test_dataset = build_fold_data(args, {}, mode="test")
    print(f"测试集: {len(test_dataset)} 样本")

    if is_fewshot:
        n_way = getattr(args, "n_way", 2)
        k_shot = getattr(args, "k_shot", 5)
        n_query = getattr(args, "n_query", 10)
        test_loader = build_fewshot_loader(
            test_dataset, n_way, k_shot, n_query, 50
        )
    elif is_flat_da:
        # MLDA / DANN / DeepCORAL 需要展平的向量输入
        test_windows = np.stack(test_dataset.windows, axis=0)
        test_flat = test_windows.reshape(test_windows.shape[0], -1).astype(np.float32)
        test_labels = np.array(test_dataset.labels, dtype=np.int64)
        test_dset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_flat), torch.from_numpy(test_labels)
        )
        test_loader = torch.utils.data.DataLoader(
            test_dset, batch_size=args.batch_size, shuffle=False
        )
    elif is_daeevit or is_lamsda or is_interpcnn or is_afmcir:
        # DAEEGViT / LA-MSDA / InterpretableCNN 需要 (B, C, W) 格式
        test_windows = np.stack(test_dataset.windows, axis=0)       # (N, W, C)
        test_bcw = test_windows.transpose(0, 2, 1).astype(np.float32)  # (N, C, W)
        test_labels = np.array(test_dataset.labels, dtype=np.int64)
        test_dset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_bcw), torch.from_numpy(test_labels)
        )
        test_loader = torch.utils.data.DataLoader(
            test_dset, batch_size=args.batch_size, shuffle=False
        )
    else:
        test_loader = build_temporal_loader(test_dataset, args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=getattr(args, "label_smoothing", 0.0)
    ).to(device)
    criterion_nll = nn.NLLLoss().to(device)

    all_fold_metrics = []

    for fold_idx in range(1, args.k_fold + 1):
        model_path = Path(args.save_dir) / f"{args.exp_name}_best_model_fold{fold_idx}.pth"
        if not model_path.exists():
            # 兼容两种命名格式
            model_path = Path(args.save_dir) / f"best_model_fold{fold_idx}.pth"
        if not model_path.exists():
            print(f"⚠️  Fold {fold_idx} 模型不存在，跳过")
            continue

        print(f"\n--- Fold {fold_idx} 测试 ---")

        model = build_model_with_kwargs(args, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # 创建 test observer
        observer = RuntimeObserver(
            log_dir=str(Path(args.save_dir) / f"test_fold{fold_idx}"),
            device=device,
            num_classes=args.num_classes,
            task="multiclass" if args.num_classes > 2 else "binary",
            average="macro" if args.num_classes > 2 else "micro",
            patience=999,
            hyperparameters=vars(args),
        )

        if is_fewshot:
            # 小样本：逐样本预测，使用 observer 记录完整指标
            _fewshot_eval_epoch(model, test_loader, device, observer, criterion_nll, mode="test")
        elif is_lamsda:
            # LA-MSDA：集成所有源域分支的预测
            with torch.no_grad():
                for batch_data, batch_label in test_loader:
                    batch_data = batch_data.to(device)
                    batch_label = batch_label.to(device)

                    avg_probs = model.ensemble_predict(batch_data)
                    loss = criterion(torch.log(avg_probs + 1e-8), batch_label)
                    _, predictions = torch.max(avg_probs, dim=1)
                    observer.test_update(loss, avg_probs, predictions, batch_label)
        elif is_flat_da:
            # MLDA / DANN / DeepCORAL：展平向量输入，推理模式 (tar_data=None)
            with torch.no_grad():
                for batch_data, batch_label in test_loader:
                    batch_data = batch_data.to(device)
                    batch_label = batch_label.to(device)

                    _, outputs = model(batch_data, None)
                    loss = criterion(outputs, batch_label)

                    prob = torch.softmax(outputs, dim=1)
                    _, predictions = torch.max(prob, dim=1)
                    observer.test_update(loss, prob, predictions, batch_label)
        elif is_daeevit:
            # DAEEGViT：(B, C, W) 输入，forward 返回 (logits, cls_features)
            with torch.no_grad():
                for batch_data, batch_label in test_loader:
                    batch_data = batch_data.to(device)
                    batch_label = batch_label.to(device)

                    outputs, _ = model(batch_data)
                    loss = criterion(outputs, batch_label)

                    prob = torch.softmax(outputs, dim=1)
                    _, predictions = torch.max(prob, dim=1)
                    observer.test_update(loss, prob, predictions, batch_label)
        elif is_interpcnn:
            # InterpretableCNN：(B, C, W) 输入，输出 log-probs
            with torch.no_grad():
                for batch_data, batch_label in test_loader:
                    batch_data = batch_data.to(device)
                    batch_label = batch_label.to(device)

                    log_probs = model(batch_data)
                    loss = criterion_nll(log_probs, batch_label)

                    prob = torch.exp(log_probs)
                    _, predictions = torch.max(prob, dim=1)
                    observer.test_update(loss, prob, predictions, batch_label)
        elif is_afmcir:
            # AFM-CIR：(B, C, W) 输入，eval 模式输出 log-probs
            with torch.no_grad():
                for batch_data, batch_label in test_loader:
                    batch_data = batch_data.to(device)
                    batch_label = batch_label.to(device)

                    log_probs = model(batch_data)
                    loss = criterion_nll(log_probs, batch_label)

                    prob = torch.exp(log_probs)
                    _, predictions = torch.max(prob, dim=1)
                    observer.test_update(loss, prob, predictions, batch_label)
        else:
            # 时序：标准前向推理，使用 observer 记录完整指标
            with torch.no_grad():
                for batch in test_loader:
                    windows = batch["window"].to(device)
                    labels = batch["label"].to(device)

                    outputs = model(windows)
                    loss = criterion(outputs, labels)

                    prob = torch.softmax(outputs, dim=1)
                    _, predictions = torch.max(prob, dim=1)
                    observer.test_update(loss, prob, predictions, labels)

        # 计算并打印测试集完整指标
        test_len = len(test_loader.dataset) if not is_fewshot else test_loader.dataset.__len__()
        observer.compute_test_result(test_len)

        # 记录该 fold 的测试集指标到 history.csv
        if recorder is not None:
            recorder.record_test(observer, fold_idx)

        # 收集该 fold 的指标
        metrics = {}
        for key in ["Accuracy", "Precision", "Recall", "Specificity",
                     "F1", "BalanceAccuracy", "CohenKappa", "AuRoc"]:
            val = observer.test_metric.get(key, 0.0)
            metrics[key] = float(val) if torch.is_tensor(val) else float(val)
        all_fold_metrics.append(metrics)

    # ---- 汇总 ----
    if all_fold_metrics:
        import numpy as np
        print(f"\n{'='*60}")
        print("📊 测试集汇总结果（所有 Fold）")
        print(f"{'='*60}")
        keys = all_fold_metrics[0].keys()
        for key in keys:
            values = [m[key] for m in all_fold_metrics]
            mean_v = np.mean(values)
            std_v = np.std(values)
            print(f"  {key}: {mean_v:.4f} +/- {std_v:.4f}")

    return all_fold_metrics


def evaluate_on_dataset(args, device, target_data_dir, checkpoint_dir=None,
                        test_subject_ids=None, is_gaipat_target=False,
                        recorder=None, tag="test"):
    """
    通用评估函数: 加载每个 fold 的模型并在指定数据集上测试

    支持跨数据集评估:
    - FatigueGuard 训练的模型 -> GAIPAT 数据测试
    - GAIPAT 训练的模型 -> FatigueGuard 数据测试

    Args:
        args: 实验配置
        device: 计算设备
        target_data_dir: 目标数据集目录
        checkpoint_dir: 模型权重目录 (None = args.save_dir)
        test_subject_ids: 测试受试者 ID 列表 (None = 使用全部)
        is_gaipat_target: 目标数据集是否为 GAIPAT
        recorder: MetricsRecorder 实例
        tag: 结果标签前缀 (如 "gaipat" / "fg" / "test")

    Returns:
        all_fold_metrics: 各 fold 的测试指标列表
    """
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else Path(args.save_dir)
    print(f"\n{'='*60}")
    print(f"  评估: {tag} (checkpoint: {ckpt_dir})")
    print(f"  数据: {target_data_dir}")
    print(f"{'='*60}")

    # ---- 构建测试数据集 ----
    is_fewshot = "fewshot" in args.dataset_name.lower()
    use_adf = getattr(args, "use_adf", True)
    local_mean_size = getattr(args, "local_mean_size", 16)
    per_sample_norm = getattr(args, "per_sample_norm", False)

    if is_gaipat_target:
        test_dataset = GaipatDataset(
            data_dir=target_data_dir,
            window_size=args.window_size,
            feature_name="deviation_cm",
            subject_ids=test_subject_ids,
            use_adf=use_adf,
            local_mean_size=local_mean_size,
            per_sample_norm=per_sample_norm,
        )
    elif is_fewshot:
        test_dataset = FewShotFatigueDataset(
            data_dir=target_data_dir,
            window_size=args.window_size,
            stride=args.stride,
            feature_name=args.feature_name,
            subject_ids=test_subject_ids,
            difficulty=getattr(args, "difficulty", None),
            use_adf=use_adf,
            local_mean_size=local_mean_size,
            per_sample_norm=per_sample_norm,
        )
    else:
        test_dataset = FatigueDataset(
            data_dir=target_data_dir,
            window_size=args.window_size,
            stride=args.stride,
            feature_name=args.feature_name,
            subject_ids=test_subject_ids,
            difficulty=getattr(args, "difficulty", None),
            use_adf=use_adf,
            local_mean_size=local_mean_size,
            per_sample_norm=per_sample_norm,
        )

    print(f"测试集: {len(test_dataset)} 样本")

    # ---- 构建 DataLoader ----
    is_mlda = args.model_name == "mlda"
    is_daeevit = args.model_name == "daeevit"
    is_lamsda = args.model_name == "lamsda"
    is_dann = args.model_name == "dann"
    is_deepcoral = args.model_name == "deepcoral"
    is_interpcnn = args.model_name == "interpcnn"
    is_afmcir = args.model_name == "afmcir"
    is_flat_da = is_mlda or is_dann or is_deepcoral

    if is_fewshot and is_gaipat_target:
        # GAIPAT + fewshot: 使用适配器
        adapter = GaipatFewShotAdapter(test_dataset)
        n_way = getattr(args, "n_way", 2)
        k_shot = getattr(args, "k_shot", 5)
        n_query = getattr(args, "n_query", 10)
        test_loader = build_fewshot_loader(adapter, n_way, k_shot, n_query, 50)
    elif is_fewshot:
        n_way = getattr(args, "n_way", 2)
        k_shot = getattr(args, "k_shot", 5)
        n_query = getattr(args, "n_query", 10)
        test_loader = build_fewshot_loader(test_dataset, n_way, k_shot, n_query, 50)
    elif is_flat_da:
        test_windows = np.stack(test_dataset.windows, axis=0)
        test_flat = test_windows.reshape(test_windows.shape[0], -1).astype(np.float32)
        test_labels = np.array(test_dataset.labels, dtype=np.int64)
        test_dset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_flat), torch.from_numpy(test_labels)
        )
        test_loader = torch.utils.data.DataLoader(
            test_dset, batch_size=args.batch_size, shuffle=False
        )
    elif is_daeevit or is_lamsda or is_interpcnn or is_afmcir:
        test_windows = np.stack(test_dataset.windows, axis=0)
        test_bcw = test_windows.transpose(0, 2, 1).astype(np.float32)
        test_labels = np.array(test_dataset.labels, dtype=np.int64)
        test_dset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_bcw), torch.from_numpy(test_labels)
        )
        test_loader = torch.utils.data.DataLoader(
            test_dset, batch_size=args.batch_size, shuffle=False
        )
    else:
        test_loader = build_temporal_loader(test_dataset, args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=getattr(args, "label_smoothing", 0.0)
    ).to(device)
    criterion_nll = nn.NLLLoss().to(device)

    all_fold_metrics = []

    for fold_idx in range(1, args.k_fold + 1):
        model_path = ckpt_dir / f"{args.exp_name}_best_model_fold{fold_idx}.pth"
        if not model_path.exists():
            model_path = ckpt_dir / f"best_model_fold{fold_idx}.pth"
        if not model_path.exists():
            print(f"  Fold {fold_idx} 模型不存在于 {ckpt_dir}，跳过")
            continue

        print(f"\n--- {tag} Fold {fold_idx} ---")

        model = build_model_with_kwargs(args, device)
        model.load_state_dict(torch.load(model_path, map_location=device,
                                          weights_only=False))
        model.eval()

        obs_log_dir = Path(args.save_dir) / f"{tag}_fold{fold_idx}"
        obs_log_dir.mkdir(parents=True, exist_ok=True)
        observer = RuntimeObserver(
            log_dir=str(obs_log_dir),
            device=device,
            num_classes=args.num_classes,
            task="multiclass" if args.num_classes > 2 else "binary",
            average="macro" if args.num_classes > 2 else "micro",
            patience=999,
            hyperparameters=vars(args),
        )

        # ---- 推理 ----
        if is_fewshot:
            _fewshot_eval_epoch(model, test_loader, device, observer,
                                criterion_nll, mode="test")
        elif is_lamsda:
            with torch.no_grad():
                for batch_data, batch_label in test_loader:
                    batch_data = batch_data.to(device)
                    batch_label = batch_label.to(device)
                    avg_probs = model.ensemble_predict(batch_data)
                    loss = criterion(torch.log(avg_probs + 1e-8), batch_label)
                    _, predictions = torch.max(avg_probs, dim=1)
                    observer.test_update(loss, avg_probs, predictions, batch_label)
        elif is_flat_da:
            with torch.no_grad():
                for batch_data, batch_label in test_loader:
                    batch_data = batch_data.to(device)
                    batch_label = batch_label.to(device)
                    _, outputs = model(batch_data, None)
                    loss = criterion(outputs, batch_label)
                    prob = torch.softmax(outputs, dim=1)
                    _, predictions = torch.max(prob, dim=1)
                    observer.test_update(loss, prob, predictions, batch_label)
        elif is_daeevit:
            with torch.no_grad():
                for batch_data, batch_label in test_loader:
                    batch_data = batch_data.to(device)
                    batch_label = batch_label.to(device)
                    outputs, _ = model(batch_data)
                    loss = criterion(outputs, batch_label)
                    prob = torch.softmax(outputs, dim=1)
                    _, predictions = torch.max(prob, dim=1)
                    observer.test_update(loss, prob, predictions, batch_label)
        elif is_interpcnn or is_afmcir:
            with torch.no_grad():
                for batch_data, batch_label in test_loader:
                    batch_data = batch_data.to(device)
                    batch_label = batch_label.to(device)
                    log_probs = model(batch_data)
                    loss = criterion_nll(log_probs, batch_label)
                    prob = torch.exp(log_probs)
                    _, predictions = torch.max(prob, dim=1)
                    observer.test_update(loss, prob, predictions, batch_label)
        else:
            with torch.no_grad():
                for batch in test_loader:
                    windows = batch["window"].to(device)
                    labels = batch["label"].to(device)
                    outputs = model(windows)
                    loss = criterion(outputs, labels)
                    prob = torch.softmax(outputs, dim=1)
                    _, predictions = torch.max(prob, dim=1)
                    observer.test_update(loss, prob, predictions, labels)

        # ---- 计算指标 ----
        test_len = (len(test_loader.dataset) if not is_fewshot
                    else test_loader.dataset.__len__())
        observer.compute_test_result(test_len)

        # 记录到 CSV (使用独立 recorder 或共享 recorder)
        if recorder is not None:
            recorder.record_test(observer, fold_idx)

        # 收集指标
        metrics = {}
        for key in ["Accuracy", "Precision", "Recall", "Specificity",
                     "F1", "BalanceAccuracy", "CohenKappa", "AuRoc"]:
            val = observer.test_metric.get(key, 0.0)
            metrics[key] = float(val) if torch.is_tensor(val) else float(val)
        all_fold_metrics.append(metrics)

    # ---- 汇总 ----
    if all_fold_metrics:
        print(f"\n{'='*60}")
        print(f"  {tag} 汇总结果 ({len(all_fold_metrics)} folds)")
        print(f"{'='*60}")
        for key in all_fold_metrics[0].keys():
            values = [m[key] for m in all_fold_metrics]
            mean_v = np.mean(values)
            std_v = np.std(values)
            print(f"  {key}: {mean_v:.4f} +/- {std_v:.4f}")

        # 保存到独立 CSV
        _save_cross_eval_csv(all_fold_metrics, args.save_dir, tag)

    return all_fold_metrics


def _save_cross_eval_csv(fold_metrics, save_dir, tag):
    """将跨数据集评估结果保存到独立 CSV 文件"""
    import csv
    if not fold_metrics:
        return

    csv_path = Path(save_dir) / f"{tag}_results.csv"
    metric_keys = list(fold_metrics[0].keys())
    header = ["fold"] + metric_keys

    rows = []
    for i, m in enumerate(fold_metrics, 1):
        row = {"fold": i}
        row.update(m)
        rows.append(row)

    # 添加 mean/std 行
    mean_row = {"fold": "mean"}
    std_row = {"fold": "std"}
    for key in metric_keys:
        values = [m[key] for m in fold_metrics]
        mean_row[key] = float(np.mean(values))
        std_row[key] = float(np.std(values))
    rows.extend([mean_row, std_row])

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  结果已保存: {csv_path}")


# ========================================================================== #
#  主流程                                                                     #
# ========================================================================== #

def main():
    # ---- 参数解析 ----
    parser = argparse.ArgumentParser(
        description="Fatigue Detection Contrast Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
评估模式 (--eval_mode):
  fatigue            FG 数据训练 + FG 数据测试 (默认)
  fatigue_to_gaipat  FG 数据训练 + GAIPAT 数据测试
  gaipat             GAIPAT 数据训练 + GAIPAT 数据测试
  gaipat_to_fatigue  GAIPAT 数据训练 + FG 数据测试

跨数据集测试时使用 --checkpoint_dir 指定已训练模型的权重目录。

示例:
  # 模式1: FG 训练 + FG 测试
  python main_fatigue.py --exp_name Fatigue_LSTM_baseline

  # 模式2: FG 训练 + GAIPAT 测试 (加载已有权重)
  python main_fatigue.py --exp_name Fatigue_LSTM_baseline \\
      --eval_mode fatigue_to_gaipat \\
      --gaipat_dir /root/autodl-tmp/shenxy/Data/gaipat/final_relabelled \\
      --checkpoint_dir ./result_20250101_120000_Fatigue_LSTM_baseline

  # 模式3: GAIPAT 训练 + GAIPAT 测试
  python main_fatigue.py --exp_name Fatigue_LSTM_baseline \\
      --eval_mode gaipat \\
      --gaipat_dir /root/autodl-tmp/shenxy/Data/gaipat/final_relabelled

  # 模式4: GAIPAT 训练 + FG 测试 (加载已有权重)
  python main_fatigue.py --exp_name Fatigue_LSTM_baseline \\
      --eval_mode gaipat_to_fatigue \\
      --gaipat_dir /root/autodl-tmp/shenxy/Data/gaipat/final_relabelled \\
      --checkpoint_dir ./result_gaipat_20250101_120000_Fatigue_LSTM_baseline
        """,
    )
    parser.add_argument("--exp_name", type=str, required=True, help="实验名称")
    parser.add_argument(
        "--eval_mode", type=str, default="fatigue",
        choices=["fatigue", "fatigue_to_gaipat", "gaipat", "gaipat_to_fatigue"],
        help="评估模式 (默认: fatigue)",
    )
    parser.add_argument(
        "--gaipat_dir", type=str, default=None,
        help="GAIPAT 数据集根目录 (含 release/ 和 grasp/ 子目录)",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="跨数据集测试时的模型权重目录 (含 best_model_fold*.pth)",
    )
    parser.add_argument(
        "--gaipat_k_fold", type=int, default=5,
        help="GAIPAT K-Fold 折数 (默认: 5)",
    )
    parser.add_argument(
        "--per_sample_norm", action="store_true", default=None,
        help="启用 per-sample Min-Max 归一化 (消除跨数据集尺度差异，跨数据集实验必须开启)",
    )
    args = parser.parse_args()

    eval_mode = args.eval_mode

    # ---- 加载实验配置 ----
    all_experiments = {}
    all_experiments.update(fatigue_temporal_experiments)
    all_experiments.update(fatigue_fewshot_experiments)
    all_experiments.update(fatigue_da_experiments)
    all_experiments.update(fatigue_dg_experiments)

    if args.exp_name not in all_experiments:
        print(f"  实验 '{args.exp_name}' 未找到。可用实验:")
        for name in all_experiments:
            print(f"  - {name}")
        sys.exit(1)

    exp_config = all_experiments[args.exp_name]
    print(f"  加载实验配置: {args.exp_name}")

    # 保存 CLI 指定的 per_sample_norm（避免被配置覆盖）
    cli_per_sample_norm = args.per_sample_norm

    for key, value in exp_config.items():
        setattr(args, key, value)

    # CLI --per_sample_norm 优先于配置文件
    if cli_per_sample_norm is not None:
        args.per_sample_norm = cli_per_sample_norm
    elif not hasattr(args, "per_sample_norm"):
        args.per_sample_norm = False

    # ---- GAIPAT 相关覆盖 ----
    gaipat_dir = args.gaipat_dir
    original_fg_dir = args.data_dir  # 保留 FG 原始路径

    if eval_mode in ("gaipat", "gaipat_to_fatigue"):
        # GAIPAT 做训练: 覆盖 data_dir 和特征
        if gaipat_dir is None:
            print("  eval_mode 为 gaipat/gaipat_to_fatigue 时必须指定 --gaipat_dir")
            sys.exit(1)
        args.data_dir = gaipat_dir
        args.data_type = "gaipat"
        args.feature_name = "deviation_cm"
        args.difficulty = None  # GAIPAT 没有 easy/hard 区分
        args.test_ids = None    # GAIPAT 测试用全部受试者
    elif eval_mode == "fatigue_to_gaipat":
        if gaipat_dir is None:
            print("  eval_mode 为 fatigue_to_gaipat 时必须指定 --gaipat_dir")
            sys.exit(1)
        args.data_type = "fatigue"  # 训练用 FG
    else:
        args.data_type = "fatigue"

    # ---- 基本设置 ----
    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"评估模式: {eval_mode}")

    # ---- 输出目录 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = f"_{eval_mode}" if eval_mode != "fatigue" else ""
    save_path = f"{args.output_dir}_{timestamp}_{args.exp_name}{mode_tag}/"
    Path(save_path).mkdir(exist_ok=True, parents=True)
    args.save_dir = save_path
    print(f"输出目录: {save_path}")

    # ---- 保存配置到 YAML ----
    save_config_yaml(args, save_path)

    # ---- CSV 指标记录器 ----
    recorder = MetricsRecorder(save_path, num_classes=args.num_classes)

    # ---- 判断是否需要训练 ----
    # fatigue: 始终训练
    # fatigue_to_gaipat: 若提供了 checkpoint_dir 则跳过训练(已有FG权重), 否则先训练
    # gaipat: 始终训练
    # gaipat_to_fatigue: 若提供了 checkpoint_dir 则跳过训练(已有GAIPAT权重), 否则先训练
    checkpoint_provided = args.checkpoint_dir is not None
    if eval_mode == "fatigue":
        needs_training = True
    elif eval_mode == "fatigue_to_gaipat":
        needs_training = not checkpoint_provided
    elif eval_mode == "gaipat":
        needs_training = True
    elif eval_mode == "gaipat_to_fatigue":
        needs_training = not checkpoint_provided
    else:
        needs_training = True

    # ---- 跳过训练时，从 checkpoint 目录推断 k_fold ----
    if not needs_training and checkpoint_provided:
        ckpt_path = Path(args.checkpoint_dir)
        if not ckpt_path.exists():
            print(f"  checkpoint_dir 不存在: {ckpt_path}")
            sys.exit(1)
        # 扫描模型文件推断 fold 数量
        fold_nums = set()
        for p in ckpt_path.glob("*best_model_fold*.pth"):
            name = p.stem
            # 匹配 ...fold{N}.pth
            import re
            m = re.search(r'fold(\d+)', name)
            if m:
                fold_nums.add(int(m.group(1)))
        if fold_nums:
            detected_k = max(fold_nums)
            args.k_fold = detected_k
            print(f"  从 checkpoint 推断 k_fold={detected_k} (folds: {sorted(fold_nums)})")
        else:
            print(f"  未在 {ckpt_path} 中找到模型文件，使用配置中的 k_fold={args.k_fold}")

    if needs_training:
        # ============================================================== #
        #  训练阶段                                                       #
        # ============================================================== #
        val_strategy = getattr(args, "val_strategy", "kfold")
        test_ids = getattr(args, "test_ids", None)
        difficulty = getattr(args, "difficulty", None)

        if difficulty:
            print(f"任务难度: {difficulty}")
        else:
            print(f"任务难度: 全部")

        is_gaipat_train = args.data_type == "gaipat"

        if is_gaipat_train:
            # ---- GAIPAT 数据划分 ----
            if val_strategy == "loso":
                folds_config, all_subject_ids = generate_gaipat_loso_folds(
                    args.data_dir, test_ids
                )
                args.all_subject_ids = all_subject_ids
                args.k_fold = len(folds_config)
                print(f"GAIPAT LOSO: {args.k_fold} 折")
            else:
                folds_config = generate_gaipat_kfold(
                    args.data_dir, k=args.gaipat_k_fold,
                    seed=args.seed, test_ids=test_ids,
                )
                args.all_subject_ids = scan_gaipat_subject_ids(args.data_dir)
                args.k_fold = len(folds_config)
                print(f"GAIPAT K-Fold: {args.k_fold} 折")
        else:
            # ---- FatigueGuard 数据划分 ----
            if val_strategy == "loso":
                folds_config, all_subject_ids = generate_loso_folds(
                    args.data_dir, test_ids, difficulty
                )
                args.all_subject_ids = all_subject_ids
                args.k_fold = len(folds_config)
                print(f"验证策略: LOSO, 共 {args.k_fold} 折")
            else:
                folds_config = getattr(args, "folds", {})
                if not folds_config:
                    folds_config = {1: {"val_ids": []}}
                args.all_subject_ids = scan_subject_ids(args.data_dir, difficulty)
                print(f"验证策略: K-Fold, 共 {len(folds_config)} 折")

        # ---- 训练循环 ----
        is_fewshot = "fewshot" in args.dataset_name.lower()
        training_type = getattr(args, "training_type", "")
        is_da = training_type == "domain_adapt"
        is_da_vit = training_type == "domain_adapt_vit"
        is_ms_da = training_type == "multi_source_da"
        is_dann = training_type == "dann"
        is_deepcoral = training_type == "deepcoral"
        is_dg_interpcnn = training_type == "dg_interpcnn"
        is_dg_afmcir = training_type == "dg_afmcir"

        start_time = time.time()
        fold_results = {}

        for fold_idx in sorted(folds_config.keys()):
            fold_config = folds_config[fold_idx]

            if is_da:
                best_val_acc = run_mlda_fold(args, device, fold_idx, fold_config,
                                             recorder=recorder)
            elif is_da_vit:
                best_val_acc = run_daeevit_fold(args, device, fold_idx, fold_config,
                                                recorder=recorder)
            elif is_ms_da:
                best_val_acc = run_lamsda_fold(args, device, fold_idx, fold_config,
                                               recorder=recorder)
            elif is_dann:
                best_val_acc = run_dann_fold(args, device, fold_idx, fold_config,
                                             recorder=recorder)
            elif is_deepcoral:
                best_val_acc = run_deepcoral_fold(args, device, fold_idx, fold_config,
                                                  recorder=recorder)
            elif is_dg_interpcnn:
                best_val_acc = run_interpcnn_fold(args, device, fold_idx, fold_config,
                                                  recorder=recorder)
            elif is_dg_afmcir:
                best_val_acc = run_afmcir_fold(args, device, fold_idx, fold_config,
                                               recorder=recorder)
            elif is_fewshot:
                best_val_acc = run_fewshot_fold(args, device, fold_idx, fold_config,
                                                recorder=recorder)
            else:
                best_val_acc = run_temporal_fold(args, device, fold_idx, fold_config,
                                                 recorder=recorder)

            fold_results[fold_idx] = best_val_acc

        # ---- 训练汇总 ----
        strategy_name = "LOSO" if val_strategy == "loso" else "K-Fold"
        data_name = "GAIPAT" if is_gaipat_train else "FatigueGuard"
        print(f"\n{'='*60}")
        print(f"  {data_name} {strategy_name} 训练完成 ({len(folds_config)} 折)")
        print(f"{'='*60}")
        for fold_idx, val_acc in fold_results.items():
            print(f"  Fold {fold_idx}: Val Acc = {val_acc:.4f}")
        accs = list(fold_results.values())
        if accs:
            print(f"\n  平均: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

        # ---- 同数据集测试评估 ----
        if is_gaipat_train:
            # GAIPAT: 设置 test_ids 为全部受试者，然后使用 batch_evaluate_folds
            args.test_ids = scan_gaipat_subject_ids(args.data_dir)
            batch_evaluate_folds(args, device, recorder=recorder)
        elif test_ids:
            # FatigueGuard: 使用配置中的 test_ids
            batch_evaluate_folds(args, device, recorder=recorder)

    # ============================================================== #
    #  跨数据集评估阶段                                               #
    # ============================================================== #

    if eval_mode == "fatigue_to_gaipat":
        # 模式2: FG 训练完成后，在 GAIPAT 上测试
        print(f"\n{'#'*60}")
        print(f"  跨数据集评估: FG -> GAIPAT")
        print(f"{'#'*60}")

        ckpt_dir = args.checkpoint_dir if args.checkpoint_dir else args.save_dir
        gaipat_all_ids = scan_gaipat_subject_ids(gaipat_dir)

        evaluate_on_dataset(
            args, device,
            target_data_dir=gaipat_dir,
            checkpoint_dir=ckpt_dir,
            test_subject_ids=None,  # 全部 GAIPAT 受试者
            is_gaipat_target=True,
            recorder=None,  # 跨数据集用独立 CSV
            tag="fg_to_gaipat",
        )

    elif eval_mode == "gaipat_to_fatigue":
        # 模式4: GAIPAT 训练完成后，在 FG 上测试
        print(f"\n{'#'*60}")
        print(f"  跨数据集评估: GAIPAT -> FatigueGuard")
        print(f"{'#'*60}")

        ckpt_dir = args.checkpoint_dir if args.checkpoint_dir else args.save_dir

        # 恢复 FG 数据配置
        args.data_dir = original_fg_dir
        args.data_type = "fatigue"
        args.feature_name = exp_config.get("feature_name", "deviation_px_before_calibrate")
        args.difficulty = exp_config.get("difficulty", "easy")

        # 获取 FG 全部受试者做测试
        fg_difficulty = exp_config.get("difficulty", None)
        fg_all_ids = scan_subject_ids(original_fg_dir, fg_difficulty)

        evaluate_on_dataset(
            args, device,
            target_data_dir=original_fg_dir,
            checkpoint_dir=ckpt_dir,
            test_subject_ids=None,  # 全部 FG 受试者
            is_gaipat_target=False,
            recorder=None,
            tag="gaipat_to_fg",
        )

    # ---- 保存 CSV 指标文件 ----
    best_path, fold_paths = recorder.save()
    if best_path.exists():
        print(f"\n  best_results: {best_path}")
    if fold_paths:
        print(f"  fold histories: {[str(p) for p in fold_paths]}")

    # ---- 耗时 ----
    if needs_training:
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        print(f"\n总耗时: {hours}h {minutes}m {seconds}s")

    print(f"\n  完成! 评估模式: {eval_mode}")


if __name__ == "__main__":
    main()
