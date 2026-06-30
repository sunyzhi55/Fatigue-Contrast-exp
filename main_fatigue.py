"""
疲劳检测对比实验训练脚本

支持的实验类型:
- 时序基线: LSTM, Transformer, Mamba
- 小样本学习: ProtoNet, RelationNet

功能:
- 训练前自动保存配置到 YAML 文件
- 支持基于受试者ID的 K-Fold / LOSO 数据划分
- 使用 RuntimeObserver 进行完整指标监控（Acc/Pre/Rec/Spe/F1/Kappa/AUC/BalAcc）
- 训练完成后批量评估所有 fold 的模型

使用方法:
    python main_fatigue.py --exp_name Fatigue_LSTM_baseline
    python main_fatigue.py --exp_name Fatigue_ProtoNet_baseline
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
from data.fatigue_dataset import (
    FatigueDataset, FewShotFatigueDataset,
    build_temporal_loader, build_fewshot_loader,
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
    use_adf = getattr(args, "use_adf", True)
    num_channels = 3 if use_adf else 1
    if is_fewshot or is_mlda:
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
        fold_config: {"val_ids": [...]}
        mode: "train" / "val" / "test"

    Returns:
        dataset: FatigueDataset 或 FewShotFatigueDataset
    """
    is_fewshot = "fewshot" in args.dataset_name.lower()
    difficulty = getattr(args, "difficulty", None)  # "easy" / "hard" / None
    use_adf = getattr(args, "use_adf", True)
    local_mean_size = getattr(args, "local_mean_size", 16)

    if mode == "test":
        subject_ids = getattr(args, "test_ids", None)
    elif mode == "val":
        subject_ids = fold_config.get("val_ids", None)
    else:
        val_ids = fold_config.get("val_ids", [])
        test_ids = getattr(args, "test_ids", []) or []
        exclude_ids = set(val_ids) | set(test_ids)
        all_subject_ids = getattr(args, "all_subject_ids", None)
        if all_subject_ids is not None:
            subject_ids = [s for s in all_subject_ids if s not in exclude_ids]
        else:
            subject_ids = None

    adf_kwargs = dict(use_adf=use_adf, local_mean_size=local_mean_size)
    if is_fewshot:
        dataset = FewShotFatigueDataset(
            data_dir=args.data_dir,
            window_size=args.window_size,
            stride=args.stride,
            feature_name=args.feature_name,
            subject_ids=subject_ids,
            difficulty=difficulty,
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
            **adf_kwargs,
        )

    if mode == "train" and subject_ids is None and not is_fewshot:
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

            # 跳过 batch size 不匹配的情况
            if src_data.size(0) != tar_data.size(0):
                continue

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
    test_dataset = build_fold_data(args, {}, mode="test")
    print(f"测试集: {len(test_dataset)} 样本")

    if is_fewshot:
        n_way = getattr(args, "n_way", 2)
        k_shot = getattr(args, "k_shot", 5)
        n_query = getattr(args, "n_query", 10)
        test_loader = build_fewshot_loader(
            test_dataset, n_way, k_shot, n_query, 50
        )
    elif is_mlda:
        # MLDA 需要展平的向量输入
        test_windows = np.stack(test_dataset.windows, axis=0)
        test_flat = test_windows.reshape(test_windows.shape[0], -1).astype(np.float32)
        test_labels = np.array(test_dataset.labels, dtype=np.int64)
        test_dset = torch.utils.data.TensorDataset(
            torch.from_numpy(test_flat), torch.from_numpy(test_labels)
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
        elif is_mlda:
            # MLDA：展平向量输入，推理模式 (tar_data=None)
            with torch.no_grad():
                for batch_data, batch_label in test_loader:
                    batch_data = batch_data.to(device)
                    batch_label = batch_label.to(device)

                    _, outputs = model(batch_data, None)
                    loss = criterion(outputs, batch_label)

                    prob = torch.softmax(outputs, dim=1)
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
            print(f"  {key}: {mean_v:.4f} ± {std_v:.4f}")


# ========================================================================== #
#  主流程                                                                     #
# ========================================================================== #

def main():
    # ---- 参数解析 ----
    parser = argparse.ArgumentParser(description="Fatigue Detection Baselines")
    parser.add_argument("--exp_name", type=str, required=True, help="实验名称")
    args = parser.parse_args()

    # ---- 加载实验配置 ----
    all_experiments = {}
    all_experiments.update(fatigue_temporal_experiments)
    all_experiments.update(fatigue_fewshot_experiments)
    all_experiments.update(fatigue_da_experiments)

    if args.exp_name not in all_experiments:
        print(f"❌ 实验 '{args.exp_name}' 未找到。可用实验:")
        for name in all_experiments:
            print(f"  - {name}")
        sys.exit(1)

    exp_config = all_experiments[args.exp_name]
    print(f"✅ 加载实验配置: {args.exp_name}")

    for key, value in exp_config.items():
        setattr(args, key, value)

    # ---- 基本设置 ----
    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---- 输出目录 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{args.output_dir}_{timestamp}_{args.exp_name}/"
    Path(save_path).mkdir(exist_ok=True, parents=True)
    args.save_dir = save_path
    print(f"输出目录: {save_path}")

    # ---- 保存配置到 YAML ----
    save_config_yaml(args, save_path)

    # ---- CSV 指标记录器（history.csv + best_results.csv）----
    recorder = MetricsRecorder(save_path, num_classes=args.num_classes)

    # ---- 数据划分 ----
    val_strategy = getattr(args, "val_strategy", "kfold")  # "kfold" 或 "loso"
    test_ids = getattr(args, "test_ids", None)
    difficulty = getattr(args, "difficulty", None)         # "easy" / "hard" / None

    if difficulty:
        print(f"任务难度: {difficulty}")
    else:
        print(f"任务难度: 全部 (easy + hard)")

    if val_strategy == "loso":
        # LOSO: 自动从数据目录读取受试者ID，生成 N 个 fold
        folds_config, all_subject_ids = generate_loso_folds(args.data_dir, test_ids, difficulty)
        args.all_subject_ids = all_subject_ids
        args.k_fold = len(folds_config)
        print(f"验证策略: LOSO (Leave-One-Subject-Out), 共 {args.k_fold} 折")
    else:
        # K-Fold: 使用配置文件中手动指定的 folds
        folds_config = getattr(args, "folds", {})
        if not folds_config:
            print("⚠️  未配置 folds，使用默认单次训练/验证划分。")
            folds_config = {1: {"val_ids": []}}
        args.all_subject_ids = scan_subject_ids(args.data_dir, difficulty)
        print(f"验证策略: K-Fold, 共 {len(folds_config)} 折")

    is_fewshot = "fewshot" in args.dataset_name.lower()
    is_da = getattr(args, "training_type", "") == "domain_adapt"

    # ---- 训练循环 ----
    start_time = time.time()
    fold_results = {}

    for fold_idx in sorted(folds_config.keys()):
        fold_config = folds_config[fold_idx]

        if is_da:
            best_val_acc = run_mlda_fold(args, device, fold_idx, fold_config, recorder=recorder)
        elif is_fewshot:
            best_val_acc = run_fewshot_fold(args, device, fold_idx, fold_config, recorder=recorder)
        else:
            best_val_acc = run_temporal_fold(args, device, fold_idx, fold_config, recorder=recorder)

        fold_results[fold_idx] = best_val_acc

    # ---- 汇总 ----
    strategy_name = "LOSO" if val_strategy == "loso" else "K-Fold"
    print(f"\n{'='*60}")
    print(f"📊 {strategy_name} 训练完成 ({len(folds_config)} 折)")
    print(f"{'='*60}")
    for fold_idx, val_acc in fold_results.items():
        print(f"  Fold {fold_idx}: Val Acc = {val_acc:.4f}")

    import numpy as np
    accs = list(fold_results.values())
    print(f"\n  平均: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # ---- 批量测试评估 ----
    batch_evaluate_folds(args, device, recorder=recorder)

    # ---- 保存 CSV 指标文件（history 各 fold 已实时落盘，此处补 best 的 mean/std）----
    best_path, fold_paths = recorder.save()
    print(f"\n✅ 各 fold 最佳结果已实时写入: {best_path}（末尾含 mean/std）")
    print(f"✅ 各 fold 每轮指标历史（实时写入）: {fold_paths}")

    # ---- 耗时 ----
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"\n总耗时: {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
