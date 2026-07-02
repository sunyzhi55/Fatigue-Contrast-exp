"""
疲劳检测对比实验训练脚本

支持的实验类型:
- 时序基线: LSTM, Transformer, Mamba
- 小样本学习: ProtoNet, RelationNet
- 域适应: MLDA, DAEEGViT, LA-MSDA, DANN, DeepCORAL
- 域泛化: InterpretableCNN

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

            if src_data.size(0) != tar_data.size(0):
                continue

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

            if src_data.size(0) != tar_data.size(0):
                continue

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

            if src_data.size(0) != tar_data.size(0):
                continue

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
    elif is_daeevit or is_lamsda or is_interpcnn:
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
    all_experiments.update(fatigue_dg_experiments)

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
    training_type = getattr(args, "training_type", "")
    is_da = training_type == "domain_adapt"
    is_da_vit = training_type == "domain_adapt_vit"
    is_ms_da = training_type == "multi_source_da"
    is_dann = training_type == "dann"
    is_deepcoral = training_type == "deepcoral"
    is_dg_interpcnn = training_type == "dg_interpcnn"

    # ---- 训练循环 ----
    start_time = time.time()
    fold_results = {}

    for fold_idx in sorted(folds_config.keys()):
        fold_config = folds_config[fold_idx]

        if is_da:
            best_val_acc = run_mlda_fold(args, device, fold_idx, fold_config, recorder=recorder)
        elif is_da_vit:
            best_val_acc = run_daeevit_fold(args, device, fold_idx, fold_config, recorder=recorder)
        elif is_ms_da:
            best_val_acc = run_lamsda_fold(args, device, fold_idx, fold_config, recorder=recorder)
        elif is_dann:
            best_val_acc = run_dann_fold(args, device, fold_idx, fold_config, recorder=recorder)
        elif is_deepcoral:
            best_val_acc = run_deepcoral_fold(args, device, fold_idx, fold_config, recorder=recorder)
        elif is_dg_interpcnn:
            best_val_acc = run_interpcnn_fold(args, device, fold_idx, fold_config, recorder=recorder)
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
