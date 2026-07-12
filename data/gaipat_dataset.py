"""
GAIPAT 公开数据集加载模块

目录结构:
    gaipat_dir/
    |-- release/
    |   |-- {subject_id}_{task}_{step}_{event}_{block_id}_{label}.jsonl
    |   ...
    |-- grasp/
        |-- {subject_id}_{task}_{step}_{event}_{block_id}_{label}.jsonl
        ...

标签规则:
    label=0  分心 (distracted)
    label=1  专注 (focused)
    其他标签 (2, 3) 丢弃

每条 JSONL 文件包含 256 帧数据，核心特征字段为 deviation_cm（偏差距离，厘米）。
每个文件直接作为一个完整样本（不做滑动窗口），与 window_size=256 配置匹配。
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Set


from data.fatigue_dataset import compute_adf


class GaipatDataset(Dataset):
    """
    GAIPAT 数据集
    加载 release/ 和 grasp/ 子目录下 label=0/1 的 JSONL 文件，
    提取 deviation_cm 特征并计算 ADF 三通道。
    每个文件 = 一个样本（256帧），不做滑动窗口。

    接口与 FatigueDataset 兼容，返回相同结构的 dict:
    {"window", "label", "subject_id", "file_id"}
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 256,
        feature_name: str = "deviation_cm",
        subject_ids: Optional[List[str]] = None,
        use_adf: bool = True,
        local_mean_size: int = 16,
        per_sample_norm: bool = True,
        **kwargs,  # 兼容 FatigueDataset 的多余参数 (stride, difficulty 等)
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.feature_name = feature_name
        self.use_adf = use_adf
        self.local_mean_size = local_mean_size
        self.per_sample_norm = per_sample_norm

        self.windows = []
        self.labels = []
        self.subject_ids = []
        self.file_ids = []

        self._load_data(subject_ids)

    def _load_data(self, subject_ids: Optional[List[str]]):
        """扫描 release/ 和 grasp/ 子目录，加载所有有效文件"""
        subject_set = set(subject_ids) if subject_ids else None

        for subdir_name in ["release", "grasp"]:
            subdir = self.data_dir / subdir_name
            if not subdir.exists():
                continue
            self._load_subdir(subdir, subject_set)

        self.num_samples = len(self.windows)
        if self.num_samples == 0:
            raise RuntimeError(
                f"未在 {self.data_dir} 下加载到任何有效样本。"
                f"请检查目录结构 (release/grasp) 和文件名格式。"
            )

        unique_subjects = set(self.subject_ids)
        class_counts = {}
        for lbl in self.labels:
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

        chan_str = "ADF*3" if self.use_adf else "1ch"
        print(f"[GaipatDataset] 加载完成: {self.num_samples} 个样本"
              f" ({len(unique_subjects)} subjects, {chan_str},"
              f" class_dist={class_counts})")

        for c in [0, 1]:
            if class_counts.get(c, 0) == 0:
                raise RuntimeError(f"类别 {c} 没有样本，请检查数据。")

    def _load_subdir(self, subdir: Path, subject_set: Optional[Set[str]]):
        """加载单个子目录下的所有有效 JSONL 文件"""
        jsonl_files = sorted(subdir.glob("*.jsonl"))

        for file_path in jsonl_files:
            stem = file_path.stem
            parts = stem.split("_")
            if len(parts) != 6:
                continue

            subj_id, task, step, event, block_id, label_str = parts

            # 标签过滤: 仅保留 0 (distracted) 和 1 (focused)
            try:
                label = int(label_str)
            except ValueError:
                continue
            if label not in (0, 1):
                continue

            # 受试者过滤
            if subject_set is not None and subj_id not in subject_set:
                continue

            # 读取 deviation_cm 序列
            drift_values = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        frame = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    value = frame.get(self.feature_name)
                    if value is not None:
                        try:
                            drift_values.append(float(value))
                        except (TypeError, ValueError):
                            continue

            num_frames = len(drift_values)
            if num_frames < self.window_size:
                continue

            # Per-sample Min-Max 归一化（消除跨数据集尺度差异，映射到 [0,1]）
            if self.per_sample_norm:
                drift_arr = np.array(drift_values, dtype=np.float32)
                d_min = drift_arr.min()
                d_max = drift_arr.max()
                rng = d_max - d_min
                if rng > 1e-8:
                    drift_values = ((drift_arr - d_min) / rng).tolist()
                else:
                    drift_values = np.zeros_like(drift_arr).tolist()

            # 计算 ADF 三通道特征（在全序列上计算）
            if self.use_adf:
                sequence = compute_adf(
                    np.array(drift_values, dtype=np.float32),
                    self.local_mean_size,
                )  # (T, 3)
            else:
                sequence = np.array(drift_values, dtype=np.float32)  # (T,)

            # 每个文件作为一个完整样本（取前 window_size 帧）
            window = sequence[:self.window_size]
            self.windows.append(np.ascontiguousarray(window, dtype=np.float32))
            self.labels.append(label)
            self.subject_ids.append(subj_id)
            self.file_ids.append(stem)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        window = torch.from_numpy(self.windows[idx].copy()).float()
        label = self.labels[idx]
        return {
            "window": window,
            "label": label,
            "subject_id": self.subject_ids[idx],
            "file_id": self.file_ids[idx],
        }


def scan_gaipat_subject_ids(data_dir: str) -> List[str]:
    """扫描 GAIPAT 数据目录，提取所有有效文件的受试者 ID"""
    data_path = Path(data_dir)
    subject_ids = set()
    for subdir_name in ["release", "grasp"]:
        subdir = data_path / subdir_name
        if not subdir.exists():
            continue
        for f in subdir.glob("*.jsonl"):
            parts = f.stem.split("_")
            if len(parts) == 6:
                try:
                    label = int(parts[5])
                    if label in (0, 1):
                        subject_ids.add(parts[0])
                except ValueError:
                    continue
    return sorted(subject_ids)


def generate_gaipat_loso_folds(
    data_dir: str, test_ids: Optional[List[str]] = None
):
    """
    为 GAIPAT 数据集生成 LOSO (Leave-One-Subject-Out) 划分

    Returns:
        folds: {1: {"val_ids": ["subj1"], "train_ids": [...]}, ...}
        all_subject_ids: 所有受试者 ID
    """
    all_ids = scan_gaipat_subject_ids(data_dir)
    if not all_ids:
        raise RuntimeError(f"未在 {data_dir} 中找到任何 GAIPAT 受试者数据")

    test_set = set(test_ids) if test_ids else set()
    loo_ids = [s for s in all_ids if s not in test_set]

    folds = {}
    for i, sid in enumerate(loo_ids, start=1):
        folds[i] = {
            "val_ids": [sid],
            "train_ids": [s for s in loo_ids if s != sid],
        }

    print(f"[GAIPAT LOSO] 检测到 {len(all_ids)} 个受试者")
    if test_ids:
        print(f"[GAIPAT LOSO] 测试集: {len(test_ids)} 个受试者 (不参与 LOSO)")
    print(f"[GAIPAT LOSO] 生成 {len(folds)} 个 fold")

    return folds, all_ids


def generate_gaipat_kfold(
    data_dir: str, k: int = 5, seed: int = 42,
    test_ids: Optional[List[str]] = None,
):
    """
    为 GAIPAT 数据集生成 K-Fold 划分（按受试者分组）

    Returns:
        folds: {1: {"val_ids": [...], "train_ids": [...]}, ...}
    """
    all_ids = scan_gaipat_subject_ids(data_dir)
    if not all_ids:
        raise RuntimeError(f"未在 {data_dir} 中找到任何 GAIPAT 受试者数据")

    test_set = set(test_ids) if test_ids else set()
    split_ids = [s for s in all_ids if s not in test_set]

    rng = np.random.RandomState(seed)
    indices = np.arange(len(split_ids))
    rng.shuffle(indices)

    fold_indices = np.array_split(indices, k)

    folds = {}
    for i in range(k):
        val_idx = set(fold_indices[i])
        train_idx = set()
        for j in range(k):
            if j != i:
                train_idx.update(fold_indices[j])

        val_subjects = [split_ids[idx] for idx in sorted(val_idx)]
        train_subjects = [split_ids[idx] for idx in sorted(train_idx)]
        folds[i + 1] = {
            "val_ids": val_subjects,
            "train_ids": train_subjects,
        }

    print(f"[GAIPAT KFold] {len(split_ids)} subjects -> {k} folds")
    for fi, fc in folds.items():
        print(f"  Fold {fi}: {len(fc['train_ids'])} train, "
              f"{len(fc['val_ids'])} val subjects")

    return folds
