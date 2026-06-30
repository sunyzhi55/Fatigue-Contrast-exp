"""
疲劳检测数据集模块
支持从JSONL文件加载时序特征数据，构建滑动窗口样本。

数据格式说明：
- 文件名格式: [id]_[easy|hard]_[alert|sleepy].jsonl
- 每行为一帧数据，包含 deviation_px_after_calibrate 等特征
- alert=0, sleepy=1
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Dict, Any


class FatigueDataset(Dataset):
    """
    疲劳检测时序数据集
    从JSONL文件加载 deviation 特征，构建滑动窗口。
    适用于 LSTM、Transformer、Mamba 等时序模型。
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 30,
        stride: int = 15,
        feature_name: str = "deviation_px_after_calibrate",
        subject_ids: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
    ):
        """
        Args:
            data_dir: JSONL 文件目录
            window_size: 滑动窗口大小（帧数）
            stride: 滑动窗口步长
            feature_name: 要提取的特征字段名
            subject_ids: 如果指定，仅加载这些受试者ID的数据
            difficulty: "easy" / "hard" / None(加载全部)
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.feature_name = feature_name

        self.windows = []       # shape: (num_samples, window_size)
        self.labels = []        # 0=alert, 1=sleepy
        self.subject_ids = []   # 受试者ID
        self.file_ids = []      # 文件ID
        self.difficulties = []  # easy/hard

        self._load_data(subject_ids, difficulty)

    def _load_data(self, subject_ids: Optional[List[str]], difficulty: Optional[str] = None):
        """加载所有JSONL文件并构建滑动窗口"""
        jsonl_files = sorted(self.data_dir.glob("*.jsonl"))
        if len(jsonl_files) == 0:
            raise RuntimeError(f"未在 {self.data_dir} 下找到任何 JSONL 文件")

        for file_path in jsonl_files:
            # 解析文件名: [id]_[easy|hard]_[alert|sleepy].jsonl
            stem = file_path.stem
            parts = stem.split("_")
            if len(parts) != 3:
                continue

            file_id, file_difficulty, state = parts
            if state not in ("alert", "sleepy"):
                continue

            # 按任务难度过滤
            if difficulty is not None and file_difficulty != difficulty:
                continue

            # 按受试者ID过滤
            if subject_ids is not None and file_id not in subject_ids:
                continue

            label = 0 if state == "alert" else 1

            # 读取 JSONL 文件
            deviation_values = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    frame = json.loads(line)
                    value = frame.get(self.feature_name)
                    if value is not None:
                        deviation_values.append(float(value))

            num_frames = len(deviation_values)
            if num_frames < self.window_size:
                continue  # 帧数不足，跳过

            # 构建滑动窗口
            for start in range(0, num_frames - self.window_size + 1, self.stride):
                window = deviation_values[start:start + self.window_size]
                self.windows.append(window)
                self.labels.append(label)
                self.subject_ids.append(file_id)
                self.file_ids.append(stem)
                self.difficulties.append(file_difficulty)

        self.num_samples = len(self.windows)
        if self.num_samples == 0:
            raise RuntimeError(
                f"未在 {self.data_dir} 下构建到任何样本"
                f"（window_size={self.window_size}, stride={self.stride}）"
            )
        diff_str = difficulty if difficulty else "easy+hard"
        print(f"[FatigueDataset] 加载完成: {self.num_samples} 个窗口样本"
              f" (难度={diff_str}, 来自 {len(set(self.file_ids))} 个文件)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        window = torch.tensor(self.windows[idx], dtype=torch.float32)
        label = self.labels[idx]
        return {
            "window": window,
            "label": label,
            "file_id": self.file_ids[idx],
            "subject_id": self.subject_ids[idx],
            "difficulty": self.difficulties[idx],
        }


class FewShotFatigueDataset(Dataset):
    """
    小样本学习疲劳检测数据集
    按类别组织样本，支持 Episodic Sampling（N-way K-shot）。
    适用于 ProtoNet、RelationNet 等小样本学习模型。
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 30,
        stride: int = 15,
        feature_name: str = "deviation_px_after_calibrate",
        subject_ids: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.feature_name = feature_name

        # 按类别组织: {0: [(window, subject_id), ...], 1: [...]}
        self.class_to_samples: Dict[int, List[Dict]] = {0: [], 1: []}
        self._load_data(subject_ids, difficulty)

        self.num_classes = 2
        total = sum(len(v) for v in self.class_to_samples.values())
        diff_str = difficulty if difficulty else "easy+hard"
        print(f"[FewShotFatigueDataset] 加载完成: {total} 个窗口样本"
              f" (难度={diff_str},"
              f" alert={len(self.class_to_samples[0])},"
              f" sleepy={len(self.class_to_samples[1])})")

    def _load_data(self, subject_ids: Optional[List[str]], difficulty: Optional[str] = None):
        """加载JSONL文件，按类别分组存储窗口样本"""
        jsonl_files = sorted(self.data_dir.glob("*.jsonl"))
        if len(jsonl_files) == 0:
            raise RuntimeError(f"未在 {self.data_dir} 下找到任何 JSONL 文件")

        for file_path in jsonl_files:
            stem = file_path.stem
            parts = stem.split("_")
            if len(parts) != 3:
                continue

            file_id, file_difficulty, state = parts
            if state not in ("alert", "sleepy"):
                continue

            # 按任务难度过滤
            if difficulty is not None and file_difficulty != difficulty:
                continue

            if subject_ids is not None and file_id not in subject_ids:
                continue

            label = 0 if state == "alert" else 1

            deviation_values = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    frame = json.loads(line)
                    value = frame.get(self.feature_name)
                    if value is not None:
                        deviation_values.append(float(value))

            num_frames = len(deviation_values)
            if num_frames < self.window_size:
                continue

            for start in range(0, num_frames - self.window_size + 1, self.stride):
                window = deviation_values[start:start + self.window_size]
                self.class_to_samples[label].append({
                    "window": window,
                    "subject_id": file_id,
                    "file_id": stem,
                    "difficulty": difficulty,
                })

        for c in [0, 1]:
            if len(self.class_to_samples[c]) == 0:
                raise RuntimeError(
                    f"类别 {c} 没有样本。请检查数据目录和受试者ID过滤条件。"
                )

    def get_num_classes(self):
        return self.num_classes

    def get_class_samples_count(self):
        return {c: len(samples) for c, samples in self.class_to_samples.items()}

    def sample_episode(
        self,
        n_way: int = 2,
        k_shot: int = 5,
        n_query: int = 10,
    ) -> Dict[str, Any]:
        """
        采样一个 Episode（N-way K-shot）

        Args:
            n_way: 类别数（默认2: alert/sleepy）
            k_shot: 每个类别的支持集样本数
            n_query: 每个类别的查询集样本数

        Returns:
            dict: {
                'support_windows': tensor (n_way*k_shot, window_size),
                'support_labels': tensor (n_way*k_shot,),
                'query_windows': tensor (n_way*n_query, window_size),
                'query_labels': tensor (n_way*n_query,),
            }
        """
        import random

        available_classes = [c for c in range(self.num_classes)
                            if len(self.class_to_samples[c]) >= k_shot + n_query]
        if len(available_classes) < n_way:
            # 降级：使用所有可用类别
            available_classes = [c for c in range(self.num_classes)
                                if len(self.class_to_samples[c]) >= k_shot + 1]
            if len(available_classes) < n_way:
                raise ValueError(
                    f"可用类别数 ({len(available_classes)}) < n_way ({n_way})。"
                    f"各类别样本数: {self.get_class_samples_count()}"
                )

        selected_classes = random.sample(available_classes, n_way)

        support_windows = []
        support_labels = []
        query_windows = []
        query_labels = []

        for new_label, cls in enumerate(selected_classes):
            samples = self.class_to_samples[cls]
            selected = random.sample(samples, min(k_shot + n_query, len(samples)))

            for item in selected[:k_shot]:
                support_windows.append(item["window"])
                support_labels.append(new_label)

            for item in selected[k_shot:k_shot + n_query]:
                query_windows.append(item["window"])
                query_labels.append(new_label)

        return {
            "support_windows": torch.tensor(support_windows, dtype=torch.float32),
            "support_labels": torch.tensor(support_labels, dtype=torch.long),
            "query_windows": torch.tensor(query_windows, dtype=torch.float32),
            "query_labels": torch.tensor(query_labels, dtype=torch.long),
        }

    def __len__(self):
        # 返回一个合理的估计值（用于 DataLoader）
        total = sum(len(v) for v in self.class_to_samples.values())
        return max(total // 10, 1)

    def __getitem__(self, idx):
        # 按需采样 episode
        return self.sample_episode()


def build_temporal_loader(
    dataset: FatigueDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """构建时序模型的 DataLoader"""

    def collate_fn(batch):
        windows = torch.stack([item["window"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return {
            "window": windows,
            "label": labels,
            "file_id": [item["file_id"] for item in batch],
            "subject_id": [item["subject_id"] for item in batch],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_fewshot_loader(
    dataset: FewShotFatigueDataset,
    n_way: int = 2,
    k_shot: int = 5,
    n_query: int = 10,
    episodes_per_epoch: int = 100,
    num_workers: int = 0,
) -> DataLoader:
    """构建小样本学习的 DataLoader（Episodic Training）"""

    def collate_fn(batch):
        # batch 中每个元素是一个 episode（来自 sample_episode）
        # 取第一个 episode（因为 batch_size=1 在 ep_loader 中）
        episode = batch[0]
        return episode

    # 创建一个简单的 wrapper dataset
    class EpisodeWrapper(Dataset):
        def __init__(self, fs_dataset, n_way, k_shot, n_query, num_episodes):
            self.fs_dataset = fs_dataset
            self.n_way = n_way
            self.k_shot = k_shot
            self.n_query = n_query
            self.num_episodes = num_episodes

        def __len__(self):
            return self.num_episodes

        def __getitem__(self, idx):
            return self.fs_dataset.sample_episode(
                self.n_way, self.k_shot, self.n_query
            )

    wrapper = EpisodeWrapper(dataset, n_way, k_shot, n_query, episodes_per_epoch)

    return DataLoader(
        wrapper,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
