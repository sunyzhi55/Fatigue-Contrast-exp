"""
CSV 指标记录器

用于疲劳检测对比实验，记录两类结果到每次运行的输出文件夹：

1. history_fold{N}.csv —— 每个 fold / LOSO **单独**一个文件，**每训练一轮实时追加**
   train + val 指标到文件末尾（测试评估完成后再追加 test 行）。长表格式，
   `split` 列区分 train / val / test，`is_best` 标记该轮是否为当前 fold 最佳轮。
   混淆矩阵按位置展平为独立列（binary 即 cm_00/cm_01/cm_10/cm_11 四个值）。

2. best_results.csv —— 每个 fold / LOSO **最佳轮**的 train + val 指标（宽表，
   train_* / val_* 并排），每折完成时实时追加一行；全部 fold 结束后追加
   mean / std 汇总行。

设计要点：
- history 按 fold 分文件，命名 `history_fold{N:02d}.csv` 区分；
- 所有写入采用「追加 + 即时关闭文件」方式，保证每轮结束即落盘，进程中断不丢失；
- best_results 跨 fold 聚合，便于直接读均值方差。
"""
import csv
from pathlib import Path
import numpy as np
import torch


# 统一指标键（顺序即 CSV 列顺序）
METRIC_KEYS = [
    "Loss", "Accuracy", "Precision", "Recall", "Specificity",
    "F1", "CohenKappa", "BalanceAccuracy", "AuRoc",
]


def _to_float(v):
    """安全地把 torch.Tensor / numpy / 数字转成 float。"""
    try:
        if v is None:
            return 0.0
        if torch.is_tensor(v):
            if v.numel() == 1:
                return float(v.item())
            return float(v.float().mean().item())
        if isinstance(v, np.ndarray):
            return float(v.mean()) if v.size else 0.0
        return float(v)
    except Exception:
        return 0.0


def _cm_columns(num_classes):
    """混淆矩阵展平后的列名，例如 binary: cm_00, cm_01, cm_10, cm_11。"""
    return [f"cm_{i}{j}" for i in range(num_classes) for j in range(num_classes)]


def _flatten_cm(cm, num_classes):
    """把混淆矩阵展平为 {cm_ij: int} 字典，缺失或异常时填 0。"""
    cols = {}
    keys = [(i, j) for i in range(num_classes) for j in range(num_classes)]
    if cm is None:
        for i, j in keys:
            cols[f"cm_{i}{j}"] = 0
        return cols
    if torch.is_tensor(cm):
        cm = cm.detach().cpu().numpy()
    cm = np.array(cm)
    expected = num_classes * num_classes
    if cm.size < expected:
        cm = np.zeros((num_classes, num_classes), dtype=cm.dtype if cm.size else int)
    if cm.ndim != 2:
        cm = cm.reshape(num_classes, num_classes)
    for i, j in keys:
        try:
            cols[f"cm_{i}{j}"] = int(cm[i, j])
        except Exception:
            cols[f"cm_{i}{j}"] = 0
    return cols


class MetricsRecorder:
    """按 fold 分文件实时记录 history，跨 fold 聚合 best_results。"""

    def __init__(self, save_dir, num_classes=2):
        self.save_dir = Path(save_dir)
        self.num_classes = num_classes
        self.metric_cols = METRIC_KEYS + _cm_columns(num_classes)
        self._fold_seen = set()        # 已创建表头的 fold history 文件
        self._best_seen = False        # best_results.csv 是否已写表头
        self.best_rows = []            # 缓存各 fold best 行，用于末尾 mean/std

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pack(metric_dict, loss, auc, balacc, num_classes):
        """从一个 metric dict + 标量拼出一组指标（含混淆矩阵展平）。"""
        pack = {"Loss": _to_float(loss)}
        for k in ["Accuracy", "Precision", "Recall", "Specificity",
                  "F1", "CohenKappa"]:
            pack[k] = _to_float(metric_dict.get(k, 0.0)) if metric_dict else 0.0
        pack["BalanceAccuracy"] = _to_float(balacc)
        pack["AuRoc"] = _to_float(auc)
        cm = metric_dict.get("confusionMatrix", None) if metric_dict else None
        pack.update(_flatten_cm(cm, num_classes))
        return pack

    def _append_csv(self, path, header, rows):
        """确保表头存在后追加若干行（即时关闭文件，保证落盘）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def _fold_history_path(self, fold):
        """单个 fold 的 history 文件路径，按命名区分。"""
        try:
            tag = int(fold)
            tag = f"{tag:02d}"
        except (TypeError, ValueError):
            tag = str(fold)
        return self.save_dir / f"history_fold{tag}.csv"

    # ------------------------------------------------------------------ #
    def record_epoch(self, observer, fold, epoch, is_best=False,
                     train_loss=None, train_acc=None):
        """记录一个 epoch 的训练集 + 验证集指标，实时追加到该 fold 的 history 文件。

        Args:
            observer: RuntimeObserver（已完成 compute_result / print_result）
            fold: fold 编号
            epoch: epoch 编号
            is_best: 该 epoch 是否为当前 fold 的最佳轮
            train_loss / train_acc: 小样本场景下 episodic 训练损失/准确率，
                用于覆盖 observer 中缺失的 train 指标（时序模型留空即可）
        """
        # ---- 训练集行 ----
        train_pack = self._pack(
            observer.train_metric,
            observer.average_train_loss if train_loss is None else train_loss,
            observer.train_auc,
            observer.train_balance_accuracy,
            self.num_classes,
        )
        if train_acc is not None:
            train_pack["Accuracy"] = _to_float(train_acc)
        row_train = {"fold": fold, "epoch": epoch, "split": "train",
                     "is_best": is_best}
        row_train.update(train_pack)

        # ---- 验证集行 ----
        val_pack = self._pack(
            observer.eval_metric,
            observer.average_eval_loss,
            observer.eval_auc,
            observer.eval_balance_accuracy,
            self.num_classes,
        )
        row_val = {"fold": fold, "epoch": epoch, "split": "val",
                   "is_best": is_best}
        row_val.update(val_pack)

        # ---- 实时追加到该 fold 的 history 文件 ----
        path = self._fold_history_path(fold)
        header = ["fold", "epoch", "split", "is_best"] + self.metric_cols
        self._append_csv(path, header, [row_train, row_val])

        if fold not in self._fold_seen:
            self._fold_seen.add(fold)
            print(f"[history] Fold {fold} 指标历史实时写入: {path}")

    def record_test(self, observer, fold):
        """记录一个 fold 的测试集指标，追加到该 fold 的 history 文件（epoch=test）。"""
        test_pack = self._pack(
            observer.test_metric,
            observer.average_test_loss,
            observer.test_auc,
            observer.test_balance_accuracy,
            self.num_classes,
        )
        row = {"fold": fold, "epoch": "test", "split": "test", "is_best": ""}
        row.update(test_pack)

        path = self._fold_history_path(fold)
        header = ["fold", "epoch", "split", "is_best"] + self.metric_cols
        self._append_csv(path, header, [row])

    def record_best(self, observer, fold):
        """记录一个 fold 最佳轮次的 train + val 指标，实时追加到 best_results.csv。

        依赖 observer.get_best() 已在最佳轮快照 train 指标。
        """
        b = observer.best_dicts
        best_epoch = b.get("epoch", 0)

        train_metric = {
            "Accuracy": b.get("train_Accuracy", 0.0),
            "Precision": b.get("train_Precision", 0.0),
            "Recall": b.get("train_Recall", 0.0),
            "Specificity": b.get("train_Specificity", 0.0),
            "F1": b.get("train_F1", 0.0),
            "CohenKappa": b.get("train_CohenKappa", 0.0),
            "confusionMatrix": b.get("train_confusionMatrix", None),
        }
        train_pack = self._pack(
            train_metric,
            b.get("train_Loss", 0.0),
            b.get("train_AuRoc", 0.0),
            b.get("train_BalanceAccuracy", 0.0),
            self.num_classes,
        )

        val_metric = {
            "Accuracy": b.get("Accuracy", 0.0),
            "Precision": b.get("Precision", 0.0),
            "Recall": b.get("Recall", 0.0),
            "Specificity": b.get("Specificity", 0.0),
            "F1": b.get("F1", 0.0),
            "CohenKappa": b.get("CohenKappa", 0.0),
            "confusionMatrix": b.get("confusionMatrix", None),
        }
        val_pack = self._pack(
            val_metric,
            b.get("val_Loss", 0.0),
            b.get("AuRoc", 0.0),
            b.get("BalanceAccuracy", 0.0),
            self.num_classes,
        )

        row = {"fold": fold, "best_epoch": best_epoch}
        for k, v in train_pack.items():
            row[f"train_{k}"] = v
        for k, v in val_pack.items():
            row[f"val_{k}"] = v
        self.best_rows.append(row)

        # 实时追加到 best_results.csv
        best_path = self.save_dir / "best_results.csv"
        header = (["fold", "best_epoch"]
                  + [f"train_{c}" for c in self.metric_cols]
                  + [f"val_{c}" for c in self.metric_cols])
        self._append_csv(best_path, header, [row])
        if not self._best_seen:
            self._best_seen = True

    # ------------------------------------------------------------------ #
    def save(self):
        """全部 fold 结束后，向 best_results.csv 追加 mean / std 汇总行。

        各 fold 的 history_fold{N}.csv 已在训练过程中实时落盘，无需再写。
        返回 best_results.csv 路径与各 fold history 路径列表。
        """
        best_path = self.save_dir / "best_results.csv"
        rows = list(self.best_rows)
        if rows:
            header = (["fold", "best_epoch"]
                      + [f"train_{c}" for c in self.metric_cols]
                      + [f"val_{c}" for c in self.metric_cols])
            mean_std_rows = []
            for agg in ("mean", "std"):
                agg_row = {"fold": agg, "best_epoch": ""}
                for col in header[2:]:
                    vals = [_to_float(r.get(col, 0.0)) for r in rows]
                    if not vals:
                        agg_row[col] = 0.0
                    elif agg == "mean":
                        agg_row[col] = float(np.mean(vals))
                    else:
                        agg_row[col] = float(np.std(vals))
                mean_std_rows.append(agg_row)
            self._append_csv(best_path, header, mean_std_rows)

        fold_paths = [self._fold_history_path(f) for f in sorted(self._fold_seen)]
        return best_path, fold_paths
