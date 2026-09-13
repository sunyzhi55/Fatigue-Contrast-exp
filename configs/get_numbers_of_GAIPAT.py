import os
import re
from pathlib import Path
"""
输入: root_dir: GAIPAT数据集根目录路径，包含所有的 JSONL 文件
输出: 个体总数和所有 subject_id 的列表
"""
def count_subjects(root_dir):
    pattern = re.compile(r'^(\d+)_[^_]+_[^_]+_[^_]+_[^_]+_[^_]+\.jsonl$')
    subject_ids = set()

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            match = pattern.match(filename)
            if match:
                subject_id = match.group(1)
                subject_ids.add(subject_id)

    return len(subject_ids), sorted(subject_ids)

# 使用示例（请替换为你的实际路径）
root_path = "/data3/wangchangmiao/shenxy/Code/gaze/GAIPAT_Data_20260719"
count, ids = count_subjects(root_path)
print(f"个体总数: {count}")
print(f"所有subject_id: {ids}")