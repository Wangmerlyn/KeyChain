#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
filter_again_from_list.py

• 针对 train_files 列表中的 parquet，读取对应 eval_results/*_results.jsonl
• 统计 avg_acc 分布
• 剔除 avg_acc == 1.0 的样本，保存 *_filter_again.parquet
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

# --------------------------------------------------
# >>> 只需根据实际情况修改下面三处变量 <<<
# --------------------------------------------------
dataset_prefix = "/mnt/longcontext/models/siyuan"
RESULT_DIR = Path("eval_results_420")  # 与评测脚本保持一致

# train_files = [
#     f"{dataset_prefix}/rl_datasets/rl_three/system/hotpotqa_qwen_filtered_start_idx0_end_idx2500_seq16384/train.parquet",
#     f"{dataset_prefix}/rl_datasets/rl_three/system/hotpotqa_filtered-uuid_default_8-numchain_64-numhop-4-distractor_256_start_idx2500_end_idx5000_seq16384/train.parquet",
#     f"{dataset_prefix}/rl_datasets/rl_three/system/musique_qwen_filtered_start_idx0_end_idx2500_seq16384/train.parquet",
#     f"{dataset_prefix}/rl_datasets/rl_three/system/musique_filtered-uuid_default_8-numchain_64-numhop-4-distractor_256_start_idx2500_end_idx5000_seq16384/train.parquet",
#     f"{dataset_prefix}/rl_datasets/rl_three/system/2wikimqa_qwen_filtered_start_idx0_end_idx2500_seq16384/train.parquet",
#     f"{dataset_prefix}/rl_datasets/rl_three/system/2wikimqa_filtered-uuid_default_8-numchain_64-numhop-4-distractor_256_start_idx2500_end_idx5000_seq16384/train.parquet",
# ]
train_files = [
    f"{dataset_prefix}/rl_datasets/rl_three/system/musique_qwen_filtered_start_idx5000_end_idx6750_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/musique_filtered-uuid_default_8-numchain_64-numhop-4-distractor_256_start_idx6750_end_idx8500_seq16384/train.parquet",
]
# --------------------------------------------------


def read_avg_list(results_path: Path) -> List[float]:
    """把 *_results.jsonl 的 avg_acc 读出来"""
    avg_list = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "avg_acc" not in obj:
                raise KeyError(f"avg_acc 字段缺失: {results_path}")
            avg_list.append(obj["avg_acc"])
    return avg_list


def main():
    for parquet_str in train_files:
        parquet_path = Path(parquet_str)

        # --------- 根据 “上级目录名_文件名” 拼 tag ---------
        tag = f"{parquet_path.parent.name}_{parquet_path.stem}"
        results_path = RESULT_DIR / f"{tag}_results.jsonl"
        if not results_path.exists():
            print(f"\n✗ 找不到 results: {results_path}   (跳过该文件)")
            continue

        print(f"\n▶ 处理 {parquet_path}")
        avg_list = read_avg_list(results_path)
        ser = pd.Series(avg_list, name="avg_acc")

        # --------- 统计分布 ---------
        print("  avg_acc 分布：")
        for acc, cnt in ser.value_counts().sort_index().items():
            print(f"    {acc:>5.3f} → {cnt}")

        # --------- 过滤 & 保存 ---------
        df = pd.read_parquet(parquet_path)
        assert len(df) == len(ser), (
            f"长度不一致：parquet({len(df)}) vs results({len(ser)})"
        )

        keep_mask = ser < 1.0  # 非 100% 正确
        df_filtered = df[keep_mask].reset_index(drop=True)

        out_path = parquet_path.with_name(
            f"{parquet_path.stem}_420_filter_again.parquet"
        )
        df_filtered.to_parquet(out_path, index=False)
        print(f"  ✓ 已保存: {out_path.name}   (剩余 {len(df_filtered)} 条)")


if __name__ == "__main__":
    main()
