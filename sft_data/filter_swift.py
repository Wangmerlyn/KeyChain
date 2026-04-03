"""
sft_data/filter_swift.py

Filter ms-swift SFT JSONL: keep one best trajectory per query.

Selection rule:
  - Among trajectories with sub_em == 1, pick the one with highest f1.
  - Ties in f1 are broken by first occurrence.
  - If all trajectories have sub_em == 0, the query is dropped.

Usage:
    python sft_data/filter_swift.py \
        --input sft_data/hotpotqa/train-num_sample_1000-max_seq_4096-LoongRL-14b-swift.jsonl

    python sft_data/filter_swift.py \
        --input sft_data/hotpotqa/train-num_sample_1000-max_seq_4096-LoongRL-14b-swift.jsonl \
        --output_dir sft_data/hotpotqa/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def select_best_trajectory(rows: list) -> dict | None:
    """Return the best trajectory row from a group, or None if all incorrect.

    'Best' = highest f1 among sub_em==1 rows; first on ties.
    Returns None if rows is empty or all have sub_em==0.
    """
    correct = [r for r in rows if r.get("sub_em") == 1]
    if not correct:
        return None
    # max() is stable: returns first element among ties (left-to-right scan)
    return max(correct, key=lambda r: r["f1"])


def filter_file(input_path: str, output_dir: str | None = None) -> Path:
    """Filter one swift JSONL file. Returns the output Path."""
    in_path = Path(input_path)
    out_dir = Path(output_dir) if output_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}-filtered.jsonl"

    # Group rows by source_index, preserving insertion order
    groups: dict = defaultdict(list)
    with open(in_path) as f:
        for line in f:
            row = json.loads(line)
            groups[row["source_index"]].append(row)

    n_total = len(groups)
    n_kept = 0
    with open(out_path, "w") as f:
        for rows in groups.values():
            best = select_best_trajectory(rows)
            if best is not None:
                f.write(json.dumps(best, ensure_ascii=False) + "\n")
                n_kept += 1

    n_dropped = n_total - n_kept
    print(
        f"  {in_path.name}\n"
        f"    queries: {n_total} total  →  {n_kept} kept  ({n_dropped} dropped)\n"
        f"    → {out_path.name}"
    )
    return out_path


def parse_args():
    p = argparse.ArgumentParser(
        description="Filter ms-swift JSONL to best trajectory per query"
    )
    p.add_argument("--input", required=True, help="Path to input swift JSONL")
    p.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: same directory as input file)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not Path(args.input).exists():
        sys.exit(f"File not found: {args.input}")
    filter_file(args.input, args.output_dir)


if __name__ == "__main__":
    main()
