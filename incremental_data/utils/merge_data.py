#!/usr/bin/env python3
"""
Merge existing 1k samples with new 1.5k incremental samples.
Creates combined 2.5k files for each dataset and length.

Usage:
    python incremental_data/utils/merge_data.py \
        --existing_dir sft_data \
        --incremental_dir sft_data_incremental \
        --output_dir sft_data_merged
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge existing 1k samples with incremental 1.5k samples"
    )
    p.add_argument(
        "--existing_dir",
        required=True,
        help="Directory containing existing 1k samples (e.g., sft_data/)",
    )
    p.add_argument(
        "--incremental_dir",
        required=True,
        help="Directory containing incremental 1.5k samples (e.g., sft_data_incremental/)",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for merged 2.5k samples (e.g., sft_data_merged/)",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpotqa", "musique", "2wikimqa"],
        help="Datasets to merge",
    )
    p.add_argument(
        "--lengths",
        nargs="+
        type=int,
        default=[4096, 8192, 16384, 32768],
        help="Context lengths to merge",
    )
    return p.parse_args()


def load_jsonl(path: Path) -> list:
    records = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping line {line_num} in {path}: {e}")
    return records


def save_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def derive_output_filename(existing_stem: str) -> str:
    return existing_stem.replace("num_sample_1000", "num_sample_2500")


def merge_files(
    existing_path: Path,
    incremental_path: Path,
    output_path: Path,
) -> dict:
    print(f"  Loading existing: {existing_path}")
    existing_records = load_jsonl(existing_path)
    print(f"    → {len(existing_records)} records")

    print(f"  Loading incremental: {incremental_path}")
    incremental_records = load_jsonl(incremental_path)
    print(f"    → {len(incremental_records)} records")

    merged = existing_records + incremental_records
    print(f"  Merged total: {len(merged)} records")

    save_jsonl(merged, output_path)
    print(f"  Saved to: {output_path}")

    return {
        "existing": len(existing_records),
        "incremental": len(incremental_records),
        "merged": len(merged),
    }


def main():
    args = parse_args()

    existing_dir = Path(args.existing_dir)
    incremental_dir = Path(args.incremental_dir)
    output_dir = Path(args.output_dir)

    if not existing_dir.exists():
        sys.exit(f"Existing directory not found: {existing_dir}")
    if not incremental_dir.exists():
        sys.exit(f"Incremental directory not found: {incremental_dir}")

    print("=" * 60)
    print("Merging Existing 1k + Incremental 1.5k → Combined 2.5k")
    print("=" * 60)
    print(f"Existing:    {existing_dir}")
    print(f"Incremental: {incremental_dir}")
    print(f"Output:      {output_dir}")
    print("")

    stats_all = []

    for dataset in args.datasets:
        for length in args.lengths:
            print(f"\n[{dataset} @ {length}]")

            existing_pattern = f"{dataset}/train-num_sample_1000-max_seq_{length}-*-filtered.jsonl"
            incremental_pattern = f"{dataset}/train-num_sample_1500-max_seq_{length}-*-filtered.jsonl"

            existing_files = list(existing_dir.glob(existing_pattern))
            incremental_files = list(incremental_dir.glob(incremental_pattern))

            if not existing_files:
                print(f"  Warning: No existing files found for {dataset} @ {length}")
                continue
            if not incremental_files:
                print(f"  Warning: No incremental files found for {dataset} @ {length}")
                continue

            existing_path = existing_files[0]
            incremental_path = incremental_files[0]

            output_filename = derive_output_filename(existing_path.name)
            output_path = output_dir / dataset / output_filename

            try:
                stats = merge_files(existing_path, incremental_path, output_path)
                stats_all.append(
                    {
                        "dataset": dataset,
                        "length": length,
                        **stats,
                    }
                )
            except Exception as e:
                print(f"  Error merging {dataset} @ {length}: {e}")
                continue

    print("\n" + "=" * 60)
    print("Merge Summary")
    print("=" * 60)
    total_existing = sum(s["existing"] for s in stats_all)
    total_incremental = sum(s["incremental"] for s in stats_all)
    total_merged = sum(s["merged"] for s in stats_all)

    print(f"Total files processed: {len(stats_all)}")
    print(f"Total existing records:    {total_existing:,}")
    print(f"Total incremental records: {total_incremental:,}")
    print(f"Total merged records:      {total_merged:,}")
    print("")
    print("Output directory:", output_dir)
    print("")
    print("Per-file breakdown:")
    for stat in stats_all:
        print(
            f"  {stat['dataset']:12} @ {stat['length']:5}: "
            f"{stat['existing']:4} + {stat['incremental']:4} = {stat['merged']:4}"
        )


if __name__ == "__main__":
    main()
