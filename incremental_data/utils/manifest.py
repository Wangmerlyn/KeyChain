#!/usr/bin/env python3
"""
Track and manage incremental data generation progress.

Usage:
    python incremental_data/utils/manifest.py --create
    python incremental_data/utils/manifest.py --status
    python incremental_data/utils/manifest.py --update hotpotqa 4096 generated
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

MANIFEST_FILE = Path(__file__).parent.parent / "manifests" / "generation_manifest.json"

DATASETS = ["hotpotqa", "musique", "2wikimqa"]
LENGTHS = [4096, 8192, 16384, 32768]
STAGES = ["plain", "trajectories", "swift", "filtered", "merged"]
TARGET_SAMPLES = {"existing": 1000, "new": 1500, "total": 2500}


def create_manifest():
    manifest = {
        "created_at": datetime.now().isoformat(),
        "target_samples": TARGET_SAMPLES,
        "datasets": {},
    }

    for dataset in DATASETS:
        manifest["datasets"][dataset] = {}
        for length in LENGTHS:
            manifest["datasets"][dataset][str(length)] = {
                "status": "pending",
                "current_stage": None,
                "stages_completed": [],
                "samples": {
                    "existing": 0,
                    "generated": 0,
                    "trajectories": 0,
                    "filtered": 0,
                },
                "files": {
                    "plain": None,
                    "trajectories": None,
                    "swift": None,
                    "filtered": None,
                    "merged": None,
                },
                "updated_at": None,
            }

    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created manifest: {MANIFEST_FILE}")
    return manifest


def load_manifest():
    if not MANIFEST_FILE.exists():
        print("Manifest not found, creating new one...")
        return create_manifest()

    with open(MANIFEST_FILE) as f:
        return json.load(f)


def save_manifest(manifest):
    manifest["updated_at"] = datetime.now().isoformat()
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


def update_status(dataset, length, stage, samples=None, filepath=None):
    manifest = load_manifest()

    if dataset not in manifest["datasets"]:
        print(f"Error: Unknown dataset {dataset}")
        return
    if str(length) not in manifest["datasets"][dataset]:
        print(f"Error: Unknown length {length}")
        return

    entry = manifest["datasets"][dataset][str(length)]
    entry["current_stage"] = stage

    if stage not in entry["stages_completed"]:
        entry["stages_completed"].append(stage)

    if samples is not None:
        entry["samples"][stage] = samples

    if filepath is not None:
        entry["files"][stage] = filepath

    entry["updated_at"] = datetime.now().isoformat()

    if stage == "merged":
        entry["status"] = "complete"
    elif entry["stages_completed"]:
        entry["status"] = "in_progress"

    save_manifest(manifest)
    print(f"Updated {dataset} @ {length}: stage={stage}, samples={samples}")


def print_status():
    manifest = load_manifest()

    print("=" * 70)
    print("Incremental Data Generation Status")
    print("=" * 70)
    print(f"Target: {TARGET_SAMPLES['total']} samples per dataset/length")
    print(
        f"        ({TARGET_SAMPLES['existing']} existing + {TARGET_SAMPLES['new']} new)"
    )
    print(f"Manifest: {MANIFEST_FILE}")
    print("")

    total_complete = 0
    total_in_progress = 0
    total_pending = 0

    for dataset in DATASETS:
        print(f"\n{dataset.upper()}")
        print("-" * 70)
        for length in LENGTHS:
            entry = manifest["datasets"][dataset][str(length)]
            status = entry["status"]
            stage = entry["current_stage"] or "-"
            samples = entry["samples"]["filtered"]

            if status == "complete":
                symbol = "✓"
                total_complete += 1
            elif status == "in_progress":
                symbol = "⋯"
                total_in_progress += 1
            else:
                symbol = "○"
                total_pending += 1

            print(
                f"  {symbol} {length:5} | {status:12} | Stage: {stage:12} | "
                f"Samples: {samples:4}/{TARGET_SAMPLES['total']}"
            )

    total = total_complete + total_in_progress + total_pending
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(
        f"Complete:    {total_complete}/{total} ({100 * total_complete / total:.1f}%)"
    )
    print(f"In Progress: {total_in_progress}/{total}")
    print(f"Pending:     {total_pending}/{total}")


def main():
    p = argparse.ArgumentParser(
        description="Manage incremental data generation manifest"
    )
    p.add_argument("--create", action="store_true", help="Create new manifest")
    p.add_argument("--status", action="store_true", help="Show current status")
    p.add_argument(
        "--update",
        nargs=3,
        metavar=("DATASET", "LENGTH", "STAGE"),
        help="Update status for dataset/length",
    )
    p.add_argument("--samples", type=int, help="Number of samples (with --update)")
    p.add_argument("--file", help="File path (with --update)")

    args = p.parse_args()

    if args.create:
        create_manifest()
    elif args.status:
        print_status()
    elif args.update:
        dataset, length, stage = args.update
        update_status(dataset, int(length), stage, args.samples, args.file)
    else:
        print_status()


if __name__ == "__main__":
    main()
