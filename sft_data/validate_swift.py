"""
sft_data/validate_swift.py

Two-layer validation for ms-swift SFT JSONL files:
  Layer 1 — Rule checks (fast, no ms-swift needed)
  Layer 2 — AutoPreprocessor dry-run (ms-swift 4.x)

Usage:
    python sft_data/validate_swift.py path/to/file-swift.jsonl
    python sft_data/validate_swift.py path/to/file-swift.jsonl --full   # check all rows

Exit code: 0 = pass, 1 = fail
"""

import argparse
import json
import sys
from pathlib import Path


# ── Layer 1: Rule checks ──────────────────────────────────────────────────────

def validate_row_rules(row: dict) -> list:
    """Return list of error strings; empty list = valid."""
    errors = []
    if "messages" not in row:
        errors.append("missing 'messages' field")
        return errors
    msgs = row["messages"]
    if not isinstance(msgs, list) or len(msgs) == 0:
        errors.append("'messages' must be a non-empty list")
        return errors
    for i, msg in enumerate(msgs):
        if "role" not in msg or "content" not in msg:
            errors.append(f"message[{i}] missing 'role' or 'content'")
        elif not msg["content"]:
            errors.append(f"message[{i}] has empty content")
    if msgs and msgs[-1].get("role") != "assistant":
        errors.append(f"last message role must be 'assistant', got '{msgs[-1].get('role')}'")
    return errors


def validate_file_rules(path: str) -> tuple[int, int]:
    """Run rule checks on all rows. Returns (n_errors, n_rows)."""
    n_errors, n_rows = 0, 0
    with open(path) as f:
        for i, line in enumerate(f, start=1):
            row = json.loads(line)
            n_rows += 1
            errors = validate_row_rules(row)
            for e in errors:
                print(f"  [rule] line {i}: {e}")
                n_errors += 1
    return n_errors, n_rows


# ── Layer 2: AutoPreprocessor dry-run ─────────────────────────────────────────

def validate_file_swift(path: str, sample_size: int = 100) -> bool:
    """Run AutoPreprocessor on up to sample_size rows. Returns True if passes."""
    try:
        from swift.dataset import AutoPreprocessor
        from datasets import Dataset
    except ImportError:
        print("  [swift] ms-swift not installed — skipping AutoPreprocessor check")
        return True

    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= sample_size:
                break

    try:
        preprocessor = AutoPreprocessor()
        preprocessor(Dataset.from_list(rows))
        print(f"  [swift] AutoPreprocessor: OK ({len(rows)} rows sampled)")
        return True
    except Exception as e:
        print(f"  [swift] AutoPreprocessor FAILED: {e}")
        return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Validate ms-swift SFT JSONL format")
    p.add_argument("file", help="Path to swift JSONL file to validate")
    p.add_argument("--full", action="store_true",
                   help="Run AutoPreprocessor on all rows (default: 100 sample)")
    return p.parse_args()


def main():
    args = parse_args()
    path = args.file

    if not Path(path).exists():
        sys.exit(f"File not found: {path}")

    print(f"Validating: {path}")

    # Layer 1
    print("Layer 1: rule checks...")
    n_errors, n_rows = validate_file_rules(path)
    if n_errors:
        print(f"  FAIL: {n_errors} rule errors in {n_rows} rows")
        sys.exit(1)
    print(f"  PASS: {n_rows} rows, no rule errors")

    # Layer 2
    # n_rows is available from Layer 1 — used for --full mode
    print("Layer 2: ms-swift AutoPreprocessor...")
    sample_size = n_rows if args.full else 100
    ok = validate_file_swift(path, sample_size=sample_size)
    if not ok:
        print("  FAIL: AutoPreprocessor rejected the format")
        sys.exit(1)

    print(f"\n✓ {Path(path).name} passed all checks")


if __name__ == "__main__":
    main()
