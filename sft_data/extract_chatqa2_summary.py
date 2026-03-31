"""
sft_data/extract_chatqa2_summary.py

Download the nvidia/ChatQA2-Long-SFT-data long_sft split from HuggingFace,
filter for summarization tasks, and save as ms-swift JSONL.

Filtering: records where `question` (lowercase) contains any of the summary keywords.
Default keywords cover the observed pattern "Can you write an appropriate summary..."
and related phrasings from the dataset.

Usage:
    python sft_data/extract_chatqa2_summary.py

    # Custom output dir or keywords:
    python sft_data/extract_chatqa2_summary.py \
        --output_dir sft_data/chatqa2_summary/ \
        --keywords "can you write an appropriate summary" "write a summary" "summarize"
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_KEYWORDS = [
    "can you write an appropriate summary",
    "write a summary",
    "summarize the above",
    "provide a summary",
    "write an appropriate summary",
]

DEFAULT_OUTPUT_DIR = "sft_data/chatqa2_summary"
DATASET_NAME = "nvidia/ChatQA2-Long-SFT-data"
CONFIG_NAME = "long_sft"


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract summarization data from ChatQA2-Long-SFT-data"
    )
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--keywords",
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help="Lowercase substrings to match in the question field",
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Which split to process (default: train)",
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming mode (default: True, avoids downloading full dataset)",
    )
    return p.parse_args()


def is_summary(question: str, keywords: list) -> bool:
    q_lower = question.lower()
    return any(kw in q_lower for kw in keywords)


def to_swift_row(record: dict) -> dict:
    """Convert a ChatQA2 record to ms-swift messages format."""
    answer = record["answer"]
    # answer may be a string or list; normalise to string
    if isinstance(answer, list):
        answer = answer[0] if answer else ""

    # The question field has the format:
    #   "User: <context>\n\nQ: <question>\n\nAssistant:"
    # Strip the leading "User: " and the trailing "\n\nAssistant:" so the
    # user message contains only the clean context + question.
    question = record["question"]

    # Strip leading "User: " prefix
    if question.startswith("User: "):
        question = question[len("User: "):]

    # Strip trailing "\n\nAssistant:" suffix
    for suffix in ("\n\nAssistant:", "\nAssistant:", "Assistant:"):
        if question.endswith(suffix):
            question = question[: -len(suffix)].rstrip()
            break

    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "dataset": "chatqa2_summary",
        "source": DATASET_NAME,
        "word_count": record.get("word count") or record.get("word_count"),
    }


def main():
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Please install the `datasets` library: pip install datasets")

    print(f"Loading {DATASET_NAME} / {CONFIG_NAME} / {args.split} (streaming)...")
    print(f"Keywords: {args.keywords}")
    print()

    ds = load_dataset(
        DATASET_NAME,
        CONFIG_NAME,
        split=args.split,
        streaming=True,
        trust_remote_code=True,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"long_sft_{args.split}_summary.jsonl"

    n_seen = n_kept = 0
    with open(out_path, "w") as f:
        for record in ds:
            n_seen += 1
            if is_summary(record.get("question", ""), args.keywords):
                row = to_swift_row(record)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_kept += 1

            if n_seen % 10_000 == 0:
                print(f"  scanned {n_seen:,} / kept {n_kept:,}", end="\r")

    print(f"\nDone: scanned {n_seen:,} records, kept {n_kept:,} summary records")
    print(f"Output: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
