"""
sft_data/convert_to_swift.py

Convert LoongRL-14b trajectory JSONL to ms-swift SFT format.
One input record with n trajectories → n output rows.

Usage:
    python sft_data/convert_to_swift.py \
        --input trajectories/hotpotqa/train-num_sample_1000-max_seq_4096-qwen14b*.jsonl \
        --output_dir sft_data/hotpotqa/ \
        --model_tag LoongRL-14b
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

PROMPT_TEMPLATE = "The following are given passages.\n{context}\n\nQuestion: {input}"


def derive_output_stem(input_stem: str, model_tag: str) -> str:
    """Strip checkpoint suffix and append model_tag.

    'train-num_sample_1000-max_seq_4096-qwen14b_...step151'
    → 'train-num_sample_1000-max_seq_4096-LoongRL-14b-swift'
    """
    match = re.match(r"(.*max_seq_\d+)", input_stem)
    if not match:
        raise ValueError(
            f"Input stem '{input_stem}' must contain 'max_seq_<number>' pattern. "
            f"Expected format: 'train-num_sample_*-max_seq_*-...'"
        )
    return f"{match.group(1)}-{model_tag}-swift"


def convert_record(record: dict, dataset: str, model_tag: str,
                   skip_empty: bool = True) -> list:
    """Convert one trajectory record into a list of ms-swift rows."""
    user_content = PROMPT_TEMPLATE.format(
        context=record["context"],
        input=record["input"],
    )
    rows = []
    for traj in record["trajectories"]:
        text = traj["text"]
        if skip_empty and not text.strip():
            continue
        rows.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": text},
            ],
            "is_correct": traj["is_correct"],
            "sub_em": traj["sub_em"],
            "em": traj["em"],
            "f1": traj["f1"],
            "extracted_answer": traj["extracted_answer"],
            "answers": record["answers"],
            "source_index": record["source_index"],
            "dataset": dataset,
            "length": record["length"],
            "model": model_tag,
        })
    return rows


def convert_file(input_path: str, output_dir: str, model_tag: str,
                 skip_empty: bool = True) -> Path:
    """Convert one trajectory JSONL file. Returns output path."""
    input_path = Path(input_path)
    dataset = input_path.parent.name
    out_stem = derive_output_stem(input_path.stem, model_tag)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_stem}.jsonl"

    total_in, total_out = 0, 0
    with open(input_path) as fin, open(out_path, "w") as fout:
        for line_num, line in enumerate(fin, start=1):
            try:
                record = json.loads(line)
                total_in += 1
                rows = convert_record(record, dataset=dataset, model_tag=model_tag,
                                       skip_empty=skip_empty)
                for row in rows:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_out += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  WARNING: skipping line {line_num}: {e}", file=sys.stderr)
                continue

    print(f"  {input_path.name} → {out_path.name}  "
          f"({total_in} records → {total_out} rows)")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Convert trajectory JSONL to ms-swift format")
    p.add_argument("--input", required=True,
                   help="Input file path (glob supported, use quotes)")
    p.add_argument("--output_dir", required=True,
                   help="Output directory (dataset subdirectory auto-created)")
    p.add_argument("--model_tag", default="LoongRL-14b",
                   help="Model tag written to 'model' field (default: LoongRL-14b)")
    p.add_argument("--no_skip_empty", action="store_true",
                   help="Keep trajectories with empty text")
    return p.parse_args()


def main():
    args = parse_args()
    files = glob.glob(args.input)
    if not files:
        sys.exit(f"No files matched: {args.input}")
    for f in sorted(files):
        convert_file(f, args.output_dir, args.model_tag,
                     skip_empty=not args.no_skip_empty)


if __name__ == "__main__":
    main()
