"""
sft_data/clean_traces.py

Two-step pipeline for LoongRL-14b trajectory files:
  1. Filter: keep best sub_em==1 trajectory per query (highest f1, first on tie).
             Drop queries where all trajectories have sub_em==0.
  2. Clean:  raw text is "reasoning...</think>\\boxed{answer}"
             output is  "<think>reasoning...</think>answer"

Usage:
    python sft_data/clean_traces.py \
        --input trajectories/hotpotqa/train-num_sample_1000-max_seq_4096-qwen14b*.jsonl \
        --output_dir sft_data_cleaned/hotpotqa/

    # model_tag written to 'model' field (default: LoongRL-14b)
    python sft_data/clean_traces.py --input ... --output_dir ... --model_tag LoongRL-14b
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

# Reuse brace-aware boxed helpers from trajectory generation
sys.path.insert(0, str(Path(__file__).parent.parent))
from trajectory_gen.generate_trajectories import last_boxed_only_string, remove_boxed

PROMPT_TEMPLATE = "The following are given passages.\n{context}\n\nQuestion: {input}"


# ── Format cleaning ───────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Transform raw model output to clean <think>…</think>answer format.

    Input:  "reasoning...</think>\\boxed{answer}"   (no leading <think>)
    Output: "<think>reasoning...</think>answer"

    Uses rsplit to split on the LAST </think>, preserving any intermediate
    reasoning in multi-</think> traces. Handles nested braces in \\boxed{}.
    """
    think_part, after = text.rsplit("</think>", 1)
    after = after.strip()
    boxed = last_boxed_only_string(after)
    content = remove_boxed(boxed) if boxed is not None else after
    return f"<think>{think_part}</think>{content}"


# ── Filter ────────────────────────────────────────────────────────────────────

def select_best(record: dict) -> dict | None:
    """Return the best trajectory from a record, or None if all incorrect.

    'Best' = highest f1 among sub_em==1 trajectories; first on tie.
    Returns None if no trajectory has sub_em==1.
    """
    correct = [t for t in record["trajectories"] if t.get("sub_em") == 1]
    if not correct:
        return None
    # max() is stable: first element wins on tie
    return max(correct, key=lambda t: t["f1"])


# ── File processing ───────────────────────────────────────────────────────────

def derive_output_stem(input_stem: str, model_tag: str) -> str:
    """Strip checkpoint suffix and append model_tag + '-cleaned'.

    'train-num_sample_1000-max_seq_4096-qwen14b_...step151'
    → 'train-num_sample_1000-max_seq_4096-LoongRL-14b-cleaned'
    """
    match = re.match(r"(.*max_seq_\d+)", input_stem)
    if not match:
        raise ValueError(
            f"Input stem '{input_stem}' must contain 'max_seq_<number>'. "
            f"Expected: 'train-num_sample_*-max_seq_*-...'"
        )
    return f"{match.group(1)}-{model_tag}-cleaned"


def process_file(input_path: str, output_dir: str, model_tag: str = "LoongRL-14b") -> Path:
    """Filter and clean one trajectory JSONL file. Returns output Path."""
    in_path = Path(input_path)
    dataset = in_path.parent.name
    out_stem = derive_output_stem(in_path.stem, model_tag)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_stem}.jsonl"

    n_total = n_kept = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line_num, line in enumerate(fin, start=1):
            try:
                record = json.loads(line)
                n_total += 1
                best = select_best(record)
                if best is None:
                    continue
                cleaned_text = clean_text(best["text"])
                user_content = PROMPT_TEMPLATE.format(
                    context=record["context"], input=record["input"]
                )
                row = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": cleaned_text},
                    ],
                    "sub_em": best["sub_em"],
                    "em": best["em"],
                    "f1": best["f1"],
                    "is_correct": best["is_correct"],
                    "extracted_answer": best["extracted_answer"],
                    "answers": record["answers"],
                    "source_index": record["source_index"],
                    "dataset": dataset,
                    "length": record["length"],
                    "model": model_tag,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_kept += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  WARNING: skipping line {line_num}: {e}", file=sys.stderr)

    n_dropped = n_total - n_kept
    print(
        f"  {in_path.name}\n"
        f"    queries: {n_total} total  →  {n_kept} kept  ({n_dropped} dropped)\n"
        f"    → {out_path}"
    )
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Filter + clean LoongRL-14b trajectory files to ms-swift format"
    )
    p.add_argument("--input", required=True,
                   help="Input trajectory JSONL path (glob supported)")
    p.add_argument("--output_dir", required=True,
                   help="Output directory")
    p.add_argument("--model_tag", default="LoongRL-14b",
                   help="Written to 'model' field (default: LoongRL-14b)")
    return p.parse_args()


def main():
    args = parse_args()
    files = glob.glob(args.input)
    if not files:
        sys.exit(f"No files matched: {args.input}")
    for f in sorted(files):
        process_file(f, args.output_dir, args.model_tag)


if __name__ == "__main__":
    main()
