"""
trajectory_gen/run_vllm_all.py

Run trajectory generation for all (dataset × length) combos with a single
vLLM model load. Much faster than spawning one process per combo.

Usage:
    python trajectory_gen/run_vllm_all.py \
        --model /path/to/model \
        --tp_size 8 \
        --lengths 4096 8192 16384 32768

All output goes to trajectories/{dataset}/{stem}-{model_tag}.jsonl
"""

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Optional

# reuse scoring helpers from generate_trajectories
import sys
sys.path.insert(0, str(Path(__file__).parent))
from generate_trajectories import (
    score_trajectory,
    derive_output_path,
    DEFAULT_MODEL,
    DEFAULT_PROMPT_TEMPLATE,
)

DATASETS = ["hotpotqa", "musique", "2wikimqa"]


def parse_args():
    p = argparse.ArgumentParser(description="Batch vLLM trajectory generation (single model load)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tp_size", type=int, default=8)
    p.add_argument("--n", type=int, default=4, help="Rollouts per sample")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    p.add_argument(
        "--lengths", type=int, nargs="+",
        default=[4096, 8192, 16384, 32768],
        help="Context lengths to process",
    )
    p.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        help="Datasets to process",
    )
    p.add_argument("--input_dir", default="output")
    p.add_argument("--output_dir", default="trajectories")
    p.add_argument("--num_samples", type=int, default=1000,
                   help="Samples per file (used to locate input filename)")
    p.add_argument(
        "--max_model_len", type=int, default=None,
        help=(
            "Override vLLM max_model_len. Defaults to max(lengths) + max_tokens "
            "so that long-context inputs don't trigger 'exceeds max model len' errors. "
            "E.g. 32k input + 4096 output = 36864, but model config says 32768 → set 40960."
        ),
    )
    return p.parse_args()


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def process_file(input_file, llm, sampling_params, prompt_template, output_dir, model_tag):
    records = load_jsonl(input_file)
    print(f"  Loaded {len(records)} records")

    prompts = [
        prompt_template.format(context=r["context"], input=r["input"])
        for r in records
    ]
    messages = [[{"role": "user", "content": p}] for p in prompts]

    outputs = llm.chat(messages=messages, sampling_params=sampling_params)

    output_records = []
    for record, out in zip(records, outputs):
        texts = [o.text for o in out.outputs]
        trajectories = [score_trajectory(t, record["answers"]) for t in texts]
        num_correct = sum(t["is_correct"] for t in trajectories)
        output_records.append({
            **record,
            "model": model_tag,
            "trajectories": trajectories,
            "num_correct": num_correct,
            "pass_rate": round(num_correct / len(trajectories), 4) if trajectories else 0.0,
        })

    out_path = derive_output_path(input_file, output_dir, model_tag)
    with open(out_path, "w") as f:
        for r in output_records:
            f.write(json.dumps(r) + "\n")

    any_correct = sum(r["num_correct"] > 0 for r in output_records)
    total = len(output_records)
    print(f"  Saved → {out_path}  |  any-correct: {any_correct}/{total} ({any_correct/total:.1%})")
    return out_path


def main():
    args = parse_args()
    model_tag = Path(args.model).name

    # Collect all input files upfront so we can report what will run
    combos = []
    for dataset in args.datasets:
        for length in args.lengths:
            path = f"{args.input_dir}/{dataset}/train-num_sample_{args.num_samples}-max_seq_{length}.jsonl"
            if Path(path).exists():
                combos.append((dataset, length, path))
            else:
                print(f"[skip] {path} not found")

    if not combos:
        print("No input files found. Check --input_dir and --num_samples.")
        return

    print(f"Model  : {args.model}")
    print(f"TP size: {args.tp_size}")
    print(f"n={args.n}, temp={args.temperature}, max_tokens={args.max_tokens}")
    print(f"Jobs   : {len(combos)} ({', '.join(f'{d}@{l}' for d,l,_ in combos)})")
    print()

    # Load model ONCE
    # max_model_len: auto = max(lengths) + max_tokens to avoid vLLM rejecting
    # requests where input + output exceeds the model's config value (e.g. 32k
    # input + 4096 output = 36864 > 32768). Passing a larger value here tells
    # vLLM to allocate KV cache for that window; quality on the extra tokens is
    # the model's business, but it won't crash or silently truncate.
    max_model_len = args.max_model_len or (max(args.lengths) + args.max_tokens)
    print(f"Loading model... (max_model_len={max_model_len})")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.n,
    )
    print("Model loaded.\n")

    # Process all combos
    for i, (dataset, length, input_file) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] {dataset} @ {length} tokens")
        process_file(input_file, llm, sampling_params, args.prompt_template, args.output_dir, model_tag)
        print()

    print("All done.")


if __name__ == "__main__":
    main()
