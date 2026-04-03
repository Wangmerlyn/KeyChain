"""
generate_trajectories.py

Generate rollout trajectories for SFT from plain multihop JSONL files.
Supports vLLM (local) and OpenAI-compatible API backends.

Usage (vllm):
    python generate_trajectories.py --input_file output/hotpotqa/train-num_sample_1000-max_seq_4096.jsonl

Usage (openai):
    python generate_trajectories.py \\
        --input_file output/hotpotqa/train-num_sample_1000-max_seq_4096.jsonl \\
        --backend openai --model gpt-4o --openai_base_url https://...
"""

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = (
    "/home/wsy0227/qwen14b_2e_1node_16k_2k_FILTERedAGAIN_dis_256bsz_grpo_SUBEM_end-step151"
)
DEFAULT_PROMPT_TEMPLATE = (
    "The following are given passages.\n{context}\n\nQuestion: {input}"
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate rollout trajectories for SFT")
    p.add_argument("--input_file", required=True, help="Path to input JSONL file")
    p.add_argument("--output_dir", default="trajectories", help="Root output directory")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model path (vllm) or name (openai)")
    p.add_argument("--backend", default="vllm", choices=["vllm", "openai"])
    p.add_argument("--tp_size", type=int, default=4, help="Tensor parallel size (vllm only)")
    p.add_argument("--n", type=int, default=4, help="Rollouts per sample")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument(
        "--prompt_template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Python .format(context=..., input=...) template",
    )
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument(
        "--end_idx",
        type=int,
        default=-1,
        help="Exclusive end index; -1 means all (sliced as records[start_idx:])",
    )
    p.add_argument(
        "--openai_base_url", default=None, help="Base URL for OpenAI-compatible API"
    )
    return p.parse_args()


# ── Scoring ───────────────────────────────────────────────────────────────────
# Adapted from LoongRL longcontext_qa_llm_judge.py (rule-based only, no LLM judge).
# NOTE: do NOT use filter_again/judge_utils.exact_match_score for `em` —
#       that function implements substring containment (sub_em), not strict equality.


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def last_boxed_only_string(s: str) -> Optional[str]:
    """Find the last \\boxed{...} in a string and return it including braces."""
    idx = s.rfind("\\boxed")
    if idx < 0:
        return None
    i, num_open, right = idx, 0, None
    while i < len(s):
        if s[i] == "{":
            num_open += 1
        elif s[i] == "}":
            num_open -= 1
            if num_open == 0:
                right = i
                break
        i += 1
    return s[idx : right + 1] if right is not None else None


def remove_boxed(s: str) -> str:
    if s.startswith("\\boxed "):
        return s[len("\\boxed ") :]
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{") : -1]
    return s


def extract_answer(text: str) -> Optional[str]:
    """Extract final answer from <think>...</think>\\boxed{answer} format.

    Returns None if </think> is absent or no \\boxed{} follows it.
    """
    if "</think>" not in text:
        return None
    after_think = text.split("</think>")[-1].strip().replace("*", "")
    boxed = last_boxed_only_string(after_think)
    return remove_boxed(boxed) if boxed is not None else None


def _f1(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)


def compute_rewards(extracted: Optional[str], gold_answers: list) -> dict:
    """Return sub_em, em, f1 — each the max over all gold answers."""
    if not extracted or not gold_answers:
        return {"sub_em": 0, "em": 0, "f1": 0.0}
    norm_pred = normalize_answer(extracted)
    sub_em = max(
        int(normalize_answer(g) in norm_pred or norm_pred in normalize_answer(g))
        for g in gold_answers
    )
    em = max(int(norm_pred == normalize_answer(g)) for g in gold_answers)
    f1 = max(_f1(extracted, g) for g in gold_answers)
    return {"sub_em": sub_em, "em": em, "f1": round(f1, 4)}


def score_trajectory(text: str, gold_answers: list) -> dict:
    """Score a single trajectory text against gold answers."""
    extracted = extract_answer(text)
    rewards = compute_rewards(extracted, gold_answers)
    return {
        "text": text,
        "extracted_answer": extracted,
        **rewards,
        "is_correct": rewards["sub_em"],
    }


# ── Backends ──────────────────────────────────────────────────────────────────


def run_vllm(
    prompts: list,
    model: str,
    tp_size: int,
    n: int,
    temperature: float,
    max_tokens: int,
) -> list:
    """Run vLLM inference. Returns List[List[str]] — n texts per sample."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, tensor_parallel_size=tp_size, gpu_memory_utilization=0.9)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n)
    messages = [[{"role": "user", "content": p}] for p in prompts]
    outputs = llm.chat(messages=messages, sampling_params=params)
    return [[o.text for o in out.outputs] for out in outputs]


def run_openai(
    prompts: list,
    model: str,
    n: int,
    temperature: float,
    max_tokens: int,
    base_url: Optional[str] = None,
) -> list:
    """Run OpenAI-compatible API inference. Returns List[List[str]]."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url) if base_url else OpenAI()
    results = []
    for prompt in prompts:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )
        results.append([c.message.content for c in resp.choices])
    return results


# ── I/O ───────────────────────────────────────────────────────────────────────


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def derive_output_path(input_file: str, output_dir: str, model_tag: str) -> Path:
    """
    input_file: output/hotpotqa/train-num_sample_1000-max_seq_4096.jsonl
    → {output_dir}/hotpotqa/train-num_sample_1000-max_seq_4096-{model_tag}.jsonl
    """
    p = Path(input_file)
    dataset = p.parent.name
    out = Path(output_dir) / dataset
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{p.stem}-{model_tag}.jsonl"


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    records = load_jsonl(args.input_file)
    records = (
        records[args.start_idx :]
        if args.end_idx == -1
        else records[args.start_idx : args.end_idx]
    )
    print(f"Loaded {len(records)} records from {args.input_file}")

    prompts = [
        args.prompt_template.format(context=r["context"], input=r["input"])
        for r in records
    ]

    print(
        f"Backend: {args.backend} | model: {Path(args.model).name} "
        f"| n={args.n} | temp={args.temperature}"
    )
    if args.backend == "vllm":
        all_texts = run_vllm(
            prompts, args.model, args.tp_size, args.n,
            args.temperature, args.max_tokens,
        )
    else:
        all_texts = run_openai(
            prompts, args.model, args.n, args.temperature,
            args.max_tokens, args.openai_base_url,
        )

    model_tag = Path(args.model).name
    output_records = []
    for record, texts in zip(records, all_texts):
        trajectories = [score_trajectory(t, record["answers"]) for t in texts]
        num_correct = sum(t["is_correct"] for t in trajectories)
        output_records.append({
            **record,
            "model": model_tag,
            "trajectories": trajectories,
            "num_correct": num_correct,
            "pass_rate": round(num_correct / len(trajectories), 4) if trajectories else 0.0,
        })

    out_path = derive_output_path(args.input_file, args.output_dir, model_tag)
    with open(out_path, "w") as f:
        for r in output_records:
            f.write(json.dumps(r) + "\n")

    total = len(output_records)
    any_correct = sum(r["num_correct"] > 0 for r in output_records)
    print(f"Saved {total} records → {out_path}")
    print(f"Any-correct: {any_correct}/{total} ({any_correct / total:.1%})")


if __name__ == "__main__":
    main()
