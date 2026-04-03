"""
trajectory_gen/run_api_all.py

Run trajectory generation for all (dataset × length) combos via an
OpenAI-compatible API. All samples within each file are sent concurrently.

Prompt template: filter_infer.py style ("The answer is <X>")
Answer extraction: "The answer is ..." pattern (suits non-thinking models)
Scoring: sub_em / em / f1

Required env vars:
    OPENAI_BASE_URL   e.g. http://host:port/v1
    OPENAI_API_KEY    API key ("dummy" if server doesn't require one)

Optional env vars:
    MODEL             default: DeepSeek-V3.2
    N_ROLLOUTS        generations per sample, default: 1
    TEMPERATURE       default: 0.6
    MAX_TOKENS        default: 8192
    CONCURRENCY       max simultaneous requests, default: 20
    INPUT_DIR         default: data/plain_multihop
    OUTPUT_DIR        default: trajectories
    NUM_SAMPLES       samples per file, default: 1000
    LENGTHS           space-separated list, default: "4096 8192 16384 32768"
    DATASETS          space-separated list, default: "hotpotqa musique 2wikimqa"

Usage:
    OPENAI_BASE_URL=http://host:port/v1 OPENAI_API_KEY=dummy \\
        python trajectory_gen/run_api_all.py
"""

import asyncio
import json
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL    = os.environ.get("OPENAI_BASE_URL")
API_KEY     = os.environ.get("OPENAI_API_KEY", "dummy")
MODEL       = os.environ.get("MODEL", "DeepSeek-V3.2")
N_ROLLOUTS  = int(os.environ.get("N_ROLLOUTS", "1"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.6"))
MAX_TOKENS  = int(os.environ.get("MAX_TOKENS", "8192"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "20"))
INPUT_DIR   = os.environ.get("INPUT_DIR", "data/plain_multihop")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", "trajectories")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "1000"))
LENGTHS     = [int(x) for x in os.environ.get("LENGTHS", "4096 8192 16384 32768").split()]
DATASETS    = os.environ.get("DATASETS", "hotpotqa musique 2wikimqa").split()

PROMPT_TEMPLATE = (
    "The following are given passages.\n{context}\n\n"
    "Answer the question based on the given passages. "
    "Please think step by step before answering. "
    "After thinking, output the final answer in the format of "
    "'The answer is <your answer here>'\n\n"
    "Question: {input}"
)

if not BASE_URL:
    sys.exit("Error: OPENAI_BASE_URL is not set.\nExample: export OPENAI_BASE_URL=http://host:port/v1")


# ── Scoring ───────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p)
    rec  = num_same / len(g)
    return 2 * prec * rec / (prec + rec)


def extract_answer(text: str) -> Optional[str]:
    """Extract from 'The answer is X' format (strips markdown bold)."""
    clean = text.replace("**", "").replace("*", "")
    lower = clean.lower()
    if "the answer is" not in lower:
        return None
    pos = lower.rfind("the answer is")
    raw = clean[pos + len("the answer is"):].strip().lstrip(":").strip()
    raw = re.split(r"\n|\.(?:\s|$)", raw)[0].strip()
    return raw if raw else None


def score_response(text: str, gold_answers: list) -> dict:
    extracted = extract_answer(text)
    if not extracted or not gold_answers:
        return {"extracted_answer": extracted, "sub_em": 0, "em": 0, "f1": 0.0, "is_correct": 0}
    norm_pred = normalize_answer(extracted)
    sub_em = max(int(normalize_answer(g) in norm_pred or norm_pred in normalize_answer(g)) for g in gold_answers)
    em_    = max(int(norm_pred == normalize_answer(g)) for g in gold_answers)
    f1_    = max(_f1(extracted, g) for g in gold_answers)
    return {
        "extracted_answer": extracted,
        "sub_em": sub_em,
        "em": em_,
        "f1": round(f1_, 4),
        "is_correct": sub_em,
    }


# ── Async API ─────────────────────────────────────────────────────────────────

MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))

async def call_api(client, semaphore, prompt: str) -> list[str]:
    """Call API with exponential backoff retry on any error."""
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    n=N_ROLLOUTS,
                )
                return [c.message.content for c in resp.choices]
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"\n[error] gave up after {MAX_RETRIES} attempts: {e}")
                    return [""] * N_ROLLOUTS   # empty placeholder, scored as incorrect
                wait = 2 ** attempt            # 1s, 2s, 4s, 8s, 16s
                print(f"\n[retry {attempt+1}/{MAX_RETRIES-1}] {type(e).__name__}: {e} — retrying in {wait}s")
                await asyncio.sleep(wait)


# ── Per-file processing ───────────────────────────────────────────────────────

async def process_file(client, semaphore, input_file: str, model_tag: str):
    with open(input_file) as f:
        records = [json.loads(line) for line in f]

    prompts = [
        PROMPT_TEMPLATE.format(context=r["context"], input=r["input"])
        for r in records
    ]

    from tqdm.asyncio import tqdm as atqdm
    tasks = [call_api(client, semaphore, p) for p in prompts]
    all_texts = await atqdm.gather(*tasks, desc=f"  {Path(input_file).parent.name}@{Path(input_file).stem.split('_')[-1]}", ncols=80)

    output_records = []
    for record, texts in zip(records, all_texts):
        scored = [score_response(t, record["answers"]) for t in texts]
        num_correct = sum(s["is_correct"] for s in scored)
        output_records.append({
            **record,
            "model": model_tag,
            "trajectories": [{"text": t, **s} for t, s in zip(texts, scored)],
            "num_correct": num_correct,
            "pass_rate": round(num_correct / len(texts), 4) if texts else 0.0,
        })

    # output path: trajectories/{dataset}/{stem}-{model_tag}.jsonl
    p = Path(input_file)
    out_dir = Path(OUTPUT_DIR) / p.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{p.stem}-{model_tag}.jsonl"

    with open(out_path, "w") as f:
        for r in output_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(output_records)
    any_correct = sum(r["num_correct"] > 0 for r in output_records)
    avg_f1 = (
        sum(t["f1"] for r in output_records for t in r["trajectories"])
        / sum(len(r["trajectories"]) for r in output_records)
    )
    print(f"  → {out_path}  |  any-correct: {any_correct}/{total} ({any_correct/total:.1%})  avg-f1: {avg_f1:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    from openai import AsyncOpenAI

    # Collect input files
    combos = []
    for dataset in DATASETS:
        for length in LENGTHS:
            path = f"{INPUT_DIR}/{dataset}/train-num_sample_{NUM_SAMPLES}-max_seq_{length}.jsonl"
            if Path(path).exists():
                combos.append((dataset, length, path))
            else:
                print(f"[skip] {path} not found")

    if not combos:
        sys.exit("No input files found. Check INPUT_DIR and NUM_SAMPLES env vars.")

    model_tag = MODEL.replace("/", "_")

    print(f"Model      : {MODEL}  ({BASE_URL})")
    print(f"n={N_ROLLOUTS}, temp={TEMPERATURE}, max_tokens={MAX_TOKENS}, concurrency={CONCURRENCY}")
    print(f"Jobs       : {len(combos)}  ({', '.join(f'{d}@{l}' for d,l,_ in combos)})")
    print()

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY) as client:
        for i, (dataset, length, input_file) in enumerate(combos, 1):
            print(f"[{i}/{len(combos)}] {dataset} @ {length} tokens  ({input_file})")
            await process_file(client, semaphore, input_file, model_tag)

    print("\nAll done.")


if __name__ == "__main__":
    asyncio.run(main())
