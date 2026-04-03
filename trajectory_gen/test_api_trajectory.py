"""
test_api_trajectory.py

Quick test: run N_SAMPLES from hotpotqa 8k through an OpenAI-compatible API
and score responses with rule-based rewards (sub_em / em / f1).

All API calls are concurrent via asyncio.

Required env vars:
    OPENAI_BASE_URL   e.g. http://host:port/v1
    OPENAI_API_KEY    API key (use "dummy" if the server doesn't require one)
    MODEL             model name served at that endpoint (default: DeepSeek-V3.2)

Optional env vars:
    N_SAMPLES     number of input records to test (default: 10)
    N_ROLLOUTS    generations per sample (default: 1)
    TEMPERATURE   sampling temperature (default: 0.6)
    MAX_TOKENS    max tokens per generation (default: 4096)
    INPUT_FILE    path to input JSONL (default: output/hotpotqa/train-num_sample_1000-max_seq_8192.jsonl)
    OUTPUT_FILE   where to save results (default: auto-derived from INPUT_FILE)

Usage:
    OPENAI_BASE_URL=http://host:port/v1 OPENAI_API_KEY=dummy python test_api_trajectory.py
"""

import asyncio
import json
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path

# ── Config (all from env, nothing hardcoded) ──────────────────────────────────

BASE_URL    = os.environ.get("OPENAI_BASE_URL")
API_KEY     = os.environ.get("OPENAI_API_KEY", "dummy")
MODEL       = os.environ.get("MODEL", "DeepSeek-V3.2")
N_SAMPLES   = int(os.environ.get("N_SAMPLES", "10"))
N_ROLLOUTS  = int(os.environ.get("N_ROLLOUTS", "1"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.6"))
MAX_TOKENS  = int(os.environ.get("MAX_TOKENS", "4096"))
INPUT_FILE  = os.environ.get(
    "INPUT_FILE",
    "output/hotpotqa/train-num_sample_1000-max_seq_8192.jsonl",
)

# Prompt template from filter_infer.py
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


def extract_answer(text: str):
    """Extract answer from 'The answer is X' format (case-insensitive).

    Handles markdown bold (**X**), leading colons, and 'The final answer is' variants.
    """
    # Strip markdown so '**the answer is:**' becomes 'the answer is:'
    clean = text.replace("**", "").replace("*", "")
    lower = clean.lower()
    if "the answer is" not in lower:
        return None
    pos = lower.rfind("the answer is")
    raw = clean[pos + len("the answer is"):].strip().lstrip(":").strip()
    # Take only the first sentence / line (stop at newline or period followed by space/end)
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

async def call_api(client, prompt: str) -> list[str]:
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        n=N_ROLLOUTS,
    )
    return [c.message.content for c in resp.choices]


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    from openai import AsyncOpenAI

    # Load data
    with open(INPUT_FILE) as f:
        records = [json.loads(line) for line in f]
    records = records[:N_SAMPLES]
    print(f"Loaded {len(records)} records from {INPUT_FILE}")
    print(f"Model: {MODEL} | base_url: {BASE_URL} | n={N_ROLLOUTS} | temp={TEMPERATURE}")
    print(f"Running {len(records)} concurrent API calls...\n")

    prompts = [PROMPT_TEMPLATE.format(context=r["context"], input=r["input"]) for r in records]

    async with AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY) as client:
        tasks   = [call_api(client, p) for p in prompts]
        all_texts = await asyncio.gather(*tasks)

    # Score and display
    results = []
    for record, texts in zip(records, all_texts):
        scored = [score_response(t, record["answers"]) for t in texts]
        num_correct = sum(s["is_correct"] for s in scored)
        results.append({
            **record,
            "model": MODEL,
            "trajectories": [{"text": t, **s} for t, s in zip(texts, scored)],
            "num_correct": num_correct,
            "pass_rate": round(num_correct / len(texts), 4) if texts else 0.0,
        })
        first = scored[0]
        status = "✓" if first["is_correct"] else "✗"
        print(
            f"[{record['index']:3d}] {status} "
            f"q: {record['input'][:55]!r:58s} "
            f"gold={record['answers']} "
            f"→ {first['extracted_answer']!r}"
        )

    # Save
    input_stem = Path(INPUT_FILE).stem
    output_file = os.environ.get(
        "OUTPUT_FILE",
        str(Path(INPUT_FILE).parent / f"{input_stem}-{MODEL.replace('/', '_')}-test{N_SAMPLES}.jsonl"),
    )
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    any_correct = sum(r["num_correct"] > 0 for r in results)
    total_traj  = sum(len(r["trajectories"]) for r in results)
    avg_f1      = sum(t["f1"] for r in results for t in r["trajectories"]) / total_traj
    print(f"\n{'='*60}")
    print(f"Any-correct : {any_correct}/{len(results)} ({any_correct/len(results):.0%})")
    print(f"Avg F1      : {avg_f1:.3f}")
    print(f"Saved to    : {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
