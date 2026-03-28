# Trajectory Generation for SFT — Design Spec

**Date:** 2026-03-28

## Goal

For each of the 15 (dataset × context_length) JSONL files in `output/`, generate `n` rollout trajectories per sample using a vLLM-served local model or an OpenAI-compatible API model. Score each trajectory with rule-based rewards. Save all trajectories with correctness labels for downstream SFT use.

---

## Context

- **Input data**: `output/{dataset}/train-num_sample_1000-max_seq_{length}.jsonl`
  Each line: `{index, input, context, answers, length}`
- **Default model**: `/home/wsy0227/qwen14b_2e_1node_16k_2k_FILTERedAGAIN_dis_256bsz_grpo_SUBEM_end-step151` (Qwen2.5-14B fine-tune, GRPO-trained, produces `<think>…</think>\boxed{…}` format)
- **Future models**: OpenAI-compatible API models (template TBD, passed via CLI)
- **Reference**: scoring logic adapted from [LoongRL longcontext_qa_llm_judge.py](https://github.com/rStar-RL/LoongRL/blob/main/verl/verl/utils/reward_score/longcontext_qa_llm_judge.py)

---

## Files

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `generate_trajectories.py` | Core script: load data → build prompts → rollout → score → save |
| Create | `scripts/gen_trajectories.sh` | Run all 15 combos sequentially |
| Read-only | `filter_again/judge_utils.py` | Reference only (not used; we implement LoongRL scoring inline) |

---

## Prompt Template

Default (LoongRL format, for the Qwen GRPO model):

```
The following are given passages.
{context}

Question: {input}
```

Overridable via `--prompt_template`. Both backends apply the template using Python `.format(context=..., input=...)` substitution — this convention is fixed and must be consistent across vllm and openai paths.

---

## `generate_trajectories.py` — CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_file` | required | Path to one input JSONL file |
| `--output_dir` | `trajectories/` | Root output directory |
| `--model` | `/home/wsy0227/qwen14b_...step151` | Model path (vllm) or name (openai) |
| `--backend` | `vllm` | `vllm` or `openai` |
| `--tp_size` | `4` | Tensor parallel size (vllm only) |
| `--n` | `4` | Rollouts per sample |
| `--temperature` | `0.6` | Sampling temperature |
| `--max_tokens` | `4096` | Max generation tokens |
| `--prompt_template` | LoongRL format | Overridable prompt template string |
| `--start_idx` | `0` | Slice input data (for distributed runs) |
| `--end_idx` | `-1` | Slice input data (`-1` means all; implemented as `records[start_idx:]` not `records[start_idx:-1]`) |
| `--openai_base_url` | `None` | Base URL for OpenAI-compatible API (stub; used when `--backend openai`) |

---

## Scoring Logic (Rule-Based, No LLM Judge)

Adapted from LoongRL `compute_score`. **Do NOT use `filter_again/judge_utils.py::exact_match_score` for `em`** — that function implements substring containment (i.e. `sub_em`), not strict equality. All three metrics are implemented from scratch:

1. **Answer extraction**:
   - If `</think>` not in response → `extracted_answer = None`, all scores = 0
   - Extract text after last `</think>`
   - Find `\boxed{…}` in that text → extract content as `extracted_answer`
   - If no `\boxed{}` → `extracted_answer = None`, all scores = 0
   - If `answers` is empty → all scores = 0

2. **Normalization**: lowercase, remove articles (`a/an/the`), remove punctuation, collapse whitespace

3. **Metrics** (computed against each gold answer in `answers`, take max):
   - **`sub_em`**: `normalize(gold) in normalize(pred)` OR `normalize(pred) in normalize(gold)` → 0 or 1
   - **`em`**: `normalize(pred) == normalize(gold)` (strict equality) → 0 or 1
   - **`f1`**: token-level F1 on normalized, split tokens → 0.0–1.0

4. **`is_correct` = `sub_em`** (matches LoongRL training reward)

---

## Output Format

**Output path**: `trajectories/{dataset}/train-num_sample_1000-max_seq_{length}-{model_tag}.jsonl`
`model_tag` = last path component of `--model` (e.g. `qwen14b_...step151`)

**Each line**:
```json
{
  "index": 0,
  "input": "Which magazine was started first...",
  "context": "Passage 1:\n...",
  "answers": ["Arthur's Magazine"],
  "length": 4096,
  "model": "qwen14b_...step151",
  "trajectories": [
    {
      "text": "<think>...</think>\\boxed{Arthur's Magazine}",
      "extracted_answer": "Arthur's Magazine",
      "sub_em": 1,
      "em": 1,
      "f1": 1.0,
      "is_correct": 1
    },
    {
      "text": "<think>...</think>\\boxed{First for Women}",
      "extracted_answer": "First for Women",
      "sub_em": 0,
      "em": 0,
      "f1": 0.0,
      "is_correct": 0
    }
  ],
  "num_correct": 1,
  "pass_rate": 0.25
}
```

**Output path derivation**: parsed from `--input_file` basename.
- Input: `output/hotpotqa/train-num_sample_1000-max_seq_4096.jsonl`
- `{dataset}` = parent directory name (`hotpotqa`)
- `{length_tag}` = stem of filename (`train-num_sample_1000-max_seq_4096`)
- Output: `{output_dir}/{dataset}/{length_tag}-{model_tag}.jsonl`

---

## Internal Architecture

```
load_jsonl(input_file)
  → build_prompts(records, prompt_template)      # list of chat messages
  → run_vllm(prompts, args)                      # or run_openai(...)
      returns List[List[str]]  (n texts per sample)
  → score_trajectories(texts, answers)
      → extract_answer(text)   → (extracted_answer, think_present)
      → compute_rewards(extracted_answer, answers)  → {sub_em, em, f1}
  → assemble output records
  → write_jsonl(output_path, records)
```

Two backend functions with identical signatures:
- `run_vllm(prompts, model, tp_size, n, temperature, max_tokens) → List[List[str]]`
- `run_openai(prompts, model, n, temperature, max_tokens) → List[List[str]]`

---

## `scripts/gen_trajectories.sh`

Loops over all 15 (dataset × length) combos and launches them **sequentially** (not in parallel) — vLLM claims all GPUs for a single job, so parallel launch would cause OOM. Configurable via env vars: `MODEL`, `BACKEND`, `TP_SIZE`, `N`, `TEMPERATURE`.

---

## Non-Goals

- No LLM judge (skipped entirely)
- No parquet output (plain JSONL only)
- No distributed multi-node coordination (single-node vLLM with `--tp_size`)
- No deduplication or filtering at this stage (save all, filter later)
