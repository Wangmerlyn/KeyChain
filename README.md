<div align="center">

# KeyChain: Key-Sentence-Driven Long-Context Data Synthesis **[ICLR 2026 Oral]**

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026%20Oral-blue)](https://iclr.cc/virtual/2026/oral/10007440)
[![arXiv](https://img.shields.io/badge/arXiv-2510.19363-b31b1b.svg)](https://arxiv.org/abs/2510.19363)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://loongrl.github.io)
[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm-dark.svg)](https://huggingface.co/papers/2510.19363)
<a href="https://huggingface.co/datasets/OldKingMeister/LoongRL-Train-Data"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg" height="20"></a>

</div>

This repository implements the **KeyChain** data creation pipeline from [LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts](https://arxiv.org/abs/2510.19363). KeyChain synthesizes high-quality, verifiable long-context QA training data by embedding questions within long contexts using UUID key-value chain linking and multi-level distractors — designed specifically for reinforcement learning over long contexts with fine-grained difficulty control.

**RL Training**: For the reinforcement learning training framework, see [**LoongRL**](https://github.com/rStar-RL/LoongRL).

---

## Table of Contents

- [KeyChain Pipeline](#keychain-pipeline) — original UUID-chain synthesis
- [Plain Multihop Synthesis](#plain-multihop-synthesis) — vanilla long-context QA data generation
- [Trajectory Generation](#trajectory-generation) — rollout generation + scoring for SFT
- [Repository Structure](#repository-structure)
- [Citation](#citation)

---

## KeyChain Pipeline

### Overview

KeyChain constructs long-context QA instances through a three-stage pipeline:

1. **Data Filtering** — Filter source multi-hop questions (HotpotQA, MuSiQue, 2WikiMQA) using Qwen2.5-32B, retaining only questions of appropriate difficulty
2. **Long Context Filling** — Compose long contexts (4K–128K tokens) by shuffling and inserting documents around the question-relevant passages
3. **KeyChain Insertion** — Generate UUID key-value chains and insert the pairs at random positions throughout the context. One chain leads to the real question; other chains lead to distractor questions. The model must follow the correct chain starting from a given UUID to locate the question, then reason over the surrounding documents to answer it

### Generated Data Example

Each instance presents the model with a long context containing documents interleaved with UUID key-value pairs. The model receives a starting UUID and must follow the chain to find the hidden question:

```
Please read the following text.
Document 0: ...
Document 3:
Who's Who? is a studio album by American jazz musician John Scofield. ...
{"bdd640fb-0667-4ad1-9c80-317fa3b1799d": "23b8c1e9-3924-46de-beb1-3b9046685257"}.
...
Document 10:
... Sonoma State offers 92 Bachelor's degrees, 19 Master's degrees ...
{"972a8469-1641-4f82-8b9d-2434e465e150": "Musician and satirist Allie Goertz
 wrote a song about the "The Simpsons" character Milhouse, who Matt Groening
 named after who?"}.
...
Document 47:
Neil Affleck
{"bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9": "972a8469-1641-4f82-8b9d-2434e465e150"}.
...

In the context above, there is one correct question to answer. The correct
question can only be found by following the correct consecutive chain of
key:value pairs encoded with UUID strings, starting from
"bdd640fb-0667-4ad1-9c80-317fa3b1799d".
Find the correct question first, then answer it.
```

### Data Sources

| Dataset   | Training QA Pairs | Unique Documents |
| --------- | ----------------: | ---------------: |
| HotpotQA  |            90,447 |          483,696 |
| MuSiQue   |            19,938 |          797,413 |
| 2WikiMQA  |           167,454 |          369,378 |

### Installation

```bash
pip install transformers tenacity openai azure-identity tqdm gdown
sudo apt update -y && sudo apt install unzip
```

### Usage

```bash
# Standard long-context QA synthesis
bash scripts/synth.sh

# Relevant-documents-only (no fillers)
bash scripts/synth_relevant_only.sh

# Full pipeline: context filling + KeyChain insertion across all datasets × lengths
bash scripts/synth_qwen_filter_core_gaussian_add_distractor.sh

# Extract GPT-4o reasoning steps
bash scripts/synth_gpt_call_reasoning.sh
```

Custom generation:

```bash
python qa.py \
    --save_dir=./ \
    --save_name=hotpotqa \
    --dataset=hotpot_train_v1.1.json \
    --tokenizer_path=Qwen/Qwen2.5-7B-Instruct \
    --tokenizer_type=hf \
    --max_seq_length=32768 \
    --tokens_to_generate=128 \
    --num_samples=100 \
    --template="{context}"
```

---

## Plain Multihop Synthesis

Synthesize vanilla long-context multihop QA data (no UUID/KeyChain) for SFT pre-training or evaluation. Covers all three datasets at five context lengths.

### Quick Start

```bash
# Download datasets + generate 1000 samples per (dataset × length) combo
# 15 jobs total: hotpotqa/musique/2wikimqa × 4k/8k/16k/32k/64k
bash scripts/synth_multihop_plain.sh
```

Datasets are auto-downloaded on first run. Override the tokenizer via env var:

```bash
TOKENIZER_PATH=/path/to/model bash scripts/synth_multihop_plain.sh
```

Scale to a different sample count:

```bash
NUM_SAMPLES=500 bash scripts/synth_multihop_plain.sh
```

### Output

```
output/
├── hotpotqa/
│   ├── train-num_sample_1000-max_seq_4096.jsonl
│   ├── train-num_sample_1000-max_seq_8192.jsonl
│   ├── train-num_sample_1000-max_seq_16384.jsonl
│   ├── train-num_sample_1000-max_seq_32768.jsonl
│   └── train-num_sample_1000-max_seq_65536.jsonl
├── musique/   (same structure)
└── 2wikimqa/  (same structure)
```

Each line:

```json
{
  "index": 0,
  "input": "Which magazine was started first Arthur's Magazine or First for Women?",
  "context": "Passage 1:\n...\n\nPassage 2:\n...",
  "answers": ["Arthur's Magazine"],
  "length": 4096
}
```

Default tokenizer: `Qwen/Qwen3-8B` (only tokenizer config downloaded, no model weights).

---

## Trajectory Generation

Generate model rollouts over the plain multihop data and score them with rule-based rewards for SFT training.

### Scoring

Rewards are adapted from [LoongRL](https://github.com/rStar-RL/LoongRL/blob/main/verl/verl/utils/reward_score/longcontext_qa_llm_judge.py) (rule-based only, no LLM judge):

| Metric | Description |
|--------|-------------|
| `sub_em` | Bidirectional substring containment after normalization — primary reward, matches LoongRL training signal |
| `em` | Strict exact match after normalization |
| `f1` | Token-level F1 score |

Answer extraction: finds `</think>` tag → extracts `\boxed{answer}` from text after it (for thinking models). `is_correct = sub_em`.

### Option A: Local vLLM model

```bash
# All 15 combos sequentially (vLLM owns all GPUs per job)
bash trajectory_gen/scripts/gen_trajectories.sh

# Override parameters
MODEL=/path/to/model TP_SIZE=8 N=8 bash trajectory_gen/scripts/gen_trajectories.sh
```

Default model: `/home/wsy0227/qwen14b_2e_1node_16k_2k_FILTERedAGAIN_dis_256bsz_grpo_SUBEM_end-step151`

Single file:

```bash
python trajectory_gen/generate_trajectories.py \
    --input_file output/hotpotqa/train-num_sample_1000-max_seq_8192.jsonl \
    --model /path/to/model \
    --tp_size 4 \
    --n 4 \
    --temperature 0.6
```

### Option B: OpenAI-compatible API

```bash
# Single file via API
python trajectory_gen/generate_trajectories.py \
    --input_file output/hotpotqa/train-num_sample_1000-max_seq_8192.jsonl \
    --backend openai \
    --model DeepSeek-V3.2 \
    --openai_base_url http://host:port/v1

# All 15 combos
BACKEND=openai MODEL=DeepSeek-V3.2 OPENAI_BASE_URL=http://host:port/v1 \
    bash trajectory_gen/scripts/gen_trajectories.sh
```

Note: for API backends, set `OPENAI_API_KEY` env var if required by the server.

### Option C: Quick API test (10 samples)

```bash
OPENAI_BASE_URL=http://host:port/v1 \
OPENAI_API_KEY=dummy \
python trajectory_gen/test_api_trajectory.py

# Optional overrides
N_SAMPLES=50 N_ROLLOUTS=4 MODEL=gpt-4o \
OPENAI_BASE_URL=http://host:port/v1 \
python trajectory_gen/test_api_trajectory.py
```

Uses a different prompt template (step-by-step reasoning, "The answer is X" format) suited for non-thinking models. Concurrent API calls via asyncio.

### Output

```
trajectories/
├── hotpotqa/
│   ├── train-num_sample_1000-max_seq_4096-{model_tag}.jsonl
│   └── ...
├── musique/   (same structure)
└── 2wikimqa/  (same structure)
```

Each line:

```json
{
  "index": 0,
  "input": "Which magazine was started first...",
  "context": "Passage 1:\n...",
  "answers": ["Arthur's Magazine"],
  "length": 8192,
  "model": "qwen14b_...step151",
  "trajectories": [
    {
      "text": "<think>...</think>\\boxed{Arthur's Magazine}",
      "extracted_answer": "Arthur's Magazine",
      "sub_em": 1,
      "em": 1,
      "f1": 1.0,
      "is_correct": 1
    }
  ],
  "num_correct": 1,
  "pass_rate": 0.25
}
```

### Tests

```bash
# Unit tests for scoring functions (no GPU required)
python -m pytest trajectory_gen/tests/test_scoring.py -v
```

---

## Repository Structure

```
KeyChain/
│
├── ── Plain Multihop Synthesis ──────────────────────────────────────────────
│
├── qa.py                                            # Long-context QA synthesis (plain, no KeyChain)
│                                                    #   --distract_questions -1 to disable distractors
├── scripts/synth_multihop_plain.sh                  # Run all 15 (dataset × length) synthesis jobs
│
├── ── Trajectory Generation ─────────────────────────────────────────────────
│
├── trajectory_gen/
│   ├── generate_trajectories.py                     # Rollout generation + rule-based scoring
│   │                                                #   Backends: vllm, openai
│   │                                                #   Rewards: sub_em (primary), em, f1
│   ├── test_api_trajectory.py                       # Quick async test against any OpenAI-compatible API
│   ├── scripts/
│   │   └── gen_trajectories.sh                      # Batch runner for all 15 combos (sequential)
│   └── tests/
│       └── test_scoring.py                          # Unit tests for scoring logic (no GPU needed)
│
├── ── KeyChain Pipeline ─────────────────────────────────────────────────────
│
├── qa_filter_data.py                                # Context generation with pre-filtering
├── qa_filter_data_core_gaussian_middle.py           # Gaussian question placement
├── qa_qwen_filtered_core_gaussian_add_distractor.py # Full pipeline: context + KeyChain
├── qa_add_distractor.py                             # Distractor injection module
├── qa_relevant_only.py                              # Supporting-facts-only generation
├── qa_musique_hard.py                               # MuSiQue-specific variant
├── uuid_test.py                                     # UUID chain/tree generation utilities
├── gpt_call.py                                      # GPT-4o reasoning extraction
├── tokenizer.py                                     # Multi-backend tokenizer (HF, NeMo, OpenAI, Gemini)
│
├── ── Quality Filtering ─────────────────────────────────────────────────────
│
├── filter_question/                                 # Stage 1: model-based QA quality filtering
│   ├── filter_infer.py                              #   vLLM distributed inference (Qwen2.5-32B)
│   ├── merge_output.py                              #   Prediction merging & metric computation
│   └── convert_*.py                                 #   Dataset format converters
├── filter_again/                                    # Stage 2: secondary quality control
│   ├── judge_filters.py                             #   Answer matching & filtering
│   └── judge_utils.py                               #   Metrics: EM, F1, CEM
│
├── ── Scripts ───────────────────────────────────────────────────────────────
│
├── scripts/
│   ├── synth_multihop_plain.sh                      # Plain multihop synthesis (NEW)
│   ├── synth.sh                                     # Basic synthesis
│   ├── synth_relevant_only.sh                       # Supporting-facts-only
│   ├── synth_qwen_filter_core_gaussian_add_distractor.sh  # Full KeyChain pipeline
│   ├── synth_context_all_lengths.sh                 # All context lengths (no filtering)
│   └── synth_gpt_call_reasoning.sh                  # GPT-4o reasoning extraction
│
└── difficulty_analysis/                             # Dataset analysis notebooks
```

---

## Citation

If you use KeyChain in your research, please cite our paper:

```bibtex
@misc{wang2025loongrlreinforcementlearningadvanced,
      title={LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts},
      author={Siyuan Wang and Gaokai Zhang and Li Lyna Zhang and Ning Shang and Fan Yang and Dongyao Chen and Mao Yang},
      year={2025},
      eprint={2510.19363},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.19363},
}
```


[![ICLR 2026](https://img.shields.io/badge/ICLR-2026%20Oral-blue)](https://iclr.cc/virtual/2026/oral/10007440)
[![arXiv](https://img.shields.io/badge/arXiv-2510.19363-b31b1b.svg)](https://arxiv.org/abs/2510.19363)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://loongrl.github.io)
[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm-dark.svg)](https://huggingface.co/papers/2510.19363)
<a href="https://huggingface.co/datasets/OldKingMeister/LoongRL-Train-Data"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg" height="20"></a>

</div>

This repository implements the **KeyChain** data creation pipeline from [LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts](https://arxiv.org/abs/2510.19363). KeyChain synthesizes high-quality, verifiable long-context QA training data by embedding questions within long contexts using UUID key-value chain linking and multi-level distractors — designed specifically for reinforcement learning over long contexts with fine-grained difficulty control.

**RL Training**: For the reinforcement learning training framework, see [**LoongRL**](https://github.com/rStar-RL/LoongRL).

## Overview

KeyChain constructs long-context QA instances through a three-stage pipeline:

1. **Data Filtering** — Filter source multi-hop questions (HotpotQA, MuSiQue, 2WikiMQA) using Qwen2.5-32B, retaining only questions of appropriate difficulty
2. **Long Context Filling** — Compose long contexts (4K–128K tokens) by shuffling and inserting documents around the question-relevant passages
3. **KeyChain Insertion** — Generate UUID key-value chains and insert the pairs at random positions throughout the context. One chain leads to the real question; other chains lead to distractor questions. The model must follow the correct chain starting from a given UUID to locate the question, then reason over the surrounding documents to answer it

### Generated Data Example

Each instance presents the model with a long context containing documents interleaved with UUID key-value pairs. The model receives a starting UUID and must follow the chain to find the hidden question:

```
Please read the following text.
Document 0: ...
Document 3:
Who's Who? is a studio album by American jazz musician John Scofield. ...
{"bdd640fb-0667-4ad1-9c80-317fa3b1799d": "23b8c1e9-3924-46de-beb1-3b9046685257"}.
...
Document 10:
... Sonoma State offers 92 Bachelor's degrees, 19 Master's degrees ...
{"972a8469-1641-4f82-8b9d-2434e465e150": "Musician and satirist Allie Goertz
 wrote a song about the "The Simpsons" character Milhouse, who Matt Groening
 named after who?"}.
...
Document 47:
Neil Affleck
{"bd9c66b3-ad3c-4d6d-9a3d-1fa7bc8960a9": "972a8469-1641-4f82-8b9d-2434e465e150"}.
...

In the context above, there is one correct question to answer. The correct
question can only be found by following the correct consecutive chain of
key:value pairs encoded with UUID strings, starting from
"bdd640fb-0667-4ad1-9c80-317fa3b1799d".
Find the correct question first, then answer it.
```

## Data Sources

| Dataset   | Training QA Pairs | Unique Documents |
| --------- | ----------------: | ---------------: |
| HotpotQA  |            90,447 |          483,696 |
| MuSiQue   |            19,938 |          797,413 |
| 2WikiMQA  |           167,454 |          369,378 |

## Installation

```bash
pip install transformers tenacity openai azure-identity tqdm gdown
sudo apt update -y && sudo apt install unzip
```

## Usage

### Basic Synthesis

```bash
# Standard long-context QA synthesis
bash scripts/synth.sh

# Relevant-documents-only (no fillers)
bash scripts/synth_relevant_only.sh
```

### Full Pipeline (Context Filling + KeyChain Insertion)

```bash
# Synthesize across all datasets x all context lengths with
# context filling and KeyChain insertion
bash scripts/synth_qwen_filter_core_gaussian_add_distractor.sh
```

### Reasoning Extraction

```bash
# Extract GPT-4o reasoning steps for training data
bash scripts/synth_gpt_call_reasoning.sh
```

### Custom Generation

```bash
python qa.py \
    --save_dir=./ \
    --save_name=hotpotqa \
    --dataset=hotpotqa/hotpot_train_v1.1.json \
    --tokenizer_path=Qwen/Qwen2.5-7B-Instruct \
    --tokenizer_type=hf \
    --max_seq_length=32768 \
    --tokens_to_generate=128 \
    --num_samples=100 \
    --template="{context}"
```

## Repository Structure

```
KeyChain/
├── qa.py                                            # Basic context composition
├── qa_filter_data.py                                # Context generation with pre-filtering
├── qa_filter_data_core_gaussian_middle.py           # Gaussian question placement
├── qa_qwen_filtered_core_gaussian_add_distractor.py # Full pipeline: context filling + KeyChain insertion
├── qa_add_distractor.py                             # Distractor injection module
├── qa_relevant_only.py                              # Supporting-facts-only generation
├── gpt_call.py                                      # GPT-4o reasoning extraction
├── tokenizer.py                                     # Multi-backend tokenizer (HF, NeMo, OpenAI)
├── filter_question/                                 # Stage 1: model-based QA quality filtering
│   ├── filter_infer.py                              #   vLLM distributed inference
│   ├── merge_output.py                              #   Prediction merging
│   └── convert_*.py                                 #   Dataset format converters
├── filter_again/                                    # Stage 2: secondary quality control
│   ├── judge_filters.py                             #   Answer matching & filtering
│   └── judge_utils.py                               #   Evaluation metrics (EM, F1, CEM)
├── scripts/                                         # Shell scripts for pipeline execution
│   ├── synth.sh                                     #   Basic synthesis
│   ├── synth_qwen_filter_core_gaussian_add_distractor.sh  # Full pipeline
│   └── synth_gpt_call_reasoning.sh                  #   Reasoning extraction
└── difficulty_analysis/                             # Dataset analysis notebooks
```

## Citation

If you use KeyChain in your research, please cite our paper:

```bibtex
@misc{wang2025loongrlreinforcementlearningadvanced,
      title={LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts},
      author={Siyuan Wang and Gaokai Zhang and Li Lyna Zhang and Ning Shang and Fan Yang and Dongyao Chen and Mao Yang},
      year={2025},
      eprint={2510.19363},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.19363},
}
```
