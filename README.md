# KeyChain: Key-Sentence-Driven Long-Context Data Synthesis **[ICLR 2026 Oral]**

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026%20Oral-blue)](https://iclr.cc/virtual/2026/oral/10007440)
[![arXiv](https://img.shields.io/badge/arXiv-2510.19363-b31b1b.svg)](https://arxiv.org/abs/2510.19363)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://loongrl.github.io)
[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm-dark.svg)](https://huggingface.co/papers/2510.19363)
<a href="https://huggingface.co/datasets/OldKingMeister/LoongRL-Train-Data"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg" height="20"></a>

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
