#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评测脚本（带 pair 复用 / 进度条 / 每-pair 平均准确率）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from judge_utils import boxed_exact_match_judge  # 你的判分函数
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ------------------------
# 工具函数
# ------------------------


def save_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    """将列表写入 jsonl 文件（覆盖写）"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def pick_first(d: Dict[str, Any], keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def build_output_pairs(
    records: List[Dict[str, Any]],
    vllm_outputs: List[Any],
    question_key_candidates=("question", "prompt", "input"),
    answer_key_candidates=("outputs", "answers", "gold", "label"),
) -> List[Dict[str, Any]]:
    """
    将 parquet 记录 + vLLM 返回结果 组装成判分所需的结构
    """
    assert len(records) == len(
        vllm_outputs
    ), f"#records({len(records)}) != #outputs({len(vllm_outputs)})"

    def extract_pred_list(vllm_request_output) -> List[str]:
        # vLLM 常见结构：outputs 为 List[RequestOutputFragment]
        return [frag.text for frag in vllm_request_output.outputs]

    output_pairs: List[Dict[str, Any]] = []
    for rec, out in zip(records, vllm_outputs):
        question = pick_first(rec, question_key_candidates)
        if question is None:
            question = rec.get("prompt", "")
        # 某些数据集 question 可能是 ndarray / Series
        if hasattr(question, "tolist"):
            question = question.tolist()

        gold = pick_first(rec, answer_key_candidates)
        if gold is None and "reward_model" in rec:
            gold = rec["reward_model"]["ground_truth"]
        gold = gold.tolist() if hasattr(gold, "tolist") else gold
        gold_list = to_list(gold)

        pred_list = extract_pred_list(out)

        output_pairs.append(
            {
                "question": question,
                "outputs": gold_list,
                "pred_list": pred_list,
            }
        )
    return output_pairs


def evaluate_pairs_with_avg(
    output_pairs: List[Dict[str, Any]]
) -> tuple[float, float, List[Dict[str, Any]]]:
    """
    逐 pair 判分，返回：
      macro_avg_acc   —— 每题 avg_acc 再求平均
      micro_any_correct —— 每题是否有任一回答正确 (1/0) 再求平均
      per_item_results —— 每题详细结果
    同时，本函数“并不”修改 output_pairs；由调用方写回 avg_acc 等字段
    """
    per_item_results: List[Dict[str, Any]] = []
    avg_list, any_correct_list = [], []

    for op in tqdm(output_pairs, desc="Scoring pairs"):
        jr_list = boxed_exact_match_judge(op)
        scores = [jr["is_correct"] for jr in jr_list]
        avg_acc = sum(scores) / max(len(scores), 1)
        any_correct = 1 if max(scores) > 0 else 0

        avg_list.append(avg_acc)
        any_correct_list.append(any_correct)

        per_item_results.append(
            {
                "question": op["question"],
                "judge_results": jr_list,
                "avg_acc": avg_acc,
                "any_correct": bool(any_correct),
                "pred_list": op["pred_list"],
                "gold": op["outputs"],
            }
        )

    macro_avg_acc = sum(avg_list) / max(len(avg_list), 1)
    micro_any_correct = sum(any_correct_list) / max(len(any_correct_list), 1)
    return macro_avg_acc, micro_any_correct, per_item_results


# ------------------------
# 路径 & 模型配置
# ------------------------

dataset_prefix = "/mnt/longcontext/models/siyuan"
PRETRAINED_MODEL_PATH = (
    "/mnt/longcontext/models/siyuan/rl_ckpts/qwen25_f1s_fix_no_entro_2node_16k_2k_math_filtered_dis_mathqa_256bsz_20ksamples_end-step420"
)

train_files = [
    f"{dataset_prefix}/rl_datasets/rl_three/system/hotpotqa_qwen_filtered_start_idx0_end_idx2500_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/hotpotqa_filtered_distractor_256_start_idx2500_end_idx5000_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/musique_qwen_filtered_start_idx0_end_idx2500_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/musique_filtered_distractor_256_start_idx2500_end_idx5000_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/2wikimqa_qwen_filtered_start_idx0_end_idx2500_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/2wikimqa_filtered_distractor_256_start_idx2500_end_idx5000_seq16384/train.parquet",
]

train_files += [
    f"{dataset_prefix}/rl_datasets/rl_three/system/musique_qwen_filtered_start_idx5000_end_idx6750_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/musique_filtered_distractor_256_start_idx6750_end_idx8500_seq16384/train.parquet",
]

RESULT_DIR = Path("eval_results_420")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# 一次性加载 tokenizer / 模型
# ------------------------

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, trust_remote_code=True)
model = LLM(
    model=PRETRAINED_MODEL_PATH,
    max_model_len=32768,
    data_parallel_size=1,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.95,
    max_num_batched_tokens=32768,
)

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=4096,
    n=8,
)

# 你可以根据显存把 prompts 再做 chunk
VLLM_BATCH = 10000000  # 足够大 → 实际一次送完

summary_all: List[Dict[str, Any]] = []

# ------------------------
# 主流程
# ------------------------

for file_path in train_files:
    file_path = Path(file_path)
    tag = f"{file_path.parent.name}_{file_path.stem}"

    pairs_path = RESULT_DIR / f"{tag}_pairs.jsonl"
    per_item_path = RESULT_DIR / f"{tag}_results.jsonl"

    # ---------- (A) 拿到 output_pairs ----------
    if pairs_path.exists():
        # 直接读取
        with pairs_path.open("r", encoding="utf-8") as f:
            output_pairs = [json.loads(line) for line in f]
        print(f"✓ 复用已存在 pairs: {pairs_path}")
    else:
        # 读取 parquet、推理
        df = pd.read_parquet(file_path)
        records = df.to_dict(orient="records")

        prompts = [r["prompt"] for r in records]  # prompt 字段本身应是 str

        outputs_all = []
        # 如果你担心显存，可改用分块循环
        for chunk_start in range(0, len(prompts), VLLM_BATCH):
            chunk_prompts = prompts[chunk_start : chunk_start + VLLM_BATCH]
            chunk_prompts = [list(p) for p in chunk_prompts] 
            chunk_out = model.chat(
                messages=chunk_prompts, sampling_params=sampling_params, use_tqdm=True
            )
            outputs_all.extend(chunk_out)

        output_pairs = build_output_pairs(records, outputs_all)

    # ---------- (B) 评测 ----------
    macro_avg_acc, micro_any_correct, per_item_results = evaluate_pairs_with_avg(
        output_pairs
    )

    # ---------- (C) 把 avg_acc / any_correct 写回 pair ----------
    for pair, res in zip(output_pairs, per_item_results):
        pair["avg_acc"] = res["avg_acc"]
        pair["any_correct"] = res["any_correct"]

    # 覆盖保存 pairs & per-item results
    save_jsonl(pairs_path, output_pairs)
    save_jsonl(per_item_path, per_item_results)

    # ---------- (D) 文件级别 summary ----------
    file_summary = {
        "file": str(file_path),
        "macro_avg_acc": macro_avg_acc,
        "micro_any_correct": micro_any_correct,
        "num_samples": len(output_pairs),
        "results_path": str(per_item_path),
        "pairs_path": str(pairs_path),
    }
    summary_all.append(file_summary)

    with (RESULT_DIR / f"{tag}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(file_summary, f, ensure_ascii=False, indent=2)

# 总汇总
with (RESULT_DIR / "all_files_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary_all, f, ensure_ascii=False, indent=2)

print("\n======  全部完成  ======")
for s in summary_all:
    print(
        f"{s['file']} -> macro_avg_acc={s['macro_avg_acc']:.4f}, "
        f"micro_any_correct={s['micro_any_correct']:.4f}, "
        f"samples={s['num_samples']}"
    )

# 若需要 keep_gpu 监控 GPU，保留；否则可删除
try:
    from keep_gpu.cli import main as keep_gpu_main
    keep_gpu_main()
except ImportError:
    pass