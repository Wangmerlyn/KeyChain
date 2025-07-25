import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from judge_utils import boxed_exact_match_judge
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ------------------------
# 工具函数
# ------------------------


def save_jsonl(path: str | Path, rows: List[Dict[str, Any]]):
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


def chunked(iterable: List[Any], n: int) -> Iterable[List[Any]]:
    """把 iterable 按 n 大小切块"""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def extract_pred_list(vllm_request_output) -> List[str]:
    # 适配 vLLM 的常见结构；若与你版本不一致，请据实修改
    return [frag.text for frag in vllm_request_output.outputs]


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
    assert len(records) == len(vllm_outputs), (
        f"#records({len(records)}) != #outputs({len(vllm_outputs)})"
    )

    output_pairs = []
    for rec, out in zip(records, vllm_outputs):
        question = pick_first(rec, question_key_candidates)
        if question is None:
            question = rec.get("prompt", "")
        question = question.tolist()

        # gold = pick_first(rec, answer_key_candidates)
        gold = rec["reward_model"]["ground_truth"].tolist()
        gold_list = to_list(gold)

        pred_list = extract_pred_list(out)

        output_pairs.append(
            {"question": question, "outputs": gold_list, "pred_list": pred_list}
        )
    return output_pairs


def evaluate_pairs_with_avg(output_pairs: List[Dict[str, Any]]):
    """
    返回：
      - macro_avg_acc：对每题 '8个回答准确率的平均值' 再求平均
      - micro_any_correct：对每题看是否有任意一个回答正确（1/0），再求平均
      - per_item_results：每题的详细 judge 结果 + 平均准确率
    """
    per_item_results = []
    avg_list = []
    any_correct_list = []

    for op in output_pairs:
        jr_list = boxed_exact_match_judge(op)  # 你已有的函数
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
# 主流程（按 parquet 分文件评测 & 保存）
# ------------------------

dataset_prefix = "/mnt/longcontext/models/siyuan"
PRETRAINED_MODEL_PATH = "/mnt/longcontext/models/siyuan/rl_ckpts/qwen25_fix_no_entro_2node_16k_2k_math_filtered_dis_mathqa_256bsz_20ksamples_end-step420"

train_files = [
    f"{dataset_prefix}/rl_datasets/rl_three/system/hotpotqa_qwen_filtered_start_idx0_end_idx2500_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/hotpotqa_filtered-uuid_default_8-numchain_64-numhop-4-distractor_256_start_idx2500_end_idx5000_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/musique_qwen_filtered_start_idx0_end_idx2500_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/musique_filtered-uuid_default_8-numchain_64-numhop-4-distractor_256_start_idx2500_end_idx5000_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/2wikimqa_qwen_filtered_start_idx0_end_idx2500_seq16384/train.parquet",
    f"{dataset_prefix}/rl_datasets/rl_three/system/2wikimqa_filtered-uuid_default_8-numchain_64-numhop-4-distractor_256_start_idx2500_end_idx5000_seq16384/train.parquet",
]

# 结果输出目录
RESULT_DIR = Path("eval_results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 一次性加载模型（避免在每个 parquet 里重复加载）
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH, trust_remote_code=True)
model = LLM(
    model=PRETRAINED_MODEL_PATH,
    max_model_len=32768,
    data_parallel_size=1,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.95,
    max_num_batched_tokens=32769,
)

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=4096,
    n=8,
)

# 为了避免一次性送太大 batch，可设置一个合适的 chunk size
VLLM_BATCH = 10000000  # 你可以根据显存调整

summary_all = []

for file_path in train_files:
    df = pd.read_parquet(file_path)
    records = df.to_dict(orient="records")
    records = records[:10]

    # 构造输入（这里假设有 "prompt" 字段；请按需调整）
    prompts = [list(r.get("prompt")) for r in records]

    # 对该 parquet 分块推理
    outputs_all = []
    out_chunk = model.chat(
        messages=prompts, sampling_params=sampling_params, use_tqdm=True
    )
    outputs_all.extend(out_chunk)

    # 组织成 boxed_exact_match_judge 所需的格式
    output_pairs = build_output_pairs(records, outputs_all)

    # 评测：返回每题平均acc，以及文件级别两个指标
    macro_avg_acc, micro_any_correct, per_item_results = evaluate_pairs_with_avg(
        output_pairs
    )

    # 保存该 parquet 的逐题结果
    stem = Path(file_path).stem  # train
    # 也许你更想把上一级目录名也带上
    tag = Path(file_path).parent.name + "_" + stem

    # 1) 每题的详细 judge 结果（含 avg_acc）
    per_item_path = RESULT_DIR / f"{tag}_results.jsonl"
    save_jsonl(per_item_path, per_item_results)

    # 2) 同时把可复现的 input-output 对象也保存
    pairs_path = RESULT_DIR / f"{tag}_pairs.jsonl"
    save_jsonl(pairs_path, output_pairs)

    # 3) 文件级别 summary
    file_summary = {
        "file": file_path,
        "macro_avg_acc": macro_avg_acc,
        "micro_any_correct": micro_any_correct,
        "num_samples": len(records),
        "results_path": str(per_item_path),
        "pairs_path": str(pairs_path),
    }
    summary_all.append(file_summary)

    # 也把这个文件的 summary 单独落盘
    with (RESULT_DIR / f"{tag}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(file_summary, f, ensure_ascii=False, indent=2)

# 把所有 parquet 的 summary 汇总到一个文件
with (RESULT_DIR / "all_files_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary_all, f, ensure_ascii=False, indent=2)

print("Done. Summaries:")
for s in summary_all:
    print(
        f"{s['file']} -> macro_avg_acc={s['macro_avg_acc']:.4f}, "
        f"micro_any_correct={s['micro_any_correct']:.4f}, "
        f"samples={s['num_samples']}"
    )

from keep_gpu.cli import main

main()
