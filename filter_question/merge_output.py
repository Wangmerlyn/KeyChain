import os
import glob
import json
import re
import string
from collections import Counter

# --- Config ---
note = "dist_run"
num_total_nodes = 10

dataset_prefix="/home/aiscuser/filter_question/data"

# Dataset lengths
dataset_lengths = {
    "hotpotqa": 90447,
    "musique": 19938,
    "2wikimqa": 167454,
}

# Output file pattern (per-node)
def get_chunk_range(total_len, node_id, num_nodes):
    chunk_size = (total_len + num_nodes - 1) // num_nodes
    start = node_id * chunk_size
    end = min(start + chunk_size, total_len)
    return start, end

# Output file pattern: <dataset>_train_merged_pred_<start>_<end>_<note>.jsonl
def get_expected_files(dataset_name, total_len):
    files = []
    for node_id in range(num_total_nodes):
        start, end = get_chunk_range(total_len, node_id, num_total_nodes)
        fname = f"{dataset_prefix}/{dataset_name}_train_merged_pred_{start}_{end}_{note}.jsonl"
        if os.path.exists(fname):
            files.append(fname)
        else:
            print(f"[Warning] Missing file: {fname}")
    return files

# Merge all jsonl files into one
def merge_jsonl_files(file_list, output_path):
    merge_dataset = [
        json.loads(line)
        for fname in file_list
        for line in open(fname, "r")
    ]
    with open(output_path, "w") as fout:
        for fname in file_list:
            with open(fname, "r") as fin:
                for line in fin:
                    fout.write(line)
    print(f"✅ Merged {len(file_list)} files into: {output_path}")
    return merge_dataset

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_answer(text: str) -> str:
    keyword = "the answer is"
    keyword_lower = keyword.lower()
    lower_text = text.lower()
    idx = lower_text.find(keyword_lower)
    if idx == -1:
        return ""  # Not found
    # Extract everything after the keyword
    return text[idx + len(keyword):].strip()

def _cem_score(pred, gt):
# cem for covered exact match
    pred = normalize_answer(pred)
    gt = normalize_answer(gt)
    if pred in gt:
        return 1.0
    else:
        return 0.0

def cem_score(entry):
    pred_list = entry['pred_list']
    gts = entry['outputs']
    for pred in pred_list:
        max_score=0
        judge_reason = extract_answer(pred['pred'])
        if judge_reason == "":
            max_score = 0
        else:
            for gt in gts:
                cem_score = _cem_score(judge_reason, gt)
                max_score = max(max_score, cem_score)
        pred['judge_reason'] = judge_reason
        pred['cem_score'] = max_score
    return pred_list

def _f1_score(prediction, ground_truth, **kwargs):
    norm_pred = normalize_answer(prediction).split()
    norm_gt = normalize_answer(ground_truth).split()
    common = Counter(norm_pred) & Counter(norm_gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(norm_pred)
    recall = 1.0 * num_same / len(norm_gt)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_score(entry):
    pred_list = entry['pred_list']
    gts = entry['outputs']
    for pred in pred_list:
        max_score=0
        judge_reason = extract_answer(pred['pred'])
        if judge_reason == "":
            max_score = 0
        else:
            for gt in gts:
                f1_score = _f1_score(judge_reason, gt)
                max_score = max(max_score, f1_score)
        pred['judge_reason'] = judge_reason
        pred['f1_score'] = max_score
    return pred_list
    



# --- Main logic ---
for dataset_name, total_len in dataset_lengths.items():
    per_node_files = get_expected_files(dataset_name, total_len)
    merged_output_name = f"{dataset_prefix}/{dataset_name}_train_merged_pred_{note}.jsonl"
    merged_score_name = f"{dataset_prefix}/{dataset_name}_train_merged_pred_{note}_score.jsonl"
    merged_dataset = merge_jsonl_files(per_node_files, merged_output_name)
    assert len(merged_dataset) == total_len, f"Expected {total_len} records, but got {len(merged_dataset)}"
    print(f"✅ Successfully merged {dataset_name} dataset with {len(merged_dataset)} records.")

    # Compute scores
    for entry in merged_dataset:
        entry['pred_list'] = cem_score(entry)
        entry['pred_list'] = f1_score(entry)

    print(f"✅ Successfully computed scores for {dataset_name} dataset.")
    # Save the merged dataset with scores
    with open(merged_score_name, "w") as fout:
        for entry in merged_dataset:
            fout.write(json.dumps(entry) + "\n")
    print(f"✅ Successfully saved the merged dataset with scores to {merged_score_name}.")
    
    print(merged_dataset[0])
    # if dataset_name == "2wikimqa":
    #     test_cem_pred_list = cem_score(merged_dataset[0])
    #     test_f1_pred_list = f1_score(merged_dataset[0])
    #     import pdb; pdb.set_trace()