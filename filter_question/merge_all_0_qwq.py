import os
import json
import re
import string
from collections import Counter

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
    # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()   
        # if qwq_pred_list not in entry, use pred_list instead
    if 'qwq_pred_list' not in entry:
        entry['qwq_pred_list'] = entry['pred_list']
    pred_list = entry['qwq_pred_list']
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
    pred_list = entry['qwq_pred_list']
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

def merge_node_outputs(dataset_name, total_len, num_total_nodes, output_path):
    base_dir = dataset_prefix
    all_data = []

    # Compute chunk size (ceiling division)
    chunk_size = (total_len + num_total_nodes - 1) // num_total_nodes

    for node_id in range(num_total_nodes):
        start_idx = node_id * chunk_size
        end_idx = min(start_idx + chunk_size, total_len)
        filename = f"{dataset_name}_train_merged_pred_{start_idx}_{end_idx}_all_0_qwq.jsonl"
        file_path = os.path.join(base_dir, filename)

        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        with open(file_path, "r") as f:
            for line in f:
                try:
                    all_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line in {file_path}: {e}")

    # Write merged output
    with open(output_path, "w") as f_out:
        for entry in all_data:
            f_out.write(json.dumps(entry) + "\n")

    print(f"✅ Merged {len(all_data)} entries to {output_path}")
    return all_data




dataset_dict = {
    "hotpotqa": 11786,
    "musique": 6736,
    "2wikimqa": 20798,
}
total_nodes = 4
dataset_prefix = "/home/aiscuser/LongContextDataSynth/filter_question/data"

if __name__ == "__main__":
    for dataset, total_len in dataset_dict.items():
        output_path = os.path.join(dataset_prefix, f"{dataset}_train_merged_pred_dist_run_0_correct_all_0_qwq.jsonl")
        # assert not os.path.exists(output_path), f"Output file already exists: {output_path}"
        merged_dataset = merge_node_outputs(dataset_name=dataset, total_len=total_len, num_total_nodes=total_nodes, output_path=output_path)

        # Compute scores
        for entry in merged_dataset:
            entry['qwq_pred_list'] = cem_score(entry)
            entry['qwq_pred_list'] = f1_score(entry)
        cem_stats = {}
        f1_stats = {}
        for entry in merged_dataset:
            pred_list = entry['pred_list']
            avg_cem_score = sum([pred['cem_score'] for pred in pred_list]) / len(pred_list)
            total_cem_score = sum([pred['cem_score'] for pred in pred_list])
            avg_f1_score = sum([pred['f1_score'] for pred in pred_list]) / len(pred_list)

            high_f1_count = sum(1 for pred in pred_list if pred['f1_score'] > 0.75)


            entry['avg_cem_score'] = avg_cem_score
            entry['total_cem_score'] = total_cem_score
            entry['avg_f1_score'] = avg_f1_score
            entry["high_f1_count"] = high_f1_count
            # since we have 8 pred in pred_list
            # count how many pred list have a total cem score in 0,1,2,3,4,5,6,7,8 each
            if total_cem_score not in cem_stats:
                cem_stats[total_cem_score] = 0
            cem_stats[total_cem_score] += 1
            if high_f1_count not in f1_stats:
                f1_stats[high_f1_count] = 0
            f1_stats[high_f1_count] += 1
            # print(f"avg_cem_score: {avg_cem_score}, total_cem_score: {total_cem_score}, avg_f1_score: {avg_f1_score}, high_f1_count: {high_f1_count}")
        print(f"dataset_name: {dataset}")
        print(f"cem_stats: {cem_stats}")
        print(f"f1_stats: {f1_stats}")

        # Filter entries where total_cem_score is not 0 or 8
        filtered_entries = [
            entry for entry in merged_dataset
            if entry["total_cem_score"] != 0 and entry["total_cem_score"] != 8
        ]

        # Save filtered entries to jsonl
        filtered_output_path = os.path.join(dataset_prefix, f"{dataset}_all_0_qwq_some_right_some_wrong_filtered_entries.jsonl")
        with open(filtered_output_path, "w") as f:
            for entry in filtered_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"✅ Saved {len(filtered_entries)} filtered entries to {filtered_output_path}")