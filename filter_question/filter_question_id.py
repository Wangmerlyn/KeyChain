import os
import json
import re
import string
from collections import Counter


def load_jsonl(file_path):
    """Load a JSONL file and return a list of dictionaries."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

dataset_prefix = "filter_question/data"

dataset_dict = {
    "hotpotqa": 11786,
    "musique": 6736,
    "2wikimqa": 20798,
}

for dataset_name in dataset_dict:
    qwq_srsw_data_path = os.path.join(dataset_prefix, f"{dataset_name}_all_0_qwq_some_right_some_wrong_filtered_entries.jsonl")
    # see if the file exists
    if not os.path.exists(qwq_srsw_data_path):
        raise FileNotFoundError(f"File not found: {qwq_srsw_data_path}")
    qwq_srsw_dataset = load_jsonl(qwq_srsw_data_path)
    print(f"Loaded {len(qwq_srsw_dataset)} entries from {qwq_srsw_data_path}")
    correct_1_6_data_path = os.path.join(dataset_prefix, f"{dataset_prefix}/{dataset_name}_train_merged_pred_dist_run_1_6_correct.jsonl")
    correct_1_6_dataset = load_jsonl(correct_1_6_data_path)
    # merge the two datasets
    merged_dataset = qwq_srsw_dataset + correct_1_6_dataset
    print(f"Merged dataset has {len(merged_dataset)} entries")
    # save the merged dataset
    merged_output_path = os.path.join(dataset_prefix, f"{dataset_name}_train_merged_pred_dist_run_1_6_all_0_qwq_srsw.jsonl")
    with open(merged_output_path, 'w') as f:
        for entry in merged_dataset:
            f.write(json.dumps(entry) + '\n')
    print(f"Saved merged dataset to {merged_output_path}")