import json
import os

# Read Hotpot QA dataset
def read_hotpotqa(file):
    with open(file) as f:
        data = json.load(f)

    # import pdb; pdb.set_trace()


    # total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]]

    # total_docs = sorted(list(set(total_docs)))
    # total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        current_context = [
            f"{t}\n{''.join(p)}" for t, p in d["context"]
        ]
        total_qas.append(
            {
                "id": d["_id"],
                "query": d["question"],
                "outputs": d["answer"]
                    if isinstance(d["answer"], list) else [d["answer"]],
                "context": "\n".join(current_context),
                "level": d["level"],
            }
        )

    return total_qas

dataset_path = "hotpot_train_v1.1.json"
dataset_save_dir = "filter_question/data"

if __name__ == "__main__":
    dataset = read_hotpotqa(dataset_path)
    os.makedirs(dataset_save_dir, exist_ok=True)
    with open(f"{dataset_save_dir}/hotpotqa_train_merged.jsonl", "w") as f:
        for d in dataset:
            f.write(json.dumps(d) + "\n")
    print(f"Saved {len(dataset)} records to {dataset_save_dir}/hotpotqa_train_merged.jsonl")
