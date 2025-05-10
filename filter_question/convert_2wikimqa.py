import os
import json

def read_2wikimqa(file):
    with open(file) as f:
        data = json.load(f)

    # total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]]
    # total_docs = sorted(list(set(total_docs)))
    # total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    # import pdb; pdb.set_trace()
    total_qas = []
    for d in data:
        current_context = [
            f"{t}\n{''.join(p)}" for t, p in d["context"]
        ]
        total_qas.append(
            {
                "id": d["_id"],
                "query": d["question"],
                "outputs": [d["answer"]],
                "context": "\n".join(current_context),
            }
        )

    return total_qas

dataset_path = "/mnt/longcontext/models/siyuan/test_code/longcontext_syth/2wikimqa/train.json"

if __name__ == "__main__":
    dataset = read_2wikimqa(dataset_path)
    # import pdb; pdb.set_trace()
    dataset_dir = "/mnt/longcontext/models/siyuan/test_code/longcontext_syth/filter_question/data"
    os.makedirs(dataset_dir, exist_ok=True)
    with open(f"{dataset_dir}/2wikimqa_train_merged.jsonl", "w") as f:
        for d in dataset:
            f.write(json.dumps(d) + "\n")
    print(f"Saved {len(dataset)} records to {dataset_dir}/2wikimqa_train_merged.jsonl")