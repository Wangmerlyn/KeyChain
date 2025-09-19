import json
import os


def read_musqiue(file):
    with open(file) as f:
        # read jsonl file
        data = [json.loads(line) for line in f]
    total_qas = []
    for d in data:
        # This only deals with questions that are answerable given the context
        if d["answerable"]:
            # context from key paragraphs
            context_list = [f"{p['title']}\n{p['paragraph_text']}" for p in d["paragraphs"]]
            total_qas.append(
                {
                    "id": d["id"],
                    "query": d["question"],
                    # check if d['answer'] is a list
                    # if is a list, then outputs = d['answer']
                    # if not, then outputs = [d['answer']]
                    "outputs": d["answer"]
                    if isinstance(d["answer"], list) else [d["answer"]],
                    "context": "\n".join(context_list),
                }
            )

    return total_qas


dataset_path = "data/musique_full_v1.0_train.jsonl"

if __name__ == "__main__":
    # read the dataset
    dataset = read_musqiue(dataset_path)
    # import pdb; pdb.set_trace()
    # save dataset to jsonl file
    dataset_dir = "filter_question/data"
    os.makedirs(dataset_dir, exist_ok=True)
    with open(f"{dataset_dir}/musique_train_merged.jsonl", "w") as f:
        for d in dataset:
            f.write(json.dumps(d) + "\n")
    print(f"Saved {len(dataset)} records to {dataset_dir}/musique_train_merged.jsonl")