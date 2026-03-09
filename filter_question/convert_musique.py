import json
import os
import argparse
from pathlib import Path


def read_musqiue(file):
    with open(file) as f:
        data = [json.loads(line) for line in f]
    total_qas = []
    for d in data:
        if d["answerable"]:
            context_list = [
                f"{p['title']}\n{p['paragraph_text']}" for p in d["paragraphs"]
            ]
            total_qas.append(
                {
                    "id": d["id"],
                    "query": d["question"],
                    "outputs": d["answer"]
                    if isinstance(d["answer"], list)
                    else [d["answer"]],
                    "context": "\n".join(context_list),
                }
            )

    return total_qas


def main():
    parser = argparse.ArgumentParser(
        description="Convert MuSiQue dataset to merged format"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="musique/musique_full_v1.0_train.jsonl",
        help="Path to the MuSiQue training dataset (default: musique/musique_full_v1.0_train.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="filter_question/data",
        help="Directory to save the output (default: filter_question/data)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset_path}. "
            "Please download MuSiQue dataset first or provide correct path."
        )

    print(f"Reading dataset from: {args.dataset_path}")
    dataset = read_musqiue(args.dataset_path)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = Path(args.output_dir) / "musique_train_merged.jsonl"

    with open(output_path, "w") as f:
        for d in dataset:
            f.write(json.dumps(d) + "\n")

    print(f"✓ Saved {len(dataset)} records to {output_path}")


if __name__ == "__main__":
    main()
