import json
import os
import argparse
from pathlib import Path


# Read Hotpot QA dataset
def read_hotpotqa(file):
    with open(file) as f:
        data = json.load(f)

    total_qas = []
    for d in data:
        current_context = [f"{t}\n{''.join(p)}" for t, p in d["context"]]
        total_qas.append(
            {
                "id": d["_id"],
                "query": d["question"],
                "outputs": d["answer"]
                if isinstance(d["answer"], list)
                else [d["answer"]],
                "context": "\n".join(current_context),
                "level": d["level"],
            }
        )

    return total_qas


def main():
    parser = argparse.ArgumentParser(
        description="Convert HotpotQA dataset to merged format"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="hotpotqa/hotpot_train_v1.1.json",
        help="Path to the HotpotQA training dataset (default: hotpotqa/hotpot_train_v1.1.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="filter_question/data",
        help="Directory to save the output (default: filter_question/data)",
    )
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset_path}. "
            "Please download HotpotQA dataset first or provide correct path."
        )

    print(f"Reading dataset from: {args.dataset_path}")
    dataset = read_hotpotqa(args.dataset_path)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = Path(args.output_dir) / "hotpotqa_train_merged.jsonl"

    with open(output_path, "w") as f:
        for d in dataset:
            f.write(json.dumps(d) + "\n")

    print(f"✓ Saved {len(dataset)} records to {output_path}")


if __name__ == "__main__":
    main()
