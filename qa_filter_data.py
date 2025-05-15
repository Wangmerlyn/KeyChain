import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np
from tokenizer import select_tokenizer

parser = argparse.ArgumentParser()
# Basic Configurations
parser.add_argument(
    "--save_dir", type=Path, required=True, help="dataset folder to save dataset"
)
parser.add_argument(
    "--save_name", type=str, required=True, help="name of the save dataset jsonl file"
)
parser.add_argument(
    "--subset", type=str, default="validation", help="Options: validation or test"
)
parser.add_argument(
    "--tokenizer_path", type=str, required=True, help="path to the tokenizer model"
)
parser.add_argument(
    "--tokenizer_type", type=str, default="nemo", help="[Options] hf, openai."
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    required=True,
    help="max sequence length including all input tokens and generated tokens.",
)
parser.add_argument(
    "--tokens_to_generate",
    type=int,
    required=True,
    help="expected generated token amount.",
)
parser.add_argument(
    "--num_samples", type=int, required=True, help="number of samples to generate"
)
parser.add_argument(
    "--pre_samples", type=int, default=0, help="number of samples are already generated"
)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--template", type=str, required=True, help="prompt template")
parser.add_argument(
    "--remove_newline_tab",
    action="store_true",
    help="remove `\n` and `\t` in all strings.",
)

# Complexity Configurations
parser.add_argument("--dataset", type=str, required=True, help="dataset file")
parser.add_argument("--filter_ids_path", type=str, default=None, help="path to filter ids file")

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)


TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)


# Read SQuAD QA dataset
def read_squad(file):
    with open(file) as f:
        data = json.load(f)

    total_docs = [p["context"] for d in data["data"] for p in d["paragraphs"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data["data"]:
        more_docs = [total_docs_dict[p["context"]] for p in d["paragraphs"]]
        for p in d["paragraphs"]:
            for qas in p["qas"]:
                if not qas["is_impossible"]:
                    total_qas.append(
                        {
                            "query": qas["question"],
                            "outputs": [a["text"] for a in qas["answers"]],
                            "context": [total_docs_dict[p["context"]]],
                            "more_context": [
                                idx
                                for idx in more_docs
                                if idx != total_docs_dict[p["context"]]
                            ],
                        }
                    )

    return total_qas, total_docs


# Read Hotpot QA dataset
def read_hotpotqa(file):
    with open(file) as f:
        data = json.load(f)

    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        total_qas.append(
            {
                "id": d["_id"],
                "query": d["question"],
                "outputs": [d["answer"]],
                "context": [
                    total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d["context"]
                ],
            }
        )

    return total_qas, total_docs


def read_musqiue(file):
    with open(file) as f:
        # read jsonl file
        data = [json.loads(line) for line in f]
    total_docs = []
    for d in data:
        for p in d["paragraphs"]:
            total_docs.append(f"{p['title']}\n{p['paragraph_text']}")
    print(len(total_docs))
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    total_qas = []
    for d in data:
        # This only deals with questions that are answerable given the context
        if d["answerable"]:
            total_qas.append(
                {
                    "id": d["id"],
                    "query": d["question"],
                    "outputs": [d["answer"]],
                    "context": [
                        total_docs_dict[f"{p['title']}\n{p['paragraph_text']}"]
                        for p in d["paragraphs"]
                    ],
                }
            )

    return total_qas, total_docs


def read_2wikimqa(file):
    with open(file) as f:
        data = json.load(f)

    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        total_qas.append(
            {
                "id":d['_id'],
                "query": d["question"],
                "outputs": [d["answer"]],
                "context": [
                    total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d["context"]
                ],
            }
        )

    return total_qas, total_docs


DOCUMENT_PROMPT = "Passage {i}:\n{document}"
if "hotpot" in args.dataset:
    QAS, DOCS = read_hotpotqa(args.dataset)
    dataset_name = "hotpotqa"
elif "musique" in args.dataset:
    print("Reading MusiqueQA dataset")
    QAS, DOCS = read_musqiue(args.dataset)
    dataset_name = "musique"
elif "2wikimqa" in args.dataset:
    QAS, DOCS = read_2wikimqa(args.dataset)
    dataset_name = "2wikimqa"
else:
    raise NotImplementedError(f"{args.dataset} is not implemented.")

assert args.filter_ids_path is not None, "Filter ids path is required."
if args.filter_ids_path is not None:
    print(f"Filtering questions using {args.filter_ids_path}")
    with open(args.filter_ids_path, "r") as f:
        filter_data_entries = [json.loads(line) for line in f]
        assert len(filter_data_entries) > 0, "No filter data entries found."
        assert "id" in filter_data_entries[0], "Filter data entries must contain 'id' field."
        filter_ids = set([filter_data_entry["id"] for filter_data_entry in filter_data_entries])
    QAS = [
        qas
        for qas in QAS
        if qas["id"] in filter_ids
    ]
    print(f"Filtered down to {len(QAS)} questions.")


def generate_input_output(index, num_docs):
    curr_q = QAS[index]["query"]
    curr_a = QAS[index]["outputs"]
    curr_docs = QAS[index]["context"]
    curr_more = QAS[index].get("more_context", [])
    if num_docs < len(DOCS):
        # If we have more documents than the number of documents we want to use
        if len(curr_docs) > num_docs:
            all_docs = random.sample(curr_docs, num_docs)

        elif (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [
                i for i, d in enumerate(DOCS) if i not in curr_docs + curr_more
            ]
            all_docs = (
                curr_docs
                + curr_more
                + random.sample(
                    addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more))
                )
            )
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))

        all_docs = [DOCS[idx] for idx in all_docs]
    else:
        all_docs = DOCS

    random.Random(args.random_seed).shuffle(all_docs)

    context = "\n\n".join(
        [DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(all_docs)]
    )
    input_text = args.template.format(
        context=context,
    )
    return input_text, curr_a, curr_q


def generate_samples(
    num_samples: int, max_seq_length: int, save_dir: str, incremental: int = 10
):

    write_jsons = []
    tokens_to_generate = args.tokens_to_generate

    # Find the perfect num_docs
    num_docs = incremental

    total_tokens = 0  # Track the total tokens generated for this example
    while total_tokens + tokens_to_generate < max_seq_length:
        input_text, answer, question = generate_input_output(0, num_docs)
        # Calculate the number of tokens in the example
        total_tokens = len(TOKENIZER.text_to_tokens(input_text + f" {answer}"))
        print(
            f"Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Docs: {num_docs}"
        )
        if total_tokens + tokens_to_generate > max_seq_length:
            num_docs -= incremental
            break

        num_docs += incremental
        if num_docs > len(DOCS):
            num_docs = len(DOCS)
            break
    print("Number of documents:", num_docs)

    # Generate samples
    for index in tqdm(range(num_samples)):
        used_docs = num_docs
        while True:
            try:
                input_text, answer, question = generate_input_output(
                    index + args.pre_samples, used_docs
                )
                length = len(TOKENIZER.text_to_tokens(input_text)) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except:
                if used_docs > incremental:
                    used_docs -= incremental
                else:
                    break

        if args.remove_newline_tab:
            input_text = " ".join(
                input_text.replace("\n", " ").replace("\t", " ").strip().split()
            )

        formatted_output = {
            "index": index,
            "input": question,
            "context": input_text,
            "answers": answer,
            "length": length,
        }
        write_jsons.append(formatted_output)

    return write_jsons


def main():
    if args.num_samples < 0:
        args.num_samples = len(QAS)
    save_file = args.save_dir / f"{args.save_name}" / f"{dataset_name}_{args.subset}-num_sample_{args.num_samples}-max_seq_{args.max_seq_length}.jsonl"
    save_file.parent.mkdir(parents=True, exist_ok=True)

    write_jsons = generate_samples(
        num_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
        save_dir=args.save_dir,
    )

    # distract_questions=100
    # if distract_questions>=0:
    #     for item in write_jsons:
    #         # Add distractor questions to the dataset
    #         # add one more entry list named distract_questions
    #         # sampled from all the questions in the dataset excluding the current question
    #         if len(QAS) <= distract_questions:
    #             continue
    #         distract_qas = random.sample(
    #             [q for i, q in enumerate(QAS) if i != item["index"]],
    #             min(distract_questions, len(QAS) - 1)
    #         )
    #         distract_questions_list = []
    #         # only keep the questions in the distractors
    #         for distract_q in distract_qas:
    #             distract_questions_list.append(distract_q["query"])
    #         item["distract_questions"] = distract_questions_list

    with open(save_file, "w") as f:
        for item in write_jsons:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
