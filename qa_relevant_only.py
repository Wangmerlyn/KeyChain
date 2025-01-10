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
    "--subset", type=str, default="relevant", help="Options: validation or test"
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
        context = [total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d["context"]]
        per_qa_total_docs = [f"{t}\n{''.join(p)}" for t, p in d["context"]]
        supporting_facts = [total_docs_dict[c] for c in per_qa_total_docs for s in d['supporting_facts'] if s[0] in c]
        total_qas.append(
            {
                "query": d["question"],
                "outputs": [d["answer"]],
                "context": context,
                "supporting_facts": supporting_facts,
            }
        )
        assert len(supporting_facts) > 0, "No supporting facts found."       

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
elif "musique" in args.dataset:
    print("Reading MusiqueQA dataset")
    QAS, DOCS = read_musqiue(args.dataset)
elif "2wikimqa" in args.dataset:
    QAS, DOCS = read_2wikimqa(args.dataset)
else:
    raise NotImplementedError(f"{args.dataset} is not implemented.")


def generate_input_output(index,):
    curr_q = QAS[index]["query"]
    curr_a = QAS[index]["outputs"]
    curr_docs = QAS[index]["context"]
    curr_more = QAS[index].get("more_context", [])
    curr_supporting_facts = QAS[index].get("supporting_facts", [])
    all_docs = curr_supporting_facts
    random.Random(args.random_seed).shuffle(all_docs)
    all_docs = [DOCS[d] for d in all_docs]
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
    input_text, answer, question = generate_input_output(0)
    # Calculate the number of tokens in the example
    total_tokens = len(TOKENIZER.text_to_tokens(input_text + f" {answer}"))
    print(
        f"Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Docs: {num_docs}"
    )
    print("Number of documents:", num_docs)

    # Generate samples
    for index in tqdm(range(num_samples)):
        input_text, answer, question = generate_input_output(
            index + args.pre_samples,
        )
        length = len(TOKENIZER.text_to_tokens(input_text)) + tokens_to_generate

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
    save_file = args.save_dir / f"{args.save_name}" / f"{args.subset}.jsonl"
    save_file.parent.mkdir(parents=True, exist_ok=True)

    write_jsons = generate_samples(
        num_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
        save_dir=args.save_dir,
    )

    with open(save_file, "w") as f:
        for item in write_jsons:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
