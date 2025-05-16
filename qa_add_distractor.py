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
    QAS, DOCS = read_musqiue(args.dataset)
elif "2wikimqa" in args.dataset:
    QAS, DOCS = read_2wikimqa(args.dataset)
else:
    raise NotImplementedError(f"{args.dataset} is not implemented.")


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
    save_file = args.save_dir / f"{args.save_name}" / f"{args.subset}-{os.path.basename(args.tokenizer_path)}-num_sample_{args.num_samples}-max_seq_{args.max_seq_length}.jsonl"
    dataset_name =""
    if "hotpotqa" in args.save_dir:
        dataset_name = "hotpotqa"
    elif "musique" in args.save_dir:
        dataset_name = "musique"
    elif "2wikimqa" in args.save_dir:
        dataset_name = "2wikimqa"
    else:
        raise NotImplementedError(f"{args.save_dir} is not implemented.")
    # read the save file to write_json
    with open(save_file, 'r') as f:
        write_jsons = [json.loads(line) for line in f]
    distract_questions=args.max_seq_length // 1024 * 16 if args.max_seq_length // 1024 > 0 else 16
    if distract_questions>=0:
        for item in write_jsons:
            # Add distractor questions to the dataset
            # add one more entry list named distract_questions
            # sampled from all the questions in the dataset excluding the current question
            if len(QAS) <= distract_questions:
                continue
            distract_qas = random.sample(
                [q for i, q in enumerate(QAS) if i != item["index"]],
                min(distract_questions, len(QAS) - 1)
            )
            distract_questions_list = []
            # only keep the questions in the distractors
            for distract_q in distract_qas:
                distract_questions_list.append(distract_q["query"])
            item["distract_questions"] = distract_questions_list

    distractor_type = "chain"
    chain_distractor_config = {
        "num_chains": args.max_seq_length // 1024, # number of chains to generate, this should be a small number
        "num_uuids": 4,
    }
    import uuid_test
    for item in write_jsons:
        if distractor_type == "chain":
            chain_list = [ uuid_test.generate_uuid_chain(chain_distractor_config['num_uuids']) for _ in range(chain_distractor_config["num_chains"])]
            chain_string_list = []
            insert_input = True
            for index, chain in enumerate(chain_list):
                # Generate a string representation of the chain
                if insert_input:
                    chain_string_list.append(
                        uuid_test.generate_uuid_string_from_chain(
                            uuids=chain,
                            end_with=item['input']
                        )
                    )
                    # get the head of the chain with the question
                    chain_head_with_question = chain[0]
                    insert_input = False
                else:
                    chain_string_list.append(
                        uuid_test.generate_uuid_string_from_chain(
                            uuids=chain,
                            end_with=item['distract_questions'][index]
                        )
                    )
            # Flatten the list of strings
            flat_chain_string_list = sum(chain_string_list, [])
            # shuffle the flat list to mix the distractors
            random.shuffle(flat_chain_string_list)
            # find all the occurrences of sentence stoppers, i.e., '.' or '?' or '\n' in the context
            # randomly insert the distractor strings into the context 
            distractor_inserted_context = insert_distractor_into_context(
                context=item["context"],
                distractor_strings=flat_chain_string_list
            )
            item['distractor_context'] = distractor_inserted_context
            item['chain_head_with_question'] = chain_head_with_question


    
    resave_file = args.save_dir / f"{args.save_name}" / f"{args.subset}-{args.save_name}-dis_{distract_questions}-{os.path.basename(args.tokenizer_path)}-num_sample_{args.num_samples}-max_seq_{args.max_seq_length}.jsonl"
    with open(resave_file, "w") as f:
        for item in write_jsons:
            f.write(json.dumps(item) + "\n")

def insert_distractor_into_context(context, distractor_strings):
    """
    Insert distractor strings into the context at random positions.
    """
    if not distractor_strings:
        return context

    # find all the sentences stoppers in the context
    sentence_stoppers = [match.start() for match in re.finditer(r'[\n.?\n]', context)]
    if not sentence_stoppers:
        raise ValueError("No sentence stoppers found in the context to insert distractors.")
    if len(sentence_stoppers) < len(distractor_strings):
        raise ValueError(
            f"Not enough sentence stoppers in the context to insert all distractors. Found {len(sentence_stoppers)} but need {len(distractor_strings)}."
        )
    # insert distractor strings after random sentence stoppers in the context
    insertion_position = random.sample(
        range(len(sentence_stoppers)),
        len(distractor_strings)
    )
    insertion_position = [sentence_stoppers[i] for i in insertion_position]
    assert len(insertion_position) == len(distractor_strings), \
        f"Mismatch in insertion positions and distractor strings length: {len(insertion_position)} vs {len(distractor_strings)}"
    distractor_string_tupple_list = [(pos, distractor) for pos, distractor in zip(insertion_position, distractor_strings)]
    # sort the tupple list by position to insert in order of from big to small
    distractor_string_tupple_list = sorted(distractor_string_tupple_list, key=lambda x: x[0], reverse=True)
    distractor_inserted_context = context
    for pos, distractor in distractor_string_tupple_list:
        # insert the distractor string after the position of the sentence stopper
        insertion_extra_char = ' ' if distractor_inserted_context[pos] != '\n' else ''  # ensure readability
        distractor_inserted_context = (
            distractor_inserted_context[:pos + 1]  # +1 to include the stopper
            + f"{insertion_extra_char}{distractor}."  # add space around the distractor for readability
            + distractor_inserted_context[pos + 1:]
        )
    return distractor_inserted_context


if __name__ == "__main__":
    main()
