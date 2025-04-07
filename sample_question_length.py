import os
import argparse
import json
import random
from tqdm import tqdm
import numpy as np


def sample_questions_by_context_length(
    question_lists, length_distribution, shuffle=True
):
    """
    Sample questions from different context length lists based on a given length distribution.

    Args:
        question_lists (dict): A dictionary where keys are context lengths and values are lists of questions (dictionaries).
        length_distribution (dict): A dictionary where keys are context lengths and values are the desired proportions.
        shuffle (bool): Whether to shuffle and randomly sample questions or to sample sequentially based on the original order.

    Returns:
        list: A list of sampled questions where each question is mapped to one context length.
    """
    # Normalize the distribution to ensure it sums to 1
    total = sum(length_distribution.values())
    normalized_distribution = {k: v / total for k, v in length_distribution.items()}

    # Determine the number of questions to sample for each context length
    total_questions = len(
        next(iter(question_lists.values()))
    )  # All lists have the same number of questions
    samples_per_length = {
        length: int(normalized_distribution[length] * total_questions)
        for length in normalized_distribution
    }

    sampled_questions = []

    if shuffle:
        # Shuffle indices for random sampling
        num_questions = len(next(iter(question_lists.values())))
        indices = list(range(num_questions))
        random.shuffle(indices)
        used_indices = set()

        for length, num_samples in samples_per_length.items():
            length_samples = []
            available_indices = [idx for idx in indices if idx not in used_indices]

            for idx in available_indices[:num_samples]:
                length_samples.append(question_lists[length][idx])
                used_indices.add(idx)

            sampled_questions.extend(length_samples)
    else:
        # Sample sequentially without shuffling
        for length in question_lists:
            sampled_questions.extend(
                question_lists[length][: samples_per_length[length]]
            )

    return sampled_questions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample questions based on context length."
    )
    parser.add_argument(
        "--question_lists_prefix", type=str, help="Prefix for the question lists."
    )
    parser.add_argument(
        "--gpt_answer_list_path", type=str, help="Path to the GPT answer list."
    )
    parser.add_argument(
        "--length_distribution", type=str, help="Desired length distribution."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--shuffle", action="store_true", help="Whether to shuffle")

    return parser.parse_args()


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


user_prompt_template = "Answer the question based on the given passages.\n\nThe following are given passages.\n{context}\n\nNow, answer the question: {input}"

if __name__ == "__main__":
    args = parse_args()
    fix_seed(args.seed)
    length_distribution = json.loads(args.length_distribution)
    question_lists_prefix = args.question_lists_prefix
    gpt_answer_list_path = args.gpt_answer_list_path
    shuffle = args.shuffle
    question_list = {}
    for length in length_distribution:
        question_list[length] = read_jsonl(f"{question_lists_prefix}_{length}.jsonl")
    sampled_questions = sample_questions_by_context_length(
        question_list, length_distribution, shuffle
    )
    gpt_answer_list = read_jsonl(gpt_answer_list_path)
    # sort by index
    gpt_answer_list = sorted(gpt_answer_list, key=lambda x: x["index"])
    sampled_questions = sorted(sampled_questions, key=lambda x: x["index"])
    # sanity check
    print("running sanity check")
    assert len(sampled_questions) == len(
        gpt_answer_list
    ), "The number of questions and answers do not match."
    for question_pair, response_pair in tqdm(zip(sampled_questions, gpt_answer_list)):
        assert (
            question_pair["index"] == response_pair["index"]
        ), "The indices do not match."
        assert (
            question_pair["input"] == response_pair["input"]
        ), "The inputs do not match."
    for question_pair, response_pair in tqdm(zip(sampled_questions, gpt_answer_list)):
        # TODO: only the first element of the reasonings is used
        question_pair["response"] = response_pair["reasonings"][0]

    output_file_path = f"{question_lists_prefix}-dist_{args.length_distribution}-shuffle_{shuffle}-sampled.jsonl"
    # Write the sampled questions to a JSONL file
    with open(output_file_path, "w") as f:
        for question in sampled_questions:
            json.dump(question, f)
            f.write("\n")

    train_file_path = f"{os.path.dirname(question_lists_prefix)}/{'_'.join([str(k)+'-'+str(v) for k, v in length_distribution.items()])}/train_sft.jsonl"
    os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
    train_file_list = []
    for question_pair in sampled_questions:
        train_pair = {}
        train_pair["prompt_id"] = str(question_pair["index"])
        train_pair["messages"] = [
            {
                "role": "user",
                "content": user_prompt_template.format(
                    context=question_pair["context"], input=question_pair["input"]
                ),
            },
            {"role": "assistant", "content": question_pair["response"]},
        ]
        train_file_list.append(train_pair)

    with open(train_file_path, "w") as f:
        for train_pair in train_file_list:
            json.dump(train_pair, f)
            f.write("\n")
