import argparse
import json
import os
from pprint import pprint
import vllm
from vllm import LLM, SamplingParams


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Script for long-context QA")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/longcontext/models/siyuan/llama3/Qwen2.5-32B-Instruct",
        help="Path to the model directory"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=['hotpotqa', 'musique', '2wikimqa'],
        default="hotpotqa",
        help="Dataset to use"
    )

    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index of the dataset"
    )

    parser.add_argument(
        "--end_idx",
        type=int,
        default=1000,
        help="End index of the dataset"
    )

    parser.add_argument(
        "--note",
        type=str,
        default="in_test",
        help="Optional note or tag for the experiment"
    )

    parser.add_argument("--dataset_path_prefix", type=str, default="/mnt/longcontext/models/siyuan/test_code/longcontext_syth/filter_question/data/", help="Path prefix for the dataset")

    parser.add_argument(
        "--prompt_template",
        type=str,
        default="The following are given passages.\n{context}\n\nAnswer the question based on the given passages. Please think step by step before answering. After thinking, output the final answer in the format of 'The answer is <your answer here>'\n\nQuestion: {input}",
        help="Prompt template to use for input formatting"
    )

    parser.add_argument(
        "--tp_size",
        type=int,
        default=4,
        help="Tensor parallel size"
    )

    return parser.parse_args()


def load_jsonl(file):
    with open(file) as f:
        data = [json.loads(line) for line in f]
    return data


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    dataset_name = args.dataset_name
    start_idx = args.start_idx
    end_idx = args.end_idx
    note = args.note
    prompt_template = args.prompt_template
    pprint(args)

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=2048,
        n=8,
        )

    model_engine = LLM(
        model=model_path,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=args.tp_size,
    )
    # dataset_path = f"/mnt/longcontext/models/siyuan/test_code/longcontext_syth/filter_question/data/{dataset_name}_train_merged.jsonl"
    dataset_path = f"{args.dataset_path_prefix}/{dataset_name}_train_merged.jsonl"
    dataset = load_jsonl(dataset_path)
    end_idx = min(end_idx, len(dataset))
    assert start_idx < end_idx, f"start_idx {start_idx} should be less than end_idx {end_idx}"
    dataset = dataset[start_idx:end_idx]
    print(f"Loaded {len(dataset)} records from {dataset_name}_train_merged.jsonl")
    print(f"Start idx: {start_idx}, End idx: {end_idx}")
    prompts_list = []
    for i, d in enumerate(dataset):
        # add a new entry to the dataset
        d['formatted_input'] = prompt_template.format(context=d['context'], input=d['query'])
        prompts_list.append([{'role': 'user', 'content': d['formatted_input']}])

    outputs = model_engine.chat(
        messages=prompts_list,
        sampling_params=sampling_params,
        add_generation_prompt=True
    )
    for output, d in zip(outputs, dataset):
        # there are multiple text for each input question
        # add the output to the dataset
        d['pred_list'] = [{"pred": choice.text} for choice in output.outputs]
        d['input_tokens'] = len(output.prompt_token_ids) 

    # import pdb; pdb.set_trace()
    output_file = f"{os.path.dirname(dataset_path)}/{dataset_name}_train_merged_pred_{start_idx}_{end_idx}_{note}.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for d in dataset:
            f.write(json.dumps(d) + "\n")