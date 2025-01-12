import argparse
import json
import asyncio
import re
import os

from openai import AzureOpenAI
from azure.identity import (
    DefaultAzureCredential,
    ChainedTokenCredential,
    AzureCliCredential,
    get_bearer_token_provider,
)

scope = "api://trapi/.default"
credential = get_bearer_token_provider(
    ChainedTokenCredential(
        AzureCliCredential(),
        DefaultAzureCredential(
            exclude_cli_credential=True,
            exclude_environment_credential=True,
            exclude_shared_token_cache_credential=True,
            exclude_developer_cli_credential=True,
            exclude_powershell_credential=True,
            exclude_interactive_browser_credential=True,
            exclude_visual_studio_code_credentials=True,
            managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
        ),
    ),
    scope,
)

api_version = "2024-10-21"
model_name = "gpt-4o"
model_version = "2024-11-20"
deployment_name = re.sub(r"[^a-zA-Z0-9-_]", "", f"{model_name}_{model_version}")
instance = "gcr/shared"
endpoint = f"https://trapi.research.microsoft.com/{instance}"

client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=credential,
    api_version=api_version,
)


def read_data(file_path):
    """Read question and answer data from a file"""

    def read_json_lines(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def read_json(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if file_path.endswith(".jsonl"):
        return read_json_lines(file_path)
    elif file_path.endswith(".json"):
        return read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


async def call_gpt_api(
    item, prompt_template, num_sequences, temperature, retry_attempts=3
):
    """Asynchronously call Azure OpenAI API to generate reasoning process for one item"""
    question = item["input"]
    context = item["context"]
    answer = item["answers"][0]

    messages = [
        {
            "role": "user",
            "content": prompt_template.format(
                question=question, context=context, answer=answer
            ),
        }
    ]

    for attempt in range(retry_attempts):
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=deployment_name,
                messages=messages,
                n=num_sequences,
                temperature=temperature,
            )

            reasonings = [choice.message.content.strip() for choice in response.choices]
            return {
                "index": item["index"],
                "input": question,
                "context": context,
                "answer": answer,
                "reasonings": reasonings,
            }

        except Exception as e:
            print(
                f"Error processing item {item['index']} on attempt {attempt + 1}: {e}"
            )
            if attempt < retry_attempts - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff
            else:
                return {
                    "index": item["index"],
                    "input": question,
                    "context": context,
                    "answer": answer,
                    "reasonings": ["Error: " + str(e)],
                }


async def generate_inference(input_data, prompt_template, num_sequences, temperature):
    """Generate reasoning process for all items asynchronously"""
    tasks = [
        call_gpt_api(item, prompt_template, num_sequences, temperature)
        for item in input_data
    ]
    results = await asyncio.gather(*tasks)
    return results


def save_results(file_path, results):
    """Save reasoning results to a file"""
    if file_path.endswith(".jsonl"):
        with open(file_path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")
    elif file_path.endswith(".json"):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate reasoning process for question-answer pairs"
    )
    parser.add_argument("--input_file", type=str, required=True, help="Input file path")
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output file path"
    )
    parser.add_argument(
        "--prompt_template_type", type=str, default="basic", help="Prompt template type"
    )
    parser.add_argument(
        "--num_sequences", type=int, default=1, help="Number of sequences"
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    return parser.parse_args()


prompt_template_dict = {
    "basic": "Question: {question}\n\nContext: {context}\n\nAnswer: {answer}\n\nProvide a detailed reasoning process to arrive at the answer based on the given context.",
    "normal": 'Question: {question}\n\nContext: {context}\n\nAnswer: {answer}\n\nYour task:\nPlease produce a clear and logically sound explanation that shows how the answer is derived from the context. Finally, present your final conclusion after "Final answer:".',
    "cot": ' Question: {question}\n\nContext: {context}\n\nAnswer: {answer}\n\nYour task:\nPlease produce a step-by-step reasoning that uncovers the path from the context to the final answer. Clearly demonstrate each inference or sub-action in your explanation. Finally, present your final conclusion after "Final answer:"',
    "cot-cite": 'Question: {question}\n\nContext: {context}\n\nAnswer: {answer}\n\nYour task:\nPlease produce a structured, step-by-step reasoning that references any relevant parts of the context in quotes ("") whenever you use them. Finally, present your final conclusion after "Final answer:".',
    "mcts": 'Question: {question}\n\nContext: {context}\n\nAnswer: {answer}\n\nYour task:\nPlease adopt a multi-phase approach to thoroughly examine the given context, refining your ideas at each stage. Provide the reasoning details step by step. Finally, present your final conclusion after "Final answer:".',
}

if __name__ == "__main__":
    args = parse_args()
    input_file = args.input_file
    output_file = args.output_file
    prompt_template = prompt_template_dict[args.prompt_template_type]
    num_sequences = args.num_sequences
    temperature = args.temperature
    input_data = read_data(input_file)

    results = asyncio.run(
        generate_inference(input_data, prompt_template, num_sequences, temperature)
    )

    save_results(output_file, results)
    print(f"Reasoning results saved to {output_file}")
