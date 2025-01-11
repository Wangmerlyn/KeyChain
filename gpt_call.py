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
            # Exclude other credentials we are not interested in.
            exclude_environment_credential=True,
            exclude_shared_token_cache_credential=True,
            exclude_developer_cli_credential=True,
            exclude_powershell_credential=True,
            exclude_interactive_browser_credential=True,
            exclude_visual_studio_code_credentials=True,
            # DEFAULT_IDENTITY_CLIENT_ID is a variable exposed in
            # Azure ML Compute jobs that has the client id of the
            # user-assigned managed identity in it.
            # See https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-authentication#compute-cluster
            # In case it is not set the ManagedIdentityCredential will
            # default to using the system-assigned managed identity, if any.
            managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
        ),
    ),
    scope,
)

api_version = "2024-10-21"  # Ensure this is a valid API version see: https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release
model_name = "gpt-4o"  # Ensure this is a valid model name
model_version = "2024-11-20"  # Ensure this is a valid model version
deployment_name = re.sub(
    r"[^a-zA-Z0-9-_]", "", f"{model_name}_{model_version}"
)  # If your Endpoint doesn't have harmonized deployment names, you can use the deployment name directly: see: https://aka.ms/trapi/models
instance = "gcr/shared"  # See https://aka.ms/trapi/models for the instance name, remove /openai (library adds it implicitly)
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


async def call_gpt_api(item, retry_attempts=3):
    """Asynchronously call Azure OpenAI API to generate reasoning process for one item"""
    question = item["input"]
    context = item["context"]
    answer = item["answers"][0]

    # Construct chat messages
    messages = [
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext: {context}\n\nAnswer: {answer}\n\nProvide a detailed reasoning process to arrive at the answer based on the given context.",
        }
    ]

    for attempt in range(retry_attempts):
        try:
            # Asynchronous GPT API call using Azure OpenAI client
            response = await asyncio.to_thread(
                client.chat.completions.create, model=deployment_name, messages=messages
            )

            reasoning = response.choices[0].message.content.strip()
            return {
                "index": item["index"],
                "input": question,
                "context": context,
                "answer": answer,
                "reasoning": reasoning,
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
                    "reasoning": "Error: " + str(e),
                }


async def generate_inference(input_data):
    """Generate reasoning process for all items asynchronously"""
    tasks = [call_gpt_api(item) for item in input_data]
    results = await asyncio.gather(*tasks)
    return results


def save_results(file_path, results):
    """Save reasoning results to a file"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Read input data
    input_file = "/mnt/longcontext/models/siyuan/test_code/longcontext_syth/hotpotqa/relevant.jsonl"  # Input file path
    output_file = "/mnt/longcontext/models/siyuan/test_code/longcontext_syth/hotpotqa/relevant_answer.jsonl"  # Output file path

    input_data = read_data(input_file)[:10]

    # Generate reasoning process asynchronously
    results = asyncio.run(generate_inference(input_data))

    # Save results
    save_results(output_file, results)
    print(f"Reasoning results saved to {output_file}")
