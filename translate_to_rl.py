import os
import json

jsonl_list = [
    "musique/validation-llama-3.1-8B-instruct-num_sample_10000-max_seq_8192.jsonl",
]
# save_dir_name = "longcontext_train_30k"
save_dir_name = "musique_8k"

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        jsonl_list = [json.loads(line) for line in f]
    return jsonl_list
            

def translate_to_rl_format(jsonl_list):
    prompt_format = "Answer the question based on the given passages following these steps: \n\n Start with a `<think>` and break down the question into key elements;\n As you reason, use the marker `wait` to pause and reflect on details when necessary;\n Provide a clear, step-by-step explanation of your reasoning, ensuring each step is backed by the passages;\n End your response with a final line starting with `Answer:` followed by your answer.\n Keep your reasoning rigorous, precise, and succinct.\n\n The following are given passages.\n {context}\n\n Question: {input}"
    new_list = []
    for item in jsonl_list:
        # input => question
        # context => context
        # new[answer] = answers[0]
        # answers => answers
        # messages = [{"role": "user", "content": prompt_format.format(context=item['context'], input=item['question'])}]
        new_item = {
            "question": item['input'],
            "context": item['context'],
            "answers": item['answers'],
            "answer": item['answers'][0],
            "messages": [{"role": "user", "content": prompt_format.format(context=item['context'], input=item['input'])}]
        }
        new_list.append(new_item)
    return new_list

# run translate on all jsonl files
# merge all jsonl files

merged_jsonl = []
for jsonl in jsonl_list:
    jsonl_data = read_jsonl(jsonl)
    print(len(jsonl_data))
    new_jsonl = translate_to_rl_format(jsonl_data)
    print(len(new_jsonl))
    merged_jsonl.extend(new_jsonl)
print(len(merged_jsonl))

# save to jsonl
os.makedirs(f"rl_datasets/{save_dir_name}", exist_ok=True)
with open(f"rl_datasets/{save_dir_name}/train.jsonl", "w") as f:
    for item in merged_jsonl:
        f.write(json.dumps(item) + "\n")
