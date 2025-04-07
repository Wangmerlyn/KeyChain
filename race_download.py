import os
from datasets import load_dataset
import json
split = "high"
ds = load_dataset("ehovy/race", split)
ds_train_json = ds['train']
# convert to json
os.makedirs("RACE_JSON", exist_ok=True)
with open(f"RACE_JSON/{split}.jsonl", "w") as f:
    for example in ds_train_json:
        f.write(json.dumps(example) + "\n")