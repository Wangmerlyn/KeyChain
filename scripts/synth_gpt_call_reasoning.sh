#!/bin/bash

# Define the dataset list
# DATASETS=("hotpotqa" "musique" "2wikimqa")
DATASETS=("hotpotqa" "2wikimqa")

# Define common parameters
PROMPT_TEMPLATE_TYPE="cot-cite"
NUM_SEQUENCES=1
TEMPERATURE=0.1

# Loop through each dataset
for DATASET_CHOICE in "${DATASETS[@]}"; do
  # Specify the input and output file paths
  INPUT_FILE="${DATASET_CHOICE}/relevant-llama-3.1-8B-instruct-10000.jsonl"
  OUTPUT_FILE="${DATASET_CHOICE}/relevant-llama-3.1-8B-instruct-10000_answer.jsonl"
  
  echo "Processing dataset: ${DATASET_CHOICE}"
  
  # Execute the Python script with the specified parameters
  python gpt_call.py \
    --input_file="${INPUT_FILE}" \
    --output_file="${OUTPUT_FILE}" \
    --prompt_template_type="${PROMPT_TEMPLATE_TYPE}" \
    --num_sequences="${NUM_SEQUENCES}" \
    --temperature="${TEMPERATURE}"
done
