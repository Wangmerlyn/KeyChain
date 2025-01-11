#!/bin/bash

# Define the dataset choice
DATASET_CHOICE="hotpotqa"

# Specify the input and output file paths
INPUT_FILE="${DATASET_CHOICE}/relevant.jsonl"
OUTPUT_FILE="${DATASET_CHOICE}/relevant_answer.jsonl"

# Define the prompt template type
PROMPT_TEMPLATE_TYPE="basic"

# Execute the Python script with the specified parameters
python gpt_call.py \
  --input_file="${INPUT_FILE}" \
  --output_file="${OUTPUT_FILE}" \
  --prompt_template_type="${PROMPT_TEMPLATE_TYPE}"
