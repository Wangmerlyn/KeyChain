#!/bin/bash

# Define the datasets and prompt templates
DATASETS=("hotpotqa" "musique" "2wikimqa")
PROMPTS=("normal" "cot" "cot-cite" "mcts")

# Loop through each dataset
for DATASET_CHOICE in "${DATASETS[@]}"; do
  
  # The input file is always named "relevant.jsonl" for each dataset folder
  INPUT_FILE="${DATASET_CHOICE}/relevant.jsonl"
  
  # Loop through each prompt (excluding "basic")
  for PROMPT_TEMPLATE_TYPE in "${PROMPTS[@]}"; do
    
    # Define the output file for each dataset+prompt
    OUTPUT_FILE="${DATASET_CHOICE}/relevant_answer_${PROMPT_TEMPLATE_TYPE}.jsonl"
    
    echo "Running dataset: ${DATASET_CHOICE} with prompt: ${PROMPT_TEMPLATE_TYPE}"
    
    # Execute the Python script
    python gpt_call.py \
      --input_file="${INPUT_FILE}" \
      --output_file="${OUTPUT_FILE}" \
      --prompt_template_type="${PROMPT_TEMPLATE_TYPE}"
    
    echo "Finished dataset: ${DATASET_CHOICE} with prompt: ${PROMPT_TEMPLATE_TYPE}"
    echo "--------------------------------------------------"
    
  done

done
