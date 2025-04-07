#!/bin/bash

# Define the dataset choice: options are hotpotqa, musique, or 2wikimqa
DATASET_CHOICE="hotpotqa"
# DATASET_CHOICE="musique"
# DATASET_CHOICE="2wikimqa"

# Define whether to shuffle or not (true/false)
SHUFFLE=true

# Define the common parameters
QUESTION_LISTS_PREFIX_BASE="${DATASET_CHOICE}/validation-llama-3.1-8B-instruct-num_sample_10000-max_seq"
GPT_ANSWER_LIST_PATH="${DATASET_CHOICE}/relevant-llama-3.1-8B-instruct-10000_answer.jsonl"
# LENGTH_DISTRIBUTION='{"4096": 0.1, "8192": 0.1, "16384": 0.1, "32768": 0.2, "65536": 0.25, "131072": 0.25}'
# LENGTH_DISTRIBUTION='{"4096": 0.2, "8192": 0.4, "16384": 0.4}'
LENGTH_DISTRIBUTION='{"4096": 0.1, "8192": 0.1, "16384": 0.8}'

# Build the command dynamically
CMD="python sample_question_length.py \
    --question_lists_prefix ${QUESTION_LISTS_PREFIX_BASE} \
    --gpt_answer_list_path ${GPT_ANSWER_LIST_PATH} \
    --length_distribution '${LENGTH_DISTRIBUTION}'"

# Add the --shuffle flag if SHUFFLE is true
if [ "$SHUFFLE" = true ]; then
    CMD+=" --shuffle"
fi

# Execute the command
eval $CMD
