#!/bin/bash
# Filter and clean all LoongRL-14b trajectory files.
# Input:  trajectories/*/*qwen14b*.jsonl   (12 files)
# Output: sft_data_cleaned/{dataset}/*-LoongRL-14b-cleaned.jsonl

set -e
cd "$(dirname "$0")/.."   # repo root

MODEL_TAG="${MODEL_TAG:-LoongRL-14b}"
OUTPUT_BASE="sft_data_cleaned"

echo "Cleaning LoongRL-14b trajectory files..."
echo ""

for input_file in trajectories/*/*qwen14b*.jsonl; do
    dataset=$(basename "$(dirname "$input_file")")
    python sft_data/clean_traces.py \
        --input="${input_file}" \
        --output_dir="${OUTPUT_BASE}/${dataset}" \
        --model_tag="${MODEL_TAG}"
    echo ""
done

echo "Done. Cleaned files:"
for f in ${OUTPUT_BASE}/*/*-cleaned.jsonl; do
    count=$(wc -l < "$f")
    echo "  ${f}  (${count} rows)"
done
