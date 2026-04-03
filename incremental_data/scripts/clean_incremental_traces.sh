#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

MODEL_TAG="${MODEL_TAG:-LoongRL-14b}"
INPUT_DIR="${INPUT_DIR:-trajectories_incremental}"
OUTPUT_BASE="${OUTPUT_BASE:-sft_data_cleaned_incremental}"

echo "=================================================="
echo "Clean Incremental Trajectories"
echo "=================================================="
echo "Model tag:    ${MODEL_TAG}"
echo "Input dir:    ${INPUT_DIR}"
echo "Output base:  ${OUTPUT_BASE}"
echo "=================================================="
echo ""

for input_file in ${INPUT_DIR}/*/*qwen14b*.jsonl; do
    if [ ! -f "$input_file" ]; then
        echo "No files found in ${INPUT_DIR}"
        exit 1
    fi
    
    dataset=$(basename "$(dirname "$input_file")")
    echo "[clean] ${input_file}"
    python sft_data/clean_traces.py \
        --input="${input_file}" \
        --output_dir="${OUTPUT_BASE}/${dataset}" \
        --model_tag="${MODEL_TAG}"
    echo ""
done

echo "=================================================="
echo "Cleaning complete!"
echo "=================================================="
echo "Output directory: ${OUTPUT_BASE}/"
echo ""
echo "Cleaned files:"
for f in ${OUTPUT_BASE}/*/*-cleaned.jsonl; do
    if [ -f "$f" ]; then
        count=$(wc -l < "$f")
        echo "  ${f}  (${count} rows)"
    fi
done
echo ""
echo "Next step: Merge with existing cleaned data"
echo "  python incremental_data/utils/merge_cleaned_data.py \\"
echo "    --existing_dir sft_data_cleaned \\"
echo "    --incremental_dir sft_data_cleaned_incremental \\"
echo "    --output_dir sft_data_cleaned_merged"
