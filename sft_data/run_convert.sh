#!/bin/bash
# Convert and validate all LoongRL-14b trajectory files to ms-swift format.
# Input:  trajectories/{dataset}/*qwen14b*.jsonl  (12 files)
# Output: sft_data/{dataset}/*LoongRL-14b-swift.jsonl

set -e
cd "$(dirname "$0")/.."   # repo root

MODEL_TAG="${MODEL_TAG:-LoongRL-14b}"
INPUT_GLOB="trajectories/*/*qwen14b*.jsonl"
OUTPUT_BASE="sft_data"

echo "Converting LoongRL-14b trajectories to ms-swift format..."
echo "Model tag: ${MODEL_TAG}"
echo ""

# Convert each file
for input_file in ${INPUT_GLOB}; do
    dataset=$(basename "$(dirname "$input_file")")
    output_dir="${OUTPUT_BASE}/${dataset}"
    echo "[convert] ${input_file}"
    python sft_data/convert_to_swift.py \
        --input="${input_file}" \
        --output_dir="${output_dir}" \
        --model_tag="${MODEL_TAG}"
done

echo ""
echo "Validating all output files..."

# Validate each output file
n_fail=0
for swift_file in ${OUTPUT_BASE}/*/*LoongRL-14b-swift.jsonl; do
    echo "[validate] ${swift_file}"
    if ! python sft_data/validate_swift.py "${swift_file}"; then
        n_fail=$((n_fail + 1))
    fi
    echo ""
done

if [ "$n_fail" -gt 0 ]; then
    echo "FAILED: ${n_fail} file(s) did not pass validation"
    exit 1
fi

echo "All files converted and validated successfully."
echo "Output: ${OUTPUT_BASE}/"
