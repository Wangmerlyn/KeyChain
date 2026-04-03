#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

MODEL_TAG="${MODEL_TAG:-LoongRL-14b}"
INPUT_DIR="${INPUT_DIR:-trajectories_incremental}"
OUTPUT_BASE="${OUTPUT_BASE:-sft_data_incremental}"

echo "=================================================="
echo "Convert Incremental Trajectories to Ms-Swift Format"
echo "=================================================="
echo "Model tag:    ${MODEL_TAG}"
echo "Input dir:    ${INPUT_DIR}"
echo "Output base:  ${OUTPUT_BASE}"
echo "=================================================="
echo ""

find "${INPUT_DIR}" -name "*.jsonl" -type f | while read -r input_file; do
    dataset=$(basename "$(dirname "$input_file")")
    output_dir="${OUTPUT_BASE}/${dataset}"
    
    echo "[convert] ${input_file}"
    python sft_data/convert_to_swift.py \
        --input="${input_file}" \
        --output_dir="${output_dir}" \
        --model_tag="${MODEL_TAG}"
done

echo ""
echo "=================================================="
echo "Validating all output files..."
echo "=================================================="

n_fail=0
for swift_file in ${OUTPUT_BASE}/*/*-swift.jsonl; do
    if [ -f "$swift_file" ]; then
        echo "[validate] ${swift_file}"
        if ! python sft_data/validate_swift.py "${swift_file}"; then
            n_fail=$((n_fail + 1))
        fi
        echo ""
    fi
done

if [ "$n_fail" -gt 0 ]; then
    echo "WARNING: ${n_fail} file(s) did not pass validation"
    exit 1
fi

echo "=================================================="
echo "Conversion and validation complete!"
echo "=================================================="
echo "Output: ${OUTPUT_BASE}/"
echo ""
echo "Next step: Filter the swift files"
echo "  bash incremental_data/scripts/filter_incremental.sh"
