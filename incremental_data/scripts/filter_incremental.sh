#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

INPUT_BASE="${INPUT_BASE:-sft_data_incremental}"

echo "=================================================="
echo "Filter Incremental Ms-Swift Data"
echo "=================================================="
echo "Input base: ${INPUT_BASE}"
echo "=================================================="
echo ""

for swift_file in ${INPUT_BASE}/*/*-swift.jsonl; do
    if [ -f "$swift_file" ]; then
        echo "[filter] ${swift_file}"
        python sft_data/filter_swift.py \
            --input="${swift_file}"
        echo ""
    fi
done

echo "=================================================="
echo "Filtering complete!"
echo "=================================================="
echo ""
echo "Next step: Merge with existing data"
echo "  bash incremental_data/scripts/merge_data.sh"
