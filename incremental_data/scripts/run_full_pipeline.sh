#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

echo "=================================================="
echo "Incremental Data Generation Pipeline (30k Total)"
echo "=================================================="
echo "This pipeline will:"
echo "  1. Generate additional 1.5k plain samples per dataset/length"
echo "  2. Generate trajectories using vLLM"
echo "  3. Convert to ms-swift format"
echo "  4. Filter to best trajectories"
echo "  5. Merge with existing 1k samples"
echo ""
echo "Target: 2.5k samples per dataset/length (30k total)"
echo "=================================================="
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

python incremental_data/utils/manifest.py --create 2>/dev/null || true

echo ""
echo "Step 1/5: Generate incremental plain samples (1.5k each)"
echo "=================================================="
bash incremental_data/scripts/gen_incremental_plain.sh

echo ""
echo "Step 2/5: Generate trajectories using vLLM"
echo "=================================================="
bash incremental_data/scripts/gen_incremental_trajectories.sh

echo ""
echo "Step 3/5: Convert to ms-swift format"
echo "=================================================="
bash incremental_data/scripts/convert_incremental.sh

echo ""
echo "Step 4/5: Filter to best trajectories"
echo "=================================================="
bash incremental_data/scripts/filter_incremental.sh

echo ""
echo "Step 5/5: Merge with existing data"
echo "=================================================="
python incremental_data/utils/merge_data.py \
    --existing_dir sft_data \
    --incremental_dir sft_data_incremental \
    --output_dir sft_data_merged

echo ""
echo "=================================================="
echo "Pipeline Complete!"
echo "=================================================="
echo ""
python incremental_data/utils/manifest.py --status
echo ""
echo "Final merged data location: sft_data_merged/"
echo "Each file contains 2.5k samples (1k existing + 1.5k new)"
