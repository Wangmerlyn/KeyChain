#!/bin/bash
# Filter all ms-swift JSONL files to best trajectory per query.
# Input:  sft_data/{dataset}/*-swift.jsonl   (12 files, 4000 rows each)
# Output: sft_data/{dataset}/*-swift-filtered.jsonl  (~900-1000 rows each)

set -e
cd "$(dirname "$0")/.."   # repo root

echo "Filtering LoongRL-14b swift files..."
echo ""

for input_file in sft_data/*/*-swift.jsonl; do
    python sft_data/filter_swift.py --input="${input_file}"
    echo ""
done

echo "Done. Filtered files:"
for f in sft_data/*/*-swift-filtered.jsonl; do
    count=$(wc -l < "$f")
    echo "  ${f}  (${count} rows)"
done
