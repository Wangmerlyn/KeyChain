#!/bin/bash
# incremental_data/scripts/gen_incremental_plain.sh
# 
# Generate ADDITIONAL 1.5k plain multihop samples per dataset per length.
# This is designed to incrementally expand existing 1k samples to 2.5k total.
#
# Usage:
#   bash incremental_data/scripts/gen_incremental_plain.sh
#
# Override parameters:
#   EXISTING_SAMPLES=1000 NEW_SAMPLES=1500 bash incremental_data/scripts/gen_incremental_plain.sh
#   TOKENIZER_PATH=/path/to/tokenizer bash incremental_data/scripts/gen_incremental_plain.sh

set -e
cd "$(dirname "$0")/../.."   # repo root

# ---------- Parameters ----------
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-8B}"
TOKENIZER_TYPE="hf"
TOKENS_TO_GENERATE=128
EXISTING_SAMPLES="${EXISTING_SAMPLES:-1000}"  # Already generated
NEW_SAMPLES="${NEW_SAMPLES:-1500}"            # Additional to generate
TOTAL_SAMPLES=$((EXISTING_SAMPLES + NEW_SAMPLES))  # 2500
SUBSET="train"
SAVE_DIR="data/plain_multihop_incremental"
RANDOM_SEED=42  # MUST match the original generation

# Only 4 lengths for the 30k dataset (not 5 like the original)
# 3 datasets × 4 lengths × 2.5k = 30k total
MAX_SEQ_LENGTHS=(4096 8192 16384 32768)

# For incremental generation, we use pre_samples=EXISTING_SAMPLES
# This ensures we generate indices [EXISTING_SAMPLES, TOTAL_SAMPLES)
# without overlapping with the original [0, EXISTING_SAMPLES)

echo "=================================================="
echo "Incremental Plain Multihop Data Generation"
echo "=================================================="
echo "Existing samples per dataset/length: ${EXISTING_SAMPLES}"
echo "New samples to generate:            ${NEW_SAMPLES}"
echo "Total target per dataset/length:    ${TOTAL_SAMPLES}"
echo "Tokenizer:                          ${TOKENIZER_PATH}"
echo "Lengths:                            ${MAX_SEQ_LENGTHS[@]}"
echo "Random seed:                        ${RANDOM_SEED}"
echo "=================================================="
echo ""

# ---------- Synthesis function ----------
run_synthesis() {
    local DATASET_CHOICE=$1
    local MAX_SEQ_LENGTH=$2

    case "$DATASET_CHOICE" in
        hotpotqa)
            DATASET="hotpot_train_v1.1.json"
            SAVE_NAME="hotpotqa"
            ;;
        musique)
            DATASET="musique/musique_full_v1.0_train.jsonl"
            SAVE_NAME="musique"
            ;;
        2wikimqa)
            DATASET="2wikimqa/train.json"
            SAVE_NAME="2wikimqa"
            ;;
        *)
            echo "ERROR: Unknown dataset: $DATASET_CHOICE"
            return 1
            ;;
    esac

    echo "[synth] $DATASET_CHOICE @ ${MAX_SEQ_LENGTH} tokens"
    echo "        Generating ${NEW_SAMPLES} new samples (pre_samples=${EXISTING_SAMPLES})"
    
    python qa.py \
        --save_dir="${SAVE_DIR}" \
        --save_name="${SAVE_NAME}" \
        --subset="${SUBSET}" \
        --tokenizer_path="${TOKENIZER_PATH}" \
        --tokenizer_type="${TOKENIZER_TYPE}" \
        --max_seq_length="${MAX_SEQ_LENGTH}" \
        --tokens_to_generate="${TOKENS_TO_GENERATE}" \
        --num_samples="${NEW_SAMPLES}" \
        --pre_samples="${EXISTING_SAMPLES}" \
        --random_seed="${RANDOM_SEED}" \
        --shuffle_qa \
        --distract_questions=-1 \
        --template="{context}" \
        --dataset="${DATASET}"
    
    echo "[done]  $DATASET_CHOICE @ ${MAX_SEQ_LENGTH}"
    echo ""
}

# ---------- Main execution ----------
echo "Starting incremental data generation..."
echo "This will generate ${NEW_SAMPLES} additional samples for each (dataset, length) combination"
echo "Total combinations: 3 datasets × ${#MAX_SEQ_LENGTHS[@]} lengths = $((3 * ${#MAX_SEQ_LENGTHS[@]}))"
echo ""

# Count total jobs
TOTAL_JOBS=0
for DATASET_CHOICE in hotpotqa musique 2wikimqa; do
    for MAX_SEQ_LENGTH in "${MAX_SEQ_LENGTHS[@]}"; do
        ((TOTAL_JOBS++))
    done
done

echo "Total jobs to run: ${TOTAL_JOBS}"
echo ""

# Track progress
JOB_NUM=0
for DATASET_CHOICE in hotpotqa musique 2wikimqa; do
    for MAX_SEQ_LENGTH in "${MAX_SEQ_LENGTHS[@]}"; do
        ((JOB_NUM++))
        echo "[${JOB_NUM}/${TOTAL_JOBS}] Processing ${DATASET_CHOICE} @ ${MAX_SEQ_LENGTH}..."
        run_synthesis "$DATASET_CHOICE" "$MAX_SEQ_LENGTH"
    done
done

echo "=================================================="
echo "Incremental data generation complete!"
echo "=================================================="
echo "Output directory: ${SAVE_DIR}/"
echo ""
echo "Generated files:"
find "${SAVE_DIR}" -name "*.jsonl" -type f | sort
echo ""
echo "Next steps:"
echo "  1. Run trajectory generation: bash incremental_data/scripts/gen_incremental_trajectories.sh"
echo "  2. Merge with existing data:  bash incremental_data/scripts/merge_data.sh"
echo "  3. Convert to swift format:   bash incremental_data/scripts/convert_incremental.sh"
