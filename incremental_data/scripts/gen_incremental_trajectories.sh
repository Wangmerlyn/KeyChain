#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

MODEL="${MODEL:-/home/wsy0227/qwen14b_2e_1node_16k_2k_FILTERedAGAIN_dis_256bsz_grpo_SUBEM_end-step151}"
BACKEND="${BACKEND:-vllm}"
TP_SIZE="${TP_SIZE:-4}"
N="${N:-4}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
INPUT_DIR="${INPUT_DIR:-data/plain_multihop_incremental}"
OUTPUT_DIR="${OUTPUT_DIR:-trajectories_incremental}"

DATASETS=(hotpotqa musique 2wikimqa)
MAX_SEQ_LENGTHS=(4096 8192 16384 32768)

echo "=================================================="
echo "Incremental Trajectory Generation"
echo "=================================================="
echo "Model:        ${MODEL}"
echo "Backend:      ${BACKEND}"
echo "TP Size:      ${TP_SIZE}"
echo "N (rollouts): ${N}"
echo "Temperature:  ${TEMPERATURE}"
echo "Input dir:    ${INPUT_DIR}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "=================================================="
echo ""

for DATASET in "${DATASETS[@]}"; do
    for MAX_SEQ in "${MAX_SEQ_LENGTHS[@]}"; do
        INPUT_FILE="${INPUT_DIR}/${DATASET}/train-num_sample_1500-max_seq_${MAX_SEQ}.jsonl"
        
        if [ ! -f "$INPUT_FILE" ]; then
            echo "[skip] $INPUT_FILE not found"
            continue
        fi
        
        echo "[gen] ${DATASET} @ ${MAX_SEQ} tokens"
        python trajectory_gen/generate_trajectories.py \
            --input_file="$INPUT_FILE" \
            --output_dir="$OUTPUT_DIR" \
            --model="$MODEL" \
            --backend="$BACKEND" \
            --tp_size="$TP_SIZE" \
            --n="$N" \
            --temperature="$TEMPERATURE" \
            --max_tokens="$MAX_TOKENS"
        echo "[done] ${DATASET} @ ${MAX_SEQ}"
        echo ""
    done
done

echo "=================================================="
echo "Incremental trajectory generation complete!"
echo "=================================================="
echo "Output directory: ${OUTPUT_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Convert to swift format: bash incremental_data/scripts/convert_incremental.sh"
echo "  2. Merge with existing data: bash incremental_data/scripts/merge_data.sh"
