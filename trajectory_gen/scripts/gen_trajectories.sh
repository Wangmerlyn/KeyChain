#!/bin/bash
# Generate rollout trajectories for all 15 (dataset × context_length) combos.
# Jobs run SEQUENTIALLY — vLLM claims all available GPUs per job.
#
# Override any parameter via environment variables, e.g.:
#   N=8 TEMPERATURE=0.7 bash scripts/gen_trajectories.sh
#
# To use the OpenAI backend:
#   BACKEND=openai MODEL=gpt-4o bash scripts/gen_trajectories.sh
#
# To process a subset of samples (e.g. for testing):
#   NUM_SAMPLES=10 bash scripts/gen_trajectories.sh  (not supported here — use --start_idx/--end_idx directly)

set -e
cd "$(dirname "$0")/../.."   # repo root

MODEL="${MODEL:-/home/wsy0227/qwen14b_2e_1node_16k_2k_FILTERedAGAIN_dis_256bsz_grpo_SUBEM_end-step151}"
BACKEND="${BACKEND:-vllm}"
TP_SIZE="${TP_SIZE:-4}"
N="${N:-4}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
INPUT_DIR="${INPUT_DIR:-output}"
OUTPUT_DIR="${OUTPUT_DIR:-trajectories}"

DATASETS=(hotpotqa musique 2wikimqa)
MAX_SEQ_LENGTHS=(4096 8192 16384 32768 65536)

for DATASET in "${DATASETS[@]}"; do
    for MAX_SEQ in "${MAX_SEQ_LENGTHS[@]}"; do
        INPUT_FILE="${INPUT_DIR}/${DATASET}/train-num_sample_1000-max_seq_${MAX_SEQ}.jsonl"
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
    done
done

echo "All trajectory generation jobs complete. Outputs in: ${OUTPUT_DIR}/"
