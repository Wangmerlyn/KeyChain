#!/bin/bash
# Plain multihop synthesis (no KeyChain) for HotpotQA, MuSiQue, 2WikiMQA
# Runs 3 datasets × 5 context lengths = 15 parallel jobs, 10 samples each.

set -e
cd "$(dirname "$0")/.."   # repo root

# ---------- Download datasets (skip if already present) ----------
if [ ! -f "hotpot_train_v1.1.json" ]; then
    echo "[download] HotpotQA training set..."
    wget -q http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
fi

if [ ! -f "musique/musique_full_v1.0_train.jsonl" ]; then
    echo "[download] MuSiQue training set..."
    pip install -q gdown --break-system-packages 2>/dev/null || true
    export PATH="$HOME/.local/bin:$PATH"
    gdown --id 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h --output musique_v1.0.zip
    unzip -q musique_v1.0.zip
    rm musique_v1.0.zip
    rm -rf __MACOSX
    mkdir -p musique
    [ -f "data/musique_full_v1.0_train.jsonl" ] && mv data/musique_full_v1.0_train.jsonl musique/
    rmdir data 2>/dev/null || true
fi

if [ ! -f "2wikimqa/train.json" ]; then
    echo "[download] 2WikiMQA training set..."
    # Note: if this Dropbox URL expires, re-fetch from the dataset's GitHub release page
    wget -q -O 2wikimqa.zip \
        "https://www.dropbox.com/scl/fi/32t7pv1dyf3o2pp0dl25u/data_ids_april7.zip?rlkey=u868q6h0jojw4djjg7ea65j46&e=1"
    unzip -q 2wikimqa.zip -d 2wikimqa
    rm 2wikimqa.zip
fi

# ---------- Parameters ----------
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-8B}"
TOKENIZER_TYPE="hf"
TOKENS_TO_GENERATE=128
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
SUBSET="train"
SAVE_DIR="data/plain_multihop"
# Each length gets a non-overlapping slice of shuffled QAS:
#   4096  → [0,      NUM_SAMPLES)
#   8192  → [1×NS,   2×NS)
#   16384 → [2×NS,   3×NS)
#   32768 → [3×NS,   4×NS)
#   65536 → [4×NS,   5×NS)
MAX_SEQ_LENGTHS=(4096 8192 16384 32768 65536)

# ---------- Synthesis function ----------
run_synthesis() {
    local DATASET_CHOICE=$1
    local MAX_SEQ_LENGTH=$2
    local PRE_SAMPLES=$3

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
            echo "Unknown dataset: $DATASET_CHOICE"; return 1
            ;;
    esac

    echo "[synth] $DATASET_CHOICE @ ${MAX_SEQ_LENGTH} tokens (pre_samples=${PRE_SAMPLES})"
    python qa.py \
        --save_dir="${SAVE_DIR}" \
        --save_name="${SAVE_NAME}" \
        --subset="${SUBSET}" \
        --tokenizer_path="${TOKENIZER_PATH}" \
        --tokenizer_type="${TOKENIZER_TYPE}" \
        --max_seq_length="${MAX_SEQ_LENGTH}" \
        --tokens_to_generate="${TOKENS_TO_GENERATE}" \
        --num_samples="${NUM_SAMPLES}" \
        --pre_samples="${PRE_SAMPLES}" \
        --shuffle_qa \
        --distract_questions=-1 \
        --template="{context}" \
        --dataset="${DATASET}"
    echo "[done]  $DATASET_CHOICE @ ${MAX_SEQ_LENGTH}"
}

# ---------- Launch all 15 jobs in parallel ----------
for DATASET_CHOICE in hotpotqa musique 2wikimqa; do
    for idx in "${!MAX_SEQ_LENGTHS[@]}"; do
        MAX_SEQ_LENGTH="${MAX_SEQ_LENGTHS[$idx]}"
        PRE_SAMPLES=$((idx * NUM_SAMPLES))
        run_synthesis "$DATASET_CHOICE" "$MAX_SEQ_LENGTH" "$PRE_SAMPLES" &
    done
done

wait
echo "All 15 synthesis jobs complete. Outputs in: ${SAVE_DIR}/"
