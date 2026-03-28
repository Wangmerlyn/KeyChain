# Plain Multihop Data Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synthesize plain multihop QA long-context data (no KeyChain/UUID) for HotpotQA, MuSiQue, and 2WikiMQA at 5 context lengths (4K/8K/16K/32K/64K), producing 10 samples per (dataset × length) combination = 150 samples total, for initial review.

**Architecture:** Use the existing `qa.py` script unchanged. Create a new shell script `scripts/synth_multihop_plain.sh` that downloads the three raw datasets locally and runs `qa.py` in parallel for all 15 (dataset × length) combinations. Outputs land in `output/<dataset>/`.

**Tech Stack:** Python (existing `qa.py`), bash, HuggingFace tokenizer (`Qwen/Qwen2.5-7B-Instruct`), wget/gdown for dataset download.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `scripts/synth_multihop_plain.sh` | Download datasets + run all 15 synthesis jobs |
| Read-only | `qa.py` | Plain multihop context composer (unchanged) |
| Output (generated) | `output/hotpotqa/train-num_sample_10-max_seq_*.jsonl` | 5 files |
| Output (generated) | `output/musique/train-num_sample_10-max_seq_*.jsonl` | 5 files |
| Output (generated) | `output/2wikimqa/train-num_sample_10-max_seq_*.jsonl` | 5 files |

---

### Task 1: Create new branch

**Files:**
- No file changes — git operation only

- [ ] **Step 1: Create and checkout the feature branch**

```bash
git checkout -b feature/plain-multihop-synthesis
```

Expected: `Switched to a new branch 'feature/plain-multihop-synthesis'`

---

### Task 2: Write the synthesis script

**Files:**
- Create: `scripts/synth_multihop_plain.sh`

- [ ] **Step 1: Write `scripts/synth_multihop_plain.sh`**

```bash
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
    pip install -q gdown
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
    wget -q -O 2wikimqa.zip \
        "https://www.dropbox.com/scl/fi/32t7pv1dyf3o2pp0dl25u/data_ids_april7.zip?rlkey=u868q6h0jojw4djjg7ea65j46&e=1"
    unzip -q 2wikimqa.zip -d 2wikimqa
    rm 2wikimqa.zip
fi

# ---------- Parameters ----------
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER_TYPE="hf"
TOKENS_TO_GENERATE=128
NUM_SAMPLES=10
SUBSET="train"
SAVE_DIR="output"
MAX_SEQ_LENGTHS=(4096 8192 16384 32768 65536)

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
            echo "Unknown dataset: $DATASET_CHOICE"; return 1
            ;;
    esac

    echo "[synth] $DATASET_CHOICE @ ${MAX_SEQ_LENGTH} tokens"
    python qa.py \
        --save_dir="${SAVE_DIR}" \
        --save_name="${SAVE_NAME}" \
        --subset="${SUBSET}" \
        --tokenizer_path="${TOKENIZER_PATH}" \
        --tokenizer_type="${TOKENIZER_TYPE}" \
        --max_seq_length="${MAX_SEQ_LENGTH}" \
        --tokens_to_generate="${TOKENS_TO_GENERATE}" \
        --num_samples="${NUM_SAMPLES}" \
        --template="{context}" \
        --dataset="${DATASET}"
    echo "[done]  $DATASET_CHOICE @ ${MAX_SEQ_LENGTH}"
}

# ---------- Launch all 15 jobs in parallel ----------
for DATASET_CHOICE in hotpotqa musique 2wikimqa; do
    for MAX_SEQ_LENGTH in "${MAX_SEQ_LENGTHS[@]}"; do
        run_synthesis "$DATASET_CHOICE" "$MAX_SEQ_LENGTH" &
    done
done

wait
echo "All 15 synthesis jobs complete. Outputs in: ${SAVE_DIR}/"
```

- [ ] **Step 2: Make the script executable**

```bash
chmod +x scripts/synth_multihop_plain.sh
```

- [ ] **Step 3: Commit the script**

```bash
git add scripts/synth_multihop_plain.sh
git commit -m "feat: add plain multihop synthesis script for 3 datasets × 5 lengths"
```

---

### Task 3: Install dependencies and run synthesis

**Files:**
- No new files — runtime step

- [ ] **Step 1: Install Python requirements**

```bash
pip install -r requirements.txt
```

- [ ] **Step 2: Run the synthesis script**

```bash
bash scripts/synth_multihop_plain.sh
```

Expected: 15 `[synth] ... @ ... tokens` lines appear, then 15 `[done]` lines, then "All 15 synthesis jobs complete."

Note: The HuggingFace tokenizer config for `Qwen/Qwen2.5-7B-Instruct` (~2 MB) will be auto-downloaded on first run.

---

### Task 4: Validate outputs

**Files:**
- No new files — validation step

- [ ] **Step 1: Verify all 15 output files exist**

```bash
find output -name "*.jsonl" | sort
```

Expected: 15 files — 5 per dataset under `output/hotpotqa/`, `output/musique/`, `output/2wikimqa/`.

- [ ] **Step 2: Verify each file has exactly 10 samples and correct keys**

```bash
python - <<'EOF'
import json, glob, sys
errors = []
for f in sorted(glob.glob("output/**/*.jsonl", recursive=True)):
    lines = open(f).readlines()
    if len(lines) != 10:
        errors.append(f"{f}: expected 10 lines, got {len(lines)}")
    for i, line in enumerate(lines):
        d = json.loads(line)
        for key in ("index", "input", "context", "answers", "length"):
            if key not in d:
                errors.append(f"{f} line {i}: missing key '{key}'")
if errors:
    print("ERRORS:"); [print(" -", e) for e in errors]; sys.exit(1)
else:
    print(f"OK: all {len(list(glob.glob('output/**/*.jsonl', recursive=True)))} files valid")
EOF
```

Expected: `OK: all 15 files valid`

- [ ] **Step 3: Spot-check a sample from each dataset**

```bash
python - <<'EOF'
import json, glob
for ds in ("hotpotqa", "musique", "2wikimqa"):
    f = sorted(glob.glob(f"output/{ds}/*.jsonl"))[0]
    d = json.loads(open(f).readline())
    print(f"\n=== {ds} (first file: {f}) ===")
    print(f"  question : {d['input'][:100]}")
    print(f"  answer   : {d['answers']}")
    print(f"  length   : {d['length']} tokens")
    print(f"  context  : {d['context'][:200]}...")
EOF
```

Expected: One sample printed per dataset showing a valid multihop question, an answer, token length near the target, and context starting with "Passage 1:".

- [ ] **Step 4: Commit outputs for user review**

```bash
git add output/
git commit -m "data: add 10-sample synthesis outputs for hotpotqa/musique/2wikimqa at 5 context lengths"
```

---

## Done

Hand off 15 output files to user for review. If the format looks correct, the full-scale run (e.g. 10K samples) can proceed by changing `NUM_SAMPLES` in the script.
