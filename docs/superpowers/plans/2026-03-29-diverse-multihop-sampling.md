# Diverse Multihop Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each (dataset × context_length) combo use a distinct non-overlapping 1000-question partition, with `source_index` recorded in every output sample for traceability.

**Architecture:** Add `--shuffle_qa` flag to `qa.py` that shuffles QAS with the existing random seed before sampling, then update `synth_multihop_plain.sh` to pass per-length `--pre_samples` offsets (0, 1000, 2000, 3000, 4000) so each length draws from a different slice of the shuffled QAS.

**Tech Stack:** Python (qa.py), bash (synth_multihop_plain.sh), pytest.

---

## File Map

| Action | Path | Change |
|--------|------|--------|
| Modify | `qa.py:54-61` | Add `--shuffle_qa` arg |
| Modify | `qa.py:183-184` | Insert shuffle block after dataset load |
| Modify | `qa.py:273-279` | Add `source_index` to `formatted_output` |
| Modify | `scripts/synth_multihop_plain.sh:43-90` | Per-length `--pre_samples`, add `--shuffle_qa` |
| Create | `tests/test_diverse_sampling.py` | Unit tests for shuffle + source_index |

---

### Task 1: Write failing tests

**Files:**
- Create: `tests/test_diverse_sampling.py`

- [ ] **Step 1: Write `tests/test_diverse_sampling.py`**

```python
"""Tests for diverse sampling: --shuffle_qa and source_index."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def run_qa(dataset_path, max_seq_length, num_samples, pre_samples=0,
           shuffle_qa=False, save_dir=None, save_name="test_out"):
    """Run qa.py with given args, return list of output records."""
    cmd = [
        sys.executable, str(REPO_ROOT / "qa.py"),
        "--save_dir", str(save_dir),
        "--save_name", save_name,
        "--subset", "test",
        "--tokenizer_path", "Qwen/Qwen3-8B",
        "--tokenizer_type", "hf",
        "--max_seq_length", str(max_seq_length),
        "--tokens_to_generate", "128",
        "--num_samples", str(num_samples),
        "--distract_questions", "-1",
        "--template", "{context}",
        "--pre_samples", str(pre_samples),
        "--dataset", str(dataset_path),
    ]
    if shuffle_qa:
        cmd.append("--shuffle_qa")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, f"qa.py failed:\n{result.stderr}"
    out_file = Path(save_dir) / save_name / f"test-num_sample_{num_samples}-max_seq_{max_seq_length}.jsonl"
    return [json.loads(l) for l in out_file.read_text().splitlines()]


def test_source_index_present():
    """Every output record must have a source_index field."""
    dataset = REPO_ROOT / "hotpot_train_v1.1.json"
    if not dataset.exists():
        import pytest; pytest.skip("hotpotqa dataset not downloaded")
    with tempfile.TemporaryDirectory() as d:
        records = run_qa(dataset, 4096, 5, save_dir=d)
    assert all("source_index" in r for r in records), "source_index missing"


def test_source_index_without_shuffle_equals_index():
    """Without --shuffle_qa, source_index should equal index + pre_samples."""
    dataset = REPO_ROOT / "hotpot_train_v1.1.json"
    if not dataset.exists():
        import pytest; pytest.skip("hotpotqa dataset not downloaded")
    with tempfile.TemporaryDirectory() as d:
        records = run_qa(dataset, 4096, 5, pre_samples=10, save_dir=d)
    for r in records:
        assert r["source_index"] == r["index"] + 10, \
            f"expected source_index={r['index']+10}, got {r['source_index']}"


def test_shuffle_produces_non_overlapping_partitions():
    """With --shuffle_qa, two runs with different pre_samples must have disjoint source_indexes."""
    dataset = REPO_ROOT / "hotpot_train_v1.1.json"
    if not dataset.exists():
        import pytest; pytest.skip("hotpotqa dataset not downloaded")
    with tempfile.TemporaryDirectory() as d:
        r1 = run_qa(dataset, 4096, 10, pre_samples=0,  shuffle_qa=True, save_dir=d, save_name="out1")
        r2 = run_qa(dataset, 4096, 10, pre_samples=10, shuffle_qa=True, save_dir=d, save_name="out2")
    idx1 = {r["source_index"] for r in r1}
    idx2 = {r["source_index"] for r in r2}
    assert idx1.isdisjoint(idx2), f"Overlapping source_indexes: {idx1 & idx2}"


def test_shuffle_changes_question_order():
    """--shuffle_qa should produce different question order than no shuffle (same pre_samples)."""
    dataset = REPO_ROOT / "hotpot_train_v1.1.json"
    if not dataset.exists():
        import pytest; pytest.skip("hotpotqa dataset not downloaded")
    with tempfile.TemporaryDirectory() as d:
        unshuffled = run_qa(dataset, 4096, 5, pre_samples=0, shuffle_qa=False, save_dir=d, save_name="uns")
        shuffled   = run_qa(dataset, 4096, 5, pre_samples=0, shuffle_qa=True,  save_dir=d, save_name="shf")
    q_uns = [r["input"] for r in unshuffled]
    q_shf = [r["input"] for r in shuffled]
    assert q_uns != q_shf, "Shuffled and unshuffled produced identical question order"
```

- [ ] **Step 2: Run tests — expect failures (qa.py missing --shuffle_qa and source_index)**

```bash
python -m pytest tests/test_diverse_sampling.py -v -k "not hotpotqa" 2>&1 | head -20
```

Expected: tests skip (dataset not downloaded) or fail with `unrecognized arguments: --shuffle_qa`.

---

### Task 2: Add `--shuffle_qa` to `qa.py` and `source_index` to output

**Files:**
- Modify: `qa.py:54-61` (add arg)
- Modify: `qa.py:183-184` (add shuffle block)
- Modify: `qa.py:273-279` (add source_index field)

- [ ] **Step 1: Add `--shuffle_qa` argument** after line 57 (`--distract_questions` arg):

```python
parser.add_argument("--shuffle_qa", action="store_true",
                    help="Shuffle QAS with random_seed before sampling, enabling "
                         "non-overlapping partitions via --pre_samples.")
```

- [ ] **Step 2: Insert shuffle block** after line 183 (after the `raise NotImplementedError` line, i.e. right after the if/elif/else block that sets `QAS, DOCS`):

```python
# Shuffle QAS for diverse sampling across context lengths.
# Re-seed explicitly here — the module-level random.seed() on line 60 has
# already fired; this second call is intentional for reproducibility.
if args.shuffle_qa:
    random.seed(args.random_seed)
    random.shuffle(QAS)
```

- [ ] **Step 3: Add `source_index` to `formatted_output`** (around line 273):

```python
        formatted_output = {
            "index": index,
            "source_index": index + args.pre_samples,  # absolute position in (shuffled) QAS
            "input": question,
            "context": input_text,
            "answers": answer,
            "length": length,
        }
```

- [ ] **Step 4: Run tests** (skip tests need dataset, but verify no import errors):

```bash
python -m pytest tests/test_diverse_sampling.py -v 2>&1 | tail -15
```

Expected: tests skip with `pytest.skip("hotpotqa dataset not downloaded")` — no errors.

- [ ] **Step 5: Smoke-check source_index on existing data** (no re-synthesis needed):

```bash
python3 -c "
import subprocess, json, tempfile, sys
from pathlib import Path
# Quick sanity: run qa.py with 3 samples, no shuffle, check source_index
with tempfile.TemporaryDirectory() as d:
    r = subprocess.run([
        sys.executable, 'qa.py',
        '--save_dir', d, '--save_name', 'chk',
        '--subset', 'test',
        '--tokenizer_path', 'Qwen/Qwen3-8B', '--tokenizer_type', 'hf',
        '--max_seq_length', '4096', '--tokens_to_generate', '128',
        '--num_samples', '3', '--distract_questions', '-1',
        '--template', '{context}', '--pre_samples', '5',
        '--dataset', 'hotpot_train_v1.1.json',
    ], capture_output=True, text=True)
    print(r.stderr[-300:] if r.returncode != 0 else 'OK')
    if r.returncode == 0:
        recs = [json.loads(l) for l in (Path(d)/'chk'/'test-num_sample_3-max_seq_4096.jsonl').read_text().splitlines()]
        for rec in recs:
            assert rec['source_index'] == rec['index'] + 5, rec
        print('source_index correct:', [r['source_index'] for r in recs])
"
```

Expected: `source_index correct: [5, 6, 7]`

- [ ] **Step 6: Commit**

```bash
git add qa.py tests/test_diverse_sampling.py
git commit -m "feat: add --shuffle_qa flag and source_index field to qa.py"
```

---

### Task 3: Update `scripts/synth_multihop_plain.sh`

**Files:**
- Modify: `scripts/synth_multihop_plain.sh:43-90`

- [ ] **Step 1: Replace the parameters + loop section** with per-length `--pre_samples` offsets:

Replace from `# ---------- Parameters ----------` through `wait` with:

```bash
# ---------- Parameters ----------
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-8B}"
TOKENIZER_TYPE="hf"
TOKENS_TO_GENERATE=128
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
SUBSET="train"
SAVE_DIR="output"
# Each length gets a non-overlapping slice of shuffled QAS:
#   4096 → [0,      NUM_SAMPLES)
#   8192 → [1×NS,   2×NS)
#  16384 → [2×NS,   3×NS)
#  32768 → [3×NS,   4×NS)
#  65536 → [4×NS,   5×NS)
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
```

- [ ] **Step 2: Commit**

```bash
git add scripts/synth_multihop_plain.sh
git commit -m "feat: partition questions across context lengths via shuffle + pre_samples"
```

---

### Task 4: Re-run synthesis and validate

**Files:** No new files — runtime step.

- [ ] **Step 1: Run synthesis**

```bash
bash scripts/synth_multihop_plain.sh
```

Expected: 15 jobs complete, each logging `pre_samples=N`.

- [ ] **Step 2: Validate non-overlapping source_indexes**

```bash
python3 - <<'EOF'
import json, glob, sys
from pathlib import Path

errors = []
for dataset in ("hotpotqa", "musique", "2wikimqa"):
    files = sorted(glob.glob(f"output/{dataset}/train-num_sample_1000-max_seq_*.jsonl"))
    all_source_indexes = {}
    for f in files:
        records = [json.loads(l) for l in open(f)]
        # Check source_index present
        missing = [i for i, r in enumerate(records) if "source_index" not in r]
        if missing:
            errors.append(f"{f}: missing source_index on lines {missing[:5]}")
            continue
        idxs = {r["source_index"] for r in records}
        # Check no overlap with previously seen files
        for other_f, other_idxs in all_source_indexes.items():
            overlap = idxs & other_idxs
            if overlap:
                errors.append(f"{f} overlaps with {other_f}: {len(overlap)} shared source_indexes")
        all_source_indexes[f] = idxs
    if not errors:
        print(f"✓ {dataset}: {len(files)} files, all source_indexes non-overlapping")

if errors:
    print("ERRORS:"); [print(" -", e) for e in errors]; sys.exit(1)
EOF
```

Expected:
```
✓ hotpotqa: 5 files, all source_indexes non-overlapping
✓ musique: 5 files, all source_indexes non-overlapping
✓ 2wikimqa: 5 files, all source_indexes non-overlapping
```

- [ ] **Step 3: Spot-check source_index values match expected offsets**

```bash
python3 - <<'EOF'
import json, glob
for dataset in ("hotpotqa", "musique", "2wikimqa"):
    print(f"\n{dataset}:")
    for f in sorted(glob.glob(f"output/{dataset}/train-num_sample_1000-max_seq_*.jsonl")):
        records = [json.loads(l) for l in open(f)]
        min_si = min(r["source_index"] for r in records)
        max_si = max(r["source_index"] for r in records)
        length = f.split("max_seq_")[1].replace(".jsonl","")
        print(f"  {length:6s} → source_index range [{min_si:4d}, {max_si:4d}]")
EOF
```

Expected: ranges are `[0,999]`, `[1000,1999]`, `[2000,2999]`, `[3000,3999]`, `[4000,4999]` for each dataset.

- [ ] **Step 4: Commit outputs and push**

```bash
git add scripts/synth_multihop_plain.sh  # if not already committed
git push https://github.com/Wangmerlyn/KeyChain-private.git feature/trajectory-generation
```

---

## Done

15 output files regenerated with non-overlapping question partitions. Each record now has `source_index` for traceability back to the shuffled QAS list (seed=42).
