# Diverse Multihop Sampling Design Spec

**Date:** 2026-03-29

## Problem

The current `synth_multihop_plain.sh` always takes `QAS[0:1000]` for every context length, so 4k/8k/16k/32k datasets all contain the **same 1000 questions**. For SFT training these should be distinct non-overlapping question sets.

## Goal

Re-synthesize plain multihop data so each (dataset × context_length) combo uses a **different, non-overlapping** 1000-question partition, while preserving traceability back to the original dataset via a `source_index` field.

---

## Changes

### 1. `qa.py` — two additions

**New CLI flag:**

```
--shuffle_qa    (store_true, default False)
                Shuffle QAS with --random_seed before sampling.
                Enables non-overlapping partitions via --pre_samples.
```

**Shuffle logic** (inserted immediately after QAS/DOCS are loaded at line ~183, **before** `generate_samples` is called — the module-level `random.seed()` on line 60 has already fired, so re-seeding here is intentional and necessary for reproducibility):

```python
if args.shuffle_qa:
    random.seed(args.random_seed)   # re-seed explicitly — intentional, not a duplicate
    random.shuffle(QAS)
```

**New `source_index` field** in each output record:

```python
formatted_output = {
    "index": index,                          # position within this output file (0-based)
    "source_index": index + args.pre_samples, # absolute position in shuffled QAS
    "input": question,
    "context": input_text,
    "answers": answer,
    "length": length,
}
```

`source_index` = `index + args.pre_samples`, which is the row index into the shuffled QAS list. Combined with dataset name + random_seed, this uniquely identifies the source question.

### 2. `scripts/synth_multihop_plain.sh` — partition by pre_samples

Add `--shuffle_qa` to all `qa.py` calls and map each length to a `pre_samples` offset:

| Context length | `--pre_samples` | QAS slice (shuffled) |
|---------------|----------------|----------------------|
| 4096          | 0              | [0, 1000)            |
| 8192          | 1000           | [1000, 2000)         |
| 16384         | 2000           | [2000, 3000)         |
| 32768         | 3000           | [3000, 4000)         |
| 65536         | 4000           | [4000, 5000)         |

All 5 lengths × 3 datasets = 15 non-overlapping partitions (4000–5000 questions per dataset; all datasets have 39k–167k QA pairs so this is well within range).

---

## Output Format

Each line in the output JSONL:

```json
{
  "index": 0,
  "source_index": 1532,
  "input": "Which magazine was started first...",
  "context": "Passage 1:\n...",
  "answers": ["Arthur's Magazine"],
  "length": 4096
}
```

- `index`: row within this file (0–999), unchanged
- `source_index`: absolute position in shuffled QAS — use this + dataset name + seed=42 to reproduce

---

## Reproducibility

The shuffle uses `random_seed=42` (default, already in the script). Given:
- dataset name
- random_seed
- source_index

the original question can always be recovered.

---

## Non-Goals

- No change to the KeyChain pipeline scripts
- No change to `generate_trajectories.py` or trajectory gen code
- No deduplication across datasets (hotpotqa/musique/2wikimqa are independent)
