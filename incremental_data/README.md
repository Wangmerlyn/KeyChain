# Incremental Data Generation System

This directory contains scripts and utilities for incrementally expanding the multihop dataset from 12k (1k per dataset/length) to 30k (2.5k per dataset/length).

## Overview

The existing dataset has:
- 3 datasets: hotpotqa, musique, 2wikimqa
- 4 lengths: 4096, 8192, 16384, 32768
- 1k samples per dataset/length = 12k total

The target is:
- Same 3 datasets × 4 lengths
- 2.5k samples per dataset/length = 30k total
- Additional 1.5k samples per dataset/length needed

## Key Design Principles

1. **Non-overlapping sampling**: Uses `pre_samples` parameter with consistent random seed to ensure new samples don't overlap with existing ones
2. **Cross-length partitioning**: Different lengths use different index ranges to avoid query overlap:
   - 4k: indices 0-999 (existing), 4000-5499 (new)
   - 8k: indices 1000-1999 (existing), 5500-6999 (new)
   - 16k: indices 2000-2999 (existing), 7000-8499 (new)
   - 32k: indices 3000-3999 (existing), 8500-9999 (new)

## Directory Structure

```
incremental_data/
├── scripts/
│   ├── gen_incremental_plain.sh         # Generate 1.5k additional plain samples
│   ├── gen_incremental_trajectories.sh  # Run vLLM on new samples
│   ├── convert_incremental.sh           # Convert to ms-swift format
│   ├── filter_incremental.sh            # Filter to best trajectories
│   ├── merge_data.sh                    # Merge 1k + 1.5k → 2.5k
│   └── run_full_pipeline.sh             # Run all steps end-to-end
├── utils/
│   ├── merge_data.py                    # Python merge utility
│   └── manifest.py                      # Track generation progress
└── manifests/                           # Generation tracking files
```

## Usage

### Option 1: Run Full Pipeline

```bash
bash incremental_data/scripts/run_full_pipeline.sh
```

This will:
1. Generate 1.5k additional plain samples per dataset/length
2. Run vLLM trajectory generation (4 rollouts per query)
3. Convert to ms-swift format
4. Filter to best trajectory per query
5. Merge with existing 1k samples

### Option 2: Step-by-Step

Generate additional plain samples:
```bash
bash incremental_data/scripts/gen_incremental_plain.sh
```

Generate trajectories:
```bash
bash incremental_data/scripts/gen_incremental_trajectories.sh
```

Convert to ms-swift:
```bash
bash incremental_data/scripts/convert_incremental.sh
```

Filter best trajectories:
```bash
bash incremental_data/scripts/filter_incremental.sh
```

Merge with existing:
```bash
python incremental_data/utils/merge_data.py \
    --existing_dir sft_data \
    --incremental_dir sft_data_incremental \
    --output_dir sft_data_merged
```

### Track Progress

Create/update manifest:
```bash
python incremental_data/utils/manifest.py --create
```

Check status:
```bash
python incremental_data/utils/manifest.py --status
```

Update specific entry:
```bash
python incremental_data/utils/manifest.py --update hotpotqa 4096 merged --samples 2500
```

## Output Structure

After running the pipeline:

```
data/plain_multihop_incremental/     # New 1.5k plain samples
trajectories_incremental/            # vLLM rollouts on new data
sft_data_incremental/                # Ms-swift formatted new data
sft_data_merged/                     # Final 2.5k combined data
```

## Environment Variables

Override defaults by setting these before running scripts:

```bash
# Data generation
export EXISTING_SAMPLES=1000
export NEW_SAMPLES=1500
export RANDOM_SEED=42
export TOKENIZER_PATH=Qwen/Qwen3-8B

# Trajectory generation
export MODEL=/path/to/your/model
export TP_SIZE=4
export N=4                    # Rollouts per query
export TEMPERATURE=0.6

# Paths
export SAVE_DIR=data/plain_multihop_incremental
export OUTPUT_DIR=trajectories_incremental
```

## Notes

- The scripts use `pre_samples=4000+` to ensure no overlap with existing data (0-3999 range)
- Random seed 42 is used consistently to maintain reproducibility
- Each query generates 4 responses via vLLM, then filtered to keep the best one (sub_em=1, highest f1)
- The original 1k data remains untouched; new data is generated separately and merged at the end
