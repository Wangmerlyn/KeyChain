# LongContextDataSynth

## Usage
```bash
bash scripts/synth.sh
```

To generate a dataset with only the relevant docs to answer the question, use:
```bash
bash scripts/synth_relevant_only.sh
```

To filter datasets for better quality, use:
```bash
bash filter_question/remote_distributed_filter.sh
```
> use this on a distributed cluster for efficiency.

To generate a dataset with distractors on the filtered dataset:
```bash
bash scripts/synth_qwen_filter_add_distractor.sh
# for debugging purposes, use synth_qwen_filter_add_distractor_debug.sh 
```

## Dataset Statistics
| Dataset   | #training qa pairs | #Total unique docs |
| --------- | ------------------ | ------------------ |
| Hotpot QA | 90447              | 483696             |
| Musique   | 19938              | 797413             |
| 2Wikimqa  | 167454             | 369378             |
