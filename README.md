# LongContextDataSynth

## Usage
```bash
bash scripts/synth.sh
```

To generate a dataset with only the relevant docs to answer the question, use:
```bash
bash scripts/synth_relevant_only.sh
```

To get gpt-4o reasoning steps from training sets:
```bash
bash scripts/synth_gpt_call_reasoning.sh
```

## Dataset Statistics
| Dataset   | #training qa pairs | #Total unique docs |
| --------- | ------------------ | ------------------ |
| Hotpot QA | 90447              | 483696             |
| Musique   | 19938              | 797413             |
| 2Wikimqa  | 167454             | 369378             |
