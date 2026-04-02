"""Tests for trace filtering and format cleaning."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def make_traj(sub_em, f1, text="reasoning</think>\\boxed{answer}"):
    return {"text": text, "sub_em": sub_em, "f1": f1, "em": sub_em,
            "is_correct": sub_em, "extracted_answer": "answer"}

def make_record(trajs, source_index=0):
    return {
        "index": 0, "source_index": source_index,
        "input": "Who?", "context": "Passage 1:\ntext.",
        "answers": ["answer"], "length": 4096,
        "model": "qwen14b_...step151",
        "num_correct": sum(t["sub_em"] for t in trajs),
        "pass_rate": 0.5,
        "trajectories": trajs,
    }


# ── clean_text ────────────────────────────────────────────────────────────────

def test_clean_text_adds_think_tag():
    """Adds <think> prefix to raw model output."""
    from sft_data.clean_traces import clean_text
    raw = "reasoning here</think>\\boxed{Paris}"
    result = clean_text(raw)
    assert result.startswith("<think>")
    assert result == "<think>reasoning here</think>Paris"

def test_clean_text_strips_boxed():
    """Strips \\boxed{} wrapper, leaving just content."""
    from sft_data.clean_traces import clean_text
    raw = "step by step</think>\\boxed{Arthur's Magazine}"
    result = clean_text(raw)
    assert "\\boxed" not in result
    assert result.endswith("Arthur's Magazine")

def test_clean_text_handles_nested_braces():
    """Handles \\boxed{content with {nested} braces}."""
    from sft_data.clean_traces import clean_text
    raw = "think</think>\\boxed{A {nested} answer}"
    result = clean_text(raw)
    assert result == "<think>think</think>A {nested} answer"

def test_clean_text_strips_whitespace_after_think():
    """Whitespace/newline between </think> and \\boxed is handled."""
    from sft_data.clean_traces import clean_text
    raw = "think</think>\n\\boxed{yes}"
    result = clean_text(raw)
    assert result == "<think>think</think>yes"


# ── select_best ───────────────────────────────────────────────────────────────

def test_select_best_picks_highest_f1():
    """Selects trajectory with highest f1 among sub_em==1."""
    from sft_data.clean_traces import select_best
    record = make_record([
        make_traj(sub_em=1, f1=0.7),
        make_traj(sub_em=1, f1=1.0),
        make_traj(sub_em=0, f1=0.9),
    ])
    result = select_best(record)
    assert result["f1"] == 1.0

def test_select_best_tiebreak_first():
    """First occurrence wins on f1 tie."""
    from sft_data.clean_traces import select_best
    t1 = make_traj(sub_em=1, f1=1.0)
    t2 = make_traj(sub_em=1, f1=1.0)
    t1["extracted_answer"] = "first"
    t2["extracted_answer"] = "second"
    result = select_best(make_record([t1, t2]))
    assert result["extracted_answer"] == "first"

def test_select_best_returns_none_if_all_wrong():
    """Returns None when all trajectories have sub_em==0."""
    from sft_data.clean_traces import select_best
    record = make_record([make_traj(sub_em=0, f1=0.8), make_traj(sub_em=0, f1=0.5)])
    assert select_best(record) is None


# ── process_file ──────────────────────────────────────────────────────────────

def test_process_file_output_format(tmp_path):
    """Output rows are valid ms-swift messages format with cleaned text."""
    import json
    from sft_data.clean_traces import process_file

    records = [
        make_record([make_traj(sub_em=1, f1=1.0, text="think</think>\\boxed{yes}")], source_index=0),
        make_record([make_traj(sub_em=0, f1=0.5)], source_index=1),  # dropped
    ]
    in_file = tmp_path / "hotpotqa" / "train-num_sample_1000-max_seq_4096-qwen14b_step.jsonl"
    in_file.parent.mkdir()
    in_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    out_path = process_file(str(in_file), str(tmp_path / "out"), model_tag="LoongRL-14b")
    rows = [json.loads(l) for l in out_path.read_text().splitlines()]

    assert len(rows) == 1  # record 1 dropped
    row = rows[0]
    assert "messages" in row
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][1]["role"] == "assistant"
    assert row["messages"][1]["content"] == "<think>think</think>yes"
    assert row["dataset"] == "hotpotqa"
    assert row["model"] == "LoongRL-14b"
    assert "\\boxed" not in row["messages"][1]["content"]

def test_process_file_output_name(tmp_path):
    """Output filename: {prefix}-{model_tag}-cleaned.jsonl."""
    import json
    from sft_data.clean_traces import process_file

    records = [make_record([make_traj(sub_em=1, f1=1.0, text="t</think>\\boxed{a}")])]
    ds_dir = tmp_path / "hotpotqa"
    ds_dir.mkdir()
    in_file = ds_dir / "train-num_sample_1000-max_seq_4096-qwen14b_long_checkpoint.jsonl"
    in_file.write_text(json.dumps(records[0]) + "\n")

    out_path = process_file(str(in_file), str(tmp_path / "out"), model_tag="LoongRL-14b")
    assert out_path.name == "train-num_sample_1000-max_seq_4096-LoongRL-14b-cleaned.jsonl"
