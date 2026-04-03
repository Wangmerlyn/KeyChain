"""Tests for ms-swift format conversion and validation."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Conversion tests ──────────────────────────────────────────────────────────

def make_trajectory_record(n=2):
    """Minimal trajectory record matching real data structure."""
    return {
        "index": 0,
        "source_index": 42,
        "input": "Who directed Titanic?",
        "context": "Passage 1:\nJames Cameron directed Titanic.",
        "answers": ["James Cameron"],
        "length": 4096,
        "model": "qwen14b_...step151",
        "num_correct": 1,
        "pass_rate": 0.5,
        "trajectories": [
            {
                "text": "<think>Titanic was directed by Cameron.</think>\\boxed{James Cameron}",
                "extracted_answer": "James Cameron",
                "sub_em": 1, "em": 1, "f1": 1.0, "is_correct": 1,
            },
            {
                "text": "<think>I'm not sure.</think>\\boxed{unknown}",
                "extracted_answer": "unknown",
                "sub_em": 0, "em": 0, "f1": 0.0, "is_correct": 0,
            },
        ][:n],
    }


def test_convert_expands_trajectories():
    """Each record with n trajectories produces n output rows."""
    from sft_data.convert_to_swift import convert_record
    record = make_trajectory_record(n=2)
    rows = convert_record(record, dataset="hotpotqa", model_tag="LoongRL-14b")
    assert len(rows) == 2


def test_convert_messages_format():
    """Output row has correct messages structure for ms-swift."""
    from sft_data.convert_to_swift import convert_record
    record = make_trajectory_record(n=1)
    rows = convert_record(record, dataset="hotpotqa", model_tag="LoongRL-14b")
    row = rows[0]
    assert "messages" in row
    msgs = row["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert record["input"] in msgs[0]["content"]
    assert record["context"] in msgs[0]["content"]
    assert msgs[1]["content"] == record["trajectories"][0]["text"]


def test_convert_preserves_scoring_fields():
    """All scoring fields are preserved in output row."""
    from sft_data.convert_to_swift import convert_record
    record = make_trajectory_record(n=1)
    row = convert_record(record, dataset="hotpotqa", model_tag="LoongRL-14b")[0]
    for field in ("is_correct", "sub_em", "em", "f1", "extracted_answer"):
        assert field in row, f"missing field: {field}"
    assert row["is_correct"] == 1
    assert row["f1"] == 1.0
    assert row["dataset"] == "hotpotqa"
    assert row["model"] == "LoongRL-14b"
    assert row["source_index"] == 42
    assert row["answers"] == ["James Cameron"]


def test_convert_output_filename():
    """Output stem is correctly derived from input path."""
    from sft_data.convert_to_swift import derive_output_stem
    stem = "train-num_sample_1000-max_seq_4096-qwen14b_2e_1node_long_checkpoint"
    result = derive_output_stem(stem, model_tag="LoongRL-14b")
    assert result == "train-num_sample_1000-max_seq_4096-LoongRL-14b-swift"


def test_convert_skips_empty_text():
    """Trajectories with empty text are skipped when skip_empty=True."""
    from sft_data.convert_to_swift import convert_record
    record = make_trajectory_record(n=1)
    record["trajectories"][0]["text"] = ""
    rows = convert_record(record, dataset="hotpotqa", model_tag="LoongRL-14b", skip_empty=True)
    assert len(rows) == 0


# ── Validation tests ──────────────────────────────────────────────────────────

def make_swift_row(role_last="assistant", content="response text"):
    return {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": role_last, "content": content},
        ],
        "is_correct": 1,
    }


def test_validate_passes_valid_row():
    from sft_data.validate_swift import validate_row_rules
    assert validate_row_rules(make_swift_row()) == []


def test_validate_missing_messages():
    from sft_data.validate_swift import validate_row_rules
    errors = validate_row_rules({"is_correct": 1})
    assert any("messages" in e for e in errors)


def test_validate_last_role_not_assistant():
    from sft_data.validate_swift import validate_row_rules
    errors = validate_row_rules(make_swift_row(role_last="user"))
    assert any("assistant" in e for e in errors)


def test_validate_empty_content():
    from sft_data.validate_swift import validate_row_rules
    errors = validate_row_rules(make_swift_row(content=""))
    assert any("empty" in e.lower() for e in errors)
