"""Tests for SFT trajectory filtering logic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def make_row(source_index, sub_em, f1, em=None):
    """Minimal swift row for testing."""
    return {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "source_index": source_index,
        "sub_em": sub_em,
        "em": em if em is not None else sub_em,
        "f1": f1,
        "is_correct": sub_em,
        "extracted_answer": "ans",
        "answers": ["ans"],
        "dataset": "hotpotqa",
        "length": 4096,
        "model": "LoongRL-14b",
    }


# ── select_best_trajectory ────────────────────────────────────────────────────

def test_selects_correct_over_wrong():
    """When some trajectories are correct, only correct ones are candidates."""
    from sft_data.filter_swift import select_best_trajectory
    rows = [make_row(0, sub_em=0, f1=0.9), make_row(0, sub_em=1, f1=0.5)]
    result = select_best_trajectory(rows)
    assert result is not None
    assert result["sub_em"] == 1


def test_selects_highest_f1_among_correct():
    """Among correct trajectories, the one with highest f1 is selected."""
    from sft_data.filter_swift import select_best_trajectory
    rows = [
        make_row(0, sub_em=1, f1=0.7),
        make_row(0, sub_em=1, f1=1.0),
        make_row(0, sub_em=1, f1=0.8),
    ]
    result = select_best_trajectory(rows)
    assert result["f1"] == 1.0


def test_tiebreak_returns_first():
    """On f1 tie, the first occurrence is selected."""
    from sft_data.filter_swift import select_best_trajectory
    rows = [
        make_row(0, sub_em=1, f1=1.0),   # first
        make_row(0, sub_em=1, f1=1.0),   # second (same f1)
    ]
    rows[0]["extracted_answer"] = "first"
    rows[1]["extracted_answer"] = "second"
    result = select_best_trajectory(rows)
    assert result["extracted_answer"] == "first"


def test_returns_none_when_all_wrong():
    """When all trajectories have sub_em=0, returns None (query dropped)."""
    from sft_data.filter_swift import select_best_trajectory
    rows = [make_row(0, sub_em=0, f1=0.8), make_row(0, sub_em=0, f1=0.6)]
    assert select_best_trajectory(rows) is None


def test_returns_none_for_empty_list():
    """Empty input returns None."""
    from sft_data.filter_swift import select_best_trajectory
    assert select_best_trajectory([]) is None


# ── filter_file ───────────────────────────────────────────────────────────────

def test_filter_file_output_count(tmp_path):
    """filter_file produces one row per kept query."""
    import json
    from sft_data.filter_swift import filter_file

    # 3 queries: q0 all-wrong (drop), q1 one correct, q2 two correct
    rows = [
        make_row(0, sub_em=0, f1=0.5),
        make_row(0, sub_em=0, f1=0.3),
        make_row(1, sub_em=1, f1=0.9),
        make_row(1, sub_em=0, f1=0.5),
        make_row(2, sub_em=1, f1=0.7),
        make_row(2, sub_em=1, f1=1.0),
    ]
    in_file = tmp_path / "test-swift.jsonl"
    in_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    out_path = filter_file(str(in_file), str(tmp_path))
    out_rows = [json.loads(l) for l in out_path.read_text().splitlines()]

    assert len(out_rows) == 2           # q0 dropped, q1 and q2 kept
    assert out_rows[0]["source_index"] == 1
    assert out_rows[1]["source_index"] == 2
    assert out_rows[1]["f1"] == 1.0     # best of q2


def test_filter_file_output_name(tmp_path):
    """Output filename ends in -filtered.jsonl."""
    import json
    from sft_data.filter_swift import filter_file

    rows = [make_row(0, sub_em=1, f1=1.0)]
    in_file = tmp_path / "train-num_sample_1000-max_seq_4096-LoongRL-14b-swift.jsonl"
    in_file.write_text(json.dumps(rows[0]) + "\n")

    out_path = filter_file(str(in_file), str(tmp_path))
    assert out_path.name == "train-num_sample_1000-max_seq_4096-LoongRL-14b-swift-filtered.jsonl"
