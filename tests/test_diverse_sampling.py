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
