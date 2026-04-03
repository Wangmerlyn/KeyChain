import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generate_trajectories import (
    normalize_answer,
    extract_answer,
    compute_rewards,
    score_trajectory,
)


# ── normalize_answer ──────────────────────────────────────────────────────────

def test_normalize_strips_articles():
    assert normalize_answer("The answer") == "answer"
    assert normalize_answer("a cat and an owl") == "cat and owl"

def test_normalize_lowercases_and_strips_punct():
    assert normalize_answer("Arthur's Magazine") == "arthurs magazine"

def test_normalize_collapses_whitespace():
    assert normalize_answer("  foo   bar  ") == "foo bar"


# ── extract_answer ────────────────────────────────────────────────────────────

def test_extract_answer_basic():
    text = "<think>some reasoning</think>\\boxed{Arthur's Magazine}"
    assert extract_answer(text) == "Arthur's Magazine"

def test_extract_answer_no_think_tag():
    assert extract_answer("\\boxed{foo}") is None

def test_extract_answer_no_boxed():
    assert extract_answer("<think>hmm</think>The answer is foo") is None

def test_extract_answer_takes_last_think():
    # multiple </think> → use the last one
    text = "<think>a</think><think>b</think>\\boxed{final}"
    assert extract_answer(text) == "final"

def test_extract_answer_strips_asterisks():
    text = "<think>r</think>**\\boxed{yes}**"
    assert extract_answer(text) == "yes"


# ── compute_rewards ───────────────────────────────────────────────────────────

def test_compute_rewards_exact():
    r = compute_rewards("Arthur's Magazine", ["Arthur's Magazine"])
    assert r["sub_em"] == 1
    assert r["em"] == 1
    assert r["f1"] == 1.0

def test_compute_rewards_sub_em_only():
    # prediction contains gold
    r = compute_rewards("Arthur's Magazine was the first", ["Arthur's Magazine"])
    assert r["sub_em"] == 1
    assert r["em"] == 0  # not strict-equal

def test_compute_rewards_wrong_answer():
    r = compute_rewards("First for Women", ["Arthur's Magazine"])
    assert r["sub_em"] == 0
    assert r["em"] == 0
    assert r["f1"] == 0.0

def test_compute_rewards_none_extracted():
    r = compute_rewards(None, ["Arthur's Magazine"])
    assert r == {"sub_em": 0, "em": 0, "f1": 0.0}

def test_compute_rewards_empty_gold():
    r = compute_rewards("foo", [])
    assert r == {"sub_em": 0, "em": 0, "f1": 0.0}

def test_compute_rewards_multi_gold_takes_max():
    # second gold matches
    r = compute_rewards("yes", ["no", "yes"])
    assert r["sub_em"] == 1
    assert r["em"] == 1

def test_compute_rewards_f1_partial():
    r = compute_rewards("new york city", ["new york"])
    assert 0.0 < r["f1"] < 1.0


# ── score_trajectory ──────────────────────────────────────────────────────────

def test_score_trajectory_correct():
    text = "<think>r</think>\\boxed{Arthur's Magazine}"
    result = score_trajectory(text, ["Arthur's Magazine"])
    assert result["is_correct"] == 1
    assert result["extracted_answer"] == "Arthur's Magazine"
    assert result["text"] == text

def test_score_trajectory_no_answer():
    result = score_trajectory("I don't know", ["Arthur's Magazine"])
    assert result["is_correct"] == 0
    assert result["extracted_answer"] is None
    assert result["sub_em"] == 0
    assert result["em"] == 0
    assert result["f1"] == 0.0
