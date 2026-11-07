"""Tests that the data pipeline has no question-level leakage.

Run from the repo root:  pytest -v tests/test_split.py
"""

import numpy as np
import pandas as pd
import pytest

from src.data import build_splits, split_by_question, to_samples


# ---------- Unit tests on small synthetic data (fast, no download) ----------

def make_fake_questions(n_questions: int, rows_per_question: int = 1) -> pd.DataFrame:
    """Question-level DataFrame shaped like MedHallu, optionally with repeated questions."""
    rows = []
    for q in range(n_questions):
        for r in range(rows_per_question):
            rows.append({
                "qid": f"fake-{q}-{r}",
                "Question": f"Question {q}?",
                "Knowledge": np.array([f"Passage A for {q}.", f"Passage B for {q}."]),
                "Ground Truth": f"Correct answer for question {q}.",
                "Hallucinated Answer": f"Wrong answer for question {q}.",
                "Difficulty Level": "easy",
                "Category of Hallucination": "Misinterpretation",
            })
    return pd.DataFrame(rows)


def test_split_keeps_repeated_questions_together():
    # Each question appears 3 times; a plain row-level split would scatter them.
    df = make_fake_questions(n_questions=50, rows_per_question=3)
    train, val = split_by_question(df, val_size=0.2, seed=0)

    assert set(train["Question"]).isdisjoint(set(val["Question"]))
    assert len(train) + len(val) == len(df)


def test_to_samples_creates_one_positive_and_one_negative_per_question():
    df = make_fake_questions(n_questions=5)
    samples = to_samples(df)

    assert len(samples) == 2 * len(df)
    for _, group in samples.groupby("qid"):
        assert sorted(group["label"]) == [0, 1]


# ---------- Integration tests on the real dataset (downloads MedHallu) ----------

@pytest.fixture(scope="module")
def splits():
    # scope="module": build once and share across the tests below.
    return build_splits(val_size=0.15, seed=42)


@pytest.mark.parametrize("a, b", [("train", "val"), ("train", "test"), ("val", "test")])
def test_no_question_overlap(splits, a, b):
    qa = set(splits[a]["question"].str.lower().str.strip())
    qb = set(splits[b]["question"].str.lower().str.strip())
    overlap = qa & qb
    assert not overlap, f"{len(overlap)} questions shared by {a} and {b}"


def test_each_question_has_both_labels_in_same_split(splits):
    for name, df in splits.items():
        counts = df.groupby("qid")["label"].agg(["count", "sum"])
        assert (counts["count"] == 2).all(), f"{name}: a question does not have exactly 2 samples"
        assert (counts["sum"] == 1).all(), f"{name}: a question does not have one sample per label"


def test_labels_are_balanced(splits):
    for name, df in splits.items():
        assert df["label"].mean() == pytest.approx(0.5), name
