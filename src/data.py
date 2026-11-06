"""Load, clean, and split the MedHallu dataset without question-level leakage."""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit

DATASET_NAME = "UTAustin-AIHealth/MedHallu"


def load_subset(name: str) -> pd.DataFrame:
    """Load one MedHallu config ("pqa_labeled" or "pqa_artificial") as a DataFrame.

    Adds a `qid` column (e.g. "pqa_artificial-2430") that keeps the original row
    number, so any sample can be traced back to the raw dataset.
    """
    df = load_dataset(DATASET_NAME, name)["train"].to_pandas()
    df.insert(0, "qid", [f"{name}-{i}" for i in range(len(df))])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicated questions (keep first) and rows where either answer is a single word.

    Returns a new DataFrame with a reset index; the input is not modified.
    """
    df = df.drop_duplicates(subset="Question", keep="first")
    gt_len = df["Ground Truth"].str.split().str.len()
    hallu_len = df["Hallucinated Answer"].str.split().str.len()
    keep = (gt_len > 1) & (hallu_len > 1)
    return df[keep].reset_index(drop=True)


def split_by_question(
    df: pd.DataFrame, val_size: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split question-level rows into train/val so no question appears in both."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["Question"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def to_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Expand each question into two samples: ground truth (label 0) and hallucinated (label 1).

    Must be called AFTER splitting, so both samples of a question stay in the same split.
    """
    base = pd.DataFrame({
        "qid": df["qid"],
        "question": df["Question"],
        "knowledge": df["Knowledge"].apply(" ".join),  # list of passages -> one string
        "difficulty": df["Difficulty Level"],
    })
    correct = base.assign(answer=df["Ground Truth"], label=0, hallu_category=None)
    hallucinated = base.assign(
        answer=df["Hallucinated Answer"], label=1,
        hallu_category=df["Category of Hallucination"],
    )
    return pd.concat([correct, hallucinated], ignore_index=True)


def build_splits(val_size: float = 0.15, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Full pipeline: train/val from pqa_artificial, test from pqa_labeled (sample-level)."""
    train_q, val_q = split_by_question(clean(load_subset("pqa_artificial")), val_size, seed)
    test_q = clean(load_subset("pqa_labeled"))
    return {
        "train": to_samples(train_q),
        "val": to_samples(val_q),
        "test": to_samples(test_q),
    }


if __name__ == "__main__":
    splits = build_splits()
    for name, df in splits.items():
        print(f"{name:5s}: {len(df):5d} samples, "
              f"{df['qid'].nunique():4d} questions, "
              f"hallucinated ratio = {df['label'].mean():.2f}")
