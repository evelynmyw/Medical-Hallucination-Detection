"""Simple non-neural baselines that the BERT model must beat.

Usage (from repo root):
    python -m src.baselines              # fit on train, report on val
    python -m src.baselines --eval-test  # also report on test (do this once, at the end)
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.data import build_splits


# ---------- Input views: which part of a sample each baseline is allowed to see ----------

def answer_length(df: pd.DataFrame) -> pd.DataFrame:
    """Single feature: log word count of the answer."""
    return np.log1p(df["answer"].str.split().str.len()).to_frame()


def answer_text(df: pd.DataFrame) -> pd.Series:
    return df["answer"]


def question_answer_text(df: pd.DataFrame) -> pd.Series:
    return df["question"] + " [SEP] " + df["answer"]


def tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50_000)


def make_baselines(seed: int = 42) -> dict:
    lr = lambda: LogisticRegression(max_iter=1000, random_state=seed)
    return {
        "length_only": make_pipeline(FunctionTransformer(answer_length), StandardScaler(), lr()),
        "tfidf_answer": make_pipeline(FunctionTransformer(answer_text), tfidf(), lr()),
        "tfidf_question_answer": make_pipeline(FunctionTransformer(question_answer_text), tfidf(), lr()),
    }


# ---------- Metrics ----------

def pairwise_accuracy(df: pd.DataFrame, prob: np.ndarray) -> float:
    """Share of questions where the hallucinated answer gets a higher score than the ground truth."""
    p = df.assign(prob=prob).pivot(index="qid", columns="label", values="prob")
    return float((p[1] > p[0]).mean())


def evaluate(model, df: pd.DataFrame, threshold: float = 0.5) -> dict:
    prob = model.predict_proba(df)[:, 1]
    pred = (prob >= threshold).astype(int)
    y = df["label"].to_numpy()

    results = {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "auc": roc_auc_score(y, prob),
        "pairwise_acc": pairwise_accuracy(df, prob),
    }
    by_diff = df.assign(correct=(pred == y)).groupby("difficulty")["correct"].mean()
    results.update({f"acc_{d}": acc for d, acc in by_diff.items()})
    return results


def top_features(model, k: int = 20) -> pd.DataFrame:
    """Words with the largest positive (-> hallucinated) and negative (-> ground truth) weights."""
    vectorizer, clf = model[-2], model[-1]
    words = vectorizer.get_feature_names_out()
    coef = clf.coef_[0]
    order = np.argsort(coef)
    return pd.DataFrame({
        "ground_truth_word": words[order[:k]],
        "gt_weight": coef[order[:k]].round(2),
        "hallucinated_word": words[order[::-1][:k]],
        "hallu_weight": coef[order[::-1][:k]].round(2),
    })


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-test", action="store_true", help="also report test metrics")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = build_splits(seed=args.seed)
    eval_sets = ["val", "test"] if args.eval_test else ["val"]

    rows = []
    models = make_baselines(args.seed)
    for name, model in models.items():
        model.fit(splits["train"], splits["train"]["label"])
        for split in eval_sets:
            rows.append({"model": name, "split": split, **evaluate(model, splits[split])})

    table = pd.DataFrame(rows).set_index(["model", "split"])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(table.round(3))

    print("\nMost predictive words for tfidf_answer:")
    print(top_features(models["tfidf_answer"]).to_string(index=False))


if __name__ == "__main__":
    main()
