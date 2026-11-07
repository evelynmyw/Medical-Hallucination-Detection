"""Error analysis of BERT test predictions across all training seeds.

Focuses on errors made by EVERY seed (systematic weaknesses, not seed noise).

Usage (from repo root):
    python -m src.error_analysis                                 # uses results/bio_clinicalbert
    python -m src.error_analysis --run-dir <dir> --n-examples 5
"""

import argparse
import textwrap
from pathlib import Path

import pandas as pd

# Hedging words: typical of the human-written ground-truth conclusions (see README findings).
HEDGE_PATTERN = r"\b(?:may|might|could|suggests?|possibly|likely)\b"


def load_predictions(run_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """One row per test sample, with prob_<seed> and pred_<seed> columns for every seed."""
    seed_dirs = sorted(p for p in run_dir.glob("seed*") if (p / "test_predictions.csv").exists())
    if not seed_dirs:
        raise SystemExit(f"No seed*/test_predictions.csv found in {run_dir}")

    merged = None
    for d in seed_dirs:
        df = pd.read_csv(d / "test_predictions.csv")
        if merged is None:
            merged = df.drop(columns=["prob", "pred"])
        else:
            # All seeds were evaluated on the same test set in the same order.
            assert (df["qid"] == merged["qid"]).all() and (df["label"] == merged["label"]).all()
        merged[f"prob_{d.name}"] = df["prob"]
        merged[f"pred_{d.name}"] = df["pred"]
    return merged, [d.name for d in seed_dirs]


def add_error_columns(df: pd.DataFrame, seeds: list[str]) -> pd.DataFrame:
    pred_cols = [f"pred_{s}" for s in seeds]
    prob_cols = [f"prob_{s}" for s in seeds]
    return df.assign(
        n_wrong=df[pred_cols].ne(df["label"], axis=0).sum(axis=1),
        mean_prob=df[prob_cols].mean(axis=1),
        n_words=df["answer"].str.split().str.len(),
        hedged=df["answer"].str.contains(HEDGE_PATTERN, case=False, regex=True),
    )


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(str(text), width=100, initial_indent=indent, subsequent_indent=indent)


def print_examples(errors: pd.DataFrame, all_rows: pd.DataFrame, n: int) -> None:
    """Show each wrong answer next to the other answer for the same question."""
    other = all_rows.set_index(["qid", "label"])
    for _, row in errors.head(n).iterrows():
        pair = other.loc[(row["qid"], 1 - row["label"])]
        print(f"\n--- {row['qid']} | difficulty: {row['difficulty']} | "
              f"category: {row['hallu_category'] if row['label'] == 1 else pair['hallu_category']}")
        print(f"  Q: {row['question']}")
        print(f"  Ground truth (P(hallu) = {(pair if row['label'] == 1 else row)['mean_prob']:.2f}):")
        print(wrap(pair["answer"] if row["label"] == 1 else row["answer"]))
        print(f"  Hallucinated (P(hallu) = {(row if row['label'] == 1 else pair)['mean_prob']:.2f}):")
        print(wrap(row["answer"] if row["label"] == 1 else pair["answer"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="results/bio_clinicalbert")
    parser.add_argument("--n-examples", type=int, default=5)
    args = parser.parse_args()
    pd.set_option("display.width", 200)

    run_dir = Path(args.run_dir)
    df, seeds = load_predictions(run_dir)
    df = add_error_columns(df, seeds)
    k = len(seeds)
    print(f"Seeds: {', '.join(seeds)} | test samples: {len(df)}")

    # 1. How consistent are the errors across seeds?
    print_section(f"1. Number of seeds (out of {k}) that got each sample wrong")
    counts = df.groupby(["label", "n_wrong"]).size().unstack(fill_value=0)
    counts.index = counts.index.map({0: "ground truth (0)", 1: "hallucinated (1)"})
    print(counts)

    missed = df[(df["label"] == 1) & (df["n_wrong"] == k)]          # false negatives, all seeds
    caught = df[(df["label"] == 1) & (df["n_wrong"] == 0)]
    false_alarm = df[(df["label"] == 0) & (df["n_wrong"] == k)]     # false positives, all seeds
    accepted = df[(df["label"] == 0) & (df["n_wrong"] == 0)]

    # 2. Which hallucination categories are missed?
    print_section("2. Hallucinated answers by category")
    hallu = df[df["label"] == 1]
    by_cat = hallu.groupby("hallu_category").agg(
        n=("qid", "size"),
        recall_mean=("n_wrong", lambda w: 1 - w.mean() / k),
        missed_by_all=("n_wrong", lambda w: int((w == k).sum())),
    ).sort_values("recall_mean")
    print(by_cat.round(3))

    # 3. Do errors differ in length or hedging?
    print_section("3. Length and hedging of errors vs correct predictions")
    groups = {
        "hallucinated, missed by all": missed,
        "hallucinated, caught by all": caught,
        "ground truth, flagged by all": false_alarm,
        "ground truth, accepted by all": accepted,
    }
    summary = pd.DataFrame({
        name: {"n": len(g), "median_words": g["n_words"].median(), "hedged_share": g["hedged"].mean()}
        for name, g in groups.items()
    }).T
    print(summary.round(3))
    print(f"\n(hedged = contains one of: {HEDGE_PATTERN})")

    # 4. Examples
    print_section(f"4a. Hallucinations missed by all seeds (first {args.n_examples})")
    print_examples(missed.sort_values("mean_prob"), df, args.n_examples)
    print_section(f"4b. Ground-truth answers flagged by all seeds (first {args.n_examples})")
    print_examples(false_alarm.sort_values("mean_prob", ascending=False), df, args.n_examples)

    out = run_dir / "consistent_errors.csv"
    pd.concat([missed, false_alarm]).to_csv(out, index=False)
    print(f"\nSaved {len(missed) + len(false_alarm)} consistent errors to {out}")


if __name__ == "__main__":
    main()
