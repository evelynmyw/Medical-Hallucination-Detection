"""Final evaluation: choose each model's threshold on val, then evaluate ONCE on test.

Compares every trained BERT seed with the baselines and reports mean ± std over seeds.
Run this only after all models are final; looking at test results and then changing
the model would turn the test set into a second validation set.

Usage (from repo root):
    python -m src.evaluate --run-dir /content/drive/MyDrive/medhallu/bio_clinicalbert
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from src.baselines import make_baselines, pairwise_accuracy
from src.data import build_splits
from src.dataset import MedHalluDataset
from src.train import get_device

THRESHOLDS = np.round(np.linspace(0.01, 0.99, 99), 2)
LENGTH_MATCH_WORDS = 3  # "length-matched" = the two answers differ by at most this many words


# ---------- Threshold and metrics ----------

def choose_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    """Threshold that maximizes accuracy on the given (validation) data."""
    accs = [accuracy_score(y, prob >= t) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(accs))])


def length_matched_qids(df: pd.DataFrame, max_diff: int) -> set:
    """Questions whose ground-truth and hallucinated answers have similar word counts."""
    n_words = df["answer"].str.split().str.len()
    lens = df.assign(n_words=n_words).pivot(index="qid", columns="label", values="n_words")
    return set(lens.index[(lens[0] - lens[1]).abs() <= max_diff])


def compute_metrics(df: pd.DataFrame, prob: np.ndarray, threshold: float, matched_qids: set) -> dict:
    y = df["label"].to_numpy()
    pred = (prob >= threshold).astype(int)
    correct = pred == y

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "auc": roc_auc_score(y, prob),
        "pairwise_acc": pairwise_accuracy(df, prob),
    }
    by_diff = df.assign(correct=correct).groupby("difficulty")["correct"].mean()
    metrics.update({f"acc_{d}": acc for d, acc in by_diff.items()})
    metrics["acc_len_matched"] = correct[df["qid"].isin(matched_qids).to_numpy()].mean()
    return metrics


# ---------- BERT inference ----------

@torch.no_grad()
def predict_proba(model, tokenizer, df: pd.DataFrame, cfg: dict, device, batch_size: int = 64) -> np.ndarray:
    ds = MedHalluDataset(df, tokenizer, cfg["max_length"], cfg["use_knowledge"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=DataCollatorWithPadding(tokenizer))
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        probs.append(torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu())
    return torch.cat(probs).numpy()


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="directory containing seed*/best/")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    seed_dirs = sorted(p for p in run_dir.glob("seed*") if (p / "best").exists())
    if not seed_dirs:
        raise SystemExit(f"No seed*/best/ found in {run_dir}")
    cfg = json.loads((seed_dirs[0] / "config.json").read_text())

    splits = build_splits(val_size=cfg["val_size"], seed=cfg["split_seed"])
    train, val, test = splits["train"], splits["val"], splits["test"]
    y_val = val["label"].to_numpy()
    matched = length_matched_qids(test, LENGTH_MATCH_WORDS)
    print(f"Test: {test['qid'].nunique()} questions, {len(matched)} length-matched "
          f"(answers within {LENGTH_MATCH_WORDS} words)\n")

    rows = []

    # Baselines: fit on train, choose threshold on val, evaluate on test.
    for name, model in make_baselines(cfg["split_seed"]).items():
        model.fit(train, train["label"])
        threshold = choose_threshold(y_val, model.predict_proba(val)[:, 1])
        rows.append({"model": name, **compute_metrics(test, model.predict_proba(test)[:, 1], threshold, matched)})

    # BERT: one row per training seed.
    device = get_device()
    for seed_dir in seed_dirs:
        seed_cfg = json.loads((seed_dir / "config.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(seed_dir / "best")
        model = AutoModelForSequenceClassification.from_pretrained(seed_dir / "best").to(device)

        threshold = choose_threshold(y_val, predict_proba(model, tokenizer, val, seed_cfg, device))
        test_prob = predict_proba(model, tokenizer, test, seed_cfg, device)

        # Saved next to the model, so inference uses the same threshold (not argmax).
        (seed_dir / "threshold.json").write_text(json.dumps({"threshold": threshold}, indent=2))
        test.drop(columns="knowledge").assign(
            prob=test_prob, pred=(test_prob >= threshold).astype(int)
        ).to_csv(seed_dir / "test_predictions.csv", index=False)

        cm = confusion_matrix(test["label"], test_prob >= threshold)
        print(f"{seed_dir.name}: threshold {threshold:.2f}\n"
              f"  confusion matrix (rows = true [gt, hallu], cols = predicted)\n{cm}\n")
        rows.append({"model": f"bert_{seed_dir.name}", **compute_metrics(test, test_prob, threshold, matched)})

    table = pd.DataFrame(rows).set_index("model")
    bert = table[table.index.str.startswith("bert_")]
    table.loc["bert_mean"] = bert.mean()
    table.loc["bert_std"] = bert.std()  # sample std (ddof=1) across seeds

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("TEST RESULTS")
    print(table.round(3))
    table.to_csv(run_dir / "test_results.csv")
    print(f"\nSaved to {run_dir / 'test_results.csv'}")


if __name__ == "__main__":
    main()
