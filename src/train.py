"""Fine-tune a BERT-style model for hallucination detection with a hand-written PyTorch loop.

Usage (from repo root):
    python -m src.train --debug 50          # quick smoke test on 50 train / 50 val questions
    python -m src.train --seed 42           # full run
    python -m src.train --seed 43 --output-dir /content/drive/MyDrive/medhallu/bio_clinicalbert
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to file, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from src.data import build_splits
from src.dataset import MedHalluDataset


# ---------- Setup helpers ----------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def subsample(df: pd.DataFrame, n_questions: int, seed: int) -> pd.DataFrame:
    """Keep n whole questions (both samples of each), for quick debugging.

    Taking df.head(n) would be wrong: samples are ordered label 0 first, then label 1.
    """
    qids = df["qid"].drop_duplicates()
    keep = qids.sample(n=min(n_questions, len(qids)), random_state=seed)
    return df[df["qid"].isin(keep)].reset_index(drop=True)


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with no weight decay on biases and LayerNorm weights (all 1-D parameters)."""
    decay, no_decay = [], []
    for param in model.parameters():
        (no_decay if param.ndim == 1 else decay).append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


# ---------- Train / eval ----------

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, grad_clip, log_every) -> float:
    model.train()  # enables dropout
    total_loss, n_seen = 0.0, 0
    for step, batch in enumerate(loader, start=1):
        batch = to_device(batch, device)
        labels = batch.pop("labels")

        logits = model(**batch).logits          # shape: (batch_size, 2)
        loss = loss_fn(logits, labels)          # mean cross-entropy over the batch

        optimizer.zero_grad()                   # clear gradients from the previous step
        loss.backward()                         # compute new gradients
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()                        # update weights
        scheduler.step()                        # update learning rate

        total_loss += loss.item() * len(labels)
        n_seen += len(labels)
        if step % log_every == 0:
            print(f"  step {step}/{len(loader)}  train_loss {total_loss / n_seen:.4f}  "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")
    return total_loss / n_seen


@torch.no_grad()  # no gradient tracking: faster and uses less memory
def evaluate(model, loader, loss_fn, device) -> dict:
    model.eval()  # disables dropout
    total_loss, all_probs, all_labels = 0.0, [], []
    for batch in loader:
        batch = to_device(batch, device)
        labels = batch.pop("labels")
        logits = model(**batch).logits
        total_loss += loss_fn(logits, labels).item() * len(labels)
        all_probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())  # P(hallucinated)
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs).numpy()
    y = torch.cat(all_labels).numpy()
    return {
        "loss": total_loss / len(y),
        "accuracy": accuracy_score(y, probs >= 0.5),
        "auc": roc_auc_score(y, probs),
        "pred_positive_rate": float((probs >= 0.5).mean()),  # ~1.0 means "always hallucinated"
    }


def plot_history(history: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    plt.plot(epochs, [h["val_loss"] for h in history], marker="o", label="val")
    plt.xticks(epochs)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ---------- Main ----------

def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # show each log line immediately (e.g. in Colab)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, help="override training seed")
    parser.add_argument("--output-dir", help="override output directory (e.g. Google Drive)")
    parser.add_argument("--debug", type=int, metavar="N", help="use only N train and N val questions")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    set_seed(cfg["seed"])
    device = get_device()
    out_dir = Path(cfg["output_dir"]) / f"seed{cfg['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Device: {device} | Output: {out_dir}")

    # Data: the test split is deliberately not used here.
    splits = build_splits(val_size=cfg["val_size"], seed=cfg["split_seed"])
    train_df, val_df = splits["train"], splits["val"]
    if args.debug:
        train_df = subsample(train_df, args.debug, cfg["seed"])
        val_df = subsample(val_df, args.debug, cfg["seed"])
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    collator = DataCollatorWithPadding(tokenizer)
    train_ds = MedHalluDataset(train_df, tokenizer, cfg["max_length"], cfg["use_knowledge"])
    val_ds = MedHalluDataset(val_df, tokenizer, cfg["max_length"], cfg["use_knowledge"])

    generator = torch.Generator().manual_seed(cfg["seed"])  # reproducible shuffling
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=collator, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
                            collate_fn=collator)

    # Model: pretrained encoder + a NEW randomly initialized 2-class head on top.
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2)
    model.to(device)

    optimizer = make_optimizer(model, cfg["lr"], cfg["weight_decay"])
    total_steps = len(train_loader) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(cfg["warmup_ratio"] * total_steps), num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss, epochs_without_improvement, history = float("inf"), 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn,
                                     device, cfg["grad_clip"], cfg["log_every"])
        val = evaluate(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        **{f"val_{k}": v for k, v in val.items()}})
        print(f"Epoch {epoch}: train_loss {train_loss:.4f} | val_loss {val['loss']:.4f} | "
              f"val_acc {val['accuracy']:.4f} | val_auc {val['auc']:.4f} | "
              f"pred_pos {val['pred_positive_rate']:.2f} | {time.time() - start:.0f}s")

        # Save logs every epoch, so an interrupted run still leaves a record.
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        plot_history(history, out_dir / "loss_curve.png")

        # Early stopping on validation loss; keep only the best checkpoint.
        if val["loss"] < best_val_loss:
            best_val_loss, epochs_without_improvement = val["loss"], 0
            model.save_pretrained(out_dir / "best")
            tokenizer.save_pretrained(out_dir / "best")
            print(f"  saved best model (val_loss {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg["patience"]:
                print(f"Early stopping: no val-loss improvement for {cfg['patience']} epochs")
                break

    print(f"Done. Best val_loss {best_val_loss:.4f}. Files in {out_dir}")


if __name__ == "__main__":
    main()
