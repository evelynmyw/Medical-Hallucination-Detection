"""Single inference entry point: load a trained seed and apply its validation-chosen threshold.

The Gradio app, the FastAPI service and the command line all use this module, so there is
one prediction path. It imports only torch + transformers, to keep the Docker image small.

Usage (from repo root):
    python -m src.predict --seed-dir <dir>/seed44 \
        --question "Does X reduce Y?" --answer "X significantly reduces Y."
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HallucinationDetector:
    """Wraps a seed directory containing best/, config.json and threshold.json."""

    def __init__(self, seed_dir: str | Path, device: torch.device | None = None):
        seed_dir = Path(seed_dir)
        cfg = json.loads((seed_dir / "config.json").read_text())
        if cfg.get("use_knowledge"):
            raise NotImplementedError("This detector only supports question + answer inputs.")

        self.max_length = cfg["max_length"]
        # Threshold chosen on the validation set by src/evaluate.py (not argmax / 0.5).
        self.threshold = json.loads((seed_dir / "threshold.json").read_text())["threshold"]
        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(seed_dir / "best")
        self.model = AutoModelForSequenceClassification.from_pretrained(seed_dir / "best")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_proba(self, questions: list[str], answers: list[str], batch_size: int = 32) -> np.ndarray:
        """P(hallucinated) for each (question, answer) pair."""
        probs = []
        for i in range(0, len(questions), batch_size):
            enc = self.tokenizer(
                questions[i:i + batch_size], answers[i:i + batch_size],
                truncation=True, max_length=self.max_length, padding=True, return_tensors="pt",
            ).to(self.device)
            probs.append(torch.softmax(self.model(**enc).logits, dim=-1)[:, 1].cpu())
        return torch.cat(probs).numpy()

    def predict(self, question: str, answer: str) -> dict:
        p = float(self.predict_proba([question], [answer])[0])
        return {
            "p_hallucinated": p,
            "threshold": self.threshold,
            "label": "hallucinated" if p >= self.threshold else "ground_truth",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--answer", required=True)
    args = parser.parse_args()

    detector = HallucinationDetector(args.seed_dir)
    print(json.dumps(detector.predict(args.question, args.answer), indent=2))


if __name__ == "__main__":
    main()
