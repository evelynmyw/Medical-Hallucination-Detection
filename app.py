"""Gradio interface for prototyping, model visualization, and evaluation.

Tabs:
  1. Try it      - score a question/answer pair with a trained model
  2. Evaluation  - test-set metrics, PR curve, confusion matrix and score distribution,
                   with an adjustable decision threshold
  3. Errors      - browse test errors made by every training seed

The Evaluation and Errors tabs read the committed files in results/ and work without a model.

Usage (from repo root):
    python app.py --seed-dir /content/drive/MyDrive/medhallu/bio_clinicalbert/seed44 --share
    python app.py            # evaluation and error tabs only
"""

import argparse
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)

RESULTS_DIR = Path("results/bio_clinicalbert")
EXAMPLE_QIDS = ["pqa_labeled-0", "pqa_labeled-160", "pqa_labeled-490"]

# Colors (light theme): categorical slots 1-2, one-hue sequential ramp, recessive chrome.
GT_COLOR, HALLU_COLOR = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#104281"])
CLASS_NAMES = ["Ground truth", "Hallucinated"]


# ---------- Data ----------

def load_seed_predictions() -> dict[str, tuple[pd.DataFrame, float]]:
    out = {}
    for d in sorted(RESULTS_DIR.glob("seed*")):
        if (d / "test_predictions.csv").exists() and (d / "threshold.json").exists():
            df = pd.read_csv(d / "test_predictions.csv")
            out[d.name] = (df, json.loads((d / "threshold.json").read_text())["threshold"])
    return out


def metrics_at_threshold(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
    }


# ---------- Plots ----------

def new_axes(title: str, figsize=(5, 4)):
    fig = Figure(figsize=figsize, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    return fig, ax


def plot_pr_curve(y, prob, threshold) -> Figure:
    precision, recall, _ = precision_recall_curve(y, prob)
    m = metrics_at_threshold(y, prob, threshold)
    fig, ax = new_axes(f"Precision–recall curve (AP = {average_precision_score(y, prob):.3f})")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(y.mean(), color=MUTED, linewidth=1, linestyle="--")
    ax.text(0.02, y.mean() + 0.01, "random classifier", color=MUTED, fontsize=8, va="bottom")
    ax.plot(recall, precision, color=HALLU_COLOR, linewidth=2)
    ax.plot(m["Recall"], m["Precision"], "o", markersize=9, color=HALLU_COLOR,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    ax.annotate(f"threshold {threshold:.2f}\nP {m['Precision']:.3f} · R {m['Recall']:.3f}",
                (m["Recall"], m["Precision"]), xytext=(-10, -14), textcoords="offset points",
                ha="right", va="top", fontsize=9, color=INK_2)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Recall (hallucinations caught)", color=INK_2)
    ax.set_ylabel("Precision", color=INK_2)
    return fig


def plot_confusion(y, prob, threshold) -> Figure:
    cm = confusion_matrix(y, prob >= threshold)
    fig, ax = new_axes(f"Confusion matrix at threshold {threshold:.2f}", figsize=(4.6, 4))
    ax.imshow(cm, cmap=BLUE_RAMP, vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            share = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]}\n{share:.1%}", ha="center", va="center", fontsize=11,
                    color="#ffffff" if cm[i, j] > cm.max() * 0.5 else INK)
    ax.set_xticks([0, 1], [f"Pred: {c.lower()}" for c in CLASS_NAMES])
    ax.set_yticks([0, 1], [f"True: {c.lower()}" for c in CLASS_NAMES], rotation=90, va="center")
    ax.tick_params(length=0, colors=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def plot_score_distribution(y, prob, threshold) -> Figure:
    fig, ax = new_axes("Predicted P(hallucinated) by true label", figsize=(5, 4))
    bins = np.linspace(0, 1, 41)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.hist(prob[y == 0], bins=bins, histtype="step", linewidth=2, color=GT_COLOR, label=CLASS_NAMES[0])
    ax.hist(prob[y == 1], bins=bins, histtype="step", linewidth=2, color=HALLU_COLOR, label=CLASS_NAMES[1])
    ax.axvline(threshold, color=INK, linewidth=1, linestyle="--", label=f"Threshold {threshold:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("P(hallucinated)", color=INK_2)
    ax.set_ylabel("Samples (log scale)", color=INK_2)
    fig.legend(loc="outside lower center", ncols=3, frameon=False, fontsize=9, labelcolor=INK_2)
    return fig


def probability_bar(p: float, threshold: float) -> str:
    """Horizontal bar of P(hallucinated) with the decision threshold marked.

    Replaces gr.Label, whose headline is the argmax class and would contradict
    the thresholded prediction whenever 0.5 <= p < threshold.
    """
    color = HALLU_COLOR if p >= threshold else GT_COLOR
    return f"""
<div style="font-family: system-ui, sans-serif; color: {INK_2}; font-size: 13px; padding: 28px 0 8px;">
  <div style="position: relative; height: 14px; background: {GRID}; border-radius: 4px;">
    <div style="width: {p * 100:.1f}%; height: 100%; background: {color}; border-radius: 4px;"></div>
    <div style="position: absolute; left: {threshold * 100:.1f}%; top: -5px; bottom: -5px;
                width: 2px; background: {INK};"></div>
    <div style="position: absolute; left: {threshold * 100:.1f}%; bottom: 22px;
                transform: translateX(-50%); white-space: nowrap;">threshold {threshold:.2f}</div>
  </div>
  <div style="display: flex; justify-content: space-between; margin-top: 8px; color: {MUTED};">
    <span>0 · ground truth</span><span>1 · hallucinated</span>
  </div>
</div>"""


# ---------- App ----------

def build_app(detector, seed_preds: dict) -> gr.Blocks:
    seeds = list(seed_preds)
    with gr.Blocks(title="MedHallu Hallucination Detector") as demo:
        gr.Markdown(
            "# MedHallu hallucination detector\n"
            "Bio_ClinicalBERT fine-tuned to flag hallucinated answers to medical questions. "
            "Research prototype, **not for clinical use**. The model relies partly on writing "
            "style (hedged vs. assertive wording), so it can miss hallucinations that change a single fact."
        )

        # ----- Tab 1: Try it -----
        with gr.Tab("Try it"):
            if detector is None:
                gr.Markdown("No model loaded. Start the app with `--seed-dir <path to seedXX>` to enable this tab.")
            else:
                gr.Markdown(f"Decision threshold: **{detector.threshold:.2f}** (chosen on the validation set).")
                with gr.Row():
                    with gr.Column():
                        question = gr.Textbox(label="Question", lines=2)
                        answer = gr.Textbox(label="Answer", lines=5)
                        reference = gr.Textbox(label="Dataset label (filled in by the examples below)",
                                               interactive=False)
                        run = gr.Button("Check answer", variant="primary")
                    with gr.Column():
                        verdict = gr.Markdown()
                        prob_bar = gr.HTML()

                def score(q, a):
                    if not q.strip() or not a.strip():
                        return "Enter both a question and an answer.", ""
                    r = detector.predict(q, a)
                    p, t = r["p_hallucinated"], r["threshold"]
                    name = "Hallucinated" if r["label"] == "hallucinated" else "Ground truth"
                    return (f"### Prediction: {name}\nP(hallucinated) = **{p:.3f}**, "
                            f"threshold {t:.2f} (hallucinated if P ≥ threshold)",
                            probability_bar(p, t))

                run.click(score, [question, answer], [verdict, prob_bar])
                # Typing your own text clears the label that came from an example.
                for box in (question, answer):
                    box.input(lambda: "", None, reference)

                if seeds:
                    test_df = seed_preds[seeds[0]][0]
                    examples = test_df[test_df["qid"].isin(EXAMPLE_QIDS)].sort_values(["qid", "label"])
                    examples = examples.assign(
                        reference=examples["qid"] + ": " + examples["label"].map(
                            {0: "ground truth", 1: "hallucinated"}))
                    gr.Examples(
                        examples=examples[["question", "answer", "reference"]].values.tolist(),
                        inputs=[question, answer, reference],
                        label="Test-set pairs. pqa_labeled-160 and -490 are known failure cases.",
                    )

        # ----- Tab 2: Evaluation -----
        with gr.Tab("Evaluation"):
            if not seeds:
                gr.Markdown(f"No predictions found in `{RESULTS_DIR}`. Run `python -m src.evaluate` first.")
            else:
                results_csv = RESULTS_DIR / "test_results.csv"
                if results_csv.exists():
                    cols = ["accuracy", "precision", "recall", "f1", "auc", "acc_len_matched"]
                    summary = (pd.read_csv(results_csv, index_col=0)[cols]
                               .map(lambda v: f"{v:.3f}").reset_index())  # fixed 3 decimals
                    gr.Markdown("### Test set (`pqa_labeled`, 2,000 samples): all models")
                    gr.Dataframe(summary, interactive=False)

                gr.Markdown("### Explore one seed\nMove the threshold to see the precision/recall trade-off. "
                            "The default is the threshold chosen on the validation set.")
                with gr.Row():
                    seed = gr.Dropdown(seeds, value=seeds[-1], label="Training seed")
                    threshold = gr.Slider(0.01, 0.99, step=0.01, value=seed_preds[seeds[-1]][1],
                                          label="Decision threshold")
                metrics_md = gr.Markdown()
                with gr.Row():
                    # show_label=False: the figure titles already name each plot, and
                    # Gradio's label badge would cover them.
                    pr_plot = gr.Plot(show_label=False)
                    cm_plot = gr.Plot(show_label=False)
                    dist_plot = gr.Plot(show_label=False)

                def update(s, t):
                    df = seed_preds[s][0]
                    y, prob = df["label"].to_numpy(), df["prob"].to_numpy()
                    m = metrics_at_threshold(y, prob, t)
                    table = ("| " + " | ".join(m) + " |\n|" + "---|" * len(m) + "\n| "
                             + " | ".join(f"{v:.3f}" for v in m.values()) + " |")
                    return (table, plot_pr_curve(y, prob, t), plot_confusion(y, prob, t),
                            plot_score_distribution(y, prob, t))

                outputs = [metrics_md, pr_plot, cm_plot, dist_plot]
                seed.change(lambda s: seed_preds[s][1], seed, threshold).then(update, [seed, threshold], outputs)
                threshold.release(update, [seed, threshold], outputs)
                demo.load(update, [seed, threshold], outputs)

        # ----- Tab 3: Errors -----
        with gr.Tab("Errors"):
            errors_csv = RESULTS_DIR / "consistent_errors.csv"
            if not errors_csv.exists():
                gr.Markdown(f"`{errors_csv}` not found. Run `python -m src.error_analysis` first.")
            else:
                errors = pd.read_csv(errors_csv)
                errors["error_type"] = np.where(errors["label"] == 1, "Missed hallucination",
                                                "Flagged ground truth")
                view_cols = ["qid", "error_type", "hallu_category", "difficulty", "mean_prob",
                             "question", "answer"]
                errors["hallu_category"] = errors["hallu_category"].str.replace("#Question#", "question")
                errors = errors[view_cols].round({"mean_prob": 3})
                gr.Markdown(f"{len(errors)} test answers misclassified by **every** training seed. "
                            "`mean_prob` is P(hallucinated) averaged over seeds.")
                kind = gr.Radio(["All", "Missed hallucination", "Flagged ground truth"], value="All",
                                label="Error type")
                table = gr.Dataframe(errors, interactive=False, wrap=True,
                                     column_widths=["130px", "150px", "190px", "100px", "100px",
                                                    "300px", "420px"])
                kind.change(lambda k: errors if k == "All" else errors[errors["error_type"] == k],
                            kind, table)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", help="seedXX dir with best/, config.json, threshold.json")
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = parser.parse_args()

    detector = None
    if args.seed_dir:
        from src.predict import HallucinationDetector  # heavy import only when a model is used
        detector = HallucinationDetector(args.seed_dir)
        print(f"Loaded model from {args.seed_dir} on {detector.device}")

    build_app(detector, load_seed_predictions()).launch(share=args.share)


if __name__ == "__main__":
    main()
