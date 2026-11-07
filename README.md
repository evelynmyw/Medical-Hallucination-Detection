# Medical Hallucination Detection

Binary classifier that flags hallucinated answers in medical QA, trained on the MedHallu dataset. A fine-tuned Bio_ClinicalBERT reaches **93.8% ± 0.7% test accuracy** (3 seeds), compared with 90.1% for a TF-IDF + logistic regression baseline. That baseline is strong because, on this dataset, much of the signal comes from surface style rather than factual content (see [Findings](#findings)).

## Task

Input: a medical question and a candidate answer. Output: the probability that the answer is hallucinated (label 1) rather than the ground truth (label 0).

## Data and split

- Dataset: [`UTAustin-AIHealth/MedHallu`](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu). Each row is one question with one ground-truth answer and one hallucinated answer.
- Train / validation: `pqa_artificial` (9,000 questions), split **by question** 85/15 with `GroupShuffleSplit`.
- Test: `pqa_labeled` (1,000 questions, human-curated). No question overlaps with `pqa_artificial`.
- Each question is expanded into two samples only *after* splitting, so both answers to a question always land in the same split. `tests/test_split.py` checks that no question is shared between splits.
- Cleaning: dropped 1 duplicated question and 2 rows with a single-word answer (truncated generation `"VSL"`, non-answer `"P<0.05."`) from `pqa_artificial`.

| Split | Questions | Samples |
|---|---|---|
| train | 7,647 | 15,294 |
| val | 1,350 | 2,700 |
| test | 1,000 | 2,000 |

## Method

**Baselines** (`src/baselines.py`): logistic regression on (1) answer word count only, (2) TF-IDF of the answer only, (3) TF-IDF of question + answer.

**Model** (`src/train.py`): `emilyalsentzer/Bio_ClinicalBERT` with a 2-class head. Input is encoded as a sentence pair `[CLS] question [SEP] answer [SEP]`, max 256 tokens, with dynamic padding. The training loop is written in plain PyTorch:

- AdamW, lr 2e-5, weight decay 0.01 (not applied to biases and LayerNorm), batch size 16
- linear schedule with 10% warmup, gradient clipping at 1.0
- up to 4 epochs, early stopping on validation loss (patience 2), best checkpoint kept
- 3 training seeds (42, 43, 44) on the same data split

Config: `configs/default.yaml`.

**Evaluation** (`src/evaluate.py`): for every model, including the baselines, the decision threshold is chosen on the validation set (maximizing accuracy) and saved to `threshold.json`. The test set was evaluated once, after all models were final.

## Results

Test set (`pqa_labeled`, 2,000 samples). BERT is reported as mean ± std over 3 seeds.

| Model | Accuracy | Precision | Recall | F1 | AUC | Acc. (length-matched) |
|---|---|---|---|---|---|---|
| Length only | 0.710 | 0.733 | 0.662 | 0.696 | 0.740 | 0.515 |
| TF-IDF + LR (answer) | 0.901 | 0.948 | 0.849 | 0.896 | 0.972 | 0.831 |
| TF-IDF + LR (question + answer) | 0.882 | 0.941 | 0.815 | 0.874 | 0.962 | 0.807 |
| **Bio_ClinicalBERT** | **0.938 ± 0.007** | **0.962 ± 0.007** | **0.911 ± 0.008** | **0.936 ± 0.007** | **0.987 ± 0.003** | **0.870 ± 0.020** |

*Length-matched*: the 166 test questions whose two answers differ by at most 3 words (332 samples).

Accuracy by difficulty level (test):

| Model | Easy | Medium | Hard |
|---|---|---|---|
| Length only | 0.732 | 0.684 | 0.717 |
| TF-IDF + LR (answer) | 0.909 | 0.901 | 0.896 |
| Bio_ClinicalBERT | 0.934 ± 0.008 | 0.939 ± 0.011 | 0.939 ± 0.005 |

Per-seed results, training curves, thresholds and test predictions: [`results/bio_clinicalbert/`](results/bio_clinicalbert/).

## Findings

**1. Answers can be told apart by surface features.** Ground-truth answers are longer than hallucinated ones in 78.4% (`pqa_labeled`) and 73.4% (`pqa_artificial`) of question pairs. A bag-of-words logistic regression on the answer alone reaches 92.8% validation accuracy, suggesting that MedHallu's hallucinated answers are largely distinguishable by surface style rather than factual content.

The most predictive TF-IDF features show the style difference. Ground-truth answers (written by paper authors) hedge: *may, be, might, could, should, our, we, results, data*. Hallucinated answers (LLM-generated) make confident causal claims: *significantly, enhances, due to, rather than, primarily, inversely, increases, exacerbates*.

**2. BERT beats the baselines, by a moderate margin.** On test, BERT is 3.7 points more accurate than the best baseline (0.938 vs 0.901), roughly 5 standard deviations across seeds. It cuts the error rate from 9.9% to 6.2%.

**3. Part of every model's accuracy comes from answer length.** On length-matched pairs, the length-only baseline falls to chance (0.515), which confirms the subset removes the length cue. TF-IDF drops by 7.0 points and BERT by 6.8 points. BERT keeps its lead (0.870 vs 0.831), but it does not escape the drop. Length-matched pairs may also be harder for other reasons, so the drop is an upper bound on how much the models rely on length.

**4. Errors are mostly missed hallucinations.** Precision (0.962) is higher than recall (0.911): across seeds the model misses 80–94 of the 1,000 hallucinated answers and wrongly flags 29–43 of the 1,000 correct ones.

**5. Overfitting starts early.** Validation loss is lowest after epoch 1 or 2 for every seed. Training loss keeps falling to 0.03–0.06 while validation loss rises, as the model becomes more confident on the examples it gets wrong.

## Error analysis

`src/error_analysis.py` looks at test errors made by **all three seeds**, which reflect systematic weaknesses rather than seed noise. 53 hallucinated answers are missed by every seed, and 16 ground-truth answers are flagged by every seed.

**Missed hallucinations by category** (recall averaged over seeds):

| Category | n | Recall | Missed by all seeds |
|---|---|---|---|
| Incomplete Information | 212 | 0.862 | 20 |
| Misinterpretation of Question | 752 | 0.922 | 33 |
| Mechanism and Pathway Misattribution | 33 | 0.980 | 0 |
| Methodological and Evidence Fabrication | 3 | 1.000 | 0 |

Mechanism-type hallucinations, which use the confident causal wording the TF-IDF baseline picked up, are almost always caught.

**Errors follow the hedging style.** Share of answers containing a hedging word (*may, might, could, suggest(s), possibly, likely*):

| Group | n | Median words | Hedged |
|---|---|---|---|
| Hallucinated, missed by all seeds | 53 | 25 | 26.4% |
| Hallucinated, caught by all seeds | 873 | 23 | 7.2% |
| Ground truth, flagged by all seeds | 16 | 21.5 | 6.2% |
| Ground truth, accepted by all seeds | 939 | 38 | 42.8% |

Hallucinations that hedge are missed more often; ground-truth answers that are short and assertive are flagged. The groups of consistent errors are small (53 and 16), and the hedging word list is a coarse proxy.

**Typical failures.** Missed hallucinations are usually near-copies of the ground truth with one fact changed, which the model scores the same as the original (P(hallucinated) ≈ 0.01 for both):

- *Ground truth:* "Systolic BP measured by the **nurse** … is significantly lower than the readings obtained by the **physician** …"
  *Hallucinated:* "Systolic BP measured by the **physician** … is significantly lower than the readings obtained by the **nurse** …"
- *Ground truth:* "VAT area measurement at a single level 3 cm **above** the lower margin of the L3 vertebra …"
  *Hallucinated:* "… 3 cm **below** the lower margin of the L3 vertebra …"
- *Ground truth:* "… use of prophylactic antibiotics for TIVAD placement is **not recommended**."
  *Hallucinated:* "… routine use of prophylactic antibiotics is **advised** …"

Flagged ground-truth answers are short, definitive or negative conclusions ("Skewed X-inactivation is **not associated** with recurrent spontaneous abortion …"; "Alcohol drinking during meals was **inversely** related with risk of acute MI …"), and the model scores both answers of the pair as hallucinated (P ≈ 0.93–1.00).

In both cases the model gives the two answers of a question nearly the same score: it judges each answer's style in isolation and cannot tell which version of a fact is correct.

**A one-word probe.** Two content-free answers to the placeholder question "Does X reduce Y?", scored by the deployed model (seed 44) through the API:

| Answer | P(hallucinated) | Prediction |
|---|---|---|
| "X **significantly** reduces Y." | 0.998 | hallucinated |
| "X **may** reduce Y." | 0.012 | ground truth |

Neither sentence contains any medical content; swapping one hedging word flips the prediction. Without the source evidence, these pairs cannot be resolved from the question and answer alone, which motivates adding the `Knowledge` passages to the input.

Full list: `results/bio_clinicalbert/consistent_errors.csv`.

## Limitations

- **Style shortcut.** The dataset's hallucinated answers differ from the ground truth in length and wording. High accuracy here does not show that the model checks facts; a hallucination written in cautious, author-like language would likely be missed.
- **Train/test generation differ.** Training uses `pqa_artificial` (automatically generated questions, some of them ungrammatical); testing uses `pqa_labeled`. Results may not transfer to other sources of hallucinations, such as answers from other LLMs.
- **Missed hallucinations matter more.** In a medical setting a false negative is usually worse than a false positive; recall (0.911) is the weaker metric.
- **Unstable thresholds.** The validation-selected threshold varies widely between seeds (0.61–0.94) while accuracy barely changes, because most predicted probabilities sit near 0 or 1. The threshold is a fragile choice.
- **Small subsets.** The length-matched subset has only 166 questions, so its numbers have a larger margin of error (std across seeds 0.020).
- **Only exact-duplicate leakage is tested.** Paraphrased questions, or questions from the same source paper, are not detected.
- **Knowledge passages are not used.** The model sees only the question and the answer.

## Demo and deployment

All three entry points share one inference module, `src/predict.py`, which loads a trained seed and applies its validation-chosen threshold from `threshold.json` (not argmax). Device selection: CUDA → MPS → CPU.

**Gradio interface** (`app.py`), for prototyping and model inspection:

- *Try it*: score a question/answer pair (needs a trained model).
- *Evaluation*: test metrics for all models, plus PR curve, confusion matrix and score distribution for one seed, with an adjustable threshold.
- *Errors*: browse test answers misclassified by every seed.

```bash
python app.py --seed-dir <path>/seed44 --share   # all tabs
python app.py                                    # evaluation and error tabs only (reads results/)
```

**REST API** (`src/api.py`, FastAPI):

```bash
MODEL_DIR=<path>/seed44 uvicorn src.api:app --port 8000
curl -X POST localhost:8000/detect -H "Content-Type: application/json" \
     -d '{"question": "Does X reduce Y?", "answer": "X significantly reduces Y."}'
# {"label": "hallucinated", "p_hallucinated": ..., "threshold": 0.87}
```

`GET /health` returns `{"status": "ok"}`. Empty or over-long inputs are rejected with HTTP 422. `tests/test_api.py` tests the HTTP contract with a fake detector.

**Docker** (CPU inference; the model is mounted at runtime, not baked into the image). *The Dockerfile has not been built or run yet; the API itself has been tested with uvicorn.*

```bash
docker build -t medhallu-api .
docker run --rm -p 8000:8000 -v <path>/seed44:/model:ro medhallu-api
```

The trained weights are not distributed with this repository; train a seed with `src/train.py` and run `src/evaluate.py` to produce `threshold.json`.

## Reproduce

```bash
pip install -r requirements.txt   # plus torch
pytest                            # data split tests
python -m src.baselines           # baselines on val
python -m src.train --seed 42     # repeat for 43, 44 (GPU recommended, ~25 min each on a T4)
python -m src.evaluate --run-dir outputs/bio_clinicalbert
```

EDA: `notebooks/01_eda.ipynb`.
