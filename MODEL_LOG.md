# Model experiment log

Updated 2026-10-24 after the full text-only benchmark and graph regeneration.

## Current implemented models

All nine entries below were executed in `textonly_baselines.ipynb`. Reusable training and evaluation
code is in `textonly_baselines.py`; data, labels, and historical features are shared through `textonly_data.py`.
The two lookup baselines produce predictions without fitting a classifier.

### Shared experiment setup

- **Task:** classify whether engagement is above the 90th percentile within a subreddit and calendar quarter.
  Engagement is `log1p(max(score, 0)) + log1p(max(num_comments, 0))`; quarters need at least 100 posts.
- **Data:** text posts from AskReddit, Showerthoughts, unpopularopinion, dadjokes,
  hypotheticalsituation, sysadmin, and TwoSentenceHorror. Deleted, removed, NSFW, and scheduled megathread
  posts are filtered; bodies are cut at edit markers to reduce outcome leakage.
- **Chronological split:** train before 2024; validation during 2024; test January 2025 through February 2026.
  Samples contain **914,038 / 207,133 / 188,947** posts, respectively. Caps are 150,000 training and
  30,000 validation/test posts per subreddit; seed is 42. Success prevalence is about 10%.
- **Metadata:** subreddit, local hour/weekday/month, previous-hour/day posting volume, smoothed training
  success rates by posting time, title/body structure, author flair, author posting history, and repeated titles.
  The current metadata matrix has 61 columns. Link flair is excluded because moderators can change it later.
- **Text:** TF-IDF word unigrams/bigrams from titles and the first 1,000 body characters; at most 50,000
  features, minimum document frequency 5. Vocabulary and feature transformations are fitted on training data.
- **Evaluation:** pooled and per-subreddit average precision, ROC-AUC, and log loss on validation and test.
  The code and graphs call average precision **PR-AUC**. Validation is used for model choices.

### Completed main comparison

| ID | Model | Inputs | Val PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---|---|---:|---:|---:|
| 0 | Constant training success rate | None | 0.0997 | 0.0996 | 0.5000 |
| 1 | Subreddit x weekday x hour success rate | Training rate lookup | 0.1132 | 0.1076 | 0.5204 |
| 2 | Multinomial Naive Bayes | Text | 0.1485 | 0.1535 | 0.6281 |
| 3 | Logistic regression | Metadata | 0.1244 | 0.1247 | 0.5650 |
| 4 | Logistic regression | Text | 0.1692 | 0.1789 | 0.6619 |
| 5 | Logistic regression | Metadata + text | 0.1829 | 0.1902 | 0.6758 |
| 6 | One-hidden-layer MLP | Metadata + text | 0.1907 | 0.1966 | 0.6800 |
| 7 | Random Forest classifier | Metadata | 0.1397 | 0.1380 | 0.5952 |
| 8 | Histogram gradient boosting classifier | Metadata | 0.1446 | 0.1434 | 0.6077 |

Settings used:

- **Naive Bayes:** smoothing `alpha=0.1`.
- **Logistic regression:** L2 regularization, `C=1`, `lbfgs`, maximum 2,000 iterations.
- **MLP:** 64 ReLU hidden units, Adam, L2 `alpha=0.0001`, learning rate 0.001, batch size 1,024.
  Maximum 8 epochs, patience 2; the best validation PR-AUC checkpoint was epoch 2, and training stopped at epoch 4.
- **Random Forest:** 300 trees, maximum depth 20, minimum leaf size 50, seed 42.
- **Gradient boosting:** 200 iterations, learning rate 0.05, maximum 31 leaves, minimum leaf size 50,
  L2 regularization 1, seed 42. Automatic early stopping is disabled to preserve the chronological validation setup.

The trees use the same metadata and sampled posts as model 3. They improve pooled metadata-only PR-AUC,
but text-based models remain stronger in the main comparison. The MLP has the highest validation and test
PR-AUC in this nine-model comparison; the content-only ablation below has a higher observed test PR-AUC.
These are initial tree settings, with hyperparameter tuning still to be done.

## Completed feature ablations

These are additional logistic-regression fits, using the same split and `C=1`. Each row includes subreddit.
The all-feature ablation concatenates feature groups in a different order from model 5; their fitted
metrics differ slightly, so both executed results are retained.

| Feature groups beyond subreddit | Val PR-AUC | Test PR-AUC |
|---|---:|---:|
| None | 0.0998 | 0.0998 |
| Timing | 0.1145 | 0.1087 |
| Structure | 0.1087 | 0.1160 |
| History | 0.1179 | 0.1182 |
| Text | 0.1802 | 0.1951 |
| Structure + history + text | 0.1862 | 0.2020 |
| Structure + history + text + timing | 0.1840 | 0.1910 |

Text supplies most of the predictive signal. Timing alone improves only slightly over the baseline;
adding timing to the content features lowers test PR-AUC in this run. This is an observed comparison,
not evidence of a causal effect of posting time.

## Earlier implemented baseline experiment

`baselines.ipynb` contains the earlier broad-Reddit 2025 experiment, which was not rerun for this update.
It uses different data, labels, and splits, so its scores are not directly comparable to the text-only run.
Its eight models are the constant baseline, subreddit-rate lookup, subreddit x post-type lookup,
text Naive Bayes, three logistic-regression feature variants, and a one-hidden-layer MLP.
The subreddit-only and subreddit x post-type lookups are additional baseline variants beyond the current ladder.
The archived combined logistic regression and MLP achieved test PR-AUC of approximately 0.1988 and 0.2012.

## Documented scratch experiments

The following experiments are recorded in `PROJECT_NOTES.md`. Their reusable training scripts and
saved estimators are not in the repository, and they were not rerun with the new graphs.

**One logistic-regression model per subreddit:** TF-IDF + timing + structure + history, training capped
at 300,000 posts per subreddit, `C` selected on validation. Documented test PR-AUC: sysadmin 0.39,
dadjokes 0.24, hypotheticalsituation 0.24, AskReddit 0.20, TwoSentenceHorror 0.20,
Showerthoughts 0.15, and unpopularopinion 0.13.

**Alternative targets/losses on dadjokes:** regressions were tuned on validation Spearman correlation.

| Target / model | Test Spearman vs. percentile | Test top-10% PR-AUC | Top-10% hit rate |
|---|---:|---:|---:|
| Binary top-10% logistic regression | 0.262 | 0.238 | 0.279 |
| Log engagement / ridge regression | 0.313 | 0.234 | 0.282 |
| Within-quarter percentile / ridge regression | 0.326 | 0.201 | 0.248 |
| Log engagement / Huber SGD | 0.226 | 0.158 | 0.188 |
| Raw score / Poisson regression, **did not converge** | 0.194 | 0.227 | 0.269 |
| Pairwise logistic ranking, 600,000 same-quarter pairs | 0.314 | 0.198 | 0.243 |

Ridge on log engagement is the proposed single-subreddit direction in the notes; the reusable pooled
benchmark currently implements classifiers. The Poisson attempt is an incomplete fit.

## Outputs and verification

- Refreshed results and graphs are saved in `textonly_baselines.ipynb`, including all nine models and
  the seven feature ablations. The notebook stores 21 executed code cells without error outputs.
- Standalone graphs, exact metric tables, predictions, sample counts, and a run manifest are in
  `C:/Users/owens/.codex/visualizations/2026/10/24/01a1700a-7682-7d31-bd12-c4867bd27063/reddit-model-comparison/`.
  Main graph files: `model_pr_auc.png`, `per_subreddit_pr_auc.png`, and `metadata_precision_recall.png`.
- Dataset revision: `e02c2b6236cbc51df27b4faf991fec836ed7fe08` of
  `zachnorton03/textonly-subreddit-predictors`; Python 3.12.14, DuckDB 2.0.0, scikit-learn 1.9.1.
- For this execution, historical posting counts used an equivalent cumulative-count/ASOF query to
  reduce runtime. Its output matched the original query on all 376,299 sysadmin rows and 70 boundary cases;
  the source feature definitions were unchanged. This runtime substitution is recorded in the run manifest.
- `tests/test_tree_models.py`: two passing integration tests cover all seven subreddits and check that
  changing held-out outcomes cannot change fitted tree predictions or input features. All six added
  notebook code cells also passed a synthetic-data execution before the full benchmark.

Remaining model work: tune tree parameters using validation, examine feature importance and errors,
and turn the selected single-subreddit regression experiments into reproducible training code.
