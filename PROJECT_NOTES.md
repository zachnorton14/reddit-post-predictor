# Project notes: EDA and baselines (handoff)

Status as of 2026-10-06. This file summarizes the work in `eda.ipynb` and `baselines.ipynb`: what was examined,
what was found, what was decided, and what was built and tested. All numbers come from executed runs of the
notebooks. The notebooks in the repo contain those outputs.

## 1. Project context

- CSC422 group project "Predicting Social Media Post Success" (proposal: `CS422 - Predicting Social Media Post Success Proposal.docx`).
- Goal: predict whether a Reddit submission performs unusually well, from information known at posting time
  (time, subreddit, form of content, text). Find which factors matter, especially how much timing adds.
- The proposal also plans: random forest and gradient boosting (week 8, 10/26), tuning (week 9), feature importance,
  topic categories, geographic subreddits (r/NYC, r/Seattle, r/Chicago) with local-time features, and
  engagement heatmaps by local hour and weekday. **None of these are built yet.**

## 2. Data source

- Hugging Face dataset `open-index/arctic`: Arctic Shift Reddit dumps as Parquet, 2005-12 to 2026-02, 1.3 TB
  (submissions 326.6 GB, ~2.8B rows). Only submissions are used. Comments are not used.
- Layout: `data/submissions/YYYY/MM/NNN.parquet`. 2025 has 73–89 shards per month (1,014 shards, ~69 GB).
  Each shard is ~70 MB and 500k rows.
- **Shards are in time order.** One shard is one contiguous block of ~10 hours. Taking the first N shards of a
  month gives only the first days (an early 6-shard sample of 2025-12 had no Tuesday posts). Always take
  shards spread evenly across the month.
- Columns: `id, author, subreddit, title, selftext, score, created_utc, created_at, title_length,
  num_comments, url, over_18, link_flair_text, author_flair_text`. Nulls occur only in the flair columns
  (`author_flair_text` 91%, `link_flair_text` 52%).
- **No images or media.** The Arctic Shift `thumbnail`, `preview`, and `media_metadata` fields are not in this
  Parquet version. For an image post, only the link in `url` exists. A check of 39 random `i.redd.it` URLs
  (2025-12) found 28 (72%) still served and 11 returned 404. Gallery posts link to `reddit.com/gallery/<id>`,
  which would need the Reddit API.
- `score` and `num_comments` are values at archival time. A check by month (section 4) found no sign that
  late-2025 scores are less mature.

## 3. Files

| File | Purpose |
|---|---|
| `reddit_data.py` | Shared loader: `select_shards(year, months, shards_per_month)`, `download(shards, data_dir)` (parallel, skips existing files), `connect(files)` → DuckDB connection with views `raw_subs` (all posts) and `subs` (filtered). Defines `NSFW_SQL`, `DELETED_SQL`, `POST_TYPE_SQL`. |
| `eda.ipynb` | EDA of 2025 submissions. Executed with outputs. **It does not import `reddit_data.py` yet.** It has its own copy of the filter and post-type SQL, which must stay in sync. |
| `baselines.ipynb` | Eight baseline models from a constant to a one-hidden-layer MLP. Executed with outputs. Imports `reddit_data`, so start Jupyter from the repo root. |
| `requirements.txt` | duckdb, huggingface_hub, ipykernel, ipywidgets, matplotlib≥3.10, numpy, pandas, pyarrow, scikit-learn, scipy. |
| `.gitignore` | Ignores `data/` (downloaded shards go to `data/arctic/`), `.venv/`, checkpoints. |

Notes:
- Data was downloaded and the notebooks were executed in a separate scratch environment. The repo has no
  `data/` folder. The first run from the repo downloads ~9.3 GB (~10 min). Set `SHARDS_PER_MONTH` lower for a
  quick test.
- Run times on the development machine (14 cores, 38 GB RAM): `eda.ipynb` ~11 min including download;
  `baselines.ipynb` ~4 min with data present.
- Git: one commit before this work (the proposal) plus the user's commit of an early `eda.ipynb`.
  Current changes to `eda.ipynb`, `requirements.txt`, and the new `baselines.ipynb`, `reddit_data.py`,
  and this file are **not committed**.

## 4. EDA (`eda.ipynb`)

### Sample and filters

- Year 2025, 12 shards per month spread evenly (144 shards, 9.3 GB), covering 188 days.
- Each month gets the same number of shards, so **post counts per month, weekday, or hour in the sample show
  where the shards fall, not real Reddit volume**. Only rates (for example, the share of posts with score ≥ 10)
  are valid across groups.
- Summary tables run in DuckDB on all filtered posts. Plots that need individual rows in pandas use a fixed
  2M-post reservoir sample (`subs_sample`, `REPEATABLE (42)`).

| Step | Posts |
|---|---|
| Raw | 69,261,822 |
| NSFW (`over_18`), excluded | 29,444,333 (42.5%) |
| SFW but deleted/removed, excluded | 4,227,885 (10.6% of SFW) |
| **Kept** | **35,589,604** (51.4%): 842,838 subreddits, 11.2M authors |

- **Decided by the user:** exclude NSFW posts. Exclude deleted/removed posts: `author = '[deleted]'`, or
  `selftext` in (`[deleted]`, `[removed]`), or `title` in (`[deleted by user]`, `[ Removed by moderator ]`).
- **Posts with an empty `url` are deleted posts, not a separate post type.** All of them (100%) have author
  `[deleted]`, and 99.3% have selftext `[deleted]`/`[removed]`. Reddit clears author, body, and URL on
  deletion but keeps title and score. Some had high scores before deletion (max seen 2,372). The filter
  removes all of them.
- No duplicate `id`s were found.

### Findings (2025, filtered)

- **Targets are very heavy-tailed.** Score quantiles: 25% = 1, median = 2, 75% = 11, 90% = 65, 99% = 970,
  99.9% = 6,650, max 242,068 (mean 62.6). `num_comments`: median 2, 90% = 22, 99% = 135, max 52,946.
  Spearman correlation between score and comments: 0.45.
- **Subreddits:** volume is spread out (top 10 subreddits = 4.6% of posts, top 1,000 = 29.6%, top 10,000 = 68.3%).
  The busiest subreddits are mostly bots and games (`chessquiz`, `FarmMergeValley`, `Pixelary`,
  `PokemonGoRaids`, `AutoNewspaper`) with a median score of 1. Raw score is not comparable across
  subreddits, so labels must be relative to the subreddit.
- **Post type** (from `url`: self, image, gallery, video, external link) is the strongest simple signal:

  | Type | Share | Median score | Score ≥ 10 |
  |---|---|---|---|
  | self | 45.3% | 1 | 12.9% |
  | image | 24.9% | 6 | 44.7% |
  | gallery | 11.9% | 5 | 40.8% |
  | external link | 9.1% | 1 | 13.9% |
  | video | 8.8% | 2 | 34.5% |

- **Time (UTC) has a weak effect.** By weekday, the share with score ≥ 10 ranges only from 25.6% (Mon) to 27.1% (Sun).
  By month it is 25–27%, with a dip in October (23.7%). Comments show the same October dip, so it looks seasonal,
  not an archiving artifact. December (26.2%) ≈ January (26.6%), so archival-time scores look mature enough.
  Note: the proposal wants **local** time for geographic subreddits. Everything so far is UTC.
- **Text features** (early 2025-12 run that still **included NSFW and deleted posts**; the current notebook has
  the full-year filtered numbers, which were not reviewed): titles with digits
  (13% vs 27% reach score ≥ 10) and all-caps titles (8.5% vs 23%) do worse. Questions are slightly worse.
  Very short (≤ 12 chars) and very long (> 88 chars) titles do worse than mid-length ones.
- **Flair:** posts with a link flair do slightly better (26% vs 20% reach score ≥ 10; 2025-12 SFW run before the
  deleted-post filter).
- **Authors:** median 1 post per author. The heaviest posters are bots (`AutoModerator`, `chess-quiz-plus`,
  `bot-bouncer`, `AutoNewspaperAdmin`). Bots are not filtered yet.
- Candidate label base rates (2025): score ≥ 10 = 26.1%, score ≥ 100 = 7.6%, comments ≥ 10 = 20.9%,
  top 10% by score within the subreddit = 8.7%.

## 5. Baselines (`baselines.ipynb`)

### Design decisions

| Decision | Choice | Who decided |
|---|---|---|
| Label metric | `engagement = ln(1 + max(score, 0)) + ln(1 + num_comments)` | User approved; matches proposal ("top 10% of engagement") |
| Success | engagement > the subreddit's 90th percentile (`SUCCESS_QUANTILE = 0.9`), threshold from training months only | Claude default; proposal says top 10% |
| Minimum subreddit size | ≥ 50 training posts (`MIN_SUB_POSTS`); posts in other subreddits are dropped | Claude default, open |
| Split | by time: train Jan–Sep, validation Oct, test Nov–Dec 2025 | Proposal requires a time split; months chosen by Claude |
| Sample | 1M train, 200k val, 200k test (reservoir, seed 42) | Claude default |
| Main metric | PR-AUC (average precision). Also ROC-AUC and log loss | Claude default; the proposal lists accuracy/precision/recall/F1/ROC-AUC, so threshold metrics are not reported yet |
| Features | Only information known at posting time. `score` and `num_comments` are never features | Required (leakage) |

- `LABEL_METRIC` switches the label between `"engagement"` (default), `"score"`, and `"comments"`. All metrics
  are log scale, so the subreddit threshold and median are used directly as features.
- **Why engagement and not score:** score and comments measure different kinds of success (upvoted vs. discussed).
  Overlap of top-10%-in-subreddit labels (training months; share of row-label successes that are also column-label successes):

  | Successful by ↓ / also by → | score | comments | engagement |
  |---|---|---|---|
  | score | 100% | 50% | 80% |
  | comments | 51% | 100% | 73% |
  | engagement | 75% | 68% | 100% |

- Label coverage (share of posts whose subreddit has a threshold): train 90.6%, val 87.7%, test 83.2%.
  Validation and test lose more because new or small subreddits have no training threshold.
- Positive rate (engagement label): train 9.2%, val 8.7%, test 9.6%. 45,903 subreddits have a threshold.

### Features

- **Metadata (49 columns):** standardized numeric features `log_title_length`, `title_words`, `log_body_length`,
  `sub_log_posts`, `sub_threshold`, `sub_median`, `sub_rate`, `sub_type_rate`; binary features
  `title_has_question`, `title_has_number`, `title_all_caps`, `has_link_flair`, `has_author_flair`; one-hot
  `post_type`, `hour` (UTC), `dow` with the first category dropped (gallery, hour 0, Monday).
- Subreddit statistics (`sub_*`) come from **all sampled training-month posts** (not only the 1M sample).
  `sub_type_rate` is the subreddit × post type success rate, smoothed toward the subreddit rate with 20 pseudo-posts.
  Minor leak: training rows contribute to their own subreddit's rate (each subreddit has ≥ 50 posts, from 24.5M
  training posts in total, so the effect is small).
- **Text:** TF-IDF over title + first 1,000 characters of selftext; word 1–2-grams, `min_df=5`,
  `max_features=50,000`, `sublinear_tf`, fit on train only.

### Models and results

Settings: Naive Bayes `MultinomialNB(alpha=0.1)`. Logistic regression `C=1`, lbfgs (not tuned). The MLP is
`MLPClassifier` with 64 ReLU units, adam, `alpha=1e-4`, batch 1024, lr 1e-3, trained with `partial_fit`
one epoch at a time. Early stopping on **validation PR-AUC**, patience 2, maximum 8 epochs.

| # | Model | Engagement label: val PR-AUC | test PR-AUC | test ROC-AUC | Score-only label: val PR-AUC | test PR-AUC |
|---|---|---|---|---|---|---|
| 0 | Constant (base rate) | 0.087 | 0.096 | 0.500 | 0.085 | 0.094 |
| 1 | Subreddit rate | 0.104 | 0.108 | 0.538 | 0.104 | 0.108 |
| 2 | Subreddit × post type rate | 0.163 | 0.169 | 0.667 | 0.182 | 0.184 |
| 3 | Naive Bayes (text) | 0.134 | 0.139 | 0.626 | 0.140 | 0.144 |
| 4 | Logistic regression (metadata) | 0.166 | 0.174 | 0.674 | 0.185 | 0.188 |
| 5 | Logistic regression (text) | 0.142 | 0.149 | 0.636 | 0.151 | 0.155 |
| 6 | Logistic regression (metadata + text) | 0.186 | 0.199 | 0.703 | 0.210 | 0.216 |
| 7 | MLP, one hidden layer (metadata + text) | **0.193** | **0.201** | **0.711** | **0.219** | **0.227** |

Interpretation:
- The ranking is the same for both labels. Validation and test agree, so the time split looks stable.
- The subreddit × post type lookup (model 2) almost matches logistic regression on all metadata (model 4).
  Post type does most of the work.
- Text is weak alone but adds about 0.02 PR-AUC on top of metadata.
- The MLP beats logistic regression by less than 0.01. Its best validation PR-AUC is at epoch 1, and it gets
  worse after that (overfitting), so it needs more regularization or fewer features.
- The engagement label is about 0.02 harder than the score-only label, mostly because post type predicts upvotes
  better than comments.
- Logistic regression coefficients (engagement label): the strongest positive features are `sub_type_rate`,
  `sub_rate`, image/gallery/video post types, and `has_author_flair`. Hours 2–8 UTC are the most negative hours.
  Negative text terms include "has anyone", "question", "recommendations", "headline", "top stories", and bot
  boilerplate ("new asteroid discovered", "opinionwarsdev"). Positive terms include "last night", "giveaway",
  "match thread", "personally", "regret".

## 6. Open questions and next steps

1. **Commit** the current work (nothing since the early `eda.ipynb` is committed).
2. Switch `eda.ipynb` to `reddit_data.py` so the filter rules exist in one place.
3. Open design questions: threshold per subreddit **and time period** (the proposal says "within each subreddit
   and time period"), minimum subreddit size, filtering bot accounts, author-history features (need a
   time-aware computation to avoid leaks).
4. Proposal items not started: random forest and gradient boosting (add them to the same comparison table),
   geographic subreddits with local-time features, topic categories, heatmaps of engagement by local
   hour × weekday, feature importance, and how much timing adds beyond content and community.
5. Tuning (logistic regression `C`, TF-IDF size, MLP units and regularization) on validation only. Look at
   test once, at the end. Report the proposal's threshold metrics (precision, recall, F1) at a threshold chosen on validation.
