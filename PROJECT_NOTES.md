# Project notes: EDA and baselines (handoff)

## Update 2026-10-24: random forest and gradient boosting

`textonly_baselines.run_tree_models(data, feats, res)` and notebook section 5 add random forest and histogram
gradient boosting to the pooled and per-subreddit comparison. Both use metadata only (no TF-IDF), with the
same samples, labels, and chronological splits as metadata logistic regression.

- **Defaults:** forest: 300 trees, depth 20; boosting: 200 iterations, learning rate 0.05, 31 leaves, L2 = 1.
  Both use minimum leaf size 50 and the configured seed. Tune `rf_params` / `gb_params` on validation.
- **Time split:** boosting disables automatic early stopping to avoid a random internal validation split.
- **Status:** full-extract benchmark and graphs complete. Test PR-AUC: forest 0.138, boosting 0.143,
  metadata logistic regression 0.125. The notebook includes all nine models; `MODEL_LOG.md` records the
  current results and earlier experiments. Two passing tests cover all seven subreddits and held-out outcome
  isolation. Run: `python -m unittest discover -s tests -v`.

## Update 2026-10-23: text-only subreddits, all years

Post type (image vs. text) dominated the first baselines. The project now uses **text posts from seven
subreddits** so that text and timing carry the signal: AskReddit, Showerthoughts, unpopularopinion, dadjokes,
hypotheticalsituation, sysadmin, TwoSentenceHorror (`text_subreddits.csv` lists the candidates that were considered).

| File | Purpose |
|---|---|
| `extract_subreddits.py` | Streams every Arctic Shift shard (2005-12 to 2026-02, 349 GB), keeps SFW, non-deleted posts of the seven subreddits, writes `data/textonly/data/<subreddit>/<YYYY>/<MM>.parquet`, pushes each finished year to HF `zachnorton03/textonly-subreddit-predictors`. Resumable. |
| `textonly_data.py` | Shared definitions: `connect()` (views `posts`, table `history`), `add_labels()`, local time (America/New_York), edit-note stripping, megathread flag, chart style. |
| `textonly_eda.ipynb` | EDA of the extract. Executed on the full extract (24.1M labeled text posts). |
| `textonly_baselines.py` / `textonly_baselines.ipynb` | Baseline ladder, ablation by feature group, plots. The notebook is a thin harness over the module. |

Decisions:
- **Label:** engagement above the 90th percentile of the post's **subreddit and quarter** (thresholds drift a lot over the years).
- **Split:** train before 2024, validation 2024, test 2025-01 to 2026-02. Capped sample per subreddit (AskReddit has ~100× more posts).
- **Text posts only**; scheduled megathreads (AutoModerator, "Moronic Monday", ...) are dropped: they get many comments automatically.
- **Edit notes leak the outcome** ("EDIT: thanks for the gold"). Bodies are cut at the first edit marker (`textonly_data.EDIT_RE`).
  In a run on 2008–2016 data, posts with an edit marker were 2–3× as likely to be successes.
- **Data gap:** the source has no posts from 2022-09 to 2023-11.

- **Moderator-removed posts** are dropped. Some subreddits flair removed posts ("Removed",
  "Rule 2 – Removed", "Spam") instead of clearing the body. Since 2024 Showerthoughts flairs ~69% of its posts this
  way, so 2025 thresholds sat on the removed posts' score of 1. `reddit_data.DELETED_SQL` now checks the flair
  (112,873 posts dropped; the HF dataset was rewritten).
- **No link flair feature:** the archived flair is the final one and moderators change it after posting
  (Showerthoughts flairs unreviewed posts "Pending"). `has_author_flair` is still used.

### Initial full-extract results (2026-10-24)

These are the earlier run's results. The refreshed nine-model comparison and exact sample counts are in
`MODEL_LOG.md` and the executed `textonly_baselines.ipynb`.

Dataset: 24.1M labeled text posts (AskReddit 20.1M, hypotheticalsituation 68k). Train sample 914k (capped at
150k per subreddit; hypotheticalsituation has only 14k), validation 207k, test 207k. Base rate ~0.10.

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|
| constant | 0.100 | 0.100 | 0.500 |
| subreddit × weekday × hour rate | 0.113 | 0.108 | 0.520 |
| naive bayes (text) | 0.149 | 0.154 | 0.628 |
| logreg (metadata) | 0.124 | 0.125 | 0.565 |
| logreg (text) | 0.166 | 0.179 | 0.658 |
| logreg (metadata + text) | 0.186 | 0.193 | 0.678 |
| MLP, 1 hidden layer (metadata + text) | 0.191 | **0.198** | 0.680 |

Per subreddit (test PR-AUC, best model): sysadmin 0.35–0.37, dadjokes 0.22, AskReddit 0.18, hypotheticalsituation
0.17, TwoSentenceHorror 0.17, Showerthoughts 0.13, unpopularopinion 0.12.

Findings:
- **Timing patterns are strong in the data.** Success peaks 6–9am New York time in every subreddit except
  AskReddit, whose best hours run from about 1am to 7am. The best weekday × hour cells have 1.2× (AskReddit) to 1.8× (dadjokes) the base
  rate. The worst cells, around midnight to 2am (and Sunday afternoon in AskReddit), have 0.6–0.8×. In AskReddit, hours with fewer posts do better
  (Spearman −0.95 over the 24 hours).
- **But timing adds little to the models.** The ablation (logistic regression): subreddit + timing 0.109 test
  PR-AUC vs. 0.100. Adding timing to content *lowers* test PR-AUC (0.202 → 0.191, mostly sysadmin 0.368 → 0.341)
  while validation ROC-AUC rises a little. Timing effects are a few hours wide and multiplicative, so a linear
  model with one-hot hours may not use them well: a tree model with interactions is the next test.
- **Text carries most of the signal** (+text: 0.196 test). The MLP is the first model to beat logistic regression
  (0.198 vs. 0.193), so more capacity helps a little. sysadmin text is strongly predictive (rants, outages,
  ransomware, "your username" threads do well; tech-support questions do poorly).
- **Reposted titles** (`is_repost`, the largest metadata coefficient, +0.71) and author flair (+0.45) do better.
  Not explained yet.
- **Possible score-maturity shift:** median score drops to 1 from 2023 on in AskReddit, Showerthoughts and
  unpopularopinion (unpopularopinion: 10 in 2022, 1 in 2024), while median comments rise in some subreddits.
  Arctic Shift may capture recent posts before their scores settle. If so, validation and test labels mean
  something different from train labels. Check this before the larger models.

### One model per subreddit, and regression targets (2026-10-24)

Direction: no general model. Build the simplest model that works for one subreddit and answers "will this post
do well?". The per-subreddit and loss runs were scratch experiments, not in the repo code.

Logistic regression trained on one subreddit only (TF-IDF + timing + structure + history; C tuned on validation;
train capped at 300k):

| Subreddit | Test PR-AUC (pooled → own model) | Top-10% hit rate |
|---|---|---|
| sysadmin | 0.35 → 0.39 | 42% |
| dadjokes | 0.22 → 0.24 | 28% |
| hypotheticalsituation | 0.17 → 0.24 | 27% |
| AskReddit | 0.18 → 0.20 | 24% |
| TwoSentenceHorror | 0.17 → 0.20 | 24% |
| Showerthoughts | 0.13 → 0.15 | 19% |
| unpopularopinion | 0.12 → 0.13 | 15% |

Chosen subreddit: **dadjokes** (stable median score 7–8 in every year, strongest remaining timing effect, 266k
training posts). sysadmin scores higher but mostly by topic (rants and outages vs. help questions).

Timing in dadjokes: the 6–9am (New York) lift fell from 1.81× (≤2018) to 1.54× (2019–23) to 1.29× (2024–26).
Timing alone: 0.118 test PR-AUC. Adding timing to content adds ~0 (0.241 vs. 0.239), also when training only on 2019+.

Targets and losses on dadjokes (same features; regressions tuned on validation Spearman):

| Target and loss | Spearman vs. percentile | PR-AUC (top 10%) | Top-10% hit rate |
|---|---|---|---|
| top-10% yes/no, logistic | 0.262 | 0.238 | 0.279 |
| log engagement, ridge (squared error + L2) | 0.313 | 0.234 | 0.282 |
| percentile in quarter, ridge | 0.326 | 0.201 | 0.248 |
| log engagement, Huber (SGD, epsilon 1) | 0.226 | 0.158 | 0.188 |
| raw score, Poisson (did not converge) | 0.194 | 0.227 | 0.269 |
| pairwise ranking (logistic on 600k same-quarter pairs) | 0.314 | 0.198 | 0.243 |

Ridge on log engagement finds the top as well as the classifier and ranks all posts better, so it is the proposed
main model, evaluated on top-10%. The loss is not the bottleneck.

### Larger models on dadjokes (2026-10-24)

Code: `dadjokes_models.py` (data, features, models, evaluation); `run_dadjokes_models.py` reproduces the table.
Target: log engagement. Train 266,053 posts (2011–2023), validation 29,760 (2024), test 32,826 (2025-01..2026-02).
Settings chosen on validation. Caches (Parquet, embeddings, fine-tune predictions) in `data/models/`.

| Model | Test Spearman | Test PR-AUC | Top-10% hit rate |
|---|---|---|---|
| ridge, TF-IDF + metadata (reference) | 0.313 | 0.234 | 0.283 |
| ridge, TF-IDF only | 0.277 | 0.212 | 0.252 |
| gradient boosting (sklearn HGB), metadata only | 0.185 | 0.164 | 0.200 |
| HGB, metadata + out-of-fold TF-IDF ridge (stacked) | 0.306 | 0.230 | 0.275 |
| ridge, MiniLM embeddings only (frozen, all-MiniLM-L6-v2) | 0.210 | 0.169 | 0.201 |
| ridge, embeddings + TF-IDF + metadata | 0.338 | 0.240 | 0.281 |
| HGB, metadata + TF-IDF and embedding ridges (stacked) | 0.322 | 0.237 | 0.281 |
| **fine-tuned MiniLM, text only** | 0.356 | 0.266 | 0.314 |
| **blend: fine-tuned MiniLM + HGB metadata** (weights 0.66 / 0.34, fit on validation) | 0.367 | **0.274** | **0.318** |
| blend: fine-tuned MiniLM + ridge TF-IDF + metadata | 0.374 | 0.272 | 0.312 |

Fine-tuning: all-MiniLM-L6-v2 (22M parameters) with mean pooling and a linear head, squared error on the
standardized target, AdamW lr 5e-5 with one-cycle schedule, batch 128, max 96 tokens, 3 epochs (~7.5 min each on
an M3 Max, MPS). Validation Spearman by epoch: 0.340, 0.359, 0.359.

Findings:
- **Fine-tuning is the first clear gain over linear models** (top-10% hit 0.283 → 0.314 with text only). Frozen
  embeddings lose to TF-IDF, so embeddings help only when they are trained on the task.
- Gradient boosting finds little beyond the linear model: no strong weekday × hour interaction.
- Metadata (timing, structure, history) adds a little on top of the fine-tuned text model (0.314 → 0.318).
- The best model's top-ranked 10% of posts are successes 32% of the time (3.2× chance); timing alone gives 1.2×.

Next: a larger encoder (for example DeBERTa-v3-base, ~1–1.5 h on the M3 Max), a notebook harness that scores
a new joke at a chosen posting time, and the score-maturity check for 2023 and later. `add_labels` now also gives `metric_pct`
(percentile in subreddit-quarter), and `load_data` returns `metric`, `metric_pct`, `score`, `num_comments`.

### d1 zero-shot on dadjokes (2026-10-24)

`run_d1_zero_shot.py` scores the 32,826 dadjokes test posts with Liquid AI's d1 decision model
(`LiquidAI/d1-omni-600M`, 587M parameters, no fine-tuning). d1 returns P(yes) for a yes/no question about a text;
that probability is the ranking score. Same test set and metrics as the table above. ~40 min on an M1 Max (MPS, fp16).

| Question | Test Spearman | Test PR-AUC (95% bootstrap CI) | Top-10% hit rate |
|---|---|---|---|
| "This is a post on r/dadjokes. Will it be in the top 10% of posts by upvotes and comments?" | 0.043 | 0.121 (0.115–0.128) | 0.134 |
| "Is this joke funny?" | 0.007 | 0.106 (0.102–0.113) | 0.113 |
| constant (base rate) | – | 0.100 | 0.100 |

- **Zero-shot d1 is close to chance** and far below TF-IDF ridge (0.234) and the fine-tuned MiniLM (0.266). A general
  model's sense of "funny" or "will do well" does not match what this community rewards; the top-10% question is a
  little better than "funny" (the two scores correlate 0.57).
- Fine-tuning d1 would test whether a 25× larger encoder helps, but the fine-tuned MiniLM shows that training on the
  task is what matters. Not tried: ~5–7 h per epoch on this machine.

### Four more subreddits (2026-10-24)

`extract_subreddits.py --subreddits wallstreetbets singularity artificial LocalLLaMA --repo jbduran/textonly-subreddit-predictors --push`
extracts r/wallstreetbets, r/singularity, r/artificial and r/LocalLLaMA with the same filters into the HF dataset
`jbduran/textonly-subreddit-predictors` (separate from the original seven). Months are tracked per set of subreddits
(`.done_months_<subreddits>`), and a push uploads and deletes only the run's subreddits. These subreddits are
mostly link and image posts; the labels keep their text posts. LocalLLaMA was created in 2023-03, inside the data
gap, so it has posts only from 2023-12 on and needs a later train/validation/test split.

Extract finished 2026-10-25 (all posts / approximate text posts): wallstreetbets 939k / 230k, singularity 74k / 24k,
LocalLLaMA 67k / 50k, artificial 55k / 17k.

Baselines from `dadjokes_models.py` (`run_new_subreddits.py`; labeled text posts, default splits; LocalLLaMA: train
before 2025, validation 2025-01..06, test 2025-07..2026-02). Base rate 0.10 everywhere.

| Subreddit | Train posts | Test PR-AUC: ridge (TF-IDF + meta) | HGB (meta only) | Top-10% hit (ridge) |
|---|---|---|---|---|
| LocalLLaMA | 19,802 | **0.315** | 0.200 | 0.344 |
| wallstreetbets | 162,563 | 0.215 | **0.221** | 0.249 |
| artificial | 9,839 | 0.159 | 0.137 | 0.210 |
| singularity | 3,955 | 0.155 | 0.122 | 0.184 |

- **LocalLLaMA has the strongest text signal measured so far** (ridge 0.315 vs. 0.234 on dadjokes, with 13× less
  training data). Its whole history is 2023-12 on, so the possible score-maturity shift (see above) needs checking first.
- **wallstreetbets is the only subreddit where metadata alone beats text**; not investigated yet.
- singularity and artificial have few labeled text posts (quarters with < 100 text posts are dropped) and weak signal.

---

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
