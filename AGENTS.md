# Working in this repo

Read this before you change anything. It applies to people and to coding agents (Claude Code, Codex, Cursor and others).
`CLAUDE.md` points here.

The project predicts which Reddit text posts will reach the top 10% of engagement in their subreddit and quarter.
The model that ships today is a fine-tuned MiniLM on r/dadjokes. A CI/CD pipeline trains it on a university GPU
(NCSU VCL), compares it with the deployed model and ships it.

## Read first, by task

| Task | Read |
|---|---|
| Any change | this file |
| Change the shipped model, `pipeline/`, or `MODEL_NAME` | `.claude/skills/model-release/SKILL.md` |
| Workflows, VCL, secrets, Hugging Face storage | `docs/pipeline.md` |
| Experiments, results, past decisions | `PROJECT_NOTES.md` (newest first), `MODEL_LOG.md` |
| Dataset checks and leakage rules | `validate_dataset.py`, `textonly_data.py` |

## Setup

```sh
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # full: torch, transformers, notebooks
pip install -r requirements-ci.txt     # lint and tests only (what CI installs, no torch)
```

- Put `HF_TOKEN` in `.env`. It is ignored by git. Never commit a token, key or `.vcl_state.json`.
- The dataset is `zachnorton03/textonly-subreddit-predictors` on Hugging Face. Local copies go in `data/` (ignored).
  `textonly_data.SOURCES` reads it from `"local"` or `"hf"`.

## Branches and pull requests

Flow: feature branch → `dev` → `release` (deploys). `main` is the default branch, but day-to-day work targets `dev`.

- Branch from `origin/dev`. Name it `<your name>/<topic>`, for example `owen/deberta-pipeline`.
- Open the PR into `dev`. Never push directly to `dev` or `release`.
- Required on `dev`: `lint-test`. Required on `release`: `trained-candidate`.
- Keep one topic per PR. Put research code and pipeline changes in separate PRs when you can.
- In the PR body, say what changed, how you tested it, and give any metric with its split.
- Before you open the PR, merge `origin/dev` into your branch and run the checks below.

## Checks to run before every push

```sh
ruff check .            # config in ruff.toml: line length 130, notebooks excluded
pytest -q tests         # with requirements-ci.txt installed, as CI does
```

- CI has **no torch, no transformers, no network and no GPU**.
- Import `torch` and `transformers` inside functions. If a test file needs them, start it with
  `pytest.importorskip("torch")` so CI skips it instead of failing.
- Use small synthetic data in tests, like `tests/test_pipeline.py` and `tests/test_tree_models.py`.
- Add a test for each new script or contract that a workflow calls.

## What a merge into `dev` does

A push to `dev` that changes any of these starts a GPU training run (`train-eval`):
- `*.py` at the repo root
- `pipeline/**`
- `docker/train.Dockerfile`
- `requirements.txt`

So a change to a root `.py` file, even a research script, retrains the shipped model. That is expected. Each run
takes the GPU queue for about 12–25 minutes, so don't merge many small root-level changes one at a time.

The run trains with `pipeline/train.py`. It scores the new model and the deployed champion with `pipeline/evaluate.py`
on the same pinned test data. The gate fails if `pr_auc` or `top10_hit_rate` drops more than 0.01 below the
champion's. The candidate is uploaded to `zachnorton03/reddit-post-predictor-models` under `models/<MODEL_NAME>/`.

## Changing the model

Follow `.claude/skills/model-release/SKILL.md`. In short:
1. Reuse or extend the code in a root module (for example `dadjokes_models.py`). Keep the existing defaults so other
   scripts keep working.
2. In `pipeline/train.py`, set `MODEL_NAME` by hand: `<source .py name>-<param count>[-<detail>...]`, lowercase, no
   dates. Example: `dadjokes_models-22M-minilm_l6`. Change it whenever the model changes.
3. Write `type` in `model.json` and add a loader for that type in `pipeline/evaluate.py`'s `LOADERS`.
   **Never remove the loader for the deployed type**, because evaluate also scores the champion.
4. Run a small local subset (`--max-train 5000 --epochs 1`), then a real run on your branch:
   `gh workflow run train-eval.yml --ref <your-branch>`. Link the run in the PR.

The commands and file contracts are in `docs/pipeline.md` under "Contracts the workflows rely on". Don't change
them without updating the workflows, `infra/ansible/train.yml` and that doc in the same PR.

## Data and evaluation rules

These rules keep the results honest. Break them and the numbers mean nothing.

- **Splits are by time:** train before 2024, validation 2024, test January 2025 to February 2026. Never shuffle
  across them.
- **Pick settings, checkpoints, blend weights and features on validation only.** The test split is for the final
  report and the pipeline gate. If you looked at test while choosing, say so.
- **Label:** top 10% of engagement within the subreddit and calendar quarter, where engagement is
  `log1p(max(score, 0)) + log1p(max(num_comments, 0))`.
- **Metrics:** `pr_auc` is average precision on that label. Some notes call it "AP" or "test AP".
  `top10_hit_rate` is the share of real top-10% posts among the 10% the model ranks highest (notes call it
  "top-10% success" or "precision top-10"). Also report `spearman` when you can.
- **Use only what is known at posting time.** These fields leak the outcome: edits (cut bodies at EDIT markers),
  removal or moderator flairs, link flair (moderators change it later) and megathreads. Never add a feature computed
  from posts after the target post.
- **Compare like with like.** Before you call a model better, check that it used the same subreddit, splits,
  training years and test set. Say how many random seeds it used. A gain of about 0.01 can be seed noise.
- **Pin the data.** Pipeline code loads data only through `pipeline/data.py: load_pinned(dataset, revision)`, never
  from a local cache.

## GPUs and compute

- **VCL (default, free):** CI reserves an L40S (48 GB). If no L40S is free within 15 minutes, it uses an
  RTX 2080 Ti (11 GB). Runs queue one at a time (concurrency group `vcl-gpu`).
  - A training job times out after 240 minutes.
  - An L40S trains the current MiniLM in about 6 minutes; a 2080 Ti takes about 15.
- **Modal (fallback, costs money):** `train-eval-modal.yml`, manual only. Use it only when VCL is down or the model
  needs more than VCL offers. Ask the repo owner first.
- **Local:** fine for small subsets. `dadjokes_models.device()` picks CUDA, then Apple MPS, then CPU.
- **Long local runs:** save checkpoints so a crash or sleep doesn't lose the run.

## Recording results

- Add a dated section at the top of `PROJECT_NOTES.md` for each finished experiment. Give the model, data, splits,
  seeds, and validation and test metrics, and say what the result means.
- Put generated tables and plots in `results/<experiment>/`. Keep them small; put large files in `data/`, which
  is ignored.
- Never edit an old result to match a new run. Add the new one.

## Code style

- Python 3.12. Ruff with the rules in `ruff.toml`.
- Keep a module docstring with a one-line summary and the command to run it.
- Prefer small changes to existing modules over new parallel copies. Keep functions' default behavior when other
  scripts call them.
- Notebooks are for exploring and plotting. Code that the pipeline or other scripts need goes in `.py` modules.

## Things agents must not do without asking the repo owner

- Push to `dev`, `release` or `main`, merge a PR, or change branch protection.
- Delete or rewrite someone else's branch. To update it, branch from it and open a new PR.
- Change secrets, GitHub environments or workflow triggers.
- Start paid compute (Modal), or start many VCL runs in a row.
- Promote a model on Hugging Face (`scripts/hf_models.py promote`), or delete anything in the HF repos.
