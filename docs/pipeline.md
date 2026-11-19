# CI/CD pipeline

Branch flow: feature branch → `dev` → `release` → automatic deploy. A model is trained once and then only scored.

1. **Experiment, on a feature branch:** push `experiments/<name>.yaml`. `experiment` trains each listed run on a
   VCL GPU, scores it on validation, stores the weights on Hugging Face and posts a table on the PR.
2. **Choose:** name the chosen run's folder in `candidate.yaml` (choose on validation, never on test).
3. **Pull request into `dev`:** CI (Ruff, pytest) must pass. Dataset validation runs when dataset code changes.
   Merging uses no GPU and trains nothing.
4. **Pull request `dev` → `release`:** `release-check` downloads the `candidate.yaml` folder at a pinned Hugging Face
   revision and scores it and the deployed champion on the same test data. It blocks the merge if the candidate is
   worse. Nothing is trained.
5. **Push to `release`** (the merge): `deploy` ships exactly the revision the check scored and makes it the
   champion.

`train-eval` (retrain `pipeline/train.py`'s model) is now manual only. How to name, change and ship a model:
`.claude/skills/model-release/SKILL.md`.

| Workflow | Trigger | Runner | What it does |
|---|---|---|---|
| `ci.yml` | pull request into `dev`; manual | GitHub-hosted | Ruff, pytest. Required on `dev`. |
| `data-validation.yml` | pull request into `dev` that changes dataset code; Hugging Face webhook relay; manual | GitHub-hosted | validates the dataset |
| `train-eval.yml` | manual only (`smoke: true` only checks VCL access) | hosted + self-hosted controller | retrains `pipeline/train.py`'s model on a VCL GPU, scores it and the champion, uploads it to `models/<name>/`, gates (advisory) |
| `release-check.yml` | pull request from `dev` into `release` (`pull_request_target`) | hosted + self-hosted controller | scores the `candidate.yaml` model and the champion on test on a VCL GPU, no training; uploads `release-<head sha>` on a pass. Job `trained-candidate` is required on `release`. |
| `train-eval-modal.yml` | manual only | GitHub-hosted | fallback: trains and evaluates on a Modal GPU, gates (see "Modal fallback") |
| `experiment.yml` | push to a feature branch that changes `experiments/*.yaml`; manual (`-f config=...`) | hosted + self-hosted controller | runs every listed research run on a VCL L40S, validation only; stores each on HF; posts a table on the PR (see "Experiments"). Advisory, never blocks. |
| `vcl-cleanup.yml` | every 30 minutes; manual | self-hosted controller | ends reservations that a pipeline job recorded but couldn't end (`scripts/vcl_ledger.py`) |
| `deploy.yml` | push to `release` | GitHub-hosted | downloads the weights at the revision `release-check` scored, builds the inference image, deploys, publishes release `model-<n>`, promotes the champion |

## Runners

- **GitHub-hosted** (`ubuntu-latest`) for everything except training.
- **One self-hosted controller** (`csc519-201-host`, labels `self-hosted, Linux, X64`). It has no GPU. In the
  `train-eval` job it reserves an L40S instance with `python -m vcl up gpu-train-large`. If no L40S is ready within
  15 minutes (VCL has 2), it falls back to an RTX 2080 Ti (`gpu-train`) and `run.json` records which (`vcl_profile`). It runs
  `infra/ansible/train.yml` against it, copies the results back, and always runs `python -m vcl down`.
  VCL lets only the IP that requested the connection reach the instance, so all of this stays in one job on
  the controller.
- The controller needs `git` and outbound SSH. The jobs get Python from `actions/setup-python`. The job installs `ansible-core`,
  `pyyaml` and `huggingface_hub` into `.venv` in its workspace.
- The repo is public. Self-hosted jobs run only on pushes by people with write access, manual runs, schedules
  and `release-check`. `release-check` uses `pull_request_target`: GitHub runs the workflow file from `release`,
  not from the pull request, and the jobs run only when the pull request comes from this repo's `dev`. So fork
  code never reaches the controller. Also set Settings → Actions → General → "Require approval for all outside
  collaborators".
- `pull_request_target` uses the copy of `release-check.yml` on `release`. A change to it takes effect after it is
  merged into `release`. The first time, `release` doesn't have it yet: the repo owner merges that first
  `dev` → `release` pull request without the check (or adds the file to `release` first).

## Secrets

Stored in GitHub, never in the repo. Locally, `HF_TOKEN` goes in `.env` (ignored by git).

| Secret | Where | Used by | Notes |
|---|---|---|---|
| `GITHUB_TOKEN` | automatic | all | GHCR push and pull, releases, artifact download. Nothing to set. |
| `HF_TOKEN` | repository | data validation, train-eval, train-eval-modal, deploy | Hugging Face read/write token: reads the dataset, writes `zachnorton03/reddit-post-predictor-models`. Must be a repository secret (not an environment secret): the jobs that use it run in other environments or none. |
| `VCL_TOKEN` | environment `vcl` | train-eval | VCL API token (vcl.ncsu.edu → Manage → User Preferences → Manage Tokens) |
| `VCL_SSH_KEY` | environment `vcl` | train-eval | private key Ansible uses to reach the GPU instance |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | environment `modal` | train-eval-modal | Modal API token (`modal token new`). Only needed if the fallback is used. |
| `PROD_SSH_KEY`, `KUBECONFIG_DATA` | environment `production` | deploy | placeholders for the production deployment |

Environments: `vcl` (no reviewers, so runs are not blocked) and `production` (deployment branch: `release`
only; optionally a required reviewer).

One-time setup (repo admin):

```sh
gh api -X PUT repos/zachnorton14/reddit-post-predictor/environments/vcl
gh api -X PUT repos/zachnorton14/reddit-post-predictor/environments/production \
  -F 'deployment_branch_policy[protected_branches]=false' -F 'deployment_branch_policy[custom_branch_policies]=true'
gh api -X POST repos/zachnorton14/reddit-post-predictor/environments/production/deployment-branch-policies -f name=release

gh secret set HF_TOKEN                        # prompts for the value
gh secret set VCL_TOKEN --env vcl
gh secret set VCL_SSH_KEY --env vcl < ~/.ssh/id_vcl
```

Branch protection: on `dev` require `lint-test` (CI); on `release` require `trained-candidate`
(release-check).

## Contracts the workflows rely on

Files marked *(exists)* are in the repo; the rest do not exist yet. Each owner can change the interface; update the workflow with it.

| File | Contract |
|---|---|
| `vcl/` (PR #5) | `python -m vcl up gpu-train --inventory <file>` and `python -m vcl down`, reading `$VCL_TOKEN` |
| `infra/ansible/train.yml` *(exists)* | needs passwordless sudo for the VCL user; on the `vcl` inventory group: installs Docker and the NVIDIA container toolkit if needed; logs in to GHCR with `$GHCR_USER`/`$GHCR_TOKEN` (`lookup('env', ...)`); runs `pipeline.train`, then `pipeline.evaluate` on the candidate and (if `champion_dir` is set) on the champion, in `{{ image }}` with `$HF_TOKEN`; copies `model/`, `metrics.json` and `champion_metrics.json` to `{{ artifacts_dir }}` on the controller. Extra vars: `image`, `dataset`, `revision`, `artifacts_dir`, optional `champion_dir`. |
| `docker/train.Dockerfile` | CUDA training environment with the repo code |
| `pipeline/train.py` | `--dataset <repo> --revision <sha> --out <dir>`; writes the model to `<dir>` with `<dir>/model.json`: `name` (the hand-set `MODEL_NAME`, format in the model-release skill), `type`, and anything the loader needs |
| `pipeline/evaluate.py` | `--model <dir> --dataset <repo> --revision <sha> --out <file>`; loads any past model by its `model.json` `type` (`minilm`, `prototype`, `d1`, `glide`); scores every subreddit's test split (`--split val` for validation, up to 40,000 posts each; `--per-sub N` scores fewer) and writes JSON with the macro averages `pr_auc`, `top10_hit_rate` and `spearman` (each subreddit with 1,000+ posts counts once), `by_subreddit`, and `rows` (hash of the scored post ids). Set `EVAL_CACHE` to share the labeled splits between containers |
| `candidate.yaml` *(exists)* | `path` (HF folder with the weights), `name`, `type`. Checked by `scripts/hf_models.py candidate` |
| `infra/ansible/gate.yml` *(exists)* | on the instance: downloads the candidate (`hf_models.py fetch`) and the champion, runs `pipeline.evaluate` on both, copies `metrics.json` and `champion_metrics.json` to `{{ results_dir }}`. Extra vars: `image`, `dataset`, `revision`, `candidate_path`, `candidate_revision`, `candidate_name`, `candidate_type`, `results_dir` |
| `docker/serve.Dockerfile` | inference image; `COPY model/` from the build context (the scored weights, with `model.json`) |
| `infra/deploy/deploy.sh <image>` | deploys the image to production |

The gate (`scripts/compare_metrics.py`) compares the candidate with the champion, both scored in the same run
by the same `evaluate.py` on the same test data. It fails if the two `rows` hashes differ (different posts). It fails if the macro `pr_auc` or `top10_hit_rate` drops more than 0.01. It warns (does not fail) when one
subreddit's `pr_auc` drops more than 0.03.
With no champion yet, it passes. The gate checks "not worse", not "better": 0.01 is near the noise on the test
split.

## Model storage

Models live in the Hugging Face repo `zachnorton03/reddit-post-predictor-models` (`scripts/hf_models.py`).

- Experiment runs: `experiments/<config>/<short commit>/<model>_s<seed>/` (see "Experiments").
- Manual train-eval runs: `models/<name>/`, where `<name>` is `MODEL_NAME` from `pipeline/train.py`: `model/`
  (with `model.json`), `metrics.json`, `champion_metrics.json` and `run.json`. Retraining the same model replaces
  that folder.
- `candidate.yaml` names one such folder (`models/<name>/model` for a train-eval run). `release-check` pins the
  repo's commit when it starts and scores the folder at that commit. `release.json` in its artifact records the
  commit as `revision`.
- Deploy downloads the folder at that `revision`, so it ships exactly what was scored. `promote` then adds
  `release.json` (and `model.json`, which experiment runs lack) next to the weights, checks that the weights there
  are unchanged from `revision`, tags that commit `model-<n>` and points `champion.json` at it
  (`{version, name, type, path, commit, revision}`). `model-<n>` matches the GitHub release `model-<n>`.
- The next check downloads the champion at its `revision` and scores it again with the new `evaluate.py`. So a
  changed metric or dataset still compares like with like.

## Experiments

Research runs (new architectures, seeds, settings) go through `experiment.yml`, not `train-eval`. They're scored on
2024 **validation only**, so choosing among many experiments never uses test. Test is used once, when a chosen
model ships, by the blocking gate.

1. On a feature branch, add `experiments/<name>.yaml` (format below, and in `pipeline/experiment.py`). Pushing the
   branch starts the workflow. To re-run without a change: `gh workflow run experiment.yml --ref <branch> -f
   config=experiments/<name>.yaml`.
2. One L40S reservation (up to 10 hours) runs every listed run in order through the config's `runner` script.
   Each finished run is uploaded from the GPU instance to `zachnorton03/reddit-post-predictor-models` at
   `experiments/<name>/<short commit>/<model>_s<seed>/`, weights included (not `resume.pt` or
   `best_checkpoint.pt`). A run that doesn't finish before the reservation ends is lost; a re-run on the same commit
   skips runs already on HF.
3. Each finished run and the deployed champion are scored by the same `pipeline.evaluate --split val` on every
   subreddit (`val_eval.json` in the run's folder). The workflow posts a table of the macro averages and the AP per
   subreddit, every run next to the champion, on the branch's open PR (or in the run summary if there's no PR).
4. Write up the result in `PROJECT_NOTES.md` in the PR, by hand.

```yaml
runner: run_pooling.py          # script at the repo root that follows the runner contract
runs:
  - {model: deberta_attn4, seeds: [42, 43, 44]}
  - {model: deberta_large_attn4, seeds: [42]}
epochs: 3                       # optional
allow_fallback: false           # optional; true allows the RTX 2080 Ti when no L40S is free (base-size models)
type: prototype                 # optional; the pipeline/evaluate.py loader that reads the saved runs
eval_per_sub: 3000              # optional; validation posts scored per subreddit (default all, up to 40,000), for slow models
```

**Runner contract.** Called once per run, from the repo root, in the training image:
`python <runner> --models <model> --seeds <seed> --output <dir>/runs --results <dir>/results --resume --dataset
<repo> --revision <sha> [--epochs N]`. It must load data only from the pinned revision, never load or score test,
write `<dir>/runs/<model>_s<seed>/best.json`, and save the weights so the config's `type` loader can score them.

**Timing.** Measure a new model with one seed before listing many. A config that doesn't fit in about 9 hours
needs splitting. The self-hosted controller runs one job at a time, so a long experiment delays other GPU jobs
(including `release-check`) until it ends.

**Shipping a winner:** set `candidate.yaml` to the run's folder, `name` and `type: prototype` on a feature branch,
merge it into `dev`, then open `dev` → `release`. `release-check` scores those saved weights and the champion on
test, with no retraining, and blocks if the candidate is worse.

## VCL reservations

Every job ends its reservation in an `always()` step, including on failure, cancel and timeout. If the controller
crashes mid-job, that step can't run. So jobs record each reservation in a ledger on the controller
(`~/.vcl-pipeline/`, `scripts/vcl_ledger.py`), and `vcl-cleanup.yml` ends any recorded reservation whose workflow run
has finished. Reservations made by hand aren't in the ledger and are never touched. `train-eval` also reserves
only 240 minutes and `release-check` 120, matching their timeouts.

## Modal fallback

`train-eval-modal.yml` runs the same `pipeline.train` and `pipeline.evaluate` on a Modal GPU, through
`infra/modal/train.py`. It builds `docker/train.Dockerfile` on Modal (no GHCR login) and adds the repo code at
`/repo`. Use it only when VCL GPUs are unavailable or too small.

- Manual only. Pick the GPU when you start the run (default `L40S`). The function times out after 10 hours.
- It uploads the same `candidate` artifact and runs the same gate, and `run.json` records `"provider": "modal"`.
- It uploads to `models/<name>/` like `train-eval`, so `candidate.yaml` can name it (`models/<name>/model`).
- Its gate compares with the champion's stored metrics (`release.json`). It does not score the champion again.
- Models trained on different GPUs score slightly differently. Compare against the provider in `run.json`.

One-time setup, only if the fallback is used:

```sh
gh api -X PUT repos/zachnorton14/reddit-post-predictor/environments/modal
gh secret set MODAL_TOKEN_ID --env modal
gh secret set MODAL_TOKEN_SECRET --env modal
```

Locally: `modal run infra/modal/train.py --dataset <repo> --revision <sha> --out artifacts --gpu L40S`
(needs `modal token new` and `HF_TOKEN`).
