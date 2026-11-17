# CI/CD pipeline

Branch flow: feature branch → `dev` → `release` → automatic deploy.

1. **Pull request into `dev`:** CI (Ruff, pytest) must pass. Dataset validation runs when dataset code changes.
   No training.
2. **Push to `dev`** (the merge) that changes model files (`*.py` at the root, `pipeline/`,
   `docker/train.Dockerfile`, `requirements.txt`): `train-eval` trains a candidate on a VCL GPU, scores it and
   the deployed champion on the same test data, and gates it. The result attaches to that commit.
3. **Pull request `dev` → `release`:** `release-check` must pass. It finds the passing train-eval run for the
   newest model change on the branch. Commits that change only docs or workflows reuse that run. Nothing is
   retrained.
4. **Push to `release`** (the merge): `deploy` ships that run's candidate and makes it the champion.

How to name, change and ship a model: `.claude/skills/model-release/SKILL.md`.

| Workflow | Trigger | Runner | What it does |
|---|---|---|---|
| `ci.yml` | pull request into `dev`; manual | GitHub-hosted | Ruff, pytest. Required on `dev`. |
| `data-validation.yml` | pull request into `dev` that changes dataset code; Hugging Face webhook relay; manual | GitHub-hosted | validates the dataset |
| `train-eval.yml` | push to `dev` that changes model files; manual (`smoke: true` only checks VCL access) | hosted + self-hosted controller | builds training image, reserves a VCL GPU, trains, scores candidate and champion, uploads the candidate, gates |
| `release-check.yml` | pull request into `release` | GitHub-hosted | requires a passing train-eval run for the newest model change (`scripts/find_candidate.sh`). Required on `release`. |
| `train-eval-modal.yml` | manual only | GitHub-hosted | fallback: trains and evaluates on a Modal GPU, gates (see "Modal fallback") |
| `experiment.yml` | push to a feature branch that changes `experiments/*.yaml`; manual (`-f config=...`) | hosted + self-hosted controller | runs every listed research run on a VCL L40S, validation only; stores each on HF; posts a table on the PR (see "Experiments"). Advisory, never blocks. |
| `vcl-cleanup.yml` | every 30 minutes; manual | self-hosted controller | ends reservations that a pipeline job recorded but couldn't end (`scripts/vcl_ledger.py`) |
| `deploy.yml` | push to `release` | GitHub-hosted | builds inference image from the gated model, deploys, publishes release `model-<n>`, promotes the champion |

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
- The repo is public. The self-hosted jobs run only on pushes to `dev` and manual runs, never on pull
  requests, so fork code never reaches the controller. Also set Settings → Actions → General → "Require
  approval for all outside collaborators".

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
| `pipeline/evaluate.py` | `--model <dir> --dataset <repo> --revision <sha> --out <file>`; loads any past model by its `model.json` `type`; writes JSON with at least `pr_auc` and `top10_hit_rate` on that revision's test split |
| `docker/serve.Dockerfile` | inference image; `COPY model/` from the build context (the gated artifact) |
| `infra/deploy/deploy.sh <image>` | deploys the image to production |

The gate (`scripts/compare_metrics.py`) compares the candidate with the champion, both scored in the same run
by the same `evaluate.py` on the same test data. It fails if `pr_auc` or `top10_hit_rate` drops more than 0.01.
With no champion yet, it passes. The gate checks "not worse", not "better": 0.01 is near the noise on the test
split.

## Model storage

Models live in the private Hugging Face repo `zachnorton03/reddit-post-predictor-models`
(`scripts/hf_models.py`; the layout is in the repo's README).

- Every train-eval run uploads its candidate to `models/<name>/`, where `<name>` is `MODEL_NAME` from
  `pipeline/train.py`: `model/` (with `model.json`), `metrics.json`, `champion_metrics.json` and `run.json`.
  Retraining the same model replaces that folder.
- After the upload, `run.json` in the GitHub `candidate` artifact gets `hf_revision`: the Hugging Face commit
  that holds this run's files. (The copy on Hugging Face can't contain its own commit, so it has no `hf_revision`.)
- Deploy promotes the candidate. It creates the git tag `model-<n>` on that `hf_revision` and points
  `champion.json` at the folder, so the tag matches the files that were deployed. The tag keeps the champion's
  files even if a later run, including a manual run on a branch, replaces the folder. `model-<n>` matches the
  GitHub release `model-<n>`.
- Runs from before `hf_revision` existed can't be promoted; `promote` fails with a clear error. Retrain on `dev`.
- The next train-eval run downloads the champion at its tag and scores it again with the new `evaluate.py`.
  So a changed metric or dataset still compares like with like.

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
3. The workflow also scores the deployed champion on validation (`pipeline.evaluate --split val`). It posts a table
   of every run next to the champion on the branch's open PR (or in the run summary if there's no PR).
4. Write up the result in `PROJECT_NOTES.md` in the PR, by hand.

```yaml
runner: run_pooling.py          # script at the repo root that follows the runner contract
runs:
  - {model: deberta_attn4, seeds: [42, 43, 44]}
  - {model: deberta_large_attn4, seeds: [42]}
epochs: 3                       # optional
allow_fallback: false           # optional; true allows the RTX 2080 Ti when no L40S is free (base-size models)
```

**Runner contract.** Called once per run, from the repo root, in the training image:
`python <runner> --models <model> --seeds <seed> --output <dir>/runs --results <dir>/results --resume --dataset
<repo> --revision <sha> [--epochs N]`. It must load data only from the pinned revision, never load or score test,
and write `<dir>/runs/<model>_s<seed>/best.json` with `ap`, `precision_top10` and `spearman`.

**Timing.** Measure a new model with one seed before listing many. A config that doesn't fit in about 9 hours
needs splitting. The self-hosted controller runs one job at a time, so a long experiment delays `train-eval` on
`dev` until it ends.

**Shipping a winner** (planned): a person names the chosen run in a PR. The gate on the `dev` → `release` PR scores
those saved weights and the champion on test, with no retraining, and blocks if the candidate is worse. Until that
exists, ship through `pipeline/train.py` as described in the model-release skill.

## VCL reservations

Every job ends its reservation in an `always()` step, including on failure, cancel and timeout. If the controller
crashes mid-job, that step can't run. So jobs record each reservation in a ledger on the controller
(`~/.vcl-pipeline/`, `scripts/vcl_ledger.py`), and `vcl-cleanup.yml` ends any recorded reservation whose workflow run
has finished. Reservations made by hand aren't in the ledger and are never touched. `train-eval` also reserves
only 240 minutes, matching its timeout.

## Modal fallback

`train-eval-modal.yml` runs the same `pipeline.train` and `pipeline.evaluate` on a Modal GPU, through
`infra/modal/train.py`. It builds `docker/train.Dockerfile` on Modal (no GHCR login) and adds the repo code at
`/repo`. Use it only when VCL GPUs are unavailable or too small.

- Manual only. Pick the GPU when you start the run (default `L40S`). The function times out after 10 hours.
- It uploads the same `candidate` artifact and runs the same gate, and `run.json` records `"provider": "modal"`.
- It does not satisfy `release-check`, and `deploy.yml` deploys only models from `train-eval.yml` runs on
  `dev`. A Modal model is for comparison until the deploy is changed to accept it.
- Its gate compares with the champion's stored metrics. It does not score the champion again.
- Models trained on different GPUs score slightly differently. Compare against the provider in `run.json`.

One-time setup, only if the fallback is used:

```sh
gh api -X PUT repos/zachnorton14/reddit-post-predictor/environments/modal
gh secret set MODAL_TOKEN_ID --env modal
gh secret set MODAL_TOKEN_SECRET --env modal
```

Locally: `modal run infra/modal/train.py --dataset <repo> --revision <sha> --out artifacts --gpu L40S`
(needs `modal token new` and `HF_TOKEN`).
