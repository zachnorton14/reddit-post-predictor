# CI/CD pipeline

Branch flow:

```
feature ──PR──► dev ──PR──► release ──PR──► main
         CI + model gate    CD: staging (VCL)    CD: production (Kubernetes, planned)
```

- **CI** runs on every pull request into `dev`: lint, tests, dataset validation, and GPU training and
  evaluation with a regression gate. A pull request merges into `dev` only if CI passes.
- **CD to staging** runs on every push to `release` (dev promoted through a pull request). It deploys the gated
  model as the dev release on one VCL instance. It does not retrain.
- **CD to production** (planned, not built yet) will run on push to `main` (release promoted through a pull
  request). It will promote the staging image to a Kubernetes cluster.

| Workflow | Trigger | Runner | What it does |
|---|---|---|---|
| `ci.yml` | pull request into `dev`; manual | GitHub-hosted | Ruff, pytest |
| `data-validation.yml` | pull request into `dev` that changes validation code; Hugging Face webhook relay; manual | GitHub-hosted | validates the dataset |
| `train-eval.yml` | pull request into `dev`; manual | hosted + self-hosted controller | builds training image, reserves a VCL GPU, trains, evaluates, gates against the latest `model-*` release |
| `deploy-staging.yml` | push to `release`; manual | hosted + self-hosted controller | finds the newest gated dev pull request in the pushed history, builds the inference image from its model, deploys it to the staging VCL instance, publishes release `model-<n>` |
| production deploy (planned) | push to `main` | — | promotes the staging image (`serve:sha-<commit>`) to Kubernetes |

`ci.yml` and `train-eval.yml` have no path filter because they are required checks: a required check that a
path filter skips never reports, and the pull request cannot merge. `data-validation.yml` is path-filtered, so
it is not required.

## How staging finds the model

A push to `release` brings in all of dev, not one pull request. `scripts/find_gated_run.sh <commit>` takes the
newest pull request merged into `dev` whose merge commit is in `<commit>`'s history, and returns the successful
`train-eval` run for that pull request's head commit. `deploy-staging` downloads that run's `candidate`
artifact (retained 90 days). This is the right model only if every pull request was up to date with `dev` when it
merged, so `dev` requires branches to be up to date (see branch protection).

The inference image is tagged `serve:model-<n>`, `serve:staging`, and `serve:sha-<release commit>`. The sha tag
is for the planned production deploy, which will retag that exact image instead of rebuilding it.

## Runners

- **GitHub-hosted** (`ubuntu-latest`) for everything except the jobs that reach VCL.
- **One self-hosted controller** (`csc519-201-host`, labels `self-hosted, Linux, X64`). It has no GPU. VCL lets
  only the IP that requested the reservation reach the instance, so every job that reserves or connects to VCL
  runs here, in one job:
  - `train-eval`: `python -m vcl up gpu-train` (PR #5), `infra/ansible/train.yml`, copies the results back,
    and always runs `python -m vcl down`.
  - `deploy-staging / deploy`: `python -m vcl up serve-staging`, then `infra/ansible/serve.yml`. The
    reservation stays up after a successful deploy and ends only if the deploy fails.
- The controller needs `python3` with `venv`, `git` and outbound SSH. The jobs install `ansible-core`,
  `pyyaml` and (train-eval) `huggingface_hub` into `.venv` in their workspace.
- The repo is public. The self-hosted jobs run only for pull requests from branches in this repo (never
  forks) and for pushes to `release`. Also set Settings → Actions → General → "Require approval for all outside
  collaborators".

## Secrets

Stored in GitHub, never in the repo. Locally, `HF_TOKEN` goes in `.env` (ignored by git).

| Secret | Where | Used by | Notes |
|---|---|---|---|
| `GITHUB_TOKEN` | automatic | all | GHCR push and pull, releases, artifact download. Nothing to set. |
| `HF_TOKEN` | repository | data validation, train-eval | Hugging Face read token |
| `VCL_TOKEN` | environments `vcl` and `staging` | train-eval, deploy-staging | VCL API token (vcl.ncsu.edu → Manage → User Preferences → Manage Tokens) |
| `VCL_SSH_KEY` | environments `vcl` and `staging` | train-eval, deploy-staging | private key Ansible uses to reach the VCL instance |
| `KUBECONFIG_DATA` | environment `production` | production deploy (planned) | not needed yet |

Environments:
- `vcl`: no reviewers, so runs are not blocked.
- `staging`: deployment branch `release` only.
- `production`: for the planned Kubernetes deploy. It will be restricted to `main`; nothing uses it yet.

One-time setup (repo admin):

```sh
gh api -X PUT repos/zachnorton14/reddit-post-predictor/environments/vcl
gh api -X PUT repos/zachnorton14/reddit-post-predictor/environments/staging \
  -F 'deployment_branch_policy[protected_branches]=false' -F 'deployment_branch_policy[custom_branch_policies]=true'
gh api -X POST repos/zachnorton14/reddit-post-predictor/environments/staging/deployment-branch-policies -f name=release

gh secret set HF_TOKEN                        # prompts for the value
gh secret set VCL_TOKEN --env vcl
gh secret set VCL_SSH_KEY --env vcl < ~/.ssh/id_vcl
gh secret set VCL_TOKEN --env staging
gh secret set VCL_SSH_KEY --env staging < ~/.ssh/id_vcl
```

Branch protection:
- `dev`: require a pull request; require the checks `Code CI / lint-test` and `train-eval / gate`; require
  branches to be up to date before merging.
- `release`: require a pull request (from `dev`). No CI runs here; the model was gated on its way into `dev`.
- `main`: require a pull request (from `release`).

## Contracts the workflows rely on

Some of these files do not exist yet. Each owner can change the interface; update the workflow with it.

| File | Contract |
|---|---|
| `vcl/` (PR #5) | `python -m vcl up <profile> --inventory <file>` and `python -m vcl down`, reading `$VCL_TOKEN` |
| `vcl/environments.yaml` | profiles `gpu-train` (training) and `serve-staging` (CPU staging instance; commented out until its image name is known) |
| `infra/ansible/train.yml` | on the `vcl` inventory group: installs Docker and the NVIDIA container toolkit if needed; logs in to GHCR with `$GHCR_USER`/`$GHCR_TOKEN` (`lookup('env', ...)`); runs `pipeline.train` then `pipeline.evaluate` in `{{ image }}` with `$HF_TOKEN`; copies the model and `metrics.json` to `{{ artifacts_dir }}` on the controller. Extra vars: `image`, `dataset`, `revision`, `artifacts_dir`. |
| `infra/ansible/serve.yml` | on the `vcl` inventory group: installs Docker if needed; logs in to GHCR with `$GHCR_USER`/`$GHCR_TOKEN`; replaces any running `serve` container with `{{ image }}` (restart policy `unless-stopped`); fails unless `/health` returns 200 on the instance within a timeout. Extra vars: `image`. |
| `docker/train.Dockerfile` | CUDA training environment with the repo code |
| `pipeline/train.py` | `--dataset <repo> --revision <sha> --out <dir>`; writes the model to `<dir>` |
| `pipeline/evaluate.py` | `--model <dir> --out <file>`; writes JSON with at least `pr_auc` and `top10_hit_rate` on the test split |
| `docker/serve.Dockerfile` | inference image; `COPY model/` from the build context (the gated artifact); serves `/health` and `/predict` |
| `scripts/find_gated_run.sh <commit>` | prints the train-eval run id to deploy for a `release` commit; writes `id`, `pr`, `head` to `$GITHUB_OUTPUT` |

The gate (`scripts/compare_metrics.py`) fails if `pr_auc` or `top10_hit_rate` drops more than 0.01 below the
latest `model-*` release. With no release yet, it passes. Until the production deploy exists, `model-<n>` is
published by `deploy-staging`. Once production exists, staging will publish `rc-<n>` pre-releases and only the
production deploy will publish `model-<n>`, so the gate compares against what is in production.

## Open items

- The staging reservation is time-capped (`length_minutes`) and each push to `release` makes a new one. The
  previous instance is not ended; it expires on its own. Decide whether to end it explicitly or tear staging
  down after the health check.
