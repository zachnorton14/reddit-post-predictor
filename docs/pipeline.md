# CI/CD pipeline

Branch flow: feature branch → `dev` (pull request, CI must pass) → `release` (pull request, the model must
pass the gate) → automatic deploy.

| Workflow | Trigger | Runner | What it does |
|---|---|---|---|
| `ci.yml` (PR #4) | code changes | GitHub-hosted | Ruff, pytest |
| `data-validation.yml` (PR #4) | code changes; Hugging Face webhook relay | GitHub-hosted | validates the dataset |
| `train-eval.yml` | pull request into `release`; manual | hosted + self-hosted controller | builds training image, reserves a VCL GPU, trains, evaluates, gates |
| `deploy.yml` | push to `release` | GitHub-hosted | builds inference image from the gated model, deploys, publishes release `model-<n>` |

## Runners

- **GitHub-hosted** (`ubuntu-latest`) for everything except training.
- **One self-hosted controller** (`csc519-201-host`, labels `self-hosted, Linux, X64`). It has no GPU. In the
  `train-eval` job it reserves an RTX 2080 Ti instance with `python -m vcl up gpu-train` (PR #5), runs
  `infra/ansible/train.yml` against it, copies the results back, and always runs `python -m vcl down`.
  VCL lets only the IP that requested the connection reach the instance, so all of this stays in one job on
  the controller.
- The controller needs `python3` with `venv`, `git` and outbound SSH. The job installs `ansible-core`,
  `pyyaml` and `huggingface_hub` into `.venv` in its workspace.
- The repo is public. The self-hosted job runs only for pull requests from branches in this repo, never
  forks. Also set Settings → Actions → General → "Require approval for all outside collaborators".

## Secrets

Stored in GitHub, never in the repo. Locally, `HF_TOKEN` goes in `.env` (ignored by git).

| Secret | Where | Used by | Notes |
|---|---|---|---|
| `GITHUB_TOKEN` | automatic | all | GHCR push and pull, releases, artifact download. Nothing to set. |
| `HF_TOKEN` | repository | data validation, train-eval | Hugging Face read token |
| `VCL_TOKEN` | environment `vcl` | train-eval | VCL API token (vcl.ncsu.edu → Manage → User Preferences → Manage Tokens) |
| `VCL_SSH_KEY` | environment `vcl` | train-eval | private key Ansible uses to reach the GPU instance |
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

Branch protection: on `dev` require the CI check; on `release` also require `train-eval / gate`.

## Contracts the workflows rely on

These files do not exist yet. Each owner can change the interface; update the workflow with it.

| File | Contract |
|---|---|
| `vcl/` (PR #5) | `python -m vcl up gpu-train --inventory <file>` and `python -m vcl down`, reading `$VCL_TOKEN` |
| `infra/ansible/train.yml` | on the `vcl` inventory group: installs Docker and the NVIDIA container toolkit if needed; logs in to GHCR with `$GHCR_USER`/`$GHCR_TOKEN` (`lookup('env', ...)`); runs `pipeline.train` then `pipeline.evaluate` in `{{ image }}` with `$HF_TOKEN`; copies the model and `metrics.json` to `{{ artifacts_dir }}` on the controller. Extra vars: `image`, `dataset`, `revision`, `artifacts_dir`. |
| `docker/train.Dockerfile` | CUDA training environment with the repo code |
| `pipeline/train.py` | `--dataset <repo> --revision <sha> --out <dir>`; writes the model to `<dir>` |
| `pipeline/evaluate.py` | `--model <dir> --out <file>`; writes JSON with at least `pr_auc` and `top10_hit_rate` on the test split |
| `docker/serve.Dockerfile` | inference image; `COPY model/` from the build context (the gated artifact) |
| `infra/deploy/deploy.sh <image>` | deploys the image to production |

The gate (`scripts/compare_metrics.py`) fails if `pr_auc` or `top10_hit_rate` drops more than 0.01 below the
latest `model-*` release. With no release yet, it passes.
