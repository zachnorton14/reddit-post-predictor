# CI/CD pipeline

Branch flow: feature branch → `dev` → `release` (staging on VCL) → `main` (production on Kubernetes on VCL).
A model is trained once and then only scored.

1. **Experiment, on a feature branch:** push `experiments/<name>.yaml`. `experiment` trains each listed run on a
   VCL GPU, scores it on validation, stores the weights on Hugging Face and posts a table on the PR.
2. **Choose:** name the chosen run's folder in `candidate.yaml` (choose on validation, never on test).
3. **Pull request into `dev`:** CI (Ruff, pytest) must pass. Dataset validation runs when dataset code changes.
   Merging uses no GPU and trains nothing.
4. **Pull request `dev` → `release`:** `release-check` downloads the `candidate.yaml` folder at a pinned Hugging Face
   revision and scores it and the model on staging (the latest `rc-<n>`) on the same test data. It blocks the merge
   if the candidate is worse. Nothing is trained.
5. **Push to `release`** (the merge): `deploy` serves exactly the revision the check scored on the VCL staging
   instance and publishes the pre-release `rc-<n>` (see "Staging"). It does not make the model the champion.
6. **Push to `main`** (the `release` → `main` merge): `deploy-production` rolls the staged image out to the k3s
   cluster on the production VCL instance under nonstop traffic, rolls back on its own if anything fails, and only
   after a rollout with zero failed requests tags the image `model-<n>`, promotes it on Hugging Face as the champion
   and publishes `model-<n>` (see "Production").

`train-eval` (retrain `pipeline/train.py`'s model) is now manual only. How to name, change and ship a model:
`.claude/skills/model-release/SKILL.md`.

| Workflow | Trigger | Runner | What it does |
|---|---|---|---|
| `ci.yml` | pull request into `dev`; manual | GitHub-hosted | Ruff, pytest. Required on `dev`. |
| `data-validation.yml` | pull request into `dev` that changes dataset code; Hugging Face webhook relay; manual | GitHub-hosted | validates the dataset |
| `train-eval.yml` | manual only (`smoke: true` only checks VCL access) | hosted + self-hosted controller | retrains `pipeline/train.py`'s model on a VCL GPU, scores it and the champion, uploads it to `models/<name>/`, gates (advisory) |
| `release-check.yml` | pull request from `dev` into `release` (`pull_request_target`) | hosted + self-hosted controller | scores the `candidate.yaml` model and the baseline (the latest `rc-<n>`, else the champion) on test on a VCL GPU, no training; uploads `release-<head sha>` on a pass. Job `trained-candidate` is required on `release`. |
| `train-eval-modal.yml` | manual only | GitHub-hosted | fallback: trains and evaluates on a Modal GPU, gates (see "Modal fallback") |
| `experiment.yml` | push to a feature branch that changes `experiments/*.yaml`; manual (`-f config=...`) | hosted + self-hosted controller | runs every listed research run on a VCL L40S, validation only; stores each on HF; posts a table on the PR (see "Experiments"). Advisory, never blocks. |
| `vcl-cleanup.yml` | every 30 minutes; manual | self-hosted controller | ends reservations that a pipeline job recorded but couldn't end (`scripts/vcl_ledger.py`) |
| `deploy.yml` | push to `release` | hosted + self-hosted controller | downloads the weights at the revision `release-check` scored, builds the CPU inference image, serves it on a VCL `cpu-staging` instance and checks it answers, publishes the pre-release `rc-<n>` with `release.json` (see "Staging"). No promotion. |
| `deploy-production.yml` | push to `main`; manual | hosted + self-hosted controller | finds the `rc-<n>` staged at the merged release commit, rolls its image out to k3s on the VCL `cpu-production` instance under traffic (`production-rollout.yml`), then tags `serve:model-<n>`, promotes on HF and publishes `model-<n>` (see "Production") |
| `production-provision.yml` | manual | hosted + self-hosted controller | brings production up with the current champion ahead of a demo; promotes nothing |
| `rollback-drill.yml` | manual (`scenario`: `broken-release`, `undo-last-release`) | self-hosted controller | on the live production cluster, under traffic: a release whose model doesn't load is rolled back automatically, or the last release is undone; passes only with zero failed requests |
| `production-rollout.yml` | called by the three above | self-hosted controller | one monitored change to production: reuse or reserve the instance, `infra/ansible/production.yml`, record it, judge the traffic report |

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
| `VCL_TOKEN` | environments `vcl` and `production` | train-eval, release-check, experiment, deploy; production-rollout | VCL API token (vcl.ncsu.edu → Manage → User Preferences → Manage Tokens) |
| `VCL_SSH_KEY` | environments `vcl` and `production` | train-eval, release-check, experiment, deploy; production-rollout | private key Ansible uses to reach the VCL instance |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | environment `modal` | train-eval-modal | Modal API token (`modal token new`). Only needed if the fallback is used. |

Environments: `vcl` (no reviewers, so runs are not blocked) and `production` (deployment branch: `main` only;
optionally a required reviewer, which then also gates the provision and drill runs). The production jobs run in
`production`, so it needs its own copy of the two VCL secrets. The cluster needs no credential in GitHub: the
playbook runs `kubectl` on the instance over SSH.

One-time setup (repo admin):

```sh
gh api -X PUT repos/zachnorton14/reddit-post-predictor/environments/vcl
gh api -X PUT repos/zachnorton14/reddit-post-predictor/environments/production \
  -F 'deployment_branch_policy[protected_branches]=false' -F 'deployment_branch_policy[custom_branch_policies]=true'
gh api -X POST repos/zachnorton14/reddit-post-predictor/environments/production/deployment-branch-policies -f name=main

gh secret set HF_TOKEN                        # prompts for the value
gh secret set VCL_TOKEN --env vcl
gh secret set VCL_SSH_KEY --env vcl < ~/.ssh/id_vcl
gh secret set VCL_TOKEN --env production
gh secret set VCL_SSH_KEY --env production < ~/.ssh/id_vcl
```

If `production` already allows `release`, delete that policy (`gh api repos/.../environments/production/deployment-branch-policies`
lists them) and add `main`. `production-provision` and `rollback-drill` must then also be run from `main`.

Branch protection: on `dev` require `lint-test` (CI); on `release` require `trained-candidate`
(release-check); on `main` require a pull request (from `release`).

## Contracts the workflows rely on

Files marked *(exists)* are in the repo; the rest do not exist yet. Each owner can change the interface; update the workflow with it.

| File | Contract |
|---|---|
| `vcl/` (PR #5) | `python -m vcl up gpu-train --inventory <file>` and `python -m vcl down`, reading `$VCL_TOKEN` |
| `infra/ansible/train.yml` *(exists)* | needs passwordless sudo for the VCL user; on the `vcl` inventory group: installs Docker and the NVIDIA container toolkit if needed; logs in to GHCR with `$GHCR_USER`/`$GHCR_TOKEN` (`lookup('env', ...)`); runs `pipeline.train`, then `pipeline.evaluate` on the candidate and (if `champion_dir` is set) on the champion, in `{{ image }}` with `$HF_TOKEN`; copies `model/`, `metrics.json` and `champion_metrics.json` to `{{ artifacts_dir }}` on the controller. Extra vars: `image`, `dataset`, `revision`, `artifacts_dir`, optional `champion_dir`. |
| `docker/train.Dockerfile` | CUDA training environment with the repo code |
| `pipeline/train.py` | `--dataset <repo> --revision <sha> --out <dir>`; writes the model to `<dir>` with `<dir>/model.json`: `name` (the hand-set `MODEL_NAME`, format in the model-release skill), `type`, and anything the loader needs |
| `pipeline/evaluate.py` | `--model <dir> --dataset <repo> --revision <sha> --out <file>`; loads any past model by its `model.json` `type` (`minilm`, `prototype`); scores every subreddit's test split (`--split val` for validation, up to 40,000 posts each) and writes JSON with the macro averages `pr_auc`, `top10_hit_rate` and `spearman` (each subreddit with 1,000+ posts counts once), `by_subreddit`, and `rows` (hash of the scored post ids). Set `EVAL_CACHE` to share the labeled splits between containers |
| `candidate.yaml` *(exists)* | `path` (HF folder with the weights), `name`, `type`. Checked by `scripts/hf_models.py candidate` |
| `infra/ansible/gate.yml` *(exists)* | on the instance: downloads the candidate (`hf_models.py fetch`) and the champion, runs `pipeline.evaluate` on both, copies `metrics.json` and `champion_metrics.json` to `{{ results_dir }}`. Extra vars: `image`, `dataset`, `revision`, `candidate_path`, `candidate_revision`, `candidate_name`, `candidate_type`, `results_dir`, and optional `baseline_path`, `baseline_revision`, `baseline_name`, `baseline_type` (downloaded instead of the champion; scored into `champion_metrics.json` either way) |
| `docker/serve.Dockerfile` *(exists)* | CPU inference image; `COPY model/` from the build context (the scored weights, with `model.json`) to `/model`; build arg `MODEL_VERSION`; loads and scores one post at build (`python -m pipeline.serve --check /model`); serves on port 8000 |
| `pipeline/serve.py` *(exists)* | `GET /health`, `POST /predict` `{"posts": [{"title", "body"}]}` → `{"predictions": [{"score"}]}`; serves the `model.json` types in `PREDICTORS` (`minilm`) |
| `infra/ansible/staging.yml` *(exists)* | on the `vcl` inventory group: installs Docker if needed, pulls `{{ image }}` with `$GHCR_USER`/`$GHCR_TOKEN`, runs it as container `serve` on port 8000 with `--restart unless-stopped`, checks `/health` and `/predict`. Extra vars: `image`, optional `port` |
| `scripts/release_version.py` *(exists)* | `next <prefix> <commit>` prints the next `<prefix>-<n>` from the git tags, or the one already at `<commit>`; `latest <prefix>` prints the highest existing one |
| `scripts/vcl_staging.py` *(exists)* | `current [--role R]` prints the recorded request id; `live [--role R]` prints it only while VCL reports it `ready`; `record <image> <version> [--role R]` makes `.vcl_state.json` that role's reservation (`~/.vcl-pipeline/<role>.json`). `R` is `staging` (default) or `production` |
| `infra/ansible/production.yml` *(exists)* | needs passwordless sudo; on the `vcl` group: installs k3s (pinned, no traefik) if needed; pulls the images into containerd with `$GHCR_USER`/`$GHCR_TOKEN`; seeds a fresh cluster with `seed_image`; starts `scripts/probe.py run` on the instance, applies `infra/k8s/serve.yaml.j2` (or `rollout undo`), waits, checks `/health` reports `version` and `/predict` answers, else `rollout undo`; writes `probe.json` and `rollout.json` to `{{ report_dir }}` on the controller. Doesn't fail on a rollback. Extra vars: `mode` (`deploy`/`undo`), `image`, `version`, optional `seed_image`, `seed_version`, `break_model`, and `report_dir`; leave unused ones out rather than empty |
| `infra/k8s/serve.yaml.j2` *(exists)* | namespace `prod`; Deployment `serve` (3 replicas, `maxUnavailable: 0`, `/health` startup/readiness/liveness probes, `preStop` sleep, `MODEL_VERSION` set to the production version, `imagePullPolicy: IfNotPresent`); NodePort Service on 8000 → 30080. Vars: `serve_image`, `serve_version`, optional `serve_break`, `replicas`, `node_port`, `progress_deadline` |
| `scripts/probe.py` *(exists)* | stdlib only. `run --url --out --stop-file` sends alternating `/health` and `/predict` until the stop file appears; `report <probe.json> <rollout.json> --expect rollout\|rollback` prints the Markdown report and exits 1 on any failed request, a wrong rollback outcome or a wrong final version; `undo-target` picks the revision to undo to |

**The baseline.** `release-check` compares the candidate with the latest release candidate `rc-<n>`, the model on
staging: it downloads `release.json` from that GitHub pre-release and passes its HF `path`, `revision`, `name` and
`type` to `gate.yml`. Each release is therefore gated against the one before it. With no `rc-<n>` yet it falls back
to the production champion (`champion.json`), and with neither it passes. `release.json` records which (`baseline`).

The gate (`scripts/compare_metrics.py`) compares the candidate with the baseline, both scored in the same run
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
- The staging deploy downloads the folder at that `revision`, so it ships exactly what was scored. It doesn't
  promote. The production deploy runs `promote` (after a rollout with zero failed requests), which adds
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

## Staging

`deploy` serves each release on a VCL CPU instance (profile `cpu-staging`: image 7392, "Ubuntu 22.04 - 16 cores
128 GB RAM", 240 minutes). The shipped MiniLM needs no GPU.

1. `build` (hosted) fetches the scored weights into `model/` and builds `docker/serve.Dockerfile` (CPU torch, model
   baked in, no network needed at start). It pushes `ghcr.io/<repo>/serve:sha-<release commit>` (the tag
   production will deploy, so it ships the bytes staging tested), `:rc-<n>` and `:staging`. It adds `serve_image`,
   `rc` and `release_commit` to `release.json`.
2. `deploy-staging` runs on the self-hosted controller, because VCL opens an instance's firewall only to the IP
   that asked for it (a hosted runner's IP changes every run). It reserves a new `cpu-staging` instance, runs
   `infra/ansible/staging.yml` and checks `/health` and `/predict` on the instance.
3. Only after that passes does it record the new reservation as staging and end the previous one. If anything
   fails, it ends the new reservation and the previous staging keeps serving, and `record` doesn't run.
4. `record` publishes the GitHub pre-release `rc-<n>` at the `release` commit, with `release.json` attached (a
   release asset doesn't expire like the 90-day workflow artifact). It doesn't touch Hugging Face.

`rc-<n>` counts up from the existing `rc-*` tags (`scripts/release_version.py`). A failed deploy makes no tag, so it
leaves no gap, and a re-run of the same commit reuses its number. The next `release-check` gates against the
latest `rc-<n>` (see "The baseline" above), so the gate works before production exists.

Staging lasts until its reservation ends; the next deploy replaces it. It is recorded in
`~/.vcl-pipeline/staging.json`, not in the cleanup ledger, so `vcl-cleanup` leaves it alone. While the deploy job
runs, the new reservation is in the ledger, so a crashed job doesn't leak it.

**Reaching it.** Only the controller can reach the instance. The job summary prints the IP and a tunnel command:

```sh
ssh -J <you>@<controller> -p 22 -L 8000:localhost:8000 <vcl user>@<staging ip>   # needs the VCL SSH key
curl localhost:8000/health
curl -X POST localhost:8000/predict -H 'Content-Type: application/json'   -d '{"posts": [{"title": "I used to hate facial hair", "body": "but then it grew on me."}]}'
```

`score` is the predicted engagement, `log1p(score) + log1p(num_comments)`. Rank posts by it; it is not a
probability. The server builds the text as training did (body cut at EDIT markers, then to 1,000 characters).
Only the `minilm` type can be served: `prototype` runs also read posting-time metadata (such as the author's
history) that a request doesn't carry, so `pipeline/serve.py` refuses them at start and the image build fails.

Locally: `python scripts/hf_models.py fetch <path> <revision> model --name <name> --type <type>`, then
`docker build -f docker/serve.Dockerfile -t serve . && docker run -p 8000:8000 serve`.

## Production

Production is a single-node k3s cluster on one VCL CPU instance (profile `cpu-production`: the staging image, 16
cores, 128 GB, 600 minutes, the student cap). It isn't permanent: the pipeline reserves it when it needs it and keeps
it until the reservation ends, and for a demo it is brought up ahead of time so it looks like it has been running.

**A deploy** (`deploy-production.yml`, on the `release` → `main` merge):

1. `resolve` (hosted) finds the merged pull request's head (the `release` commit), the `rc-<n>` tag at it and that
   pre-release's `release.json`. The image is its `serve_image`, `serve:sha-<release commit>`: the bytes staging
   tested; nothing is rebuilt. It fails if the commit was never staged. The version is the next `model-<n>`
   (`scripts/release_version.py next model`). The seed is the current champion's image (the latest `model-<n>`).
2. `rollout` (`production-rollout.yml` on the controller) reuses the production reservation if VCL still reports it
   ready (`scripts/vcl_staging.py live --role production`), otherwise reserves a new one (a cold start). It runs
   `infra/ansible/production.yml`:
   - installs k3s if needed and pulls the images into containerd with the run's GHCR token, so the cluster never
     holds a registry credential (`imagePullPolicy: IfNotPresent`);
   - on a fresh cluster, deploys the champion first, so the new release still arrives as a rolling update;
   - starts `scripts/probe.py` on the instance, which sends an alternating `/health` and `/predict` every 0.1 s to the
     Service and records each answer's version, applies the new release, and waits for the rollout;
   - checks that `/health` reports the new version and `/predict` answers. If the new pods aren't Ready by the
     progress deadline (180 s), or a check fails, it runs `kubectl rollout undo` on its own;
   - copies `probe.json` and `rollout.json` back.

   Whatever serves at the end is recorded as production (`~/.vcl-pipeline/production.json`), even after a rollback,
   because it's a release that works. A new reservation is ended only if the playbook itself failed.
3. `probe.py report --expect rollout` judges the run: any failed request, a rollback or a wrong final version fails
   the job, and `record` doesn't run. The report (requests, failures, availability, longest outage, latency, and
   which version answered when) is in the job summary and the `production-report` artifact.
4. `record` (hosted) tags the image `serve:model-<n>` and `serve:production` (`docker buildx imagetools create`, no
   rebuild), adds `model`, `production_commit` and `staged_as` to `release.json`, promotes it on Hugging Face
   (`scripts/hf_models.py promote`) and publishes the GitHub release `model-<n>` with `release.json` and the report.

**Why a rollout has no downtime** (`infra/k8s/serve.yaml.j2`): `maxUnavailable: 0` keeps the old pods until a new one
is Ready; the readiness probe holds traffic until the model has loaded; the `preStop` sleep keeps a stopping pod
serving while the Service drops it; and the progress deadline turns a release that never gets Ready into a failed
rollout that can be undone. The probe runs on the instance, so it measures the Service, not the network from outside.

**Before a demo**, run `production-provision` (less than 10 hours ahead; extend with `python -m vcl extend <minutes>
<request id>` on the controller). It reserves or reuses production and serves the champion (or, before the first
production release, the latest `rc-<n>`). It promotes nothing. Then a `release` → `main` merge rolls the new release
onto it, and `rollback-drill` works on it.

**Rollback drill** (`rollback-drill.yml`, manual, needs a live production; reserves nothing):

- `broken-release` deploys the live image with no model (`serve_break`: `MODEL_DIR` points at an empty folder). The
  one new pod never becomes Ready, so the old pods keep serving, the rollout passes its deadline and the playbook
  rolls back. Passes only if it was rolled back, no request failed and the old version answers at the end.
- `undo-last-release` undoes to the previous release under traffic (the newest earlier revision that the
  broken-release drill didn't break: `probe.py undo-target`). Passes only with zero failed requests and the previous
  version answering at the end. Production then serves that release until the next deploy; nothing is re-promoted.

**Reaching it.** Only the controller can reach the instance. The job summary prints the IP, the Service address and
a tunnel command (`ssh -J <you>@<controller> -p <port> -L 8000:<service ip>:8000 <vcl user>@<ip>`), then
`curl localhost:8000/health`. On the instance: `k3s kubectl -n prod get pods`, `k3s kubectl -n prod rollout history
deployment/serve` (the change cause is the version).

**Limits.** One node, so no node-failure demo. A reservation that ends takes production with it; the next deploy
cold-starts. If the VCL API can't be reached, `live` reports nothing and the deploy reserves a new instance.

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
