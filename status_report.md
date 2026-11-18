# Project status report

Team: Jonathan Duran-Ortiz (jbduran), Zach Norton (zanorton), Cooper Sizemore (cgsizem3).
Proposal: [`DevOps Project Proposal.txt`](DevOps%20Project%20Proposal.txt). Pipeline details: [`docs/pipeline.md`](docs/pipeline.md).

## Where we are

As shown in our deployment pipeline figure, the first four stages are built and running. Stage 1 (triggers and
checks) works: a pull request into `dev` runs Ruff and pytest, and a change to the code or to the Hugging Face
dataset runs data validation. For a dataset change, a webhook relay starts the run. Stages 2 and 3 (build,
provision, train, evaluate) work too. A workflow builds the training Docker image and reserves a VCL GPU from a
self-hosted controller. Ansible then sets up Docker and the NVIDIA runtime on the instance, and the job trains and
evaluates there. It always ends the reservation, and a cleanup job ends any reservation that a crashed job left open.
The first real run trained our fine-tuned MiniLM in about 6 minutes on an L40S. Stage 4 (the gate) compares the
candidate with the deployed model on the same pinned test data. It fails if PR-AUC or the top-10% hit rate drops by
more than 0.01, and it is the required check on the `dev` → `release` pull request. What remains is stage 5
(deploy) and most of stage 6 (records). The deploy workflow exists, but the inference container, the serve
script and the production target are not built. We have not decided between a VCL staging instance and Kubernetes
yet (issue #22; steps in #23). The record format is defined (`champion.json`, `release.json`, a `model-<n>` tag and
a GitHub release with model version, Git commit, dataset revision and metrics). It will first be written by the
first release, which hasn't happened. From the testing depth area, integration tests and artifact checks are done.
The inference latency benchmark and the A/B traffic simulation are not started.

## Changes from the proposal

The biggest change is that a model is trained once, then only scored. In the proposal, every push trained a model
and the gate decided whether it went to `release`. That trained the same model many times. It also left no good
way to run research experiments, which are scored many times while we choose among them. Now experiments run on
feature branches from a YAML config. Each run is scored on validation only, its weights are stored in a Hugging
Face model repo, and the results go to a table on the pull request. To ship, we name the chosen run in
`candidate.yaml`. The release gate then scores those saved weights on test against the deployed model, and deploy
ships exactly that version. So test is used only at release, and merging into `dev` no longer uses a GPU. Smaller
changes:

- Model storage: models are stored and versioned in a Hugging Face model repo, with tags, not in a record file in Git.
- VCL access: we use our own Python VCL client (`vcl/`) instead of WolfCTL, so CI can reserve and end instances
  without anyone at the keyboard.
- GPU: the pipeline asks for an L40S first and falls back to the RTX 2080 Ti from the proposal.
- Modal fallback: a manual workflow can train on Modal when VCL has no GPU free.

## Contributions

- **Jonathan Duran-Ortiz:**
  - Code CI, dataset validation and the Hugging Face webhook relay:
    [b16fe51](https://github.com/zachnorton14/reddit-post-predictor/commit/b16fe517212ef45a3ec629097a37bf46e21ca05c)
    (PR #4)
  - Training and evaluation scripts and the training image:
    [0a00a05](https://github.com/zachnorton14/reddit-post-predictor/commit/0a00a058391065dd1e9c7befe963246427d9bf39)
    (PR #17)
- **Cooper Sizemore:**
  - VCL provisioning client:
    [cb631b1](https://github.com/zachnorton14/reddit-post-predictor/commit/cb631b10dc6719aa529f1be9cb02ce78c684523f)
  - Its tests:
    [309b3f2](https://github.com/zachnorton14/reddit-post-predictor/commit/309b3f252fbbbbc8f06db93a887a6477dd09d17f)
    (PR #5)
  - Workflow triggers (CI on pull requests into `dev`, CD on pushes to `release`):
    [4777774](https://github.com/zachnorton14/reddit-post-predictor/commit/4777774ee4ddb896305a4507172ac90097e1d835)
- **Zach Norton:**
  - Train-and-evaluate and deploy workflows with the release gate:
    [e7aa4bf](https://github.com/zachnorton14/reddit-post-predictor/commit/e7aa4bf9e5ed5d489536ed149597215b6ea30216)
    (PR #6)
  - Hugging Face model storage:
    [301d406](https://github.com/zachnorton14/reddit-post-predictor/commit/301d406cbf32428e494b2a6d5e2aa33ad83fd9ed)
    (PR #15)
  - Experiment workflow:
    [01b427a](https://github.com/zachnorton14/reddit-post-predictor/commit/01b427ae9c1cea8018e6bd1dc4892c90737dc41f)
    (PR #24)
  - Release gate on saved weights:
    [d86a929](https://github.com/zachnorton14/reddit-post-predictor/commit/d86a92990e0f5f47096a6045dc2f91e2d60502a4)
    (PR #26)
