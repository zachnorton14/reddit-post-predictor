---
name: model-release
description: Name, change or ship the model that the CI/CD pipeline trains. Use when editing pipeline/train.py, switching the shipped model, setting MODEL_NAME or model.json, or asking how a model gets from a branch to production.
---

# Model release

The pipeline trains whatever `pipeline/train.py` builds. Other models in the repo are experiments until
`train.py` uses them.

## Name the model (by hand, every time the model changes)

Set `MODEL_NAME` at the top of `pipeline/train.py`. `train.py` writes it into `model/model.json`.

Format: `<source>-<params>[-<detail>...]`

- `<source>`: the `.py` file that defines the model, without `.py` (for example `dadjokes_models`).
- `<params>`: total trainable parameters, rounded, with `K`, `M` or `B` (for example `22M`, `184M`, `0.6B`).
- `<detail>`: optional parts that tell two models apart: base model, blend, LoRA rank, max tokens.
- Lowercase letters, digits, `_` and `.` only. Parts are joined with `-`. No dates.

Examples: `dadjokes_models-22M-minilm_l6-hgb_blend`, `deberta-184M-v3_base`, `d1-0.6B-lora_r16`.
`scripts/hf_models.py path` checks the format, and the workflow fails on a bad name.

**Change the name whenever the model changes.** Retraining the same model on new data keeps the name. The
candidate then replaces `models/<name>/` on Hugging Face. Deployed champions are safe because each one has a
`model-<n>` tag.

## `model/model.json`

Written by `train.py` next to the model files. `evaluate.py` and serving read it to load any past model.

```json
{"name": "dadjokes_models-22M-minilm_l6-hgb_blend", "type": "minilm_hgb_blend", "params": 22700000}
```

- `type` picks the loader in `pipeline/evaluate.py`. Add a loader for each new type. Keep the loader for the
  current champion's type until a different model is deployed.
- Add any settings the loader needs (for example `base_model`, `max_len`).

## Change the shipped model

1. On a feature branch, put the model code in its own module (or a function in an existing one).
2. In `pipeline/train.py`, call the new code and set a new `MODEL_NAME`.
3. In `pipeline/evaluate.py`, add a loader for the new `type` if needed. Keep the champion's loader.
4. Choose settings on the validation split. Never tune on the test split.
5. Open a PR into `dev`. CI (Ruff and pytest) must pass. No training happens yet.
6. Merge. The push to `dev` runs `train-eval`:
   - it trains the candidate on a VCL GPU and uploads it to `models/<name>/`;
   - it scores the candidate and the champion on the same test data;
   - the gate fails if either metric drops more than 0.01.
7. Open a PR from `dev` into `release`. `release-check` passes only if the latest model change on `dev` was
   trained and gated.
8. Merge. `deploy` ships that candidate, publishes the GitHub release `model-<n>`, points `champion.json` at
   the candidate and tags the Hugging Face repo `model-<n>`.

## Where things are

- Models: Hugging Face `zachnorton03/reddit-post-predictor-models` (private). `champion.json` names the
  deployed model.
- Storage script: `scripts/hf_models.py`. Gate: `scripts/compare_metrics.py`.
- Pipeline overview: `docs/pipeline.md`.
