---
name: model-release
description: Name, change or ship the model that the CI/CD pipeline deploys. Use when editing candidate.yaml, pipeline/train.py or pipeline/evaluate.py LOADERS, switching the shipped model, setting a model name or model.json, or asking how a model gets from a branch to production.
---

# Model release

A model is trained once and then only scored. `candidate.yaml` at the repo root names the trained weights that the
next release ships: a folder in the Hugging Face repo `zachnorton03/reddit-post-predictor-models`.

```yaml
path: experiments/deberta_ladder/abc1234/deberta_large_attn4_s42   # an experiment run
name: run_pooling-435M-deberta_large_attn4
type: prototype
```

`scripts/hf_models.py candidate candidate.yaml` checks the file. A test checks it too.

## Name the model (by hand, every time the model changes)

Format: `<source>-<params>[-<detail>...]`

- `<source>`: the `.py` file that trains the model, without `.py` (for example `run_pooling`, `dadjokes_models`).
- `<params>`: total trainable parameters, rounded, with `K`, `M` or `B` (for example `22M`, `435M`, `0.6B`).
- `<detail>`: optional parts that tell two models apart: base model, pooling, LoRA rank, max tokens.
- Lowercase letters, digits, `_` and `.` only. Parts are joined with `-`. No dates.

Examples: `run_pooling-435M-deberta_large_attn4`, `dadjokes_models-22M-minilm_l6`, `d1-0.6B-lora_r16`.

## `model.json` and loaders

`pipeline/evaluate.py` loads a model by the `type` in its folder's `model.json`.

- Experiment runs have no `model.json`. The release check writes one from `candidate.yaml`
  (`{"type", "name", "hf_path"}`), and promote stores it next to the weights.
- `train-eval` runs write it in `pipeline/train.py` (`MODEL_NAME`, `type: minilm`, and the loader's settings).
- `type: prototype` loads a run saved by `prototype_models.train` (`encoder/`, `tokenizer/`, `head.pt`,
  `extra.pt`, `spec.json`, `preprocessing.npz`) and reads the full post body (`BODY_CHARS`).
- Add a loader for each new type. **Keep the loader for the current champion's type**: the release check scores
  the champion with the new code.

## Ship a model

1. Train it in an experiment (`experiments/<name>.yaml` on a feature branch; see `docs/pipeline.md`). Choose on the
   validation table. Never choose on test.
2. On a feature branch, set `candidate.yaml` (`path` from the results table, `name`, `type`). Open a PR into `dev`.
   CI must pass. Merging trains nothing.
3. Open a PR from `dev` into `release`. `release-check` downloads the folder at a pinned revision, scores it and the
   model on staging (the latest `rc-<n>`, else the champion) on the same test data, and fails if `pr_auc` or `top10_hit_rate` drops more than 0.01.
4. Merge. `deploy` downloads the same revision, serves it on the VCL staging instance, and publishes the GitHub
   pre-release `rc-<n>` with `release.json`. It does not make the model the champion.
5. Production (planned, issue #22): a push to `main` will deploy the staged image to Kubernetes, publish `model-<n>`
   and promote it on Hugging Face (`release.json` next to the weights, tag `model-<n>`, `champion.json`). Until then
   there is no champion.

To retrain `pipeline/train.py`'s model instead (for example on new data), run
`gh workflow run train-eval.yml --ref <branch>` and set `path: models/<MODEL_NAME>/model`, `type: minilm`.

## Where things are

- Models: Hugging Face `zachnorton03/reddit-post-predictor-models`. `champion.json` names the deployed model.
- Storage script: `scripts/hf_models.py`. Gate: `scripts/compare_metrics.py`. Check: `release-check.yml`.
- Pipeline overview: `docs/pipeline.md`.
