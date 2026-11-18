# Experiment configs

Each `<name>.yaml` here lists research runs for `.github/workflows/experiment.yml`. Pushing a feature branch that
adds or changes one runs it on a VCL L40S, scored on validation only, and posts the results on the PR.

```yaml
runner: run_pooling.py
runs:
  - {model: deberta_large_attn4, seeds: [42]}
```

The file name names the results folder on Hugging Face (`experiments/<name>/<short commit>/`), so use lowercase
letters, digits and `_`. Check a config locally with `python -m pipeline.experiment --check experiments/<name>.yaml`.
Full format, runner contract and timing: `docs/pipeline.md` "Experiments".

To ship a run, put its folder in `candidate.yaml` at the repo root (see `.claude/skills/model-release/SKILL.md`).

## Overnight DeBERTa setup/punchline experiment

`deberta_setup_punchline.yaml` tests whether the setup/punchline representation that helped the earlier MiniLM
blend also helps DeBERTa-v3. At each encoder size (base and large), the joint control reads the title and body
together in up to 256 tokens. The interaction model uses the same encoder twice, with up to 256 tokens each for
the title and body, keeping the end of a long body. Its head reads both vectors, their absolute difference and
their product. This tests the whole representation, including the separate token budgets and larger input to the head.

All four runs use fresh seed 45, four-query attention pooling, timing embeddings, an effective batch of 128 and
three epochs on the same 2019–2023 r/dadjokes posts. Each selects its checkpoint by 2024 validation AP. Read the
large interaction-minus-joint AP difference first, then check top-10% hit rate, Spearman and whether base agrees.
One seed per model makes this a screen for replication. The saved four-model ensemble's 0.2803 validation AP
is a reference for scale. The paired controls are the comparison for this experiment.

Plan for roughly 5–8 hours on one L40S. The previous five large joint runs took 8 h 35 min from start to finish.
The two-pass runtime is an estimate. A push with this new config starts the workflow. The Actions summary and
open PR receive the validation table, and each completed run saves predictions, post IDs and weights under
`experiments/deberta_setup_punchline/<short commit>/` in the models repo.
