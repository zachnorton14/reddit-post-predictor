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
