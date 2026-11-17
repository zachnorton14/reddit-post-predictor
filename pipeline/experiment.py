"""Run the training runs listed in an experiment config, scored on validation only, and store each on Hugging Face.

    python -m pipeline.experiment --config experiments/<name>.yaml --dataset <hf repo> --revision <sha> \\
        --commit <git sha> --out <dir>
    python -m pipeline.experiment --check experiments/a.yaml [...]     # validate configs (no GPU, no network)

Config (experiments/<name>.yaml):

    runner: run_pooling.py          # script at the repo root that follows the runner contract below
    runs:
      - {model: deberta_attn4, seeds: [42, 43, 44]}
      - {model: deberta_large_attn4, seeds: [42]}
    epochs: 3                       # optional, passed to every run
    allow_fallback: false           # optional; true lets the workflow use the RTX 2080 Ti when no L40S is free

Runner contract, called once per run from the repo root:

    python <runner> --models <model> --seeds <seed> --output <dir>/runs --results <dir>/results --resume \\
        --dataset <hf repo> --revision <sha> [--epochs N]

It must write <dir>/runs/<model>_s<seed>/best.json with `ap`, `precision_top10` and `spearman` on 2024 validation,
and must never load or score test. Each finished run is uploaded straight away to
experiments/<config name>/<short commit>/<model>_s<seed>/ in the models repo (all but resume.pt and
best_checkpoint.pt, which duplicate the weights), so a
reservation that runs out loses only the run in progress. Runs already uploaded for this commit are skipped.

The deployed champion is scored on validation too (pipeline.evaluate --split val). The summary
(<dir>/summary.json, <dir>/summary.md) compares each run with it. That comparison is advisory: the blocking gate
scores the chosen model on test when it ships (docs/pipeline.md).
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "zachnorton03/reddit-post-predictor-models"
NAME = re.compile(r"^[a-z0-9_]+$")
RUNNER = re.compile(r"^[a-z0-9_]+\.py$")
MAX_RUNS = 50
KEYS = {"runner", "runs", "epochs", "allow_fallback"}


class ConfigError(ValueError):
    pass


def load_config(path) -> dict:
    """Read and check a config. Raises ConfigError with a message that names the problem."""
    path = Path(path)
    if not NAME.match(path.stem):
        raise ConfigError(f"{path}: file name must be lowercase letters, digits and _ (it names the results folder)")
    try:
        cfg = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML: {e}") from e
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: must be a mapping with `runner` and `runs`")
    if unknown := set(cfg) - KEYS:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}; allowed: {sorted(KEYS)}")
    runner = cfg.get("runner")
    if not isinstance(runner, str) or not RUNNER.match(runner):
        raise ConfigError(f"{path}: `runner` must be a script at the repo root, like run_pooling.py")
    runs = cfg.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ConfigError(f"{path}: `runs` must be a non-empty list of {{model, seeds}}")
    for i, run in enumerate(runs):
        if not isinstance(run, dict) or set(run) != {"model", "seeds"}:
            raise ConfigError(f"{path}: runs[{i}] must have exactly `model` and `seeds`")
        if not isinstance(run["model"], str) or not NAME.match(run["model"]):
            raise ConfigError(f"{path}: runs[{i}].model must be lowercase letters, digits and _")
        seeds = run["seeds"]
        if not isinstance(seeds, list) or not seeds or not all(type(s) is int and s >= 0 for s in seeds):
            raise ConfigError(f"{path}: runs[{i}].seeds must be a non-empty list of whole numbers")
    epochs = cfg.get("epochs")
    if epochs is not None and (type(epochs) is not int or epochs < 1):
        raise ConfigError(f"{path}: `epochs` must be a positive whole number")
    if type(cfg.get("allow_fallback", False)) is not bool:
        raise ConfigError(f"{path}: `allow_fallback` must be true or false")
    if len(plan(cfg)) > MAX_RUNS:
        raise ConfigError(f"{path}: more than {MAX_RUNS} runs; split it into several configs")
    return {"name": path.stem, "runner": runner, "runs": runs, "epochs": epochs,
            "allow_fallback": cfg.get("allow_fallback", False)}


def plan(cfg) -> list[tuple[str, int]]:
    """(model, seed) pairs in config order, without repeats."""
    seen = []
    for run in cfg["runs"]:
        for seed in run["seeds"]:
            if (run["model"], seed) not in seen:
                seen.append((run["model"], seed))
    return seen


def run_dir_name(model, seed) -> str:
    return f"{model}_s{seed}"


def hf_prefix(cfg, commit) -> str:
    return f"experiments/{cfg['name']}/{commit[:7]}"


def runner_command(cfg, model, seed, out, dataset, revision) -> list[str]:
    cmd = [sys.executable, cfg["runner"], "--models", model, "--seeds", str(seed), "--output", str(out / "runs"),
           "--results", str(out / "results"), "--resume", "--dataset", dataset, "--revision", revision]
    if cfg["epochs"]:
        cmd += ["--epochs", str(cfg["epochs"])]
    return cmd


def summarize(cfg, commit, rows, champion) -> str:
    """Markdown table of the runs (and per-model means) next to the champion's validation scores."""
    lines = [f"### Experiment `{cfg['name']}` at `{commit[:7]}` (2024 validation)", "",
             "| Run | Status | Val AP | Top-10% hit | Spearman | Minutes |", "|---|---|---:|---:|---:|---:|"]
    fmt = lambda v, f: format(v, f) if isinstance(v, (int, float)) else "–"  # noqa: E731
    if champion:
        lines.append(f"| **champion** {champion['name']} | deployed | {fmt(champion['pr_auc'], '.4f')} | "
                     f"{fmt(champion['top10_hit_rate'], '.4f')} | {fmt(champion['spearman'], '.4f')} | |")
    for r in rows:
        minutes = r["seconds"] / 60 if isinstance(r.get("seconds"), (int, float)) else None
        lines.append(f"| {r['run']} | {r['status']} | {fmt(r.get('ap'), '.4f')} | {fmt(r.get('precision_top10'), '.4f')}"
                     f" | {fmt(r.get('spearman'), '.4f')} | {fmt(minutes, '.0f')} |")
    done = {}
    for r in rows:
        if isinstance(r.get("ap"), (int, float)):
            done.setdefault(r["model"], []).append(r)
    means = [(m, rs) for m, rs in done.items() if len(rs) > 1]
    if means:
        lines += ["", "| Model (mean over seeds) | Seeds | Val AP | Top-10% hit |", "|---|---:|---:|---:|"]
        for model, rs in means:
            ap = sum(r["ap"] for r in rs) / len(rs)
            top = sum(r["precision_top10"] for r in rs) / len(rs)
            lines.append(f"| {model} | {len(rs)} | {ap:.4f} | {top:.4f} |")
    if champion:
        lines += ["", "Advisory only: experiments are chosen on validation. The gate that blocks `release` scores the "
                  "chosen model on test against the champion."]
    else:
        lines += ["", "No champion is deployed yet, so there is nothing to compare with."]
    lines += ["", f"Files: `{MODEL_REPO}` → `{hf_prefix(cfg, commit)}/`"]
    return "\n".join(lines) + "\n"


def score_champion(dataset, revision, out) -> dict | None:
    """The deployed model's validation scores, or None if nothing is deployed."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import hf_models

    with tempfile.TemporaryDirectory() as tmp:
        champ = hf_models.fetch_champion(Path(tmp))
        if not champ:
            return None
        result = out / "champion_val.json"
        subprocess.run([sys.executable, "-m", "pipeline.evaluate", "--model", str(Path(tmp) / "model"), "--split", "val",
                        "--dataset", dataset, "--revision", revision, "--out", str(result)], check=True, cwd=ROOT)
    return {"name": champ["name"], "version": champ["version"], **json.loads(result.read_text())}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", nargs="+", metavar="CONFIG", help="only validate these configs")
    p.add_argument("--config")
    p.add_argument("--dataset")
    p.add_argument("--revision")
    p.add_argument("--commit", help="git commit of the code that runs (names the results folder)")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)

    if args.check:
        ok = True
        for path in args.check:
            try:
                cfg = load_config(path)
                print(f"{path}: {len(plan(cfg))} runs with {cfg['runner']}")
            except ConfigError as e:
                print(f"::error::{e}", file=sys.stderr)
                ok = False
        return 0 if ok else 1
    if not all([args.config, args.dataset, args.revision, args.commit, args.out]):
        p.error("--config, --dataset, --revision, --commit and --out are required (or use --check)")

    from huggingface_hub import HfApi, hf_hub_download

    cfg = load_config(args.config)
    api, prefix = HfApi(), hf_prefix(cfg, args.commit)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, seed in plan(cfg):
        run, row = run_dir_name(model, seed), {"model": model, "seed": seed}
        row["run"] = run
        remote = f"{prefix}/{run}"
        if api.file_exists(MODEL_REPO, f"{remote}/best.json"):
            row.update(json.loads(Path(hf_hub_download(MODEL_REPO, f"{remote}/best.json")).read_text()))
            rows.append({**row, "status": "already done"})
            print(f"{run}: already on Hugging Face, skipping", flush=True)
            continue
        start = time.monotonic()
        print(f"{run}: training", flush=True)
        done = subprocess.run(runner_command(cfg, model, seed, args.out, args.dataset, args.revision), cwd=ROOT)
        best = args.out / "runs" / run / "best.json"
        if done.returncode != 0 or not best.exists():
            rows.append({**row, "status": f"failed (exit {done.returncode})"})
            continue
        row.update(json.loads(best.read_text()))
        (args.out / "runs" / run / "experiment.json").write_text(json.dumps(
            {"config": cfg["name"], "commit": args.commit, "dataset": args.dataset, "revision": args.revision,
             "runner": cfg["runner"], "model": model, "seed": seed, "epochs": cfg["epochs"]}, indent=2))
        api.upload_folder(repo_id=MODEL_REPO, folder_path=args.out / "runs" / run, path_in_repo=remote,
                          ignore_patterns=["resume.pt", "best_checkpoint.pt"], commit_message=f"Experiment {remote}")
        rows.append({**row, "status": "done"})
        print(f"{run}: done in {(time.monotonic() - start) / 60:.0f} min, uploaded to {remote}", flush=True)

    try:
        champion = score_champion(args.dataset, args.revision, args.out)
    except Exception as e:  # the runs are already stored; a champion problem shouldn't hide them
        print(f"::warning::could not score the champion on validation: {e}", file=sys.stderr)
        champion = None
    (args.out / "summary.json").write_text(json.dumps({"config": cfg["name"], "commit": args.commit,
                                                       "champion": champion, "runs": rows}, indent=2))
    (args.out / "summary.md").write_text(summarize(cfg, args.commit, rows, champion))
    print((args.out / "summary.md").read_text())
    return 1 if any(r["status"].startswith("failed") for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
