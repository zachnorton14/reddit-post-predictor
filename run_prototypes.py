"""Train two architectures, freeze validation choices, then report historical test.

Usage: python run_prototypes.py --report
Small integration run: python run_prototypes.py --output data/prototypes/smoke
    --max-train 256 --max-val 128 --epochs 1 --models interaction
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import prototype_models as pm
from training_recovery import keep_system_awake

LEGACY_RUNNER_HASHES = {"973b555ae99f414bb89071a96180a3a53cdddd6cb7873d06a93b58bd26effa98"}

ROOT = Path(__file__).resolve().parent
SPECS = {
    "interaction": pm.Spec(name="interaction", architecture="interaction", max_length=64,
                           hidden=128, body_chars=None, batch_size=64, accumulation=2),
    "deberta": pm.Spec(name="deberta", source="data/pretrained/deberta", hidden=128,
                       batch_size=32, accumulation=4, learning_rate=2e-5, checkpointing=False,
                       use_fast_tokenizer=False),
}


def load_data(path=None):
    """Reuse the explicitly manifested full-body extract, or rebuild from raw shards."""
    if path is None:
        candidates = []
        for manifest in (ROOT / "data/models").glob("dadjokes_*.json"):
            cfg = json.loads(manifest.read_text()).get("config", {})
            if (cfg.get("body_chars", 1000) is None and cfg.get("train_end") == "2024-01-01"
                    and cfg.get("val_end") == "2025-01-01" and manifest.with_suffix(".parquet").exists()):
                candidates.append(manifest.with_suffix(".parquet"))
        if len(candidates) > 1:
            raise ValueError("Multiple full-body caches: specify --data explicitly")
        path = candidates[0] if candidates else None
    if path is not None:
        frame = pd.read_parquet(path)
    else:
        import textonly_baselines as tb
        cfg = tb.Config(source=str(ROOT / "data/textonly/data/dadjokes/*/*.parquet"),
                        body_chars=None, max_train_per_sub=10**9, max_eval_per_sub=10**9)
        data = tb.load_data(cfg)
        frame = pd.concat([getattr(data, s).assign(split=s) for s in ("train", "val", "test")])
        data.con.close()
    required = {"id", "title", "body", "created_at", "label", "metric", "metric_pct", *pm.META}
    if required - set(frame):
        raise ValueError(f"Missing dataset columns: {required - set(frame)}")
    if frame.id.duplicated().any():
        raise ValueError("Duplicate post IDs")
    parts = {s: frame[frame.split == s].sort_values("created_at").reset_index(drop=True)
             for s in ("train", "val", "test")}
    if not (parts["train"].created_at.max() < pd.Timestamp("2024-01-01")
            <= parts["val"].created_at.min() <= parts["val"].created_at.max() < pd.Timestamp("2025-01-01")
            <= parts["test"].created_at.min()):
        raise ValueError("Dataset does not have the intended chronological splits")
    return parts


def reference_predictions(reference, val, verify=True):
    """Read saved validation scores; verify row alignment against checkpoint inference."""
    selected = json.loads((reference / "selection.json").read_text())
    # Resolve by basename so a copied checkout does not depend on its former absolute path.
    paths = [reference / "runs" / Path(p).name for p in selected["finalists"]]
    neural = []
    for path in paths:
        scores = np.load(path / "val_predictions.npy")
        if len(scores) != len(val):
            raise ValueError("Reference validation rows differ")
        if verify:
            indices = np.linspace(0, len(val) - 1, 128).astype(int)
            reproduced = pm.predict_checkpoint(path, val.iloc[indices], legacy=True)
            # BF16 vs the legacy FP16 run introduces small numerical differences.
            if not np.allclose(scores[indices], reproduced, atol=.08, rtol=.01):
                raise ValueError("Reference predictions do not match current row order/checkpoint")
        neural.append(scores)
    ensemble = np.mean(neural, axis=0)
    hgb = np.load(reference / "classical/predictions.npz")["hgb_val"]
    blend = joblib.load(reference / "classical/finalist_blend.joblib")
    incumbent = blend.predict(np.column_stack([ensemble, hgb]))
    return incumbent, ensemble, hgb


def select_pipeline(val, incumbent, ensemble, hgb, prototypes):
    """First-half blend tuning, second-half gate, all frozen before test scoring."""
    fit = (val.created_at < pd.Timestamp("2024-07-01")).to_numpy()
    gate = ~fit
    if min(fit.sum(), gate.sum()) < 100 or min(val.label[fit].sum(), val.label[gate].sum()) < 10:
        raise ValueError("Need both validation halves with enough positive examples")
    y = val.label.to_numpy()
    neural_ref, hgb_ref = ensemble[fit], hgb[fit]
    neural_rank = pm.empirical_ranks(neural_ref, ensemble)
    hgb_rank = pm.empirical_ranks(hgb_ref, hgb)
    weight, trials = pm.choose_blend(y[fit], neural_rank[fit], hgb_rank[fit], np.arange(0, .501, .05))
    reweighted = (1 - weight) * neural_rank + weight * hgb_rank
    scores = {"incumbent": incumbent, "ranking_blend": reweighted}
    specs = {"ranking_blend": {"weight_hgb": weight, "trials": trials}}
    # Use the same blend search for both prototypes, without test-based retries.
    base_rank = pm.empirical_ranks(incumbent[fit], incumbent)
    for name, pv in prototypes.items():
        scores[name] = pv
        rank = pm.empirical_ranks(pv[fit], pv)
        w, history = pm.choose_blend(y[fit], base_rank[fit], rank[fit])
        scores[name + "_blend"] = (1 - w) * base_rank + w * rank
        specs[name + "_blend"] = {"weight_prototype": w, "trials": history}
    gate_ap = {name: float(average_precision_score(y[gate], p[gate])) for name, p in scores.items()}
    best = max(gate_ap, key=gate_ap.get)
    # Require a predeclared material validation gain before replacing the incumbent.
    chosen = best if gate_ap[best] >= gate_ap["incumbent"] + .002 else "incumbent"
    selection = {"selected": chosen, "best_gate_candidate": best, "gate_ap": gate_ap,
                 "minimum_gate_ap_gain": .002, "blend_specs": specs,
                 "fit_posts": int(fit.sum()), "gate_posts": int(gate.sum()),
                 "rule": "Tune weights Jan-Jun 2024, choose by Jul-Dec AP, require +0.002 AP; no test selection"}
    refs = dict(neural=neural_ref, hgb=hgb_ref, incumbent=incumbent[fit],
                **{name: p[fit] for name, p in prototypes.items()})
    return selection, scores, refs


def apply_pipeline(selection, refs, incumbent, ensemble, hgb, prototypes):
    weight = selection["blend_specs"]["ranking_blend"]["weight_hgb"]
    result = {"incumbent": incumbent,
              "ranking_blend": (1 - weight) * pm.empirical_ranks(refs["neural"], ensemble)
                                + weight * pm.empirical_ranks(refs["hgb"], hgb)}
    for name, pred in prototypes.items():
        result[name] = pred
        w = selection["blend_specs"][name + "_blend"]["weight_prototype"]
        result[name + "_blend"] = ((1 - w) * pm.empirical_ranks(refs["incumbent"], incumbent)
                                   + w * pm.empirical_ranks(refs[name], pred))
    return result


def report(output, reference, data, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    val, test = data["val"], data["test"]
    incumbent, ensemble, hgb = reference_predictions(reference, val)
    proto_val = {name: np.load(path / "val_predictions.npy") for name, path in paths.items()}
    for name, path in paths.items():
        rows = pd.read_parquet(path / "val_rows.parquet")
        if not np.array_equal(rows.id, val.id):
            raise ValueError(f"Validation row mismatch for {name}")
    selection, val_scores, refs = select_pipeline(val, incumbent, ensemble, hgb, proto_val)
    # Freeze decisions before loading any historical test scores or evaluating test labels.
    selection_path = output / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Frozen selection changed; use a new experiment output directory")
    pm.write_json(selection_path, selection)
    np.savez(output / "rank_references.npz", **refs)
    print("Frozen validation selection:", selection["selected"], selection["gate_ap"], flush=True)
    original = pd.read_parquet(reference / "test_predictions.parquet")
    if not np.array_equal(original.id, test.id) or not np.array_equal(original.label, test.label):
        raise ValueError("Historical test predictions are not aligned with the dataset")
    proto_test = {}
    for name, path in paths.items():
        cache = path / "test_predictions.npz"
        if cache.exists():
            loaded = np.load(cache)
            if str(loaded["fingerprint"]) != pm.fingerprint(test):
                raise ValueError("Cached test rows changed")
            proto_test[name] = loaded["predictions"]
        else:
            print(f"Evaluating frozen {name} checkpoint", flush=True)
            proto_test[name] = pm.predict_checkpoint(path, test)
            np.savez(cache, predictions=proto_test[name], fingerprint=pm.fingerprint(test))
    test_scores = apply_pipeline(selection, refs, original.finalist_metadata_blend.to_numpy(),
                                 original.finalist_ensemble.to_numpy(), original.metadata_hgb.to_numpy(), proto_test)
    destination = ROOT / "results/prototypes" / output.name
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, pred in test_scores.items():
        rows.append(dict(model=name, **{f"val_{k}": v for k, v in pm.metrics(val, val_scores[name]).items()},
                         gate_ap=selection["gate_ap"][name], **{f"test_{k}": v for k, v in pm.metrics(test, pred).items()}))
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(destination / "metrics.csv")
    intervals = {name: pm.bootstrap_delta(test, test_scores["incumbent"], pred)
                 for name, pred in test_scores.items() if name != "incumbent"}
    pm.write_json(destination / "uncertainty.json", intervals)
    pm.write_json(destination / "selection.json", selection)
    manifest = {"protocol": json.loads((output / "protocol.json").read_text()),
                "execution": json.loads((output / "execution.json").read_text())
                if (output / "execution.json").exists() else None,
                "runs": {name: {"identity": json.loads((p / "identity.json").read_text()),
                                "complete": json.loads((p / "complete.json").read_text()),
                                "best": json.loads((p / "best.json").read_text())} for name, p in paths.items()}}
    pm.write_json(destination / "run_manifest.json", manifest)
    pd.DataFrame({"id": test.id, "created_at": test.created_at, "label": test.label,
                  **test_scores}).to_parquet(output / "test_predictions.parquet", index=False)
    names = {"incumbent": "Previous ensemble + HGB", "ranking_blend": "Reweighted ranking blend",
             "interaction": "Setup / punchline MiniLM", "deberta": "DeBERTa + metadata",
             "interaction_blend": "Previous + setup / punchline", "deberta_blend": "Previous + DeBERTa"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    labels = [names[name] for name in table.index]
    for ax, col, title in zip(axes, ["test_ap", "test_precision_top10"], ["Test average precision", "Success rate in top-ranked 10%"]):
        bars = ax.barh(labels, table[col], color=["#187a67" if name == selection["selected"] else "#6683ad" for name in table.index])
        ax.invert_yaxis()
        ax.set_xlim(0, max(table[col]) * 1.2)
        ax.set_title(title)
        ax.bar_label(bars, fmt="%.3f", padding=3)
    fig.suptitle("Prototype comparison | selected on validation")
    fig.savefig(destination / "comparison.png", dpi=180)
    fig.savefig(destination / "comparison.svg")
    plt.close(fig)
    lines = ["# Neural prototypes", "", f"Validation-selected pipeline: **{names[selection['selected']]}**.", "",
             "Train: 2019–2023; validation: 2024; historical test: Jan 2025–Feb 2026.", "",
             "| Model | Validation AP | Jul–Dec gate AP | Test AP | Test top-10% success |",
             "|---|---:|---:|---:|---:|"]
    for name, row in table.iterrows():
        lines.append(f"| {names[name]} | {row.val_ap:.4f} | {row.gate_ap:.4f} | {row.test_ap:.4f} | {row.test_precision_top10:.1%} |")
    lines += ["", "AP is average precision (called PR-AUC elsewhere in the repo). Top-10% success is precision among the highest-ranked tenth, not accuracy.",
              "", "New prototypes use single-seed runs; the reference averages three seeds. Neural checkpoints use full-2024 validation AP. Blend weights use Jan–Jun only; Jul–Dec gates replacement at +0.002 AP. That gate is not untouched because neural checkpoints and the old reference already used 2024.",
              "", "Test has been inspected in earlier project work. These are exploratory historical comparisons, not a fresh prospective benchmark. Paired week-bootstrap intervals in uncertainty.json do not include model-selection uncertainty.",
              "", "A candidate is not promoted because it happens to have the best historical test score. Read selection.json for the frozen validation decision."]
    (destination / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table.round(4).to_string(), flush=True)
    print(f"Report: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/prototypes/full"))
    parser.add_argument("--reference", type=Path, default=Path("data/improvements/full"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--models", nargs="+", choices=list(SPECS), default=list(SPECS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4, help="CPU threads feeding the GPU")
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Recover an interrupted run; legacy epochs reset optimizer")
    parser.add_argument("--checkpoint-seconds", type=float, default=300, help="Save full training state about this often")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.checkpoint_seconds <= 0:
        parser.error("--checkpoint-seconds must be positive")
    pm.torch.set_num_threads(args.threads)
    if (args.report or args.report_only) and (args.max_train or args.max_val):
        parser.error("Sampled smoke runs cannot produce the full-data comparison")
    data = load_data(args.data)
    train = data["train"][data["train"].created_at >= pd.Timestamp("2019-01-01")].reset_index(drop=True)
    val = data["val"]
    if args.max_train:
        train = train.sample(min(args.max_train, len(train)), random_state=42).sort_values("created_at").reset_index(drop=True)
    if args.max_val:
        val = val.sample(min(args.max_val, len(val)), random_state=42).sort_values("created_at").reset_index(drop=True)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"train_posts": len(train), "val_posts": len(val), "train_fingerprint": pm.fingerprint(train),
                "val_fingerprint": pm.fingerprint(val), "test_fingerprint": pm.fingerprint(data["test"]),
                "epochs": args.epochs, "models": args.models, "seed": 42, "cpu_threads": args.threads,
                "test_status": "Previously inspected historical test; exploratory only",
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        if old != protocol:
            # Retain the original run's provenance; only this audited recovery-only
            # upgrade may reuse its checkpoints under a different runner hash.
            same_protocol = {k: v for k, v in old.items() if k != "runner_sha256"} == {
                k: v for k, v in protocol.items() if k != "runner_sha256"}
            if not ((args.resume or args.report_only) and same_protocol
                    and old.get("runner_sha256") in LEGACY_RUNNER_HASHES):
                raise ValueError("Protocol changed: use a new output directory")
    else:
        pm.write_json(protocol_path, protocol)
    pm.write_json(args.output / "execution.json", {"runner_sha256": protocol["runner_sha256"],
                  "resume": args.resume, "checkpoint_seconds": args.checkpoint_seconds})
    print({"train": len(train), "val": len(val), "models": args.models}, flush=True)
    paths = {}
    for name in args.models:
        path = args.output / name
        spec = replace(SPECS[name], epochs=args.epochs, source=str(ROOT / SPECS[name].source))
        if not args.report_only:
            with keep_system_awake():
                pm.train(spec, train, val, path, resume=args.resume, checkpoint_seconds=args.checkpoint_seconds)
        if not (path / "complete.json").exists():
            raise ValueError(f"Missing completed run: {path}")
        paths[name] = path
    if args.report or args.report_only:
        with keep_system_awake():
            report(args.output, args.reference, data, paths)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "CUDA" in str(exc):
            print("CUDA failed. End this process and rerun the same command with --resume. "
                  "The last successfully saved recovery snapshot is retained. "
                  "Keep the laptop plugged in and its lid open.", flush=True)
        raise
