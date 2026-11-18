"""Ladder of input, timing and pooling changes to the prototype regression network, compared on 2024 validation.

    python run_pooling.py --models minilm_base minilm_long minilm_time minilm_attn1 minilm_attn4 minilm_attn10
    python run_pooling.py --models deberta_large_attn4 --seeds 42 43 44      # on a CUDA GPU
    python run_pooling.py --output data/pooling/smoke --max-train 512 --max-val 256 --epochs 1 --models minilm_attn4

The experiment workflow (.github/workflows/experiment.yml) calls it once per run on pinned data:
    python run_pooling.py --models <model> --seeds <seed> --output /out/runs --results /out/results --resume
        --dataset <hf dataset repo> --revision <sha> [--epochs N]

Each rung changes one thing from the one before (see LADDER). Train 2019-2023, validation 2024, checkpoints and
choices by validation AP. Test is never loaded or scored here. Runs go to data/pooling/<model>_s<seed>/ (resumable);
the validation table goes to results/pooling/val_metrics.csv.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

import prototype_models as pm
from run_prototypes import load_data

ROOT = Path(__file__).resolve().parent
PRETRAINED = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "deberta": "microsoft/deberta-v3-base",
              "deberta_large": "microsoft/deberta-v3-large"}
CACHE = ROOT / "data/models/dadjokes_fullbody.parquet"

_minilm = pm.Spec(name="minilm_base", source="data/pretrained/minilm", hidden=128, batch_size=64, accumulation=2,
                  learning_rate=3e-5)
_long = dict(max_length=256, body_chars=None)
_deberta = dict(source="data/pretrained/deberta", batch_size=32, accumulation=4, learning_rate=2e-5,
                use_fast_tokenizer=False)
_large = dict(source="data/pretrained/deberta_large", batch_size=16, accumulation=8, learning_rate=1e-5,
              use_fast_tokenizer=False, checkpointing=True)
LADDER = {
    # MiniLM ladder: each rung adds one change to the previous rung
    "minilm_base": _minilm,                                                       # Owen's DeBERTa recipe on MiniLM
    "minilm_long": replace(_minilm, name="minilm_long", **_long),                 # + 256 tokens, full body
    "minilm_time": replace(_minilm, name="minilm_time", **_long, time_embed=True),  # + timing embeddings
    **{f"minilm_attn{k}": replace(_minilm, name=f"minilm_attn{k}", **_long, time_embed=True, pooling="attention",
                                  queries=k) for k in (1, 4, 10)},                # + attention pooling, k queries
    # Scaled encoders with the same head options
    "deberta_base": replace(_minilm, name="deberta_base", **_deberta),            # = Owen's "deberta" spec
    **{f"deberta_attn{k}": replace(_minilm, name=f"deberta_attn{k}", **_deberta, **_long, time_embed=True,
                                   pooling="attention", queries=k) for k in (1, 4, 10)},
    **{f"deberta_large_attn{k}": replace(_minilm, name=f"deberta_large_attn{k}", **_large, **_long, time_embed=True,
                                         pooling="attention", queries=k) for k in (1, 4, 10)},
    "deberta_large_mean": replace(_minilm, name="deberta_large_mean", **_large, **_long, time_embed=True),
}
# Same seed, optimizer, metadata and pooling as each joint control. The shared encoder reads
# the title and body separately and the interaction head compares the two representations.
for control in ("deberta_attn4", "deberta_large_attn4"):
    name = f"{control}_interaction"
    LADDER[name] = replace(LADDER[control], name=name, architecture="interaction")


def ensure_pretrained(spec):
    """Download the base encoder into spec.source once (the trainer loads local files only)."""
    path = ROOT / spec.source
    if not (path / "config.json").exists():
        from huggingface_hub import snapshot_download
        repo = PRETRAINED[Path(spec.source).name]
        snapshot_download(repo, local_dir=path, allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors",
                                                                 "pytorch_model.bin"])
    return path


def data_splits(dataset=None, revision=None):
    """Full-body labeled dadjokes splits.

    With `dataset` and `revision`: from that pinned Hugging Face revision, never from a local cache (the experiment
    workflow). Without: from the local extract, cached once as Parquet with the manifest load_data looks for.
    """
    if dataset:
        from pipeline.data import load_pinned

        data = load_pinned(dataset, revision, body_chars=None, validation_only=True)
        if pd.concat([data.train.id, data.val.id]).duplicated().any():
            raise ValueError("Duplicate post IDs in the experiment data")
        return {s: getattr(data, s).sort_values(["created_at", "id"]).reset_index(drop=True) for s in ("train", "val")}
    if not CACHE.exists():
        parts = load_data()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([p.assign(split=s) for s, p in parts.items()]).to_parquet(CACHE, index=False)
        pm.write_json(CACHE.with_suffix(".json"), {"config": {"body_chars": None, "train_end": "2024-01-01",
                                                              "val_end": "2025-01-01"}})
    return load_data(CACHE)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, choices=list(LADDER))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output", type=Path, default=ROOT / "data/pooling")
    parser.add_argument("--results", type=Path, default=ROOT / "results/pooling")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train", type=int, help="small subset runs only")
    parser.add_argument("--max-val", type=int, help="small subset runs only")
    parser.add_argument("--resume", action="store_true", help="continue interrupted runs from their snapshots")
    parser.add_argument("--dataset", help="Hugging Face dataset repo; load this pinned revision instead of local data")
    parser.add_argument("--revision", help="dataset commit (required with --dataset)")
    args = parser.parse_args(argv)
    if bool(args.dataset) != bool(args.revision):
        parser.error("--dataset and --revision go together")

    data = data_splits(args.dataset, args.revision)
    train = data["train"][data["train"].created_at >= pd.Timestamp("2019-01-01")].reset_index(drop=True)
    val = data["val"]
    print(f"full splits: {len(train):,} train posts (2019-2023), {len(val):,} validation posts", flush=True)
    if args.max_train:
        train = train.sample(min(args.max_train, len(train)), random_state=42).sort_values("created_at") \
            .reset_index(drop=True)
    if args.max_val:
        val = val.sample(min(args.max_val, len(val)), random_state=42).sort_values("created_at").reset_index(drop=True)
    print(f"train {len(train):,} posts (2019-2023), validation {len(val):,} posts (2024)", flush=True)

    rows = []
    for name in args.models:
        for seed in args.seeds:
            spec = replace(LADDER[name], seed=seed, **({"epochs": args.epochs} if args.epochs else {}))
            ensure_pretrained(spec)
            out = pm.train(spec, train, val, args.output / f"{name}_s{seed}", resume=args.resume)
            best = pd.read_json(out / "best.json", typ="series")
            rows.append({"model": name, "seed": seed, "train_posts": len(train), "val_posts": len(val),
                         **best.to_dict()})
            print(f"{name} seed {seed}: best epoch {best.epoch}, val AP {best.ap:.4f}, "
                  f"top-10% {best.precision_top10:.4f}", flush=True)
    table = pd.DataFrame(rows)
    args.results.mkdir(parents=True, exist_ok=True)
    path = args.results / "val_metrics.csv"
    if path.exists():  # keep earlier rows; a rerun of the same model and seed replaces its row
        old = pd.read_csv(path)
        old = old[~old.set_index(["model", "seed"]).index.isin(table.set_index(["model", "seed"]).index)]
        table = pd.concat([old, table], ignore_index=True)
    table.to_csv(path, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
