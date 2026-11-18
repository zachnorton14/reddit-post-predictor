"""Evaluate or score fixed averages of saved r/dadjokes DeBERTa checkpoints.

    python -m scripts.deberta_ensemble
    python predict_prototypes.py features.parquet scores.parquet --model large3

Input to predict is the same posting-time feature table as predict_prototypes.py.
All members return log-engagement scores on the original scale. Each gets one vote.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

import dadjokes_models as dm
import textonly_data as td

MODEL_REPO = "zachnorton03/reddit-post-predictor-models"
MODEL_REVISION = "cb322b7ed38f7142eb4a609d158a05b713c4a146"
DATASET = "zachnorton03/textonly-subreddit-predictors"
DATA_REVISION = "e02c2b6236cbc51df27b4faf991fec836ed7fe08"
LADDER = "experiments/deberta_ladder/44db5e4"
SWEEP = "experiments/deberta_large_seeds/5e83692"
LARGE = (f"{LADDER}/deberta_large_attn4_s42", f"{SWEEP}/deberta_large_attn4_s43", f"{SWEEP}/deberta_large_attn4_s44")
MODELS = {"large3": LARGE, "large3_base": (*LARGE, f"{LADDER}/deberta_base_s42")}
CACHE = Path("data/deberta_ensemble")


def checkpoint_files(member, files):
    root = snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=CACHE / MODEL_REVISION,
                             allow_patterns=[f"{member}/{file}" for file in files])
    return Path(root) / member


def validation_frame():
    root = snapshot_download(DATASET, repo_type="dataset", revision=DATA_REVISION, local_dir=CACHE / DATA_REVISION,
                             allow_patterns="data/dadjokes/2024/*.parquet")
    with td.connect((Path(root) / "data/dadjokes/2024/*.parquet").as_posix(), history=False) as con:
        td.add_labels(con)
        return con.sql("SELECT id, created_at, label, metric_pct FROM labeled ORDER BY created_at, id").df()


def aligned_predictions(frame, rows, scores):
    if scores.shape != (len(rows),) or not np.isfinite(scores).all():
        raise ValueError("Invalid saved validation predictions")
    by_id = rows.assign(score=scores).set_index("id")
    if not frame.id.is_unique or not by_id.index.is_unique or len(by_id) != len(frame) or not frame.id.isin(by_id.index).all():
        raise ValueError("Validation post IDs differ between checkpoint and dataset")
    aligned = by_id.loc[frame.id]
    if not np.array_equal(aligned.created_at.to_numpy(), frame.created_at.to_numpy()):
        raise ValueError("Validation timestamps differ between checkpoint and dataset")
    return aligned.score.to_numpy()


def mean_scores(scores):
    # Use the same arithmetic for cached validation predictions and live inference.
    return np.mean(np.stack(scores), axis=0, dtype=np.float64)


def evaluate():
    frame = validation_frame()
    if frame.empty or not frame.created_at.dt.year.eq(2024).all():
        raise ValueError("Expected only 2024 validation posts")
    scores = {}
    for member in MODELS["large3_base"]:
        path = checkpoint_files(member, ["val_predictions.npy", "val_rows.parquet", "experiment.json"])
        provenance = json.loads((path / "experiment.json").read_text())
        if (provenance["dataset"], provenance["revision"]) != (DATASET, DATA_REVISION):
            raise ValueError(f"Different dataset revision for {member}")
        scores[member] = aligned_predictions(frame, pd.read_parquet(path / "val_rows.parquet"),
                                             np.load(path / "val_predictions.npy"))
    for name, members in MODELS.items():
        scores[name] = mean_scores([scores[member] for member in members])

    rows = []
    halves = frame.created_at.dt.month.le(6)
    for period, mask in [("2024", np.ones(len(frame), dtype=bool)), ("Jan-Jun", halves), ("Jul-Dec", ~halves)]:
        for name, pred in scores.items():
            rows.append({"model": name.split("/")[-1], "period": period, "posts": int(mask.sum()),
                         **dm.scores(frame.loc[mask], pred[mask])})
    return pd.DataFrame(rows)


def predict(frame, model):
    from prototype_models import predict_checkpoint

    scores = []
    for member in MODELS[model]:
        path = checkpoint_files(member, ["spec.json", "encoder/*", "tokenizer/*", "preprocessing.npz", "head.pt", "extra.pt"])
        scores.append(predict_checkpoint(path, frame))
    return mean_scores(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/deberta_ensemble/validation.csv"))
    args = parser.parse_args()
    table = evaluate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
