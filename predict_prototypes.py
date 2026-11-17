"""Score derived posting-time features with a saved prototype or selected pipeline.

Input is a Parquet file containing title, cleaned body and dadjokes_models.META.
Scores rank posts; they are not calibrated probabilities of success.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dadjokes_models import CATEGORICAL, META
import prototype_models as pm
from scripts import deberta_ensemble


def score_frame(frame, run=Path("data/prototypes/full"), reference=Path("data/improvements/full"), model="selected"):
    run, reference = Path(run), Path(reference)
    required = {"title", "body", *META}
    if required - set(frame.columns):
        raise ValueError(f"Missing posting-time features: {sorted(required - set(frame.columns))}")
    if not np.isfinite(frame[META].to_numpy(dtype=np.float32)).all():
        raise ValueError("Posting-time metadata must be finite")
    if frame.empty:
        return np.array([], dtype=np.float32)
    selection = json.loads((run / "selection.json").read_text()) if model == "selected" else None
    if selection:
        model = selection["selected"]
    if model in deberta_ensemble.MODELS:
        return deberta_ensemble.predict(frame, model)
    if model in ("interaction", "deberta"):
        if not (run / model / "complete.json").exists():
            raise ValueError(f"Missing completed {model} checkpoint")
        return pm.predict_checkpoint(run / model, frame)
    if model not in ("incumbent", "ranking_blend", "interaction_blend", "deberta_blend"):
        raise ValueError(f"Unknown pipeline: {model}")
    selected_reference = json.loads((reference / "selection.json").read_text())
    paths = [reference / "runs" / Path(p).name for p in selected_reference["finalists"]]
    ensemble = np.mean([pm.predict_checkpoint(p, frame, legacy=True) for p in paths], axis=0)
    metadata = frame[META].copy()
    for name in CATEGORICAL:
        metadata[name] = metadata[name].astype("category")
    hgb = joblib.load(reference / "classical/models.joblib")["hgb"].predict(metadata)
    incumbent = joblib.load(reference / "classical/finalist_blend.joblib").predict(np.column_stack((ensemble, hgb)))
    if model == "incumbent":
        return incumbent
    selection = selection or json.loads((run / "selection.json").read_text())
    refs = np.load(run / "rank_references.npz")
    if model == "ranking_blend":
        weight = selection["blend_specs"][model]["weight_hgb"]
        return ((1 - weight) * pm.empirical_ranks(refs["neural"], ensemble)
                + weight * pm.empirical_ranks(refs["hgb"], hgb))
    name = model.removesuffix("_blend")
    pred = pm.predict_checkpoint(run / name, frame)
    weight = selection["blend_specs"][model]["weight_prototype"]
    return ((1 - weight) * pm.empirical_ranks(refs["incumbent"], incumbent)
            + weight * pm.empirical_ranks(refs[name], pred))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", type=Path, default=Path("data/prototypes/full"))
    parser.add_argument("--reference", type=Path, default=Path("data/improvements/full"))
    parser.add_argument("--model", default="selected", choices=["selected", "interaction", "deberta", "incumbent",
                        "ranking_blend", "interaction_blend", "deberta_blend", *deberta_ensemble.MODELS])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    frame = pd.read_parquet(args.input)
    prediction = score_frame(frame, args.run, args.reference, args.model)
    result = frame[[column for column in ("id", "created_at", "title") if column in frame]].copy()
    result["ranking_score"] = prediction
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(f"Scored {len(result):,} posts -> {args.output}")


if __name__ == "__main__":
    main()
