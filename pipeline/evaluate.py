"""Score a saved model on every subreddit's test split (or, with --split val, validation) of a pinned dataset revision.

    python -m pipeline.evaluate --model <dir> --dataset <hf dataset repo> --revision <sha> --out <file> [--split val]

The gate uses test. Experiments (pipeline/experiment.py) use val, so choosing among them never looks at test.

Loads any past model by the `type` in <dir>/model.json, so the deployed champion can be scored with this code.
`pr_auc`, `top10_hit_rate` and `spearman` are macro averages: each subreddit with at least MIN_POSTS posts counts
equally, so AskReddit doesn't drown out the small ones. `by_subreddit` has each one's scores.

Never writes into <dir> (the champion is mounted read-only). The metrics include `rows`, a hash of the scored post
ids, so the gate can check that two models were scored on the same posts.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

import dadjokes_models as dm
from pipeline.data import load_eval


def predict_minilm(model_dir, meta, data):
    import torch
    from transformers import AutoTokenizer

    net = dm.regression_net(meta["base_model"])
    net.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    net.to(dm.device())
    tok = AutoTokenizer.from_pretrained(model_dir)
    return dm.predict_net(net, tok, dm.text_of(data.test), meta["max_len"], meta["mu"], meta["sd"])


def predict_prototype(model_dir, meta, data):
    """A run saved by prototype_models.train (experiment runners such as run_pooling.py): encoder/, head.pt, spec.json."""
    import prototype_models as pm

    return pm.predict_checkpoint(model_dir, data.test)


# model.json type -> loader(model_dir, meta, data) returning test predictions. Keep the deployed champion's type.
LOADERS = {"minilm": predict_minilm, "prototype": predict_prototype}
# Body characters each type reads (default 1000). Prototype runs read the full body and cut it themselves (spec.json).
BODY_CHARS = {"prototype": None}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="model directory with model.json")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset repo")
    parser.add_argument("--revision", required=True, help="dataset commit")
    parser.add_argument("--out", required=True, type=Path, help="metrics JSON file")
    parser.add_argument("--split", choices=["test", "val"], default="test", help="split to score (default test)")
    return parser.parse_args(argv)


def rows_hash(frame) -> str | None:
    if "id" not in frame:
        return None
    return hashlib.sha256("\n".join(sorted(frame.id.astype(str))).encode()).hexdigest()[:16]


MIN_POSTS = 1000  # fewer scored posts (about 100 top-10% posts) is too noisy to count in the macro average
METRICS = ("pr_auc", "top10_hit_rate", "spearman")


def scores(frame, p):
    s = dm.scores(frame, p)
    return {"pr_auc": float(s["pr_auc"]), "top10_hit_rate": float(s["top10_hit"]), "spearman": float(s["spearman"]),
            "n": len(frame)}


def metrics(frame, p, split="test"):
    p = np.asarray(p)
    by_sub = {sub: scores(frame[mask], p[mask])
              for sub in sorted(frame.subreddit.unique()) if (mask := (frame.subreddit == sub).to_numpy()).any()}
    counted = [sub for sub, s in by_sub.items() if s["n"] >= MIN_POSTS]
    if not counted:
        raise SystemExit(f"no subreddit has {MIN_POSTS} or more {split} posts")
    # compare_metrics.py gates on the macro pr_auc and top10_hit_rate and checks that `rows` match
    result = {m: float(np.mean([by_sub[sub][m] for sub in counted])) for m in METRICS}
    result |= {f"n_{split}": len(frame), "macro_subreddits": counted, "by_subreddit": by_sub}
    if rows_hash(frame):
        result["rows"] = rows_hash(frame)
    return result


def main(argv=None):
    args = parse_args(argv)
    meta = json.loads((args.model / "model.json").read_text())
    if meta.get("type") not in LOADERS:
        raise SystemExit(f"unknown model type {meta.get('type')!r} in {args.model}/model.json; "
                         f"known types: {', '.join(LOADERS)}")
    data = load_eval(args.dataset, args.revision, body_chars=BODY_CHARS.get(meta["type"], 1000))
    if args.split == "val":
        # Loaders predict data.test; point it at validation so the loaders need no split argument
        data = copy.copy(data)
        data.test = data.val
    result = metrics(data.test, LOADERS[meta["type"]](args.model, meta, data), args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "by_subreddit"}))
    for sub, sc in result["by_subreddit"].items():
        print(f"  {sub:24} n={sc['n']:6} pr_auc={sc['pr_auc']:.4f} top10={sc['top10_hit_rate']:.4f} "
              f"spearman={sc['spearman']:.4f}")


if __name__ == "__main__":
    main()
