"""Score a saved model on the test split of a pinned dataset revision.

    python -m pipeline.evaluate --model <dir> --dataset <hf dataset repo> --revision <sha> --out <file>

Loads any past model by the `type` in <dir>/model.json, so the deployed champion can be scored with this code.
Never writes into <dir> (the champion is mounted read-only).
"""

import argparse
import json
from pathlib import Path

import dadjokes_models as dm
from pipeline.data import load_pinned


def predict_minilm(model_dir, meta, data):
    import torch
    from transformers import AutoTokenizer

    net = dm.regression_net(meta["base_model"])
    net.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    net.to(dm.device())
    tok = AutoTokenizer.from_pretrained(model_dir)
    return dm.predict_net(net, tok, dm.text_of(data.test), meta["max_len"], meta["mu"], meta["sd"])


# model.json type -> loader(model_dir, meta, data) returning test predictions. Keep the deployed champion's type.
LOADERS = {"minilm": predict_minilm}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="model directory with model.json")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset repo")
    parser.add_argument("--revision", required=True, help="dataset commit")
    parser.add_argument("--out", required=True, type=Path, help="metrics JSON file")
    return parser.parse_args(argv)


def metrics(test, p):
    s = dm.scores(test, p)
    # compare_metrics.py gates on top10_hit_rate
    return {"pr_auc": float(s["pr_auc"]), "top10_hit_rate": float(s["top10_hit"]),
            "spearman": float(s["spearman"]), "n_test": len(test)}


def main(argv=None):
    args = parse_args(argv)
    meta = json.loads((args.model / "model.json").read_text())
    if meta.get("type") not in LOADERS:
        raise SystemExit(f"unknown model type {meta.get('type')!r} in {args.model}/model.json; "
                         f"known types: {', '.join(LOADERS)}")
    data = load_pinned(args.dataset, args.revision)
    result = metrics(data.test, LOADERS[meta["type"]](args.model, meta, data))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
