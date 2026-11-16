"""Train the shipped model on a pinned dataset revision and save it with model.json.

    python -m pipeline.train --dataset <hf dataset repo> --revision <sha> --out <dir>

Settings are the ones chosen on validation in PROJECT_NOTES.md; the test split is never used here.
"""

import argparse
import json
from pathlib import Path

import dadjokes_models as dm
from pipeline.data import load_pinned

# Set by hand whenever the model changes (format: .claude/skills/model-release/SKILL.md)
MODEL_NAME = "dadjokes_models-22M-minilm_l6"
MODEL_TYPE = "minilm"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset repo")
    parser.add_argument("--revision", required=True, help="dataset commit")
    parser.add_argument("--out", required=True, type=Path, help="directory for the model")
    # Local subset runs only; the pipeline never sets these
    parser.add_argument("--max-train", type=int, help="train on a random sample of N posts")
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args(argv)


def model_json(params, mu, sd, max_len):
    return {"name": MODEL_NAME, "type": MODEL_TYPE, "params": params, "base_model": BASE_MODEL,
            "max_len": max_len, "mu": mu, "sd": sd}


def main(argv=None):
    args = parse_args(argv)
    import torch

    data = load_pinned(args.dataset, args.revision)
    if args.max_train and args.max_train < len(data.train):
        data.train = data.train.sample(args.max_train, random_state=data.cfg.seed) \
            .sort_values("created_at").reset_index(drop=True)
    m = dm.finetune(data, model_name=BASE_MODEL, epochs=args.epochs, return_model=True)["model"]
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(m["net"].state_dict(), args.out / "model.pt")
    m["tok"].save_pretrained(args.out)
    meta = model_json(sum(p.numel() for p in m["net"].parameters()), m["mu"], m["sd"], m["max_len"])
    (args.out / "model.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {MODEL_NAME} to {args.out}")


if __name__ == "__main__":
    main()
