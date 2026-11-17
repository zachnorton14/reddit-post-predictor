"""Does a context-free "good post" score predict how a joke does on average, better than one post's own result does?

    python scripts/good_post_check.py --checkpoint data/pooling/minilm_attn4_s42

The quality score is the model's prediction with the posting context (hour, weekday, month, subreddit activity,
author history; prototype_models.CONTEXT) fixed to the most typical training values, so it depends only on the post.
It is reported as a percentile of the validation scores.

Check on repeated jokes in 2024 validation: for each post whose exact joke (normalized title and body) appears again
in validation, the target is the mean engagement percentile of the *other* copies. One post's own result is the
luck-bound baseline. Train and validation only; test is never loaded. Writes results/good_post/<run name>/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def normalize(text):
    return text.fillna("").str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.split().str.join(" ")


def main(argv=None):
    import prototype_models as pm
    import run_pooling as rp

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="a finished run_pooling run directory")
    parser.add_argument("--out", type=Path, help="default: results/good_post/<checkpoint folder name>")
    args = parser.parse_args(argv)
    args.out = args.out or ROOT / "results/good_post" / args.checkpoint.name

    parts = rp.data_splits()
    train = parts["train"][parts["train"].created_at >= pd.Timestamp("2019-01-01")]
    val = parts["val"].reset_index(drop=True)
    context = pm.typical_context(train)

    val["full"] = pm.predict_checkpoint(args.checkpoint, val)
    val["quality"] = pm.predict_checkpoint(args.checkpoint, pm.neutral_context(val, context))
    val["quality_pct"] = pm.empirical_ranks(val.quality, val.quality)

    # Mean outcome of the other copies of the same joke in validation
    val["joke"] = normalize(val.title) + "|" + normalize(val.body)
    g = val.groupby("joke").metric_pct
    copies = g.transform("size")
    val["others_mean_pct"] = (g.transform("sum") - val.metric_pct) / (copies - 1).replace(0, np.nan)
    rep = val[copies > 1]

    def rho(a, b):
        return float(a.corr(b, method="spearman"))

    summary = {
        "checkpoint": str(args.checkpoint.relative_to(ROOT) if args.checkpoint.is_absolute() else args.checkpoint),
        "context": {k: float(v) for k, v in context.items()},
        "val_posts": len(val), "repeated_posts": len(rep), "repeated_jokes": int(rep.joke.nunique()),
        "spearman_vs_own_outcome": {"full": rho(val.full, val.metric_pct), "quality": rho(val.quality, val.metric_pct)},
        "spearman_vs_other_copies": {"own_outcome": rho(rep.metric_pct, rep.others_mean_pct),
                                     "full": rho(rep.full, rep.others_mean_pct),
                                     "quality": rho(rep.quality, rep.others_mean_pct)},
    }
    bands = pd.cut(val.quality_pct, [0, .3, .7, .9, 1], labels=["bad (<30%)", "average (30-70%)", "good (70-90%)",
                                                                  "great (top 10%)"], include_lowest=True)
    table = val.groupby(bands, observed=True).agg(posts=("label", "size"), top10_rate=("label", "mean"),
                                                  median_outcome_pct=("metric_pct", "median")).reset_index()
    table = table.rename(columns={"quality_pct": "quality band"})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table.to_csv(args.out / "bands.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
