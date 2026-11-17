"""How much posting-time signal is left: repost agreement, author history, and their gain over model predictions.

    python scripts/ceiling_check.py --predictions data/pooling/minilm_attn4_s42/val_predictions.npy

Train and validation only (test is never loaded). History features use only earlier posts at least 7 days old, so
their scores have settled. The gain check fits a logistic regression on one half of 2024 and scores the other half
(two folds by time). Writes results/ceiling/summary.json and results/ceiling/gain.csv.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SETTLE = np.timedelta64(7, "D")


def normalize(text):
    return text.fillna("").str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.split().str.join(" ")


def settled_history(frame, by):
    """Per post: count, mean percentile and top-10% rate of earlier posts in the same group that are >= 7 days old."""
    out = pd.DataFrame(index=frame.index, columns=["n", "mean_pct", "hit"], dtype=float)
    for _, g in frame.groupby(by):
        t = g.created_at.values
        pct = np.concatenate([[0], np.cumsum(g.metric_pct.values)])
        hit = np.concatenate([[0], np.cumsum(g.label.values)])
        j = np.searchsorted(t, t - SETTLE, side="right")
        with np.errstate(invalid="ignore", divide="ignore"):
            out.loc[g.index] = np.column_stack([j, pct[j] / j, hit[j] / j])
    return out


def main(argv=None):
    from scipy.stats import rankdata
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    import run_pooling as rp

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True, help="val_predictions.npy from a run_pooling run")
    parser.add_argument("--out", type=Path, default=ROOT / "results/ceiling")
    args = parser.parse_args(argv)

    parts = rp.data_splits()
    frame = pd.concat([parts["train"].assign(split="train"), parts["val"].assign(split="val")])
    authors = pd.concat(pd.read_parquet(f, columns=["id", "author"])
                        for f in glob.glob(str(ROOT / "data/**/dadjokes/**/*.parquet"), recursive=True))
    frame = frame.merge(authors.drop_duplicates("id"), on="id").sort_values("created_at").reset_index(drop=True)
    frame = frame[frame.created_at >= pd.Timestamp("2019-01-01")].reset_index(drop=True)
    frame["joke"] = normalize(frame.title) + "|" + normalize(frame.body)

    known = frame.author != "[deleted]"
    author = settled_history(frame[known], "author").reindex(frame.index)
    repost = settled_history(frame, "joke")
    val = frame.split == "val"

    # Agreement between copies of the same joke (the latest settled earlier copy vs this one)
    frame["prev_pct"] = frame.groupby("joke").metric_pct.shift(1)
    copies = frame[val & frame.prev_pct.notna()]
    summary = {
        "val_posts": int(val.sum()), "val_base_rate": float(frame.label[val].mean()),
        "val_reposts": int(len(copies)),
        "repost_spearman_prev_vs_this": float(copies.prev_pct.corr(copies.metric_pct, method="spearman")),
        "repost_label_rate_prev_top10": float(copies[copies.prev_pct >= 0.9].label.mean()),
        "repost_label_rate_prev_not_top10": float(copies[copies.prev_pct < 0.9].label.mean()),
        "val_share_with_settled_author_history": float((author.n[val] > 0).mean()),
    }
    h = frame[val & (author.n > 0)]
    summary["author_spearman_prev_mean_vs_this"] = float(author.mean_pct[h.index].corr(h.metric_pct, method="spearman"))

    # Gain over the model's own validation predictions
    pred = np.load(args.predictions)
    if len(pred) != len(parts["val"]):
        raise ValueError("predictions do not match the validation split")
    history = pd.concat([frame[["id", "label"]], author.add_prefix("a_"), repost.add_prefix("r_")], axis=1)
    v = history.set_index("id").loc[parts["val"].id.values].reset_index()  # the order the predictions are in
    features = pd.DataFrame({
        "model": rankdata(pred) / len(pred),
        "author_log_n": np.log1p(v.a_n.fillna(0)), "author_mean_pct": v.a_mean_pct.fillna(0.5),
        "author_hit": v.a_hit.fillna(0.1), "author_has": v.a_mean_pct.notna().astype(float),
        "repost_mean_pct": v.r_mean_pct.fillna(0), "repost_has": v.r_mean_pct.notna().astype(float)})
    y = v.label.values
    half = len(y) // 2
    folds = [(np.arange(half), np.arange(half, len(y))), (np.arange(half, len(y)), np.arange(half))]
    sets = {"model": ["model"],
            "+ author history": ["model", "author_log_n", "author_mean_pct", "author_hit", "author_has"],
            "+ repost history": ["model", "repost_mean_pct", "repost_has"]}
    sets["+ both"] = sorted(set(sets["+ author history"] + sets["+ repost history"]), key=list(features).index)
    rows = []
    for name, cols in sets.items():
        score = np.zeros(len(y))
        for fit, held in folds:
            model = LogisticRegression(max_iter=1000).fit(features.iloc[fit][cols], y[fit])
            score[held] = model.predict_proba(features.iloc[held][cols])[:, 1]
        top = np.argsort(-score)[:len(y) // 10]
        rows.append({"inputs": name, "val_ap": average_precision_score(y, score), "top10_hit_rate": y[top].mean()})

    args.out.mkdir(parents=True, exist_ok=True)
    summary["predictions"] = str(args.predictions.relative_to(ROOT) if args.predictions.is_absolute() else args.predictions)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "gain.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
