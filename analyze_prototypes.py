"""Presentation diagnostics for already-frozen prototype predictions.

This script never chooses or tunes a model.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import prototype_models as pm
from run_prototypes import ROOT, load_data


def analyze(run, data_path=None):
    run = Path(run)
    data = load_data(data_path)
    test = data["test"]
    scores = pd.read_parquet(run / "test_predictions.parquet")
    if not np.array_equal(scores.id, test.id):
        raise ValueError("Prediction IDs do not match evaluation rows")
    selected = json.loads((run / "selection.json").read_text())["selected"]
    destination = ROOT / "results/prototypes" / run.name
    models = [c for c in scores if c not in ("id", "created_at", "label")]
    rows = []
    for quarter, indices in test.groupby(test.created_at.dt.to_period("Q")).groups.items():
        for name in models:
            rows.append(dict(quarter=str(quarter), model=name, posts=len(indices),
                             **pm.metrics(test.loc[indices], scores.loc[indices, name].to_numpy())))
    quarters = pd.DataFrame(rows)
    quarters.to_csv(destination / "by_quarter.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    display = list(dict.fromkeys(["incumbent", "interaction", "deberta", selected]))
    for name in display:
        if name in models:
            part = quarters[quarters.model == name]
            ax.plot(part.quarter, part.ap, marker="o", label=name.replace("_", " "))
    ax.set(title="Historical test AP by quarter", ylabel="Average precision", xlabel="Quarter")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.savefig(destination / "quarter_stability.png", dpi=180)
    fig.savefig(destination / "quarter_stability.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name in ("interaction", "deberta"):
        history_path = run / name / "history.json"
        if history_path.exists():
            history = pd.DataFrame(json.loads(history_path.read_text()))
            axes[0].plot(history.epoch, history.train_loss, marker="o", label=name)
            axes[1].plot(history.epoch, history.ap, marker="o", label=name)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks([1, 2, 3])
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    axes[0].set(title="Training loss", ylabel="Standardized engagement MSE")
    axes[1].set(title="Validation performance", ylabel="Average precision")
    fig.savefig(destination / "learning_curves.png", dpi=180)
    fig.savefig(destination / "learning_curves.svg")
    plt.close(fig)
    normalize = lambda s: s.fillna("").str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    recent = data["train"][data["train"].created_at >= pd.Timestamp("2019-01-01")]
    seen = normalize(test.title).isin(set(normalize(recent.title)))
    rows = []
    for group, mask in [("seen_title", seen), ("novel_title", ~seen)]:
        for name in models:
            rows.append(dict(group=group, model=name, posts=int(mask.sum()),
                             **pm.metrics(test[mask], scores.loc[mask, name].to_numpy())))
    pd.DataFrame(rows).to_csv(destination / "seen_vs_novel.csv", index=False)
    count = max(1, len(test) // 10)
    masks = {}
    for name in ("incumbent", selected):
        masks[name] = np.zeros(len(test), dtype=bool)
        masks[name][np.argsort(-scores[name].to_numpy(), kind="stable")[:count]] = True
    old, new = masks["incumbent"], masks[selected]
    labels = test.label.to_numpy().astype(bool)
    transitions = {"selected": selected, "posts_reviewed": count,
                   "old_successes": int((old & labels).sum()), "selected_successes": int((new & labels).sum()),
                   "new_successes_entering": int((new & ~old & labels).sum()),
                   "old_successes_leaving": int((old & ~new & labels).sum())}
    pm.write_json(destination / "top10_transitions.json", transitions)
    examples = []
    for category, mask in [("new_success", new & ~old & labels), ("new_false_positive", new & ~old & ~labels),
                           ("lost_success", old & ~new & labels)]:
        indices = np.flatnonzero(mask)
        indices = indices[np.argsort(-scores.loc[indices, selected].to_numpy(), kind="stable")[:20]]
        for i in indices:
            examples.append({"category": category, "id": test.id.iloc[i], "title": test.title.iloc[i],
                             "body_excerpt": test.body.iloc[i][:250], "label": int(labels[i]),
                             "seen_title": bool(seen.iloc[i]), "old_score": scores.incumbent.iloc[i],
                             "selected_score": scores[selected].iloc[i]})
    pd.DataFrame(examples, columns=["category", "id", "title", "body_excerpt", "label", "seen_title", "old_score",
                                    "selected_score"]).to_csv(destination / "changed_picks.csv", index=False)
    print(transitions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("data/prototypes/full"))
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    analyze(args.run, args.data)
