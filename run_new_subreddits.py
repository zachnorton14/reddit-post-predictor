"""Ridge and gradient-boosting baselines from dadjokes_models.py on the four added subreddits.

    python run_new_subreddits.py
"""

import pandas as pd

import dadjokes_models as dm

# LocalLLaMA starts in 2023-12 (created inside the source's data gap), so it gets later split dates
SPLITS = {
    "wallstreetbets": {},
    "singularity": {},
    "artificial": {},
    "LocalLLaMA": {"train_end": "2025-01-01", "val_end": "2025-07-01"},
}


def main():
    rows = []
    for sub, dates in SPLITS.items():
        print(f"\n== {sub}")
        data = dm.load(dm.Config(subreddit=sub, **dates))
        X = dm.Features(data)
        res = dm.Results(data)
        res.add("ridge (tf-idf + meta)", *dm.fit_ridge(data, X.sparse(["meta", "tfidf"])))
        res.add("hgb (meta only)", *dm.fit_hgb(data)[:3])
        t = res.table().assign(subreddit=sub, train=len(data.train), test=len(data.test),
                               base_rate=data.test.label.mean())
        rows.append(t)
    out = pd.concat(rows)
    print(out[["subreddit", "train", "test", "base_rate", "test_pr_auc", "test_top10_hit", "test_spearman"]].round(3)
          .to_string())


if __name__ == "__main__":
    main()
