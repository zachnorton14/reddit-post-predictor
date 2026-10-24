"""Reproduce the dadjokes model comparison in PROJECT_NOTES.md. Caches go to data/models/.

    python run_dadjokes_models.py            # all models (fine-tuning takes ~25 min on an M3 Max)
    python run_dadjokes_models.py --no-finetune
"""

import argparse
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression

import dadjokes_models as dm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-finetune", action="store_true")
    args = parser.parse_args()

    cfg = dm.Config()
    data = dm.load(cfg)
    X = dm.Features(data)
    res = dm.Results(data)

    res.add("ridge (tf-idf + meta)", *dm.fit_ridge(data, X.sparse(["meta", "tfidf"])))
    res.add("hgb (meta only)", *dm.fit_hgb(data)[:3])
    text = dm.oof_ridge(data, X.sparse(["tfidf"]), alpha=3)
    res.add("ridge (tf-idf only)", text["val"], text["test"])
    res.add("hgb (meta + tf-idf ridge, stacked)", *dm.fit_hgb(data, {"text_tfidf": text})[:3])

    X.add("embed", dm.embed(data, cfg.embed_model))
    res.add("ridge (embeddings only)", *dm.fit_ridge(data, X.sparse(["embed"]), alphas=(1, 3, 10, 30)))
    res.add("ridge (embeddings + tf-idf + meta)", *dm.fit_ridge(data, X.sparse(["meta", "tfidf", "embed"])))
    emb = dm.oof_ridge(data, X.sparse(["embed"]), alpha=10)
    res.add("hgb (meta + tf-idf + embedding ridges, stacked)",
            *dm.fit_hgb(data, {"text_tfidf": text, "text_embed": emb})[:3])

    if not args.no_finetune:
        ft = dm.finetune(data, epochs=3)
        pickle.dump(ft, open(dm.CACHE_DIR / "finetune_minilm.pkl", "wb"))
        res.add("fine-tuned MiniLM (text only)", ft["val"], ft["test"], ft["seconds"])
        # Blend with the metadata model; weights are fit on validation, so test stays untouched
        P = res.preds
        for combo in [["fine-tuned MiniLM (text only)", "hgb (meta only)"],
                      ["fine-tuned MiniLM (text only)", "ridge (tf-idf + meta)"]]:
            V = np.column_stack([P[c]["val"] for c in combo])
            T = np.column_stack([P[c]["test"] for c in combo])
            m = LinearRegression().fit(V, data.val.metric)
            res.add("blend: " + " + ".join(combo), m.predict(V), m.predict(T))

    print(res.table().to_string())


if __name__ == "__main__":
    main()
