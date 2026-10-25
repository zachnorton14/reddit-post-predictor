"""Score dadjokes test posts with Liquid AI's d1 decision model, zero-shot (no fine-tuning). Caches go to data/models/.

d1 (LiquidAI/d1-omni-600M) answers yes/no questions about a text with a probability. Each question's P(yes) is
used as a ranking score and evaluated like the models in dadjokes_models.py.

    python run_d1_zero_shot.py
"""

import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

import dadjokes_models as dm

QUESTIONS = {
    "top10": {"type": "noul", "instructions": "This is a post on r/dadjokes. Will it be in the top 10% of posts "
                                              "by upvotes and comments?"},
    "funny": {"type": "noul", "instructions": "Is this joke funny?"},
}


def main(batch=32):
    data = dm.load(dm.Config())
    path = dm.CACHE_DIR / f"d1_zero_shot_test_{len(data.test)}.parquet"
    if not path.exists():
        model = AutoModel.from_pretrained("LiquidAI/d1-omni-600M", trust_remote_code=True,
                                          dtype=torch.float16).to(dm.device())
        texts = [f"Title: {t}\nBody: {b}" for t, b in zip(data.test.title, data.test.body)]
        rows, t0 = [], time.time()
        for i in range(0, len(texts), batch):
            out = model.system_one_batch([(t, QUESTIONS) for t in texts[i:i + batch]])
            rows += [{q: a["answers"][q]["noul"] for q in QUESTIONS} for a in out]
            if i % (batch * 100) == 0:
                print(f"{i}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).assign(seconds=time.time() - t0).to_parquet(path)
    p = pd.read_parquet(path)
    print(pd.DataFrame({q: dm.scores(data.test, p[q].to_numpy()) for q in QUESTIONS}).T.round(3).to_string())


if __name__ == "__main__":
    main()
