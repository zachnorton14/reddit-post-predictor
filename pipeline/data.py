"""The one way the pipeline loads data: labeled splits from a pinned revision of the HF dataset, never local caches."""

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

import dadjokes_models as dm
import textonly_baselines as tb
import textonly_data as td

ROOT = Path(__file__).resolve().parents[1]

# Validation and test posts scored per subreddit. Big enough to keep every dadjokes post (29,760 val, 32,826 test).
EVAL_PER_SUB = 40_000


def load_pinned(dataset, revision, subreddit="dadjokes", body_chars=1000):
    """One subreddit's train/val/test splits (dm.Config dates) from `dataset` at `revision`. Used for training.

    `body_chars` cuts each body to that many characters (None keeps it whole). It changes only the text, never which
    posts are in a split.
    """
    root = snapshot_download(dataset, repo_type="dataset", revision=revision,
                             allow_patterns=f"data/{subreddit}/**")
    return dm.load(dm.Config(subreddit=subreddit, body_chars=body_chars), source=f"{root}/data/{subreddit}/*/*.parquet")


def eval_cache_key(dataset, revision, body_chars, per_sub):
    """Names the cached eval splits: the pinned revision, the settings and the code that labels them."""
    code = b"".join(Path(m.__file__).read_bytes() for m in (td, tb, dm)) + Path(__file__).read_bytes()
    parts = f"{dataset}@{revision}|{body_chars}|{per_sub}|".encode() + code
    return hashlib.sha256(parts).hexdigest()[:20]


def load_eval(dataset, revision, body_chars=1000, per_sub=EVAL_PER_SUB):
    """Validation and test posts of every subreddit in the revision, for scoring (`subreddit` column says which).

    Labels are computed on all posts of a subreddit and quarter before sampling, so the sample doesn't move the
    top-10% threshold. Train is left nearly empty: scoring never needs it.

    Labeling every post takes a long time (AskReddit is most of the 2.9 GB), so the splits are saved in $EVAL_CACHE
    (default data/eval_cache) under eval_cache_key: a new revision, setting or labeling code gets a new file.
    """
    cfg = dm.Config()
    tcfg = tb.Config(source="", train_end=cfg.train_end, val_end=cfg.val_end, body_chars=body_chars,
                     max_train_per_sub=1, max_eval_per_sub=per_sub)
    path = Path(os.environ.get("EVAL_CACHE", ROOT / "data/eval_cache")) / \
        f"{eval_cache_key(dataset, revision, body_chars, per_sub)}.parquet"
    if not path.exists():
        root = snapshot_download(dataset, repo_type="dataset", revision=revision, allow_patterns="data/**")
        data = tb.load_data(replace(tcfg, source=f"{root}/data/*/*/*.parquet"))
        data.con.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        pd.concat([getattr(data, s).assign(split=s) for s in tb.SPLITS]).to_parquet(tmp, index=False)
        tmp.replace(path)
    df = pd.read_parquet(path)
    parts = {s: df[df.split == s].drop(columns="split").reset_index(drop=True) for s in tb.SPLITS}
    return tb.Data(tcfg, None, parts["train"], parts["val"], parts["test"])
