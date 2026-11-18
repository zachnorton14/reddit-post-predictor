"""The one way the pipeline loads data: labeled splits from a pinned revision of the HF dataset, never local caches."""

from huggingface_hub import snapshot_download

import dadjokes_models as dm
import textonly_baselines as tb

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


def load_eval(dataset, revision, body_chars=1000, per_sub=EVAL_PER_SUB):
    """Validation and test posts of every subreddit in the revision, for scoring (`subreddit` column says which).

    Labels are computed on all posts of a subreddit and quarter before sampling, so the sample doesn't move the
    top-10% threshold. Train is left nearly empty: scoring never needs it.
    """
    root = snapshot_download(dataset, repo_type="dataset", revision=revision, allow_patterns="data/**")
    cfg = dm.Config()
    data = tb.load_data(tb.Config(source=f"{root}/data/*/*/*.parquet", train_end=cfg.train_end, val_end=cfg.val_end,
                                  body_chars=body_chars, max_train_per_sub=1, max_eval_per_sub=per_sub))
    data.con.close()
    data.con = None
    return data
