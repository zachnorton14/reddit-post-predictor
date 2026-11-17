"""The one way the pipeline loads data: the dadjokes splits from a pinned revision of the HF dataset."""

from huggingface_hub import snapshot_download

import dadjokes_models as dm


def load_pinned(dataset, revision, subreddit="dadjokes", body_chars=1000):
    """Labeled train/val/test splits (dm.Config dates) from `dataset` at `revision`, never from local caches.

    `body_chars` cuts each body to that many characters (None keeps it whole). It changes only the text, never which
    posts are in a split.
    """
    root = snapshot_download(dataset, repo_type="dataset", revision=revision,
                             allow_patterns=f"data/{subreddit}/**")
    return dm.load(dm.Config(subreddit=subreddit, body_chars=body_chars), source=f"{root}/data/{subreddit}/*/*.parquet")
