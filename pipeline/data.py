"""The one way the pipeline loads data: the dadjokes splits from a pinned revision of the HF dataset."""

from huggingface_hub import snapshot_download

import dadjokes_models as dm


def load_pinned(dataset, revision, subreddit="dadjokes"):
    """Labeled train/val/test splits (dm.Config dates) from `dataset` at `revision`, never from local caches."""
    root = snapshot_download(dataset, repo_type="dataset", revision=revision,
                             allow_patterns=f"data/{subreddit}/**")
    return dm.load(dm.Config(subreddit=subreddit), source=f"{root}/data/{subreddit}/*/*.parquet")
