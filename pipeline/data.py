"""The one way the pipeline loads data: the dadjokes splits from a pinned revision of the HF dataset."""

from pathlib import Path

from huggingface_hub import snapshot_download

import dadjokes_models as dm


def load_pinned(dataset, revision, subreddit="dadjokes", body_chars=1000, *, validation_only=False):
    """Labeled train/val/test splits (dm.Config dates) from `dataset` at `revision`, never from local caches.

    `body_chars` cuts each body to that many characters (None keeps it whole). It changes only the text, never which
    posts are in a split. Validation jobs download and open only pre-test shards, including the earlier years
    needed for posting-history features, even when the snapshot cache also contains test files.
    """
    cfg = dm.Config(subreddit=subreddit, body_chars=body_chars)
    if validation_only:
        # The Reddit archive starts in 2005. The train/validation boundary and end are whole years.
        patterns = [f"data/{subreddit}/{year}/*.parquet" for year in range(2005, int(cfg.val_end[:4]))]
        root = Path(snapshot_download(dataset, repo_type="dataset", revision=revision, allow_patterns=patterns))
        files = sorted(str(file) for pattern in patterns for file in root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No pre-test {subreddit} shards at {dataset}@{revision}")
        return dm.load(cfg, source=files, validation_only=True)
    root = snapshot_download(dataset, repo_type="dataset", revision=revision, allow_patterns=f"data/{subreddit}/**")
    return dm.load(cfg, source=f"{root}/data/{subreddit}/*/*.parquet")
