"""Load sampled Arctic Shift submission shards into DuckDB.

Shards in a month are in time order (one shard of 500k rows covers ~10 hours),
so the shards are taken evenly spread across each month.
"""

from pathlib import Path

import duckdb
import numpy as np
from huggingface_hub import HfApi, snapshot_download

REPO = "open-index/arctic"

# Exclusion rules: NSFW posts, and deleted/removed posts (Reddit clears their author, body, and URL)
NSFW_SQL = "coalesce(over_18, false)"
DELETED_SQL = """(
    author = '[deleted]'
    OR coalesce(selftext, '') IN ('[deleted]', '[removed]')
    OR title IN ('[deleted by user]', '[ Removed by moderator ]')
)"""

POST_TYPE_SQL = """
CASE
  WHEN url IS NULL OR url = '' THEN 'no url'
  WHEN url LIKE '%reddit.com/r/%/comments/%' OR url LIKE '%reddit.com/r/%/s/%' THEN 'self'
  WHEN url LIKE '%i.redd.it%' OR regexp_matches(lower(url), '\\.(jpg|jpeg|png|gif|webp)$') OR url LIKE '%imgur.com%' THEN 'image'
  WHEN url LIKE '%v.redd.it%' OR url LIKE '%youtube.com%' OR url LIKE '%youtu.be%' THEN 'video'
  WHEN url LIKE '%reddit.com/gallery/%' THEN 'gallery'
  ELSE 'external link'
END
"""


def select_shards(year, months=range(1, 13), shards_per_month=12):
    """Return shard paths spread evenly across each month. shards_per_month=None takes all shards."""
    api = HfApi()
    shards = []
    for m in months:
        month_shards = sorted(
            f.path
            for f in api.list_repo_tree(REPO, path_in_repo=f"data/submissions/{year}/{m:02d}", repo_type="dataset")
            if f.path.endswith(".parquet")
        )
        n = len(month_shards) if shards_per_month is None else min(shards_per_month, len(month_shards))
        idx = np.unique(np.linspace(0, len(month_shards) - 1, n).round().astype(int))
        shards += [month_shards[i] for i in idx]
    return shards


def download(shards, data_dir="data/arctic"):
    """Download shards that are not already in data_dir. Return the local paths."""
    snapshot_download(REPO, repo_type="dataset", local_dir=data_dir, allow_patterns=shards, max_workers=8)
    return [str(Path(data_dir) / s) for s in shards]


def connect(local_files):
    """Return a DuckDB connection with two views: raw_subs (all posts) and subs (filtered)."""
    con = duckdb.connect()
    file_list = ", ".join(f"'{f}'" for f in local_files)
    con.execute(f"CREATE OR REPLACE VIEW raw_subs AS SELECT * FROM read_parquet([{file_list}])")
    con.execute(f"CREATE OR REPLACE VIEW subs AS SELECT * FROM raw_subs WHERE NOT {NSFW_SQL} AND NOT {DELETED_SQL}")
    return con
