"""Extract all posts of a few text-only subreddits from every Arctic Shift submission shard.

Each shard is downloaded, filtered, and deleted, so disk use stays small. Output is one Parquet file
per subreddit per month: OUT_DIR/data/<subreddit>/<YYYY>/<MM>.parquet. NSFW and deleted/removed posts are
dropped with the rules in reddit_data.py. All columns are kept.

Usage:
    python extract_subreddits.py --months 2025-12          # one month, for a test
    python extract_subreddits.py --push                    # all months, push to HF after each year
    python extract_subreddits.py --push-only               # upload what is already extracted
"""

import argparse
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from reddit_data import DELETED_SQL, NSFW_SQL

SOURCE_REPO = "open-index/arctic"
TARGET_REPO = "zachnorton03/textonly-subreddit-predictors"
SUBREDDITS = [
    "AskReddit", "Showerthoughts", "unpopularopinion", "dadjokes",
    "hypotheticalsituation", "sysadmin", "TwoSentenceHorror",
]
OUT_DIR = Path("data/textonly")
TMP_DIR = Path("data/_tmp_shards")
DONE_FILE = OUT_DIR / ".done_months"

CANONICAL = {s.lower(): s for s in SUBREDDITS}
SUB_LIST_SQL = ", ".join(f"'{s}'" for s in CANONICAL)


def list_shards():
    """Return {"YYYY-MM": [shard paths]} for all submission shards."""
    files = HfApi().list_repo_tree(SOURCE_REPO, path_in_repo="data/submissions", repo_type="dataset", recursive=True)
    months = defaultdict(list)
    for f in files:
        if f.path.endswith(".parquet"):
            _, _, year, month, _ = f.path.split("/")
            months[f"{year}-{month}"].append(f.path)
    return {m: sorted(p) for m, p in sorted(months.items())}


def filter_shard(shard):
    """Download one shard, keep SFW, non-deleted rows of the chosen subreddits, delete the shard."""
    for attempt in range(5):
        try:
            local = hf_hub_download(SOURCE_REPO, shard, repo_type="dataset", local_dir=TMP_DIR)
            break
        except Exception as e:
            if attempt == 4:
                raise
            print(f"  retry {shard}: {e}")
            time.sleep(10 * (attempt + 1))
    try:
        con = duckdb.connect()
        return con.execute(
            f"SELECT * FROM read_parquet('{local}') WHERE lower(subreddit) IN ({SUB_LIST_SQL})"
            f" AND NOT {NSFW_SQL} AND NOT {DELETED_SQL}"
        ).to_arrow_table()
    finally:
        os.remove(local)


def extract_month(month, shards, workers):
    start = time.time()
    with ThreadPoolExecutor(workers) as pool:
        tables = list(pool.map(filter_shard, shards))
    table = pa.concat_tables(tables)
    con = duckdb.connect()
    con.register("t", table)
    year, mm = month.split("-")
    counts = {}
    for key, name in CANONICAL.items():
        part = con.execute(
            f"SELECT * FROM t WHERE lower(subreddit) = '{key}' ORDER BY created_utc, id"
        ).to_arrow_table()
        if part.num_rows:
            path = OUT_DIR / "data" / name / year / f"{mm}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(part, path, compression="zstd")
            counts[name] = part.num_rows
    with DONE_FILE.open("a") as f:
        f.write(month + "\n")
    print(f"{month}: {len(shards)} shards, {time.time() - start:.0f}s, {counts}", flush=True)


def write_readme():
    configs = "\n".join(
        f"- config_name: {s}\n  data_files: data/{s}/*/*.parquet" for s in SUBREDDITS
    )
    (OUT_DIR / "README.md").write_text(f"""---
configs:
- config_name: all
  default: true
  data_files: data/*/*/*.parquet
{configs}
---

# Text-only subreddit submissions

All submissions from {len(SUBREDDITS)} text-focused subreddits, extracted from the Arctic Shift Reddit dumps
in [`{SOURCE_REPO}`](https://huggingface.co/datasets/{SOURCE_REPO}). Used for the CSC422 project
"Predicting Social Media Post Success".

Subreddits: {", ".join(SUBREDDITS)}.

Layout: `data/<subreddit>/<YYYY>/<MM>.parquet`, rows sorted by `created_utc`. Columns are unchanged from the
source. NSFW posts (`over_18`) and deleted/removed posts are excluded: author `[deleted]`, selftext `[deleted]` or
`[removed]`, or title `[deleted by user]` or `[ Removed by moderator ]`. `score` and `num_comments` are the
values at archival time.

The source has no submissions for 2022-09 through 2023-11, so this dataset has the same gap.
""")


def push(year=None):
    """Upload the README and the extracted files of one year (or all years) in one commit."""
    write_readme()
    files = f"data/*/{year}/*.parquet" if year else "data/*/*/*.parquet"
    HfApi().upload_folder(
        repo_id=TARGET_REPO, repo_type="dataset", folder_path=OUT_DIR,
        allow_patterns=["README.md", files], delete_patterns=[files],
        commit_message=f"Add {year or 'all years'}",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="*", help="YYYY-MM months to extract (default: all)")
    parser.add_argument("--workers", type=int, default=8, help="parallel shard downloads")
    parser.add_argument("--push", action="store_true", help="upload to HF after each finished year")
    parser.add_argument("--push-only", action="store_true", help="only upload what is already extracted")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.push_only:
        push()
        return

    all_months = list_shards()
    months = args.months or list(all_months)
    done = set(DONE_FILE.read_text().split()) if DONE_FILE.exists() else set()
    todo = [m for m in months if m not in done]
    print(f"{len(todo)} months to extract ({len(months) - len(todo)} already done)", flush=True)

    for i, month in enumerate(todo):
        extract_month(month, all_months[month], args.workers)
        year_finished = i + 1 == len(todo) or todo[i + 1][:4] != month[:4]
        if args.push and year_finished:
            push(month[:4])


if __name__ == "__main__":
    main()
