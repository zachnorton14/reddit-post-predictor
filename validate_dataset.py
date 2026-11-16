"""Validate the text-only subreddit dataset: columns, missing values, chronological splits, and label leakage.

Runs on the local extract or on a Hugging Face dataset repo (downloaded first). Exits with status 1 if any check
fails, so CI can gate on it.

    python validate_dataset.py                                         # local extract in data/textonly
    python validate_dataset.py --repo jbduran/textonly-subreddit-predictors
"""

import argparse
import re
import sys
from pathlib import Path

import duckdb
from huggingface_hub import snapshot_download

import textonly_baselines as tb
import textonly_data as td
from extract_subreddits import OUT_DIR
from reddit_data import DELETED_SQL, NSFW_SQL

# "INT" accepts any integer width: the source stores title_length as INTEGER in 2021-05..2022-08
COLUMNS = {
    "id": "VARCHAR", "author": "VARCHAR", "subreddit": "VARCHAR", "title": "VARCHAR", "selftext": "VARCHAR",
    "score": "BIGINT", "created_utc": "BIGINT", "created_at": "TIMESTAMP", "title_length": "INT",
    "num_comments": "BIGINT", "url": "VARCHAR", "over_18": "BOOLEAN", "link_flair_text": "VARCHAR",
    "author_flair_text": "VARCHAR",
}
REQUIRED = ["id", "subreddit", "title", "score", "num_comments", "created_utc", "created_at", "url"]
# Outcomes, or values that change after posting (the archived flair is the final one); none may be a model feature
OUTCOMES = {"score", "num_comments", "metric", "metric_pct", "label", "threshold", "period_median", "engagement",
            "score_metric", "comments_metric", "link_flair_text", "has_link_flair"}
PERIOD_START = {"month": r"\d{4}-\d{2}-01", "quarter": r"\d{4}-(01|04|07|10)-01", "year": r"\d{4}-01-01"}


def count(con, sql):
    return con.execute(sql).fetchone()[0]


def type_ok(actual, expected):
    return actual in ("TINYINT", "SMALLINT", "INTEGER", "BIGINT") if expected == "INT" else actual == expected


def check_columns(con, files):
    """Every file has exactly the expected columns and types."""
    errors = []
    for f in files:
        cols = dict(con.execute(f"SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM '{f}')").fetchall())
        missing, extra = sorted(set(COLUMNS) - set(cols)), sorted(set(cols) - set(COLUMNS))
        wrong = sorted(f"{c} is {cols[c]}" for c in set(cols) & set(COLUMNS) if not type_ok(cols[c], COLUMNS[c]))
        if missing or extra or wrong:
            errors.append(f"{f}: missing {missing}, unexpected {extra}, wrong types {wrong}")
    return errors


def check_missing(con):
    """Required columns have no nulls, titles are not empty, and ids are unique."""
    errors = [f"{c}: {n:,} nulls" for c in REQUIRED if (n := count(con, f"SELECT count(*) FROM raw WHERE {c} IS NULL"))]
    if n := count(con, "SELECT count(*) FROM raw WHERE trim(title) = ''"):
        errors.append(f"title: {n:,} empty")
    if n := count(con, "SELECT count(*) - count(DISTINCT id) FROM raw"):
        errors.append(f"id: {n:,} duplicates")
    return errors


def check_layout(con):
    """Files follow data/<subreddit>/<YYYY>/<MM>.parquet, and the extract's NSFW/deleted filters held."""
    checks = {
        "rows in another subreddit's folder": "subreddit <> split_part(filename, '/', -3)",
        "rows in another month's file": "NOT ends_with(filename, strftime(created_at, '/%Y/%m.parquet'))",
        "created_at not equal to created_utc": "created_at <> make_timestamp(created_utc * 1000000)",
        "NSFW rows": NSFW_SQL,
        "deleted/removed rows": DELETED_SQL,
    }
    return [f"{name}: {n:,}" for name, cond in checks.items() if (n := count(con, f"SELECT count(*) FROM raw WHERE {cond}"))]


def check_splits(con, cfg):
    """Train < validation < test in time, period boundaries line up with the split dates, and the source gap holds."""
    errors = []
    if not cfg.train_end < cfg.val_end:
        errors.append(f"train_end {cfg.train_end} is not before val_end {cfg.val_end}")
    # A label threshold is computed per period; a period that straddles a split date mixes train and test outcomes
    for name in ("train_end", "val_end"):
        if not re.fullmatch(PERIOD_START[cfg.period], getattr(cfg, name)):
            errors.append(f"{name} {getattr(cfg, name)} is not the start of a {cfg.period}, so a label period "
                          f"spans two splits")
    if n := count(con, f"SELECT count(*) FROM raw WHERE created_at >= TIMESTAMP '{td.GAP_START}' "
                       f"AND created_at < TIMESTAMP '{td.GAP_END}'"):
        errors.append(f"{n:,} posts inside the source gap {td.GAP_START}..{td.GAP_END}")
    if n := count(con, "SELECT count(*) FROM raw WHERE created_at > now()::TIMESTAMP"):
        errors.append(f"{n:,} posts dated in the future")
    return errors


def split_summary(con, cfg):
    """Posts per subreddit and split (informational: a subreddit may have no train or test posts)."""
    return con.execute(f"""
        SELECT subreddit,
               count(*) FILTER (WHERE created_at < TIMESTAMP '{cfg.train_end}') AS train,
               count(*) FILTER (WHERE created_at >= TIMESTAMP '{cfg.train_end}'
                                  AND created_at < TIMESTAMP '{cfg.val_end}') AS val,
               count(*) FILTER (WHERE created_at >= TIMESTAMP '{cfg.val_end}') AS test
        FROM raw GROUP BY 1 ORDER BY 1
    """).df()


def check_leakage(glob):
    """No outcome is a model feature, and no post-hoc edit text is left in the cleaned body."""
    features = {c for g in tb.GROUPS.values() for cols in g.values() for c in cols}
    errors = [f"feature {c} is an outcome or changes after posting" for c in sorted(features & OUTCOMES)]
    con = td.connect(glob, history=False)
    if n := count(con, f"SELECT count(*) FROM posts WHERE regexp_matches(body_clean, '{td.EDIT_RE}')"):
        errors.append(f"{n:,} cleaned bodies still contain an EDIT/UPDATE note")
    return errors


def validate(root, cfg=None):
    """Run all checks on the Parquet files under root/data. Returns (errors by check, split summary)."""
    cfg = cfg or tb.Config()
    files = sorted(str(f) for f in Path(root, "data").glob("*/*/*.parquet"))
    if not files:
        return {"files": [f"no Parquet files under {root}/data"]}, None
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    columns = check_columns(con, files)
    if columns:
        return {"columns": columns}, None
    con.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet({files}, filename = true)")
    errors = {"columns": columns, "missing values": check_missing(con), "layout": check_layout(con),
              "chronological splits": check_splits(con, cfg), "label leakage": check_leakage(f"{root}/data/*/*/*.parquet")}
    return errors, split_summary(con, cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", help="Hugging Face dataset repo to download and check (default: local extract)")
    parser.add_argument("--revision", help="commit of the HF repo to check (default: latest)")
    args = parser.parse_args()
    root = OUT_DIR
    if args.repo:
        root = snapshot_download(args.repo, repo_type="dataset", revision=args.revision,
                                 allow_patterns=["data/*/*/*.parquet"])
    errors, summary = validate(root)
    if summary is not None:
        print(summary.to_string(index=False), "\n")
    for check, errs in errors.items():
        print(f"{'FAIL' if errs else 'ok  '} {check}")
        for e in errs:
            print(f"     - {e}")
    sys.exit(1 if any(errors.values()) else 0)


if __name__ == "__main__":
    main()
