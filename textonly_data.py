"""Load the text-only subreddit dataset (output of extract_subreddits.py) into DuckDB.

The extract already drops NSFW and deleted/removed posts. This module adds the derived columns that the EDA
and the baselines share, so the definitions exist in one place:

- `posts` view: every post with its post type, outcome metrics, local time, and title/body measurements.
- `history` table: time-aware counts that only use earlier posts (competition, author history, reposts).
- `add_labels()`: success labels relative to each subreddit and time period.
"""

from pathlib import Path

import duckdb

from extract_subreddits import OUT_DIR, SUBREDDITS, TARGET_REPO
from reddit_data import POST_TYPE_SQL

SOURCES = {
    "local": str(OUT_DIR / "data/*/*/*.parquet"),
    "hf": f"hf://datasets/{TARGET_REPO}/data/*/*/*.parquet",
}
# Most of the audience of these subreddits is in North America. Local time also removes the one-hour
# shift that daylight saving time causes in UTC.
TIMEZONE = "America/New_York"
# The source archive has no submissions in this range
GAP_START, GAP_END = "2022-09-01", "2023-12-01"

# Chart colors: a validated categorical palette, and a fixed color per subreddit in this order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
INK, MUTED = "#0b0b0b", "#52514e"
COLORS = dict(zip(SUBREDDITS, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET]))

METRICS = {
    "score": "ln(1 + greatest(score, 0))",
    "comments": "ln(1 + greatest(num_comments, 0))",
    "engagement": "ln(1 + greatest(score, 0)) + ln(1 + greatest(num_comments, 0))",
}
PERIODS = {
    "month": "strftime(created_at, '%Y-%m')",
    "quarter": "year(created_at) || '-Q' || quarter(created_at)",
    "year": "year(created_at)::VARCHAR",
}

# Scheduled community threads (posted by AutoModerator or on a fixed schedule) get many comments
# automatically, so they are not ordinary posts to predict. add_labels() drops them by default.
MEGATHREAD_SQL = r"""(
    author = 'AutoModerator'
    OR regexp_matches(lower(title), 'megathread|moronic monday|thickhead(ed)? thursday|(daily|weekly|monthly) .*thread')
)"""

# Edit notes that authors add after posting, often because the post did well ("EDIT: thanks for the gold").
# Text after the first marker leaks the outcome. Markers: "EDIT:", "**Edit**", "[edit]", "Edit #2 -",
# "UPDATE:", "ETA:", or "EDIT Since..." at the start of a line, and "edit:" anywhere.
EDIT_RE = (r"((^|\n)[\s*_~#>\[(]*(?i:edit|update|eta)\s*#?\d*\s*([:;\-–\]*)]|\s+[A-Z])"
           r"|(?i:\bedit)\s*#?\d*\s*:)")
BODY_CLEAN_SQL = f"regexp_replace(regexp_replace(coalesce(selftext, ''), '(?s){EDIT_RE}.*$', ''), '^\\s+|\\s+$', '', 'g')"


def set_style():
    """Matplotlib style shared by the notebooks: thin marks, recessive grid."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.figsize": (10, 4.5), "figure.dpi": 110,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "lines.linewidth": 2,
        "axes.prop_cycle": plt.cycler(color=list(COLORS.values())),
    })


def connect(source="local", history=True):
    """Return a DuckDB connection with the `posts` view (and the `history` table if `history`)."""
    if source == "local":
        # Fix the file list now, so files that a running extract adds later do not change the views
        local = sorted(str(f) for f in Path(OUT_DIR / "data").glob("*/*/*.parquet"))
        if not local:
            raise FileNotFoundError(f"No files in {OUT_DIR}/data. Run extract_subreddits.py or use source='hf'.")
        files = "[" + ", ".join(f"'{f}'" for f in local) + "]"
    else:
        files = f"'{SOURCES.get(source, source)}'"
    con = duckdb.connect()
    # The progress bar is saved into notebook outputs as a widget
    con.execute("SET enable_progress_bar = false")
    con.execute(f"CREATE OR REPLACE VIEW raw_posts AS SELECT * FROM read_parquet({files})")
    if history:
        add_history(con)
    hist_cols = "h.posts_prev_hour, h.posts_prev_day, h.author_prev_posts, h.title_prev_count," if history else ""
    hist_join = "LEFT JOIN history h USING (id)" if history else ""
    con.execute(f"""
    CREATE OR REPLACE VIEW posts AS
    WITH p AS (
      SELECT *, timezone('{TIMEZONE}', to_timestamp(created_utc)) AS local_ts, {BODY_CLEAN_SQL} AS body_clean
      FROM raw_posts
    )
    SELECT p.*,
           {POST_TYPE_SQL}                                          AS post_type,
           {METRICS["score"]}                                       AS score_metric,
           {METRICS["comments"]}                                    AS comments_metric,
           {METRICS["engagement"]}                                  AS engagement,
           year(created_at)                                         AS year,
           hour(local_ts)                                           AS local_hour,
           isodow(local_ts)                                         AS local_dow,
           month(local_ts)                                          AS local_month,
           len(string_split(title, ' '))                            AS title_words,
           contains(title, '?')::INT                                AS title_has_question,
           regexp_matches(title, '\\d')::INT                        AS title_has_number,
           (title = upper(title) AND regexp_matches(title, '[A-Z]'))::INT AS title_all_caps,
           length(body_clean)                                       AS body_length,
           (length(body_clean) > 0)::INT                            AS has_body,
           regexp_matches(body_clean, 'https?://')::INT             AS body_has_url,
           regexp_matches(coalesce(selftext, ''), '{EDIT_RE}')::INT AS body_has_edit,
           {MEGATHREAD_SQL}::INT                                    AS is_megathread,
           (link_flair_text IS NOT NULL)::INT                       AS has_link_flair,
           (author_flair_text IS NOT NULL)::INT                     AS has_author_flair,
           {hist_cols}
           created_at >= TIMESTAMP '{GAP_START}' AND created_at < TIMESTAMP '{GAP_END}' AS in_gap
    FROM p {hist_join}
    """)
    return con


def add_history(con):
    """Create the `history` table. Each count uses only posts created before the post itself.

    - posts_prev_hour / posts_prev_day: posts in the same subreddit in the previous hour / 24 hours
      (how crowded /new is). Deleted posts were dropped by the extract, so these are slight undercounts.
    - author_prev_posts: earlier posts by the same author in the same subreddit.
    - title_prev_count: earlier posts in the same subreddit with the same normalized title (reposts).
    """
    con.execute("""
    CREATE OR REPLACE TABLE history AS
    SELECT id,
           count(*) OVER (PARTITION BY subreddit ORDER BY created_utc
                          RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING)  AS posts_prev_hour,
           count(*) OVER (PARTITION BY subreddit ORDER BY created_utc
                          RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING) AS posts_prev_day,
           row_number() OVER (PARTITION BY subreddit, author ORDER BY created_utc, id) - 1 AS author_prev_posts,
           row_number() OVER (PARTITION BY subreddit, title_key ORDER BY created_utc, id) - 1 AS title_prev_count
    FROM (
      SELECT id, subreddit, author, created_utc,
             hash(trim(regexp_replace(lower(title), '[^a-z0-9]+', ' ', 'g'))) AS title_key
      FROM raw_posts
    )
    """)


def add_labels(con, metric="engagement", quantile=0.9, period="quarter", min_posts=100, self_only=True,
               drop_megathreads=True):
    """Create the `thresholds` table and the `labeled` view.

    A post is a success if its metric is above the `quantile` of its subreddit in the same time period.
    Subreddit-periods with fewer than `min_posts` posts have no reliable threshold, so their posts are dropped.
    With `self_only`, only text posts are kept, and the thresholds come from text posts. With
    `drop_megathreads`, scheduled threads (MEGATHREAD_SQL) are dropped before the thresholds are computed.

    The threshold uses outcomes of the post's own period. That is part of the label definition, not a feature:
    no feature may use outcomes from the post's period or later.
    """
    conditions = (["post_type = 'self'"] if self_only else []) + (["is_megathread = 0"] if drop_megathreads else [])
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    con.execute(f"""
    CREATE OR REPLACE VIEW label_base AS
    SELECT *, {PERIODS[period]} AS period, {METRICS[metric]} AS metric FROM posts {where}
    """)
    con.execute(f"""
    CREATE OR REPLACE TABLE thresholds AS
    SELECT subreddit, period, count(*) AS period_posts,
           quantile_cont(metric, {quantile}) AS threshold, median(metric) AS period_median
    FROM label_base GROUP BY 1, 2 HAVING count(*) >= {min_posts}
    """)
    con.execute("""
    CREATE OR REPLACE VIEW labeled AS
    SELECT b.*, t.threshold, (b.metric > t.threshold)::INT AS label
    FROM label_base b JOIN thresholds t USING (subreddit, period)
    """)
