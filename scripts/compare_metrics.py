"""Release gate: fail if the candidate model regresses beyond tolerance against the production model.

    python scripts/compare_metrics.py candidate/metrics.json production/metrics.json

Both files are JSON objects with at least the keys in GATED (pipeline.evaluate's macro averages over subreddits).
A missing production file (no model has been released yet) passes. If both have `rows`
(pipeline.evaluate's hash of the scored post ids), they must match.
Exit status 1 blocks the merge into `release`.
"""

import json
import sys
from pathlib import Path

# Metric -> largest allowed drop (absolute). Higher is better for every gated metric.
GATED = {"pr_auc": 0.01, "top10_hit_rate": 0.01}
# A single subreddit's pr_auc drop that is flagged (a warning, not a failure: one subreddit's test set is noisy)
SUBREDDIT_WARN = 0.03


def compare(candidate: dict, production: dict | None) -> list[str]:
    """Return one message per regression; an empty list means the candidate passes."""
    missing = [m for m in GATED if m not in candidate]
    if missing:
        return [f"candidate metrics lack {', '.join(missing)}"]
    if production is None:
        return []
    if candidate.get("rows") and production.get("rows") and candidate["rows"] != production["rows"]:
        return ["candidate and production were scored on different posts (rows differ)"]
    return [
        f"{m}: {candidate[m]:.4f} vs. production {production[m]:.4f} (drop > {tol})"
        for m, tol in GATED.items()
        if m in production and candidate[m] < production[m] - tol
    ]


def subreddit_drops(candidate: dict, production: dict | None) -> list[str]:
    """One message per subreddit whose pr_auc dropped more than SUBREDDIT_WARN."""
    if production is None:
        return []
    cand, prod = candidate.get("by_subreddit", {}), production.get("by_subreddit", {})
    return [f"{sub} pr_auc: {cand[sub]['pr_auc']:.4f} vs. production {prod[sub]['pr_auc']:.4f}"
            for sub in sorted(set(cand) & set(prod)) if cand[sub]["pr_auc"] < prod[sub]["pr_auc"] - SUBREDDIT_WARN]


def main(candidate_path: str, production_path: str) -> int:
    candidate = json.loads(Path(candidate_path).read_text())
    production = json.loads(Path(production_path).read_text()) if Path(production_path).exists() else None
    for m in GATED:
        prod = "n/a" if production is None or m not in production else f"{production[m]:.4f}"
        print(f"{m}: candidate {candidate.get(m, float('nan')):.4f}, production {prod}")
    for w in subreddit_drops(candidate, production):
        print(f"::warning::{w} (drop > {SUBREDDIT_WARN})")
    failures = compare(candidate, production)
    for f in failures:
        print(f"::error::{f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
