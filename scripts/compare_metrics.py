"""Model gate: fail if the candidate model regresses beyond tolerance against the production model.

    python scripts/compare_metrics.py candidate/metrics.json production/metrics.json

Both files are JSON objects with at least the keys in GATED. A missing production file (no model has been
released yet) passes. Exit status 1 blocks the merge into `dev`.
"""

import json
import sys
from pathlib import Path

# Metric -> largest allowed drop (absolute). Higher is better for every gated metric.
GATED = {"pr_auc": 0.01, "top10_hit_rate": 0.01}


def compare(candidate: dict, production: dict | None) -> list[str]:
    """Return one message per regression; an empty list means the candidate passes."""
    missing = [m for m in GATED if m not in candidate]
    if missing:
        return [f"candidate metrics lack {', '.join(missing)}"]
    if production is None:
        return []
    return [
        f"{m}: {candidate[m]:.4f} vs. production {production[m]:.4f} (drop > {tol})"
        for m, tol in GATED.items()
        if m in production and candidate[m] < production[m] - tol
    ]


def main(candidate_path: str, production_path: str) -> int:
    candidate = json.loads(Path(candidate_path).read_text())
    production = json.loads(Path(production_path).read_text()) if Path(production_path).exists() else None
    for m in GATED:
        prod = "n/a" if production is None or m not in production else f"{production[m]:.4f}"
        print(f"{m}: candidate {candidate.get(m, float('nan')):.4f}, production {prod}")
    failures = compare(candidate, production)
    for f in failures:
        print(f"::error::{f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
