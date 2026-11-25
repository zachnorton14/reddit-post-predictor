"""Measure downtime during a production rollout: send nonstop traffic, then report what failed and which version answered.

    python3 scripts/probe.py run --url http://<service cluster ip>:8000 --out probe.json --stop-file /tmp/probe.stop
    python3 scripts/probe.py report probe.json rollout.json --expect rollout [--version model-4] [--out report.md]

`run` goes on the production VCL instance (infra/ansible/production.yml starts it in the background before the
rollout and creates the stop file after it). It alternates GET /health and POST /predict every --interval seconds,
each with a --timeout, and records every request: time, endpoint, success, HTTP status, the `version` the server
reported (pipeline/serve.py returns it from both endpoints), latency and error. It runs on the instance itself, so
it measures whether the Kubernetes Service kept answering, not the network path from outside.

`report` runs on the controller. It reads the requests and rollout.json (written by the playbook: `from`, `to`,
`from_version`, `to_version`, `rolled_back`, `reason`, `measured`) and prints a Markdown summary. Exit status 1 if:
  --expect rollout   any request failed, the rollout was rolled back, or the last version answering isn't the new one
  --expect rollback  any request failed, the rollout was NOT rolled back, or the last version isn't the old one
The new and old versions come from rollout.json (`to_version`, `from_version`); --version overrides.
A run with nothing to measure (first deploy ever, or the image was already live) passes with a note.

    k3s kubectl -n prod get rs -l app=serve -o json | python3 scripts/probe.py undo-target

prints the revision the undo drill goes back to: the newest earlier release that the broken-release drill didn't
break (a plain `rollout undo` would go back to the broken one, which Kubernetes keeps in the history).

Standard library only, Python 3.10+: the instance runs it with the system python3.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TEST_POST = {"posts": [{"title": "I used to hate facial hair", "body": "but then it grew on me."}]}


# -- run ------------------------------------------------------------------------------------------------------------


def request_once(url: str, kind: str, timeout: float) -> dict:
    """One request. Returns {kind, ok, status, version, ms, error}; never raises."""
    start = time.perf_counter()
    rec = {"kind": kind, "ok": False, "status": None, "version": None, "ms": None, "error": None}
    try:
        if kind == "health":
            req = urllib.request.Request(f"{url}/health")
        else:
            req = urllib.request.Request(f"{url}/predict", data=json.dumps(TEST_POST).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.load(resp)
            rec["status"] = resp.status
        rec["version"] = body.get("version")
        rec["ok"] = rec["status"] == 200 and (kind == "health" or len(body.get("predictions") or []) == 1)
        if not rec["ok"]:
            rec["error"] = "unexpected answer"
    except urllib.error.HTTPError as e:
        rec["status"], rec["error"] = e.code, f"HTTP {e.code}"
    except Exception as e:  # connection refused/reset, timeout, bad JSON: all count as downtime
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    rec["ms"] = round(1000 * (time.perf_counter() - start), 1)
    return rec


def run(url: str, out: Path, stop_file: Path, interval: float = 0.1, timeout: float = 2.0, max_seconds: float = 1200,
        send=request_once, clock=time.monotonic, sleep=time.sleep) -> list[dict]:
    """Probe until stop_file exists (or max_seconds pass), then write the requests to `out`."""
    records, t0, i = [], clock(), 0
    while not stop_file.exists() and clock() - t0 < max_seconds:
        rec = send(url, "health" if i % 2 == 0 else "predict", timeout)
        rec["t"] = round(clock() - t0, 3)
        records.append(rec)
        i += 1
        sleep(interval)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"url": url, "interval": interval, "timeout": timeout, "requests": records}))
    tmp.replace(out)
    return records


# -- report ---------------------------------------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def summarize(records: list[dict]) -> dict:
    """Totals, the longest outage, latency, and which version answered when (in order)."""
    failed = [r for r in records if not r["ok"]]
    longest, start = 0.0, None
    for r in records:
        if not r["ok"] and start is None:
            start = r["t"]
        elif r["ok"] and start is not None:
            longest, start = max(longest, r["t"] - start), None
    if start is not None:  # still down when the probe stopped
        longest = max(longest, records[-1]["t"] - start)
    versions: list[dict] = []  # runs of consecutive successful answers from one version
    for r in records:
        if not r["ok"]:
            continue
        if versions and versions[-1]["version"] == r["version"]:
            versions[-1].update(last=r["t"], count=versions[-1]["count"] + 1)
        else:
            versions.append({"version": r["version"], "first": r["t"], "last": r["t"], "count": 1})
    ok_ms = [r["ms"] for r in records if r["ok"]]
    return {"total": len(records), "failed": len(failed), "longest_outage_s": round(longest, 3),
            "duration_s": records[-1]["t"] if records else 0.0,
            "p50_ms": percentile(ok_ms, 0.5), "p95_ms": percentile(ok_ms, 0.95),
            "final_version": versions[-1]["version"] if versions else None, "versions": versions,
            "errors": sorted({r["error"] for r in failed if r["error"]})[:5]}


def report(summary: dict | None, rollout: dict, expect: str, version: str | None) -> tuple[str, list[str]]:
    """Markdown report and the list of problems (empty means it passed)."""
    problems: list[str] = []
    lines = [f"### Rollout `{rollout.get('from_version') or '-'}` → `{rollout.get('to_version') or '-'}`"]
    if rollout.get("rolled_back"):
        lines += ["", f"**Rolled back automatically**: {rollout.get('reason') or 'the rollout failed'}"]
    if not rollout.get("measured"):
        lines += ["", f"Nothing to measure: {rollout.get('reason') or 'no rollout under traffic'}."]
        if expect == "rollback":
            problems.append("no rollout ran under traffic, so there was nothing to roll back")
        lines += ["", *[f"- ❌ {p}" for p in problems]] if problems else ["", "✅ pass"]
        return "\n".join(lines), problems
    if summary is None:
        problems.append("a rollout ran but the probe left no results (probe.json missing)")
        return "\n".join([*lines, "", *[f"- ❌ {p}" for p in problems]]), problems

    s = summary
    rate = 100 * (s["total"] - s["failed"]) / s["total"] if s["total"] else 0.0
    lines += ["", "| Requests | Failed | Availability | Longest outage | p50 | p95 | Duration |",
              "|---|---|---|---|---|---|---|",
              f"| {s['total']} | {s['failed']} | {rate:.2f}% | {s['longest_outage_s']} s | {s['p50_ms']} ms "
              f"| {s['p95_ms']} ms | {s['duration_s']} s |",
              "", "Which version answered (in order):", "", "| Version | From | To | Requests |", "|---|---|---|---|"]
    lines += [f"| `{v['version']}` | {v['first']} s | {v['last']} s | {v['count']} |" for v in s["versions"]]
    if s["errors"]:
        lines += ["", "Errors: " + "; ".join(f"`{e}`" for e in s["errors"])]

    if s["total"] == 0:
        problems.append("the probe sent no requests")
    if s["failed"]:
        problems.append(f"{s['failed']} of {s['total']} requests failed (longest outage {s['longest_outage_s']} s)")
    if expect == "rollout" and rollout.get("rolled_back"):
        problems.append("the rollout was rolled back")
    if expect == "rollback" and not rollout.get("rolled_back"):
        problems.append("expected an automatic rollback, but the rollout was not rolled back")
    expected = version or rollout.get("from_version" if expect == "rollback" else "to_version")
    if expected and s["final_version"] != expected:
        problems.append(f"the last version answering was {s['final_version']!r}, expected {expected!r}")
    lines += ["", *[f"- ❌ {p}" for p in problems]] if problems else ["", "✅ pass: zero failed requests"]
    return "\n".join(lines), problems


# -- undo-target ----------------------------------------------------------------------------------------------------

BROKEN = {"name": "MODEL_DIR", "value": "/model-missing"}  # what infra/k8s/serve.yaml.j2 sets for serve_break


def undo_target(replicasets: list[dict]) -> int | None:
    """The revision `rollout undo` should go back to: the newest one before the current that the drill didn't break.

    The current revision is the highest. A plain `rollout undo` goes to the one before it, which after a
    broken-release drill is the broken one (Kubernetes keeps it in the history).
    """
    revs = []
    for rs in replicasets:
        rev = int(rs["metadata"].get("annotations", {}).get("deployment.kubernetes.io/revision", 0))
        env = rs["spec"]["template"]["spec"]["containers"][0].get("env") or []
        revs.append((rev, BROKEN in env))
    if not revs:
        return None
    current = max(r for r, _ in revs)
    good = [r for r, broken in revs if r < current and not broken]
    return max(good) if good else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("undo-target", help="read `kubectl get rs -o json` on stdin, print the revision to undo to")
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--stop-file", required=True, type=Path)
    r.add_argument("--interval", type=float, default=0.1)
    r.add_argument("--timeout", type=float, default=2.0)
    r.add_argument("--max-seconds", type=float, default=1200)
    rep = sub.add_parser("report")
    rep.add_argument("probe", type=Path, help="probe.json (may be missing when nothing was measured)")
    rep.add_argument("rollout", type=Path)
    rep.add_argument("--expect", choices=["rollout", "rollback"], required=True)
    rep.add_argument("--version", help="the version that must be answering at the end")
    rep.add_argument("--out", type=Path, help="also write the Markdown here")
    a = p.parse_args(argv)

    if a.cmd == "undo-target":
        rev = undo_target(json.load(sys.stdin)["items"])
        if rev is None:
            print("no earlier working release to undo to", file=sys.stderr)
            return 1
        print(rev)
        return 0
    if a.cmd == "run":
        recs = run(a.url, a.out, a.stop_file, a.interval, a.timeout, a.max_seconds)
        print(f"{len(recs)} requests, {sum(not r['ok'] for r in recs)} failed")
        return 0
    rollout = json.loads(a.rollout.read_text())
    summary = summarize(json.loads(a.probe.read_text())["requests"]) if a.probe.exists() else None
    text, problems = report(summary, rollout, a.expect, a.version)
    print(text)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
