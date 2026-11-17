"""Keep a list of the VCL reservations that pipeline jobs hold, so a reservation whose job died can still be ended.

    python scripts/vcl_ledger.py add                 after `vcl up`: records .vcl_state.json's request with this run
    python scripts/vcl_ledger.py remove <request id> after `vcl down`
    python scripts/vcl_ledger.py sweep               ends recorded reservations whose workflow run has finished

A job ends its own reservation in an `always()` step. That step can't run if the controller crashes or the runner
loses its connection, and the reservation would then stay open until its length runs out (up to 10 hours).
vcl-cleanup.yml runs `sweep` on the controller every 30 minutes. It ends only reservations recorded here, so
reservations people make by hand are never touched. Entries live in ~/.vcl-pipeline/ on the controller
($VCL_LEDGER_DIR to override). Needs $GITHUB_REPOSITORY and $GITHUB_RUN_ID (add) or $GH_TOKEN (sweep).
"""

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


def ledger_dir() -> Path:
    return Path(os.environ.get("VCL_LEDGER_DIR", Path.home() / ".vcl-pipeline"))


def add(state_file=".vcl_state.json") -> Path:
    request_id = json.loads(Path(state_file).read_text())["request_id"]
    entry = {"request_id": request_id, "repo": os.environ["GITHUB_REPOSITORY"], "run_id": os.environ["GITHUB_RUN_ID"]}
    ledger_dir().mkdir(parents=True, exist_ok=True)
    path = ledger_dir() / f"{request_id}.json"
    path.write_text(json.dumps(entry))
    return path


def remove(request_id) -> None:
    (ledger_dir() / f"{request_id}.json").unlink(missing_ok=True)


def run_finished(repo, run_id, token) -> bool:
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/actions/runs/{run_id}",
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)["status"] == "completed"


def end_reservation(request_id) -> bool:
    return subprocess.run([sys.executable, "-m", "vcl", "down", str(request_id)]).returncode == 0


def sweep(token, finished=run_finished, end=end_reservation) -> list[int]:
    """End and forget the reservations whose run has finished. Returns their request ids."""
    ended = []
    for path in sorted(ledger_dir().glob("*.json")) if ledger_dir().exists() else []:
        entry = json.loads(path.read_text())
        try:
            done = finished(entry["repo"], entry["run_id"], token)
        except Exception as e:  # try again next sweep
            print(f"::warning::can't check run {entry['run_id']}: {e}", file=sys.stderr)
            continue
        if not done:
            print(f"reservation {entry['request_id']}: run {entry['run_id']} still going")
            continue
        print(f"::warning::run {entry['run_id']} finished without ending reservation {entry['request_id']}; ending it")
        if end(entry["request_id"]):
            path.unlink()
            ended.append(entry["request_id"])
    return ended


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["add"]:
        print(f"recorded {add()}")
    elif argv[:1] == ["remove"] and len(argv) == 2:
        remove(argv[1])
    elif argv[:1] == ["sweep"]:
        print(f"ended {len(sweep(os.environ['GH_TOKEN']))} leftover reservations")
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
