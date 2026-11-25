"""Remember which VCL reservation is the staging (or production) server, so the next deploy can replace or reuse it.

    python scripts/vcl_staging.py current [--role R]                  prints the recorded request id (nothing if none)
    python scripts/vcl_staging.py live [--role R]                     prints it only if VCL says it is still ready
    python scripts/vcl_staging.py record <image> <version> [--role R] after a deploy passes: .vcl_state.json becomes R

R is `staging` (default, deploy.yml) or `production` (deploy-production.yml, production-provision.yml,
rollback-drill.yml). Neither is in the vcl_ledger.py ledger: the cleanup sweep would end it once the deploy run
finished. Each lives until its reservation ends (the profile's length_minutes) or the next deploy replaces it. Staging
is replaced by every deploy; production is reused while it is `live` and rolled onto in place. The record is
~/.vcl-pipeline/<role>.json ($VCL_LEDGER_DIR to override, like the ledger). `live` needs $VCL_TOKEN.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROLES = ("staging", "production")


def record_path(role: str = "staging") -> Path:
    if role not in ROLES:
        raise ValueError(f"role {role!r}: expected one of {', '.join(ROLES)}")
    return Path(os.environ.get("VCL_LEDGER_DIR", Path.home() / ".vcl-pipeline")) / f"{role}.json"


def current(role: str = "staging") -> dict | None:
    path = record_path(role)
    return json.loads(path.read_text()) if path.exists() else None


def vcl_status(request_id: int) -> str | None:
    """VCL's status for a reservation ("ready", "loading", ...), or None if VCL no longer knows it (ended)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo's vcl package
    from vcl.client import VCLClient, VCLError
    from vcl.config import resolve_token

    try:
        return VCLClient(resolve_token()).get_request_status(request_id).get("status")
    except VCLError:
        return None


def live(role: str = "staging", status=vcl_status) -> dict | None:
    """The recorded reservation if VCL still reports it ready, else None (expired, ended, or never recorded)."""
    entry = current(role)
    return entry if entry and status(int(entry["request_id"])) == "ready" else None


def record(image: str, version: str, state_file=".vcl_state.json", now=None, role: str = "staging") -> dict:
    state = json.loads(Path(state_file).read_text())
    entry = {"request_id": state["request_id"], "ip": state.get("ip"), "user": state.get("user"),
             "port": state.get("port"), "image": image, "version": version,
             "repo": os.environ.get("GITHUB_REPOSITORY"), "run_id": os.environ.get("GITHUB_RUN_ID"),
             "deployed_at": int(time.time() if now is None else now)}
    path = record_path(role)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=2) + "\n")
    return entry


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["current", "live", "record"])
    p.add_argument("args", nargs="*")
    p.add_argument("--role", choices=ROLES, default="staging")
    p.add_argument("--json", action="store_true", help="print the whole record instead of the request id")
    try:
        a = p.parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as e:
        return 2 if e.code else 0
    if a.cmd in ("current", "live") and not a.args:
        entry = current(a.role) if a.cmd == "current" else live(a.role)
        if entry:
            print(json.dumps(entry) if a.json else entry["request_id"])
    elif a.cmd == "record" and len(a.args) == 2:
        print(json.dumps(record(a.args[0], a.args[1], role=a.role)))
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
