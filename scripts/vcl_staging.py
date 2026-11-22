"""Remember which VCL reservation is the staging server, so the next deploy can replace it.

    python scripts/vcl_staging.py current                  prints the live staging request id (nothing if none)
    python scripts/vcl_staging.py record <image> <version> after a deploy passes: .vcl_state.json becomes staging

deploy.yml runs these on the self-hosted controller. Staging is deliberately not in the vcl_ledger.py ledger: the
cleanup sweep would end it once the deploy run finished. It lives until its reservation ends (cpu-staging's
length_minutes) or the next deploy replaces it. The record is ~/.vcl-pipeline/staging.json ($VCL_LEDGER_DIR to
override, like the ledger).
"""

import json
import os
import sys
import time
from pathlib import Path


def record_path() -> Path:
    return Path(os.environ.get("VCL_LEDGER_DIR", Path.home() / ".vcl-pipeline")) / "staging.json"


def current() -> dict | None:
    path = record_path()
    return json.loads(path.read_text()) if path.exists() else None


def record(image: str, version: str, state_file=".vcl_state.json", now=None) -> dict:
    state = json.loads(Path(state_file).read_text())
    entry = {"request_id": state["request_id"], "ip": state.get("ip"), "user": state.get("user"),
             "port": state.get("port"), "image": image, "version": version,
             "repo": os.environ.get("GITHUB_REPOSITORY"), "run_id": os.environ.get("GITHUB_RUN_ID"),
             "deployed_at": int(time.time() if now is None else now)}
    path = record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=2) + "\n")
    return entry


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["current"]:
        entry = current()
        if entry:
            print(entry["request_id"])
    elif argv[:1] == ["record"] and len(argv) == 3:
        print(json.dumps(record(argv[1], argv[2])))
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
