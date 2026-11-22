"""The staging record names the live staging reservation, and stays out of the cleanup ledger."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("vcl_staging", Path(__file__).parents[1] / "scripts" / "vcl_staging.py")
staging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staging)


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("VCL_LEDGER_DIR", str(tmp_path / "ledger"))
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_RUN_ID", "77")
    path = tmp_path / ".vcl_state.json"
    path.write_text(json.dumps({"request_id": 555, "env": "cpu-staging", "ip": "10.0.0.5", "user": "u", "port": 22}))
    return path


def test_no_staging_yet(state, capsys):
    assert staging.current() is None
    assert staging.main(["current"]) == 0
    assert capsys.readouterr().out == ""


def test_record_replaces_the_previous_staging(state, capsys):
    staging.record("ghcr.io/o/r/serve:model-1", "model-1", state, now=100)
    state.write_text(json.dumps({"request_id": 556, "ip": "10.0.0.6"}))
    staging.record("ghcr.io/o/r/serve:model-2", "model-2", state, now=200)
    assert staging.current() == {"request_id": 556, "ip": "10.0.0.6", "user": None, "port": None,
                                 "image": "ghcr.io/o/r/serve:model-2", "version": "model-2", "repo": "o/r",
                                 "run_id": "77", "deployed_at": 200}
    staging.main(["current"])
    assert capsys.readouterr().out == "556\n"


def test_staging_is_not_in_the_ledger(state):
    staging.record("img", "model-1", state)
    assert [p.name for p in staging.record_path().parent.iterdir()] == ["staging.json"]
    assert not list(staging.record_path().parent.glob("[0-9]*.json"))  # vcl_ledger.py's entries are <id>.json


def test_bad_arguments():
    assert staging.main(["record", "img"]) == 2
