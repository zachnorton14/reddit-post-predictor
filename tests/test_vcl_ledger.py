import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("vcl_ledger", Path(__file__).parents[1] / "scripts" / "vcl_ledger.py")
ledger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("VCL_LEDGER_DIR", str(tmp_path / "ledger"))
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_RUN_ID", "77")
    state = tmp_path / ".vcl_state.json"
    state.write_text(json.dumps({"request_id": 4237635, "env": "gpu-train-large"}))
    return state


def test_add_and_remove(env):
    path = ledger.add(env)
    assert json.loads(path.read_text()) == {"request_id": 4237635, "repo": "o/r", "run_id": "77"}
    ledger.remove(4237635)
    assert not path.exists()
    ledger.remove(4237635)  # already gone: no error


def test_sweep_ends_only_reservations_of_finished_runs(env, monkeypatch):
    ledger.add(env)
    env.write_text(json.dumps({"request_id": 99}))
    monkeypatch.setenv("GITHUB_RUN_ID", "78")
    ledger.add(env)
    ended = []
    finished = lambda repo, run_id, token: run_id == "77"  # noqa: E731
    assert ledger.sweep("t", finished=finished, end=lambda rid: ended.append(rid) or True) == [4237635]
    assert ended == [4237635]
    assert [p.name for p in ledger.ledger_dir().iterdir()] == ["99.json"]


def test_sweep_keeps_entries_it_cannot_check_or_end(env):
    ledger.add(env)

    def broken(*args):
        raise OSError("network down")

    assert ledger.sweep("t", finished=broken, end=lambda rid: True) == []
    assert ledger.sweep("t", finished=lambda *a: True, end=lambda rid: False) == []
    assert len(list(ledger.ledger_dir().iterdir())) == 1


def test_sweep_with_no_ledger(env):
    assert ledger.sweep("t", finished=lambda *a: True, end=lambda rid: True) == []
