"""The rollout probe counts downtime, tracks which version answered, and judges deploys and rollback drills."""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("probe", Path(__file__).parents[1] / "scripts" / "probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def reqs(*runs):
    """Records from runs of (count, version or None for a failure), 0.1 s apart."""
    out, t = [], 0.0
    for count, version in runs:
        for _ in range(count):
            out.append({"t": round(t, 3), "kind": "health", "ok": version is not None,
                        "status": 200 if version else None, "version": version, "ms": 5.0,
                        "error": None if version else "ConnectionRefusedError: refused"})
            t += 0.1
    return out


ROLLOUT = {"from_version": "model-1", "to_version": "model-2", "rolled_back": False, "measured": True}
ROLLED_BACK = {**ROLLOUT, "to_version": "model-1", "rolled_back": True, "reason": "new pods never became Ready"}


def test_summary_of_a_clean_rollout():
    s = probe.summarize(reqs((50, "model-1"), (3, "model-2"), (2, "model-1"), (40, "model-2")))
    assert (s["total"], s["failed"], s["longest_outage_s"]) == (95, 0, 0.0)
    assert [(v["version"], v["count"]) for v in s["versions"]] == [("model-1", 50), ("model-2", 3), ("model-1", 2),
                                                                   ("model-2", 40)]
    assert s["final_version"] == "model-2" and s["p50_ms"] == 5.0


def test_outage_length():
    s = probe.summarize(reqs((10, "model-1"), (4, None), (10, "model-2"), (2, None)))
    assert s["failed"] == 6
    assert s["longest_outage_s"] == pytest.approx(0.4)
    assert s["errors"] == ["ConnectionRefusedError: refused"]


def test_deploy_passes_with_zero_failures_and_the_new_version():
    text, problems = probe.report(probe.summarize(reqs((20, "model-1"), (20, "model-2"))), ROLLOUT, "rollout", None)
    assert problems == [] and "zero failed requests" in text


def test_deploy_fails_on_any_failed_request():
    _, problems = probe.report(probe.summarize(reqs((20, "model-1"), (1, None), (20, "model-2"))), ROLLOUT,
                               "rollout", None)
    assert problems == ["1 of 41 requests failed (longest outage 0.1 s)"]


def test_deploy_fails_when_rolled_back_or_on_the_wrong_version():
    s = probe.summarize(reqs((40, "model-1")))
    _, problems = probe.report(s, ROLLED_BACK, "rollout", None)
    assert "the rollout was rolled back" in problems
    _, problems = probe.report(s, ROLLOUT, "rollout", None)
    assert problems == ["the last version answering was 'model-1', expected 'model-2'"]


def test_drill_needs_a_rollback_and_the_old_version():
    s = probe.summarize(reqs((40, "model-1")))
    assert probe.report(s, ROLLED_BACK, "rollback", None)[1] == []
    _, problems = probe.report(s, ROLLOUT, "rollback", None)
    assert "expected an automatic rollback, but the rollout was not rolled back" in problems
    _, problems = probe.report(probe.summarize(reqs((20, "model-1"), (2, None), (20, "model-1"))), ROLLED_BACK,
                               "rollback", None)
    assert problems == ["2 of 42 requests failed (longest outage 0.2 s)"]


def test_nothing_to_measure():
    first = {"measured": False, "reason": "first production deploy: no earlier release to roll from"}
    text, problems = probe.report(None, first, "rollout", None)
    assert problems == [] and "first production deploy" in text
    assert probe.report(None, first, "rollback", None)[1]  # a drill must roll something back
    _, problems = probe.report(None, ROLLOUT, "rollout", None)  # measured, but the probe left nothing
    assert problems == ["a rollout ran but the probe left no results (probe.json missing)"]


def test_version_override():
    s = probe.summarize(reqs((20, "model-2")))
    assert probe.report(s, ROLLOUT, "rollout", "model-2")[1] == []
    assert probe.report(s, ROLLOUT, "rollout", "model-9")[1]


def rs(rev, broken=False):
    env = [{"name": "MODEL_VERSION", "value": f"model-{rev}"}] + ([probe.BROKEN] if broken else [])
    return {"metadata": {"annotations": {"deployment.kubernetes.io/revision": str(rev)}},
            "spec": {"template": {"spec": {"containers": [{"env": env}]}}}}


def test_undo_target_skips_the_drills_broken_release():
    assert probe.undo_target([rs(1), rs(2)]) == 1
    # broken-release drill: rev 2 broken, rolled back, so rev 1's template came back as rev 3
    assert probe.undo_target([rs(2, broken=True), rs(3), rs(4)]) == 3
    assert probe.undo_target([rs(2, broken=True), rs(3)]) is None
    assert probe.undo_target([rs(1)]) is None
    assert probe.undo_target([]) is None


def test_undo_target_cli(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps({"items": [rs(1), rs(2)]})))
    assert probe.main(["undo-target"]) == 0
    assert capsys.readouterr().out == "1\n"


class Server(BaseHTTPRequestHandler):
    version = "model-7"

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"ok": True, "version": self.version})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self._send({"version": self.version, "predictions": [{"score": 1.5}]})

    def log_message(self, *args):
        pass


def test_run_against_a_server(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    stop = tmp_path / "stop"
    sent = []

    def send(u, kind, timeout):
        if len(sent) == 6:
            stop.touch()
        sent.append(kind)
        return probe.request_once(u, kind, timeout)

    try:
        recs = probe.run(url, tmp_path / "probe.json", stop, interval=0, send=send)
    finally:
        server.shutdown()
    assert [r["kind"] for r in recs] == ["health", "predict"] * 3 + ["health"]
    assert all(r["ok"] and r["version"] == "model-7" for r in recs)
    assert len(json.loads((tmp_path / "probe.json").read_text())["requests"]) == 7
    down = probe.request_once("http://127.0.0.1:1", "health", 0.5)  # nothing listens on port 1
    assert not down["ok"] and down["error"]


def test_cli_report(tmp_path, capsys):
    (tmp_path / "probe.json").write_text(json.dumps({"requests": reqs((5, "model-1"), (5, "model-2"))}))
    (tmp_path / "rollout.json").write_text(json.dumps(ROLLOUT))
    args = ["report", str(tmp_path / "probe.json"), str(tmp_path / "rollout.json"), "--expect"]
    assert probe.main([*args, "rollout", "--out", str(tmp_path / "report.md")]) == 0
    assert "| 10 | 0 | 100.00%" in (tmp_path / "report.md").read_text()
    assert probe.main([*args, "rollback"]) == 1
    capsys.readouterr()
