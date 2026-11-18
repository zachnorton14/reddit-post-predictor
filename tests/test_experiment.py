"""pipeline/experiment.py without a GPU or network: config checks, the run plan, the runner command, the summary."""

import sys
from pathlib import Path

import pytest

from pipeline import experiment as ex

GOOD = """runner: run_pooling.py
runs:
  - {model: deberta_attn4, seeds: [42, 43]}
  - {model: deberta_large_attn4, seeds: [42]}
"""


def write(tmp_path, text, name="pooling_round1.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_good_config(tmp_path):
    cfg = ex.load_config(write(tmp_path, GOOD + "epochs: 2\nallow_fallback: true\n"))
    assert cfg["name"] == "pooling_round1"
    assert (cfg["runner"], cfg["epochs"], cfg["allow_fallback"]) == ("run_pooling.py", 2, True)
    assert ex.plan(cfg) == [("deberta_attn4", 42), ("deberta_attn4", 43), ("deberta_large_attn4", 42)]


def test_defaults_and_repeats(tmp_path):
    cfg = ex.load_config(write(tmp_path, "runner: run_pooling.py\nruns:\n  - {model: m, seeds: [1, 1]}\n"
                                         "  - {model: m, seeds: [1, 2]}\n"))
    assert (cfg["epochs"], cfg["allow_fallback"], cfg["type"]) == (None, False, "prototype")
    assert ex.plan(cfg) == [("m", 1), ("m", 2)]


@pytest.mark.parametrize("text, message", [
    ("runs: [{model: m, seeds: [1]}]\n", "runner"),
    ("runner: ../evil.py\nruns: [{model: m, seeds: [1]}]\n", "runner"),
    ("runner: run_pooling.py\n", "runs"),
    ("runner: run_pooling.py\nruns: []\n", "runs"),
    ("runner: run_pooling.py\nruns: [{model: m}]\n", "exactly"),
    ("runner: run_pooling.py\nruns: [{model: 'm; rm -rf /', seeds: [1]}]\n", "model"),
    ("runner: run_pooling.py\nruns: [{model: m, seeds: [one]}]\n", "seeds"),
    ("runner: run_pooling.py\nruns: [{model: m, seeds: [true]}]\n", "seeds"),
    ("runner: run_pooling.py\nruns: [{model: m, seeds: [1]}]\nepochs: 0\n", "epochs"),
    ("runner: run_pooling.py\nruns: [{model: m, seeds: [1]}]\nallow_fallback: yes please\n", "allow_fallback"),
    ("runner: run_pooling.py\nruns: [{model: m, seeds: [1]}]\ngpu: h100\n", "unknown keys"),
    ("runner: run_pooling.py\nruns: [{model: m, seeds: [1]}]\ntype: ../x\n", "type"),
    ("runner: run_pooling.py\nruns: [{model: m, seeds: [1]}\n", "YAML"),
    ("- just a list\n", "mapping"),
])
def test_bad_configs_fail_clearly(tmp_path, text, message):
    with pytest.raises(ex.ConfigError, match=message):
        ex.load_config(write(tmp_path, text))


def test_bad_file_name(tmp_path):
    with pytest.raises(ex.ConfigError, match="file name"):
        ex.load_config(write(tmp_path, GOOD, name="Round 1.yaml"))


def test_too_many_runs(tmp_path):
    seeds = list(range(ex.MAX_RUNS + 1))
    with pytest.raises(ex.ConfigError, match="more than"):
        ex.load_config(write(tmp_path, f"runner: run_pooling.py\nruns: [{{model: m, seeds: {seeds}}}]\n"))


def test_check_command(tmp_path, capsys):
    good, bad = write(tmp_path, GOOD), write(tmp_path, "runs: []\n", name="bad.yaml")
    assert ex.main(["--check", str(good)]) == 0
    assert "3 runs with run_pooling.py" in capsys.readouterr().out
    assert ex.main(["--check", str(good), str(bad)]) == 1
    assert "::error::" in capsys.readouterr().err


def test_runner_command_follows_the_contract(tmp_path):
    cfg = ex.load_config(write(tmp_path, GOOD + "epochs: 3\n"))
    cmd = ex.runner_command(cfg, "deberta_attn4", 43, Path("/out"), "o/d", "abc123")
    assert cmd == [sys.executable, "run_pooling.py", "--models", "deberta_attn4", "--seeds", "43", "--output", "/out/runs",
                   "--results", "/out/results", "--resume", "--dataset", "o/d", "--revision", "abc123", "--epochs", "3"]
    assert ex.hf_prefix(cfg, "0123456789abcdef") == "experiments/pooling_round1/0123456"


def ev(ap, top, rho, **by):
    return {"pr_auc": ap, "top10_hit_rate": top, "spearman": rho, "by_subreddit": {k: {"pr_auc": v} for k, v in by.items()}}


def test_summary_table(tmp_path):
    cfg = ex.load_config(write(tmp_path, GOOD))
    rows = [{"run": "deberta_attn4_s42", "model": "deberta_attn4", "status": "done", "seconds": 3600,
             "eval": ev(0.30, 0.33, 0.41, a=0.35, b=0.25)},
            {"run": "deberta_attn4_s43", "model": "deberta_attn4", "status": "already done", "seconds": 3000,
             "eval": ev(0.28, 0.31, 0.40, a=0.30)},
            {"run": "deberta_large_attn4_s42", "model": "deberta_large_attn4", "status": "failed (exit 1)"}]
    champion = {"name": "dadjokes_models-22M-minilm_l6", **ev(0.25, 0.30, 0.36, a=0.2, b=0.3)}
    md = ex.summarize(cfg, "0123456789", rows, champion)
    assert "| **champion** dadjokes_models-22M-minilm_l6 | deployed | 0.2500 | 0.3000 | 0.3600 | – |" in md
    assert "| deberta_attn4_s42 | done | 0.3000 | 0.3300 | 0.4100 | 60 |" in md
    assert "| deberta_large_attn4_s42 | failed (exit 1) | – | – | – | – |" in md
    assert "| deberta_attn4 | 2 | 0.2900 | 0.3200 | 0.4050 |" in md  # mean over the two seeds
    assert "| Run | a | b |" in md and "| deberta_attn4_s43 | 0.3000 | – |" in md  # per-subreddit AP
    assert "Advisory only" in md and "experiments/pooling_round1/0123456/" in md
    assert "No champion" in ex.summarize(cfg, "0123456789", rows, None)


FAKE_RUNNER = '''import argparse, json, pathlib, sys
p = argparse.ArgumentParser()
for a in ["--models", "--seeds", "--output", "--results", "--dataset", "--revision", "--epochs"]:
    p.add_argument(a)
p.add_argument("--resume", action="store_true")
a = p.parse_args()
if a.models == "broken":
    sys.exit(3)
run = pathlib.Path(a.output) / f"{a.models}_s{a.seeds}"
run.mkdir(parents=True)
(run / "best.json").write_text(json.dumps({"ap": 0.3, "precision_top10": 0.33, "spearman": 0.4, "seconds": 60}))
(run / "resume.pt").write_text("big")
'''


def test_main_runs_uploads_and_skips(tmp_path, monkeypatch):
    import huggingface_hub

    (tmp_path / "fake_runner.py").write_text(FAKE_RUNNER)
    cfg = write(tmp_path, "runner: fake_runner.py\nruns:\n  - {model: m, seeds: [1, 2]}\n"
                          "  - {model: broken, seeds: [1]}\n", name="round1.yaml")
    done = tmp_path / "done_best.json"
    done.write_text('{"ap": 0.2, "precision_top10": 0.25, "spearman": 0.3}')
    uploads = []

    class FakeApi:
        def file_exists(self, repo, path):
            return path == "experiments/round1/abcdef0/m_s2/best.json"  # m_s2 finished in an earlier reservation

        def upload_folder(self, **kw):
            uploads.append((kw["path_in_repo"], sorted(p.name for p in Path(kw["folder_path"]).iterdir()),
                            kw["ignore_patterns"]))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, path: str(done))
    monkeypatch.setattr(ex, "ROOT", tmp_path)
    monkeypatch.setattr(ex, "score_champion", lambda *a: None)
    scored = []

    def fake_eval(model_dir, dataset, revision, result, meta=None):
        scored.append((Path(model_dir).name, meta))
        Path(result).write_text("{}")
        return ev(0.31, 0.34, 0.42)

    monkeypatch.setattr(ex, "evaluate_val", fake_eval)
    out = tmp_path / "out"
    code = ex.main(["--config", str(cfg), "--dataset", "o/d", "--revision", "r", "--commit", "abcdef0123", "--out", str(out)])

    assert code == 1  # one run failed
    assert scored == [("m_s1", {"name": "m_s1", "type": "prototype"})]
    assert uploads == [("experiments/round1/abcdef0/m_s1", ["best.json", "experiment.json", "resume.pt", "val_eval.json"],
                        ["resume.pt", "best_checkpoint.pt"])]
    summary = (out / "summary.md").read_text()
    assert "| m_s1 | done | 0.3100 |" in summary
    assert "| m_s2 | already done | – |" in summary  # finished before validation scoring existed
    assert "| broken_s1 | failed (exit 3) |" in summary
