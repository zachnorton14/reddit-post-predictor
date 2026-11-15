"""scripts/find_candidate.sh against a throwaway git repo and a fake `gh` that lists train-eval runs."""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "find_candidate.sh"


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def commit(repo, path, text):
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text)
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", path)
    return git(repo, "rev-parse", "HEAD")


def run(repo, tmp_path, runs, head):
    """runs: newest first, as (run_id, sha)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text("#!/bin/sh\ncat <<'EOF'\n" + "".join(f"{i} {s}\n" for i, s in runs) + "EOF\n")
    gh.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    return subprocess.run(["bash", str(SCRIPT), head], cwd=repo, env=env, capture_output=True, text=True)


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    return repo


def test_docs_only_change_reuses_the_trained_candidate(tmp_path):
    repo = make_repo(tmp_path)
    trained = commit(repo, "pipeline/train.py", "v1")
    head = commit(repo, "docs/pipeline.md", "notes")
    r = run(repo, tmp_path, [("101", trained)], head)
    assert r.returncode == 0 and r.stdout.strip() == "101"


def test_model_change_after_training_fails(tmp_path):
    repo = make_repo(tmp_path)
    trained = commit(repo, "pipeline/train.py", "v1")
    head = commit(repo, "dadjokes_models.py", "new model")
    r = run(repo, tmp_path, [("101", trained)], head)
    assert r.returncode == 1 and "dadjokes_models.py" in r.stderr


def test_picks_newest_ancestor_and_skips_other_branches(tmp_path):
    repo = make_repo(tmp_path)
    old = commit(repo, "pipeline/train.py", "v1")
    new = commit(repo, "pipeline/train.py", "v2")
    git(repo, "checkout", "-qb", "other", old)
    unrelated = commit(repo, "pipeline/train.py", "side")
    git(repo, "checkout", "-q", "-")
    r = run(repo, tmp_path, [("300", unrelated), ("200", new), ("100", old)], new)
    assert r.returncode == 0 and r.stdout.strip() == "200"


def test_no_passing_run_fails(tmp_path):
    repo = make_repo(tmp_path)
    head = commit(repo, "pipeline/train.py", "v1")
    r = run(repo, tmp_path, [], head)
    assert r.returncode == 1 and "No passing train-eval run" in r.stderr
