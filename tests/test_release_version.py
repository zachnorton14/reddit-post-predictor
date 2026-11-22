"""Deploy versions count up from existing tags, a re-run of the same commit keeps its number, and the gate finds the
latest one."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("release_version",
                                              Path(__file__).parents[1] / "scripts" / "release_version.py")
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)


def test_first_version():
    assert rv.pick("rc", [], []) == "rc-1"


def test_counts_up_from_the_highest_number_not_the_newest_tag():
    assert rv.pick("rc", ["rc-2", "rc-10", "rc-9"], []) == "rc-11"


def test_other_prefixes_and_lookalikes_are_ignored():
    assert rv.pick("rc", ["model-7", "rc-3", "rc-4-hotfix", "xrc-9"], []) == "rc-4"


def test_rerun_of_the_same_commit_reuses_its_version():
    assert rv.pick("rc", ["rc-1", "rc-2", "rc-3"], ["rc-2"]) == "rc-2"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "a")
    git("tag", "rc-1")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "b")
    monkeypatch.chdir(tmp_path)
    return git


def test_latest():
    assert rv.latest("rc", ["rc-2", "rc-10", "model-30", "rc-9"]) == "rc-10"
    assert rv.latest("rc", ["model-1"]) is None


def test_main_reads_git_tags(repo, capsys):
    assert rv.main(["next", "rc", "HEAD"]) == 0
    assert capsys.readouterr().out == "rc-2\n"
    assert rv.main(["next", "rc", "HEAD~1"]) == 0
    assert capsys.readouterr().out == "rc-1\n"
    assert rv.main(["latest", "rc"]) == 0
    assert capsys.readouterr().out == "rc-1\n"
    assert rv.main(["latest", "model"]) == 0
    assert capsys.readouterr().out == ""


def test_bad_arguments():
    assert rv.main(["rc", "HEAD"]) == 2
    assert rv.main(["next", "rc"]) == 2
