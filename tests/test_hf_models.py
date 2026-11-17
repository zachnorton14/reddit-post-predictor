import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("hf_models", Path(__file__).parents[1] / "scripts" / "hf_models.py")
hf_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf_models)


@pytest.mark.parametrize("name", ["dadjokes_models-22M-minilm_l6-hgb_blend", "deberta-184M", "d1-0.6B-lora_r16"])
def test_good_names_map_to_models_folder(name):
    assert hf_models.model_path({"name": name}) == f"models/{name}"


@pytest.mark.parametrize("name", ["", "DadJokes-22M", "dadjokes_models", "dadjokes_models-22", "x-22M-Has Space",
                                  "20261114-d585ba9-43628212078"])
def test_bad_names_are_rejected(name):
    with pytest.raises(ValueError):
        hf_models.model_path({"name": name})


ROOT = Path(__file__).parents[1]
RELEASE = {"path": "experiments/ladder/abc1234/deberta_large_attn4_s42", "revision": "r0", "name": "run_pooling-435M-x",
           "type": "prototype", "commit": "abc1234def"}


def test_repo_candidate_file_is_valid():
    from pipeline import evaluate

    cand = hf_models.load_candidate(ROOT / "candidate.yaml")
    assert cand["type"] in evaluate.LOADERS


@pytest.mark.parametrize("text, message", [
    ("path: models/x\nname: x-1M\n", "exactly"),
    ("path: ../secrets\nname: x-1M\ntype: minilm\n", "path"),
    ("path: experiments/a/../../b\nname: x-1M\ntype: minilm\n", "path"),
    ("path: other/x\nname: x-1M\ntype: minilm\n", "path"),
    ("path: models/x\nname: X\ntype: minilm\n", "name"),
    ("path: models/x\nname: x-1M\ntype: 'a; rm'\n", "type"),
])
def test_bad_candidate_files(tmp_path, text, message):
    (tmp_path / "c.yaml").write_text(text)
    with pytest.raises(ValueError, match=message):
        hf_models.load_candidate(tmp_path / "c.yaml")


def fake_download(files):
    def snapshot_download(repo, revision, allow_patterns, local_dir):
        for rel, text in files.items():
            (Path(local_dir) / rel).parent.mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / rel).write_text(text)
    return snapshot_download


def test_fetch_writes_model_json_for_experiment_runs(tmp_path, monkeypatch):
    import huggingface_hub

    path = RELEASE["path"]
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download({f"{path}/head.pt": "w"}))
    meta = hf_models.fetch(path, "r0", tmp_path / "c", "run_pooling-435M-x", "prototype")
    assert meta == json.loads((tmp_path / "c" / "model.json").read_text()) == {
        "type": "prototype", "name": "run_pooling-435M-x", "hf_path": path}
    assert (tmp_path / "c" / "head.pt").exists()
    with pytest.raises(ValueError, match="--name and --type"):
        hf_models.fetch(path, "r0", tmp_path / "d")


def test_fetch_checks_an_existing_model_json(tmp_path, monkeypatch):
    import huggingface_hub

    files = {"models/m/model/model.json": json.dumps({"type": "minilm", "name": "m-1M", "mu": 1})}
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download(files))
    assert hf_models.fetch("models/m/model", "r0", tmp_path / "a", "m-1M", "minilm")["mu"] == 1
    with pytest.raises(ValueError, match="does not match"):
        hf_models.fetch("models/m/model", "r0", tmp_path / "b", "other-1M", "minilm")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download({}))
    with pytest.raises(ValueError, match="not found"):
        hf_models.fetch("models/gone", "r0", tmp_path / "c")


class FakeApi:
    """The repo as {revision: {path: blob id}}. create_commit adds revision r1."""

    def __init__(self, change_weights=False):
        p = RELEASE["path"]
        self.trees = {"r0": {f"{p}/head.pt": "h", f"{p}/encoder/model.safetensors": "e"}}
        self.change_weights, self.calls = change_weights, []

    def list_repo_tree(self, repo, path_in_repo, revision, recursive):
        from types import SimpleNamespace
        return [SimpleNamespace(path=k, blob_id=v) for k, v in self.trees[revision].items() if k.startswith(path_in_repo)]

    def file_exists(self, repo, path, revision=None):
        return path in self.trees[revision or "r1"]

    def create_commit(self, repo, operations, commit_message):
        from types import SimpleNamespace
        tree = dict(self.trees["r0"], **{op.path_in_repo: "new" for op in operations})
        if self.change_weights:
            tree[f"{RELEASE['path']}/head.pt"] = "retrained"
        self.trees["r1"] = tree
        self.calls.append(("commit", sorted(op.path_in_repo for op in operations)))
        return SimpleNamespace(oid="r1")

    def create_tag(self, repo, tag, revision):
        self.calls.append(("tag", tag, revision))

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, commit_message):
        self.calls.append(("upload", path_in_repo, json.loads(path_or_fileobj)))


def test_promote_records_tags_and_points_champion_at_the_scored_weights():
    api = FakeApi()
    champ = hf_models.promote(api, RELEASE, "model-3")
    p = RELEASE["path"]
    assert champ == {"version": "model-3", "name": "run_pooling-435M-x", "type": "prototype", "path": p,
                     "commit": "abc1234def", "revision": "r1"}
    assert api.calls == [("commit", [f"{p}/model.json", f"{p}/release.json"]), ("tag", "model-3", "r1"),
                         ("upload", "champion.json", champ)]


def test_promote_refuses_weights_that_changed_after_the_gate():
    api = FakeApi(change_weights=True)
    with pytest.raises(ValueError, match="changed after the gate"):
        hf_models.promote(api, RELEASE, "model-3")
    assert [c[0] for c in api.calls] == ["commit"]  # no tag, no champion.json


def test_promote_without_revision_fails_before_touching_hf(tmp_path, capsys):
    (tmp_path / "release.json").write_text(json.dumps({k: v for k, v in RELEASE.items() if k != "revision"}))
    assert hf_models.main(["promote", str(tmp_path / "release.json"), "model-1"]) == 1
    assert "revision" in capsys.readouterr().err


def test_record_revision(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"commit": "abc"}))
    hf_models.record_revision(tmp_path, "f00d")
    assert json.loads((tmp_path / "run.json").read_text()) == {"commit": "abc", "hf_revision": "f00d"}
    hf_models.record_revision(tmp_path / "missing", "f00d")  # no run.json: nothing to do


def test_path_command(tmp_path, capsys):
    f = tmp_path / "model.json"
    f.write_text(json.dumps({"name": "deberta-184M", "type": "deberta"}))
    assert hf_models.main(["path", str(f)]) == 0
    assert capsys.readouterr().out.strip() == "models/deberta-184M"
    f.write_text(json.dumps({"type": "deberta"}))
    assert hf_models.main(["path", str(f)]) == 1
