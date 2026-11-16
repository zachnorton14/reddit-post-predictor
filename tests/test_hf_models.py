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


def test_champion_record():
    run = {"commit": "abc1234def", "hf_path": "models/deberta-184M", "hf_revision": "f00d"}
    assert hf_models.champion_record(run, {"name": "deberta-184M"}, "model-3") == {
        "version": "model-3", "name": "deberta-184M", "path": "models/deberta-184M", "commit": "abc1234def",
        "revision": "f00d"}


def test_champion_record_needs_the_upload_commit():
    with pytest.raises(ValueError, match="hf_revision"):
        hf_models.champion_record({"commit": "abc", "hf_path": "models/x-1M"}, {"name": "x-1M"}, "model-1")


def test_promote_without_revision_fails_before_touching_hf(tmp_path, capsys):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "model.json").write_text(json.dumps({"name": "x-1M", "type": "minilm"}))
    (tmp_path / "run.json").write_text(json.dumps({"commit": "abc", "hf_path": "models/x-1M"}))
    assert hf_models.main(["promote", str(tmp_path / "run.json"), "model-1"]) == 1
    assert "hf_revision" in capsys.readouterr().err


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
