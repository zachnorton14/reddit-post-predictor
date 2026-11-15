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
    run = {"commit": "abc1234def", "hf_path": "models/deberta-184M"}
    assert hf_models.champion_record(run, {"name": "deberta-184M"}, "model-3") == {
        "version": "model-3", "name": "deberta-184M", "path": "models/deberta-184M", "commit": "abc1234def"}


def test_path_command(tmp_path, capsys):
    f = tmp_path / "model.json"
    f.write_text(json.dumps({"name": "deberta-184M", "type": "deberta"}))
    assert hf_models.main(["path", str(f)]) == 0
    assert capsys.readouterr().out.strip() == "models/deberta-184M"
    f.write_text(json.dumps({"type": "deberta"}))
    assert hf_models.main(["path", str(f)]) == 1
