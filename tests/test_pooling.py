# ruff: noqa: E402  (torch imports are skipped first when torch is missing)
"""run_pooling.py: the ladder definitions and the command-line contract the experiment workflow calls."""
from dataclasses import replace

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import run_pooling as rp
from pipeline import experiment


def test_ladder_names_match_specs_and_pretrained_sources():
    for name, spec in rp.LADDER.items():
        assert spec.name == name
        assert spec.source.split("/")[-1] in rp.PRETRAINED


def test_ladder_rungs_add_one_change_each():
    base, long, time, attn = (rp.LADDER[k] for k in ("minilm_base", "minilm_long", "minilm_time", "minilm_attn4"))
    assert (base.max_length, long.max_length, long.body_chars) == (96, 256, None)
    assert not long.time_embed and time.time_embed
    assert time.pooling == "mean" and (attn.pooling, attn.queries) == ("attention", 4)


def test_overnight_interactions_match_their_joint_controls():
    cfg = experiment.load_config(rp.ROOT / "experiments/deberta_setup_punchline.yaml")
    plan = experiment.plan(cfg)
    assert cfg["runner"] == "run_pooling.py" and cfg["epochs"] == 3 and not cfg["allow_fallback"]
    assert len(plan) == 4
    for (control, seed), (candidate, candidate_seed) in (plan[:2], plan[2:]):
        assert seed == candidate_seed == 45
        assert rp.LADDER[candidate] == replace(rp.LADDER[control], name=candidate, architecture="interaction")


@pytest.mark.parametrize("args", [["--dataset", "repo"], ["--revision", "abc"]])
def test_dataset_needs_revision(args, monkeypatch):
    monkeypatch.setattr(rp, "data_splits", lambda *a: pytest.fail("must not load data"))
    with pytest.raises(SystemExit):
        rp.main(["--models", "minilm_base", *args])


def test_pinned_dataset_is_passed_through(monkeypatch):
    seen = []

    def stop(*a):
        seen.append(a)
        raise RuntimeError("stop")

    monkeypatch.setattr(rp, "data_splits", stop)
    with pytest.raises(RuntimeError):
        rp.main(["--models", "minilm_base", "--dataset", "repo", "--revision", "abc"])
    assert seen == [("repo", "abc")]
