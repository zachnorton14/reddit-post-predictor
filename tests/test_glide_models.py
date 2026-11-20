"""glide_models.py: option order, targets, micro-batching and the LoRA layer (torch); the 9B model itself is not loaded."""

import numpy as np
import pandas as pd
import pytest

import glide_models as gm
from pipeline.evaluate import BODY_CHARS, LOADERS


def test_noul_option_order_is_flipped_from_d1():
    """GLiDE lists [true, false]; a high "true" logit must rank a post higher, and the label maps to option 0."""
    s = gm.rank_scores("noul", [[2.0, -1.0], [-1.0, 2.0]])["yes_margin"]
    assert s[0] > s[1]
    frame = pd.DataFrame({"label": [0, 1], "metric_pct": [0.1, 0.95]})
    assert gm.targets(frame, "noul").tolist() == [1, 0]
    assert gm.targets(frame, "score10").tolist() == [1, 9]
    assert all(set(gm.rank_scores(spec.readout, np.zeros((1, 10 if spec.readout == "score10" else 2))))
               == set(gm.RANKERS[spec.readout]) for spec in gm.MODELS.values())


def test_micro_batches_respect_the_token_budget():
    lengths = [100, 400, 50, 300, 200, 250]
    mbs = gm.micro_batches(range(len(lengths)), lengths, 600)
    assert sorted(i for mb in mbs for i in mb) == list(range(len(lengths)))
    assert all(len(mb) * max(lengths[i] for i in mb) <= 600 or len(mb) == 1 for mb in mbs)


def test_loader_registered():
    assert "glide" in LOADERS and BODY_CHARS["glide"] is None


def test_lora_starts_as_identity_and_trains():
    torch = pytest.importorskip("torch")
    net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    net.requires_grad_(False)
    x = torch.randn(3, 4)
    before = net(x)
    for name in ("0", "2"):
        gm.LoRALinear(net.get_submodule(name), rank=2, alpha=4)
    assert torch.allclose(net(x), before)  # B starts at zero
    params = [p for n, p in net.named_parameters() if n.endswith(("lora_A", "lora_B"))]
    assert len(params) == 4 and all(p.requires_grad for p in params)
    net(x).sum().backward()
    assert params[1].grad is not None and params[1].grad.abs().sum() > 0


def test_long_prompts_are_shortened_to_fit():
    """A fake tokenizer (one token per character) and prompter: only the over-long post is cut, and posts keep their order."""
    from types import SimpleNamespace

    frame = pd.DataFrame({"subreddit": ["a", "b", "c"], "title": ["x" * 50, "y" * 900, "z" * 10], "body": ["", "", ""],
                          "local_dow": 0, "local_hour": 0, "local_month": 1, "log_posts_prev_hour": 0.0,
                          "log_posts_prev_day": 0.0, "log_author_prev_posts": 0.0, "has_author_flair": 0, "is_repost": 0})
    glide = SimpleNamespace(prompter=SimpleNamespace(text=lambda state, q, opts: state),
                            tokenizer=lambda texts, add_special_tokens: {"input_ids": [list(t) for t in texts]})
    spec = gm.Spec("noul", max_len=400, state_chars=2000)
    ids = gm.encode_frame(frame, glide, lambda q: [], spec)
    assert all(len(x) <= 400 for x in ids)
    assert "x" * 50 in "".join(ids[0]) and "z" * 10 in "".join(ids[2]) and "y" in "".join(ids[1])


def test_clean_train_drops_only_score0_commented_posts():
    frame = pd.DataFrame({"score": [0, 0, 5, 1], "num_comments": [3, 0, 0, 2]})
    assert gm.silent_removals_dropped(frame).index.tolist() == [1, 2, 3]


def test_prompts_fit_when_the_question_fills_most_tokens():
    """A fixed 380-token question leaves 20 tokens for the state: shrinking still converges."""
    from types import SimpleNamespace

    frame = pd.DataFrame({"subreddit": ["a"], "title": ["y" * 3000], "body": [""], "local_dow": 0, "local_hour": 0,
                          "local_month": 1, "log_posts_prev_hour": 0.0, "log_posts_prev_day": 0.0,
                          "log_author_prev_posts": 0.0, "has_author_flair": 0, "is_repost": 0})
    glide = SimpleNamespace(prompter=SimpleNamespace(text=lambda state, q, opts: "q" * 380 + state),
                            tokenizer=lambda texts, add_special_tokens: {"input_ids": [list(t) for t in texts]})
    ids = gm.encode_frame(frame, glide, lambda q: [], gm.Spec("noul", max_len=400, state_chars=2000))
    assert len(ids[0]) <= 400


def test_checkpoint_path_is_unique_per_run_and_saves_without_a_token(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    a = gm.checkpoint_remote(gm.MODELS["glide_score10"], 1000)
    assert a != gm.checkpoint_remote(gm.MODELS["glide_score10_clean8k"], 1000)
    assert a != gm.checkpoint_remote(gm.MODELS["glide_score10"], 999)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    local = tmp_path / "resume.pt"
    gm.fetch_checkpoint(local, a)  # no token: starts fresh
    assert not local.exists()
    gm.save_checkpoint(local, a, {"epoch": 1, "seen": 64, "net": {"w": torch.ones(2)}})
    assert torch.load(local, weights_only=False)["seen"] == 64 and not local.with_suffix(".tmp").exists()
