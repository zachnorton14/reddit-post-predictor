"""d1_models.py: the state text, questions, targets and ranking scores without torch; the trained readout with torch."""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import d1_models as d1
from pipeline.evaluate import BODY_CHARS, LOADERS


def post(**kw):
    base = dict(subreddit="dadjokes", local_dow=6, local_hour=9, local_month=7, log_posts_prev_hour=math.log1p(3),
                log_posts_prev_day=math.log1p(40), log_author_prev_posts=math.log1p(2), has_author_flair=1,
                is_repost=0, title="Why did the chicken cross the road?", body="To get to the other side.")
    return SimpleNamespace(**(base | kw))


def test_state_text_carries_the_posting_context():
    text = d1.state_text(post())
    assert "r/dadjokes" in text and "Saturday 09:00" in text and "July" in text
    assert "3 posts in the previous hour, 40 in the previous day" in text
    assert "2 earlier posts in this subreddit, has a user flair" in text
    assert text.endswith("Title: Why did the chicken cross the road?\nBody: To get to the other side.")
    bare = d1.state_text(post(body="", is_repost=1, has_author_flair=0))
    assert "Body:" not in bare and "posted here before" in bare and "flair" not in bare


def test_questions_name_the_subreddit_and_fit_d1_schema():
    score = d1.question("sysadmin", "score10")
    assert score["type"] == "score" and len(score["criteria"]) == d1.LEVELS and "r/sysadmin" in score["instructions"]
    assert d1.question("sysadmin", "noul")["type"] == "noul"
    assert d1.question("sysadmin", "reg")["type"] == "noul"


def test_targets():
    frame = pd.DataFrame({"label": [0, 1, 0], "metric_pct": [0.0, 0.95, 1.0]})
    assert d1.targets(frame, "noul").tolist() == [0, 1, 0]
    assert d1.targets(frame, "score10").tolist() == [0, 9, 9]  # percentile 1.0 is the top level, not 10
    assert d1.targets(frame, "reg").dtype == np.float32


def test_rank_scores_order_posts_by_engagement():
    noul = d1.rank_scores("noul", [[2.0, -1.0], [-1.0, 2.0]])["yes_margin"]
    assert noul[1] > noul[0]
    low, high = np.full(d1.LEVELS, -5.0), np.full(d1.LEVELS, -5.0)
    low[0], high[-1] = 5.0, 5.0
    s = d1.rank_scores("score10", [low, high])
    assert set(s) == set(d1.RANKERS["score10"])
    assert s["expected_level"][1] > s["expected_level"][0] and s["top_level"][1] > s["top_level"][0]
    assert set(d1.MODELS) == {"d1_noul", "d1_score10", "d1_reg"}
    assert all(set(d1.rank_scores(spec.readout, np.zeros((1, 10)))) == set(d1.RANKERS[spec.readout])
               for spec in d1.MODELS.values())


def test_loader_registered():
    assert "d1" in LOADERS and BODY_CHARS["d1"] is None


def test_batches_cover_every_post_once():
    lengths = [5, 1, 9, 3, 3, 7, 2]
    for rng in (None, np.random.default_rng(0)):
        got = np.concatenate(d1.batches(lengths, 3, rng))
        assert sorted(got.tolist()) == list(range(len(lengths)))


def test_readout_trains_and_round_trips():
    """A two-layer stand-in for d1 (same trunk/head interface) learns, and its saved weights reload."""
    torch = pytest.importorskip("torch")
    from torch import nn

    class Trunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(20, 8)
            self.lin = nn.Linear(8, 8)

        def forward(self, h, pad, prefix):
            return self.lin(h)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.score = nn.Linear(8, 1)

        def forward(self, h, pad, pos, mask, qtype):
            return self.score(torch.gather(h, 1, pos[..., None].expand(-1, -1, 8))).squeeze(-1)

    model = SimpleNamespace(encoder=Trunk(), head=Head(), device="cpu", config=SimpleNamespace(text_config={"hidden_size": 8}),
                            train=lambda mode=True: None)
    for readout in ("noul", "score10", "reg"):
        net = d1.Readout(model, readout)
        k = 10 if readout == "score10" else 2
        ids, markers = [[1, 2, 3, *range(4, 4 + k)], [1, 5, *range(4, 4 + k)]], [list(range(3, 3 + k)), list(range(2, 2 + k))]
        out = net(ids, markers)
        assert out.shape == ((2, 1) if readout == "reg" else (2, k))
        clone = d1.Readout(SimpleNamespace(**vars(model), ), readout)
        clone.load_state_dict(net.state_dict())
        assert torch.allclose(clone(ids, markers), out)
