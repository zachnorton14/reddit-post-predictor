"""The inference server builds the model's text as training did and answers /health and /predict."""

import json

import pytest
from fastapi.testclient import TestClient

from pipeline import serve


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "model.json").write_text(json.dumps({"type": "fake", "name": "fake-1"}))
    seen = []

    def load_fake(model_dir, meta):
        def predict(texts):
            seen.extend(texts)
            return [float(len(t)) for t in texts]
        return predict

    monkeypatch.setitem(serve.PREDICTORS, "fake", load_fake)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_VERSION", "model-7")
    serve.state.clear()
    with TestClient(serve.app) as c:
        c.seen = seen
        yield c
    serve.state.clear()


def test_health(client):
    assert client.get("/health").json() == {"ok": True, "model": "fake-1", "type": "fake", "version": "model-7"}


def test_predict_scores_each_post_with_training_text(client):
    r = client.post("/predict", json={"posts": [{"title": "Hi", "body": "there\nEDIT: thanks for the gold!"},
                                                 {"title": "No body"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fake-1" and body["version"] == "model-7"
    assert client.seen == ["Hi \n there", "No body \n "]
    assert [p["score"] for p in body["predictions"]] == [len("Hi \n there"), len("No body \n ")]


@pytest.mark.parametrize("payload", [{"posts": []}, {"posts": [{"title": ""}]}, {"posts": [{"body": "x"}]},
                                     {"posts": [{"title": "t"}] * (serve.MAX_POSTS + 1)}])
def test_bad_requests_are_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_clean_body_cuts_edits_and_length():
    assert serve.clean_body("  joke\n\n**Edit**: wow, front page  ") == "joke"
    assert serve.clean_body("a" * 1500) == "a" * serve.BODY_CHARS
    assert serve.clean_body(None) == ""
    assert serve.model_text("T", "b") == "T \n b"


def test_unservable_type_fails_at_start(tmp_path):
    (tmp_path / "model.json").write_text(json.dumps({"type": "prototype", "name": "p"}))
    with pytest.raises(ValueError, match="can't be served"):
        serve.Model(tmp_path)
