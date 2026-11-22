"""HTTP inference for a saved model: send a post's title and body, get its predicted engagement.

    MODEL_DIR=<dir with model.json> uvicorn pipeline.serve:app --host 0.0.0.0 --port 8000
    python -m pipeline.serve --check <dir>     load the model and score one post (the serve image runs this at build)

docker/serve.Dockerfile bakes the deployed model into the image and runs this. Endpoints:

    GET  /health    {"ok": true, "model": <name>, "type": <type>, "version": <MODEL_VERSION>}
    POST /predict   {"posts": [{"title": "...", "body": "..."}]} -> {"model": ..., "predictions": [{"score": ...}]}

`score` is the predicted engagement, log1p(score) + log1p(comments), on the training target's scale. Higher means
more engagement. Rank posts by it; it is not a probability.

The text is built the way training built it: the body is cut at the first EDIT marker (textonly_data.EDIT_RE) and
then to BODY_CHARS characters, and the model reads "<title> \\n <body>". Only types whose inputs are just the title
and body can be served (PREDICTORS). Types that also read posting-time metadata (`prototype`) can't, because a
request doesn't carry the poster's history.
"""

import argparse
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BODY_CHARS = 1000  # what the minilm type trained on (pipeline.data.load_pinned's default)
MAX_POSTS = 256
_EDIT = None


def clean_body(body: str, chars: int | None = BODY_CHARS) -> str:
    """The body as training saw it: cut at the first EDIT marker, stripped, then cut to `chars` characters."""
    global _EDIT
    if _EDIT is None:
        from textonly_data import EDIT_RE

        _EDIT = re.compile(f"(?s){EDIT_RE}.*$")
    body = _EDIT.sub("", body or "").strip()
    return body if chars is None else body[:chars]


def model_text(title: str, body: str) -> str:
    """dadjokes_models.text_of for one post."""
    return f"{title} \n {clean_body(body)}"


def load_minilm(model_dir: Path, meta: dict):
    import dadjokes_models as dm
    from pipeline.evaluate import load_minilm as load

    net, tok = load(model_dir, meta)
    return lambda texts: dm.predict_net(net, tok, texts, meta["max_len"], meta["mu"], meta["sd"])


# model.json type -> loader(model_dir, meta) returning predict(list of texts) -> array of scores. Load once, at start.
PREDICTORS = {"minilm": load_minilm}


class Model:
    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        self.meta = json.loads((self.dir / "model.json").read_text())
        kind = self.meta.get("type")
        if kind not in PREDICTORS:
            raise ValueError(f"{self.dir}: type {kind!r} can't be served; servable types: {', '.join(PREDICTORS)}")
        self.predict_texts = PREDICTORS[kind](self.dir, self.meta)

    def predict(self, posts) -> list[float]:
        return [float(s) for s in self.predict_texts([model_text(p["title"], p.get("body", "")) for p in posts])]


class Post(BaseModel):
    title: str = Field(min_length=1)
    body: str = ""


class PredictRequest(BaseModel):
    posts: list[Post] = Field(min_length=1, max_length=MAX_POSTS)


state: dict = {}


@asynccontextmanager
async def lifespan(_app):
    if "model" not in state:
        state["model"] = Model(os.environ.get("MODEL_DIR", "/model"))
    yield


app = FastAPI(title="reddit-post-predictor", lifespan=lifespan)


def _model() -> Model:
    if "model" not in state:
        raise HTTPException(503, "model not loaded")
    return state["model"]


@app.get("/health")
def health():
    m = _model()
    return {"ok": True, "model": m.meta.get("name"), "type": m.meta.get("type"),
            "version": os.environ.get("MODEL_VERSION")}


@app.post("/predict")
def predict(req: PredictRequest):
    m = _model()
    start = time.perf_counter()
    scores = m.predict([p.model_dump() for p in req.posts])
    return {"model": m.meta.get("name"), "version": os.environ.get("MODEL_VERSION"),
            "predictions": [{"score": s} for s in scores], "ms": round(1000 * (time.perf_counter() - start), 1)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", required=True, type=Path, help="model directory: load it and score one post")
    args = p.parse_args(argv)
    m = Model(args.check)
    print(json.dumps({"model": m.meta.get("name"), "score": m.predict([{"title": "Why did the chicken cross the road?",
                                                                        "body": "To get to the other side."}])[0]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
