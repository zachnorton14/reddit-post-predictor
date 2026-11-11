"""Forward Hugging Face dataset webhooks to GitHub as a repository_dispatch event.

Hugging Face webhooks POST a JSON payload with an X-Webhook-Secret header. GitHub's dispatch API needs a bearer
token and its own body, so this relay sits in between (deployed as a Docker Space). Environment:

    WEBHOOK_SECRET   the secret set on the Hugging Face webhook
    GH_TOKEN         GitHub token that may create dispatch events on GH_REPO
    GH_REPO          e.g. zachnorton14/reddit-post-predictor
"""

import hmac
import os

import httpx
from fastapi import FastAPI, Header, HTTPException, Request

EVENT_TYPE = "hf-dataset-updated"
app = FastAPI()


def dispatch_body(payload):
    """The repository_dispatch body for a dataset content change, or None for any other event."""
    repo, event = payload.get("repo", {}), payload.get("event", {})
    if repo.get("type") != "dataset" or event.get("scope") != "repo.content":
        return None
    return {"event_type": EVENT_TYPE, "client_payload": {"repo": repo["name"], "revision": repo.get("headSha")}}


@app.get("/")
def health():
    return {"ok": True}


@app.post("/webhook")
async def webhook(request: Request, x_webhook_secret: str = Header(default="")):
    if not hmac.compare_digest(x_webhook_secret, os.environ["WEBHOOK_SECRET"]):
        raise HTTPException(401, "bad webhook secret")
    body = dispatch_body(await request.json())
    if body is None:
        return {"forwarded": False}
    r = httpx.post(f"https://api.github.com/repos/{os.environ['GH_REPO']}/dispatches", json=body, timeout=10,
                   headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                            "Accept": "application/vnd.github+json"})
    if r.status_code != 204:
        raise HTTPException(502, f"GitHub dispatch failed: {r.status_code} {r.text[:200]}")
    return {"forwarded": True, **body["client_payload"]}
