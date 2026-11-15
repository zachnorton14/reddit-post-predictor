"""The webhook relay forwards only dataset content changes, and only with the right secret."""

import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from webhook_relay import app as relay

PUSH = {"event": {"action": "update", "scope": "repo.content"},
        "repo": {"type": "dataset", "name": "jbduran/textonly-subreddit-predictors", "headSha": "abc123"}}


@mock.patch.dict(os.environ, {"WEBHOOK_SECRET": "s3cret", "GH_TOKEN": "t", "GH_REPO": "o/r"})
class WebhookRelayTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(relay.app)

    def test_dataset_push_is_forwarded(self):
        with mock.patch.object(relay.httpx, "post", return_value=mock.Mock(status_code=204)) as post:
            r = self.client.post("/webhook", json=PUSH, headers={"X-Webhook-Secret": "s3cret"})
        self.assertEqual(r.json(), {"forwarded": True, "repo": PUSH["repo"]["name"], "revision": "abc123"})
        self.assertEqual(post.call_args.args[0], "https://api.github.com/repos/o/r/dispatches")
        self.assertEqual(post.call_args.kwargs["json"]["event_type"], "hf-dataset-updated")

    def test_other_events_are_ignored(self):
        for payload in ({**PUSH, "event": {"action": "create", "scope": "discussion"}},
                        {**PUSH, "repo": {**PUSH["repo"], "type": "model"}}):
            r = self.client.post("/webhook", json=payload, headers={"X-Webhook-Secret": "s3cret"})
            self.assertEqual(r.json(), {"forwarded": False})

    def test_wrong_secret_is_rejected(self):
        r = self.client.post("/webhook", json=PUSH, headers={"X-Webhook-Secret": "nope"})
        self.assertEqual(r.status_code, 401)


if __name__ == "__main__":
    unittest.main()
