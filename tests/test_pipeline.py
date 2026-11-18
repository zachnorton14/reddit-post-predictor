"""pipeline/train.py and pipeline/evaluate.py without torch, downloads or a GPU (synthetic data, fake loaders)."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from pipeline import data as pdata
from pipeline import evaluate, train
from scripts.hf_models import NAME


def fake_test_split(n=2400, subs=("a", "b")):
    """n posts split evenly over `subs`, each labeled top 10% within its subreddit."""
    rng = np.random.default_rng(0)
    parts = []
    for sub in subs:
        metric = pd.Series(rng.normal(size=n // len(subs)))
        parts.append(pd.DataFrame({"subreddit": sub, "metric_pct": metric.rank(pct=True),
                                   "label": (metric > metric.quantile(0.9)).astype(int), "title": "t", "body": "b"}))
    return pd.concat(parts, ignore_index=True)


class TrainTest(unittest.TestCase):
    def test_args(self):
        a = train.parse_args(["--dataset", "o/d", "--revision", "abc", "--out", "/tmp/m"])
        self.assertEqual((a.dataset, a.revision, a.out, a.max_train, a.epochs), ("o/d", "abc", Path("/tmp/m"), None, 3))
        with self.assertRaises(SystemExit):
            train.parse_args(["--dataset", "o/d", "--out", "/tmp/m"])

    def test_model_json(self):
        meta = train.model_json(22_713_601, 1.5, 0.9, 96)
        self.assertRegex(meta["name"], NAME)
        self.assertIn(meta["type"], evaluate.LOADERS)
        self.assertEqual({k: meta[k] for k in ("params", "max_len", "mu", "sd")},
                         {"params": 22_713_601, "max_len": 96, "mu": 1.5, "sd": 0.9})
        json.dumps(meta)


class EvaluateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name, "model")
        self.model.mkdir()
        self.out = Path(self.tmp.name, "metrics.json")
        self.data = SimpleNamespace(test=fake_test_split())

    def tearDown(self):
        self.tmp.cleanup()

    def run_evaluate(self, meta, loaders):
        (self.model / "model.json").write_text(json.dumps(meta))
        with mock.patch.object(evaluate, "load_eval", return_value=self.data) as load, \
                mock.patch.dict(evaluate.LOADERS, loaders, clear=True):
            evaluate.main(["--model", str(self.model), "--dataset", "o/d", "--revision", "abc", "--out", str(self.out)])
        load.assert_called_once_with("o/d", "abc", body_chars=1000)

    def test_dispatches_on_type_and_writes_gated_metrics(self):
        fake = mock.Mock(side_effect=lambda d, m, data: data.test.metric_pct.to_numpy())
        self.run_evaluate({"type": "fake"}, {"fake": fake})
        self.assertEqual(fake.call_args.args[0], self.model)
        result = json.loads(self.out.read_text())
        self.assertEqual(set(result), {"pr_auc", "top10_hit_rate", "spearman", "n_test", "macro_subreddits",
                                       "by_subreddit"})
        self.assertEqual(result["top10_hit_rate"], 1.0)  # a perfect ranking hits every top-10% post
        self.assertEqual(result["n_test"], 2400)
        self.assertEqual(result["macro_subreddits"], ["a", "b"])
        self.assertFalse(any(f.name != "model.json" for f in self.model.iterdir()))

    def test_val_split_scores_validation(self):
        self.data.val = fake_test_split(2000)
        fake = mock.Mock(side_effect=lambda d, m, data: data.test.metric_pct.to_numpy())
        (self.model / "model.json").write_text(json.dumps({"type": "fake"}))
        with mock.patch.object(evaluate, "load_eval", return_value=self.data), \
                mock.patch.dict(evaluate.LOADERS, {"fake": fake}, clear=True):
            evaluate.main(["--model", str(self.model), "--dataset", "o/d", "--revision", "abc", "--out", str(self.out),
                           "--split", "val"])
        result = json.loads(self.out.read_text())
        self.assertEqual(result["n_val"], 2000)
        self.assertNotIn("n_test", result)
        self.assertEqual(len(self.data.test), 2400)  # the caller's data is not changed

    def test_prototype_runs_read_full_bodies_and_rows_are_hashed(self):
        self.assertIn("prototype", evaluate.LOADERS)
        self.data.test["id"] = [f"p{i}" for i in range(2400)]
        fake = mock.Mock(side_effect=lambda d, m, data: data.test.metric_pct.to_numpy())
        (self.model / "model.json").write_text(json.dumps({"type": "prototype"}))
        with mock.patch.object(evaluate, "load_eval", return_value=self.data) as load, \
                mock.patch.dict(evaluate.LOADERS, {"prototype": fake}, clear=True):
            evaluate.main(["--model", str(self.model), "--dataset", "o/d", "--revision", "abc", "--out", str(self.out)])
        load.assert_called_once_with("o/d", "abc", body_chars=None)
        rows = json.loads(self.out.read_text())["rows"]
        self.assertEqual(rows, evaluate.rows_hash(self.data.test.iloc[::-1]))  # order doesn't matter
        self.assertNotEqual(rows, evaluate.rows_hash(self.data.test.iloc[1:]))

    def test_macro_average_counts_each_big_subreddit_once(self):
        # "a" ranked perfectly, "b" ranked backwards, "tiny" (under MIN_POSTS) ranked perfectly but left out
        frame = pd.concat([fake_test_split(2400, ("a", "b")), fake_test_split(100, ("tiny",))], ignore_index=True)
        p = np.where(frame.subreddit == "b", -frame.metric_pct, frame.metric_pct)
        result = evaluate.metrics(frame, p)
        self.assertEqual(result["macro_subreddits"], ["a", "b"])
        self.assertEqual(result["by_subreddit"]["a"]["top10_hit_rate"], 1.0)
        self.assertEqual(result["by_subreddit"]["b"]["top10_hit_rate"], 0.0)
        self.assertEqual(result["by_subreddit"]["tiny"]["n"], 100)
        self.assertAlmostEqual(result["top10_hit_rate"], 0.5)
        self.assertAlmostEqual(result["spearman"], 0.0)

    def test_unknown_type_fails_clearly(self):
        with self.assertRaises(SystemExit) as e:
            self.run_evaluate({"type": "nope"}, {"fake": mock.Mock()})
        self.assertIn("unknown model type 'nope'", str(e.exception))


class LoadEvalTest(unittest.TestCase):
    def test_labels_once_per_revision_then_reads_the_cache(self):
        frames = {"train": fake_test_split(4), "val": fake_test_split(2000), "test": fake_test_split(2400)}
        fake = SimpleNamespace(con=mock.Mock(), **frames)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {"EVAL_CACHE": tmp}), \
                mock.patch.object(pdata, "snapshot_download", return_value=tmp) as download, \
                mock.patch.object(pdata.tb, "load_data", return_value=fake) as load:
            first = pdata.load_eval("o/d", "r1")
            again = pdata.load_eval("o/d", "r1")
            self.assertEqual(load.call_count, 1)
            self.assertEqual(download.call_count, 1)
            self.assertEqual(load.call_args.args[0].source, f"{tmp}/data/*/*/*.parquet")
            pdata.load_eval("o/d", "r2")  # another revision is labeled again
            self.assertEqual(load.call_count, 2)
        for d in (first, again):
            pd.testing.assert_frame_equal(d.test, frames["test"])
            pd.testing.assert_frame_equal(d.val, frames["val"])
        self.assertNotEqual(pdata.eval_cache_key("o/d", "r1", 1000, 10), pdata.eval_cache_key("o/d", "r1", None, 10))


if __name__ == "__main__":
    unittest.main()
