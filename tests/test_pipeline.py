"""pipeline/train.py and pipeline/evaluate.py without torch, downloads or a GPU (synthetic data, fake loaders)."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from pipeline import evaluate, train
from scripts.hf_models import NAME


def fake_test_split(n=200):
    rng = np.random.default_rng(0)
    metric = rng.normal(size=n)
    return pd.DataFrame({"metric_pct": pd.Series(metric).rank(pct=True), "label": (metric > np.quantile(metric, 0.9))
                         .astype(int), "title": "t", "body": "b"})


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
        with mock.patch.object(evaluate, "load_pinned", return_value=self.data) as load, \
                mock.patch.dict(evaluate.LOADERS, loaders, clear=True):
            evaluate.main(["--model", str(self.model), "--dataset", "o/d", "--revision", "abc", "--out", str(self.out)])
        load.assert_called_once_with("o/d", "abc", body_chars=1000)

    def test_dispatches_on_type_and_writes_gated_metrics(self):
        fake = mock.Mock(side_effect=lambda d, m, data: data.test.metric_pct.to_numpy())
        self.run_evaluate({"type": "fake"}, {"fake": fake})
        self.assertEqual(fake.call_args.args[0], self.model)
        result = json.loads(self.out.read_text())
        self.assertEqual(set(result), {"pr_auc", "top10_hit_rate", "spearman", "n_test"})
        self.assertEqual(result["top10_hit_rate"], 1.0)  # a perfect ranking hits every top-10% post
        self.assertEqual(result["n_test"], 200)
        self.assertFalse(any(f.name != "model.json" for f in self.model.iterdir()))

    def test_val_split_scores_validation(self):
        self.data.val = fake_test_split(120)
        fake = mock.Mock(side_effect=lambda d, m, data: data.test.metric_pct.to_numpy())
        (self.model / "model.json").write_text(json.dumps({"type": "fake"}))
        with mock.patch.object(evaluate, "load_pinned", return_value=self.data), \
                mock.patch.dict(evaluate.LOADERS, {"fake": fake}, clear=True):
            evaluate.main(["--model", str(self.model), "--dataset", "o/d", "--revision", "abc", "--out", str(self.out),
                           "--split", "val"])
        result = json.loads(self.out.read_text())
        self.assertEqual(result["n_val"], 120)
        self.assertNotIn("n_test", result)
        self.assertEqual(len(self.data.test), 200)  # the caller's data is not changed

    def test_prototype_runs_read_full_bodies_and_rows_are_hashed(self):
        self.assertIn("prototype", evaluate.LOADERS)
        self.data.test["id"] = [f"p{i}" for i in range(200)]
        fake = mock.Mock(side_effect=lambda d, m, data: data.test.metric_pct.to_numpy())
        (self.model / "model.json").write_text(json.dumps({"type": "prototype"}))
        with mock.patch.object(evaluate, "load_pinned", return_value=self.data) as load, \
                mock.patch.dict(evaluate.LOADERS, {"prototype": fake}, clear=True):
            evaluate.main(["--model", str(self.model), "--dataset", "o/d", "--revision", "abc", "--out", str(self.out)])
        load.assert_called_once_with("o/d", "abc", body_chars=None)
        rows = json.loads(self.out.read_text())["rows"]
        self.assertEqual(rows, evaluate.rows_hash(self.data.test.iloc[::-1]))  # order doesn't matter
        self.assertNotEqual(rows, evaluate.rows_hash(self.data.test.iloc[1:]))

    def test_unknown_type_fails_clearly(self):
        with self.assertRaises(SystemExit) as e:
            self.run_evaluate({"type": "nope"}, {"fake": mock.Mock()})
        self.assertIn("unknown model type 'nope'", str(e.exception))


if __name__ == "__main__":
    unittest.main()
