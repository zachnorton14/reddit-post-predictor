"""Integration checks using synthetic Parquet posts; no downloads or full extract needed."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import textonly_baselines as tb
import textonly_data as td


class TreeModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "posts.parquet"
        rows = []
        for sub in td.SUBREDDITS:
            for year in (2021, 2024, 2025):
                for i in range(60):
                    created = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(hours=i * 9)
                    post_id = f"{sub}_{year}_{i}"
                    title = f"Why did the chicken cross {sub} road number {i % 11}?"
                    rows.append({
                        "id": post_id, "author": f"author_{i % 13}", "subreddit": sub,
                        "title": title, "selftext": "A short story. " * (i % 5),
                        "score": (i % 10) ** 2, "num_comments": i % 7,
                        "created_utc": created.value // 10**9, "created_at": created,
                        "title_length": len(title), "over_18": False,
                        "url": f"https://www.reddit.com/r/{sub}/comments/{post_id}/title/",
                        "link_flair_text": None, "author_flair_text": "regular" if i % 3 else None,
                    })
        pd.DataFrame(rows).to_parquet(path, index=False)
        cls.cfg = tb.Config(source=str(path), min_period_posts=20,
                            max_train_per_sub=60, max_eval_per_sub=60)
        cls.rf_params = dict(n_estimators=12, max_depth=4, min_samples_leaf=2, n_jobs=1)
        cls.gb_params = dict(max_iter=12, max_leaf_nodes=7, min_samples_leaf=3)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.data = tb.load_data(self.cfg)
        self.addCleanup(self.data.con.close)
        self.feats = tb.build_features(self.data, text=False)

    def fit(self, data, feats, results):
        return tb.run_tree_models(data, feats, results,
                                  rf_params=self.rf_params, gb_params=self.gb_params)

    def test_both_models_report_all_subreddits_without_text_or_outcome_features(self):
        self.assertNotIn("text", self.feats.matrices)
        names = self.feats.feature_names(tb.META)
        self.assertFalse(set(names) & {"label", "score", "num_comments", "metric", "metric_pct"})
        results = tb.Results(self.data)
        results.evaluate("reference", np.full(len(self.data.val), 0.1),
                         np.full(len(self.data.test), 0.1))
        models = self.fit(self.data, self.feats, results)

        self.assertEqual(set(models), {"rf_meta", "gb_meta"})
        self.assertIn("reference", results.rows)
        self.assertEqual(len(results.rows), 3)
        for model in models.values():
            self.assertEqual(model.n_features_in_, len(names))
            np.testing.assert_array_equal(model.classes_, [0, 1])
        self.assertFalse(models["gb_meta"].do_early_stopping_)
        self.assertEqual(models["gb_meta"].n_iter_, self.gb_params["max_iter"])
        for split in ("val", "test"):
            per_sub = results.per_subreddit(split)
            self.assertEqual(set(per_sub.index), set(td.SUBREDDITS))
            self.assertTrue(np.isfinite(per_sub.to_numpy()).all())
            for predictions in results.preds.values():
                p = predictions[split]
                self.assertEqual(len(p), len(getattr(self.data, split)))
                self.assertTrue(((p > 0) & (p < 1)).all())
        self.assertTrue(np.isfinite(results.table().filter(regex="_auc|_log_loss").to_numpy()).all())

    def test_future_outcomes_do_not_change_features_or_fitted_predictions(self):
        original_results = tb.Results(self.data)
        self.fit(self.data, self.feats, original_results)
        changed = replace(self.data, val=self.data.val.copy(), test=self.data.test.copy())
        for split in ("val", "test"):
            frame = getattr(changed, split)
            frame["label"] = 1 - frame.label
            frame[["metric", "metric_pct", "score", "num_comments"]] = 999
        changed_feats = tb.build_features(changed, text=False)
        for group in tb.META:
            for split in tb.SPLITS:
                np.testing.assert_array_equal(self.feats[group][split].toarray(),
                                              changed_feats[group][split].toarray())
        changed_results = tb.Results(changed)
        self.fit(changed, changed_feats, changed_results)
        for name in original_results.rows:
            for split in ("val", "test"):
                np.testing.assert_allclose(original_results.preds[name][split],
                                           changed_results.preds[name][split], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
