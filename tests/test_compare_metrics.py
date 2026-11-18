"""The release gate passes improvements and small drops, and fails real regressions."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_metrics import compare, subreddit_drops  # noqa: E402

PROD = {"pr_auc": 0.274, "top10_hit_rate": 0.318}


class CompareMetricsTest(unittest.TestCase):
    def test_first_release_passes(self):
        self.assertEqual(compare(PROD, None), [])

    def test_small_drop_passes(self):
        self.assertEqual(compare({"pr_auc": 0.268, "top10_hit_rate": 0.320}, PROD), [])

    def test_regression_fails(self):
        self.assertEqual(len(compare({"pr_auc": 0.25, "top10_hit_rate": 0.318}, PROD)), 1)

    def test_missing_metric_fails(self):
        self.assertTrue(compare({"pr_auc": 0.3}, PROD))

    def test_different_posts_fail(self):
        self.assertTrue(compare({"pr_auc": 0.3, "top10_hit_rate": 0.4, "rows": "a"}, {**PROD, "rows": "b"}))
        self.assertEqual(compare({"pr_auc": 0.3, "top10_hit_rate": 0.4, "rows": "a"}, {**PROD, "rows": "a"}), [])

    def test_subreddit_drop_is_flagged(self):
        prod = {**PROD, "by_subreddit": {"a": {"pr_auc": 0.30}, "b": {"pr_auc": 0.30}}}
        cand = {**PROD, "by_subreddit": {"a": {"pr_auc": 0.25}, "b": {"pr_auc": 0.29}, "new": {"pr_auc": 0.1}}}
        self.assertEqual(len(subreddit_drops(cand, prod)), 1)
        self.assertIn("a pr_auc", subreddit_drops(cand, prod)[0])
        self.assertEqual(subreddit_drops(cand, None), [])
        self.assertEqual(compare(cand, prod), [])  # a warning only; the macro metrics decide


if __name__ == "__main__":
    unittest.main()
