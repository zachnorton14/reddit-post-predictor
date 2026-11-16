"""The release gate passes improvements and small drops, and fails real regressions."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_metrics import compare  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
