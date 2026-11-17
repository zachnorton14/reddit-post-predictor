"""Offline regressions for combining checkpoints from separate workflow runs."""
import numpy as np
import pandas as pd
import pytest

from scripts import deberta_ensemble as ensemble


def test_reorders_equal_timestamp_posts_before_blending():
    rows = pd.DataFrame({"id": ["a", "b"], "created_at": pd.to_datetime(["2024-01-01"] * 2)})
    first = ensemble.aligned_predictions(rows, rows.iloc[::-1], np.array([1., 5.]))
    second = ensemble.aligned_predictions(rows, rows, np.array([3., 5.]))
    np.testing.assert_array_equal(ensemble.mean_scores([first, second]), [4., 3.])


@pytest.mark.parametrize("ids,times", [
    (["a", "a"], ["2024-01-01"] * 2),
    (["a", "c"], ["2024-01-01"] * 2),
    (["a", "b"], ["2024-01-01", "2024-02-01"]),
])
def test_rejects_different_validation_rows(ids, times):
    frame = pd.DataFrame({"id": ["a", "b"], "created_at": pd.to_datetime(["2024-01-01"] * 2)})
    rows = pd.DataFrame({"id": ids, "created_at": pd.to_datetime(times)})
    with pytest.raises(ValueError, match="Validation"):
        ensemble.aligned_predictions(frame, rows, np.array([1., 2.]))
