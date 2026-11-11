"""validate_dataset.py on a small synthetic dataset: a clean one passes, and each kind of defect is caught."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import duckdb

import textonly_baselines as tb
import validate_dataset as vd

# Two text posts a day in 2021-01, with the dataset's exact column types
ROWS_SQL = """
SELECT 'p' || i::VARCHAR AS id, 'author_' || (i % 5)::VARCHAR AS author, 'dadjokes' AS subreddit,
       'A joke number ' || i::VARCHAR AS title, 'Body text.' AS selftext, (i % 10)::BIGINT AS score,
       (1609459200 + i * 43200)::BIGINT AS created_utc, make_timestamp((1609459200 + i * 43200) * 1000000) AS created_at,
       14::BIGINT AS title_length, (i % 4)::BIGINT AS num_comments,
       'https://www.reddit.com/r/dadjokes/comments/p' || i::VARCHAR || '/x/' AS url, false AS over_18,
       NULL::VARCHAR AS link_flair_text, NULL::VARCHAR AS author_flair_text
FROM range(60) t(i)
"""


class ValidateDatasetTest(unittest.TestCase):
    def write(self, select=ROWS_SQL, path="data/dadjokes/2021/01.parquet"):
        root = Path(tempfile.mkdtemp(dir=self.tmp.name))
        (root / path).parent.mkdir(parents=True)
        duckdb.execute(f"COPY ({select}) TO '{root / path}' (FORMAT parquet)")
        return root

    def failures(self, root, cfg=None):
        errors, _ = vd.validate(root, cfg)
        return {check for check, errs in errors.items() if errs}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_dataset_passes(self):
        self.assertEqual(self.failures(self.write()), set())

    def test_wrong_column_type(self):
        root = self.write(f"SELECT * REPLACE (score::VARCHAR AS score) FROM ({ROWS_SQL})")
        self.assertEqual(self.failures(root), {"columns"})

    def test_missing_values_and_duplicates(self):
        root = self.write(f"SELECT * REPLACE (CASE WHEN score = 0 THEN NULL ELSE title END AS title) FROM ({ROWS_SQL})")
        self.assertEqual(self.failures(root), {"missing values"})
        root = self.write(f"SELECT * REPLACE ('same' AS id) FROM ({ROWS_SQL})")
        self.assertEqual(self.failures(root), {"missing values"})

    def test_rows_in_wrong_file(self):
        self.assertEqual(self.failures(self.write(path="data/dadjokes/2021/02.parquet")), {"layout"})
        self.assertEqual(self.failures(self.write(path="data/sysadmin/2021/01.parquet")), {"layout"})

    def test_filtered_rows_present(self):
        root = self.write(f"SELECT * REPLACE (score = 3 AS over_18) FROM ({ROWS_SQL})")
        self.assertEqual(self.failures(root), {"layout"})

    def test_split_dates_must_start_a_period(self):
        # A quarter that straddles train_end mixes train and validation outcomes in one label threshold
        self.assertEqual(self.failures(self.write(), replace(tb.Config(), train_end="2024-02-01")),
                         {"chronological splits"})
        self.assertEqual(self.failures(self.write(), replace(tb.Config(), val_end="2023-01-01")),
                         {"chronological splits"})

    def test_outcome_as_feature(self):
        groups = tb.GROUPS
        tb.GROUPS = {**groups, "leaky": {"numeric": ["num_comments"]}}
        try:
            self.assertEqual(self.failures(self.write()), {"label leakage"})
        finally:
            tb.GROUPS = groups


if __name__ == "__main__":
    unittest.main()
