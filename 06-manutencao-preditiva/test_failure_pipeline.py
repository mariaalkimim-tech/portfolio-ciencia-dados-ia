import unittest

import numpy as np
import pandas as pd

from failure_pipeline import capacity_metrics, chronological_indices


class ChronologicalSplitTests(unittest.TestCase):
    def test_splits_are_disjoint_ordered_and_cover_all_rows(self):
        train, validation, test = chronological_indices(10)
        self.assertEqual(train.tolist(), list(range(6)))
        self.assertEqual(validation.tolist(), list(range(6, 8)))
        self.assertEqual(test.tolist(), list(range(8, 10)))
        self.assertEqual(len(set(train) | set(validation) | set(test)), 10)

    def test_capacity_metrics_match_known_ranking(self):
        y = pd.Series([1, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.1])
        result = capacity_metrics(y, scores, fractions=(0.5,))[0]
        self.assertEqual(result["alerts"], 2)
        self.assertEqual(result["failures_captured"], 1)
        self.assertAlmostEqual(result["precision_at_capacity"], 0.5)
        self.assertAlmostEqual(result["recall_at_capacity"], 0.5)


if __name__ == "__main__":
    unittest.main()
