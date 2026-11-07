import unittest

import numpy as np
import pandas as pd

from default_risk_pipeline import group_audit, split_data


class CreditRiskTests(unittest.TestCase):
    def test_stratified_splits_are_disjoint_and_preserve_target_classes(self):
        n = 200
        X = pd.DataFrame({"X1": np.arange(n), "X3": np.tile([1, 2], n // 2)})
        y = pd.Series(np.tile([0, 1], n // 2))
        audit = pd.Series(np.tile([1, 2], n // 2))
        splits = split_data(X, y, audit)
        self.assertEqual({k: len(v[1]) for k, v in splits.items()}, {"train": 100, "validation": 50, "test": 50})
        self.assertTrue(all(v[1].nunique() == 2 for v in splits.values()))
        indices = [set(v[0].index) for v in splits.values()]
        self.assertEqual(len(indices[0] | indices[1] | indices[2]), n)
        self.assertFalse(indices[0] & indices[1] or indices[0] & indices[2] or indices[1] & indices[2])

    def test_group_audit_reports_descriptive_recall_and_false_positive_rate(self):
        y = pd.Series([1, 0, 1, 0])
        pred = np.array([1, 1, 0, 0])
        groups = pd.Series([1, 1, 2, 2])
        result = group_audit(y, pred, groups)
        self.assertEqual(result["1"]["rows"], 2)
        self.assertEqual(result["1"]["recall_at_threshold_0_5_descriptive"], 1.0)
        self.assertEqual(result["1"]["false_positive_rate_at_threshold_0_5_descriptive"], 1.0)
        self.assertEqual(result["2"]["recall_at_threshold_0_5_descriptive"], 0.0)


if __name__ == "__main__":
    unittest.main()
