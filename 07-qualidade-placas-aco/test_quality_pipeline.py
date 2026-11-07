import unittest

import numpy as np
import pandas as pd

from quality_pipeline import evaluate


class QualityMetricTests(unittest.TestCase):
    def test_top2_and_per_class_support_follow_label_order(self):
        classes = ["Pastry", "Z_Scratch", "K_Scratch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
        y = pd.Series(classes)
        prediction = np.array(classes, dtype=object)
        probabilities = np.eye(len(classes), dtype=float) * 0.8 + 0.2 / len(classes)
        report = evaluate(y, prediction, probabilities, classes)
        self.assertEqual(report["rows"], len(classes))
        self.assertAlmostEqual(report["top_2_accuracy"], 1.0)
        self.assertAlmostEqual(report["macro_f1"], 1.0)
        self.assertEqual(report["per_class"]["Dirtiness"]["support"], 1)
        self.assertEqual(report["confusion_matrix"][5][5], 1)


if __name__ == "__main__":
    unittest.main()
