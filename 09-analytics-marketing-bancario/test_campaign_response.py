import unittest

import numpy as np
import pandas as pd

from campaign_response_pipeline import evaluate, prepare_features


class CampaignResponseTests(unittest.TestCase):
    def test_post_contact_fields_are_removed(self):
        X = pd.DataFrame({"age": [35, 52], "duration": [120, 90], "campaign": [1, 2], "previous": [0, 1]})
        result = prepare_features(X)
        self.assertEqual(list(result.columns), ["age", "previous"])

    def test_capacity_metrics_follow_score_order(self):
        y = pd.Series([1, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.1])
        report = evaluate(y, scores)
        top = report["capacity"][0]
        self.assertEqual(top["contacts"], 1)
        self.assertEqual(top["observed_subscriptions_in_top_k"], 1)
        self.assertEqual(top["response_rate_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
