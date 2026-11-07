import unittest

import pandas as pd

from itsm_pipeline import build_cases, case_survival_summary, temporal_distribution_diagnostic


class ITSMAnalyticsTests(unittest.TestCase):
    def test_case_kpis_measure_wait_proxy_and_group_handoff(self):
        events = pd.DataFrame([
            {"case_id": "case-a", "time:timestamp": pd.Timestamp("2020-01-01T00:00:00Z"), "concept:name": "Accepted", "lifecycle:transition": "Wait - User", "org:group": "A", "impact": "High", "product": "P1", "organization involved": "Org A", "org:resource": "person-1"},
            {"case_id": "case-a", "time:timestamp": pd.Timestamp("2020-01-01T02:00:00Z"), "concept:name": "Completed", "lifecycle:transition": "In Progress", "org:group": "B", "impact": "High", "product": "P1", "organization involved": "Org B", "org:resource": "person-2"},
            {"case_id": "case-a", "time:timestamp": pd.Timestamp("2020-01-01T03:00:00Z"), "concept:name": "Completed", "lifecycle:transition": "Resolved", "org:group": "B", "impact": "High", "product": "P1", "organization involved": "Org B", "org:resource": "person-2"},
        ])
        case = build_cases(events).iloc[0]
        self.assertTrue(case["is_resolved"])
        self.assertEqual(case["resolution_hours"], 3.0)
        self.assertEqual(case["wait_hours_proxy"], 2.0)
        self.assertAlmostEqual(case["wait_share_proxy"], 2 / 3)
        self.assertEqual(case["handoff_count"], 1)
        self.assertNotIn("org:resource", case.index)

    def test_open_case_is_right_censored_not_mislabeled_resolved(self):
        events = pd.DataFrame([
            {"case_id": "case-open", "time:timestamp": pd.Timestamp("2020-01-01T00:00:00Z"), "concept:name": "Accepted", "lifecycle:transition": "In Progress", "org:group": "A"},
            {"case_id": "case-open", "time:timestamp": pd.Timestamp("2020-01-01T03:00:00Z"), "concept:name": "Accepted", "lifecycle:transition": "In Call", "org:group": "A"},
        ])
        case = build_cases(events).iloc[0]
        self.assertFalse(case["is_resolved"])
        self.assertTrue(pd.isna(case["resolution_hours"]))
        self.assertEqual(case["observed_span_hours"], 3.0)
        result = case_survival_summary(pd.DataFrame([case]))
        self.assertEqual(result["resolved_events"], 0)
        self.assertEqual(result["right_censored_cases"], 1)

    def test_kaplan_meier_handles_censoring(self):
        cases = pd.DataFrame({
            "resolution_hours": [1.0, None, 3.0],
            "observed_span_hours": [1.0, 2.0, 3.0],
            "is_resolved": [True, False, True],
        })
        result = case_survival_summary(cases)
        self.assertEqual(result["resolved_events"], 2)
        self.assertEqual(result["right_censored_cases"], 1)
        self.assertEqual(result["kaplan_meier_median_resolution_hours"], 3.0)

    def test_temporal_diagnostic_flags_missing_positive_in_later_window(self):
        durations = [1, 2, 3, 4, 5, 100, 1, 2, 1, 2]
        cases = pd.DataFrame({
            "is_resolved": [True] * 10,
            "resolution_hours": durations,
        })
        result = temporal_distribution_diagnostic(cases)
        self.assertEqual(result["windows"]["train"]["closed_cases_above_train_p75"], 2)
        self.assertEqual(result["windows"]["later_period"]["closed_cases_above_train_p75"], 0)
        self.assertIn("Strong temporal distribution shift", result["interpretation"])


if __name__ == "__main__":
    unittest.main()
