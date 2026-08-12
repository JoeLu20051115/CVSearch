import copy
import math
import unittest

from cvsearch.eval.eval_adaptive_search import evaluate_adaptive_rows


class AdaptiveSearchEvaluationTest(unittest.TestCase):
    SHA_A = "a" * 64
    SHA_B = "b" * 64

    @classmethod
    def rows(cls):
        return [
            {
                "row_id": "topic-1",
                "baseline_correct": False,
                "selected_correct": True,
                "support_label": 1,
                "raw_support": 0.6,
                "calibrated_support": 0.8,
                "stopped": True,
                "action": "ZOOM",
                "support_before": 0.4,
                "support_after": 0.8,
                "backtracked": False,
                "recovered_after_backtrack": False,
                "trajectory_length": 3,
                "added_calls": 2,
                "added_pixels": 100,
                "added_latency_seconds": 1.0,
                "baseline_rank_sha256": cls.SHA_A,
                "selected_rank_sha256": cls.SHA_A,
            },
            {
                "row_id": "topic-2",
                "baseline_correct": True,
                "selected_correct": True,
                "support_label": 0,
                "raw_support": 0.4,
                "calibrated_support": 0.3,
                "stopped": True,
                "action": "STOP",
                "support_before": 0.3,
                "support_after": 0.3,
                "backtracked": False,
                "recovered_after_backtrack": False,
                "trajectory_length": 1,
                "added_calls": 1,
                "added_pixels": 50,
                "added_latency_seconds": 0.5,
                "baseline_rank_sha256": cls.SHA_B,
                "selected_rank_sha256": cls.SHA_B,
            },
        ]

    def test_reports_accuracy_calibration_false_stop_cost_and_rank_identity(self):
        report = evaluate_adaptive_rows(self.rows())

        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["paired"]["corrections"], 1)
        self.assertEqual(report["paired"]["corruptions"], 0)
        self.assertEqual(report["paired"]["baseline_accuracy"], 0.5)
        self.assertEqual(report["paired"]["selected_accuracy"], 1.0)
        self.assertEqual(report["safety"]["false_stops"], 1)
        self.assertEqual(report["safety"]["false_stop_rate"], 0.5)
        self.assertAlmostEqual(report["calibration"]["raw_brier"], 0.16)
        self.assertAlmostEqual(report["calibration"]["calibrated_brier"], 0.065)
        self.assertLess(
            report["calibration"]["calibrated_brier"],
            report["calibration"]["raw_brier"],
        )
        self.assertEqual(report["actions"]["ZOOM"]["mean_support_gain"], 0.4)
        self.assertEqual(report["actions"]["ZOOM"]["corrections"], 1)
        self.assertEqual(report["cost"]["total_added_calls"], 3)
        self.assertEqual(report["cost"]["mean_trajectory_length"], 2.0)
        self.assertTrue(report["phase1"]["all_rank_traces_preserved"])

    def test_reports_backtrack_recovery_and_rank_drift(self):
        rows = self.rows()
        rows[0]["backtracked"] = True
        rows[0]["recovered_after_backtrack"] = True
        rows[1]["selected_rank_sha256"] = self.SHA_A

        report = evaluate_adaptive_rows(rows)

        self.assertEqual(report["trajectory"]["backtracks"], 1)
        self.assertEqual(report["trajectory"]["recoveries"], 1)
        self.assertEqual(report["trajectory"]["backtrack_recovery_rate"], 1.0)
        self.assertFalse(report["phase1"]["all_rank_traces_preserved"])
        self.assertEqual(report["phase1"]["rank_trace_drifts"], 1)

    def test_ece_uses_fixed_ten_bins_and_includes_zero_and_one(self):
        rows = self.rows()
        rows[0]["calibrated_support"] = 1.0
        rows[1]["calibrated_support"] = 0.0

        report = evaluate_adaptive_rows(rows)

        self.assertEqual(report["calibration"]["calibrated_ece_10"], 0.0)

    def test_rejects_duplicate_ids_malformed_values_and_schema_drift(self):
        duplicate = self.rows()
        duplicate[1]["row_id"] = duplicate[0]["row_id"]
        bad_cases = [duplicate]
        for field, value in (
            ("calibrated_support", 1.1),
            ("raw_support", math.nan),
            ("added_calls", True),
            ("added_pixels", -1),
            ("added_latency_seconds", math.inf),
            ("support_label", True),
            ("baseline_correct", 1),
            ("baseline_rank_sha256", "short"),
            ("action", "DELETE"),
        ):
            rows = self.rows()
            rows[0][field] = value
            bad_cases.append(rows)
        extra = self.rows()
        extra[0]["benchmark"] = "forbidden-at-policy-boundary"
        bad_cases.append(extra)

        for rows in bad_cases:
            with self.subTest(rows=rows):
                with self.assertRaises((TypeError, ValueError)):
                    evaluate_adaptive_rows(rows)

    def test_does_not_mutate_rows(self):
        rows = self.rows()
        before = copy.deepcopy(rows)

        evaluate_adaptive_rows(rows)

        self.assertEqual(rows, before)


if __name__ == "__main__":
    unittest.main()
