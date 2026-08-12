import unittest

from cvsearch.eval.score_stage3_transfer import (
    evaluate_stage3_rows,
    split_candidate_oracle,
)


def row(
    row_id, backbone, dataset, stage2, stage3, candidates, *, source="P0",
    raw=0.4, calibrated=0.4, sufficient=0, trajectory=None,
    calls=0, pixels=0, latency=0.0, branches=0, depth=0, backtracks=0,
):
    return {
        "row_id": row_id,
        "backbone": backbone,
        "dataset": dataset,
        "stage2_correct": tuple(stage2),
        "stage3_correct": tuple(stage3),
        "candidate_correct": tuple(tuple(value) for value in candidates),
        "selected_source": source,
        "raw_support": raw,
        "calibrated_support": calibrated,
        "support_sufficient": sufficient,
        "trajectory_s": trajectory,
        "calls": calls,
        "pixels": pixels,
        "latency_seconds": latency,
        "branch_count": branches,
        "max_depth": depth,
        "backtracks": backtracks,
    }


class Stage3TransferScoringTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            row(
                "q-v-1", "qwen", "vstar", (False,), (True,), ((True,),),
                source="SPLIT", raw=0.8, calibrated=0.75, sufficient=1,
                trajectory=3, calls=16, pixels=1600, latency=1.5,
                branches=2, depth=2, backtracks=0,
            ),
            row(
                "q-h-1", "qwen", "hr_bench_4k", (True, False), (True, False),
                ((True, True),), raw=0.3, calibrated=0.25, sufficient=0,
            ),
            row(
                "i-v-1", "internvl", "vstar", (True,), (True,), ((True,),),
                raw=0.7, calibrated=0.65, sufficient=1,
            ),
            row(
                "i-h-1", "internvl", "hr_bench_4k", (False, True), (False, True),
                ((False, True),), raw=0.2, calibrated=0.20, sufficient=0,
            ),
        ]

    def test_reports_cells_pooled_accuracy_calibration_trajectory_and_cost(self):
        report = evaluate_stage3_rows(self.rows)
        self.assertEqual(report["aggregate"]["official_units"], 6)
        self.assertEqual(report["aggregate"]["stage2_correct"], 3)
        self.assertEqual(report["aggregate"]["stage3_correct"], 4)
        self.assertEqual(report["aggregate"]["corrections"], 1)
        self.assertEqual(report["aggregate"]["corruptions"], 0)
        self.assertEqual(report["cells"]["qwen"]["vstar"]["accuracy_delta"], 1.0)
        self.assertEqual(report["by_backbone"]["qwen"]["accuracy_delta"], 1 / 3)
        self.assertEqual(report["by_dataset"]["vstar"]["accuracy_delta"], 0.5)
        self.assertEqual(report["trajectory"]["positive"], 1)
        self.assertEqual(report["cost"]["total_calls"], 16)
        self.assertIn("raw_brier", report["calibration"])
        self.assertIn("calibrated_auroc", report["calibration"])

    def test_candidate_oracle_counts_new_fixes_by_backbone_and_dataset(self):
        oracle = split_candidate_oracle(self.rows)
        self.assertEqual(oracle["cells"]["qwen"]["vstar"]["baseline_errors"], 1)
        self.assertEqual(oracle["cells"]["qwen"]["vstar"]["oracle_fixes"], 1)
        self.assertEqual(
            oracle["cells"]["qwen"]["hr_bench_4k"]["oracle_fixes"], 1,
        )
        self.assertEqual(oracle["by_backbone"]["internvl"]["oracle_fixes"], 0)

    def test_rejects_duplicate_rows_shape_drift_and_nonfinite_cost(self):
        cases = (
            self.rows + [self.rows[0]],
            [dict(self.rows[0], stage3_correct=(True, False))],
            [dict(self.rows[0], latency_seconds=float("nan"))],
            [dict(self.rows[0], selected_source="ZOOM")],
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                evaluate_stage3_rows(values)


if __name__ == "__main__":
    unittest.main()
