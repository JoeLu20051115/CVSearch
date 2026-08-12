import unittest

from cvsearch.eval.score_cross_backbone_transfer import (
    calibration_metrics,
    extract_support_rows,
    score_paired_dataset,
)
from cvsearch.eval.replay_adaptive_search import freeze_selected_calibration


class CrossBackboneTransferScoringTest(unittest.TestCase):
    def test_extracts_current_and_candidate_support_without_answer_fields(self):
        row = {
            "_eg_ordinal": 8,
            "input_image": "direct_attributes/example.jpg",
            "test_type": "direct_attributes",
            "bbox": [[10, 10, 5, 5]],
            "answer": "must-not-be-copied",
            "method_trace": {"steps": [{
                "action": "ZOOM",
                "zoom_audit": {
                    "batch_result": {"status": "success"},
                    "coordinate_mapping": [{
                        "current_crop_xyxy": [0, 0, 1000, 1000],
                        "candidate_crop_xyxy": [0, 0, 100, 100],
                    }],
                    "current_gap_support": {"p_yes": 0.2},
                    "candidate_gap_support": {"p_yes": 0.8},
                },
            }]},
        }

        support = extract_support_rows("vstar", [row])

        self.assertEqual(len(support), 2)
        self.assertEqual(
            [item["support_sufficient"] for item in support], [0, 1],
        )
        self.assertEqual(
            [item["raw_support"] for item in support], [0.2, 0.8],
        )
        self.assertTrue(all(item["source_group"] == row["input_image"] for item in support))
        self.assertTrue(all(set(item) == {
            "row_id", "source_group", "raw_support", "support_sufficient",
        } for item in support))

    def test_calibration_metrics_reports_no_harm_identity(self):
        calibration = freeze_selected_calibration([
            {"row_id": "a", "source_group": "g1", "raw_support": 0.1, "support_sufficient": 0},
            {"row_id": "b", "source_group": "g1", "raw_support": 0.9, "support_sufficient": 1},
            {"row_id": "c", "source_group": "g2", "raw_support": 0.2, "support_sufficient": 0},
            {"row_id": "d", "source_group": "g2", "raw_support": 0.8, "support_sufficient": 1},
        ])
        rows = [
            {"raw_support": 0.25, "support_sufficient": 0},
            {"raw_support": 0.75, "support_sufficient": 1},
        ]

        result = calibration_metrics(rows, calibration)

        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["positive"], 1)
        self.assertGreaterEqual(result["raw_auroc"], 0.5)
        self.assertGreaterEqual(result["calibrated_auroc"], 0.5)
        self.assertIn("raw_brier", result)
        self.assertIn("calibrated_ece_10", result)

    def test_scores_vstar_and_treebench_at_official_units(self):
        vstar_baseline = [{"_eg_ordinal": 4, "output": 1}]
        vstar_selected = [{
            "_eg_ordinal": 4,
            "selected_output": 0,
            "selected_source": "ZOOM",
            "reason": "test",
        }]
        treebench_baseline = [{
            "_eg_ordinal": 7, "output": "B", "answer": "A",
        }]
        treebench_selected = [{
            "_eg_ordinal": 7,
            "selected_output": "A",
            "selected_source": "EXPAND",
            "reason": "test",
        }]

        vstar = score_paired_dataset("vstar", vstar_baseline, vstar_selected)
        treebench = score_paired_dataset(
            "treebench", treebench_baseline, treebench_selected,
        )

        self.assertEqual(vstar["official_units"], 1)
        self.assertEqual(vstar["corrections"], 1)
        self.assertEqual(vstar["accuracy_delta"], 1.0)
        self.assertEqual(treebench["corrections"], 1)
        self.assertEqual(treebench["parseable_outputs"], 1)
        self.assertEqual(treebench["selected_actions"], {"EXPAND": 1})

    def test_scores_hr_cycles_without_topic_averaging(self):
        baseline = [{
            "_eg_ordinal": 2,
            "answer": ["A", "B", "C", "D"],
            "output": ["A", "B", "x", "D"],
        }]
        selected = [{
            "_eg_ordinal": 2,
            "selected_output": ["A", "B", "C", "D"],
            "selected_source": "ZOOM",
            "reason": "test",
        }]

        score = score_paired_dataset("hr-bench_4k", baseline, selected)

        self.assertEqual(score["topics"], 1)
        self.assertEqual(score["official_units"], 4)
        self.assertEqual(score["baseline_correct"], 3)
        self.assertEqual(score["selected_correct"], 4)
        self.assertEqual(score["corrections"], 1)


if __name__ == "__main__":
    unittest.main()
