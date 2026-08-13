import copy
import unittest

from cvsearch.eval.freeze_split_calibration import (
    extract_split_support_rows,
    freeze_split_calibration_suite,
    split_geometry_ordering_metrics,
)


def observed_row(*, ordinal=2, support=(0.1, 0.9)):
    return {
        "_eg_ordinal": ordinal,
        "input_image": f"images/{ordinal}.jpg",
        "bbox": [[10, 10, 10, 10]],
        "test_type": "direct_attributes",
        "method_trace": {"steps": [{
            "action": "SPLIT",
            "split_search_audit": {
                "render_policy": (
                    "native_2x2_overlap_support_screen_two_scale_depth2_v2"
                ),
                "branches": [{
                    "visit_index": 0,
                    "tight_view": {
                        "role": "tight", "crop_xyxy": [0, 0, 20, 20],
                        "raw_support": support[0],
                    },
                    "context_view": {
                        "role": "context", "crop_xyxy": [0, 0, 2000, 2000],
                        "raw_support": support[1],
                    },
                }],
            },
        }]},
    }


class SplitCalibrationFreezeTest(unittest.TestCase):
    def test_scores_root_and_branch_ordering_without_answer_labels(self):
        row = observed_row()
        row["method_trace"]["steps"][0]["split_search_audit"] = {
            "render_policy": (
                "native_2x2_overlap_support_screen_three_scale_all_roots_depth2_v3"
            ),
            "root_ranked_siblings": [
                {"box": [0, 0, 20, 20]},
                {"box": [20, 0, 40, 20]},
                {"box": [0, 20, 20, 40]},
                {"box": [20, 20, 40, 40]},
            ],
            "branches": [
                {"visit_index": index, "tight_view": {
                    "role": "tight", "crop_xyxy": crop,
                    "raw_support": 0.5,
                }, "context_view": {
                    "role": "context", "crop_xyxy": crop,
                    "raw_support": 0.5,
                }}
                for index, crop in enumerate((
                    [20, 0, 40, 20], [0, 20, 20, 40],
                    [20, 20, 40, 40], [0, 0, 20, 20],
                    [0, 0, 20, 20], [0, 0, 20, 20],
                ))
            ],
        }
        metrics = split_geometry_ordering_metrics("vstar", [row])
        self.assertEqual(metrics["topics"], 1)
        self.assertEqual(metrics["root_recall_at_1"], 1)
        self.assertEqual(metrics["root_recall_at_2"], 1)
        self.assertEqual(metrics["branch_recall_at_1"], 0)
        self.assertEqual(metrics["branch_recall_at_4"], 1)
        self.assertEqual(metrics["branch_recall_at_6"], 1)
        self.assertEqual(metrics["first_root_evidence_rank"], {"1": 1})
        self.assertEqual(metrics["first_branch_evidence_rank"], {"4": 1})
        self.assertEqual(metrics["rescue_only_recovered"], 0)

    def test_extracts_support_only_rows_from_evaluator_geometry(self):
        rows = extract_split_support_rows("vstar", [observed_row()])
        self.assertEqual(rows, [
            {
                "row_id": "vstar:2:0:tight",
                "source_group": "vstar:images/2.jpg",
                "raw_support": 0.1,
                "support_sufficient": 1,
            },
            {
                "row_id": "vstar:2:0:context",
                "source_group": "vstar:images/2.jpg",
                "raw_support": 0.9,
                "support_sufficient": 0,
            },
        ])
        self.assertFalse(any(
            forbidden in item
            for item in rows
            for forbidden in ("bbox", "answer", "correct", "benchmark")
        ))

    def test_rejects_non_development_policy_and_malformed_views(self):
        row = observed_row()
        audit = row["method_trace"]["steps"][0]["split_search_audit"]
        cases = []
        changed = copy.deepcopy(row)
        changed["split"] = "validation"
        cases.append(changed)
        changed = copy.deepcopy(row)
        changed["method_trace"]["steps"][0]["split_search_audit"][
            "render_policy"
        ] = "unknown"
        cases.append(changed)
        changed = copy.deepcopy(row)
        changed["method_trace"]["steps"][0]["split_search_audit"][
            "branches"
        ][0]["tight_view"]["role"] = "context"
        cases.append(changed)
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaises(
                (TypeError, ValueError)
            ):
                extract_split_support_rows("vstar", [changed])
        self.assertEqual(audit["branches"][0]["visit_index"], 0)

    def test_suite_freezes_independent_backbone_mappings_and_hashes(self):
        rows = {
            "qwen": {
                "vstar": [observed_row(ordinal=2), observed_row(ordinal=3)],
            },
            "internvl": {
                "vstar": [
                    observed_row(ordinal=2, support=(0.2, 0.8)),
                    observed_row(ordinal=3, support=(0.3, 0.7)),
                ],
            },
        }
        first = freeze_split_calibration_suite(rows)
        second = freeze_split_calibration_suite(copy.deepcopy(rows))
        self.assertEqual(first, second)
        self.assertEqual(set(first["backbones"]), {"qwen", "internvl"})
        self.assertEqual(first["data_scope"], "opened_development_only")
        def keys(value):
            if isinstance(value, dict):
                return set(value).union(*(keys(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value))
            return set()

        self.assertTrue({"bbox", "answer", "correct", "calibration_rows"}.isdisjoint(
            keys(first)
        ))
        for value in first["backbones"].values():
            self.assertIn(value["selected_weight"], (0.0, 0.125, 0.25, 0.5, 0.75, 1.0))
            self.assertEqual(len(value["manifest_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
