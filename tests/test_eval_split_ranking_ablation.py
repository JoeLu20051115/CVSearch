import copy
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cvsearch.eval.eval_split_ranking_ablation import (
    decompose_frozen_failures,
    evaluate_fixed_pool,
    exact_random_metrics,
    fixed_pool_scores,
)


def _box(root, child):
    root_x = 20 * (root % 2)
    root_y = 20 * (root // 2)
    child_x = root_x + 10 * (child % 2)
    child_y = root_y + 10 * (child // 2)
    return [child_x, child_y, child_x + 10, child_y + 10]


def _ranking_score(rank):
    clip = (3 - rank) / 3
    return 0.7 * clip + 0.3 * 0.5


def observed_row(*, ordinal=2):
    ranked_paths = (3, 2, 1, 0)
    roots = [
        {
            "path": [root],
            "box": [20 * (root % 2), 20 * (root // 2),
                    20 * (root % 2) + 20, 20 * (root // 2) + 20],
            "score": _ranking_score(rank),
        }
        for rank, root in enumerate(ranked_paths)
    ]
    probes = []
    for root in ranked_paths:
        for rank, child in enumerate(ranked_paths):
            probes.append({
                "patch_path": [root, child],
                "target_box_xyxy": _box(root, child),
                "ranking_score": _ranking_score(rank),
                "raw_support": 0.5,
            })
    branch_paths = ((3, 3), (2, 3), (3, 2), (3, 1), (1, 3), (0, 3))
    branches = []
    for index, path in enumerate(branch_paths):
        view = {
            "role": "tight",
            "patch_path": list(path),
            "crop_xyxy": _box(*path),
            "answer": 1,
        }
        branches.append({
            "visit_index": index,
            "observed_path": list(path),
            "tight_view": view,
            "context_view": {**view, "role": "context"},
        })
    return {
        "_eg_ordinal": ordinal,
        "input_image": "image.jpg",
        "question": "What color is the target?",
        "bbox": [[35, 35, 2, 2]],
        "test_type": "direct_attributes",
        "output": 0,
        "method_trace": {"steps": [{
            "action": "SPLIT",
            "split_search_audit": {
                "render_policy": (
                    "native_2x2_overlap_support_screen_three_scale_"
                    "all_roots_depth2_v3"
                ),
                "root_ranked_siblings": roots,
                "screening_probes": probes,
                "branches": branches,
            },
        }]},
    }


def _vstar_failure_row(ordinal, bbox, winners):
    row = observed_row(ordinal=ordinal)
    row.update({
        "bbox": [list(bbox)],
        "answer_type": "logits_match",
        "output": 1,
    })
    audit = row["method_trace"]["steps"][0]["split_search_audit"]
    for branch, winner in zip(audit["branches"], winners):
        answer = {"winner": winner, "losses": [float(winner), float(1 - winner)]}
        branch["tight_view"]["answer"] = copy.deepcopy(answer)
        branch["context_view"]["answer"] = copy.deepcopy(answer)
    return row


def _decision(ordinal, before, after, *, selected_source="P0"):
    return {
        "ordinal": ordinal,
        "stage2_output": 0 if before else 1,
        "stage2_correct": [before],
        "stage3b_output": 0 if after else 1,
        "stage3b_correct": [after],
        "selected_source": selected_source,
        "selected_branch": 0 if selected_source == "SPLIT" else None,
        "reason": "test",
    }


def _score_report(rows, *, corrections, corruptions):
    return {
        "aggregate": {
            "topics": len(rows),
            "official_units": len(rows),
            "stage2_correct": sum(row["stage2_correct"][0] for row in rows),
            "stage3b_correct": sum(row["stage3b_correct"][0] for row in rows),
            "corrections": corrections,
            "corruptions": corruptions,
        },
        "cells": {"qwen/vstar": {"rows": rows}},
    }


class FixedPoolReconstructionTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (40, 40), "gray")

    def test_recovers_clip_percentiles_from_frozen_combined_scores(self):
        scores = fixed_pool_scores(observed_row(), self.image)
        self.assertEqual(len(scores), 16)
        self.assertAlmostEqual(scores[(3, 3)]["clip_only"], 1.0)
        self.assertAlmostEqual(scores[(0, 0)]["clip_only"], 0.0)
        self.assertAlmostEqual(scores[(3, 3)]["visual_only"], 0.5)
        self.assertAlmostEqual(scores[(3, 3)]["combined"], 0.85)

    def test_rejects_any_candidate_pool_other_than_exact_sixteen(self):
        row = observed_row()
        audit = row["method_trace"]["steps"][0]["split_search_audit"]
        audit["screening_probes"].pop()
        with self.assertRaisesRegex(ValueError, "sixteen"):
            fixed_pool_scores(row, self.image)

    def test_rejects_duplicate_candidate_identity(self):
        row = observed_row()
        probes = row["method_trace"]["steps"][0]["split_search_audit"][
            "screening_probes"
        ]
        probes[-1] = copy.deepcopy(probes[0])
        with self.assertRaisesRegex(ValueError, "unique"):
            fixed_pool_scores(row, self.image)


class FixedPoolMetricTest(unittest.TestCase):
    def test_exact_random_baseline_uses_closed_form_not_sampling(self):
        metrics = exact_random_metrics([True] + [False] * 15, (1, 3, 6))
        self.assertEqual(metrics["recall_at"], {
            "1": 1 / 16,
            "3": 3 / 16,
            "6": 6 / 16,
        })
        self.assertAlmostEqual(
            metrics["mrr"], sum(1 / rank for rank in range(1, 17)) / 16,
        )
        self.assertEqual(metrics["pool_recall"], 1.0)

    def test_scores_each_backbone_over_the_same_candidate_set(self):
        row = observed_row()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (40, 40), "gray").save(root / "image.jpg")
            report = evaluate_fixed_pool(
                {
                    "qwen": {"vstar": [row]},
                    "internvl": {"vstar": [copy.deepcopy(row)]},
                },
                {"vstar": root},
            )
        self.assertEqual(report["source_topics"], 1)
        self.assertEqual(report["topic_backbone_evaluations"], 2)
        self.assertEqual(report["candidate_pool_size"], 16)
        self.assertEqual(report["backbone_copies_verified"], 2)
        self.assertEqual(report["policies"]["combined"]["recall_at"]["1"], 1.0)
        self.assertEqual(report["policies"]["grid"]["recall_at"]["6"], 0.0)
        self.assertEqual(
            report["policies"]["random_expected"]["recall_at"]["3"],
            3 / 16,
        )

    def test_allows_backbone_specific_scores(self):
        first = observed_row()
        second = copy.deepcopy(first)
        second["method_trace"]["steps"][0]["split_search_audit"][
            "screening_probes"
        ][0]["ranking_score"] -= 0.01
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (40, 40), "gray").save(root / "image.jpg")
            report = evaluate_fixed_pool(
                {
                    "qwen": {"vstar": [first]},
                    "internvl": {"vstar": [second]},
                },
                {"vstar": root},
            )
        self.assertEqual(report["topic_backbone_evaluations"], 2)

    def test_rejects_cross_backbone_candidate_geometry_drift(self):
        first = observed_row()
        second = copy.deepcopy(first)
        second["method_trace"]["steps"][0]["split_search_audit"][
            "screening_probes"
        ][0]["target_box_xyxy"][0] += 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (40, 40), "gray").save(root / "image.jpg")
            with self.assertRaisesRegex(ValueError, "candidate geometry"):
                evaluate_fixed_pool(
                    {
                        "qwen": {"vstar": [first]},
                        "internvl": {"vstar": [second]},
                    },
                    {"vstar": root},
                )


class FrozenFailureDecompositionTest(unittest.TestCase):
    def test_partitions_errors_and_refines_geometry_failures(self):
        decisions = [
            _decision(1, False, True, selected_source="SPLIT"),
            _decision(2, False, False, selected_source="ZOOM"),
            _decision(3, False, False, selected_source="SPLIT"),
            _decision(4, False, False),
            _decision(5, False, False),
            _decision(6, False, False),
            _decision(7, True, False, selected_source="SPLIT"),
        ]
        wrong = [1] * 6
        correct_available = [0, 1, 1, 1, 1, 1]
        observations = [
            _vstar_failure_row(1, [35, 35, 2, 2], correct_available),
            _vstar_failure_row(2, [35, 35, 2, 2], correct_available),
            _vstar_failure_row(3, [35, 35, 2, 2], correct_available),
            _vstar_failure_row(4, [100, 100, 2, 2], wrong),
            _vstar_failure_row(5, [5, 5, 2, 2], wrong),
            _vstar_failure_row(6, [35, 35, 2, 2], wrong),
            _vstar_failure_row(7, [35, 35, 2, 2], wrong),
        ]
        report = decompose_frozen_failures(
            _score_report(decisions, corrections=1, corruptions=1),
            {"qwen/vstar": observations},
        )
        self.assertEqual(report["aggregate"]["stage2_errors"], 6)
        self.assertEqual(report["aggregate"]["converted"], 1)
        self.assertEqual(report["aggregate"]["selector_abstained"], 1)
        self.assertEqual(report["aggregate"]["selector_wrong_choice"], 1)
        self.assertEqual(report["aggregate"]["no_correct_observed_answer"], 3)
        self.assertEqual(report["aggregate"]["geometry_pool_miss"], 1)
        self.assertEqual(report["aggregate"]["six_branch_budget_miss"], 1)
        self.assertEqual(report["aggregate"]["vlm_answer_miss"], 1)
        self.assertEqual(report["aggregate"]["corruption"], 1)
        self.assertTrue(all(report["gates"].values()))

    def test_rejects_frozen_report_accounting_drift(self):
        decision = _decision(1, False, True, selected_source="SPLIT")
        raw = _vstar_failure_row(1, [35, 35, 2, 2], [0] * 6)
        with self.assertRaisesRegex(ValueError, "corrections"):
            decompose_frozen_failures(
                _score_report([decision], corrections=0, corruptions=0),
                {"qwen/vstar": [raw]},
            )


if __name__ == "__main__":
    unittest.main()
