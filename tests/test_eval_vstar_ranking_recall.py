import unittest

from cvsearch.eval.eval_vstar_ranking_recall import (
    center_hit,
    evaluate_rows,
    group_rank_events,
)


def _detail(target, stage, bbox, rank):
    return {
        "target": target,
        "stage": stage,
        "tree_scope": "main",
        "crop_origin": [0, 0],
        "bbox_original": list(bbox),
        "score": {"rank": rank},
    }


def _row(*, target_object, bbox, details, output=0, test_type="direct_attributes"):
    return {
        "target_object": list(target_object),
        "bbox": [list(box) for box in bbox],
        "output": output,
        "test_type": test_type,
        "method_trace": {"candidate_ranks": list(details)},
    }


class RankingRecallGeometryTest(unittest.TestCase):
    def test_center_hit_uses_xywh_and_includes_boundary(self):
        self.assertTrue(center_hit([10, 20, 4, 6], [12, 23, 8, 8]))
        self.assertFalse(center_hit([10, 20, 4, 6], [0, 0, 11, 22]))

    def test_rank_events_split_when_runtime_context_changes(self):
        details = [
            _detail("comb", "Depth 2", [0, 0, 5, 5], 0.9),
            _detail("comb", "Depth 2", [5, 0, 5, 5], 0.8),
            _detail("comb", "Depth 1", [0, 0, 10, 10], 0.7),
        ]

        events = group_rank_events(details)

        self.assertEqual([len(event) for event in events], [2, 1])


class RankingRecallEvaluationTest(unittest.TestCase):
    def test_relation_metrics_distinguish_any_hit_from_complete_evidence(self):
        row = _row(
            target_object=["cup", "plate"],
            bbox=[[1, 1, 1, 1], [11, 1, 1, 1]],
            details=[
                _detail("cup", "Depth 2", [10, 0, 3, 3], 0.9),
                _detail("plate", "Depth 1", [10, 0, 3, 3], 0.8),
            ],
            test_type="relative_position",
        )

        report = evaluate_rows([row])

        self.assertEqual(report["topic"]["recall_at"]["3"], 1.0)
        self.assertEqual(
            report["topic"]["all_required_targets_recall_at"]["3"], 0.0,
        )
        self.assertEqual(
            report["event"]["target_conditioned_recall_at"]["3"], 0.5,
        )

    def test_reports_topic_event_recall_and_failure_taxonomy(self):
        hit_second = _row(
            target_object=["comb"],
            bbox=[[10, 10, 2, 2]],
            details=[
                _detail("comb", "Depth 2", [0, 0, 2, 2], 0.9),
                _detail("comb", "Depth 2", [9, 9, 4, 4], 0.8),
            ],
        )
        planner_mismatch = _row(
            target_object=["guard's glove"],
            bbox=[[10, 10, 2, 2]],
            details=[_detail("guard", "Depth 2", [9, 9, 4, 4], 0.9)],
        )
        quick = _row(target_object=["cup"], bbox=[[0, 0, 1, 1]], details=[])
        candidate_miss = _row(
            target_object=["cup"],
            bbox=[[10, 10, 2, 2]],
            details=[_detail("cup", "Depth 2", [0, 0, 2, 2], 0.9)],
        )
        rank_miss = _row(
            target_object=["ball"],
            bbox=[[10, 10, 2, 2]],
            details=[
                _detail("ball", "Depth 2", [0, 0, 2, 2], 0.9),
                _detail("ball", "Depth 2", [2, 0, 2, 2], 0.8),
                _detail("ball", "Depth 2", [4, 0, 2, 2], 0.7),
                _detail("ball", "Depth 2", [9, 9, 4, 4], 0.6),
            ],
            test_type="relative_position",
        )
        answer_miss = _row(
            target_object=["hat"],
            bbox=[[10, 10, 2, 2]],
            details=[_detail("hat", "Depth 2", [9, 9, 4, 4], 0.9)],
            output=1,
        )

        report = evaluate_rows([
            hit_second, planner_mismatch, quick, candidate_miss, rank_miss, answer_miss,
        ])

        self.assertEqual(report["row_count"], 6)
        self.assertEqual(report["ranked_topic_count"], 5)
        self.assertEqual(report["rank_event_count"], 5)
        self.assertEqual(report["topic"]["recall_at"]["1"], 2 / 5)
        self.assertEqual(report["topic"]["recall_at"]["3"], 3 / 5)
        self.assertEqual(report["topic"]["pool_upper_bound"], 4 / 5)
        self.assertEqual(report["event"]["recall_at"]["3"], 3 / 5)
        self.assertIn("all_required_targets_recall_at", report["topic"])
        self.assertIn("target_conditioned_recall_at", report["event"])
        self.assertEqual(report["failure_counts"], {
            "quick_or_unranked": 1,
            "planner_mismatch": 1,
            "candidate_miss": 1,
            "rank_miss": 1,
            "answer_miss": 1,
            "success": 1,
        })
        self.assertEqual(report["answer_accuracy"], 5 / 6)
        self.assertIn("direct_attributes", report["by_test_type"])
        self.assertIn("relative_position", report["by_test_type"])


if __name__ == "__main__":
    unittest.main()
