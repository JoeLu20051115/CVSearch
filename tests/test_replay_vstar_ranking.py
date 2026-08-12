import copy
import unittest

from cvsearch.eval.replay_vstar_ranking import (
    _event_query_profile,
    replay_event,
    replay_rows,
)
from cvsearch.evidence_gap.query_profile import adaptive_alpha, infer_query_profile


def _detail(identifier, *, main, augmented, complexity, edge, rank=0.0):
    return {
        "node_id": identifier,
        "target": "comb",
        "stage": "Depth 2",
        "tree_scope": "main",
        "crop_origin": [0, 0],
        "bbox_original": [0, 0, 4, 4] if identifier == "hit" else [10, 10, 2, 2],
        "score": {
            "main": main,
            "augmented": augmented,
            "complexity": complexity,
            "edge_density": edge,
            "rank": rank,
        },
    }


def _row(details):
    return {
        "question": "What is the color of the comb?",
        "target_object": ["comb"],
        "bbox": [[1, 1, 1, 1]],
        "output": 0,
        "test_type": "direct_attributes",
        "method_trace": {"candidate_ranks": details},
    }


class RankingReplayTest(unittest.TestCase):
    def test_reconstructs_fusion_and_preserves_ties(self):
        low = _detail(
            "low", main=0.0, augmented=1.0, complexity=0.0, edge=1.0,
        )
        high = _detail(
            "high", main=1.0, augmented=0.0, complexity=1.0, edge=0.0,
        )

        ranked = replay_event([low, high], beta=0.6, alpha=0.65, visual_lambda=0.5)
        tied = replay_event([low, high], beta=0.5, alpha=0.5, visual_lambda=0.5)

        self.assertEqual([item["node_id"] for item in ranked], ["high", "low"])
        self.assertEqual([item["node_id"] for item in tied], ["low", "high"])

    def test_replay_is_read_only_and_reports_component_configs(self):
        miss = _detail(
            "miss", main=1.0, augmented=1.0, complexity=0.0, edge=0.0,
        )
        hit = _detail(
            "hit", main=0.0, augmented=0.0, complexity=1.0, edge=1.0,
        )
        rows = [_row([miss, hit])]
        before = copy.deepcopy(rows)
        configs = [
            {"name": "relevance_only", "beta": 0.5, "alpha": 1.0, "visual_lambda": 0.5},
            {"name": "visual_only", "beta": 0.5, "alpha": 0.0, "visual_lambda": 0.5},
        ]

        report = replay_rows(rows, configs, k=3)

        self.assertEqual(rows, before)
        self.assertEqual([item["config"]["name"] for item in report["candidates"]], [
            "relevance_only", "visual_only",
        ])
        self.assertEqual(report["candidates"][0]["metrics"]["topic"]["recall_at"]["1"], 0.0)
        self.assertEqual(report["candidates"][1]["metrics"]["topic"]["recall_at"]["1"], 1.0)
        self.assertEqual(len(rows[0]["method_trace"]["candidate_ranks"]), 2)

    def test_adaptive_config_changes_alpha_from_question_only(self):
        miss = _detail(
            "miss", main=1.0, augmented=1.0, complexity=0.0, edge=0.0,
        )
        hit = _detail(
            "hit", main=0.0, augmented=0.0, complexity=1.0, edge=1.0,
        )
        configs = [{
            "name": "adaptive",
            "beta": 0.5,
            "alpha": 0.65,
            "visual_lambda": 0.5,
            "detail_alpha_discount": 0.65,
            "context_alpha_gain": 0.0,
        }]

        report = replay_rows([_row([miss, hit])], configs, k=3)

        candidate = report["candidates"][0]
        self.assertEqual(candidate["metrics"]["topic"]["recall_at"]["1"], 1.0)
        self.assertEqual(candidate["config"]["detail_alpha_discount"], 0.65)

    def test_replay_profile_matches_runtime_augmented_queries(self):
        row = _row([_detail(
            "hit", main=1.0, augmented=1.0, complexity=1.0, edge=1.0,
        )])
        row["question"] = (
            "Which one is closer to the camera, the black vehicle or the silver vehicle?"
        )
        row["method_trace"]["query_plan"] = {
            "main_query": row["question"],
            "augmented_queries": ["black vehicle", "silver vehicle"],
        }
        event = row["method_trace"]["candidate_ranks"]

        replay_profile = _event_query_profile(row, event)
        runtime_profile = infer_query_profile(
            row["method_trace"]["query_plan"]["main_query"],
            row["method_trace"]["query_plan"]["augmented_queries"],
        )

        self.assertEqual(replay_profile, runtime_profile)
        self.assertEqual(replay_profile.context_demand, 1.0)
        self.assertEqual(
            adaptive_alpha(0.25, replay_profile, 0.15, 0.45),
            0.7,
        )

    def test_context_visual_discount_replays_edge_weight(self):
        complexity_hit = _detail(
            "hit", main=0.5, augmented=0.5, complexity=1.0, edge=0.0,
        )
        edge_miss = _detail(
            "miss", main=0.5, augmented=0.5, complexity=0.0, edge=1.0,
        )
        row = _row([complexity_hit, edge_miss])
        row["question"] = "Is the comb left of the cup?"
        row["method_trace"]["query_plan"] = {
            "main_query": row["question"],
            "augmented_queries": ["comb", "cup"],
        }
        report = replay_rows([row], [{
            "name": "phase1-v2",
            "beta": 1.0,
            "alpha": 0.0,
            "visual_lambda": 1.0,
            "detail_alpha_discount": 0.0,
            "context_alpha_gain": 0.0,
            "context_visual_discount": 1.0,
        }])

        candidate = report["candidates"][0]
        self.assertEqual(candidate["config"]["context_visual_discount"], 1.0)
        self.assertEqual(candidate["metrics"]["topic"]["recall_at"]["1"], 0.0)


if __name__ == "__main__":
    unittest.main()
