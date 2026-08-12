import inspect
import math
import unittest

import cvsearch.evidence_gap.split_search as split_search
from cvsearch.evidence_gap.split_search import (
    SplitPatch,
    confirm_split_branch,
    generate_split_children,
    mann_kendall_s,
    rank_split_children,
)
from cvsearch.eval.phase12_generated_query_lazy_search import (
    AdaptivePatch,
    generate_lazy_children,
)


class SplitGeometryTest(unittest.TestCase):
    def test_geometry_matches_frozen_phase12_contract_at_two_depths(self):
        for box in ((0, 0, 100, 80), (3, 7, 104, 90), (9, 11, 14, 18)):
            with self.subTest(box=box):
                production_root = SplitPatch((), box)
                historical_root = AdaptivePatch((), box)
                production = generate_split_children(production_root)
                historical = generate_lazy_children(historical_root)
                self.assertEqual(
                    [(patch.path, patch.box) for patch in production],
                    [(patch.path, patch.box) for patch in historical],
                )
                self.assertEqual(
                    [
                        (patch.path, patch.box)
                        for patch in generate_split_children(production[0])
                    ],
                    [
                        (patch.path, patch.box)
                        for patch in generate_lazy_children(historical[0])
                    ],
                )

    def test_children_are_unique_contained_and_have_stable_identities(self):
        root = SplitPatch((), (10, 20, 111, 101))
        children = generate_split_children(root)
        self.assertEqual([patch.identity for patch in children], [
            "p0", "p1", "p2", "p3",
        ])
        self.assertEqual(len({patch.box for patch in children}), 4)
        for patch in children:
            x0, y0, x1, y1 = patch.box
            self.assertTrue(10 <= x0 < x1 <= 111)
            self.assertTrue(20 <= y0 < y1 <= 101)

    def test_invalid_or_too_small_parent_fails_closed(self):
        invalid = (
            ((), (0, 0, 0, 1)),
            ((), (-1, 0, 4, 4)),
            ((4,), (0, 0, 4, 4)),
            ((0, 1, 2), (0, 0, 4, 4)),
        )
        for path, box in invalid:
            with self.subTest(path=path, box=box), self.assertRaises(
                (TypeError, ValueError)
            ):
                generate_split_children(SplitPatch(path, box))

    def test_inference_module_has_no_evaluator_dependency(self):
        source = inspect.getsource(split_search)
        self.assertNotIn("cvsearch.eval", source)


class SplitRankingTest(unittest.TestCase):
    def setUp(self):
        self.children = generate_split_children(SplitPatch((), (0, 0, 100, 100)))

    def test_query_relevance_dominates_visual_information(self):
        ranked = rank_split_children(
            self.children,
            relevance=(0.1, 0.9, 0.2, 0.3),
            edge_density=(1.0, 0.0, 0.0, 0.0),
            feature_deviation=(1.0, 0.0, 0.0, 0.0),
        )
        self.assertEqual(ranked[0].patch.path, (1,))
        self.assertGreater(ranked[0].score, ranked[1].score)

    def test_visual_score_uses_edge_and_feature_deviation_equally(self):
        ranked = rank_split_children(
            self.children,
            relevance=(0.0, 0.0, 0.0, 0.0),
            edge_density=(1.0, 0.0, 0.4, 0.4),
            feature_deviation=(0.0, 1.0, 0.4, 0.4),
        )
        by_path = {item.patch.path: item for item in ranked}
        self.assertAlmostEqual(
            by_path[(0,)].visual_information,
            by_path[(1,)].visual_information,
        )
        self.assertEqual(
            [item.patch.path for item in ranked if item.score == ranked[0].score],
            [(0,), (1,), (2,), (3,)],
        )

    def test_ranking_rejects_misaligned_or_nonfinite_measurements(self):
        cases = (
            ((0.1,), (0.1,) * 4, (0.1,) * 4),
            ((0.1,) * 4, (0.1, math.nan, 0.1, 0.1), (0.1,) * 4),
            ((0.1,) * 4, (0.1,) * 4, (0.1, True, 0.1, 0.1)),
        )
        for relevance, edge, deviation in cases:
            with self.subTest(values=(relevance, edge, deviation)), self.assertRaises(
                (TypeError, ValueError)
            ):
                rank_split_children(self.children, relevance, edge, deviation)


class SplitTrajectoryTest(unittest.TestCase):
    def test_mann_kendall_direction(self):
        self.assertEqual(mann_kendall_s((0.1, 0.4, 0.8)), 3)
        self.assertEqual(mann_kendall_s((0.8, 0.4, 0.1)), -3)
        self.assertEqual(mann_kendall_s((0.4, 0.4, 0.4)), 0)
        self.assertEqual(mann_kendall_s((0.2, 0.8, 0.5)), 1)
        for values in ((), (0.2,), (0.1, math.inf), (0.1, True)):
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                mann_kendall_s(values)

    def confirmation(self, **overrides):
        arguments = {
            "p0_answer": "A",
            "p0_support": 0.30,
            "tight_answer": "B",
            "tight_support": 0.65,
            "tight_view_sha256": "1" * 64,
            "context_answer": "B",
            "context_support": 0.80,
            "context_view_sha256": "2" * 64,
            "minimum_final_support": 0.60,
            "minimum_support_gain": 0.10,
            "maximum_support_drop": 0.0,
            "minimum_conflict_margin": 1.0,
            "minimum_uncontested_support": 1.0,
        }
        arguments.update(overrides)
        return confirm_split_branch(**arguments)

    def test_two_distinct_agreeing_views_confirm_a_positive_trajectory(self):
        result = self.confirmation()
        self.assertTrue(result.confirmed)
        self.assertEqual(result.canonical_answer, "B")
        self.assertEqual(result.reason, "confirmed_two_view_trajectory")
        self.assertEqual(result.trajectory_s, 3)
        self.assertAlmostEqual(result.support_gain, 0.35)
        self.assertAlmostEqual(result.selection_score, 0.65)

    def test_scalar_confirmation_fails_closed_for_each_missing_condition(self):
        cases = (
            ({"context_answer": "C"}, "view_answer_disagreement"),
            ({"context_view_sha256": "1" * 64}, "duplicate_render"),
            ({"tight_support": 0.80, "context_support": 0.20}, "decreasing_trajectory"),
            ({"tight_support": 0.40, "context_support": 0.50}, "insufficient_support"),
            ({"p0_support": 0.70}, "insufficient_support_gain"),
            ({"tight_answer": None, "context_answer": None}, "invalid_answer"),
            ({
                "conflict_answer": "A",
                "conflict_selection_score": 0.65,
            }, "equally_strong_p0_conflict"),
        )
        for changes, reason in cases:
            with self.subTest(changes=changes):
                result = self.confirmation(**changes)
                self.assertFalse(result.confirmed)
                self.assertEqual(result.canonical_answer, "A")
                self.assertEqual(result.reason, reason)

    def test_hr_components_change_only_with_two_view_agreement(self):
        result = self.confirmation(
            p0_answer=("A", "B", "C", "D"),
            tight_answer=("A", "X", None, "Y"),
            context_answer=("A", "X", "Z", "Q"),
        )
        self.assertTrue(result.confirmed)
        self.assertEqual(result.canonical_answer, ("A", "X", "C", "D"))

    def test_small_support_drop_can_yield_to_dominant_counterevidence(self):
        result = self.confirmation(
            p0_support=0.93,
            tight_support=0.89,
            context_support=0.88,
            conflict_answer="A",
            conflict_selection_score=0.74,
            maximum_support_drop=0.06,
            minimum_conflict_margin=0.10,
        )
        self.assertTrue(result.confirmed)
        self.assertEqual(
            result.reason, "confirmed_two_view_dominant_counterevidence",
        )
        self.assertLess(result.trajectory_s, 0)

    def test_counterevidence_still_fails_without_a_clear_conflict_margin(self):
        result = self.confirmation(
            p0_support=0.93,
            tight_support=0.89,
            context_support=0.88,
            conflict_answer="A",
            conflict_selection_score=0.82,
            maximum_support_drop=0.06,
            minimum_conflict_margin=0.10,
        )
        self.assertFalse(result.confirmed)
        self.assertEqual(result.reason, "insufficient_conflict_margin")

    def test_uncontested_counterevidence_requires_very_high_support(self):
        result = self.confirmation(
            p0_support=0.99,
            tight_support=0.95,
            context_support=0.94,
            maximum_support_drop=0.06,
            minimum_uncontested_support=0.90,
        )
        self.assertTrue(result.confirmed)
        self.assertEqual(
            result.reason, "confirmed_two_view_high_support_counterevidence",
        )

    def test_uncontested_tiny_positive_gain_does_not_override_p0(self):
        result = self.confirmation(
            p0_support=0.93,
            tight_support=0.94,
            context_support=0.95,
            maximum_support_drop=0.06,
            minimum_uncontested_support=0.90,
        )
        self.assertFalse(result.confirmed)
        self.assertEqual(result.reason, "insufficient_support_gain")

    def test_invalid_thresholds_hashes_and_conflicts_are_rejected(self):
        cases = (
            {"minimum_final_support": 1.1},
            {"minimum_support_gain": -0.1},
            {"maximum_support_drop": -0.1},
            {"minimum_conflict_margin": 1.1},
            {"minimum_uncontested_support": -0.1},
            {"tight_view_sha256": "not-a-hash"},
            {"conflict_answer": "A"},
            {"conflict_selection_score": math.nan, "conflict_answer": "A"},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(
                (TypeError, ValueError)
            ):
                self.confirmation(**changes)


if __name__ == "__main__":
    unittest.main()
