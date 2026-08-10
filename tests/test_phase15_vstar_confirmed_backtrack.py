from copy import deepcopy
import unittest

from cvsearch.eval.phase15_vstar_confirmed_backtrack import (
    select_vstar_confirmed_backtrack,
)


OPTIONS = ["red", "green", "blue"]


def p0(output):
    return {
        "action": "P0",
        "output": output,
        "p0_stability": {"confidence": 0.2},
    }


def b5(action="LAZY", output=2):
    return {
        "action": action,
        "status": (
            "selected_generated_query_lazy_search"
            if action == "LAZY" else "retained_p0"
        ),
        "output": output,
        "confidence": 0.5000000000000001 if action == "LAZY" else None,
        "confidence_gain": 0.3000000000000001 if action == "LAZY" else None,
        "path_consensus": 2 / 3 if action == "LAZY" else None,
    }


def observation(winner):
    losses = [3.0, 3.0, 3.0]
    losses[winner] = 1.0
    return {"winner": winner, "losses": losses}


class VstarConfirmedBacktrackTests(unittest.TestCase):
    def test_p0_child_backtrack_pattern_selects_frozen_b5_candidate(self):
        decision = select_vstar_confirmed_backtrack(
            p0(1), OPTIONS,
            [observation(1), observation(2), observation(2)], b5(),
        )
        self.assertEqual(decision.action, "VSTAR_BACKTRACK")
        self.assertEqual(decision.status, "selected_confirmed_backtrack")
        self.assertEqual(decision.output, 2)

    def test_every_non_recovery_trajectory_retains_exact_p0(self):
        trajectories = (
            [1, 2],
            [2, 2],
            [2, 1, 2],
            [1, 2, 0],
            [0, 2, 2],
        )
        for winners in trajectories:
            with self.subTest(winners=winners):
                decision = select_vstar_confirmed_backtrack(
                    p0(1), OPTIONS, [observation(x) for x in winners], b5(),
                )
                self.assertEqual(decision.action, "P0")
                self.assertEqual(decision.output, 1)

    def test_unselected_or_inconsistent_b5_and_malformed_view_fail_closed(self):
        views = [observation(1), observation(2), observation(2)]
        self.assertEqual(
            select_vstar_confirmed_backtrack(
                p0(1), OPTIONS, views, b5("P0", 1),
            ).action,
            "P0",
        )
        self.assertEqual(
            select_vstar_confirmed_backtrack(
                p0(1), OPTIONS, views, b5(output=0),
            ).action,
            "P0",
        )
        malformed = deepcopy(views)
        malformed[-1]["winner"] = 0
        self.assertEqual(
            select_vstar_confirmed_backtrack(
                p0(1), OPTIONS, malformed, b5(),
            ).action,
            "P0",
        )

    def test_invalid_frozen_rule_metadata_fails_closed(self):
        views = [observation(1), observation(2), observation(2)]
        invalid = (
            {"confidence": None},
            {"confidence": 0.049},
            {"confidence": float("nan")},
            {"confidence_gain": None},
            {"confidence_gain": -0.1001},
            {"path_consensus": None},
            {"path_consensus": 1.0},
        )
        for replacement in invalid:
            with self.subTest(replacement=replacement):
                frozen = b5()
                frozen.update(replacement)
                self.assertEqual(
                    select_vstar_confirmed_backtrack(
                        p0(1), OPTIONS, views, frozen,
                    ).action,
                    "P0",
                )

    def test_confidence_and_gain_must_match_replayed_projection_and_p0(self):
        views = [observation(1), observation(2), observation(2)]
        inconsistent = b5()
        inconsistent["confidence_gain"] = -0.1
        self.assertEqual(
            select_vstar_confirmed_backtrack(
                p0(1), OPTIONS, views, inconsistent,
            ).action,
            "P0",
        )
        wrong_confidence = b5()
        wrong_confidence["confidence"] = 0.6
        wrong_confidence["confidence_gain"] = 0.4
        self.assertEqual(
            select_vstar_confirmed_backtrack(
                p0(1), OPTIONS, views, wrong_confidence,
            ).action,
            "P0",
        )

    def test_selection_is_option_permutation_equivariant_and_immutable(self):
        state = p0(1)
        views = [observation(1), observation(2), observation(2)]
        frozen = b5()
        before = deepcopy((state, OPTIONS, views, frozen))
        original = select_vstar_confirmed_backtrack(
            state, OPTIONS, views, frozen,
        )
        permutation = {0: 2, 1: 0, 2: 1}
        permuted_options = [OPTIONS[1], OPTIONS[2], OPTIONS[0]]
        permuted_views = []
        for view in views:
            losses = [0.0] * 3
            for old, value in enumerate(view["losses"]):
                losses[permutation[old]] = value
            permuted_views.append({
                "winner": permutation[view["winner"]], "losses": losses,
            })
        permuted = select_vstar_confirmed_backtrack(
            p0(permutation[1]), permuted_options, permuted_views,
            b5(output=permutation[2]),
        )
        self.assertEqual(permuted.output, permutation[original.output])
        self.assertEqual((state, OPTIONS, views, frozen), before)


if __name__ == "__main__":
    unittest.main()
