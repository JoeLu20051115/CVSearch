import math
import copy
import unittest

from cvsearch.eval.replay_uncertainty_support import (
    PROFILES,
    AdvantageFeatures,
    UnifiedPolicy,
    UtilityIsotonicCalibrator,
    candidate_snapshots,
    fit_utility_isotonic,
    raw_advantage,
    replay_uncertainty_support,
)
from tests.test_replay_split_search import calibration, make_hr, rescue_rows


class UtilityPrimitiveTests(unittest.TestCase):
    def test_pava_accepts_continuous_targets_and_is_monotone(self):
        fitted = fit_utility_isotonic(
            ((0.1, 0.75), (0.2, 0.25), (0.3, 1.0)),
        )

        predictions = [fitted.predict(value) for value in (0.1, 0.2, 0.3)]

        self.assertEqual(predictions, sorted(predictions))
        self.assertEqual(predictions, [0.5, 0.5, 1.0])

    def test_equal_scores_are_averaged_before_pava(self):
        fitted = fit_utility_isotonic(
            ((0.1, 0.25), (0.1, 0.75), (0.2, 1.0)),
        )

        self.assertEqual(fitted.upper_bounds, (0.1, 0.2))
        self.assertEqual(fitted.utilities, (0.5, 1.0))

    def test_pava_compacts_adjacent_equal_utility_blocks(self):
        fitted = fit_utility_isotonic(
            ((0.1, 0.75), (0.2, 0.25), (0.3, 1.0)),
        )

        self.assertEqual(fitted.upper_bounds, (0.2, 0.3))
        self.assertEqual(fitted.utilities, (0.5, 1.0))

    def test_raw_advantage_is_the_only_weighted_numeric_score(self):
        features = AdvantageFeatures(0.8, 0.5, 0.7, 0.6, 0.4)

        self.assertAlmostEqual(
            raw_advantage(features, PROFILES["balanced"]),
            0.6,
        )

    def test_invalid_feature_or_profile_is_rejected(self):
        for bad in (-0.1, 1.1, math.inf, math.nan, True):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    AdvantageFeatures(bad, 0.5, 0.5, 0.5, 0.5)

        with self.assertRaises(ValueError):
            raw_advantage(
                AdvantageFeatures(0.5, 0.5, 0.5, 0.5, 0.5),
                (1.0, 0.0, 0.0, 0.0, 0.1),
            )

    def test_calibrator_rejects_bad_samples_and_payload(self):
        for samples in (
            (),
            ((math.nan, 0.5),),
            ((0.1, -0.1),),
            ((0.1, 1.1),),
            ((True, 0.5),),
        ):
            with self.subTest(samples=samples):
                with self.assertRaises((TypeError, ValueError)):
                    fit_utility_isotonic(samples)

        for calibrator in (
            UtilityIsotonicCalibrator((), ()),
            UtilityIsotonicCalibrator((0.2, 0.1), (0.4, 0.5)),
            UtilityIsotonicCalibrator((0.1,), (1.1,)),
        ):
            with self.subTest(calibrator=calibrator):
                with self.assertRaises(ValueError):
                    calibrator.predict(0.1)

    def test_calibrator_round_trip_is_deterministic(self):
        fitted = fit_utility_isotonic(((0.2, 0.7), (0.1, 0.3)))

        self.assertEqual(
            UtilityIsotonicCalibrator.from_dict(fitted.to_dict()),
            fitted,
        )
        self.assertEqual(fitted.to_dict(), fitted.to_dict())


def policy(*, utility=1.0, threshold=0.0, raw_support_floor=0.2):
    return UnifiedPolicy(
        profile="balanced",
        threshold=threshold,
        raw_support_floor=raw_support_floor,
        utility_calibrator=UtilityIsotonicCalibrator((1.0,), (utility,)),
    )


def audit_value(row):
    return row["method_trace"]["steps"][0]["split_search_audit"]


class UnifiedStateMachineTests(unittest.TestCase):
    def test_two_distinct_agreeing_views_replace_p0(self):
        phase1, split = rescue_rows()
        branch = audit_value(split)["branches"][0]
        for role, support in (("tight_view", 0.9), ("context_view", 0.8)):
            branch[role]["answer"] = "B"
            branch[role]["raw_support"] = support

        decision = replay_uncertainty_support(
            phase1, split, calibration(), policy(),
        )

        self.assertEqual(decision["selected_output"], "B")
        self.assertEqual(decision["selected_source"], "SPLIT")
        self.assertEqual(decision["selected_branch"], 0)
        self.assertEqual(
            [step["action"] for step in decision["transitions"]],
            ["OBSERVE", "CONTINUE", "REPLACE"],
        )
        self.assertEqual(decision["observations"], 2)

    def test_conflict_backtracks_then_later_branch_replaces(self):
        phase1, split = rescue_rows()
        branches = audit_value(split)["branches"]
        for role, answer in (("tight_view", "B"), ("context_view", "C")):
            branches[0][role]["answer"] = answer
            branches[0][role]["raw_support"] = 0.9
        for role in ("tight_view", "context_view"):
            branches[1][role]["answer"] = "D"
            branches[1][role]["raw_support"] = 0.9

        decision = replay_uncertainty_support(
            phase1, split, calibration(), policy(),
        )

        self.assertEqual(decision["selected_output"], "D")
        self.assertEqual(decision["selected_branch"], 1)
        self.assertIn("BACKTRACK", [
            step["action"] for step in decision["transitions"]
        ])
        self.assertTrue(decision["used_backtrack"])

    def test_backtrack_keeps_prior_views_for_cross_branch_agreement(self):
        phase1, split = rescue_rows()
        branches = audit_value(split)["branches"]
        for branch in branches:
            for role in ("tight_view", "medium_view", "context_view"):
                if role in branch:
                    branch[role]["answer"] = "A"
                    branch[role]["raw_support"] = 0.9
        branches[0]["tight_view"]["answer"] = "B"
        branches[0]["context_view"]["answer"] = "C"
        branches[1]["tight_view"]["answer"] = "B"
        branches[1]["context_view"]["answer"] = "D"

        decision = replay_uncertainty_support(
            phase1, split, calibration(), policy(),
        )

        self.assertEqual(decision["selected_output"], "B")
        self.assertEqual(decision["selected_branch"], 1)
        self.assertEqual(decision["observations"], 3)
        self.assertEqual(
            decision["transitions"][-1]["features"]["agreement"],
            2 / 3,
        )

    def test_low_optimistic_utility_stops_after_one_view_per_branch(self):
        phase1, split = rescue_rows()

        decision = replay_uncertainty_support(
            phase1, split, calibration(), policy(utility=0.0),
        )

        self.assertEqual(decision["selected_output"], phase1["output"])
        self.assertEqual(decision["selected_source"], "P0")
        self.assertEqual(decision["reason"], "all_branches_exhausted")
        self.assertEqual(decision["observations"], 6)
        self.assertEqual(decision["transitions"][-1]["action"], "STOP_P0")

    def test_duplicate_render_hash_falls_back_to_exact_stage2_output(self):
        phase1, split = rescue_rows()
        branches = audit_value(split)["branches"]
        branches[1]["tight_view"]["render_sha256"] = branches[0][
            "tight_view"
        ]["render_sha256"]
        expected = copy.deepcopy(phase1["output"])

        decision = replay_uncertainty_support(
            phase1, split, calibration(), policy(),
        )

        self.assertEqual(decision["selected_output"], expected)
        self.assertEqual(decision["stage2_selected_output"], expected)
        self.assertEqual(decision["selected_source"], "P0")
        self.assertEqual(decision["transitions"][-1]["action"], "FALLBACK_P0")

    def test_unparseable_branch_backtracks_without_poisoning_later_branch(self):
        phase1, split = rescue_rows()
        branches = audit_value(split)["branches"]
        branches[0]["tight_view"]["answer"] = "not an option"
        for role in ("tight_view", "context_view"):
            branches[1][role]["answer"] = "B"
            branches[1][role]["raw_support"] = 0.9

        decision = replay_uncertainty_support(
            phase1, split, calibration(), policy(),
        )

        self.assertEqual(decision["selected_output"], "B")
        self.assertEqual(decision["selected_branch"], 1)
        self.assertTrue(any(
            step["reason"] == "unparseable_observation"
            for step in decision["transitions"]
        ))

    def test_raw_support_floor_is_a_structural_check(self):
        phase1, split = rescue_rows()
        branches = audit_value(split)["branches"]
        for branch in branches:
            for role in ("tight_view", "medium_view", "context_view"):
                if role in branch:
                    branch[role]["answer"] = "A"
        branch = branches[0]
        for role in ("tight_view", "context_view"):
            branch[role]["answer"] = "B"
            branch[role]["raw_support"] = 0.1

        decision = replay_uncertainty_support(
            phase1, split, calibration(), policy(raw_support_floor=0.2),
        )

        self.assertEqual(decision["selected_source"], "P0")
        eligible = [
            item for item in candidate_snapshots(
                phase1, split, calibration(), policy(raw_support_floor=0.2),
            )
            if item.branch_index == 0 and len(item.revealed_roles) == 2
        ]
        self.assertEqual(len(eligible), 1)
        self.assertFalse(eligible[0].structurally_eligible)

    def test_hr_replacement_preserves_atomic_list_projection(self):
        phase1, split = rescue_rows()
        make_hr(phase1, split)
        branch = audit_value(split)["branches"][0]
        for role in ("tight_view", "context_view"):
            branch[role]["answer"] = ["B"] * 4
            branch[role]["raw_support"] = 0.9

        decision = replay_uncertainty_support(
            phase1, split, calibration(), policy(),
        )

        self.assertEqual(decision["stage2_selected_output"], ["A"] * 4)
        self.assertEqual(decision["selected_output"], ["B"] * 4)

    def test_trace_is_deterministic_label_blind_and_visits_fixed_order(self):
        phase1, split = rescue_rows()
        first = replay_uncertainty_support(
            phase1, split, calibration(), policy(utility=0.0),
        )
        phase1["answer"] = "B"
        phase1["category"] = "poison"
        split["answer"] = "C"
        split["category"] = "different"
        second = replay_uncertainty_support(
            phase1, split, calibration(), policy(utility=0.0),
        )

        self.assertEqual(first, second)
        self.assertEqual(
            [step["branch"] for step in first["transitions"] if step["action"] == "BACKTRACK"],
            list(range(6)),
        )
        serialized = repr(first["transitions"]).lower()
        self.assertNotIn("correct", serialized)
        self.assertNotIn("category", serialized)


if __name__ == "__main__":
    unittest.main()
