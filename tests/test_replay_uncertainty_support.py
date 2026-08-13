import math
import copy
import unittest

from cvsearch.eval.replay_uncertainty_support import (
    PROFILES,
    AggregateEvidenceFeatures,
    AdvantageFeatures,
    RiskCalibrator,
    RiskLogisticHead,
    RiskLinearHead,
    RiskRegion,
    UnifiedPolicy,
    UtilityIsotonicCalibrator,
    candidate_snapshots,
    fit_utility_isotonic,
    raw_advantage,
    replay_uncertainty_support,
    sanitize_replay_row,
)
from cvsearch.eval.replay_adaptive_search import replay_adaptive_search
from tests.test_replay_adaptive_search import action_step
from tests.test_replay_split_search import calibration, make_hr, rescue_rows


class UtilityPrimitiveTests(unittest.TestCase):
    def test_logistic_head_round_trip_scores_aggregate_evidence(self):
        evidence = AggregateEvidenceFeatures(*([0.5] * 14))
        head = RiskLogisticHead(
            intercept=0.0,
            coefficients=(2.0,) + (0.0,) * 13,
            means=(0.0,) * 14,
            scales=(1.0,) * 14,
        )

        restored = RiskLogisticHead.from_dict(head.to_dict())

        self.assertEqual(restored, head)
        self.assertAlmostEqual(restored.predict(evidence), 0.7310585786)

    def test_risk_calibrator_separates_benefit_harm_and_boundary(self):
        features = AdvantageFeatures(0.8, 0.5, 0.7, 0.6, 0.4)
        risk = RiskCalibrator(
            benefit_head=RiskLinearHead((0.8,) + (0.0,) * 17),
            harm_head=RiskLinearHead((0.1,) + (0.0,) * 17),
            risk_penalty=2.0,
            decision_boundary=0.25,
        )

        benefit, harm, margin = risk.predict(features)

        self.assertEqual((benefit, harm), (0.8, 0.1))
        self.assertAlmostEqual(margin, 0.35)

    def test_piecewise_risk_region_is_a_generic_calibration_leaf(self):
        low_scale = RiskRegion(
            lower_bounds=(0.0, 0.0, 0.005, 0.005, 0.15),
            upper_bounds=(0.0, 0.25, 0.025, 0.02, 1.0),
            expected_benefit=1.0,
            corruption_risk=0.0,
            margin=1.0,
        )
        zero = RiskLinearHead((0.0,) * 18)
        risk = RiskCalibrator(
            zero, zero, risk_penalty=1.0, decision_boundary=0.5,
            regions=(low_scale,),
        )

        self.assertEqual(
            risk.predict(AdvantageFeatures(0.0, 0.2, 0.018, 0.009, 0.17)),
            (1.0, 0.0, 1.0),
        )
        self.assertEqual(
            risk.predict(AdvantageFeatures(0.0, 0.5, 0.018, 0.009, 0.17)),
            (0.0, 0.0, -0.5),
        )

    def test_risk_policy_uses_exact_then_backbone_then_global_stratum(self):
        def constant(value):
            return RiskCalibrator(
                RiskLinearHead((value,) + (0.0,) * 17),
                RiskLinearHead((0.0,) * 18),
                risk_penalty=1.0,
                decision_boundary=0.0,
            )

        configured = UnifiedPolicy(
            profile="balanced",
            threshold=0.0,
            raw_support_floor=0.0,
            utility_calibrator=UtilityIsotonicCalibrator((1.0,), (0.5,)),
            risk_calibrators=(
                ("*/*", constant(0.1)),
                ("qwen/*", constant(0.2)),
                ("qwen/option_list", constant(0.3)),
            ),
        )
        features = AdvantageFeatures(0.5, 0.5, 0.5, 0.5, 0.5)

        self.assertEqual(
            configured.predict_risk(features, "qwen", "option_list")[2],
            0.3,
        )
        self.assertEqual(
            configured.predict_risk(features, "qwen", "option_single")[2],
            0.2,
        )
        self.assertEqual(
            configured.predict_risk(features, "internvl", "logits_match")[2],
            0.1,
        )

    def test_replay_row_projection_excludes_evaluator_fields(self):
        row = {
            "_eg_ordinal": 1,
            "answer_type": "option_single",
            "options": "A. x\nB. y",
            "output": "A",
            "method_trace": {"steps": []},
            "answer": "B",
            "category": "poison",
            "target_box": [1, 2, 3, 4],
        }

        projected = sanitize_replay_row(row)

        self.assertEqual(
            set(projected),
            {"_eg_ordinal", "answer_type", "options", "output", "method_trace"},
        )
        self.assertNotIn("answer", projected)
        self.assertNotIn("category", projected)
        self.assertNotIn("target_box", projected)

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


def confirmation_policy():
    return UnifiedPolicy(
        profile="balanced",
        threshold=0.25,
        raw_support_floor=0.0,
        utility_calibrator=UtilityIsotonicCalibrator(
            (0.75, 1.0), (0.5, 0.7),
        ),
    )


def audit_value(row):
    return next(
        step["split_search_audit"]
        for step in row["method_trace"]["steps"]
        if step.get("action") == "SPLIT"
    )


class UnifiedStateMachineTests(unittest.TestCase):
    @staticmethod
    def _aggregate_risk_policy(*, minimum_views=2, maximum_observations=14):
        positive = RiskLogisticHead(
            intercept=10.0,
            coefficients=(0.0,) * 14,
            means=(0.0,) * 14,
            scales=(1.0,) * 14,
        )
        negative = RiskLogisticHead(
            intercept=-10.0,
            coefficients=(0.0,) * 14,
            means=(0.0,) * 14,
            scales=(1.0,) * 14,
        )
        zero = RiskLinearHead((0.0,) * 18)
        return UnifiedPolicy(
            profile="balanced",
            threshold=0.0,
            raw_support_floor=0.0,
            utility_calibrator=UtilityIsotonicCalibrator((1.0,), (0.5,)),
            risk_calibrators=(("*/*", RiskCalibrator(
                zero, zero, 1.0, 0.0,
                evidence_benefit_head=positive,
                evidence_harm_head=negative,
                minimum_observations=1,
                maximum_observations=maximum_observations,
                minimum_agreeing_views=minimum_views,
            )),),
        )

    def test_aggregate_risk_requires_two_independent_agreeing_views(self):
        phase1, split = rescue_rows()
        branch = audit_value(split)["branches"][0]
        for role in ("tight_view", "context_view"):
            branch[role]["answer"] = "B"
            branch[role]["raw_support"] = 0.9

        decision = replay_uncertainty_support(
            phase1, split, calibration(),
            self._aggregate_risk_policy(), backbone="qwen",
        )

        self.assertEqual(decision["selected_source"], "SPLIT")
        self.assertEqual(decision["observations"], 2)
        replacement = decision["transitions"][-1]
        self.assertEqual(replacement["action"], "REPLACE")
        self.assertEqual(replacement["evidence_features"]["agreeing_fraction"], 2 / 14)

    def test_legacy_four_branch_budget_replays_but_five_branches_fail_closed(self):
        phase1, split = rescue_rows()
        audit = audit_value(split)
        audit["branches"] = audit["branches"][:4]
        for role in ("tight_view", "context_view"):
            audit["branches"][0][role]["answer"] = "B"
            audit["branches"][0][role]["raw_support"] = 0.9

        snapshots = candidate_snapshots(
            phase1, split, calibration(), policy(raw_support_floor=0.0),
        )

        self.assertTrue(snapshots)
        malformed = copy.deepcopy(split)
        malformed_audit = audit_value(malformed)
        malformed_audit["branches"].append(
            copy.deepcopy(malformed_audit["branches"][-1]),
        )
        malformed_audit["branches"][-1]["visit_index"] = 4
        decision = replay_uncertainty_support(
            phase1, malformed, calibration(), policy(),
        )
        self.assertEqual(decision["selected_source"], "P0")
        self.assertIn("four or six", decision["failure_detail"])

    def test_global_observation_budget_stops_at_exact_p0(self):
        phase1, split = rescue_rows()

        decision = replay_uncertainty_support(
            phase1, split, calibration(),
            self._aggregate_risk_policy(
                minimum_views=3, maximum_observations=2,
            ),
            backbone="qwen",
        )

        self.assertEqual(decision["selected_source"], "P0")
        self.assertEqual(decision["selected_output"], phase1["output"])
        self.assertEqual(decision["observations"], 2)
        self.assertEqual(decision["reason"], "observation_budget_exhausted")

    def test_candidate_snapshots_stop_branch_after_unparseable_view(self):
        phase1, split = rescue_rows()
        branches = audit_value(split)["branches"]
        branches[0]["tight_view"]["answer"] = "not an option"
        branches[0]["context_view"]["answer"] = "B"
        branches[0]["context_view"]["raw_support"] = 0.9

        snapshots = candidate_snapshots(
            phase1, split, calibration(), policy(),
        )

        self.assertFalse(any(
            snapshot.branch_index == 0 for snapshot in snapshots
        ))

    def test_snapshot_observations_match_runtime_replacement_cost(self):
        phase1, split = rescue_rows()
        branches = audit_value(split)["branches"]
        branches[0]["tight_view"]["answer"] = "not an option"
        for role in ("tight_view", "context_view"):
            branches[1][role]["answer"] = "B"
            branches[1][role]["raw_support"] = 0.9

        snapshots = candidate_snapshots(
            phase1, split, calibration(), policy(),
        )
        decision = replay_uncertainty_support(
            phase1, split, calibration(), policy(),
        )
        replacement = next(
            transition for transition in decision["transitions"]
            if transition["action"] == "REPLACE"
        )
        selected = next(
            snapshot for snapshot in snapshots
            if snapshot.branch_index == decision["selected_branch"]
            and snapshot.revealed_roles == tuple(
                replacement["revealed_roles"],
            )
            and snapshot.output == decision["selected_output"]
        )

        self.assertEqual(selected.observations, decision["observations"])

    def test_each_changed_answer_gets_an_independent_checkpoint_snapshot(self):
        phase1, split = rescue_rows()
        branches = audit_value(split)["branches"]
        branches[0]["tight_view"]["answer"] = "B"
        branches[0]["context_view"]["answer"] = "C"
        branches[0]["tight_view"]["raw_support"] = 0.8
        branches[0]["context_view"]["raw_support"] = 0.9

        snapshots = [
            snapshot
            for snapshot in candidate_snapshots(
                phase1, split, calibration(), policy(raw_support_floor=0.0),
            )
            if snapshot.branch_index == 0
            and snapshot.revealed_roles == ("tight", "context")
        ]

        self.assertEqual(
            {snapshot.canonical_answer for snapshot in snapshots},
            {"B", "C"},
        )
        self.assertEqual(len(snapshots), 2)

    def test_single_low_raw_support_view_is_soft_evidence_not_a_veto(self):
        phase1, split = rescue_rows()
        branches = audit_value(split)["branches"]
        for branch in branches:
            for role in ("tight_view", "medium_view", "context_view"):
                if role in branch:
                    branch[role]["answer"] = "A"
        branches[0]["tight_view"]["answer"] = "B"
        branches[0]["tight_view"]["raw_support"] = 0.001

        first = candidate_snapshots(
            phase1, split, calibration(), policy(raw_support_floor=0.2),
        )[0]

        self.assertEqual(first.canonical_answer, "B")
        self.assertEqual(first.features.agreement, 0.5)
        self.assertTrue(first.structurally_eligible)

    def test_two_distinct_agreeing_views_replace_p0(self):
        phase1, split = rescue_rows()
        branch = audit_value(split)["branches"][0]
        for role, support in (("tight_view", 0.9), ("context_view", 0.8)):
            branch[role]["answer"] = "B"
            branch[role]["raw_support"] = support

        decision = replay_uncertainty_support(
            phase1, split, calibration(), confirmation_policy(),
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
            phase1, split, calibration(), confirmation_policy(),
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
            phase1, split, calibration(), confirmation_policy(),
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
        self.assertIn("globally distinct", decision["failure_detail"])
        self.assertIn(
            "globally distinct", decision["transitions"][-1]["reason"],
        )

    def test_invalid_split_fallback_preserves_reconstructed_stage2_selection(self):
        phase1, split = rescue_rows()
        zoom = action_step(
            "ZOOM", current=0.1, candidate=0.9, output="B",
        )
        expand = action_step(
            "EXPAND", current=0.1, candidate=0.8, output="B",
        )
        split["method_trace"]["steps"][:0] = [zoom, expand]
        stage2_row = copy.deepcopy(split)
        stage2_row["method_trace"]["steps"] = [zoom, expand]
        stage2 = replay_adaptive_search(stage2_row, stage2_row, calibration())
        self.assertEqual(stage2["selected_output"], "B")
        split["method_trace"]["query_plan"] = None
        audit_value(split)["branches"][0]["tight_view"][
            "render_sha256"
        ] = "bad"

        decision = replay_uncertainty_support(
            stage2_row, split, calibration(), policy(),
        )

        self.assertEqual(decision["stage2_selected_output"], "B")
        self.assertEqual(decision["selected_output"], "B")
        self.assertEqual(decision["stage2_selected_source"], "ZOOM")
        self.assertEqual(decision["selected_source"], "P0")

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

    def test_raw_support_remains_a_continuous_feature_not_a_structural_check(self):
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

        self.assertEqual(decision["selected_source"], "SPLIT")
        eligible = [
            item for item in candidate_snapshots(
                phase1, split, calibration(), policy(raw_support_floor=0.2),
            )
            if item.branch_index == 0 and len(item.revealed_roles) == 2
        ]
        self.assertEqual(len(eligible), 1)
        self.assertTrue(eligible[0].structurally_eligible)

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
