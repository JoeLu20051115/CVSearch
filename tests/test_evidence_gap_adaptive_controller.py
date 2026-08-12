import copy
import math
import unittest

from cvsearch.evidence_gap.adaptive_controller import (
    BACKTRACK,
    EXPAND,
    SPLIT,
    STOP,
    ZOOM,
    EvidenceDemand,
    SupportObservation,
    fit_isotonic,
    infer_evidence_demand,
    select_adaptive_action,
)


class AdaptiveDemandTest(unittest.TestCase):
    def test_relation_and_detail_create_mixed_soft_prior(self):
        requirements = (
            {
                "kind": "target_detail",
                "target": "number",
                "requirements": ["presence", "visual_detail"],
            },
            {
                "kind": "relation_context",
                "targets": ["entrance", "woman"],
            },
        )

        demand = infer_evidence_demand(requirements)

        self.assertGreater(demand.detail, 0.0)
        self.assertGreater(demand.context, 0.0)
        self.assertGreater(demand.localization, 0.0)
        self.assertAlmostEqual(sum(demand.action_prior().values()), 1.0)
        self.assertEqual(set(demand.action_prior()), {ZOOM, EXPAND, SPLIT})

    def test_coverage_prefers_context_and_empty_plan_is_neutral(self):
        coverage = infer_evidence_demand((
            {"kind": "coverage", "requirement": "global_scope"},
        ))
        neutral = infer_evidence_demand(())

        self.assertGreater(coverage.context, coverage.detail)
        self.assertEqual(
            neutral.action_prior(),
            {ZOOM: 1 / 3, EXPAND: 1 / 3, SPLIT: 1 / 3},
        )

    def test_policy_input_rejects_evaluator_metadata_before_reading_values(self):
        class ExplodingValue:
            def __str__(self):
                raise AssertionError("forbidden value was inspected")

        for key in (
            "benchmark", "resolution", "category", "ordinal", "label",
            "ground_truth", "target_box", "correctness",
        ):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    infer_evidence_demand(({
                        "kind": "target_detail", key: ExplodingValue(),
                    },))

    def test_demand_and_prior_are_bounded_and_immutable(self):
        demand = EvidenceDemand(detail=1, context=0.25, localization=0)
        prior = demand.action_prior()
        prior[ZOOM] = 0.0

        self.assertGreater(demand.action_prior()[ZOOM], 0.0)
        for malformed in (True, -0.1, 1.1, math.nan, math.inf):
            with self.subTest(malformed=malformed):
                with self.assertRaises((TypeError, ValueError)):
                    EvidenceDemand(malformed, 0.0, 0.0)


class AdaptiveCalibrationTest(unittest.TestCase):
    def test_support_observation_raw_score_uses_both_consistencies(self):
        observation = SupportObservation(
            p_full=0.8,
            p_partial=0.1,
            p_none=0.1,
            support_consistency=0.25,
            answer_consistency=1.0,
            missing_reason="detail_unreadable",
            normalized_cost=0.2,
        )

        self.assertAlmostEqual(observation.raw_support, 0.4)
        self.assertAlmostEqual(observation.uncertainty, 0.6)

    def test_support_observation_rejects_bad_distribution_and_reason(self):
        common = dict(
            p_full=0.8, p_partial=0.1, p_none=0.1,
            support_consistency=1.0, answer_consistency=1.0,
            missing_reason="none", normalized_cost=0.0,
        )
        for update in (
            {"p_full": 0.7},
            {"p_none": math.nan},
            {"answer_consistency": 1.1},
            {"missing_reason": "benchmark_specific"},
            {"normalized_cost": True},
        ):
            with self.subTest(update=update):
                with self.assertRaises((TypeError, ValueError)):
                    SupportObservation(**{**common, **update})

    def test_isotonic_fit_is_monotone_bounded_and_deterministic(self):
        samples = ((0.1, 0), (0.2, 1), (0.3, 0), (0.9, 1))

        calibrator = fit_isotonic(samples)
        reversed_calibrator = fit_isotonic(tuple(reversed(samples)))
        values = [calibrator.predict(value) for value in (0.0, 0.2, 0.3, 1.0)]

        self.assertEqual(values, sorted(values))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertEqual(calibrator, reversed_calibrator)
        self.assertEqual(calibrator.predict(0.2), 0.5)
        self.assertEqual(calibrator.predict(0.9), 1.0)

    def test_isotonic_fit_rejects_empty_nonbinary_and_nonfinite_samples(self):
        for samples in (
            (),
            ((0.1, 2),),
            ((math.nan, 1),),
            ((True, 1),),
            ((0.1, True),),
        ):
            with self.subTest(samples=samples):
                with self.assertRaises((TypeError, ValueError)):
                    fit_isotonic(samples)


class AdaptiveActionPolicyTest(unittest.TestCase):
    @staticmethod
    def observation(reason="detail_unreadable", p_full=0.2, cost=0.0):
        remaining = 1.0 - p_full
        return SupportObservation(
            p_full=p_full,
            p_partial=remaining,
            p_none=0.0,
            support_consistency=1.0,
            answer_consistency=1.0,
            missing_reason=reason,
            normalized_cost=cost,
        )

    def test_observed_missing_reason_overrides_soft_prior(self):
        decision = select_adaptive_action(
            EvidenceDemand(detail=1.0, context=0.0, localization=0.0),
            self.observation(reason="context_missing"),
            available=(ZOOM, EXPAND),
            trajectory=(0.2,),
            has_unvisited_branch=True,
        )

        self.assertEqual(decision.action, EXPAND)
        self.assertEqual(decision.reason, "observed_context_missing")

    def test_sufficient_stable_evidence_stops_even_with_strong_prior(self):
        decision = select_adaptive_action(
            EvidenceDemand(detail=1.0, context=0.0, localization=0.0),
            self.observation(reason="none", p_full=0.95),
            available=(ZOOM, STOP),
            trajectory=(0.82, 0.90, 0.95),
            has_unvisited_branch=True,
            calibrated_support=0.92,
        )

        self.assertEqual(decision.action, STOP)
        self.assertEqual(decision.reason, "calibrated_support_sufficient")

    def test_two_stalled_gains_backtrack_without_deleting_patch(self):
        decision = select_adaptive_action(
            EvidenceDemand(0.3, 0.3, 0.4),
            self.observation(reason="conflict"),
            available=(ZOOM, EXPAND, BACKTRACK),
            trajectory=(0.40, 0.41, 0.40),
            has_unvisited_branch=True,
        )

        self.assertEqual(decision.action, BACKTRACK)
        self.assertEqual(decision.reason, "support_trajectory_stalled")
        self.assertNotIn("delete", decision.to_dict())

    def test_location_ambiguity_uses_split_only_when_available(self):
        split = select_adaptive_action(
            EvidenceDemand(0.1, 0.1, 0.8),
            self.observation(reason="location_ambiguous"),
            available=(ZOOM, SPLIT), trajectory=(0.2,),
            has_unvisited_branch=False,
        )
        fallback = select_adaptive_action(
            EvidenceDemand(0.1, 0.1, 0.8),
            self.observation(reason="location_ambiguous"),
            available=(ZOOM,), trajectory=(0.2,),
            has_unvisited_branch=False,
        )

        self.assertEqual(split.action, SPLIT)
        self.assertEqual(fallback.action, ZOOM)

    def test_prior_weight_decays_and_cost_breaks_remaining_ties(self):
        demand = EvidenceDemand(0.5, 0.5, 0.0)
        observation = self.observation(reason="target_missing")
        first = select_adaptive_action(
            demand, observation, available=(ZOOM, EXPAND), trajectory=(),
            has_unvisited_branch=False, step=0,
            action_costs={ZOOM: 0.5, EXPAND: 0.1},
        )
        later = select_adaptive_action(
            demand, observation, available=(ZOOM, EXPAND), trajectory=(0.1,),
            has_unvisited_branch=False, step=3,
            action_costs={ZOOM: 0.5, EXPAND: 0.1},
        )

        self.assertEqual(first.action, EXPAND)
        self.assertEqual(later.action, EXPAND)
        self.assertGreater(first.prior_weight, later.prior_weight)

    def test_selection_never_mutates_inputs(self):
        costs = {ZOOM: 0.2, EXPAND: 0.3}
        before = copy.deepcopy(costs)
        select_adaptive_action(
            EvidenceDemand(0.5, 0.5, 0.0), self.observation(),
            available=(ZOOM, EXPAND), trajectory=(0.1,),
            has_unvisited_branch=False, action_costs=costs,
        )
        self.assertEqual(costs, before)


if __name__ == "__main__":
    unittest.main()
