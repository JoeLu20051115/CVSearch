import unittest

from cvsearch.eval.freeze_uncertainty_support import PolicyMetrics
from cvsearch.eval.candidate_free_verifier_selector import (
    CandidateFreeVerifierEvidence,
)
from cvsearch.eval.robust_transfer_selector import (
    AcceptanceCriteria,
    evaluate_acceptance,
    nested_partition_validation,
    robust_rank,
    verifier_evidence_key,
    select_shared_configuration,
)
from tests.test_freeze_uncertainty_support import record


BACKBONES = ("internvl", "qwen")
BENCHMARKS = ("hr_bench_4k", "hr_bench_8k", "treebench", "vstar")


def all_cells(value=0):
    return {
        f"{backbone}/{benchmark}": value
        for backbone in BACKBONES
        for benchmark in BENCHMARKS
    }


def metrics(
    *,
    net_gain=10,
    corrections=10,
    corruptions=0,
    observations=1120,
    backbone_deltas=None,
    cell_deltas=None,
):
    return PolicyMetrics(
        net_gain=net_gain,
        corrections=corrections,
        corruptions=corruptions,
        observations=observations,
        selections=corrections + corruptions,
        cell_deltas=tuple(sorted(
            (cell_deltas or all_cells()).items(),
        )),
        dataset_deltas=(),
        backbone_deltas=tuple(sorted(
            (backbone_deltas or {"qwen": 4, "internvl": 6}).items(),
        )),
    )


def development_partitions():
    return {
        "development": (
            record("dev-0", "qwen", helpful=True, ordinal=0),
            record("dev-0", "internvl", helpful=True, ordinal=0),
            record("dev-1", "qwen", helpful=False, ordinal=1),
            record("dev-1", "internvl", helpful=False, ordinal=1),
        ),
        "validation_v3": (
            record("val-0", "qwen", helpful=True, ordinal=2),
            record("val-0", "internvl", helpful=True, ordinal=2),
            record("val-1", "qwen", helpful=False, ordinal=3),
            record("val-1", "internvl", helpful=False, ordinal=3),
        ),
    }


class AcceptanceTests(unittest.TestCase):
    def test_default_contract_matches_declared_gate(self):
        criteria = AcceptanceCriteria()

        self.assertEqual(criteria.minimum_net_gain, 10)
        self.assertEqual(criteria.max_mean_observations, 12.8)
        self.assertEqual(criteria.preferred_mean_observations, 11.52)

    def test_gate_requires_positive_both_backbones_and_nonnegative_cells(self):
        cells = all_cells()
        cells["internvl/treebench"] = -1

        failures = evaluate_acceptance(metrics(
            net_gain=12,
            backbone_deltas={"qwen": 0, "internvl": 12},
            cell_deltas=cells,
        ), topics=112)

        self.assertIn("qwen backbone is not strictly positive", failures)
        self.assertIn("internvl/treebench cell is negative", failures)

    def test_gate_requires_net_gain_and_more_corrections_than_corruptions(self):
        failures = evaluate_acceptance(metrics(
            net_gain=9, corrections=2, corruptions=2,
        ), topics=112)

        self.assertIn("net gain is below 10", failures)
        self.assertIn("corrections do not exceed corruptions", failures)

    def test_gate_uses_actual_mean_observations(self):
        self.assertIn(
            "mean observations exceed 12.8",
            evaluate_acceptance(metrics(observations=1434), topics=112),
        )

    def test_complete_safe_metrics_pass(self):
        self.assertEqual(evaluate_acceptance(metrics(), topics=112), ())

    def test_gate_rejects_missing_declared_cell(self):
        cells = all_cells()
        cells.pop("qwen/vstar")

        self.assertIn(
            "required cells are incomplete",
            evaluate_acceptance(metrics(cell_deltas=cells), topics=112),
        )


class RankingTests(unittest.TestCase):
    def test_rank_prefers_larger_worst_backbone_before_pooled_gain(self):
        balanced = metrics(
            net_gain=10,
            backbone_deltas={"qwen": 4, "internvl": 6},
        )
        concentrated = metrics(
            net_gain=20,
            backbone_deltas={"qwen": 1, "internvl": 19},
        )

        self.assertLess(
            robust_rank(balanced, 112, 0),
            robust_rank(concentrated, 112, 1),
        )

    def test_rank_penalizes_concentration_when_worst_backbone_is_tied(self):
        less_concentrated = metrics(
            net_gain=4,
            backbone_deltas={"qwen": 0, "internvl": 4},
        )
        concentrated = metrics(
            net_gain=9,
            backbone_deltas={"qwen": 0, "internvl": 9},
        )

        self.assertLess(
            robust_rank(less_concentrated, 112, 1),
            robust_rank(concentrated, 112, 0),
        )

    def test_rank_prefers_zero_corruption_after_hard_gates(self):
        clean = metrics(corrections=10, corruptions=0)
        noisy = metrics(corrections=12, corruptions=1)

        self.assertLess(
            robust_rank(clean, 112, 1),
            robust_rank(noisy, 112, 0),
        )

    def test_rank_never_prefers_invalid_clean_policy_over_feasible_policy(self):
        invalid_clean = metrics(
            net_gain=20,
            backbone_deltas={"qwen": 0, "internvl": 20},
        )
        feasible_with_harm = metrics(
            corrections=12, corruptions=1,
            backbone_deltas={"qwen": 4, "internvl": 7},
        )

        self.assertLess(
            robust_rank(feasible_with_harm, 112, 1),
            robust_rank(invalid_clean, 112, 0),
        )

    def test_rank_uses_stable_grid_order_as_final_tie_break(self):
        value = metrics()

        self.assertLess(
            robust_rank(value, 112, 3),
            robust_rank(value, 112, 4),
        )


class NestedSelectionTests(unittest.TestCase):
    def test_verifier_aware_selection_requires_complete_bound_evidence(self):
        development = development_partitions()["development"]

        with self.assertRaises(ValueError):
            select_shared_configuration(
                development, verifier_evidence={},
            )

        unavailable = CandidateFreeVerifierEvidence(
            verifier_feasible=False,
            verifier_canonical=None,
            verifier_confidence=0.0,
            proposal_feasible=False,
            proposal_canonical=None,
            proposal_agreement=0.0,
            proposal_corrections=0,
            proposal_corruptions=0,
            observations=8,
        )
        result = select_shared_configuration(
            development,
            verifier_evidence={
                verifier_evidence_key(value): unavailable
                for value in development
            },
        )

        self.assertTrue(result.verifier_cascade)

    def test_outer_partition_never_enters_train_groups(self):
        result = nested_partition_validation(development_partitions())

        for fold in result.outer_folds:
            self.assertTrue(
                set(fold.train_groups).isdisjoint(fold.held_out_groups)
            )

    def test_penalty_and_boundary_are_shared_across_calibration_heads(self):
        development = development_partitions()["development"]

        result = select_shared_configuration(development)
        values = {
            (head.risk_penalty, head.decision_boundary)
            for _, head in result.refit_calibrators
        }

        self.assertEqual(len(values), 1)

    def test_v2_uses_one_global_aggregate_head_and_shared_safety_budget(self):
        result = select_shared_configuration(
            development_partitions()["development"],
        )

        self.assertEqual(
            [key for key, _ in result.refit_calibrators], ["*/*"],
        )
        calibrator = result.refit_calibrators[0][1]
        self.assertIsNotNone(calibrator.evidence_benefit_head)
        self.assertIsNotNone(calibrator.evidence_harm_head)
        self.assertEqual(calibrator.minimum_agreeing_views, 2)
        self.assertLessEqual(calibrator.maximum_observations, 14)

    def test_inner_validation_uses_four_disjoint_source_group_folds(self):
        records = tuple(
            record(
                f"group-{group}", backbone,
                helpful=group % 2 == 0,
                ordinal=group * 2 + (backbone == "internvl"),
            )
            for group in range(6)
            for backbone in BACKBONES
        )

        result = select_shared_configuration(records)

        self.assertEqual(len(result.folds), 4)
        for fold in result.folds:
            self.assertTrue(
                set(fold.held_out_groups).isdisjoint(fold.train_groups),
            )

    def test_nested_selection_is_byte_deterministic(self):
        partitions = development_partitions()

        first = nested_partition_validation(partitions).to_dict()
        second = nested_partition_validation(partitions).to_dict()

        self.assertEqual(first, second)

    def test_failed_nested_gate_does_not_emit_a_promotable_policy(self):
        result = nested_partition_validation(development_partitions())

        self.assertTrue(result.failures)
        self.assertIsNone(result.refit_policy)


if __name__ == "__main__":
    unittest.main()
