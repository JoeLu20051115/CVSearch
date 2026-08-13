import unittest

from cvsearch.eval.freeze_uncertainty_support import PolicyMetrics
from cvsearch.eval.robust_transfer_selector import (
    AcceptanceCriteria,
    evaluate_acceptance,
    robust_rank,
)


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


if __name__ == "__main__":
    unittest.main()
