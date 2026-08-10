import unittest

from cvsearch.eval.phase10_paired_verifier import (
    PAIRED_AVG_MARGINS,
    PAIRED_MIN_MARGINS,
    paired_rule_key,
    paired_support_stats,
    proposed_answer_text,
    select_paired_candidate,
    verifier_prompt,
)


def p0(output=0):
    return {
        "action": "P0",
        "output": output,
        "p0_stability": {"confidence": 0.2},
    }


def candidate(output=1, feasible=True):
    return {
        "action": "DENSE",
        "feasible": feasible,
        "output": output if feasible else None,
        "stability": {"confidence": 0.8} if feasible else None,
        "tile_majority_fraction": 2 / 3 if feasible else None,
        "sheet_sha256": ["1" * 64, "2" * 64, "3" * 64],
        "rank_sha256": "4" * 64,
    }


class ProposedAnswerTextTests(unittest.TestCase):
    def test_projects_vstar_and_hr_outputs_without_labels(self):
        self.assertEqual(
            proposed_answer_text("logits_match", ["red", " blue  sign "], 1),
            "blue sign",
        )
        blocks = [
            "A. Blue\nB. Red\nC. Green\nD. Black\n",
            "A. Red\nB. Blue\nC. Black\nD. Green\n",
            "A. Green\nB. Red\nC. Blue\nD. Black\n",
            "A. Black\nB. Green\nC. Red\nD. Blue\n",
        ]
        self.assertEqual(
            proposed_answer_text("option_list", blocks, ["A", "B", "C", "D"]),
            "blue",
        )

    def test_unprojectable_hr_answer_fails_closed(self):
        blocks = ["A. Blue\nB. Red\n"] * 4
        self.assertIsNone(
            proposed_answer_text("option_list", blocks, ["invalid"] * 4),
        )

    def test_prompt_is_fixed_answer_conditioned_visual_support(self):
        value = verifier_prompt("What is written?", "main street")
        self.assertIn("Question: What is written?", value)
        self.assertIn("Proposed answer: main street", value)
        self.assertTrue(value.endswith("Answer Yes or No."))


class PairedSupportSelectorTests(unittest.TestCase):
    def test_computes_paired_average_minimum_and_sheet_wins(self):
        value = paired_support_stats([0.2, 0.3, 0.4], [0.5, 0.6, 0.4])
        self.assertAlmostEqual(value.p0_avg, 0.3)
        self.assertAlmostEqual(value.candidate_avg, 0.5)
        self.assertAlmostEqual(value.avg_delta, 0.2)
        self.assertAlmostEqual(value.min_delta, 0.2)
        self.assertEqual(value.candidate_sheet_wins, 2)

    def test_selects_only_paired_average_and_minimum_dominance(self):
        decision = select_paired_candidate(
            p0(), candidate(),
            p0_support=[0.2, 0.3, 0.4],
            candidate_support=[0.5, 0.6, 0.4],
            min_avg_delta=0.1,
            min_min_delta=0.05,
        )
        self.assertEqual(decision.action, "DENSE")
        self.assertEqual(decision.output, 1)
        self.assertEqual(decision.candidate_sheet_wins, 2)

        for candidate_support in (
            [0.5, 0.2, 0.4],
            [0.31, 0.31, 0.41],
            [0.1, 0.8, 0.8],
        ):
            with self.subTest(candidate_support=candidate_support):
                retained = select_paired_candidate(
                    p0(), candidate(),
                    p0_support=[0.2, 0.3, 0.4],
                    candidate_support=candidate_support,
                    min_avg_delta=0.1,
                    min_min_delta=0.05,
                )
                self.assertEqual(retained.action, "P0")
                self.assertEqual(retained.output, 0)

    def test_rule_grid_is_exact_and_rejects_other_thresholds(self):
        self.assertEqual(PAIRED_AVG_MARGINS, frozenset({0.0, 0.05, 0.1}))
        self.assertEqual(PAIRED_MIN_MARGINS, frozenset({0.0, 0.05}))
        self.assertEqual(paired_rule_key(0.1, 0.05), "a0.1-m0.05")
        with self.assertRaisesRegex(ValueError, "frozen grid"):
            paired_rule_key(0.2, 0.05)

    def test_malformed_support_or_contaminated_candidate_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "exactly three"):
            paired_support_stats([0.2], [0.5])
        contaminated = {**candidate(), "benchmark": "vstar"}
        with self.assertRaisesRegex(ValueError, "invalid exact schema"):
            select_paired_candidate(
                p0(), contaminated,
                p0_support=[0.2, 0.3, 0.4],
                candidate_support=[0.5, 0.6, 0.4],
                min_avg_delta=0.0,
                min_min_delta=0.0,
            )


if __name__ == "__main__":
    unittest.main()
