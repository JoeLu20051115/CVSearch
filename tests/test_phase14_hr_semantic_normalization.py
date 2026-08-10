from copy import deepcopy
import unittest

from cvsearch.eval.phase14_hr_semantic_normalization import (
    HR_NORMALIZATION_MIN_FREQUENCY,
    project_hr_semantic_normalization,
    select_hr_semantic_normalization,
)


BLOCKS = [
    "A. Blue\nB. Red\nC. Green\nD. Black\n",
    "A. Green\nB. Blue\nC. Black\nD. Red\n",
    "A. Green\nB. Black\nC. Blue\nD. Red\n",
    "A. Green\nB. Black\nC. Red\nD. Blue\n",
]


def p0(output):
    return {
        "action": "P0",
        "output": output,
        "p0_stability": {"confidence": 0.75},
    }


class HRSemanticNormalizationTests(unittest.TestCase):
    def test_three_of_four_semantic_votes_normalize_all_shuffle_letters(self):
        projection = project_hr_semantic_normalization(
            BLOCKS, ["A", "A", "C", "D"],
        )
        self.assertEqual(HR_NORMALIZATION_MIN_FREQUENCY, 0.75)
        self.assertTrue(projection["feasible"])
        self.assertEqual(projection["output"], ["A", "B", "C", "D"])
        self.assertEqual(projection["canonical_answer"], "blue")
        self.assertEqual(projection["frequency"], 0.75)
        decision = select_hr_semantic_normalization(
            p0(["A", "A", "C", "D"]), BLOCKS,
        )
        self.assertEqual(decision.action, "HR_NORMALIZE")
        self.assertEqual(decision.output, ["A", "B", "C", "D"])

    def test_four_of_four_votes_canonicalize_punctuation_only(self):
        decision = select_hr_semantic_normalization(
            p0(["B.", "D", "D answer", "C"]), BLOCKS,
        )
        self.assertEqual(decision.action, "HR_NORMALIZE")
        self.assertEqual(decision.output, ["B", "D", "D", "C"])
        self.assertEqual(decision.frequency, 1.0)

    def test_two_of_four_tie_retains_exact_p0(self):
        raw = ["A", "A", "A", "C"]
        decision = select_hr_semantic_normalization(p0(raw), BLOCKS)
        self.assertEqual(decision.action, "P0")
        self.assertEqual(decision.status, "retained_p0")
        self.assertEqual(decision.output, raw)

    def test_ambiguous_winner_projection_retains_exact_p0(self):
        blocks = [
            "A. Red\nB. red\nC. Blue\n",
            "A. Blue\nB. red\nC. Green\n",
            "A. Green\nB. Blue\nC. red\n",
            "A. red\nB. Blue\nC. Green\n",
        ]
        raw = ["A", "B.", "C answer", "A"]
        projection = project_hr_semantic_normalization(blocks, raw)
        self.assertFalse(projection["feasible"])
        self.assertEqual(projection["reason"], "ambiguous_winner_projection")
        self.assertEqual(
            select_hr_semantic_normalization(p0(raw), blocks).output, raw,
        )

    def test_malformed_options_fail_closed_and_inputs_are_immutable(self):
        raw = ["A", "B", "C", "D"]
        state = p0(raw)
        before = deepcopy((state, BLOCKS))
        projection = project_hr_semantic_normalization(
            ["A. Blue\nA. Red"] * 4, raw,
        )
        self.assertFalse(projection["feasible"])
        self.assertEqual(projection["reason"], "invalid_option_schema")
        decision = select_hr_semantic_normalization(
            state, ["A. Blue\nA. Red"] * 4,
        )
        self.assertEqual(decision.action, "P0")
        self.assertEqual((state, BLOCKS), before)


if __name__ == "__main__":
    unittest.main()
