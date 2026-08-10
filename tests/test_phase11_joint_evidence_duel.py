import unittest

from PIL import Image

from cvsearch.eval.phase9_dense_evidence_search import DenseTile
from cvsearch.eval.phase11_joint_evidence_duel import (
    JOINT_PANEL_SIZE,
    JOINT_SEPARATOR_PIXELS,
    duel_prompts,
    project_joint_duel,
    render_joint_evidence_sheet,
    select_joint_candidate,
)


def p0(output=0):
    return {
        "action": "P0", "output": output,
        "p0_stability": {"confidence": 0.2},
    }


def candidate(output=1):
    return {
        "action": "DENSE", "feasible": True, "output": output,
        "stability": {"confidence": 0.8},
        "tile_majority_fraction": 2 / 3,
        "sheet_sha256": ["1" * 64, "2" * 64, "3" * 64],
        "rank_sha256": "4" * 64,
    }


class JointEvidenceRendererTests(unittest.TestCase):
    def test_renders_three_tiles_and_marked_context_in_exact_grid(self):
        source = Image.new("RGB", (120, 80), (10, 20, 30))
        tiles = (
            DenseTile(2, 0, 0, (0, 0, 70, 50)),
            DenseTile(3, 1, 1, (30, 20, 90, 65)),
            DenseTile(4, 2, 3, (75, 35, 120, 80)),
        )
        before = source.tobytes()

        sheet, audit = render_joint_evidence_sheet(source, tiles)

        expected = JOINT_PANEL_SIZE * 2 + JOINT_SEPARATOR_PIXELS
        self.assertEqual(sheet.size, (expected, expected))
        self.assertEqual(source.tobytes(), before)
        self.assertEqual(audit["tile_identities"], [tile.identity for tile in tiles])
        self.assertEqual(audit["tile_boxes"], [list(tile.box) for tile in tiles])
        self.assertEqual(len(audit["panel_sha256"]), 4)
        self.assertEqual(len(set(audit["panel_sha256"])), 4)
        self.assertEqual(audit["sheet_size"], [expected, expected])


class JointEvidenceDuelTests(unittest.TestCase):
    def test_prompts_swap_answer_order_and_keep_question_fixed(self):
        prompts = duel_prompts("What is written?", "main", "market")
        self.assertIn("A. main\nB. market", prompts["p0_first"])
        self.assertIn("A. market\nB. main", prompts["candidate_first"])
        self.assertEqual(prompts["p0_first"].count("What is written?"), 1)
        self.assertTrue(prompts["candidate_first"].endswith("Answer A or B."))

    def test_unanimous_order_balance_and_generator_loss_select_candidate(self):
        projection = project_joint_duel(
            p0_first={"winner": 1, "a_logit": 1.0, "b_logit": 2.0},
            candidate_first={"winner": 0, "a_logit": 3.0, "b_logit": 1.0},
            generator={"winner": 1, "losses": [2.0, 1.0]},
        )
        self.assertTrue(projection["feasible"])
        decision = select_joint_candidate(p0(), candidate(), projection)
        self.assertEqual(decision.action, "DENSE")
        self.assertEqual(decision.output, 1)

    def test_any_disagreement_tie_or_nonfinite_value_retains_or_rejects(self):
        cases = (
            ({"winner": 0, "a_logit": 2.0, "b_logit": 1.0},
             {"winner": 0, "a_logit": 2.0, "b_logit": 1.0},
             {"winner": 1, "losses": [2.0, 1.0]}),
            ({"winner": 1, "a_logit": 1.0, "b_logit": 2.0},
             {"winner": 0, "a_logit": 2.0, "b_logit": 1.0},
             {"winner": 0, "losses": [1.0, 2.0]}),
        )
        for first, second, generator in cases:
            with self.subTest(generator=generator):
                projection = project_joint_duel(
                    p0_first=first, candidate_first=second, generator=generator,
                )
                self.assertFalse(projection["feasible"])
                self.assertEqual(
                    select_joint_candidate(p0(), candidate(), projection).action,
                    "P0",
                )
        with self.assertRaisesRegex(ValueError, "finite"):
            project_joint_duel(
                p0_first={"winner": 1, "a_logit": float("nan"), "b_logit": 2.0},
                candidate_first={"winner": 0, "a_logit": 2.0, "b_logit": 1.0},
                generator={"winner": 1, "losses": [2.0, 1.0]},
            )

    def test_contaminated_candidate_is_rejected(self):
        projection = project_joint_duel(
            p0_first={"winner": 1, "a_logit": 1.0, "b_logit": 2.0},
            candidate_first={"winner": 0, "a_logit": 2.0, "b_logit": 1.0},
            generator={"winner": 1, "losses": [2.0, 1.0]},
        )
        with self.assertRaisesRegex(ValueError, "invalid exact schema"):
            select_joint_candidate(
                p0(), {**candidate(), "benchmark": "vstar"}, projection,
            )


if __name__ == "__main__":
    unittest.main()
