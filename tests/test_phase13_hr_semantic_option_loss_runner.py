import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cvsearch.eval.phase13_hr_semantic_option_loss import build_semantic_option_set
from cvsearch.eval.phase13_hr_semantic_option_loss_runner import (
    _answer_hr_loss,
    _observe_hr_loss_path,
    _partition_projectable_pairs,
)
from tests.test_phase13_hr_semantic_option_loss import BLOCKS, DUPLICATE_BLOCKS


class FakeModel:
    def multiple_choices_with_losses(self, image, question, options, nodes):
        self.call = (image, question, options, nodes)
        return 1, [2.0, 1.0, 3.0, 4.0]


class HRSemanticOptionLossRunnerTests(unittest.TestCase):
    def test_unprojectable_semantic_blocks_retain_p0_without_a_model_call(self):
        pairs = [SimpleNamespace(ordinal=1), SimpleNamespace(ordinal=2)]
        rows = {
            1: {"options": BLOCKS},
            2: {"options": DUPLICATE_BLOCKS},
        }
        projectable, excluded = _partition_projectable_pairs(pairs, rows)
        self.assertEqual([pair.ordinal for pair, _ in projectable], [1])
        self.assertEqual(projectable[0][1]["choices"], [
            "Blue", "Red", "Green", "Black",
        ])
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["source_ordinal"], 2)
        self.assertEqual(
            excluded[0]["reason"], "non_unique_semantic_projection",
        )
        self.assertEqual(len(excluded[0]["options_sha256"]), 64)

    def test_answer_call_scores_semantic_texts_once(self):
        model = FakeModel()
        material = build_semantic_option_set(BLOCKS)
        result = _answer_hr_loss(
            model, {"question": "What color?"}, "sheet", material,
        )
        self.assertEqual(result["winner"], 1)
        self.assertEqual(model.call[2], ["Blue", "Red", "Green", "Black"])
        self.assertEqual(model.call[3], [])

    def test_backtrack_is_conditional_on_parent_child_winner(self):
        material = build_semantic_option_set(BLOCKS)
        agree = [
            {"winner": 1, "losses": [2.0, 1.0, 3.0, 4.0]},
            {"winner": 1, "losses": [3.0, 1.0, 2.0, 4.0]},
        ]
        with patch(
            "cvsearch.eval.phase13_hr_semantic_option_loss_runner._answer_hr_loss",
            side_effect=agree,
        ) as answer:
            observations, used = _observe_hr_loss_path(
                object(), {"question": "q"}, [1, 2, 3], material,
            )
        self.assertEqual(len(observations), 2)
        self.assertFalse(used)
        self.assertEqual(answer.call_count, 2)

        disagree = [
            {"winner": 0, "losses": [1.0, 2.0, 3.0, 4.0]},
            {"winner": 1, "losses": [2.0, 1.0, 3.0, 4.0]},
            {"winner": 0, "losses": [1.0, 3.0, 4.0, 5.0]},
        ]
        with patch(
            "cvsearch.eval.phase13_hr_semantic_option_loss_runner._answer_hr_loss",
            side_effect=disagree,
        ) as answer:
            observations, used = _observe_hr_loss_path(
                object(), {"question": "q"}, [1, 2, 3], material,
            )
        self.assertEqual(len(observations), 3)
        self.assertTrue(used)
        self.assertEqual(answer.call_count, 3)


if __name__ == "__main__":
    unittest.main()
