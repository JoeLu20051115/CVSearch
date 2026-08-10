import unittest
from unittest.mock import patch

from cvsearch.eval.phase13_hr_semantic_option_loss import build_semantic_option_set
from cvsearch.eval.phase13_hr_semantic_option_loss_runner import (
    _answer_hr_loss,
    _observe_hr_loss_path,
)
from tests.test_phase13_hr_semantic_option_loss import BLOCKS


class FakeModel:
    def multiple_choices_with_losses(self, image, question, options, nodes):
        self.call = (image, question, options, nodes)
        return 1, [2.0, 1.0, 3.0, 4.0]


class HRSemanticOptionLossRunnerTests(unittest.TestCase):
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
