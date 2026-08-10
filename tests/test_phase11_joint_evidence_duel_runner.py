from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from cvsearch.eval.phase11_joint_evidence_duel_runner import (
    JOINT_CALLS_PER_RECORD,
    _produce_record,
    answer_duel_logits,
)
from tests.test_evidence_gap_support import support_adapter


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


class ABTokenizer:
    name_or_path = "frozen/checkpoint"

    def __call__(self, text):
        return SimpleNamespace(input_ids={"A": [9454], "B": [2753]}[text])


class Generator:
    def multiple_choices_with_losses(self, image, question, options, nodes):
        self.call = (image, question, options, nodes)
        return 1, [2.0, 1.0]


class Phase11JointEvidenceDuelRunnerTests(unittest.TestCase):
    def test_answer_duel_uses_one_final_position_ab_logit_forward(self):
        model = support_adapter(yes_logit=1.0, no_logit=2.0)
        model.tokenizer = ABTokenizer()
        result = answer_duel_logits(
            model, Image.new("RGB", (4, 3)), "Answer A or B.",
        )
        self.assertEqual(model.model.calls, 1)
        self.assertEqual(result["winner"], 1)
        self.assertEqual((result["a_token_id"], result["b_token_id"]), (9454, 2753))

    def test_produce_record_requires_all_three_cross_model_votes(self):
        from cvsearch.eval import phase11_joint_evidence_duel_runner as runner

        pair = SimpleNamespace(
            ordinal=7, p0=p0(), input_identity={"ordinal": 7},
            extracted_digest="5" * 64,
        )
        row = {
            "answer_type": "logits_match", "options": ["red", "blue"],
            "question": "What color?",
        }
        dense = {"candidate": candidate()}
        first = {"winner": 1, "a_logit": 1.0, "b_logit": 2.0}
        second = {"winner": 0, "a_logit": 2.0, "b_logit": 1.0}
        with (
            patch.object(
                runner, "_render_bound_joint_sheet",
                return_value=(object(), {"sheet_sha256": "6" * 64}),
            ),
            patch.object(runner, "answer_duel_logits", side_effect=[first, second]),
        ):
            record = _produce_record(pair, row, {}, dense, Generator(), object())
        self.assertEqual(record["decision"]["action"], "DENSE")
        self.assertEqual(record["cost"], {"planned_calls": 3, "charged_calls": 3})
        self.assertEqual(record["projection"]["generator"]["winner"], 1)

    def test_partial_failure_fully_charges_and_retains_p0(self):
        from cvsearch.eval import phase11_joint_evidence_duel_runner as runner

        pair = SimpleNamespace(
            ordinal=7, p0=p0(), input_identity={"ordinal": 7},
            extracted_digest="5" * 64,
        )
        row = {
            "answer_type": "logits_match", "options": ["red", "blue"],
            "question": "What color?",
        }
        with (
            patch.object(
                runner, "_render_bound_joint_sheet",
                return_value=(object(), {"sheet_sha256": "6" * 64}),
            ),
            patch.object(
                runner, "answer_duel_logits", side_effect=RuntimeError("failed"),
            ),
        ):
            record = _produce_record(
                pair, row, {}, {"candidate": candidate()}, Generator(), object(),
            )
        self.assertEqual(JOINT_CALLS_PER_RECORD, 3)
        self.assertEqual(record["cost"]["charged_calls"], 3)
        self.assertEqual(record["decision"]["action"], "P0")
        self.assertEqual(record["failure"]["type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
