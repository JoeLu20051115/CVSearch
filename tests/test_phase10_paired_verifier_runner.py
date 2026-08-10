import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from cvsearch.eval.phase10_paired_verifier_runner import (
    answer_support_probability,
    is_verifier_candidate,
    paired_decisions,
    record_answer_texts,
)
from tests.test_evidence_gap_support import support_adapter


def p0(output=0):
    return {
        "action": "P0", "output": output,
        "p0_stability": {"confidence": 0.2},
    }


def candidate(output=1, feasible=True):
    return {
        "action": "DENSE", "feasible": feasible,
        "output": output if feasible else None,
        "stability": {"confidence": 0.8} if feasible else None,
        "tile_majority_fraction": 2 / 3 if feasible else None,
        "sheet_sha256": ["1" * 64, "2" * 64, "3" * 64],
        "rank_sha256": "4" * 64,
    }


class Phase10PairedVerifierRunnerTests(unittest.TestCase):
    def test_answer_support_uses_one_answer_conditioned_yes_no_forward(self):
        model = support_adapter(yes_logit=2.0, no_logit=-1.0)
        result = answer_support_probability(
            model, Image.new("RGB", (4, 3), (1, 2, 3)),
            question="What is written?", proposed_answer="main street",
        )
        self.assertEqual(model.model.calls, 1)
        self.assertAlmostEqual(
            result["p_yes"], math.exp(2) / (math.exp(2) + math.exp(-1)),
        )
        self.assertEqual((result["yes_token_id"], result["no_token_id"]), (9454, 2753))
        prompt = model.processor.calls[0]["text"][0]
        self.assertIn("Proposed answer: main street", prompt)
        self.assertIn("Answer Yes or No.", prompt)

    def test_eligibility_and_answer_texts_are_label_blind(self):
        pair = SimpleNamespace(p0=p0(), candidates=())
        row = {
            "answer_type": "logits_match",
            "options": ["red", "blue"],
        }
        dense = {"p0": p0(), "candidate": candidate()}
        self.assertTrue(is_verifier_candidate(pair, row, dense))
        self.assertEqual(record_answer_texts(pair, row, dense), ("red", "blue"))
        self.assertFalse(is_verifier_candidate(
            pair, row, {**dense, "candidate": candidate(output=0)},
        ))
        self.assertFalse(is_verifier_candidate(
            pair, row, {**dense, "candidate": candidate(feasible=False)},
        ))

    def test_materializes_exact_six_rule_decisions(self):
        decisions = paired_decisions(
            p0(), candidate(),
            p0_support=[0.2, 0.3, 0.4],
            candidate_support=[0.5, 0.6, 0.4],
        )
        self.assertEqual(len(decisions), 6)
        self.assertEqual(decisions["a0.1-m0.05"]["action"], "DENSE")
        self.assertEqual(decisions["a0.0-m0.0"]["output"], 1)

    def test_partial_verifier_failure_is_conservatively_fully_charged(self):
        from cvsearch.eval import phase10_paired_verifier_runner as runner

        pair = SimpleNamespace(
            ordinal=7,
            p0=p0(),
            input_identity={"ordinal": 7},
            extracted_digest="5" * 64,
        )
        row = {
            "answer_type": "logits_match",
            "options": ["red", "blue"],
            "question": "What color?",
        }
        dense = {"candidate": candidate()}
        support = {
            "p_yes": 0.8,
        }
        with (
            patch.object(runner, "_render_bound_sheets", return_value=[object()] * 3),
            patch.object(
                runner, "answer_support_probability",
                side_effect=[support, support, RuntimeError("forward failed")],
            ) as forward,
        ):
            record = runner._produce_record(pair, row, {}, dense, object())

        self.assertEqual(forward.call_count, 3)
        self.assertEqual(record["cost"]["charged_calls"], 6)
        self.assertEqual(record["decisions"]["a0.0-m0.0"]["action"], "P0")
        self.assertEqual(record["failure"]["type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
