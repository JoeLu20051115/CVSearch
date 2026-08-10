import json
import math
import tempfile
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from cvsearch.evidence_gap.method import MINIMAL_V1, get_evidence_gap_response
from cvsearch.evidence_gap.state import (
    EvidenceState,
    EvidenceStateScore,
    score_state,
    select_state,
)
from cvsearch.evidence_gap.types import AnswerRecord, MethodTrace


def _config(**overrides):
    result = deepcopy(MINIMAL_V1)
    result.update(overrides)
    return result


class EvidenceStateTest(unittest.TestCase):
    def test_score_rewards_independent_support_and_penalizes_uncertainty_and_cost(self):
        strong = EvidenceStateScore(.1, .8, .7, .75, .2)
        weak = EvidenceStateScore(.4, .5, .3, .50, .4)

        self.assertGreater(score_state(strong), score_state(weak))

    def test_equal_score_keeps_the_exact_anchor_object(self):
        features = EvidenceStateScore(.2, .5, .5, .5, .2)
        anchor = EvidenceState(AnswerRecord(output="A"), features)
        candidate = EvidenceState(AnswerRecord(output="B"), features)

        selected, margin = select_state(anchor, candidate, tau=0.0)

        self.assertIs(selected, anchor)
        self.assertEqual(margin, 0.0)

    def test_state_contracts_are_frozen_and_reject_invalid_feature_values(self):
        score = EvidenceStateScore(.1, .2, .3, .4, .5)
        state = EvidenceState(AnswerRecord(output="A"), score)
        with self.assertRaises(FrozenInstanceError):
            score.coverage = .6
        with self.assertRaises(FrozenInstanceError):
            state.answer = AnswerRecord(output="B")

        invalid = (True, float("nan"), float("inf"), float("-inf"), -.01, 1.01, "0.5")
        for index in range(5):
            for value in invalid:
                values = [.1, .2, .3, .4, .5]
                values[index] = value
                with self.subTest(index=index, value=value):
                    with self.assertRaises((TypeError, ValueError)):
                        EvidenceStateScore(*values)

    def test_state_copies_answer_into_a_deeply_immutable_snapshot(self):
        original = AnswerRecord(
            output={"items": ["A"]},
            raw_outputs=({"raw": ["A"]},),
            groups={"cat": ["A"]},
        )
        state = EvidenceState(original, EvidenceStateScore(.2, .5, .5, .5, .2))
        before = state.answer.to_dict()

        original.output["items"].append("B")
        original.raw_outputs[0]["raw"].append("B")
        original.groups["cat"].append("B")

        self.assertEqual(state.answer.to_dict(), before)

    def test_state_exposes_no_mutable_answer_or_serialization_alias(self):
        state = EvidenceState(
            AnswerRecord(output={"items": ["A"]}, groups={"cat": ["A"]}),
            EvidenceStateScore(.2, .5, .5, .5, .2),
        )

        with self.assertRaises((AttributeError, TypeError, FrozenInstanceError)):
            state.answer.output["items"].append("B")
        with self.assertRaises((AttributeError, TypeError, FrozenInstanceError)):
            state.answer.groups["cat"] = ["B"]
        serialized = state.answer.to_dict()
        serialized["output"]["items"].append("B")

        self.assertEqual(state.answer.to_dict()["output"], {"items": ["A"]})

    def test_select_state_rejects_invalid_tau_without_changing_anchor_rule(self):
        features = EvidenceStateScore(.2, .5, .5, .5, .2)
        anchor = EvidenceState(AnswerRecord(output="A"), features)
        candidate = EvidenceState(AnswerRecord(output="B"), features)
        for tau in (True, float("nan"), float("inf"), float("-inf"), -.01, "0"):
            with self.subTest(tau=tau):
                with self.assertRaises((TypeError, ValueError)):
                    select_state(anchor, candidate, tau=tau)

    def test_trace_serializes_anchor_and_audit_fields_as_strict_json(self):
        trace = MethodTrace(
            anchor_answer=AnswerRecord(output="A"),
            anchor_state_score=.4,
            selected_state_score=.5,
            replacement_margin=.1,
            support_status="not_observed",
        )

        payload = trace.to_dict()
        encoded = json.dumps(payload, allow_nan=False)

        self.assertEqual(payload["anchor_answer"]["output"], "A")
        self.assertEqual(payload["replacement_margin"], .1)
        self.assertEqual(payload["support_status"], "not_observed")
        self.assertIn("anchor_state_score", encoded)
        self.assertNotIn("category", encoded)


class EvidenceStateRuntimeAuditTest(unittest.TestCase):
    def _assert_audit_is_inert(self, output, expected, trace):
        self.assertEqual(output, expected)
        self.assertEqual(trace.final_answer.output, expected)
        self.assertIsNot(trace.anchor_answer, trace.final_answer)
        self.assertEqual(trace.anchor_answer.to_dict(), trace.final_answer.to_dict())
        self.assertEqual(trace.anchor_state_score, trace.selected_state_score)
        self.assertEqual(trace.replacement_margin, 0.0)
        self.assertEqual(trace.support_status, "not_observed")
        json.dumps(trace.to_dict(), allow_nan=False)

    def test_three_existing_runtime_paths_preserve_output_and_only_record_audit_score(self):
        rerank_policy = {
            "question": "Return the existing opaque response.",
            "options": ["one", "two"],
            "answer_type": "opaque_test_type",
            "input_image": "not-needed.jpg",
        }
        output, trace = get_evidence_gap_response(
            sam_model=object(), zoom_model=object(), nlp_model=object(),
            policy_annotation=rerank_policy, original_annotation={}, ic_examples=[],
            decomposed_question_template="{}", config=_config(),
            cvsearch_fn=lambda **_: {"answer": "unchanged"},
        )
        self._assert_audit_is_inert(output, {"answer": "unchanged"}, trace)

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)

            class LogitsZoom:
                def multiple_choices_with_losses(self, image_pil, question, options, searched_nodes=None):
                    return 0, [.1, .9]

            logits_policy = {
                "question": "Which sign is visible?",
                "options": ["red", "blue"],
                "answer_type": "logits_match",
                "input_image": str(image_path),
            }
            output, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=LogitsZoom(), nlp_model=object(),
                policy_annotation=logits_policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}",
                config=_config(mode="root_search_fallback", rerank_enabled=False),
                cvsearch_fn=lambda **_: 0,
            )
            self._assert_audit_is_inert(output, 0, trace)

            class HrZoom:
                def __init__(self):
                    self.outputs = iter(("A", "B", "A", "B"))

                def free_form_using_nodes(self, image_pil, question, searched_nodes):
                    return next(self.outputs)

            blocks = [
                "A. cat\nB. dog", "A. dog\nB. cat",
                "A. cat\nB. dog", "A. dog\nB. cat",
            ]
            raw = ["B", "A", "B", "A"]
            hr_policy = {
                "question": "What animal is shown?", "options": blocks,
                "answer_type": "option_list", "input_image": str(image_path),
            }
            output, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=HrZoom(), nlp_model=object(),
                policy_annotation=hr_policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}",
                config=_config(mode="root_search_fallback", rerank_enabled=False),
                cvsearch_fn=lambda **_: deepcopy(raw),
            )
            self._assert_audit_is_inert(output, raw, trace)

    def test_unobserved_support_never_executes_state_selection(self):
        policy = {
            "question": "Return raw response.", "options": ["one", "two"],
            "answer_type": "opaque_test_type", "input_image": "not-needed.jpg",
        }
        with patch(
            "cvsearch.evidence_gap.method.select_state",
            side_effect=AssertionError("unobserved support must not select a state"),
        ):
            output, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=object(), nlp_model=object(),
                policy_annotation=policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", config=_config(),
                cvsearch_fn=lambda **_: "raw",
            )

        self._assert_audit_is_inert(output, "raw", trace)


if __name__ == "__main__":
    unittest.main()
