import hashlib
import math
import unittest

from PIL import Image

from cvsearch.evidence_gap.pdf_runtime import (
    IndependentSupportResult,
    PDFQueryPlan,
    PDFStateEvaluator,
    TreeActionAdapter,
    score_evidence_gaps,
    verify_answer_support,
)
from cvsearch.evidence_gap.pdf_types import SearchStateRecord
from cvsearch.evidence_gap.types import sanitize_evidence_requirements
from tests.test_pdf_runtime import FakeAnswerModel, make_full_catalog


ITEMS = (
    {"kind": "target_detail", "target": "small sign",
     "requirements": ["presence", "visual_detail"]},
    {"kind": "relation_context", "targets": ["sign", "door"]},
)
REQUIREMENTS = sanitize_evidence_requirements(ITEMS)


class EvidenceGapScorerTest(unittest.TestCase):
    def test_one_answer_free_call_parses_four_independent_gaps(self):
        prompts = []
        result = score_evidence_gaps(
            q0="What is written above the door?",
            requirements=REQUIREMENTS,
            generator=lambda prompt: prompts.append(prompt) or (
                '{"zoom":0.8,"split":0.6,"expand":0.4,"next":0.2}'
            ),
            analytic={"zoom": 0.1, "split": 0.1, "expand": 0.1, "next": 0.1},
        )
        self.assertEqual(result.mode, "model_json")
        self.assertEqual(result.scores.to_dict(), {
            "zoom": 0.8, "split": 0.6, "expand": 0.4, "next": 0.2,
        })
        self.assertEqual(len(prompts), 1)
        self.assertNotIn("proposed answer", prompts[0].casefold())
        self.assertNotIn("augmented", prompts[0].casefold())
        self.assertEqual(result.model_calls, 1)

    def test_parse_failure_uses_logged_four_score_analytic_fallback(self):
        analytic = {"zoom": 0.7, "split": 0.5, "expand": 0.3, "next": 0.1}
        result = score_evidence_gaps(
            q0="Question", requirements=REQUIREMENTS,
            generator=lambda prompt: "not-json", analytic=analytic,
        )
        self.assertEqual(result.mode, "analytic_fallback")
        self.assertEqual(result.scores.to_dict(), analytic)
        self.assertEqual(result.raw_response_sha256, hashlib.sha256(b"not-json").hexdigest())
        self.assertTrue(result.fallback_reason)

    def test_rejects_partial_or_nonfinite_analytic_scores(self):
        for analytic in (
            {"zoom": 0.1},
            {"zoom": 0.1, "split": 0.2, "expand": 0.3, "next": math.nan},
        ):
            with self.subTest(analytic=analytic), self.assertRaises((TypeError, ValueError)):
                score_evidence_gaps(
                    q0="Question", requirements=REQUIREMENTS,
                    generator=lambda prompt: "bad", analytic=analytic,
                )


class IndependentVerifierTest(unittest.TestCase):
    def test_scores_every_requirement_and_keeps_average_and_minimum(self):
        prompts = []
        values = iter((0.9, 0.4))
        result = verify_answer_support(
            q0="What is written above the door?",
            proposed_answer="OPEN",
            requirements=REQUIREMENTS,
            rendered_observation=Image.new("RGB", (8, 8), "white"),
            probability=lambda image, prompt: prompts.append(prompt) or next(values),
            checkpoint_sha256="b" * 64,
            generator_checkpoint_sha256="a" * 64,
        )
        self.assertIsInstance(result, IndependentSupportResult)
        self.assertEqual(result.per_requirement, (0.9, 0.4))
        self.assertAlmostEqual(result.support_avg, 0.65)
        self.assertEqual(result.support_min, 0.4)
        self.assertTrue(result.independent)
        self.assertFalse(result.fallback_used)
        self.assertEqual(len(prompts), 2)
        self.assertTrue(all("Proposed answer: OPEN" in prompt for prompt in prompts))
        self.assertNotEqual(prompts[0], prompts[1])

    def test_mechanical_coverage_caps_coverage_requirement(self):
        requirements = sanitize_evidence_requirements((
            {"kind": "coverage", "requirement": "global_scope"},
        ))
        result = verify_answer_support(
            q0="Are all signs visible?", proposed_answer="yes",
            requirements=requirements,
            rendered_observation=Image.new("RGB", (8, 8)),
            probability=lambda image, prompt: 0.95,
            checkpoint_sha256="b" * 64,
            generator_checkpoint_sha256="a" * 64,
            coverage_fraction=0.25,
        )
        self.assertEqual(result.per_requirement, (0.25,))
        self.assertEqual(result.support_min, 0.25)

    def test_fallback_is_audited_and_ineligible_for_certified_stop(self):
        calls = []

        def broken(image, prompt):
            raise RuntimeError("verifier failed")

        result = verify_answer_support(
            q0="Question", proposed_answer="answer", requirements=REQUIREMENTS,
            rendered_observation=Image.new("RGB", (8, 8)), probability=broken,
            fallback_probability=lambda image, prompt: calls.append(prompt) or 0.5,
            checkpoint_sha256="b" * 64,
            generator_checkpoint_sha256="a" * 64,
        )
        self.assertFalse(result.independent)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.per_requirement, (0.5, 0.5))
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.failure_type, "RuntimeError")

    def test_same_checkpoint_is_rejected_before_any_support_call(self):
        with self.assertRaisesRegex(ValueError, "independent"):
            verify_answer_support(
                q0="Question", proposed_answer="answer", requirements=REQUIREMENTS,
                rendered_observation=Image.new("RGB", (8, 8)),
                probability=lambda image, prompt: self.fail("must not run"),
                checkpoint_sha256="a" * 64,
                generator_checkpoint_sha256="a" * 64,
            )


class StateEvaluatorTest(unittest.TestCase):
    def test_every_state_reanswers_scores_gaps_and_runs_independent_support(self):
        image, catalog, (root_key, _, _) = make_full_catalog()
        plan = PDFQueryPlan(
            main_query="Which side?", targets=("object",),
            augmented_queries=("locate object", "object detail", "object context"),
            evidence_items=(ITEMS[0],), global_scope_required=False,
            fallback_used=False, fallback_reason=None,
            raw_response_sha256="c" * 64,
        )

        def ranker(nodes, image_pil, main_query, augmented_queries):
            return list(nodes), [
                {"node_id": getattr(node, "id", None), "score": {"rank": 0.8 - index * 0.2}}
                for index, node in enumerate(nodes)
            ]

        adapter = TreeActionAdapter(catalog, image, plan, ranker)
        model = FakeAnswerModel(
            losses=([0.1, 0.9], [0.2, 0.8], [0.15, 0.85]),
            outputs=('{"zoom":0.1,"split":0.8,"expand":0.2,"next":0.3}',),
        )
        policy = {
            "question": "Which side?", "options": ["left", "right"],
            "answer_type": "logits_match", "input_image": "x.jpg",
        }
        evaluator = PDFStateEvaluator(
            generator_model=model, adapter=adapter, policy_annotation=policy,
            query_plan=plan, verifier_probability=lambda image, prompt: 0.9,
            verifier_checkpoint_sha256="b" * 64,
            generator_checkpoint_sha256="a" * 64,
        )
        state = SearchStateRecord(
            state_id=0, focus_keys=(root_key,), path_keys=(root_key,),
            context_keys=(), visited_keys=(root_key,),
            observation_keys=(f"{root_key}@root",), remaining_steps=8,
            remaining_model_calls=40, remaining_pixels=10000000,
        )
        result = evaluator(state)
        self.assertEqual(result.answer, 0)
        self.assertEqual(result.gaps.split, 0.8)
        self.assertEqual(result.support_min, 0.9)
        self.assertTrue(result.verifier_independent)
        self.assertEqual(result.model_calls, 5)
        self.assertEqual(len(evaluator.records), 1)
        self.assertEqual(evaluator.records[0]["answer"]["groups"]["prompt_count"], 3)


if __name__ == "__main__":
    unittest.main()
