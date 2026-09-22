import hashlib
import math
import unittest

from PIL import Image

from qavs.evidence_gap.pdf_runtime import (
    IndependentSupportResult,
    PDFQueryPlan,
    PDFStateEvaluator,
    TreeActionAdapter,
    score_evidence_gaps,
    verify_answer_support,
    wrapper_yes_no_probability,
)
from qavs.evidence_gap.pdf_types import SearchStateRecord
from qavs.evidence_gap.types import sanitize_evidence_requirements
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
        self.assertIn("exactly the four keys zoom, split, expand, next", prompts[0])
        self.assertIn("Do not describe evidence", prompts[0])
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

    def test_nondiscriminative_model_scores_use_operational_analytic_fallback(self):
        analytic = {"zoom": 0.7, "split": 0.5, "expand": 0.3, "next": 0.1}
        result = score_evidence_gaps(
            q0="Question", requirements=REQUIREMENTS,
            generator=lambda prompt: (
                '{"zoom":0.0,"split":0.0,"expand":0.0,"next":0.0}'
            ),
            analytic=analytic,
        )
        self.assertEqual(result.mode, "analytic_fallback")
        self.assertEqual(result.scores.to_dict(), analytic)
        self.assertEqual(result.fallback_reason, "ValueError")

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




    def test_wrapper_prefers_direct_yes_no_logits_over_answerability_proxy(self):
        class DirectVerifier:
            def direct_yes_no_probability(self, image, prompt):
                self.seen = prompt
                return 0.73

            def get_confidence_value(self, *args, **kwargs):
                raise AssertionError("answerability proxy must not be used")

        verifier = DirectVerifier()
        result = wrapper_yes_no_probability(
            verifier, Image.new("RGB", (8, 8)), "Does the crop support A?",
        )
        self.assertEqual(result, 0.73)
        self.assertEqual(verifier.seen, "Does the crop support A?")

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
    def test_wrapper_scores_exact_three_semantic_labels_with_single_token_codes(self):
        from qavs.evidence_gap.pdf_runtime import wrapper_option_label_losses

        class Model:
            def label_token_losses(self, image, prompt, codes, nodes):
                self.seen = (image.size, prompt, codes, nodes)
                return 0, [0.1, 0.7, 1.2]

            def multiple_choices_with_losses(self, *args):
                raise AssertionError("sequence losses must not score verifier labels")

        model = Model()
        result = wrapper_option_label_losses(
            model, Image.new("RGB", (7, 5)), "fixed prompt",
            ("Support", "Refute", "Insufficient"),
        )

        self.assertEqual(result, (0, [0.1, 0.7, 1.2], 1))
        self.assertEqual(model.seen[2], ["A", "B", "C"])
        self.assertTrue(model.seen[3][0].is_root)

    def test_revision3_state_records_complete_option_support_vector(self):
        from qavs.independent_search.grounding import TargetInstanceRegistry
        from qavs.independent_search.semantics import build_option_catalog

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
                {"score": {"rank": 0.8 - index * 0.2}}
                for index, _ in enumerate(nodes)
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
        option_rows = iter((
            (0, [0.0, 2.0, 4.0], 4),
            (2, [4.0, 3.0, 0.0], 4),
        ))
        registry = TargetInstanceRegistry.from_frontend(
            image=image,
            targets=plan.targets,
            candidates=({
                "canonical_key": root_key,
                "bbox_original": [0, 0, image.width, image.height],
                "source": "sam_proposal",
                "sam_target_id": 17,
                "sam_target": "object",
                "sam_has_mask": True,
            },),
            sam_model=None,
            target_instance_iou=0.5,
        )
        evaluator = PDFStateEvaluator(
            generator_model=model, adapter=adapter, policy_annotation=policy,
            query_plan=plan, verifier_probability=None,
            option_catalog=build_option_catalog(policy),
            conditional_losses=lambda image, prompt, labels: next(option_rows),
            target_registry=registry,
            grounding_conditional_losses=(
                lambda image, prompt, labels: (0, [0.0, 2.0, 4.0], 4)
            ),
            grounding_threshold=0.65,
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

        vector = evaluator.option_support_results[0]
        self.assertTrue(vector.valid)
        self.assertEqual(vector.top_key, "index:0")
        self.assertEqual(len(evaluator.records[0]["option_support"]["options"]), 2)
        self.assertTrue(evaluator.records[0]["grounding"]["record"]["valid"])
        self.assertAlmostEqual(result.support_avg, vector.options[0].raw_support)
        self.assertEqual(result.model_calls, 16)

    def test_revision3_relation_bundle_is_jointly_generated_and_reverified(self):
        from qavs.independent_search.grounding import TargetInstanceRegistry
        from qavs.independent_search.semantics import build_option_catalog

        image, catalog, (root_key, left_key, right_key) = make_full_catalog()
        plan = PDFQueryPlan(
            main_query="Is the man beside the bicycle?",
            targets=("man", "bicycle"),
            augmented_queries=("locate man", "locate bicycle", "man bicycle relation"),
            evidence_items=({
                "kind": "relation_context", "targets": ["man", "bicycle"],
            },),
            global_scope_required=False,
            fallback_used=False, fallback_reason=None,
            raw_response_sha256="c" * 64,
            detail_demand=0.5, context_demand=1.0,
        )

        def ranker(nodes, image_pil, main_query, augmented_queries):
            return list(nodes), [
                {"score": {"rank": 0.8 - index * 0.2}}
                for index, _ in enumerate(nodes)
            ]

        model = FakeAnswerModel(
            losses=tuple([0.1, 0.9] for _ in range(9)),
            outputs=(
                '{"zoom":0.1,"split":0.1,"expand":0.1,"next":0.9}',
                '{"zoom":0.1,"split":0.1,"expand":0.1,"next":0.9}',
            ),
        )
        option_rows = iter((
            (0, [0.0, 2.0, 4.0], 4), (1, [2.0, 0.0, 4.0], 4),
            (0, [0.0, 2.0, 4.0], 4), (1, [2.0, 0.0, 4.0], 4),
            (0, [0.0, 2.0, 4.0], 4), (1, [2.0, 0.0, 4.0], 4),
        ))
        grounding_rows = iter((
            (0, [0.0, 2.0, 4.0], 4), (1, [2.0, 0.0, 4.0], 4),
            (1, [2.0, 0.0, 4.0], 4), (0, [0.0, 2.0, 4.0], 4),
        ))
        registry = TargetInstanceRegistry.from_frontend(
            image=image, targets=plan.targets,
            candidates=(
                {
                    "canonical_key": left_key,
                    "bbox_original": [0, 0, 4, 8],
                    "source": "sam_proposal", "sam_target_id": 1,
                    "sam_target": "man", "sam_has_mask": True,
                },
                {
                    "canonical_key": right_key,
                    "bbox_original": [4, 0, 4, 8],
                    "source": "sam_proposal", "sam_target_id": 2,
                    "sam_target": "bicycle", "sam_has_mask": True,
                },
            ),
            sam_model=None, target_instance_iou=0.5,
        )
        evaluator = PDFStateEvaluator(
            generator_model=model,
            adapter=TreeActionAdapter(catalog, image, plan, ranker),
            policy_annotation={
                "question": plan.main_query, "options": ["yes", "no"],
                "answer_type": "logits_match", "input_image": "x.jpg",
            },
            query_plan=plan, verifier_probability=None,
            option_catalog=build_option_catalog({
                "answer_type": "logits_match", "options": ["yes", "no"],
            }),
            conditional_losses=lambda *args: next(option_rows),
            target_registry=registry,
            grounding_conditional_losses=lambda *args: next(grounding_rows),
            grounding_threshold=0.65,
            verifier_checkpoint_sha256="b" * 64,
            generator_checkpoint_sha256="a" * 64,
        )
        states = (
            SearchStateRecord(
                state_id=1, focus_keys=(left_key,),
                path_keys=(root_key, left_key), context_keys=(),
                visited_keys=(root_key, left_key),
                observation_keys=("root@root", f"{left_key}@base"),
                remaining_steps=8, remaining_model_calls=100,
                remaining_pixels=100000000,
            ),
            SearchStateRecord(
                state_id=2, focus_keys=(right_key,),
                path_keys=(root_key, right_key), context_keys=(),
                visited_keys=(root_key, left_key, right_key),
                observation_keys=(
                    "root@root", f"{left_key}@base", f"{right_key}@base",
                ),
                remaining_steps=7, remaining_model_calls=80,
                remaining_pixels=90000000,
            ),
        )

        first = evaluator(states[0])
        second = evaluator(states[1])

        self.assertNotIn("evidence_bundle", evaluator.records[0])
        bundle = evaluator.records[1]["evidence_bundle"]
        self.assertEqual(bundle["source"], "joint_bundle_verification")
        self.assertEqual(bundle["plan"]["constituent_state_ids"], [1, 2])
        self.assertEqual(len(bundle["option_support"]["options"]), 2)
        self.assertEqual(first.model_calls, 20)
        self.assertEqual(second.model_calls, 31)

    def test_pure_target_detail_support_uses_one_local_focus_view(self):
        image, catalog, (root_key, left_key, _) = make_full_catalog()
        plan = PDFQueryPlan(
            main_query="What color is the object?", targets=("object",),
            augmented_queries=("locate object", "object color", "object detail"),
            evidence_items=(ITEMS[0],), global_scope_required=False,
            fallback_used=False, fallback_reason=None,
            raw_response_sha256="c" * 64,
        )

        def ranker(nodes, image_pil, main_query, augmented_queries):
            return list(nodes), [
                {"score": {"rank": 1.0 - index * 0.1}}
                for index, _ in enumerate(nodes)
            ]

        seen_sizes = []
        evaluator = PDFStateEvaluator(
            generator_model=object(),
            adapter=TreeActionAdapter(catalog, image, plan, ranker),
            policy_annotation={
                "question": "What color is the object?",
                "options": ["red", "blue"],
                "answer_type": "logits_match", "input_image": "x.jpg",
            },
            query_plan=plan,
            verifier_probability=lambda view, prompt: seen_sizes.append(view.size) or 0.9,
            verifier_checkpoint_sha256="b" * 64,
            generator_checkpoint_sha256="a" * 64,
        )
        state = SearchStateRecord(
            state_id=1, focus_keys=(left_key,), path_keys=(root_key, left_key),
            context_keys=(), visited_keys=(root_key, left_key),
            observation_keys=(f"{root_key}@root", f"{left_key}@base"),
            remaining_steps=7, remaining_model_calls=40, remaining_pixels=100000,
        )

        evaluator.verify_output_support(state, 0)
        self.assertEqual(seen_sizes, [(4, 8)])

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
