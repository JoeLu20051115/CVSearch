import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from cvsearch.evidence_gap.method import (
    MINIMAL_V1,
    build_query_plan,
    compose_output_record,
    get_evidence_gap_response,
    load_method_config,
)
from cvsearch.evidence_gap.input import split_bucket
from cvsearch.evidence_gap.types import AnswerRecord, BudgetExceeded, FORCED_RETURN, MethodTrace
from cvsearch.perform_EGSearch import build_parser, parse_ordinals, select_annotations
from cvsearch import perform_EGSearch as eg_cli


ROOT = Path(__file__).resolve().parents[1]


class AccessCanary:
    def __init__(self):
        self.allowed = False
        self.accesses = 0

    def __deepcopy__(self, memo):
        self.accesses += 1
        if not self.allowed:
            raise AssertionError("evaluator-only field accessed during policy")
        return "restored-truth"


class FakeScorer:
    def score(self, images, texts):
        return [[float(index + 1) for _ in texts] for index in range(len(images))]


class FakeNode:
    def __init__(self, identifier, complexity):
        self.id = identifier
        self.complexity = complexity
        self.state = SimpleNamespace(bbox=(0, 0, 2, 2))


def base_config(**overrides):
    config = deepcopy(MINIMAL_V1)
    config.update(overrides)
    return config


class QueryPlanTest(unittest.TestCase):
    def setUp(self):
        self.policy = {
            "question": "How many signs are beside the bus?",
            "options": ["one", "two"],
            "answer_type": "logits_match",
            "input_image": "image.jpg",
        }

    def test_plan_is_deterministic_answer_free_deduplicated_and_strict_json(self):
        targets = [" blue sign ", "BUS", "blue   sign", ""]
        first = build_query_plan(self.policy, targets)
        second = build_query_plan(deepcopy(self.policy), tuple(targets))

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.main_query, self.policy["question"])
        self.assertEqual(first.targets, ("blue sign", "BUS"))
        self.assertTrue(first.augmented_queries)
        self.assertTrue(first.evidence_items)
        self.assertTrue(first.global_scope_required)
        encoded = json.dumps(first.to_dict(), allow_nan=False)
        self.assertNotIn("one", encoded.casefold())
        self.assertNotIn("two", encoded.casefold())

    def test_plan_rejects_unsanitized_views_without_using_options_as_policy_input(self):
        with self.assertRaises(ValueError):
            build_query_plan(dict(self.policy, answer=1), ["sign"])
        with self.assertRaises(ValueError):
            build_query_plan({key: value for key, value in self.policy.items() if key != "input_image"}, ["sign"])

    def test_plan_never_iterates_or_deepcopies_options(self):
        class RaisingOptions:
            def __iter__(self):
                raise AssertionError("options iterated")

            def __deepcopy__(self, memo):
                raise AssertionError("options deep-copied")

        policy = dict(self.policy, options=RaisingOptions())
        plan = build_query_plan(policy, ["street sign"])
        self.assertEqual(plan.main_query, self.policy["question"])
        self.assertEqual(plan.targets, ("street sign",))

    def test_option_collision_filter_is_deferred_for_question_derived_targets(self):
        plan = build_query_plan(self.policy, ["one"])
        self.assertEqual(plan.targets, ("one",))

    def test_empty_targets_get_safe_localization_fallback(self):
        plan = build_query_plan(self.policy, [])
        self.assertEqual(plan.targets, ())
        self.assertEqual(plan.augmented_queries, ())
        self.assertTrue(plan.fallback_used)


class MethodCompositionTest(unittest.TestCase):
    def setUp(self):
        self.policy = {
            "question": "Which sign is visible?",
            "options": ["red", "blue"],
            "answer_type": "logits_match",
            "input_image": "image.jpg",
        }

    def test_default_config_pins_minimal_v1_and_rejects_unknown_keys(self):
        config = load_method_config("minimal_v1")
        self.assertEqual(
            {key: config[key] for key in (
                "beta", "alpha", "visual_lambda", "quick_gate", "root_fallback_tolerance"
            )},
            {"beta": 0.6, "alpha": 0.65, "visual_lambda": 0.5,
             "quick_gate": 0.8, "root_fallback_tolerance": 0.05},
        )
        self.assertFalse(config["enable_zoom"])
        self.assertFalse(config["enable_split"])
        self.assertFalse(config["enable_expand"])
        self.assertEqual(config["mode"], "rerank_only")
        self.assertGreater(config["max_mllm_calls"], 0)
        self.assertGreater(config["max_processed_pixels"], 0)
        with self.assertRaises(ValueError):
            load_method_config(dict(config, secret_setting=True))

    def test_version_safe_external_config_id_is_preserved_in_trace(self):
        frozen = base_config(config_id="frozen_v1", rerank_enabled=False)
        loaded = load_method_config(frozen)
        self.assertEqual(loaded["config_id"], "frozen_v1")
        _, trace = get_evidence_gap_response(
            sam_model=object(), zoom_model=object(), nlp_model=object(),
            policy_annotation=self.policy, original_annotation={}, ic_examples=[],
            decomposed_question_template="{}", config=frozen,
            cvsearch_fn=lambda **_: 0,
        )
        self.assertEqual(trace.config_id, "frozen_v1")
        self.assertEqual(trace.effective_config, loaded)
        for invalid in ("", "with space", "../escape", "x" * 65):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    load_method_config(base_config(config_id=invalid))

    def test_policy_callbacks_receive_isolated_sanitized_copies_and_truth_is_untouched(self):
        truth = AccessCanary()
        original = dict(self.policy, answer=truth, category=truth, index=truth)
        caller_policy = deepcopy(self.policy)
        seen = []

        def planner(annotation, targets):
            seen.append(("planner", set(annotation)))
            annotation["question"] = "mutated planner copy"
            return build_query_plan(self.policy, targets)

        def policy_callback(annotation):
            seen.append(("policy", set(annotation)))
            annotation["options"].clear()
            return None

        def fake_cvsearch(**kwargs):
            seen.append(("cvsearch", set(kwargs["annotation"])))
            kwargs["annotation"]["question"] = "mutated CVSearch copy"
            kwargs["answer_observer"]("search", [], 1)
            return 1

        response, trace = get_evidence_gap_response(
            sam_model=object(), zoom_model=object(), nlp_model=object(),
            policy_annotation=caller_policy, original_annotation=original,
            ic_examples=[], decomposed_question_template="What is the appearance of the {}?",
            config=base_config(rerank_enabled=False), cvsearch_fn=fake_cvsearch,
            planner=planner, policy_callback=policy_callback,
        )

        self.assertEqual(response, 1)
        self.assertEqual(caller_policy, self.policy)
        self.assertEqual(truth.accesses, 0)
        self.assertEqual([keys for _, keys in seen], [set(self.policy)] * 3)
        json.dumps(trace.to_dict(), allow_nan=False)

        truth.allowed = True
        record = compose_output_record(original, response, trace)
        self.assertEqual(record["answer"], "restored-truth")
        self.assertEqual(record["output"], 1)
        self.assertIn("method_trace", record)
        self.assertNotIn("_eg_ordinal", record)
        self.assertNotIn("_eg_run_fingerprint", record)

    def test_rerank_enabled_preserves_pool_records_details_and_answer_observations(self):
        nodes = [FakeNode("low", 0.1), FakeNode("high", 0.9)]

        def fake_cvsearch(**kwargs):
            ranked, details = kwargs["node_ranker"](
                nodes, Image.new("RGB", (4, 4), "white"),
                kwargs["annotation"]["question"], kwargs["method_trace"].query_plan.augmented_queries,
            )
            self.assertCountEqual(ranked, nodes)
            kwargs["method_trace"].candidate_ranks.extend(details)
            kwargs["answer_observer"]("search", [nodes[1]], 1)
            return 1

        def calibrated_record(annotation, phase, raw_answer):
            self.assertEqual(set(annotation), set(self.policy))
            confidence = 0.5
            return AnswerRecord(
                output=raw_answer, canonical_answer=raw_answer,
                confidence=confidence, uncertainty=1.0 - confidence,
            )

        response, trace = get_evidence_gap_response(
            sam_model=object(), zoom_model=object(), nlp_model=object(),
            policy_annotation=self.policy, original_annotation=dict(self.policy, answer=0),
            ic_examples=[], decomposed_question_template="What is the appearance of the {}?",
            config=base_config(rerank_enabled=True), cvsearch_fn=fake_cvsearch,
            scorer=FakeScorer(), targets=("sign",), answer_record_factory=calibrated_record,
        )

        self.assertEqual(response, 1)
        self.assertEqual(len(trace.candidate_ranks), 2)
        self.assertEqual(trace.final_answer.selected_from, "search")
        self.assertEqual(trace.history, [])
        self.assertEqual(trace.termination, FORCED_RETURN)
        self.assertEqual(trace.steps[-1].action, FORCED_RETURN)
        self.assertEqual(trace.final_boxes, ((0, 0, 2, 2),))
        json.dumps(trace.to_dict(), allow_nan=False)

    def test_rerank_enabled_requires_a_process_wide_injected_rank_dependency(self):
        with patch(
            "cvsearch.evidence_gap.clip_scorer.ClipScorer",
            side_effect=AssertionError("per-sample CLIP construction"),
        ) as clip_constructor:
            with self.assertRaisesRegex(ValueError, "injected scorer or node_ranker"):
                get_evidence_gap_response(
                    sam_model=object(), zoom_model=object(), nlp_model=object(),
                    policy_annotation=self.policy, original_annotation={}, ic_examples=[],
                    decomposed_question_template="{}", config=base_config(rerank_enabled=True),
                    cvsearch_fn=lambda **_: 0,
                )
        clip_constructor.assert_not_called()

    def test_root_search_fallback_uses_calibrated_records_and_always_forces(self):
        class Zoom:
            def __init__(self):
                self.loss_rows = iter(([0.1, 0.9], [0.4, 0.6]))

            def multiple_choices_with_losses(self, image_pil, question, options, searched_nodes=None):
                losses = next(self.loss_rows)
                return min(range(len(losses)), key=losses.__getitem__), losses

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (4, 4), "white").save(image_path)
            policy = dict(self.policy, input_image=str(image_path))

            def fake_cvsearch(**kwargs):
                node = FakeNode("searched", 0.5)
                kwargs["annotation"]["searched_bbox"] = [[0, 0, 2, 2]]
                kwargs["answer_observer"]("search", [node], 0)
                return 0

            response, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=Zoom(), nlp_model=object(),
                policy_annotation=policy, original_annotation=dict(policy, answer=0),
                ic_examples=[], decomposed_question_template="{}",
                config=base_config(mode="root_search_fallback", rerank_enabled=False),
                cvsearch_fn=fake_cvsearch,
            )

        self.assertEqual(response, 0)
        self.assertEqual([item.answer.selected_from for item in trace.history], ["root", "search"])
        self.assertEqual([item.cost for item in trace.history], [3, 6])
        self.assertEqual(trace.final_answer.selected_from, "root")
        self.assertEqual(trace.termination, FORCED_RETURN)
        json.dumps(trace.to_dict(), allow_nan=False)

    def test_root_search_fallback_selects_search_at_tolerance_boundary(self):
        class Zoom:
            def __init__(self):
                self.loss_rows = iter(([0.1, 0.9], [0.125, 0.875]))

            def multiple_choices_with_losses(self, image_pil, question, options, searched_nodes=None):
                losses = next(self.loss_rows)
                return 0, losses

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            policy = dict(self.policy, input_image=str(image_path))

            def fake_cvsearch(**kwargs):
                kwargs["answer_observer"]("search", [FakeNode("searched", 0.5)], 0)
                return 0

            _, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=Zoom(), nlp_model=object(),
                policy_annotation=policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", cvsearch_fn=fake_cvsearch,
                config=base_config(mode="root_search_fallback", rerank_enabled=False),
            )

        self.assertAlmostEqual(trace.history[0].answer.confidence, 0.8)
        self.assertAlmostEqual(trace.history[1].answer.confidence, 0.75)
        self.assertEqual(trace.final_answer.selected_from, "search")

    def test_root_search_fallback_rejects_loss_winner_mismatch(self):
        class Zoom:
            def multiple_choices_with_losses(self, image_pil, question, options, searched_nodes=None):
                return 0, [0.1, 0.9]

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            policy = dict(self.policy, input_image=str(image_path))

            def contradictory_cvsearch(**kwargs):
                kwargs["answer_observer"]("search", [FakeNode("searched", 0.5)], 1)
                return 1

            with self.assertRaisesRegex(ValueError, "disagrees"):
                get_evidence_gap_response(
                    sam_model=object(), zoom_model=Zoom(), nlp_model=object(),
                    policy_annotation=policy, original_annotation={}, ic_examples=[],
                    decomposed_question_template="{}", cvsearch_fn=contradictory_cvsearch,
                    config=base_config(mode="root_search_fallback", rerank_enabled=False),
                )

    def test_hr_root_search_fallback_returns_four_semantically_aligned_letters(self):
        option_blocks = [
            "A. cat\nB. dog", "A. dog\nB. cat",
            "A. cat\nB. dog", "A. dog\nB. cat",
        ]

        class Zoom:
            def __init__(self):
                self.outputs = iter(("A", "B", "A", "B"))

            def free_form_using_nodes(self, image_pil, question, searched_nodes):
                return next(self.outputs)

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            policy = {
                "question": "Which animal is visible?", "options": option_blocks,
                "answer_type": "option_list", "input_image": str(image_path),
            }

            def fake_cvsearch(**kwargs):
                raw = ["A", "B", "A", "B"]
                kwargs["answer_observer"]("search", [FakeNode("searched", 0.5)], raw)
                return raw

            response, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=Zoom(), nlp_model=object(),
                policy_annotation=policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", cvsearch_fn=fake_cvsearch,
                config=base_config(mode="root_search_fallback", rerank_enabled=False),
            )

        self.assertEqual(response, ["A", "B", "A", "B"])
        self.assertEqual(trace.final_answer.selected_from, "search")
        self.assertEqual(len(response), len(option_blocks))

    def test_hr_root_batch_is_fully_preauthorized_before_any_generation(self):
        blocks = [
            "A. cat\nB. dog", "A. dog\nB. cat",
            "A. cat\nB. dog", "A. dog\nB. cat",
        ]
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            policy = {
                "question": "Which animal is visible?", "options": blocks,
                "answer_type": "option_list", "input_image": str(image_path),
            }

            class Zoom:
                def __init__(self):
                    self.calls = 0

                def free_form_using_nodes(self, image_pil, question, searched_nodes):
                    self.calls += 1
                    return ("A", "B", "A", "B")[self.calls - 1]

            for max_calls, max_pixels in ((3, 16), (4, 15)):
                with self.subTest(max_calls=max_calls, max_pixels=max_pixels):
                    model = Zoom()
                    with self.assertRaises(BudgetExceeded):
                        get_evidence_gap_response(
                            sam_model=object(), zoom_model=model, nlp_model=object(),
                            policy_annotation=policy, original_annotation={}, ic_examples=[],
                            decomposed_question_template="{}", cvsearch_fn=lambda **_: [],
                            config=base_config(
                                mode="root_search_fallback", rerank_enabled=False,
                                max_mllm_calls=max_calls, max_processed_pixels=max_pixels,
                            ),
                        )
                    self.assertEqual(model.calls, 0)

    def test_hr_root_mode_search_terminal_batch_rejects_before_first_call_when_only_three_fit(self):
        blocks = [
            "A. cat\nB. dog", "A. dog\nB. cat",
            "A. cat\nB. dog", "A. dog\nB. cat",
        ]

        class Zoom:
            def __init__(self):
                self.calls = []
                self.root_outputs = iter(("A", "B", "A", "B"))

            def free_form_using_nodes(self, image_pil, question, searched_nodes):
                self.calls.append(tuple(searched_nodes))
                if searched_nodes:
                    raise AssertionError("partial searched-node HR answer reached the model")
                return next(self.root_outputs)

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            policy = {
                "question": "Which animal?", "options": blocks,
                "answer_type": "option_list", "input_image": str(image_path),
            }
            def first_search_terminal_call(**kwargs):
                return kwargs["zoom_model"].free_form_using_nodes(
                    Image.new("RGB", (2, 2)), "q", ["search"]
                )

            for max_calls, max_pixels in ((7, 32), (8, 31)):
                with self.subTest(max_calls=max_calls, max_pixels=max_pixels):
                    model = Zoom()
                    response, trace = get_evidence_gap_response(
                        sam_model=object(), zoom_model=model, nlp_model=object(),
                        policy_annotation=policy, original_annotation={}, ic_examples=[],
                        decomposed_question_template="{}", cvsearch_fn=first_search_terminal_call,
                        config=base_config(
                            mode="root_search_fallback", rerank_enabled=False,
                            max_mllm_calls=max_calls, max_processed_pixels=max_pixels,
                        ),
                    )
                    self.assertEqual(response, ["A", "B", "A", "B"])
                    self.assertEqual(model.calls, [(), (), (), ()])
                    self.assertEqual(
                        (trace.budget.mllm_calls, trace.budget.processed_pixels), (4, 16)
                    )
                    self.assertTrue(trace.budget_interrupted)
                    self.assertEqual(trace.final_answer.selected_from, "root")

    def test_hr_root_mode_search_terminal_executes_exactly_four_then_rejects_fifth(self):
        blocks = [
            "A. cat\nB. dog", "A. dog\nB. cat",
            "A. cat\nB. dog", "A. dog\nB. cat",
        ]

        class Zoom:
            def __init__(self):
                self.calls = []
                self.outputs = iter(("A", "B", "A", "B", "A", "B", "A", "B"))

            def free_form_using_nodes(self, image_pil, question, searched_nodes):
                self.calls.append(tuple(searched_nodes))
                return next(self.outputs)

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            policy = {
                "question": "Which animal?", "options": blocks,
                "answer_type": "option_list", "input_image": str(image_path),
            }
            model = Zoom()

            def five_search_terminal_calls(**kwargs):
                image = Image.new("RGB", (2, 2))
                raw = [
                    kwargs["zoom_model"].free_form_using_nodes(image, "q", ["search"])
                    for _ in range(4)
                ]
                kwargs["answer_observer"]("search", [], raw)
                kwargs["zoom_model"].free_form_using_nodes(image, "q", ["fifth"])

            response, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=model, nlp_model=object(),
                policy_annotation=policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", cvsearch_fn=five_search_terminal_calls,
                config=base_config(
                    mode="root_search_fallback", rerank_enabled=False,
                    max_mllm_calls=12, max_processed_pixels=48,
                ),
            )
        self.assertEqual(response, ["A", "B", "A", "B"])
        self.assertEqual(model.calls, [(), (), (), ()] + [("search",)] * 4)
        self.assertEqual((trace.budget.mllm_calls, trace.budget.processed_pixels), (8, 32))
        self.assertTrue(trace.budget_interrupted)
        self.assertEqual(trace.final_answer.selected_from, "search")

    def test_root_search_fallback_returns_root_when_search_budget_is_exhausted(self):
        class Zoom:
            def multiple_choices_with_losses(self, image_pil, question, options, searched_nodes=None):
                return 0, [0.1, 0.9]

            def get_confidence_value(self, *args, **kwargs):
                return 0.0

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            policy = dict(self.policy, input_image=str(image_path))

            def exhausted_cvsearch(**kwargs):
                kwargs["annotation"].update({
                    "search_mode": 2,
                    "num_pop": [1],
                    "num_zoom_in": [0],
                    "num_zoom_out": [0],
                })
                kwargs["zoom_model"].get_confidence_value([], Image.new("RGB", (2, 2)), "answering", "q")
                kwargs["zoom_model"].get_confidence_value([], Image.new("RGB", (2, 2)), "answering", "q")

            response, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=Zoom(), nlp_model=object(),
                policy_annotation=policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", cvsearch_fn=exhausted_cvsearch,
                config=base_config(
                    mode="root_search_fallback", rerank_enabled=False,
                    max_mllm_calls=4, max_processed_pixels=40,
                ),
            )

        self.assertEqual(response, 0)
        self.assertEqual(trace.final_answer.selected_from, "root")
        self.assertEqual(trace.termination, FORCED_RETURN)
        self.assertTrue(trace.budget_interrupted)
        self.assertEqual(trace.cvsearch_search_mode, 2)
        record = compose_output_record({}, response, trace)
        self.assertEqual(record["num_pop"], [1])

    def test_root_search_fallback_returns_root_when_loss_recheck_exceeds_budget(self):
        class Zoom:
            def multiple_choices_with_losses(self, image_pil, question, options, searched_nodes=None):
                return 0, [0.1, 0.9]

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            policy = dict(self.policy, input_image=str(image_path))

            def observed_search(**kwargs):
                kwargs["answer_observer"]("search", [FakeNode("searched", 0.5)], 0)
                return 0

            response, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=Zoom(), nlp_model=object(),
                policy_annotation=policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", cvsearch_fn=observed_search,
                config=base_config(
                    mode="root_search_fallback", rerank_enabled=False,
                    max_mllm_calls=3, max_processed_pixels=12,
                ),
            )

        self.assertEqual(response, 0)
        self.assertEqual(trace.final_answer.selected_from, "root")
        self.assertEqual(len(trace.history), 1)

    def test_first_cvsearch_call_budget_interrupt_has_strict_empty_runtime_cost_defaults(self):
        class Zoom:
            def multiple_choices_with_losses(self, image_pil, question, options, searched_nodes=None):
                return 0, [0.1, 0.9]

            def get_confidence_value(self, *args, **kwargs):
                raise AssertionError("budget must fail before model execution")

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            policy = dict(self.policy, input_image=str(image_path))

            def first_call(**kwargs):
                return kwargs["zoom_model"].get_confidence_value(
                    [], Image.new("RGB", (2, 2)), "answering", "q"
                )

            response, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=Zoom(), nlp_model=object(),
                policy_annotation=policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", cvsearch_fn=first_call,
                config=base_config(
                    mode="root_search_fallback", rerank_enabled=False,
                    max_mllm_calls=3, max_processed_pixels=12,
                ),
            )
        self.assertEqual(response, 0)
        self.assertTrue(trace.budget_interrupted)
        self.assertIsNone(trace.cvsearch_search_mode)
        self.assertEqual((trace.num_pop, trace.num_zoom_in, trace.num_zoom_out), ([], [], []))
        payload = trace.to_dict()
        json.dumps(payload, allow_nan=False)
        record = compose_output_record({}, response, trace)
        self.assertEqual((record["num_pop"], record["num_zoom_in"], record["num_zoom_out"]), ([], [], []))

    def test_rerank_disabled_passes_none_and_preserves_raw_output(self):
        def fake_cvsearch(**kwargs):
            self.assertIsNone(kwargs["node_ranker"])
            kwargs["annotation"]["searched_bbox"] = [[1, 2, 3, 4]]
            return {"tokens": ["raw"]}

        response, trace = get_evidence_gap_response(
            sam_model=object(), zoom_model=object(), nlp_model=object(),
            policy_annotation=self.policy, original_annotation=dict(self.policy, answer=0),
            ic_examples=[], decomposed_question_template="What is the appearance of the {}?",
            config=base_config(rerank_enabled=False), cvsearch_fn=fake_cvsearch,
        )
        self.assertEqual(response, {"tokens": ["raw"]})
        self.assertEqual(trace.final_answer.output, response)
        self.assertEqual(trace.final_boxes, ((1, 2, 3, 4),))
        self.assertEqual(trace.termination, FORCED_RETURN)

    def test_hr_rerank_only_trace_output_matches_raw_quick_and_search_response(self):
        blocks = [
            "A. cat\nB. dog", "A. dog\nB. cat",
            "A. cat\nB. dog", "A. dog\nB. cat",
        ]
        policy = dict(self.policy, answer_type="option_list", options=blocks)
        raw = ["A", "A", "A", "A"]
        for phase in ("quick", "search"):
            with self.subTest(phase=phase):
                def fake_cvsearch(**kwargs):
                    kwargs["answer_observer"](phase, [], raw)
                    return deepcopy(raw)

                response, trace = get_evidence_gap_response(
                    sam_model=object(), zoom_model=object(), nlp_model=object(),
                    policy_annotation=policy, original_annotation={}, ic_examples=[],
                    decomposed_question_template="{}", config=base_config(rerank_enabled=False),
                    cvsearch_fn=fake_cvsearch,
                )
                self.assertEqual(response, raw)
                self.assertEqual(trace.final_answer.output, response)
                self.assertEqual(trace.final_answer.canonical_answer, "cat")
                self.assertTrue(trace.final_answer.groups)

    def test_composition_rejects_output_trace_disagreement(self):
        trace = MethodTrace(final_answer=AnswerRecord(output=1), termination=FORCED_RETURN)
        with self.assertRaisesRegex(ValueError, "disagree"):
            compose_output_record({}, 0, trace)

    def test_posthoc_runtime_targets_disclose_the_effective_ranking_query(self):
        def fake_cvsearch(**kwargs):
            self.assertEqual(kwargs["method_trace"].query_plan.augmented_queries, ())
            kwargs["annotation"]["targets"] = ["street sign"]
            return 0

        _, trace = get_evidence_gap_response(
            sam_model=object(), zoom_model=object(), nlp_model=object(),
            policy_annotation=self.policy, original_annotation={}, ic_examples=[],
            decomposed_question_template="{}", config=base_config(rerank_enabled=False),
            cvsearch_fn=fake_cvsearch,
        )
        runtime_context = trace.query_plan.evidence_items[-1]
        self.assertEqual(runtime_context["kind"], "runtime_ranking_context")
        self.assertEqual(runtime_context["query_source"], "main_query_plus_current_visual_cue")
        self.assertFalse(runtime_context["planned_augmented_queries_used"])

    def test_runtime_diagnostics_are_allowlisted_traced_and_restored_for_official_costs(self):
        for search_mode, num_pop in ((1, [0]), (2, [[6, 2]])):
            with self.subTest(search_mode=search_mode):
                def fake_cvsearch(**kwargs):
                    kwargs["annotation"].update({
                        "search_mode": search_mode,
                        "root_ans_conf": 0.25,
                        "num_pop": deepcopy(num_pop),
                        "num_zoom_in": [1],
                        "num_zoom_out": [0],
                        "answer": "runtime truth must not escape",
                        "secret": {"leak": True},
                    })
                    return 0

                response, trace = get_evidence_gap_response(
                    sam_model=object(), zoom_model=object(), nlp_model=object(),
                    policy_annotation=self.policy, original_annotation={}, ic_examples=[],
                    decomposed_question_template="{}", config=base_config(rerank_enabled=False),
                    cvsearch_fn=fake_cvsearch,
                )
                self.assertEqual(trace.method_mode, "rerank_only")
                self.assertEqual(trace.config_id, "minimal_v1")
                self.assertEqual(trace.effective_config["alpha"], 0.65)
                self.assertEqual(trace.cvsearch_search_mode, search_mode)
                self.assertEqual(trace.root_ans_conf, 0.25)
                self.assertEqual(trace.num_pop, num_pop)
                self.assertEqual(trace.num_zoom_in, [1])
                self.assertEqual(trace.num_zoom_out, [0])
                self.assertFalse(trace.budget_interrupted)
                self.assertEqual(trace.effective_ranking_query, "cvsearch_default_order")
                self.assertEqual(
                    trace.pixel_accounting,
                    "source_image_area_per_logical_forward_approximation",
                )
                record = compose_output_record({}, response, trace)
                self.assertEqual(record["search_mode"], search_mode)
                self.assertEqual(record["num_pop"], num_pop)
                self.assertEqual(record["num_zoom_in"], [1])
                self.assertEqual(record["num_zoom_out"], [0])
                self.assertNotIn("secret", record)
                self.assertNotIn("answer", record)

    def test_budget_failure_and_cvsearch_errors_propagate(self):
        class Zoom:
            def get_confidence_value(self, *args, **kwargs):
                return 0.0

        def budgeted_cvsearch(**kwargs):
            kwargs["zoom_model"].get_confidence_value(
                [], Image.new("RGB", (2, 2)), "answering", "q"
            )

        with self.assertRaises(BudgetExceeded):
            get_evidence_gap_response(
                sam_model=object(), zoom_model=Zoom(), nlp_model=object(),
                policy_annotation=self.policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", config=base_config(
                    rerank_enabled=False, max_mllm_calls=0, max_processed_pixels=0
                ), cvsearch_fn=budgeted_cvsearch,
            )

        with self.assertRaisesRegex(RuntimeError, "model failed"):
            get_evidence_gap_response(
                sam_model=object(), zoom_model=object(), nlp_model=object(),
                policy_annotation=self.policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", config=base_config(rerank_enabled=False),
                cvsearch_fn=lambda **_: (_ for _ in ()).throw(RuntimeError("model failed")),
            )

    def test_rerank_only_vstar_reserves_complete_answer_and_root_falls_back_at_boundary(self):
        class Zoom:
            def __init__(self):
                self.calls = []

            def get_confidence_value(self, *args, **kwargs):
                self.calls.append("search")
                return 0.0

            def multiple_choices_with_losses(self, image_pil, question, options, searched_nodes=None):
                self.calls.append(("root", tuple(searched_nodes)))
                return 0, [0.1, 0.9]

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            policy = dict(self.policy, input_image=str(image_path))
            model = Zoom()

            def exhaust_search_reserve(**kwargs):
                for _ in range(3):
                    kwargs["zoom_model"].get_confidence_value(
                        [], Image.new("RGB", (2, 2)), "answering", "q"
                    )

            response, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=model, nlp_model=object(),
                policy_annotation=policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", cvsearch_fn=exhaust_search_reserve,
                config=base_config(
                    rerank_enabled=False, max_mllm_calls=5, max_processed_pixels=20,
                ),
            )

        self.assertEqual(response, 0)
        self.assertEqual(model.calls, ["search", "search", ("root", ())])
        self.assertEqual((trace.budget.mllm_calls, trace.budget.processed_pixels), (5, 20))
        self.assertTrue(trace.budget_interrupted)
        self.assertEqual(trace.final_answer.selected_from, "root")
        self.assertEqual(trace.history[-1].answer.output, 0)
        self.assertEqual(trace.steps[-1].action, FORCED_RETURN)
        self.assertEqual(response, trace.final_answer.output)

    def test_rerank_only_hr_reserves_atomic_four_call_answer_at_boundary(self):
        blocks = [
            "A. cat\nB. dog", "A. dog\nB. cat",
            "A. cat\nB. dog", "A. dog\nB. cat",
        ]

        class Zoom:
            def __init__(self):
                self.calls = []
                self.outputs = iter(("A", "B", "A", "B"))

            def get_confidence_value(self, *args, **kwargs):
                self.calls.append("search")
                return 0.0

            def free_form_using_nodes(self, image_pil, question, searched_nodes):
                self.calls.append(("root", tuple(searched_nodes)))
                return next(self.outputs)

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            policy = {
                "question": "Which animal?", "options": blocks,
                "answer_type": "option_list", "input_image": str(image_path),
            }
            model = Zoom()

            def exhaust_search_reserve(**kwargs):
                for _ in range(3):
                    kwargs["zoom_model"].get_confidence_value(
                        [], Image.new("RGB", (2, 2)), "answering", "q"
                    )

            response, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=model, nlp_model=object(),
                policy_annotation=policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", cvsearch_fn=exhaust_search_reserve,
                config=base_config(
                    rerank_enabled=False, max_mllm_calls=6, max_processed_pixels=24,
                ),
            )

        self.assertEqual(response, ["A", "B", "A", "B"])
        self.assertEqual(model.calls[:2], ["search", "search"])
        self.assertEqual(len(model.calls), 6)
        self.assertTrue(all(call == ("root", ()) for call in model.calls[2:]))
        self.assertEqual((trace.budget.mllm_calls, trace.budget.processed_pixels), (6, 24))
        self.assertTrue(trace.budget_interrupted)
        self.assertEqual(trace.final_answer.selected_from, "root")
        self.assertEqual(response, trace.final_answer.output)

    def test_rerank_only_rejects_inadequate_answer_reserve_before_first_model_call(self):
        class Zoom:
            def __init__(self):
                self.calls = 0

            def get_confidence_value(self, *args, **kwargs):
                self.calls += 1
                return 0.0

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)

            def first_search_call(**kwargs):
                return kwargs["zoom_model"].get_confidence_value(
                    [], Image.new("RGB", (2, 2)), "answering", "q"
                )

            cases = [
                (dict(self.policy, input_image=str(image_path)), 2, 100),
                (dict(self.policy, input_image=str(image_path)), 3, 11),
                ({
                    "question": "Which animal?",
                    "options": ["A. cat\nB. dog", "A. dog\nB. cat"] * 2,
                    "answer_type": "option_list",
                    "input_image": str(image_path),
                }, 3, 100),
                ({
                    "question": "Which animal?",
                    "options": ["A. cat\nB. dog", "A. dog\nB. cat"] * 2,
                    "answer_type": "option_list",
                    "input_image": str(image_path),
                }, 4, 15),
            ]
            for policy, max_calls, max_pixels in cases:
                with self.subTest(answer_type=policy["answer_type"], calls=max_calls, pixels=max_pixels):
                    model = Zoom()
                    with self.assertRaises(BudgetExceeded):
                        get_evidence_gap_response(
                            sam_model=object(), zoom_model=model, nlp_model=object(),
                            policy_annotation=policy, original_annotation={}, ic_examples=[],
                            decomposed_question_template="{}", cvsearch_fn=first_search_call,
                            config=base_config(
                                rerank_enabled=False,
                                max_mllm_calls=max_calls,
                                max_processed_pixels=max_pixels,
                            ),
                        )
                    self.assertEqual(model.calls, 0)

    def test_budget_interrupt_returns_latest_observation_without_more_inference(self):
        calls = []

        def observed_then_interrupted(**kwargs):
            kwargs["answer_observer"]("quick", [], 0)
            kwargs["answer_observer"]("search", [], 1)
            calls.append("before-interrupt")
            raise BudgetExceeded("synthetic search boundary")

        response, trace = get_evidence_gap_response(
            sam_model=object(), zoom_model=object(), nlp_model=object(),
            policy_annotation=self.policy, original_annotation={}, ic_examples=[],
            decomposed_question_template="{}", cvsearch_fn=observed_then_interrupted,
            config=base_config(rerank_enabled=False),
        )
        self.assertEqual(calls, ["before-interrupt"])
        self.assertEqual(response, 1)
        self.assertEqual(trace.final_answer.output, 1)
        self.assertEqual(trace.final_answer.selected_from, "search")
        self.assertEqual(trace.history[-1].answer.output, 1)
        self.assertTrue(trace.budget_interrupted)

    def test_rerank_only_normal_vstar_boundary_preserves_call_order_and_real_ledger(self):
        class Zoom:
            def __init__(self):
                self.calls = []

            def get_confidence_value(self, *args, **kwargs):
                self.calls.append("search")
                return 0.0

            def multiple_choices_inference(self, image_pil, question, options, searched_nodes=None):
                self.calls.append(("answer", tuple(searched_nodes)))
                return 0

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            policy = dict(self.policy, input_image=str(image_path))
            model = Zoom()

            def complete_at_boundary(**kwargs):
                image = Image.new("RGB", (2, 2))
                kwargs["zoom_model"].get_confidence_value([], image, "answering", "q")
                kwargs["zoom_model"].get_confidence_value([], image, "answering", "q")
                raw = kwargs["zoom_model"].multiple_choices_inference(
                    image, "q", policy["options"], ["searched"]
                )
                kwargs["answer_observer"]("search", [], raw)
                return raw

            response, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=model, nlp_model=object(),
                policy_annotation=policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", cvsearch_fn=complete_at_boundary,
                config=base_config(
                    rerank_enabled=False, max_mllm_calls=5, max_processed_pixels=20,
                ),
            )
        self.assertEqual(response, 0)
        self.assertEqual(model.calls, ["search", "search", ("answer", ("searched",))])
        self.assertEqual((trace.budget.mllm_calls, trace.budget.processed_pixels), (5, 20))
        self.assertFalse(trace.budget_interrupted)

    def test_hr_rerank_only_duplicate_loser_is_traceable_and_raw_output_is_unchanged(self):
        blocks = [
            "A. Brown\nB. Red\nC. red\nD. Blue",
            "A. Blue\nB. Brown\nC. red\nD. Red",
            "A. Blue\nB. red\nC. Brown\nD. Red",
            "A. Red\nB. red\nC. Blue\nD. Brown",
        ]
        raw = ["A", "B", "C", "D"]
        policy = dict(self.policy, answer_type="option_list", options=blocks)

        def fake_cvsearch(**kwargs):
            kwargs["answer_observer"]("search", [], raw)
            return deepcopy(raw)

        response, trace = get_evidence_gap_response(
            sam_model=object(), zoom_model=object(), nlp_model=object(),
            policy_annotation=policy, original_annotation={}, ic_examples=[],
            decomposed_question_template="{}", cvsearch_fn=fake_cvsearch,
            config=base_config(rerank_enabled=False),
        )
        self.assertEqual(response, raw)
        self.assertEqual(trace.final_answer.output, raw)
        self.assertIs(trace.final_answer.aggregation_available, True)
        self.assertEqual(trace.final_answer.canonical_answer, "brown")

    def test_rerank_only_normal_hr_boundary_atomically_charges_four_terminal_calls(self):
        blocks = [
            "A. cat\nB. dog", "A. dog\nB. cat",
            "A. cat\nB. dog", "A. dog\nB. cat",
        ]

        class Zoom:
            def __init__(self):
                self.calls = []
                self.outputs = iter(("A", "B", "A", "B"))

            def get_confidence_value(self, *args, **kwargs):
                self.calls.append("search")
                return 0.0

            def free_form_using_nodes(self, image_pil, question, searched_nodes):
                self.calls.append(("answer", tuple(searched_nodes)))
                return next(self.outputs)

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            policy = {
                "question": "Which animal?", "options": blocks,
                "answer_type": "option_list", "input_image": str(image_path),
            }
            model = Zoom()

            def complete_at_boundary(**kwargs):
                image = Image.new("RGB", (2, 2))
                kwargs["zoom_model"].get_confidence_value([], image, "answering", "q")
                kwargs["zoom_model"].get_confidence_value([], image, "answering", "q")
                raw = [
                    kwargs["zoom_model"].free_form_using_nodes(image, "q", ["searched"])
                    for _ in range(4)
                ]
                kwargs["answer_observer"]("search", [], raw)
                return raw

            response, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=model, nlp_model=object(),
                policy_annotation=policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", cvsearch_fn=complete_at_boundary,
                config=base_config(
                    rerank_enabled=False, max_mllm_calls=6, max_processed_pixels=24,
                ),
            )
        self.assertEqual(response, ["A", "B", "A", "B"])
        self.assertEqual(model.calls[:2], ["search", "search"])
        self.assertEqual(model.calls[2:], [("answer", ("searched",))] * 4)
        self.assertEqual((trace.budget.mllm_calls, trace.budget.processed_pixels), (6, 24))
        self.assertFalse(trace.budget_interrupted)

    def test_hr_rerank_only_nonprojectable_winner_preserves_exact_raw_response(self):
        blocks = [
            "A. Red\nB. red\nC. Blue",
            "A. Blue\nB. red\nC. Green",
            "A. Green\nB. Blue\nC. red",
            "A. red\nB. Blue\nC. Green",
        ]
        raw = ["A", "B.", "C answer", "A"]
        policy = dict(self.policy, answer_type="option_list", options=blocks)

        def fake_cvsearch(**kwargs):
            kwargs["answer_observer"]("search", [], raw)
            return deepcopy(raw)

        response, trace = get_evidence_gap_response(
            sam_model=object(), zoom_model=object(), nlp_model=object(),
            policy_annotation=policy, original_annotation={}, ic_examples=[],
            decomposed_question_template="{}", cvsearch_fn=fake_cvsearch,
            config=base_config(rerank_enabled=False),
        )
        self.assertEqual(response, raw)
        self.assertEqual(trace.final_answer.output, raw)
        self.assertIs(trace.final_answer.aggregation_available, False)
        self.assertEqual(trace.final_answer.aggregation_reason, "ambiguous_winner_projection")

    def test_hr_root_fallback_disables_confidence_comparison_when_either_aggregation_is_unavailable(self):
        blocks = [
            "A. Red\nB. red\nC. Blue",
            "A. Blue\nB. red\nC. Green",
            "A. Green\nB. Blue\nC. red",
            "A. red\nB. Blue\nC. Green",
        ]
        cases = (
            (["A", "B", "C", "A"], ["C raw", "A", "B", "B"], False, True),
            (["C", "A", "B", "B"], ["A raw", "B", "C", "A"], True, False),
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            policy = {
                "question": "Which color?", "options": blocks,
                "answer_type": "option_list", "input_image": str(image_path),
            }
            for root_raw, search_raw, root_available, search_available in cases:
                with self.subTest(root=root_available, search=search_available):
                    class Zoom:
                        def __init__(self):
                            self.outputs = iter(root_raw)

                        def free_form_using_nodes(self, image_pil, question, searched_nodes):
                            return next(self.outputs)

                    def fake_cvsearch(**kwargs):
                        kwargs["answer_observer"]("search", [], search_raw)
                        return deepcopy(search_raw)

                    response, trace = get_evidence_gap_response(
                        sam_model=object(), zoom_model=Zoom(), nlp_model=object(),
                        policy_annotation=policy, original_annotation={}, ic_examples=[],
                        decomposed_question_template="{}", cvsearch_fn=fake_cvsearch,
                        config=base_config(mode="root_search_fallback", rerank_enabled=False),
                    )
                    self.assertEqual(response, search_raw)
                    self.assertEqual(trace.final_answer.output, search_raw)
                    self.assertEqual(trace.final_answer.selected_from, "search")
                    self.assertIs(trace.history[0].answer.aggregation_available, root_available)
                    self.assertIs(trace.history[1].answer.aggregation_available, search_available)
                    self.assertEqual(trace.history[1].answer.output, search_raw)

    def test_multiple_choice_budget_is_precharged_once_without_nested_double_count(self):
        class Zoom:
            def __init__(self):
                self.calls = 0

            def multiple_choices_inference(self, image_pil, question, options, searched_nodes=None):
                self.calls += 1
                return 0

        model = Zoom()

        def one_choice_call(**kwargs):
            image = Image.new("RGB", (2, 2))
            return kwargs["zoom_model"].multiple_choices_inference(
                image, "q", ["a", "b"], []
            )

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            policy = dict(self.policy, input_image=str(image_path))
            response, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=model, nlp_model=object(),
                policy_annotation=policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", cvsearch_fn=one_choice_call,
                config=base_config(
                    rerank_enabled=False, max_mllm_calls=3, max_processed_pixels=12
                ),
            )
        self.assertEqual(response, 0)
        self.assertEqual(model.calls, 1)
        self.assertEqual((trace.budget.mllm_calls, trace.budget.processed_pixels), (3, 12))
        self.assertEqual(
            trace.pixel_accounting,
            "source_image_area_per_logical_forward_approximation",
        )

        blocked = Zoom()
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (2, 2), "white").save(image_path)
            blocked_policy = dict(self.policy, input_image=str(image_path))
            with self.assertRaises(BudgetExceeded):
                get_evidence_gap_response(
                    sam_model=object(), zoom_model=blocked, nlp_model=object(),
                    policy_annotation=blocked_policy, original_annotation={}, ic_examples=[],
                    decomposed_question_template="{}", cvsearch_fn=one_choice_call,
                    config=base_config(
                        rerank_enabled=False, max_mllm_calls=3, max_processed_pixels=11
                    ),
                )
        self.assertEqual(blocked.calls, 0)

    def test_text_only_and_crop_calls_use_declared_source_area_approximation(self):
        class Zoom:
            def __init__(self):
                self.calls = []

            def generate_visual_cues_using_ic(self, examples, question):
                self.calls.append("text")
                return ["sign"]

            def get_confidence_value(self, nodes, image_pil, *args, **kwargs):
                self.calls.append(("visual", image_pil.size))
                return 0.0

        model = Zoom()

        def mixed_calls(**kwargs):
            kwargs["zoom_model"].generate_visual_cues_using_ic([], "q")
            kwargs["zoom_model"].get_confidence_value(
                [], Image.new("RGB", (3, 2)), "answering", "q"
            )
            return 0

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (3, 2), "white").save(image_path)
            policy = dict(self.policy, input_image=str(image_path))
            _, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=model, nlp_model=object(),
                policy_annotation=policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", cvsearch_fn=mixed_calls,
                config=base_config(
                    rerank_enabled=False, max_mllm_calls=5, max_processed_pixels=24
                ),
            )
        self.assertEqual(model.calls, ["text", ("visual", (3, 2))])
        self.assertEqual((trace.budget.mllm_calls, trace.budget.processed_pixels), (2, 6))
        self.assertEqual(
            trace.effective_config["pixel_accounting"],
            "source_image_area_per_logical_forward_approximation",
        )


class CliHelpersAndLauncherTest(unittest.TestCase):
    def test_ordinal_parser_accepts_commas_ranges_and_rejects_ambiguous_input(self):
        self.assertEqual(parse_ordinals("4,1-3,7"), (1, 2, 3, 4, 7))
        self.assertIsNone(parse_ordinals(None))
        for spec in ("", "1,,2", "3-1", "-1", "1.5", "1-2-3", "1,1"):
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError):
                    parse_ordinals(spec)

    def test_selection_uses_global_ordinals_and_source_grouped_split(self):
        rows = [
            {"input_image": "same.jpg", "question": "q0"},
            {"input_image": "other.jpg", "question": "q1"},
            {"input_image": "same.jpg", "question": "q2"},
        ]
        selected = select_annotations(rows, "hr-bench_4k", (0, 2), "all")
        self.assertEqual([(ordinal, row["question"]) for ordinal, row in selected], [(0, "q0"), (2, "q2")])
        bucket = split_bucket("hr-bench_4k", "same.jpg")
        same_source = select_annotations(rows, "hr-bench_4k", (0, 2), bucket)
        self.assertEqual([ordinal for ordinal, _ in same_source], [0, 2])
        opposite = "holdout" if bucket == "dev" else "dev"
        with self.assertRaises(ValueError):
            select_annotations(rows, "hr-bench_4k", (0, 2), opposite)
        with self.assertRaises(ValueError):
            select_annotations(rows, "hr-bench_4k", (99,), "all")
        for invalid in ((0, 0), (-1,), (True,)):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    select_annotations(rows, "hr-bench_4k", invalid, "all")

    def test_hr_benchmark_boundary_requires_exactly_four_blocks_and_outputs(self):
        for count in (0, 2, 3, 5):
            with self.subTest(count=count):
                policy = {"options": [f"A. value {index}" for index in range(count)]}
                with self.assertRaises(ValueError):
                    eg_cli._validate_output("hr-bench_4k", policy, ["A"] * count)
        eg_cli._validate_output(
            "hr-bench_8k",
            {"options": ["A. x"] * 4},
            ["A", "A", "A", "A"],
        )

    def test_parser_requires_explicit_artifact_paths(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        parsed = parser.parse_args([
            "--root-path", "/root", "--model-path", "/model", "--annotation-path", "/data",
            "--benchmark", "vstar", "--answers-file", "/answers.jsonl", "--config", "minimal_v1",
        ])
        self.assertEqual(parsed.split, "all")
        self.assertFalse(parsed.resume)

    def test_cli_imports_as_script_and_launcher_fails_closed(self):
        cli = ROOT / "cvsearch" / "perform_EGSearch.py"
        help_result = subprocess.run(
            [sys.executable, str(cli), "--help"], cwd=ROOT,
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)

        launcher = ROOT / "cvsearch" / "run_eval_evidence_gap.sh"
        syntax = subprocess.run(["bash", "-n", str(launcher)], text=True, capture_output=True, check=False)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        missing = subprocess.run(["bash", str(launcher)], cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertNotEqual(missing.returncode, 0)
        source = launcher.read_text(encoding="utf-8")
        self.assertEqual(source.splitlines()[:2], ["#!/usr/bin/env bash", "set -euo pipefail"])
        self.assertNotIn("eval ", source)

    def test_cli_closes_writer_when_runtime_loading_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Qwen-model").mkdir()
            (root / "data" / "vstar").mkdir(parents=True)
            (root / "data" / "vstar" / "annotation_vstar.json").write_text(
                json.dumps([{
                    "question": "q", "options": ["a", "b"], "answer_type": "logits_match",
                    "input_image": "missing.jpg", "test_type": "x",
                }]), encoding="utf-8",
            )
            (root / "sam.pt").write_bytes(b"sam")
            (root / "nlp").mkdir()
            config = root / "config.json"
            config.write_text(json.dumps(base_config(rerank_enabled=False)), encoding="utf-8")
            instances = []

            class TrackingWriter(eg_cli.JsonlCheckpointWriter):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.was_closed = False
                    instances.append(self)

                def close(self):
                    self.was_closed = True
                    return super().close()

            argv = [
                "--root-path", str(root), "--model-path", "Qwen-model",
                "--annotation-path", "data", "--sam-model-path", "sam.pt",
                "--nlp-model-path", "nlp", "--benchmark", "vstar",
                "--answers-file", str(root / "answers.jsonl"), "--config", str(config),
            ]
            with patch.object(eg_cli, "JsonlCheckpointWriter", TrackingWriter), patch.object(
                eg_cli, "_load_runtime", side_effect=RuntimeError("load failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "load failed"):
                    eg_cli.main(argv)
            self.assertEqual(len(instances), 1)
            self.assertTrue(instances[0].was_closed)

    def test_cli_resume_invokes_method_only_for_missing_global_ordinals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Qwen-model").mkdir()
            (root / "data" / "vstar").mkdir(parents=True)
            rows = [{
                "question": f"q{index}", "options": ["a", "b"],
                "answer_type": "logits_match", "input_image": f"{index}.jpg", "test_type": "x",
            } for index in range(3)]
            (root / "data" / "vstar" / "annotation_vstar.json").write_text(
                json.dumps(rows), encoding="utf-8",
            )
            (root / "sam.pt").write_bytes(b"sam")
            (root / "nlp").mkdir()
            config = root / "config.json"
            config.write_text(json.dumps(base_config(rerank_enabled=False)), encoding="utf-8")
            answers = root / "answers.jsonl"
            argv = [
                "--root-path", str(root), "--model-path", "Qwen-model",
                "--annotation-path", "data", "--sam-model-path", "sam.pt",
                "--nlp-model-path", "nlp", "--benchmark", "vstar",
                "--answers-file", str(answers), "--config", str(config),
            ]
            first_seen = []

            def crash_after_one(**kwargs):
                image = kwargs["policy_annotation"]["input_image"]
                first_seen.append(image)
                if len(first_seen) == 2:
                    raise RuntimeError("synthetic crash")
                return 0, MethodTrace(termination=FORCED_RETURN)

            fake_runtime = (object(), object(), object(), None, object())
            with patch.object(eg_cli, "_load_runtime", return_value=fake_runtime), patch.object(
                eg_cli, "get_evidence_gap_response", side_effect=crash_after_one
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic crash"):
                    eg_cli.main(argv)
            self.assertEqual(first_seen, ["0.jpg", "1.jpg"])
            self.assertFalse(answers.exists())

            resumed_seen = []

            def finish(**kwargs):
                resumed_seen.append(kwargs["policy_annotation"]["input_image"])
                return 0, MethodTrace(termination=FORCED_RETURN)

            with patch.object(eg_cli, "_load_runtime", return_value=fake_runtime), patch.object(
                eg_cli, "get_evidence_gap_response", side_effect=finish
            ):
                self.assertEqual(eg_cli.main(argv + ["--resume"]), 0)
            self.assertEqual(resumed_seen, ["1.jpg", "2.jpg"])
            written = [json.loads(line) for line in answers.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["_eg_ordinal"] for row in written], [0, 1, 2])

    def test_launcher_forwards_split_seed_without_losing_exit_status(self):
        launcher = ROOT / "cvsearch" / "run_eval_evidence_gap.sh"
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            fake_python = temporary / "fake-python"
            capture = temporary / "argv.txt"
            fake_python.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$@" >"$CAPTURE"\nexit "${FAKE_EXIT:-0}"\n',
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = dict(os.environ, PYTHON_BIN=str(fake_python), CAPTURE=str(capture))
            result = subprocess.run([
                "bash", str(launcher),
                "--root-path", "/root", "--model-path", "/model",
                "--annotation-path", "/data", "--sam-model-path", "/sam",
                "--nlp-model-path", "/nlp", "--clip-model-path", "/clip",
                "--benchmark", "vstar", "--gpu", "0",
                "--answers-file", str(temporary / "answers.jsonl"),
                "--log-file", str(temporary / "run.log"), "--config", "minimal_v1",
                "--mode", "rerank_only", "--split", "dev", "--ordinals", "1,4-5",
                "--split-seed", "7", "--num-chunks", "3", "--chunk-idx", "1", "--force",
            ], cwd=ROOT, env=env, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            forwarded = capture.read_text(encoding="utf-8").splitlines()
            index = forwarded.index("--split-seed")
            self.assertEqual(forwarded[index + 1], "7")
            expected_pairs = {
                "--root-path": "/root", "--model-path": "/model",
                "--annotation-path": "/data", "--sam-model-path": "/sam",
                "--nlp-model-path": "/nlp", "--clip-model-path": "/clip",
                "--benchmark": "vstar", "--answers-file": str(temporary / "answers.jsonl"),
                "--config": "minimal_v1", "--mode": "rerank_only", "--split": "dev",
                "--ordinals": "1,4-5", "--num-chunks": "3", "--chunk-idx": "1",
            }
            for option, value in expected_pairs.items():
                with self.subTest(option=option):
                    position = forwarded.index(option)
                    self.assertEqual(forwarded[position + 1], value)
            self.assertIn("--force", forwarded)
            log = (temporary / "run.log").read_text(encoding="utf-8")
            self.assertIn("CUDA_VISIBLE_DEVICES=0", log)
            self.assertIn(f"python={fake_python}", log)
            self.assertIn("resume=0 force=1", log)
            self.assertIn("argv=", log)
            for value in ("/data", "/sam", "/nlp", "/clip", "--split-seed", "--force"):
                self.assertIn(value, log)

            failed_env = dict(env, FAKE_EXIT="7")
            failed = subprocess.run([
                "bash", str(launcher),
                "--root-path", "/root", "--model-path", "/model",
                "--annotation-path", "/data", "--sam-model-path", "/sam",
                "--nlp-model-path", "/nlp", "--clip-model-path", "/clip",
                "--benchmark", "vstar", "--gpu", "0",
                "--answers-file", str(temporary / "answers-2.jsonl"),
                "--log-file", str(temporary / "run-2.log"), "--config", "minimal_v1",
            ], cwd=ROOT, env=failed_env, text=True, capture_output=True, check=False)
            self.assertEqual(failed.returncode, 7)

    def test_launcher_rejects_answer_log_alias_before_writing(self):
        launcher = ROOT / "cvsearch" / "run_eval_evidence_gap.sh"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "same.jsonl"
            result = subprocess.run([
                "bash", str(launcher),
                "--root-path", "/root", "--model-path", "/model",
                "--annotation-path", "/data", "--sam-model-path", "/sam",
                "--nlp-model-path", "/nlp", "--clip-model-path", "/clip",
                "--benchmark", "vstar", "--gpu", "0",
                "--answers-file", str(target), "--log-file", str(target),
                "--config", "minimal_v1",
            ], cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
