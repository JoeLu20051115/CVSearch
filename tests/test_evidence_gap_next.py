import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from cvsearch.evidence_gap.method import (
    MINIMAL_V1,
    NEXT_CONFIG_KEYS,
    get_evidence_gap_response,
    load_method_config,
)
from cvsearch.evidence_gap.types import (
    BudgetExceeded,
    BudgetLedger,
    FORCED_RETURN,
    NEXT,
    QueryPlan,
)
from tests.test_evidence_gap_search_state import candidate_snapshot, event_for
from tests.test_evidence_gap_support import RawObservationModel


SUPPORT_PROMPT_VERSION = "qwen_answer_free_evidence_support_v1"
SUPPORT_PROCESSOR_MODE = "single_rendered_view_chat_left_padding_final_yes_no_logits"
SUPPORT_TEMPLATE_SHA256 = (
    "0ba54d5ef5190f162d2036721693ae1af1a499f088e12a5106f8c9b692dfcb86"
)
SUPPORT_TRANSFORM = (
    "v1:p_yes=softmax(final_position_two_logits[Yes,No],dim=-1,"
    "preserve_model_dtype,no_float32_cast)[0];no_legacy_2p_minus_1"
)


def legacy_config(**overrides):
    config = deepcopy(MINIMAL_V1)
    config.update(overrides)
    return config


def next_extension(*, enabled=True, **overrides):
    extension = {
        "next_enabled": enabled,
        "next_admission_mode": "all_feasible" if enabled else "disabled",
        "next_replacement_enabled": False,
        "evidence_support_prompt_version": SUPPORT_PROMPT_VERSION,
        "evidence_support_processor_mode": SUPPORT_PROCESSOR_MODE,
        "evidence_support_prompt_template_sha256": SUPPORT_TEMPLATE_SHA256,
        "evidence_support_probability_transform": SUPPORT_TRANSFORM,
    }
    extension.update(overrides)
    return extension


def enabled_config(**overrides):
    config = legacy_config(
        config_id="unified-next-oracle-gamma000-v1",
        mode="root_search_fallback",
        rerank_enabled=False,
        ranking_mode="cvsearch",
        ranking_rho=0.0,
        ranking_max_displacement=0,
        quick_gate=0.6,
        root_fallback_tolerance=0.05,
        enable_zoom=False,
        enable_split=False,
        enable_expand=False,
        enable_certified_stop=False,
        hr_fusion_mode="off",
        hr_fusion_gamma=0.0,
        max_mllm_calls=512,
        pixel_accounting="source_image_area_per_logical_forward_approximation",
    )
    config.update(next_extension())
    config.update(overrides)
    return config


class NextConfigTest(unittest.TestCase):
    def test_next_extension_is_strict_all_or_none_and_legacy_has_no_defaults(self):
        legacy = load_method_config(legacy_config())
        self.assertTrue(set(NEXT_CONFIG_KEYS).isdisjoint(legacy))
        for missing in NEXT_CONFIG_KEYS:
            with self.subTest(missing=missing):
                partial = legacy_config(**next_extension())
                partial.pop(missing)
                with self.assertRaisesRegex(ValueError, "all-or-none"):
                    load_method_config(partial)

    def test_enabled_p2a_accepts_only_the_frozen_unified_observation_contract(self):
        loaded = load_method_config(enabled_config())
        self.assertEqual(
            {key: loaded[key] for key in NEXT_CONFIG_KEYS}, next_extension()
        )
        invalid = (
            {"next_admission_mode": "threshold"},
            {"next_replacement_enabled": True},
            {"mode": "rerank_only"},
            {"quick_gate": 0.8},
            {"root_fallback_tolerance": 0.1},
            {"rerank_enabled": True, "ranking_mode": "query_linear"},
            {"enable_zoom": True},
            {"enable_split": True},
            {"enable_expand": True},
            {"enable_certified_stop": True},
            {"hr_fusion_mode": "global_soft", "hr_fusion_gamma": 2.1},
            {"max_mllm_calls": 511},
            {"max_processed_pixels": 9_999_999_999},
            {"max_processed_pixels": 10_000_000_001},
            {"max_processed_pixels": 0},
            {"evidence_support_prompt_version": "different"},
            {"evidence_support_processor_mode": "different"},
            {"evidence_support_prompt_template_sha256": "0" * 64},
            {"evidence_support_probability_transform": "different"},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                load_method_config(enabled_config(**changes))

    def test_explicitly_disabled_next_does_not_construct_source_or_install_sink(self):
        policy = {
            "question": "Which sign is visible?",
            "options": ["red", "blue"],
            "answer_type": "logits_match",
            "input_image": "/path/that/must/not/be/opened.png",
        }

        def fake_cvsearch(**kwargs):
            self.assertIsNone(kwargs["search_state_sink"])
            return 0

        config = legacy_config(**next_extension(enabled=False))
        output, trace = get_evidence_gap_response(
            sam_model=object(), zoom_model=object(), nlp_model=object(),
            policy_annotation=policy, original_annotation={}, ic_examples=[],
            decomposed_question_template="{}", config=config,
            cvsearch_fn=fake_cvsearch,
        )
        self.assertEqual(output, 0)
        self.assertEqual(trace.steps[-1].action, "FORCED_RETURN")
        self.assertFalse(any(step.action == "NEXT" for step in trace.steps))
        self.assertEqual(trace.anchor_state_score, 0.0)
        self.assertEqual(trace.selected_state_score, 0.0)
        self.assertEqual(trace.replacement_margin, 0.0)

    def test_legacy_trace_matches_the_frozen_pre_next_payload(self):
        policy = {
            "question": "Which sign is visible?",
            "options": ["red", "blue"],
            "answer_type": "logits_match",
            "input_image": "/path/that/must/not/be/opened.png",
        }

        with patch("cvsearch.evidence_gap.method.time.perf_counter", side_effect=(1.0, 2.0)):
            output, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=object(), nlp_model=object(),
                policy_annotation=policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", config=legacy_config(),
                cvsearch_fn=lambda **kwargs: (
                    self.assertIsNone(kwargs["search_state_sink"]) or 0
                ),
            )
        answer = {
            "output": 0, "canonical_answer": 0, "raw_outputs": [], "groups": {},
            "frequency": 0.0, "margin": 0.0, "confidence": 0.0,
            "uncertainty": 1.0, "losses": [], "selected_from": "response",
            "aggregation_available": None, "aggregation_reason": None,
        }
        budget = {
            "max_mllm_calls": 256, "max_processed_pixels": 10_000_000_000,
            "mllm_calls": 0, "processed_pixels": 0,
        }
        expected = {
            "query_plan": {
                "main_query": "Which sign is visible?", "targets": [],
                "augmented_queries": [], "evidence_items": [{
                    "kind": "question_evidence", "requirement": "visual_detail",
                }], "global_scope_required": False, "fallback_used": True,
            },
            "candidate_ranks": [],
            "steps": [{
                "step": 0, "action": FORCED_RETURN, "gap_fallback_used": False,
                "elapsed_seconds": 0.0, "focus_key": None, "feasible_actions": [],
                "gaps": {}, "no_op_reason": "minimal_v1 has no certified-stop controller",
                "answer": answer, "support_avg": 0.0, "support_min": 0.0,
                "certified": False, "budget": budget,
            }],
            "history": [], "final_answer": answer, "budget": budget,
            "elapsed_seconds": 1.0, "termination": FORCED_RETURN, "final_boxes": [],
            "method_mode": "rerank_only", "config_id": "minimal_v1",
            "effective_config": legacy_config(), "cvsearch_search_mode": None,
            "root_ans_conf": None, "num_pop": [], "num_zoom_in": [],
            "num_zoom_out": [], "budget_interrupted": False,
            "effective_ranking_query": "cvsearch_default_order",
            "pixel_accounting": "source_image_area_per_logical_forward_approximation",
            "anchor_answer": answer, "anchor_state_score": 0.0,
            "selected_state_score": 0.0, "replacement_margin": 0.0,
            "support_status": "not_observed",
        }
        self.assertEqual(output, 0)
        self.assertEqual(trace.to_dict(), expected)
        encoded = json.dumps(trace.to_dict(), sort_keys=True)
        self.assertNotIn("next_audit", encoded)
        for key in NEXT_CONFIG_KEYS:
            self.assertNotIn(key, trace.effective_config)


class IntegrationRaw(RawObservationModel):
    def __init__(self, *, fail_phase=None):
        super().__init__(fail_phase=fail_phase)
        self.root_hr_outputs = iter(("A", "B", "A", "B"))
        self.candidate_hr_outputs = iter(("candidate-A", "candidate-B", "candidate-C", "candidate-D"))
        self.candidate_answer_calls = 0

    def free_form_using_nodes(self, image, question, nodes):
        if not nodes:
            return next(self.root_hr_outputs)
        self.candidate_answer_calls += 1
        phase = f"hr_answer_{self.candidate_answer_calls - 1}"
        if self.fail_phase == phase:
            raise RuntimeError(f"failed {phase}")
        return next(self.candidate_hr_outputs)

    def multiple_choices_with_losses(self, image, question, options, nodes):
        if not nodes:
            return 0, [0.1, 0.9][:len(options)]
        self.candidate_answer_calls += 1
        if self.fail_phase == "vstar_answer":
            raise RuntimeError("failed vstar_answer")
        losses = [0.8, 0.1, 0.9, 1.0][:len(options)]
        return min(range(len(losses)), key=losses.__getitem__), losses


class UnifiedNextRuntimeTest(unittest.TestCase):
    HR_OPTIONS = [
        "A. cat\nB. dog\nC. bird\nD. fish",
        "A. dog\nB. cat\nC. fish\nD. bird",
        "A. cat\nB. dog\nC. fish\nD. bird",
        "A. dog\nB. cat\nC. bird\nD. fish",
    ]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image_path = Path(self.directory.name) / "source.png"
        image = Image.new("RGB", (8, 6))
        for y in range(image.height):
            for x in range(image.width):
                image.putpixel((x, y), (x * 20, y * 30, x + y))
        image.save(self.image_path)

    def _policy(self, answer_type):
        return {
            "question": "What is visible?",
            "options": self.HR_OPTIONS if answer_type == "option_list" else ["cat", "dog"],
            "answer_type": answer_type,
            "input_image": str(self.image_path),
        }

    @staticmethod
    def _emit_states(sink, image, *, include_p0=True, include_candidate=True,
                     current_source="fast", candidate_source="fine"):
        current = candidate_snapshot(
            (0, 0, 3, 3), posterior=0.95, depth=1, source=current_source,
        )
        candidate = candidate_snapshot(
            (4, 1, 3, 3), posterior=0.8, depth=1, source=candidate_source,
        )
        if include_candidate:
            refs, snapshot = event_for(image, [candidate], event="stage_ready")
            sink(refs, snapshot)
        if include_p0:
            refs, snapshot = event_for(
                image, [current], event="p0_selected",
                selected=(current["canonical_key"],), remaining=(), ordinal=0,
            )
            sink(refs, snapshot)
        return current, candidate

    def _run(self, *, answer_type="option_list", raw=None, planner=None,
             include_p0=True, include_candidate=True,
             budget_reject=False, current_source="fast", candidate_source="fine",
             observation_phase="search", ambiguous_p0=False,
             cvsearch_interrupt=False, omit_observer=False,
             observed_raw=None, observed_bbox=None, observed_depth=1,
             runtime_search_mode=None):
        raw = IntegrationRaw() if raw is None else raw
        policy = self._policy(answer_type)
        raw_response = ["A", "B", "A", "B"] if answer_type == "option_list" else 1

        def fake_cvsearch(**kwargs):
            self.assertIsNotNone(kwargs["search_state_sink"])
            with Image.open(self.image_path) as opened:
                image = opened.convert("RGB")
            current, _ = self._emit_states(
                kwargs["search_state_sink"], image,
                include_p0=include_p0, include_candidate=include_candidate,
                current_source=current_source, candidate_source=candidate_source,
            )
            if ambiguous_p0 and include_p0:
                refs, snapshot = event_for(
                    image, [current], event="p0_selected",
                    selected=(current["canonical_key"],), remaining=(), ordinal=0,
                )
                kwargs["search_state_sink"](refs, snapshot)
            if budget_reject:
                kwargs["zoom_model"]._ledger.consume("mllm_calls", 503)
            kwargs["annotation"]["search_mode"] = (
                runtime_search_mode if runtime_search_mode is not None
                else (0 if observation_phase == "quick" else 1)
            )
            node = current if include_p0 else candidate_snapshot(
                (0, 0, 3, 3), posterior=0.95, depth=1, source=current_source,
            )
            live_node = type("LiveNode", (), {})()
            live_node.depth = observed_depth
            live_node.state = type("State", (), {
                "bbox": node["bbox_original"] if observed_bbox is None else observed_bbox,
                "original_image_pil": image.copy(),
            })()
            if not omit_observer:
                kwargs["answer_observer"](
                    observation_phase, [live_node], deepcopy(
                        raw_response if observed_raw is None else observed_raw
                    )
                )
            if cvsearch_interrupt:
                raise BudgetExceeded("interrupted after the last complete answer")
            return deepcopy(raw_response)

        output, trace = get_evidence_gap_response(
            sam_model=object(), zoom_model=raw, nlp_model=object(),
            policy_annotation=policy, original_annotation={}, ic_examples=[],
            decomposed_question_template="{}", config=enabled_config(),
            cvsearch_fn=fake_cvsearch,
            planner=(planner if planner is not None else (
                lambda policy_snapshot, targets: QueryPlan(
                    main_query=policy_snapshot["question"],
                    targets=("object",),
                    evidence_items=({
                        "kind": "target_detail", "target": "object",
                        "requirements": ["presence", "visual_detail"],
                    },),
                )
            )),
        )
        return output, trace, raw, raw_response

    def test_hr_next_preserves_exact_p0_and_records_four_raw_candidate_shuffles(self):
        with patch(
            "cvsearch.evidence_gap.method._audit_score",
            side_effect=AssertionError("enabled NEXT called the legacy selector score"),
        ):
            output, trace, raw, p0 = self._run(answer_type="option_list")

        self.assertEqual(output, p0)
        self.assertEqual(trace.final_answer.output, p0)
        next_step = next(step for step in trace.steps if step.action == NEXT)
        audit = next_step.next_audit.to_dict()
        self.assertEqual(audit["p0_anchor"]["emitted_answer"], p0)
        self.assertEqual(audit["p0_anchor"]["cvsearch_raw"], p0)
        self.assertEqual(audit["p0_stability"]["raw_outputs"], p0)
        self.assertEqual(
            audit["batch_result"]["candidate_answer"],
            ["candidate-A", "candidate-B", "candidate-C", "candidate-D"],
        )
        self.assertEqual(audit["candidate_stability"]["raw_outputs"],
                         ["candidate-A", "candidate-B", "candidate-C", "candidate-D"])
        self.assertEqual(audit["replacement_reason"], "replacement_disabled_p2a")
        self.assertEqual(audit["score_margin"], None)
        self.assertEqual(audit["score_status"], "unavailable_missing_verifier_coverage")
        self.assertEqual(audit["verifier_status"],
                         "disabled_same_checkpoint_unpromoted")
        self.assertEqual(audit["coverage_status"], "not_observed")
        self.assertEqual(audit["normalized_actual_cost"], 6 / 512)
        self.assertIsNone(trace.anchor_state_score)
        self.assertIsNone(trace.selected_state_score)
        self.assertIsNone(trace.replacement_margin)
        self.assertEqual(next_step.feasible_actions, (NEXT,))
        self.assertEqual(trace.termination, FORCED_RETURN)
        self.assertEqual(trace.steps[-1].action, FORCED_RETURN)
        json.dumps(trace.to_dict(), allow_nan=False)

    def test_vstar_search_candidate_keeps_finite_losses_and_exact_argmin_winner(self):
        output, trace, raw, p0 = self._run(answer_type="logits_match")

        self.assertEqual(output, p0)
        audit = next(step for step in trace.steps if step.action == NEXT).next_audit.to_dict()
        self.assertEqual(audit["batch_result"]["candidate_answer"],
                         {"winner": 1, "losses": [0.8, 0.1]})
        self.assertEqual(audit["candidate_stability"]["output"], 1)
        self.assertEqual(audit["candidate_stability"]["losses"], [0.8, 0.1])

    def test_exact_p0_producer_matrix_keeps_hr_quick_search_and_vstar_root_search_views(self):
        for phase in ("quick", "search"):
            with self.subTest(answer_type="option_list", phase=phase):
                _, trace, _, _ = self._run(
                    answer_type="option_list", observation_phase=phase,
                )
                audit = next(
                    step.next_audit for step in trace.steps if step.action == NEXT
                ).to_dict()
                self.assertEqual(audit["p0_anchor"]["producing_phase"], "cvsearch_raw")
                self.assertEqual(len(audit["current_keys"]), 1)
                self.assertEqual(
                    audit["p0_anchor"]["support_view"][0]["canonical_key"],
                    audit["current_keys"][0],
                )

        _, search_trace, _, _ = self._run(answer_type="logits_match")
        search_audit = next(
            step.next_audit for step in search_trace.steps if step.action == NEXT
        ).to_dict()
        self.assertEqual(search_audit["p0_anchor"]["producing_phase"], "search")
        self.assertEqual(len(search_audit["current_keys"]), 1)

        class RootWinningRaw(IntegrationRaw):
            def multiple_choices_with_losses(self, image, question, options, nodes):
                if not nodes:
                    return 0, [0.0, 1.0]
                return super().multiple_choices_with_losses(
                    image, question, options, nodes
                )

        _, root_trace, _, _ = self._run(
            answer_type="logits_match", raw=RootWinningRaw(),
        )
        root_audit = next(
            step.next_audit for step in root_trace.steps if step.action == NEXT
        ).to_dict()
        self.assertEqual(root_audit["p0_anchor"]["producing_phase"], "root")
        self.assertEqual(root_audit["current_keys"], [])
        self.assertEqual(root_audit["p0_anchor"]["support_view"], [])
        self.assertEqual(
            root_audit["batch_result"]["batch_plan"]["current_observation"][
                "canonical_keys"
            ],
            [],
        )

    def test_zero_requirements_missing_view_and_no_candidate_do_not_consume_next(self):
        cases = (
            ("zero_requirements", lambda policy, targets: QueryPlan(
                main_query=policy["question"], evidence_items=(),
            ), True, True, "next_no_evidence_requirements"),
            ("missing_view", None, False, True, "next_p0_support_view_unavailable"),
            ("no_candidate", None, True, False, "next_all_candidates_visited"),
        )
        for name, planner, include_p0, include_candidate, reason in cases:
            with self.subTest(name=name):
                output, trace, raw, p0 = self._run(
                    planner=planner, include_p0=include_p0,
                    include_candidate=include_candidate,
                )
                self.assertEqual(output, p0)
                next_step = next(step for step in trace.steps if step.action == NEXT)
                self.assertEqual(next_step.no_op_reason, reason)
                self.assertFalse(next_step.next_audit.feasible)
                self.assertIsNone(next_step.next_audit.batch_result)
                self.assertEqual(raw.support.model.calls, 0)
                self.assertEqual(raw.candidate_answer_calls, 0)

    def test_budget_support_and_answer_failures_remain_charged_or_rejected_and_restore_p0(self):
        cases = (
            ("budget", IntegrationRaw(), True, "budget_rejected"),
            ("support", IntegrationRaw(fail_phase="current_support"), False, "model_failed"),
            ("answer", IntegrationRaw(fail_phase="hr_answer_2"), False, "model_failed"),
        )
        for name, raw, reject, status in cases:
            with self.subTest(name=name):
                output, trace, _, p0 = self._run(raw=raw, budget_reject=reject)
                self.assertEqual(output, p0)
                self.assertEqual(trace.final_answer.output, p0)
                audit = next(step for step in trace.steps if step.action == NEXT).next_audit
                self.assertEqual(audit.batch_result.status, status)
                self.assertIsNone(audit.g_next)
                self.assertIsNone(audit.support_delta)
                self.assertIsNone(audit.replacement_reason)
                if status == "model_failed":
                    self.assertTrue(audit.batch_result.charged)

    def test_same_renderer_next_noop_does_not_execute_atomic_batch(self):
        output, trace, raw, p0 = self._run(
            current_source="global", candidate_source="global",
        )
        self.assertEqual(output, p0)
        next_step = next(step for step in trace.steps if step.action == NEXT)
        self.assertEqual(next_step.no_op_reason, "next_all_observations_visited")
        self.assertIsNone(next_step.next_audit.batch_result)
        self.assertEqual(raw.support.model.calls, 0)
        self.assertEqual(raw.candidate_answer_calls, 0)

    def test_budget_interruption_and_ambiguous_p0_event_fail_closed_without_batch(self):
        cases = (
            ("interrupted", {"cvsearch_interrupt": True}),
            ("ambiguous", {"ambiguous_p0": True}),
        )
        for name, arguments in cases:
            with self.subTest(name=name):
                output, trace, raw, p0 = self._run(**arguments)
                self.assertEqual(output, p0)
                audit_step = next(step for step in trace.steps if step.action == NEXT)
                self.assertEqual(
                    audit_step.no_op_reason, "next_p0_support_view_unavailable"
                )
                self.assertIsNone(audit_step.next_audit.batch_result)
                self.assertEqual(raw.support.model.calls, 0)
                self.assertEqual(raw.candidate_answer_calls, 0)

    def test_missing_or_mismatched_answer_producer_fails_before_next_candidate(self):
        cases = (
            ("missing", {"omit_observer": True}),
            ("raw", {"observed_raw": ["D", "D", "D", "D"]}),
            ("bbox", {"observed_bbox": [1, 0, 3, 3]}),
            ("depth", {"observed_depth": 2}),
            ("phase", {"observation_phase": "quick", "runtime_search_mode": 1}),
        )
        for name, arguments in cases:
            with self.subTest(name=name):
                output, trace, raw, p0 = self._run(**arguments)
                self.assertEqual(output, p0)
                step = next(step for step in trace.steps if step.action == NEXT)
                self.assertEqual(step.no_op_reason, "next_p0_support_view_unavailable")
                self.assertIsNone(step.next_audit.batch_result)
                self.assertEqual(raw.support.model.calls, 0)
                self.assertEqual(raw.candidate_answer_calls, 0)

    def test_runtime_support_contract_mismatch_is_charged_but_never_measured_or_promoted(self):
        class MismatchedSupportRaw(IntegrationRaw):
            def _prepare_evidence_support(self, question, requirements):
                prepared = super()._prepare_evidence_support(question, requirements)
                prepared["prompt_version"] = "different_answer_free_prompt_v2"
                return prepared

            def evidence_support(self, **kwargs):
                result = super().evidence_support(**kwargs)
                return replace(
                    result, prompt_version="different_answer_free_prompt_v2"
                )

        output, trace, raw, p0 = self._run(raw=MismatchedSupportRaw())
        self.assertEqual(output, p0)
        step = next(step for step in trace.steps if step.action == NEXT)
        audit = step.next_audit
        self.assertEqual(audit.batch_result.status, "success")
        self.assertTrue(audit.batch_result.charged)
        self.assertEqual(audit.support_contract_status, "mismatch")
        self.assertEqual(step.no_op_reason, "next_support_contract_mismatch")
        self.assertIsNone(audit.g_next)
        self.assertIsNone(audit.support_delta)
        self.assertIsNone(audit.candidate_stability)
        self.assertIsNone(audit.replacement_reason)

    def test_strict_next_audit_rejects_fabricated_keys_cost_support_material_and_action(self):
        _, trace, _, _ = self._run()
        step = next(step for step in trace.steps if step.action == NEXT)
        audit = step.next_audit
        corrupt_stability = deepcopy(audit.candidate_stability)
        corrupt_stability.raw_outputs = ("forged",) * 4
        forged_fields = deepcopy(audit.candidate_stability)
        forged_fields.groups = {"forged": {"count": 4}}
        forged_fields.confidence = 0.123
        forged_fields.uncertainty = 0.877
        invalid_audits = (
            {"candidate_keys": ("forged-key",)},
            {"normalized_actual_cost": 0.0},
            {"g_next": min(1.0, audit.g_next + 0.1)},
            {"candidate_stability": corrupt_stability},
            {"candidate_stability": forged_fields},
            {"replacement_reason": "promoted"},
        )
        for changes in invalid_audits:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(audit, **changes)
        with self.assertRaisesRegex(ValueError, "action=NEXT"):
            replace(step, action=FORCED_RETURN)
        with self.assertRaisesRegex(ValueError, "focus"):
            replace(step, focus_key="forged-key")
        with self.assertRaisesRegex(ValueError, "batch ledger"):
            replace(
                step,
                budget=BudgetLedger(
                    max_mllm_calls=512,
                    max_processed_pixels=10_000_000_000,
                    mllm_calls=0,
                    processed_pixels=0,
                ),
            )

        _, no_batch_trace, _, _ = self._run(include_candidate=False)
        no_batch_step = next(
            item for item in no_batch_trace.steps if item.action == NEXT
        )
        with self.assertRaisesRegex(ValueError, "not_observed"):
            replace(
                no_batch_step.next_audit,
                support_contract_status="matched",
            )
        with self.assertRaisesRegex(ValueError, "no-op"):
            replace(no_batch_step, no_op_reason="forged_reason")

        audit.candidate_stability.groups["post_constructed"] = {"count": 99}
        with self.assertRaisesRegex(ValueError, "mutated"):
            audit.to_dict()

    def test_hr_p0_stability_is_canonical_even_without_a_candidate_batch(self):
        for include_candidate in (True, False):
            with self.subTest(include_candidate=include_candidate):
                _, trace, _, _ = self._run(include_candidate=include_candidate)
                audit = next(
                    item for item in trace.steps if item.action == NEXT
                ).next_audit
                forged = deepcopy(audit.p0_stability)
                forged.groups = {"forged": {"count": 4}}
                forged.frequency = 0.25
                forged.confidence = 0.25
                with self.assertRaisesRegex(ValueError, "P0 stability"):
                    replace(audit, p0_stability=forged).to_dict()
                with self.assertRaises((TypeError, ValueError)):
                    replace(audit, _p0_options=list(self.HR_OPTIONS))
                with self.assertRaises(ValueError):
                    replace(audit, _p0_options=tuple(self.HR_OPTIONS[:-1]))

    def test_vstar_p0_stability_is_bound_for_search_and_root_without_batch(self):
        class RootWinningRaw(IntegrationRaw):
            def multiple_choices_with_losses(self, image, question, options, nodes):
                if not nodes:
                    return 0, [0.0, 1.0]
                return super().multiple_choices_with_losses(
                    image, question, options, nodes
                )

        cases = (
            ("search", {}),
            ("root_no_batch", {
                "raw": RootWinningRaw(), "include_candidate": False,
            }),
        )
        for name, arguments in cases:
            with self.subTest(name=name):
                _, trace, _, _ = self._run(
                    answer_type="logits_match", **arguments,
                )
                audit = next(
                    item.next_audit for item in trace.steps if item.action == NEXT
                )
                forged = deepcopy(audit.p0_stability)
                forged.groups = {"forged": 999}
                forged.frequency = 0.123
                forged.confidence = 0.999
                with self.assertRaisesRegex(ValueError, "P0 stability"):
                    replace(audit, p0_stability=forged).to_dict()
                self.assertNotIn("_expected_p0_stability_json", repr(audit))
                self.assertNotIn(
                    "_expected_p0_stability_json", audit.to_dict(),
                )

    def test_next_step_rejects_a_replaced_or_mutated_full_answer_snapshot(self):
        _, trace, _, _ = self._run()
        step = next(item for item in trace.steps if item.action == NEXT)
        forged = deepcopy(step.answer)
        forged.groups = {"forged": {"count": 4}}
        forged.frequency = 0.25
        forged.confidence = 0.25
        with self.assertRaisesRegex(ValueError, "answer snapshot"):
            replace(step, answer=forged).to_dict()

        step.answer.groups["post_constructed"] = {"count": 99}
        with self.assertRaisesRegex(ValueError, "answer snapshot"):
            step.to_dict()


if __name__ == "__main__":
    unittest.main()
