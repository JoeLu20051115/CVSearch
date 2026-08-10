import hashlib
import inspect
import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from PIL import Image

import cvsearch.evidence_gap.method as method_module
import cvsearch.evidence_gap.types as types_module
from cvsearch.evidence_gap.method import (
    MINIMAL_V1,
    _BudgetedZoomModel,
    get_evidence_gap_response,
    load_method_config,
)
from cvsearch.evidence_gap.search_state import SearchStateCollector
from cvsearch.evidence_gap.types import (
    BudgetLedger,
    EXPAND,
    FORCED_RETURN,
    QueryPlan,
    _validate_batch_plan,
)

from tests.test_evidence_gap_next import next_extension
from tests.test_evidence_gap_search_state import candidate_snapshot, event_for
from tests.test_evidence_gap_support import evidence_items, next_candidate_for
from tests.test_evidence_gap_zoom_observation import (
    CoordinateZoomRaw,
    RuntimeCoordinateZoomRaw,
    gradient_image,
    zoom_observation_extension,
)


EXPAND_CONFIG_KEYS = (
    "p4a_expand_enabled",
    "p4a_expand_admission_mode",
    "p4a_expand_replacement_enabled",
    "p4a_expand_selection_policy",
)
COMPOSITION_POLICY = "focus_top_blank_or_context_bottom_native_pixels_v1"
SELECTION_POLICY = "nearest_spatial_native_cvsearch_context_v1"
ANSWER_SUFFIX = "Answer the option letter directly."


def expand_extension(*, enabled=True, **overrides):
    extension = {
        "p4a_expand_enabled": enabled,
        "p4a_expand_admission_mode": "all_feasible" if enabled else "disabled",
        "p4a_expand_replacement_enabled": False,
        "p4a_expand_selection_policy": SELECTION_POLICY,
    }
    extension.update(overrides)
    return extension


def expand_config(*, enabled=True, **overrides):
    config = deepcopy(MINIMAL_V1)
    config.update({
        "config_id": (
            "unified-expand-context-oracle-gamma000-v1"
            if enabled else "unified-expand-context-disabled-gamma000-v1"
        ),
        "mode": "root_search_fallback",
        "rerank_enabled": False,
        "ranking_mode": "cvsearch",
        "ranking_rho": 0.0,
        "ranking_max_displacement": 0,
        "quick_gate": 0.6,
        "root_fallback_tolerance": 0.05,
        "enable_zoom": False,
        "enable_split": False,
        "enable_expand": False,
        "enable_certified_stop": False,
        "hr_fusion_mode": "off",
        "hr_fusion_gamma": 0.0,
        "max_mllm_calls": 512,
        "max_processed_pixels": 10_000_000_000,
    })
    config.update(next_extension(enabled=False))
    config.update(zoom_observation_extension(enabled=False))
    config.update(expand_extension(enabled=enabled))
    config.update(overrides)
    return config


def strict_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def observation_sha256(mode, size, pixels):
    payload = mode.encode("utf-8") + b"\x00"
    payload += f"{size[0]}x{size[1]}".encode("ascii") + b"\x00" + pixels
    return hashlib.sha256(payload).hexdigest()


class ExpandRaw(CoordinateZoomRaw):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = (17, 29, 41)
        self.answer_prompts = []
        self.answer_options = []

    def free_form_using_nodes(self, image, question, nodes):
        self.answer_prompts.append(question)
        return super().free_form_using_nodes(image, question, nodes)

    def multiple_choices_with_losses(self, image, question, options, nodes):
        self.answer_prompts.append(question)
        self.answer_options.append(tuple(options))
        return super().multiple_choices_with_losses(image, question, options, nodes)


class BlankContextRaw(ExpandRaw):
    def process_nodes_to_image_list(self, nodes, image, root_anyres=True):
        if len(nodes) == 1 and nodes[0].state.bbox[0] >= 14:
            return [Image.new("RGB", (10, 12), self.background_color)]
        return super().process_nodes_to_image_list(nodes, image, root_anyres=root_anyres)


class ExpandSearchStateTest(unittest.TestCase):
    def setUp(self):
        self.image = gradient_image((100, 100))

    def collector(self, candidates, *, selected=(), popped=()):
        collector = SearchStateCollector(self.image)
        refs, snapshot = event_for(
            self.image, candidates, event="p0_selected", selected=selected,
            popped=popped,
            remaining=tuple(
                item["canonical_key"] for item in candidates
                if item["canonical_key"] not in set(selected) | set(popped)
            ),
        )
        collector(refs, snapshot)
        return collector

    def test_spatial_peek_is_pure_threshold_free_and_excludes_next_like_boxes(self):
        focus = candidate_snapshot((40, 40, 10, 10), posterior=0.7, source="fine")
        contained = candidate_snapshot((42, 42, 2, 2), posterior=0.99, depth=2,
                                       source="fine")
        contains_focus = candidate_snapshot((30, 30, 40, 40), posterior=0.99,
                                             depth=3, source="fine")
        overlap = candidate_snapshot((48, 44, 10, 4), posterior=0.1, depth=4,
                                     source="fine")
        adjacent = candidate_snapshot((51, 44, 5, 4), posterior=0.99, depth=5,
                                      source="fine")
        collector = self.collector(
            (focus, contained, contains_focus, overlap, adjacent),
            selected=(focus["canonical_key"],),
        )
        before = strict_bytes(collector.to_dict())

        method = getattr(collector, "peek_expand_candidate", None)
        self.assertIsNotNone(method, "collector must expose pure EXPAND peek")
        self.assertEqual(tuple(inspect.signature(method).parameters), ("current_keys",))
        decision = method((focus["canonical_key"],))

        self.assertEqual(decision.candidate.canonical_key, overlap["canonical_key"])
        self.assertEqual(decision.normalized_edge_gap, 0.0)
        self.assertGreater(decision.positive_outside_area, 0.0)
        self.assertEqual(decision.rank_tuple[0], 0.0)
        self.assertEqual(strict_bytes(collector.to_dict()), before)
        self.assertEqual(strict_bytes(method((focus["canonical_key"],)).to_dict()),
                         strict_bytes(decision.to_dict()))
        self.assertEqual(strict_bytes(collector.to_dict()), before)

    def test_spatial_ties_use_posterior_then_native_order_without_state_mutation(self):
        focus = candidate_snapshot((40, 40, 10, 10), posterior=0.7, source="fine")
        missing = candidate_snapshot((51, 40, 4, 4), posterior=None, depth=2,
                                     source="fine")
        lower = candidate_snapshot((40, 51, 4, 4), posterior=0.3, depth=3,
                                   source="fine")
        higher = candidate_snapshot((45, 51, 4, 4), posterior=0.8, depth=4,
                                    source="fine")
        collector = self.collector(
            (focus, missing, lower, higher), selected=(focus["canonical_key"],),
        )
        before = strict_bytes(collector.to_dict())
        decision = collector.peek_expand_candidate((focus["canonical_key"],))
        self.assertEqual(decision.candidate.canonical_key, higher["canonical_key"])
        self.assertEqual(decision.rank_tuple[1:3], (False, -0.8))
        self.assertEqual(strict_bytes(collector.to_dict()), before)

    def test_visited_duplicates_root_empty_and_unbound_fail_closed_byte_exact(self):
        focus = candidate_snapshot((40, 40, 10, 10), posterior=0.7, source="fine")
        same_renderer = candidate_snapshot((40, 40, 10, 10), posterior=0.9,
                                           depth=2, source="fine")
        visited = candidate_snapshot((52, 40, 5, 5), posterior=0.8, depth=3,
                                     source="fine")
        collector = self.collector(
            (focus, same_renderer, visited), selected=(focus["canonical_key"],),
            popped=(visited["canonical_key"],),
        )
        cases = (
            (collector, (focus["canonical_key"],)),
            (collector, ()),
            (collector, ("missing",)),
            (SearchStateCollector(self.image), (focus["canonical_key"],)),
        )
        root = candidate_snapshot((0, 0, 100, 100), posterior=0.9, depth=0,
                                  source="global")
        root_collector = self.collector((root,), selected=(root["canonical_key"],))
        cases += ((root_collector, (root["canonical_key"],)),)
        for candidate_collector, keys in cases:
            with self.subTest(keys=keys):
                before = strict_bytes(candidate_collector.to_dict())
                decision = candidate_collector.peek_expand_candidate(keys)
                self.assertIsNone(decision.candidate)
                self.assertIsNotNone(decision.no_op_reason)
                self.assertEqual(strict_bytes(candidate_collector.to_dict()), before)


class ExpandConfigTest(unittest.TestCase):
    def test_expand_group_and_checked_in_siblings_are_exact(self):
        keys = getattr(method_module, "EXPAND_OBSERVATION_CONFIG_KEYS", None)
        self.assertEqual(keys, EXPAND_CONFIG_KEYS)
        enabled = load_method_config(expand_config(enabled=True))
        disabled = load_method_config(expand_config(enabled=False))
        self.assertEqual({key for key in enabled if enabled[key] != disabled[key]}, {
            "config_id", "p4a_expand_enabled", "p4a_expand_admission_mode",
        })
        for missing in EXPAND_CONFIG_KEYS:
            partial = expand_config()
            partial.pop(missing)
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "all-or-none"):
                load_method_config(partial)
        invalid = (
            {"p4a_expand_admission_mode": "threshold"},
            {"p4a_expand_replacement_enabled": True},
            {"p4a_expand_selection_policy": "question_router"},
            {"next_enabled": True, "next_admission_mode": "all_feasible"},
            {"p2c_zoom_enabled": True, "p2c_zoom_admission_mode": "all_feasible"},
            {"quick_gate": 0.8},
            {"max_mllm_calls": 511},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                load_method_config(expand_config(**changes))

        root = Path(__file__).parents[1] / "reproduction/evidence_gap/configs"
        checked = {
            state: root / filename for state, filename in (
                (True, "dev_unified_expand_context_oracle_gamma000_budget512.json"),
                (False, "dev_unified_expand_context_disabled_gamma000_budget512.json"),
            )
        }
        for state, path in checked.items():
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             expand_config(enabled=state))


class ExpandBatchTest(unittest.TestCase):
    HR_OPTIONS = tuple(f"A. choice {index}\nB. other" for index in range(4))

    def run_batch(self, *, raw=None, answer_type="option_list", max_calls=None,
                  max_pixels=None, requirements=evidence_items()):
        image = gradient_image()
        focus = next_candidate_for(image, bbox=(6, 4, 4, 3), source="fine")
        context = next_candidate_for(
            image, bbox=(18, 10, 4, 4), depth=2, source="fine",
        )
        collector = SearchStateCollector(image)
        refs, snapshot = event_for(
            image,
            (candidate_snapshot((6, 4, 4, 3), posterior=0.8, source="fine"),
             candidate_snapshot((18, 10, 4, 4), posterior=0.8, depth=2,
                                source="fine")),
            event="p0_selected", selected=(focus.canonical_key,),
            remaining=(context.canonical_key,),
        )
        collector(refs, snapshot)
        decision = collector.peek_expand_candidate((focus.canonical_key,))
        raw = ExpandRaw() if raw is None else raw
        options = self.HR_OPTIONS if answer_type == "option_list" else ("cat", "dog")
        calls = 6 if answer_type == "option_list" else 5
        area = image.width * image.height
        ledger = BudgetLedger(
            calls if max_calls is None else max_calls,
            calls * area if max_pixels is None else max_pixels,
        )
        budgeted = _BudgetedZoomModel(raw, ledger, answer_type=answer_type)
        method = getattr(budgeted, "expand_context_observation_batch", None)
        self.assertIsNotNone(method, "budgeted model must expose P4A EXPAND batch")
        result = method(
            source_image=image,
            q0="What evidence is visible?",
            query_plan=QueryPlan(
                main_query="What evidence is visible?", evidence_items=requirements,
            ),
            current_support_view=(focus,),
            expand_decision=decision,
            answer_type=answer_type,
            options=options,
            composition_policy=COMPOSITION_POLICY,
        )
        return result, ledger, raw, image, focus, context, decision

    def test_same_shape_composites_preserve_focus_and_bind_full_geometry(self):
        result, ledger, raw, image, focus, context, decision = self.run_batch()
        self.assertEqual(result.status, "success")
        self.assertEqual((ledger.mllm_calls, ledger.processed_pixels),
                         (6, 6 * image.width * image.height))
        plan = result.batch_plan
        self.assertEqual(plan["batch_kind"], "p4a_post_anchor_expand_context")
        self.assertEqual(plan["selection_policy"], SELECTION_POLICY)
        self.assertEqual(plan["composition_identity"]["policy"], COMPOSITION_POLICY)
        self.assertEqual(plan["current_keys"], [focus.canonical_key])
        self.assertEqual(plan["candidate_keys"], [focus.canonical_key, context.canonical_key])
        self.assertEqual(plan["focus_role"]["canonical_keys"], [focus.canonical_key])
        self.assertEqual(plan["context_role"]["canonical_key"], context.canonical_key)
        self.assertEqual(plan["context_role"]["rank_tuple"], list(decision.rank_tuple))
        composition = plan["composition_identity"]
        self.assertEqual(plan["current_observation"]["rendered_size"],
                         plan["candidate_observation"]["rendered_size"])
        self.assertEqual(composition["focus_offset_xy"], [0, 0])
        self.assertEqual(composition["context_offset_xy"][1],
                         composition["focus_size"][1] + 8)
        self.assertEqual(composition["focus_current_rectangle_sha256"],
                         composition["focus_panel_sha256"])
        self.assertEqual(composition["focus_candidate_rectangle_sha256"],
                         composition["focus_panel_sha256"])
        self.assertNotEqual(plan["current_observation"]["view_sha256"],
                            plan["candidate_observation"]["view_sha256"])
        self.assertEqual(plan["candidate_answer_input_sha256"],
                         plan["candidate_observation"]["view_sha256"])
        self.assertEqual(len(raw.render_events), 2)
        self.assertEqual(len(raw.render_events[0][1]), 1)
        self.assertEqual(len(raw.render_events[1][1]), 1)
        self.assertEqual(plan["focus_merge_identity"]["per_descriptor_crop_xyxy"],
                         plan["focus_role"]["native_crops_xyxy"])
        self.assertEqual(plan["context_merge_identity"]["per_descriptor_crop_xyxy"],
                         [plan["context_role"]["native_crop_xyxy"]])
        _validate_batch_plan(plan)

    def test_hr_and_vstar_answers_use_materialized_candidate_rgb_nodes_empty_and_bind_prompts(self):
        hr, _, raw, _, _, _, _ = self.run_batch()
        plan = hr.batch_plan
        candidate_hash = plan["candidate_observation"]["view_sha256"]
        self.assertEqual(hr.candidate_answer, ["zoom-A", "zoom-B", "zoom-C", "zoom-D"])
        self.assertTrue(all(
            observation_sha256(record[1], record[2], record[3]) == candidate_hash
            for record in raw.answer_inputs
        ))
        self.assertTrue(all(record[-1] == () for record in raw.answer_inputs))
        expected_prompts = [
            "What evidence is visible?\n" + option + ANSWER_SUFFIX
            for option in self.HR_OPTIONS
        ]
        self.assertEqual(raw.answer_prompts, expected_prompts)
        self.assertEqual(plan["options"], list(self.HR_OPTIONS))
        self.assertEqual(plan["answer_prompt_sha256"], [
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for prompt in expected_prompts
        ])

        vstar, _, raw, _, _, _, _ = self.run_batch(answer_type="logits_match")
        plan = vstar.batch_plan
        self.assertEqual(vstar.candidate_answer, {"winner": 1, "losses": [0.8, 0.1]})
        self.assertEqual(raw.answer_options, [("cat", "dog")])
        self.assertEqual(raw.answer_inputs[0][-1], ())
        self.assertEqual(
            observation_sha256(raw.answer_inputs[0][1], raw.answer_inputs[0][2],
                               raw.answer_inputs[0][3]),
            plan["candidate_answer_input_sha256"],
        )
        self.assertEqual(plan["answer_prompt_sha256"], [hashlib.sha256(strict_bytes({
            "q0": "What evidence is visible?", "options": ["cat", "dog"],
        })).hexdigest()])

    def test_preflight_budget_render_noop_and_admitted_failures_close_atomically(self):
        no_requirements, ledger, raw, *_ = self.run_batch(requirements=())
        self.assertEqual(no_requirements.status, "no_requirements")
        self.assertEqual((ledger.mllm_calls, ledger.processed_pixels), (0, 0))
        self.assertEqual((raw.support_inputs, raw.answer_inputs), ([], []))

        noop, ledger, raw, *_ = self.run_batch(raw=BlankContextRaw())
        self.assertEqual(noop.status, "render_noop")
        self.assertEqual((ledger.mllm_calls, ledger.processed_pixels), (0, 0))
        self.assertEqual((raw.support_inputs, raw.answer_inputs), ([], []))

        area = 24 * 18
        for max_calls, max_pixels in ((5, 6 * area), (6, 6 * area - 1)):
            rejected, ledger, raw, *_ = self.run_batch(
                max_calls=max_calls, max_pixels=max_pixels,
            )
            self.assertEqual(rejected.status, "budget_rejected")
            self.assertEqual((ledger.mllm_calls, ledger.processed_pixels), (0, 0))
            self.assertEqual((raw.support_inputs, raw.answer_inputs), ([], []))

        for phase, expected_prefix in (
            ("current_support", ()),
            ("candidate_support", ("current_support",)),
            ("hr_answer_2", ("current_support", "candidate_support",
                             "hr_answer_0", "hr_answer_1")),
            ("vstar_answer", ("current_support", "candidate_support")),
        ):
            answer_type = "logits_match" if phase == "vstar_answer" else "option_list"
            failed, ledger, _, *_ = self.run_batch(
                raw=ExpandRaw(fail_phase=phase), answer_type=answer_type,
            )
            calls = 5 if answer_type == "logits_match" else 6
            self.assertEqual(failed.status, "model_failed")
            self.assertEqual(failed.failure_phase, phase)
            self.assertEqual(failed.executed_stages, expected_prefix)
            self.assertEqual((ledger.mllm_calls, ledger.processed_pixels),
                             (calls, calls * area))

    def test_plan_rejects_geometry_composition_options_and_prompt_forgery(self):
        result, *_ = self.run_batch()
        cases = []
        for mutation in (
            lambda plan: plan["focus_merge_identity"].__setitem__(
                "union_crop_xyxy", [0, 0, 1, 1]),
            lambda plan: plan["composition_identity"].__setitem__(
                "focus_candidate_rectangle_sha256", "f" * 64),
            lambda plan: plan.__setitem__("candidate_answer_input_sha256", "f" * 64),
            lambda plan: plan["options"].__setitem__(0, "A. poisoned"),
            lambda plan: plan["answer_prompt_sha256"].__setitem__(0, "f" * 64),
        ):
            forged = deepcopy(result.batch_plan)
            forged.pop("plan_hash", None)
            mutation(forged)
            cases.append(forged)
        for plan in cases:
            with self.assertRaises(ValueError):
                _validate_batch_plan(plan)


class RuntimeExpandRaw(RuntimeCoordinateZoomRaw):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = (17, 29, 41)


class UnifiedExpandRuntimeTest(unittest.TestCase):
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
        gradient_image().save(self.image_path)

    def run_runtime(self, *, answer_type="option_list", raw=None, enabled=True,
                    source="fine", include_p0=True, original_annotation=None,
                    interrupt=False):
        raw = RuntimeExpandRaw() if raw is None else raw
        options = self.HR_OPTIONS if answer_type == "option_list" else ["cat", "dog"]
        policy = {
            "question": "What evidence is visible?", "options": options,
            "answer_type": answer_type, "input_image": str(self.image_path),
        }
        raw_response = ["A", "B", "A", "B"] if answer_type == "option_list" else 1

        def fake_cvsearch(**kwargs):
            if not enabled:
                self.assertIsNone(kwargs["search_state_sink"])
                return deepcopy(raw_response)
            image = gradient_image()
            focus = candidate_snapshot((5, 4, 4, 3), posterior=0.9, source=source)
            context = candidate_snapshot((18, 10, 4, 4), posterior=0.8, depth=2,
                                         source="fine")
            if include_p0:
                refs, snapshot = event_for(
                    image, (focus, context), event="p0_selected",
                    selected=(focus["canonical_key"],),
                    remaining=(context["canonical_key"],),
                )
                kwargs["search_state_sink"](refs, snapshot)
            node = type("LiveNode", (), {})()
            node.depth = focus["depth"]
            node.state = type("State", (), {
                "bbox": focus["bbox_original"],
                "original_image_pil": image.copy(),
            })()
            kwargs["annotation"]["search_mode"] = 1
            kwargs["answer_observer"]("search", [node], deepcopy(raw_response))
            if interrupt:
                from cvsearch.evidence_gap.types import BudgetExceeded
                raise BudgetExceeded("interrupted after P0")
            return deepcopy(raw_response)

        output, trace = get_evidence_gap_response(
            sam_model=object(), zoom_model=raw, nlp_model=object(),
            policy_annotation=policy,
            original_annotation=(
                {"benchmark": "poison", "resolution": "8K", "category": "poison"}
                if original_annotation is None else original_annotation
            ),
            ic_examples=[], decomposed_question_template="{}",
            config=expand_config(enabled=enabled), cvsearch_fn=fake_cvsearch,
            planner=lambda policy_snapshot, targets: QueryPlan(
                main_query=policy_snapshot["question"], targets=("object",),
                evidence_items=({
                    "kind": "target_detail", "target": "object",
                    "requirements": ["presence", "visual_detail"],
                },),
            ),
        )
        return output, trace, raw, raw_response

    def test_success_is_exact_p0_and_has_dedicated_immutable_expand_audit(self):
        output, trace, raw, p0 = self.run_runtime()
        self.assertEqual(output, p0)
        self.assertEqual(trace.final_answer.output, p0)
        self.assertEqual(trace.anchor_answer.output, p0)
        self.assertEqual([step.action for step in trace.steps], [EXPAND, FORCED_RETURN])
        step = trace.steps[0]
        audit = getattr(step, "expand_audit", None)
        self.assertIsNotNone(audit)
        payload = audit.to_dict()
        self.assertEqual(payload["p0_anchor"]["emitted_answer"], p0)
        self.assertEqual(payload["current_keys"], payload["p0_anchor"]["node_keys"])
        self.assertEqual(payload["candidate_keys"][:-1], payload["current_keys"])
        self.assertEqual(payload["replacement_reason"], "replacement_disabled_p4a")
        self.assertEqual(payload["support_proxy_status"], "audit_only_uncalibrated")
        self.assertEqual(step.feasible_actions, (EXPAND,))
        self.assertEqual(step.answer.output, p0)
        self.assertTrue(all(record[-1] == () for record in raw.answer_inputs[-4:]))

        with self.assertRaises(ValueError):
            replace(audit, replacement_reason=None)
        with self.assertRaises(ValueError):
            replace(step, action=FORCED_RETURN)
        with self.assertRaises(ValueError):
            replace(step, zoom_audit=object())
        forged = deepcopy(step.answer)
        forged.groups = {"poison": 1}
        with self.assertRaises(ValueError):
            replace(step, answer=forged).to_dict()

    def test_root_empty_unavailable_nonlocal_and_interrupted_focus_fail_closed_zero_expand_calls(self):
        cases = (
            {"answer_type": "logits_match", "raw": RuntimeExpandRaw(root_wins=True)},
            {"include_p0": False},
            {"source": "global"},
            {"interrupt": True},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                output, trace, raw, _ = self.run_runtime(**arguments)
                step = next(step for step in trace.steps if step.action == EXPAND)
                self.assertEqual(output, step.answer.output)
                self.assertFalse(step.expand_audit.feasible)
                self.assertIsNone(step.expand_audit.batch_result)
                self.assertIsNotNone(step.no_op_reason)
                self.assertEqual(raw.support_inputs, [])

    def test_disabled_sibling_is_exact_p0_and_installs_no_observer_action(self):
        output, trace, _, p0 = self.run_runtime(enabled=False)
        self.assertEqual(output, p0)
        self.assertEqual([step.action for step in trace.steps], [FORCED_RETURN])

    def test_evaluator_poison_cannot_change_or_serialize_decision_plan_or_audit(self):
        poisons = (
            {
                "benchmark": "POISON_BENCHMARK_ALPHA",
                "resolution": "POISON_RESOLUTION_ALPHA",
                "category": "POISON_CATEGORY_ALPHA",
                "index": 1, "answer": "POISON_ANSWER_ALPHA",
                "target_object": ["POISON_TARGET_ALPHA"],
                "bbox": [[1, 2, 3, 4]],
            },
            {
                "benchmark": "POISON_BENCHMARK_OMEGA",
                "resolution": "POISON_RESOLUTION_OMEGA",
                "category": "POISON_CATEGORY_OMEGA",
                "index": 999, "answer": "POISON_ANSWER_OMEGA",
                "target_object": ["POISON_TARGET_OMEGA"],
                "bbox": [[90, 90, 1, 1]],
            },
        )
        payloads = []
        for poison in poisons:
            output, trace, _, p0 = self.run_runtime(original_annotation=poison)
            self.assertEqual(output, p0)
            audit = trace.steps[0].expand_audit.to_dict()
            payloads.append({
                "focus_role": audit["focus_role"],
                "context_role": audit["context_role"],
                "plan": audit["batch_result"]["batch_plan"],
            })
            encoded = json.dumps(audit, sort_keys=True)
            for value in poison.values():
                if isinstance(value, str):
                    self.assertNotIn(value, encoded)
        self.assertEqual(payloads[0], payloads[1])


if __name__ == "__main__":
    unittest.main()
