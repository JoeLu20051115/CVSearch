import hashlib
import json
import math
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from PIL import Image

from cvsearch.evidence_gap.method import (
    _BudgetedZoomModel,
    MINIMAL_V1,
    get_evidence_gap_response,
    load_method_config,
)
from cvsearch.evidence_gap.types import BudgetLedger, FORCED_RETURN, QueryPlan, ZOOM

from tests.test_evidence_gap_next import next_extension
from tests.test_evidence_gap_search_state import candidate_snapshot, event_for
from tests.test_evidence_gap_support import (
    RawObservationModel,
    evidence_items,
    next_candidate_for,
    support_adapter,
)


ZOOM_OBSERVATION_KEYS = (
    "p2c_zoom_enabled",
    "p2c_zoom_admission_mode",
    "p2c_zoom_replacement_enabled",
    "p2c_zoom_render_policy",
)


def zoom_observation_extension(*, enabled=True, **overrides):
    extension = {
        "p2c_zoom_enabled": enabled,
        "p2c_zoom_admission_mode": "all_feasible" if enabled else "disabled",
        "p2c_zoom_replacement_enabled": False,
        "p2c_zoom_render_policy": "native_coordinate_crop_view_size_div3",
    }
    extension.update(overrides)
    return extension


def zoom_observation_config(*, enabled=True, **overrides):
    config = deepcopy(MINIMAL_V1)
    config.update({
        "config_id": (
            "unified-zoom-oracle-gamma000-v1"
            if enabled else "unified-zoom-disabled-gamma000-v1"
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
    config.update(zoom_observation_extension(enabled=enabled))
    config.update(overrides)
    return config


class ZoomObservationConfigTest(unittest.TestCase):
    def test_zoom_observation_group_is_strict_and_requires_frozen_unified_contract(self):
        loaded = load_method_config(zoom_observation_config())
        self.assertEqual(
            {key: loaded[key] for key in ZOOM_OBSERVATION_KEYS},
            zoom_observation_extension(),
        )

        for missing in ZOOM_OBSERVATION_KEYS:
            with self.subTest(missing=missing):
                partial = zoom_observation_config()
                partial.pop(missing)
                with self.assertRaisesRegex(ValueError, "all-or-none"):
                    load_method_config(partial)

        invalid = (
            {"p2c_zoom_admission_mode": "threshold"},
            {"p2c_zoom_replacement_enabled": True},
            {"p2c_zoom_render_policy": "same_extent_super_resolution"},
            {"next_enabled": True, "next_admission_mode": "all_feasible"},
            {"enable_zoom": True},
            {"quick_gate": 0.8},
            {"root_fallback_tolerance": 0.1},
            {"hr_fusion_mode": "global_soft", "hr_fusion_gamma": 2.1},
            {"max_mllm_calls": 511},
            {"evidence_support_prompt_version": "different"},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                load_method_config(zoom_observation_config(**changes))

    def test_disabled_sibling_differs_only_in_id_enabled_and_admission(self):
        enabled = load_method_config(zoom_observation_config(enabled=True))
        disabled = load_method_config(zoom_observation_config(enabled=False))
        changed = {
            key for key in enabled
            if enabled[key] != disabled[key]
        }
        self.assertEqual(changed, {
            "config_id", "p2c_zoom_enabled", "p2c_zoom_admission_mode",
        })

    def test_checked_in_enabled_and_disabled_configs_are_exact_siblings(self):
        config_directory = (
            Path(__file__).parents[1] / "reproduction" / "evidence_gap" / "configs"
        )
        paths = {
            enabled: config_directory / filename
            for enabled, filename in (
                (True, "dev_unified_zoom_oracle_gamma000_budget512.json"),
                (False, "dev_unified_zoom_disabled_gamma000_budget512.json"),
            )
        }
        raw = {
            enabled: json.loads(path.read_text(encoding="utf-8"))
            for enabled, path in paths.items()
        }
        loaded = {
            enabled: load_method_config(path)
            for enabled, path in paths.items()
        }
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                self.assertEqual(raw[enabled], zoom_observation_config(enabled=enabled))
                self.assertEqual(loaded[enabled], zoom_observation_config(enabled=enabled))
        self.assertEqual(
            {
                key for key in loaded[True]
                if loaded[True][key] != loaded[False][key]
            },
            {"config_id", "p2c_zoom_enabled", "p2c_zoom_admission_mode"},
        )


def gradient_image(size=(24, 18)):
    image = Image.new("RGB", size)
    for y in range(image.height):
        for x in range(image.width):
            image.putpixel((x, y), ((x * 11) % 256, (y * 17) % 256, (x + y) % 256))
    return image


class CoordinateZoomRaw(RawObservationModel):
    def __init__(self, *, fail_phase=None, identical_render=False, nonfinite_losses=False):
        super().__init__(fail_phase=fail_phase)
        self.view_size = 12
        self.patch_scale = None
        self.identical_render = identical_render
        self.nonfinite_losses = nonfinite_losses
        self.render_events = []
        self.support_inputs = []
        self.answer_inputs = []
        self.answer_index = 0

    @staticmethod
    def get_patch(bbox, image_width, image_height, patch_size, patch_scale=None):
        object_width = math.ceil(bbox[2])
        object_height = math.ceil(bbox[3])
        center_x = int(bbox[0] + bbox[2] / 2)
        center_y = int(bbox[1] + bbox[3] / 2)
        patch_width = max(object_width, patch_size)
        patch_height = max(object_height, patch_size)
        if patch_scale is not None:
            patch_width = int(patch_width * patch_scale)
            patch_height = int(patch_height * patch_scale)
        left = max(0, center_x - patch_width // 2)
        top = max(0, center_y - patch_height // 2)
        return [
            left, top, min(left + patch_width, image_width),
            min(top + patch_height, image_height),
        ]

    def process_nodes_to_image_list(self, nodes, image, root_anyres=True):
        self.render_events.append((self.view_size, tuple(nodes), image.tobytes()))
        if not nodes:
            return [image.copy()]
        boxes = []
        for node in nodes:
            source = getattr(node, "search_source", "fine")
            patch_size = self.view_size // 3 if source == "fast" else self.view_size
            patch_scale = None if source == "fast" else self.patch_scale
            boxes.append(self.get_patch(
                node.state.bbox, image.width, image.height,
                patch_size=patch_size, patch_scale=patch_scale,
            ))
        if self.identical_render:
            return [image.copy(), image.crop((0, 0, 4, 4))]
        union = (
            min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes),
        )
        return [image.copy(), image.crop(union)]

    def evidence_support(self, **kwargs):
        image = kwargs["rendered_observation"]
        self.support_inputs.append((self.view_size, image.mode, image.size, image.tobytes()))
        return super().evidence_support(**kwargs)

    def free_form_using_nodes(self, image, question, nodes):
        phase = f"hr_answer_{self.answer_index}"
        self.answer_inputs.append((self.view_size, image.mode, image.size, image.tobytes(), tuple(nodes)))
        self.answer_index += 1
        if self.fail_phase == phase:
            raise RuntimeError(f"failed {phase}")
        return ("zoom-A", "zoom-B", "zoom-C", "zoom-D")[self.answer_index - 1]

    def multiple_choices_with_losses(self, image, question, options, nodes):
        self.answer_inputs.append((self.view_size, image.mode, image.size, image.tobytes(), tuple(nodes)))
        if self.fail_phase == "vstar_answer":
            raise RuntimeError("failed vstar_answer")
        losses = [0.8, float("nan") if self.nonfinite_losses else 0.1]
        losses.extend(0.9 + index for index in range(max(0, len(options) - 2)))
        losses = losses[:len(options)]
        return min(range(len(losses)), key=losses.__getitem__), losses


class CandidateRenderFailureRaw(CoordinateZoomRaw):
    def process_nodes_to_image_list(self, nodes, image, root_anyres=True):
        if nodes and self.view_size == 4:
            raise RuntimeError("candidate renderer failed")
        return super().process_nodes_to_image_list(nodes, image, root_anyres=root_anyres)


class CoordinateZoomBatchTest(unittest.TestCase):
    HR_OPTIONS = tuple(f"A. choice {index}\nB. other" for index in range(4))

    def _run_batch(
        self, *, raw=None, answer_type="option_list", descriptors=None,
        max_calls=None, max_pixels=None, requirements=evidence_items(), image=None,
    ):
        image = gradient_image() if image is None else image
        raw = CoordinateZoomRaw() if raw is None else raw
        if descriptors is None:
            descriptors = (
                next_candidate_for(
                    image, bbox=(6, 4, 4, 3), source="fine",
                    scope="cropped", crop_origin=(2, 1),
                ),
            )
        options = self.HR_OPTIONS if answer_type == "option_list" else ("cat", "dog")
        calls = 6 if answer_type == "option_list" else 5
        area = image.width * image.height
        ledger = BudgetLedger(
            calls if max_calls is None else max_calls,
            calls * area if max_pixels is None else max_pixels,
        )
        budgeted = _BudgetedZoomModel(raw, ledger, answer_type=answer_type)
        method = getattr(budgeted, "coordinate_zoom_observation_batch", None)
        self.assertIsNotNone(
            method, "_BudgetedZoomModel must expose the P2C coordinate ZOOM batch",
        )
        result = method(
            source_image=image,
            q0="What evidence is visible?",
            query_plan=QueryPlan(
                main_query="What evidence is visible?", evidence_items=requirements,
            ),
            current_support_view=tuple(descriptors),
            answer_type=answer_type,
            options=options,
            render_policy="native_coordinate_crop_view_size_div3",
        )
        return result, ledger, raw, image

    @staticmethod
    def _input_sha256(record):
        mode, size, pixels = record[1], record[2], record[3]
        payload = mode.encode("utf-8") + b"\x00"
        payload += f"{size[0]}x{size[1]}".encode("ascii") + b"\x00" + pixels
        return hashlib.sha256(payload).hexdigest()

    def test_hr_single_local_view_freezes_coordinate_zoom_and_exact_raw_candidate(self):
        result, ledger, raw, image = self._run_batch()

        self.assertEqual(result.status, "success")
        self.assertEqual(result.candidate_answer, [
            "zoom-A", "zoom-B", "zoom-C", "zoom-D",
        ])
        self.assertEqual((ledger.mllm_calls, ledger.processed_pixels),
                         (6, 6 * image.width * image.height))
        plan = result.batch_plan
        self.assertEqual(plan["batch_kind"], "p2c_post_anchor_coordinate_zoom")
        self.assertEqual((plan["base_view_size"], plan["candidate_view_size"]), (12, 4))
        self.assertEqual(plan["render_policy"], "native_coordinate_crop_view_size_div3")
        self.assertEqual(len(plan["coordinate_mapping"]), 1)
        mapping = plan["coordinate_mapping"][0]
        self.assertEqual(mapping["bbox_original"], [6, 4, 4, 3])
        self.assertEqual(mapping["tree_scope"], "cropped")
        self.assertEqual(mapping["crop_origin"], [2, 1])
        self.assertNotEqual(mapping["current_crop_xyxy"], mapping["candidate_crop_xyxy"])
        self.assertNotEqual(
            plan["current_observation"]["view_sha256"],
            plan["candidate_observation"]["view_sha256"],
        )
        self.assertEqual(
            result.candidate_support.view_sha256,
            plan["candidate_observation"]["view_sha256"],
        )
        candidate_hash = plan["candidate_observation"]["view_sha256"]
        self.assertEqual(self._input_sha256(raw.support_inputs[1]), candidate_hash)
        self.assertTrue(all(self._input_sha256(record) == candidate_hash
                            for record in raw.answer_inputs))
        self.assertTrue(all(record[-1] == () for record in raw.answer_inputs))
        self.assertTrue(all(record[0] == 12 for record in raw.support_inputs + raw.answer_inputs))
        self.assertEqual(raw.view_size, 12)

    def test_vstar_multiple_overlapping_views_preserve_order_and_raw_losses(self):
        image = gradient_image()
        descriptors = (
            next_candidate_for(image, bbox=(5, 4, 5, 4), source="fine"),
            next_candidate_for(
                image, bbox=(8, 5, 5, 4), source="fast",
                depth=2, render_level=1,
            ),
        )
        result, _, raw, _ = self._run_batch(
            raw=CoordinateZoomRaw(), answer_type="logits_match",
            descriptors=descriptors, image=image,
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.candidate_answer, {"winner": 1, "losses": [0.8, 0.1]})
        plan = result.batch_plan
        self.assertEqual(plan["current_keys"], [item.canonical_key for item in descriptors])
        self.assertEqual(
            [item["descriptor_index"] for item in plan["coordinate_mapping"]], [0, 1],
        )
        self.assertEqual(
            [item["source_renderer_identity"] for item in plan["coordinate_mapping"]],
            [item.renderer_identity for item in descriptors],
        )
        self.assertEqual(len(set(plan["zoom_keys"])), 2)
        self.assertEqual(plan["candidate_merge_identity"]["per_descriptor_crop_xyxy"], [
            item["candidate_crop_xyxy"] for item in plan["coordinate_mapping"]
        ])
        self.assertEqual(self._input_sha256(raw.answer_inputs[0]),
                         plan["candidate_observation"]["view_sha256"])
        self.assertEqual(raw.view_size, 12)

    def test_render_noop_missing_requirements_and_budget_boundaries_make_no_forward(self):
        no_requirements, ledger, raw, _ = self._run_batch(requirements=())
        self.assertEqual(no_requirements.status, "no_requirements")
        self.assertEqual((ledger.mllm_calls, ledger.processed_pixels), (0, 0))
        self.assertEqual((raw.support_inputs, raw.answer_inputs), ([], []))

        noop, ledger, raw, _ = self._run_batch(
            raw=CoordinateZoomRaw(identical_render=True),
        )
        self.assertEqual(noop.status, "render_noop")
        self.assertEqual((ledger.mllm_calls, ledger.processed_pixels), (0, 0))
        self.assertEqual((raw.support_inputs, raw.answer_inputs), ([], []))
        self.assertEqual(raw.view_size, 12)

        area = 24 * 18
        for max_calls, max_pixels, reason in (
            (5, 6 * area, "mllm_calls"),
            (6, 6 * area - 1, "processed_pixels"),
        ):
            with self.subTest(reason=reason):
                result, ledger, raw, _ = self._run_batch(
                    max_calls=max_calls, max_pixels=max_pixels,
                )
                self.assertEqual(result.status, "budget_rejected")
                self.assertIn(reason, result.failure_reason)
                self.assertEqual((ledger.mllm_calls, ledger.processed_pixels), (0, 0))
                self.assertEqual((raw.support_inputs, raw.answer_inputs), ([], []))
                self.assertEqual(raw.view_size, 12)

    def test_malformed_local_descriptor_rejects_before_render_charge_or_forward(self):
        image = gradient_image()
        safe = next_candidate_for(image, bbox=(6, 4, 4, 3), source="fine")
        malformed = replace(safe, renderer_identity="forged")
        raw = CoordinateZoomRaw()
        with self.assertRaisesRegex(ValueError, "descriptor"):
            self._run_batch(raw=raw, descriptors=(malformed,), image=image)
        self.assertEqual(raw.render_events, [])
        self.assertEqual((raw.support_inputs, raw.answer_inputs), ([], []))
        self.assertEqual(raw.view_size, 12)

    def test_every_answer_failure_is_fully_charged_and_restores_view_size(self):
        area = 24 * 18
        for phase in tuple(f"hr_answer_{index}" for index in range(4)) + ("vstar_answer",):
            with self.subTest(phase=phase):
                answer_type = "logits_match" if phase == "vstar_answer" else "option_list"
                raw = CoordinateZoomRaw(fail_phase=phase)
                result, ledger, _, _ = self._run_batch(
                    raw=raw, answer_type=answer_type,
                )
                calls = 5 if answer_type == "logits_match" else 6
                self.assertEqual(result.status, "model_failed")
                self.assertEqual(result.failure_phase, phase)
                self.assertEqual((ledger.mllm_calls, ledger.processed_pixels),
                                 (calls, calls * area))
                self.assertIsNone(result.candidate_answer)
                self.assertEqual(raw.view_size, 12)
                self.assertEqual(result.executed_stages[:2],
                                 ("current_support", "candidate_support"))

        nonfinite = CoordinateZoomRaw(nonfinite_losses=True)
        result, ledger, _, _ = self._run_batch(
            raw=nonfinite, answer_type="logits_match",
        )
        self.assertEqual(result.status, "model_failed")
        self.assertEqual(result.failure_phase, "vstar_answer")
        self.assertEqual((ledger.mllm_calls, ledger.processed_pixels), (5, 5 * area))
        self.assertEqual(nonfinite.view_size, 12)

    def test_support_failures_and_nonfinite_support_charge_full_batch(self):
        area = 24 * 18
        expected = {
            "current_support": (),
            "candidate_support": ("current_support",),
        }
        for phase, executed_stages in expected.items():
            with self.subTest(phase=phase):
                raw = CoordinateZoomRaw(fail_phase=phase)
                result, ledger, _, _ = self._run_batch(raw=raw)
                self.assertEqual(result.status, "model_failed")
                self.assertEqual(result.failure_phase, phase)
                self.assertEqual(result.executed_stages, executed_stages)
                self.assertEqual((ledger.mllm_calls, ledger.processed_pixels),
                                 (6, 6 * area))
                self.assertEqual(raw.view_size, 12)

        raw = CoordinateZoomRaw()
        raw.support = support_adapter(yes_logit=float("nan"))
        result, ledger, _, _ = self._run_batch(raw=raw)
        self.assertEqual(result.status, "model_failed")
        self.assertEqual(result.failure_phase, "current_support")
        self.assertEqual(result.executed_stages, ())
        self.assertEqual((ledger.mllm_calls, ledger.processed_pixels), (6, 6 * area))
        self.assertEqual(raw.view_size, 12)

    def test_candidate_renderer_exception_is_uncharged_and_restores_view_size(self):
        raw = CandidateRenderFailureRaw()
        with self.assertRaisesRegex(RuntimeError, "candidate renderer failed"):
            self._run_batch(raw=raw)
        self.assertEqual((raw.support_inputs, raw.answer_inputs), ([], []))
        self.assertEqual(raw.view_size, 12)


class RuntimeCoordinateZoomRaw(CoordinateZoomRaw):
    def __init__(self, *, root_wins=False, **kwargs):
        super().__init__(**kwargs)
        self.root_wins = root_wins
        self.root_hr_index = 0
        self.zoom_hr_index = 0

    def free_form_using_nodes(self, image, question, nodes):
        self.answer_inputs.append((self.view_size, image.mode, image.size, image.tobytes(), tuple(nodes)))
        if self.root_hr_index < 4 and image.size == (24, 18):
            output = ("A", "B", "A", "B")[self.root_hr_index]
            self.root_hr_index += 1
            return output
        phase = f"hr_answer_{self.zoom_hr_index}"
        if self.fail_phase == phase:
            raise RuntimeError(f"failed {phase}")
        output = ("zoom-A", "zoom-B", "zoom-C", "zoom-D")[self.zoom_hr_index]
        self.zoom_hr_index += 1
        return output

    def multiple_choices_with_losses(self, image, question, options, nodes):
        self.answer_inputs.append((self.view_size, image.mode, image.size, image.tobytes(), tuple(nodes)))
        if nodes:
            return 1, [0.8, 0.1]
        if image.size == (24, 18):
            return (0, [0.0, 1.0]) if self.root_wins else (0, [0.45, 0.55])
        if self.fail_phase == "vstar_answer":
            raise RuntimeError("failed vstar_answer")
        return 0, [0.1, 0.9]


class UnifiedCoordinateZoomRuntimeTest(unittest.TestCase):
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

    def _run(
        self, *, answer_type="option_list", raw=None, question="What evidence is visible?",
        sources=("fine",), include_p0=True, original_annotation=None, enabled=True,
    ):
        raw = RuntimeCoordinateZoomRaw() if raw is None else raw
        options = self.HR_OPTIONS if answer_type == "option_list" else ["cat", "dog"]
        policy = {
            "question": question,
            "options": options,
            "answer_type": answer_type,
            "input_image": str(self.image_path),
        }
        raw_response = ["A", "B", "A", "B"] if answer_type == "option_list" else 1

        def fake_cvsearch(**kwargs):
            if not enabled:
                self.assertIsNone(kwargs["search_state_sink"])
                return deepcopy(raw_response)
            self.assertIsNotNone(kwargs["search_state_sink"])
            image = gradient_image()
            candidates = [
                candidate_snapshot(
                    (5 + index * 4, 4 + index, 4, 3), posterior=0.9 - index * 0.1,
                    depth=index + 1, render_level=0, source=source,
                )
                for index, source in enumerate(sources)
            ]
            if include_p0:
                refs, snapshot = event_for(
                    image, candidates, event="p0_selected",
                    selected=tuple(item["canonical_key"] for item in candidates),
                    remaining=(), ordinal=0,
                )
                kwargs["search_state_sink"](refs, snapshot)
            live_nodes = []
            for candidate in candidates:
                node = type("LiveNode", (), {})()
                node.depth = candidate["depth"]
                node.state = type("State", (), {
                    "bbox": candidate["bbox_original"],
                    "original_image_pil": image.copy(),
                })()
                live_nodes.append(node)
            kwargs["annotation"]["search_mode"] = 1
            kwargs["answer_observer"]("search", live_nodes, deepcopy(raw_response))
            return deepcopy(raw_response)

        output, trace = get_evidence_gap_response(
            sam_model=object(), zoom_model=raw, nlp_model=object(),
            policy_annotation=policy,
            original_annotation=(
                {"benchmark": "forbidden", "category": "forbidden", "resolution": "8K"}
                if original_annotation is None else original_annotation
            ),
            ic_examples=[], decomposed_question_template="{}",
            config=zoom_observation_config(enabled=enabled), cvsearch_fn=fake_cvsearch,
            planner=lambda policy_snapshot, targets: QueryPlan(
                main_query=policy_snapshot["question"],
                targets=("object",),
                evidence_items=({
                    "kind": "target_detail", "target": "object",
                    "requirements": ["presence", "visual_detail"],
                },),
                global_scope_required="how many" in policy_snapshot["question"].casefold(),
            ),
        )
        expected_p0 = (
            0 if answer_type == "logits_match" and raw.root_wins else raw_response
        )
        return output, trace, raw, expected_p0

    def test_hr_unified_zoom_keeps_exact_p0_and_records_dedicated_audit(self):
        output, trace, raw, p0 = self._run(answer_type="option_list")

        self.assertEqual(output, p0)
        self.assertEqual(trace.final_answer.output, p0)
        zoom_step = next(step for step in trace.steps if step.action == ZOOM)
        self.assertIsNotNone(getattr(zoom_step, "zoom_audit", None))
        audit = zoom_step.zoom_audit.to_dict()
        self.assertEqual(audit["p0_anchor"]["emitted_answer"], p0)
        self.assertEqual(audit["p0_anchor"]["cvsearch_raw"], p0)
        self.assertEqual(audit["current_keys"], audit["p0_anchor"]["node_keys"])
        self.assertEqual(audit["render_policy"], "native_coordinate_crop_view_size_div3")
        self.assertEqual((audit["base_view_size"], audit["candidate_view_size"]), (12, 4))
        self.assertEqual(audit["batch_result"]["candidate_answer"], [
            "zoom-A", "zoom-B", "zoom-C", "zoom-D",
        ])
        self.assertEqual(audit["coordinate_mapping"],
                         audit["batch_result"]["batch_plan"]["coordinate_mapping"])
        self.assertEqual(audit["replacement_reason"], "replacement_disabled_p2c")
        self.assertEqual(audit["support_proxy_status"], "audit_only_uncalibrated")
        self.assertEqual(
            audit["uncalibrated_g_zoom_proxy"],
            1.0 - audit["current_gap_support"]["p_yes"],
        )
        self.assertEqual(
            audit["support_delta"],
            audit["candidate_gap_support"]["p_yes"]
            - audit["current_gap_support"]["p_yes"],
        )
        self.assertEqual(audit["normalized_actual_cost"], 6 / 512)
        self.assertEqual(zoom_step.feasible_actions, (ZOOM,))
        self.assertEqual(zoom_step.gaps, {
            "g_zoom_proxy_audit_only": audit["uncalibrated_g_zoom_proxy"],
        })
        self.assertEqual(zoom_step.answer.output, p0)
        self.assertEqual(trace.termination, FORCED_RETURN)
        self.assertEqual(trace.steps[-1].action, FORCED_RETURN)
        candidate_hash = audit["batch_result"]["batch_plan"]["candidate_observation"]["view_sha256"]
        zoom_answer_inputs = raw.answer_inputs[-4:]
        self.assertTrue(all(CoordinateZoomBatchTest._input_sha256(item) == candidate_hash
                            for item in zoom_answer_inputs))

    def test_vstar_multiple_local_views_keep_raw_candidate_and_exact_p0(self):
        output, trace, _, p0 = self._run(
            answer_type="logits_match", sources=("fine", "fast"),
        )

        self.assertEqual(output, p0)
        audit = next(step.zoom_audit for step in trace.steps if step.action == ZOOM).to_dict()
        self.assertEqual(audit["batch_result"]["candidate_answer"], {
            "winner": 0, "losses": [0.1, 0.9],
        })
        self.assertEqual(audit["candidate_stability"]["output"], 0)
        self.assertEqual(len(audit["zoom_keys"]), 2)
        self.assertEqual([item["descriptor_index"] for item in audit["coordinate_mapping"]], [0, 1])
        self.assertEqual(trace.final_answer.output, p0)

    def test_empty_global_and_unbound_p0_views_fail_closed_without_zoom_forward(self):
        cases = (
            ("global", {"sources": ("global",)}),
            ("unbound", {"include_p0": False}),
            ("empty_root", {
                "answer_type": "logits_match",
                "raw": RuntimeCoordinateZoomRaw(root_wins=True),
            }),
        )
        for name, arguments in cases:
            with self.subTest(name=name):
                output, trace, raw, p0 = self._run(**arguments)
                self.assertEqual(output, p0)
                step = next(step for step in trace.steps if step.action == ZOOM)
                self.assertFalse(step.zoom_audit.feasible)
                self.assertIsNotNone(step.no_op_reason)
                self.assertIsNone(step.zoom_audit.batch_result)
                self.assertEqual(raw.support_inputs, [])
                self.assertEqual(raw.view_size, 12)

    def test_question_resolution_benchmark_and_category_never_route_the_action(self):
        questions = (
            "What color is visible?",
            "How many objects are visible throughout the image?",
            "Where is the object relative to the sign?",
        )
        for question in questions:
            with self.subTest(question=question):
                output, trace, _, p0 = self._run(
                    question=question,
                    original_annotation={
                        "benchmark": "HR-Bench-8K", "resolution": "8192x8192",
                        "category": "count-or-relation", "ground_truth": "forbidden",
                    },
                )
                self.assertEqual(output, p0)
                step = next(step for step in trace.steps if step.action == ZOOM)
                self.assertTrue(step.zoom_audit.feasible)
                self.assertIsNone(step.no_op_reason)

    def test_explicitly_disabled_observation_zoom_installs_no_sink_or_action(self):
        output, trace, _, p0 = self._run(enabled=False)
        self.assertEqual(output, p0)
        self.assertFalse(any(step.action in {ZOOM, "NEXT"} for step in trace.steps))
        self.assertEqual([step.action for step in trace.steps], [FORCED_RETURN])

    def test_renderer_exception_fails_closed_to_exact_p0_without_zoom_charge(self):
        raw = CandidateRenderFailureRaw()
        output, trace, raw, p0 = self._run(raw=raw)

        self.assertEqual(output, p0)
        step = next(step for step in trace.steps if step.action == ZOOM)
        self.assertEqual(step.no_op_reason, "zoom_batch_preflight_failed")
        self.assertIsNone(step.zoom_audit.batch_result)
        self.assertFalse(step.zoom_audit.feasible)
        self.assertEqual(step.budget.mllm_calls, 4)
        self.assertEqual(raw.support_inputs, [])
        self.assertEqual(len(raw.answer_inputs), 4)
        self.assertEqual(raw.view_size, 12)

    def test_zoom_audit_and_step_trace_reject_tampering_and_mutation(self):
        _, trace, _, _ = self._run(answer_type="option_list")
        step = next(step for step in trace.steps if step.action == ZOOM)
        audit = step.zoom_audit

        tampered_audits = (
            {"zoom_keys": ()},
            {"normalized_actual_cost": 0.0},
            {"uncalibrated_g_zoom_proxy": 0.0},
            {"replacement_reason": None},
        )
        for changes in tampered_audits:
            with self.subTest(audit_changes=changes), self.assertRaises(ValueError):
                replace(audit, **changes)

        tampered_steps = (
            {"action": FORCED_RETURN},
            {"focus_key": "forged"},
            {"feasible_actions": ()},
            {"gaps": {}},
        )
        for changes in tampered_steps:
            with self.subTest(step_changes=changes), self.assertRaises(ValueError):
                replace(step, **changes)

        mutated = deepcopy(audit)
        mutated.candidate_stability.output = ["forged"]
        with self.assertRaisesRegex(ValueError, "mutated"):
            mutated.to_dict()


if __name__ == "__main__":
    unittest.main()
