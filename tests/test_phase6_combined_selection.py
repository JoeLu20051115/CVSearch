import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import cvsearch.eval.phase4_expand_oracle as phase4
import cvsearch.eval.phase5_unified_selector as phase5
import cvsearch.eval.phase6_combined_selection as phase6
from cvsearch.eval.phase6_combined_selection import (
    freeze_combined_decisions,
    validate_and_extract_combined_pairs,
    validate_combined_launch_pair,
)
from cvsearch.evidence_gap.method import load_method_config
from cvsearch.evidence_gap.provenance import canonical_sha256
from tests.test_phase3_zoom_oracle import (
    _as_unavailable_noop,
    _merge_identity,
    _row as zoom_pair,
    _support_payload,
)
from tests.test_phase4_expand_oracle import (
    FROZEN_REVISION,
    PROMPT_SHA,
    PairFactory,
    _audit as expand_audit,
    _as_budget_rejected as expand_budget_rejected,
    _as_model_failed as expand_model_failed,
    _as_no_batch as expand_no_batch,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "reproduction/evidence_gap/configs"
DISABLED_CONFIG = CONFIG_ROOT / (
    "dev_unified_zoom_expand_disabled_gamma000_budget512.json"
)
ENABLED_CONFIG = CONFIG_ROOT / (
    "dev_unified_zoom_expand_observation_gamma000_budget512.json"
)


def _configs():
    return load_method_config(DISABLED_CONFIG), load_method_config(ENABLED_CONFIG)


def _config_identity(config, path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    files = [{"path": path.name, "sha256": digest, "size": path.stat().st_size}]
    material = {"kind": "file", "files": files}
    source = {
        "kind": "file", "path": str(path), "files": files,
        "sha256": canonical_sha256(material),
    }
    return {
        "loaded": copy.deepcopy(config),
        "loaded_sha256": canonical_sha256(config),
        "source": source,
    }


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _retarget_zoom_success(zoom_step, enabled, disabled_budget, source_image):
    audit = zoom_step["zoom_audit"]
    batch = audit["batch_result"]
    plan = batch["batch_plan"]
    expand = expand_audit(enabled)
    descriptor = copy.deepcopy(expand["p0_anchor"]["support_view"][0])
    source = phase4._source_identity(source_image)
    source_key = _canonical(source)
    self_key = descriptor["canonical_key"]
    bbox = descriptor["bbox_original"]
    current_crop = [0, 0, source_image.width, source_image.height]
    candidate_crop = [
        int(bbox[0]), int(bbox[1]),
        int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]),
    ]
    renderer = _canonical({
        "action": "P2C_ZOOM",
        "candidate_crop_xyxy": candidate_crop,
        "candidate_view_size": 112,
        "current_key": self_key,
        "render_policy": "native_coordinate_crop_view_size_div3",
        "source_image_key": source_key,
    })
    zoom_key = "zoom-" + hashlib.sha256(renderer.encode()).hexdigest()
    mapping = {
        "descriptor_index": 0,
        "current_key": self_key,
        "zoom_key": zoom_key,
        "bbox_original": copy.deepcopy(bbox),
        "depth": descriptor["depth"],
        "render_level": descriptor["render_level"],
        "posterior_score": descriptor["posterior_score"],
        "first_seen_ordinal": descriptor["first_seen_ordinal"],
        "tree_scope": descriptor["tree_scope"],
        "crop_origin": copy.deepcopy(descriptor["crop_origin"]),
        "source_image_key": source_key,
        "source": descriptor["source"],
        "renderer_kind": descriptor["renderer_kind"],
        "source_renderer_identity": descriptor["renderer_identity"],
        "zoom_renderer_identity": renderer,
        "current_crop_xyxy": current_crop,
        "candidate_crop_xyxy": candidate_crop,
    }
    plan.update(
        source_identity=source,
        source_width=source_image.width,
        source_height=source_image.height,
        accounted_source_area=source_image.width * source_image.height,
        pixels_per_logical_forward=source_image.width * source_image.height,
        current_keys=[self_key],
        zoom_keys=[zoom_key],
        coordinate_mapping=[mapping],
        current_merge_identity=_merge_identity(current_crop),
        candidate_merge_identity=_merge_identity(candidate_crop),
        current_observation={
            "canonical_keys": [self_key],
            "renderer_identities": [descriptor["renderer_identity"]],
            "rendered_mode": "RGB", "rendered_size": [64, 64],
            "view_sha256": "1" * 64,
            "descriptors": [copy.deepcopy(descriptor)],
        },
        candidate_observation={
            "canonical_keys": [zoom_key],
            "renderer_identities": [renderer],
            "rendered_mode": "RGB", "rendered_size": [48, 48],
            "view_sha256": "2" * 64,
            "descriptors": [{
                "canonical_key": zoom_key,
                "renderer_identity": renderer,
                "source_descriptor": copy.deepcopy(descriptor),
            }],
        },
        candidate_answer_input_sha256="2" * 64,
    )
    plan["total_pixels"] = plan["total_calls"] * plan["pixels_per_logical_forward"]
    plan.pop("plan_hash", None)
    plan_hash = hashlib.sha256(_canonical(plan).encode()).hexdigest()
    plan["plan_hash"] = plan_hash
    current_support = _support_payload(plan, plan_hash, "current", 0.4)
    candidate_support = _support_payload(plan, plan_hash, "candidate", 0.8)
    after = copy.deepcopy(disabled_budget)
    after["mllm_calls"] += plan["total_calls"]
    after["processed_pixels"] += plan["total_pixels"]
    batch.update(
        batch_plan_hash=plan_hash,
        current_support=current_support,
        candidate_support=candidate_support,
        ledger_before=copy.deepcopy(disabled_budget),
        ledger_after=copy.deepcopy(after),
    )
    audit.update(
        p0_anchor=copy.deepcopy(expand["p0_anchor"]),
        current_keys=[self_key], zoom_keys=[zoom_key],
        coordinate_mapping=[mapping],
        current_gap_support=current_support,
        candidate_gap_support=candidate_support,
        p0_stability=copy.deepcopy(expand["p0_stability"]),
        uncertainty=expand["p0_stability"]["uncertainty"],
        normalized_actual_cost=plan["total_calls"] / 512,
    )
    zoom_step["focus_key"] = "zoom-aggregate-" + hashlib.sha256(
        json.dumps([zoom_key], separators=(",", ":")).encode()
    ).hexdigest()
    zoom_step["budget"] = copy.deepcopy(after)
    return after


class CombinedPairFactory:
    def __init__(self, directory):
        self.expand = PairFactory(directory)

    def pair(
        self, benchmark="vstar", ordinal=0, *, zoom_feasible=False,
        expand_mode="success",
    ):
        disabled, enabled, _ = self.expand.pair(benchmark, ordinal)
        if expand_mode == "model_failed":
            enabled = expand_model_failed(enabled)
        elif expand_mode == "budget_rejected":
            disabled, enabled = expand_budget_rejected(disabled, enabled)
        elif expand_mode == "no_batch":
            enabled = expand_no_batch(enabled, "no_candidate")
        elif expand_mode != "success":
            raise ValueError(expand_mode)
        disabled_config, enabled_config = _configs()

        _, standalone_zoom = zoom_pair(benchmark, ordinal)
        if not zoom_feasible:
            standalone_zoom = _as_unavailable_noop(standalone_zoom)
        zoom_step = copy.deepcopy(standalone_zoom["method_trace"]["steps"][0])
        zoom_step["step"] = 0
        zoom_step["answer"] = copy.deepcopy(enabled["method_trace"]["final_answer"])
        zoom = zoom_step["zoom_audit"]
        expand = expand_audit(enabled)
        disabled_budget = disabled["method_trace"]["budget"]
        if zoom_feasible:
            with Image.open(self.expand.image_path) as opened:
                zoom_after = _retarget_zoom_success(
                    zoom_step, enabled, disabled_budget, opened.convert("RGB"),
                )
            expand_batch = expand["batch_result"]
            expand_delta_calls = (
                expand_batch["ledger_after"]["mllm_calls"]
                - expand_batch["ledger_before"]["mllm_calls"]
            )
            expand_delta_pixels = (
                expand_batch["ledger_after"]["processed_pixels"]
                - expand_batch["ledger_before"]["processed_pixels"]
            )
            expand_after = copy.deepcopy(zoom_after)
            expand_after["mllm_calls"] += expand_delta_calls
            expand_after["processed_pixels"] += expand_delta_pixels
            expand_batch["ledger_before"] = copy.deepcopy(zoom_after)
            expand_batch["ledger_after"] = copy.deepcopy(expand_after)
            enabled["method_trace"]["steps"][0]["budget"] = copy.deepcopy(expand_after)
            enabled["method_trace"]["steps"][1]["budget"] = copy.deepcopy(expand_after)
            enabled["method_trace"]["budget"] = copy.deepcopy(expand_after)
        else:
            zoom_step.update(
                no_op_reason="zoom_batch_preflight_failed",
                budget=copy.deepcopy(disabled_budget),
            )
            zoom.update(
                p0_anchor=copy.deepcopy(expand["p0_anchor"]),
                current_keys=copy.deepcopy(expand["current_keys"]),
                p0_stability=copy.deepcopy(expand["p0_stability"]),
                uncertainty=expand["p0_stability"]["uncertainty"],
            )

        expand_step = enabled["method_trace"]["steps"][0]
        forced_step = enabled["method_trace"]["steps"][1]
        expand_step["step"] = 1
        forced_step["step"] = 2
        enabled["method_trace"]["steps"] = [zoom_step, expand_step, forced_step]
        enabled["method_trace"]["effective_config"] = copy.deepcopy(enabled_config)
        enabled["method_trace"]["config_id"] = enabled_config["config_id"]
        disabled["method_trace"]["effective_config"] = copy.deepcopy(disabled_config)
        disabled["method_trace"]["config_id"] = disabled_config["config_id"]

        disabled_manifest = self.expand.manifest(
            config=None, benchmark=benchmark, ordinal=ordinal,
        )
        enabled_manifest = self.expand.manifest(
            config=None, benchmark=benchmark, ordinal=ordinal,
        )
        # PairFactory has no public config hook for the combined siblings; retain
        # every non-config launch field and bind the checked-in bytes here.
        disabled_manifest["config"] = _config_identity(disabled_config, DISABLED_CONFIG)
        enabled_manifest["config"] = _config_identity(enabled_config, ENABLED_CONFIG)
        for manifest in (disabled_manifest, enabled_manifest):
            manifest["code"]["revision"] = FROZEN_REVISION
        disabled["_eg_code_revision"] = FROZEN_REVISION
        enabled["_eg_code_revision"] = FROZEN_REVISION
        disabled["_eg_run_fingerprint"] = canonical_sha256(disabled_manifest)
        enabled["_eg_run_fingerprint"] = canonical_sha256(enabled_manifest)
        return disabled, enabled, disabled_manifest, enabled_manifest


class Phase6CombinedSelectionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.factory = CombinedPairFactory(self.directory.name)

    def validate(self, benchmark, disabled, enabled, left, right):
        with (
            patch.object(phase6, "_FROZEN_COMBINED_INFERENCE_REVISION", FROZEN_REVISION),
            patch.object(phase4, "_support_prompt_sha256", return_value=PROMPT_SHA),
        ):
            launch = validate_combined_launch_pair(
                [disabled], [enabled], left, right,
            )
            pairs = validate_and_extract_combined_pairs(
                benchmark, [disabled], [enabled],
                disabled_launch_manifest=left,
                enabled_launch_manifest=right,
            )
            frozen = freeze_combined_decisions(
                benchmark, [disabled], [enabled],
                disabled_launch_manifest=left,
                enabled_launch_manifest=right,
            )
            return launch, pairs, frozen

    def assert_rejected(self, benchmark, disabled, enabled, left, right):
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            self.validate(benchmark, disabled, enabled, left, right)

    def test_valid_exact_combined_trace_extracts_dtos_and_freezes_p0(self):
        disabled, enabled, left, right = self.factory.pair("vstar")
        before = json.dumps(
            [disabled, enabled, left, right], sort_keys=True, separators=(",", ":"),
        )

        launch, pairs, decisions = self.validate(
            "vstar", disabled, enabled, left, right,
        )

        self.assertEqual(launch.revision, FROZEN_REVISION)
        self.assertEqual(launch.gpu_uuid, "4319ce22-3516-44fb-5efe-b0f88b98a042")
        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        self.assertEqual(pair.ordinal, 0)
        self.assertEqual(set(pair.p0), {"action", "output", "p0_stability"})
        self.assertEqual(pair.p0["action"], "P0")
        self.assertEqual(
            [candidate["action"] for candidate in pair.candidates],
            ["ZOOM", "EXPAND"],
        )
        self.assertFalse(pair.candidates[0]["feasible"])
        self.assertTrue(pair.candidates[1]["feasible"])
        self.assertEqual(
            pair.candidates[1]["output"],
            enabled["method_trace"]["steps"][1]["expand_audit"]
            ["candidate_stability"]["output"],
        )
        self.assertEqual(decisions[0].decision.action, "P0")
        self.assertEqual(
            before,
            json.dumps(
                [disabled, enabled, left, right], sort_keys=True, separators=(",", ":"),
            ),
        )

    def test_both_feasible_actions_extract_canonical_vstar_projection(self):
        disabled, enabled, left, right = self.factory.pair(
            "vstar", zoom_feasible=True,
        )

        _, pairs, decisions = self.validate(
            "vstar", disabled, enabled, left, right,
        )

        self.assertEqual(
            [(item["action"], item["feasible"]) for item in pairs[0].candidates],
            [("ZOOM", True), ("EXPAND", True)],
        )
        for index, action in enumerate(("ZOOM", "EXPAND")):
            audit = enabled["method_trace"]["steps"][index][f"{action.lower()}_audit"]
            self.assertEqual(
                pairs[0].candidates[index]["output"],
                audit["candidate_stability"]["output"],
            )
        self.assertEqual(decisions[0].decision.action, "P0")

    def test_freeze_fails_closed_if_actual_phase5_tie_order_drifts(self):
        disabled, enabled, left, right = self.factory.pair("vstar")
        with (
            patch.object(phase6, "_FROZEN_COMBINED_INFERENCE_REVISION", FROZEN_REVISION),
            patch.object(phase4, "_support_prompt_sha256", return_value=PROMPT_SHA),
        ):
            pairs = validate_and_extract_combined_pairs(
                "vstar", [disabled], [enabled],
                disabled_launch_manifest=left, enabled_launch_manifest=right,
            )
            with (
                patch.object(phase5, "_ACTION_NAME_ORDER", ("ZOOM", "EXPAND")),
                self.assertRaises(RuntimeError),
            ):
                freeze_combined_decisions(
                    "vstar", [disabled], [enabled],
                    disabled_launch_manifest=left,
                    enabled_launch_manifest=right,
                )
            with (
                patch.object(phase6, "SELECTOR_SOURCE_SHA256", "0" * 64),
                self.assertRaises(RuntimeError),
            ):
                freeze_combined_decisions(
                    "vstar", [disabled], [enabled],
                    disabled_launch_manifest=left,
                    enabled_launch_manifest=right,
                )

    def test_exact_phase5_binding_and_validated_dtos_are_delegated_unchanged(self):
        disabled, enabled, left, right = self.factory.pair("vstar")
        _, pairs, _ = self.validate("vstar", disabled, enabled, left, right)
        pair = pairs[0]
        with (
            patch.object(phase6, "_FROZEN_COMBINED_INFERENCE_REVISION", FROZEN_REVISION),
            patch.object(phase4, "_support_prompt_sha256", return_value=PROMPT_SHA),
            patch.object(
                phase6, "select_unified_state", wraps=phase5.select_unified_state,
            ) as selector,
        ):
            frozen = freeze_combined_decisions(
                "vstar", [disabled], [enabled],
                disabled_launch_manifest=left,
                enabled_launch_manifest=right,
            )
        selector.assert_called_once_with(pair.p0, list(pair.candidates))
        self.assertEqual(frozen[0].decision.action, "P0")
        self.assertEqual(phase6.STABILITY_GAIN_THRESHOLD, 0.25)
        self.assertEqual(phase6.TIE_ORDER, ("EXPAND", "ZOOM"))
        self.assertIs(phase6.select_unified_state, phase5.select_unified_state)

    def test_post_validation_dto_replacement_cannot_reuse_trusted_provenance(self):
        disabled, enabled, left, right = self.factory.pair("vstar")
        _, pairs, _ = self.validate("vstar", disabled, enabled, left, right)
        self.assertNotIn("_VALIDATION_SEAL_KEY", vars(phase6))
        self.assertNotIn("_pair_validation_seal", vars(phase6))
        pair = pairs[0]
        for forged in (
            replace(
                pair,
                _p0_json=_canonical({
                    "action": "P0", "output": "forged",
                    "p0_stability": {"confidence": 0.0},
                }),
            ),
            replace(
                pair,
                _candidate_jsons=(
                    pair._candidate_jsons[0],
                    _canonical({
                        "action": "EXPAND", "feasible": True,
                        "output": "forged", "candidate_stability": {"confidence": 1.0},
                    }),
                ),
            ),
        ):
            with self.subTest(field=forged):
                with self.assertRaises(TypeError):
                    freeze_combined_decisions((forged,))

    def test_hr_four_shuffle_projection_and_all_infeasible_edges_are_canonical(self):
        disabled, enabled, left, right = self.factory.pair("hr-bench_4k")
        _, pairs, decisions = self.validate(
            "hr-bench_4k", disabled, enabled, left, right,
        )
        expected = enabled["method_trace"]["steps"][1]["expand_audit"]
        self.assertEqual(pairs[0].candidates[1]["output"], expected["candidate_stability"]["output"])
        self.assertIsInstance(pairs[0].candidates[1]["output"], list)
        self.assertEqual(decisions[0].decision.action, "P0")

        disabled, enabled, left, right = self.factory.pair(
            "hr-bench_8k", expand_mode="no_batch",
        )
        _, pairs, decisions = self.validate(
            "hr-bench_8k", disabled, enabled, left, right,
        )
        self.assertEqual(
            [(item["feasible"], item["output"], item["candidate_stability"])
             for item in pairs[0].candidates],
            [(False, None, None), (False, None, None)],
        )
        self.assertEqual(decisions[0].decision.output, pairs[0].p0["output"])

    def test_expand_failure_and_budget_edges_preserve_cumulative_chain(self):
        for mode, expected_status in (
            ("model_failed", "model_failed"),
            ("budget_rejected", "budget_rejected"),
        ):
            with self.subTest(mode=mode):
                disabled, enabled, left, right = self.factory.pair(
                    "vstar", expand_mode=mode,
                )
                _, pairs, decisions = self.validate(
                    "vstar", disabled, enabled, left, right,
                )
                audit = enabled["method_trace"]["steps"][1]["expand_audit"]
                self.assertEqual(audit["batch_result"]["status"], expected_status)
                self.assertFalse(pairs[0].candidates[1]["feasible"])
                self.assertEqual(decisions[0].decision.action, "P0")

    def test_action_order_step_indices_and_standalone_trace_stitching_reject(self):
        disabled, enabled, left, right = self.factory.pair("vstar")
        cases = []
        forged = copy.deepcopy(enabled)
        forged["method_trace"]["steps"][0], forged["method_trace"]["steps"][1] = (
            forged["method_trace"]["steps"][1], forged["method_trace"]["steps"][0]
        )
        cases.append(forged)
        forged = copy.deepcopy(enabled)
        forged["method_trace"]["steps"][1]["step"] = 0
        cases.append(forged)
        forged = copy.deepcopy(enabled)
        forged["method_trace"]["steps"] = [
            forged["method_trace"]["steps"][0], forged["method_trace"]["steps"][2],
        ]
        cases.append(forged)
        for forged in cases:
            forged["_eg_run_fingerprint"] = enabled["_eg_run_fingerprint"]
            self.assert_rejected("vstar", disabled, forged, left, right)

    def test_ledger_chain_p0_anchor_answer_and_action_state_forgery_reject(self):
        disabled, enabled, left, right = self.factory.pair(
            "vstar", zoom_feasible=True,
        )
        mutations = []

        forged = copy.deepcopy(enabled)
        forged["method_trace"]["steps"][1]["expand_audit"]["batch_result"][
            "ledger_before"
        ]["mllm_calls"] += 1
        mutations.append(forged)

        forged = copy.deepcopy(enabled)
        forged["method_trace"]["steps"][0]["zoom_audit"]["p0_anchor"][
            "emitted_answer"
        ] = 0
        mutations.append(forged)

        forged = copy.deepcopy(enabled)
        forged["method_trace"]["steps"][1]["expand_audit"]["p0_stability"][
            "confidence"
        ] = 0.0
        mutations.append(forged)

        forged = copy.deepcopy(enabled)
        forged["method_trace"]["steps"][1]["expand_audit"]["candidate_stability"][
            "confidence"
        ] = 0.0
        mutations.append(forged)

        forged = copy.deepcopy(enabled)
        forged["method_trace"]["steps"][1]["expand_audit"]["batch_result"][
            "candidate_answer"
        ]["losses"] = [0.0, 10.0]
        mutations.append(forged)

        forged = copy.deepcopy(enabled)
        forged["method_trace"]["steps"][1]["expand_audit"]["support_contract_status"] = (
            "not_observed"
        )
        mutations.append(forged)

        forged = copy.deepcopy(enabled)
        forged["method_trace"]["steps"][1]["expand_audit"]["selection_decision"][
            "normalized_edge_gap"
        ] = 0.5
        mutations.append(forged)

        forged = copy.deepcopy(enabled)
        forged["method_trace"]["steps"][1]["expand_audit"]["batch_result"][
            "batch_plan"
        ]["source_width"] += 1
        mutations.append(forged)

        for forged in mutations:
            self.assert_rejected("vstar", disabled, forged, left, right)

    def test_launch_config_revision_fingerprint_partition_gpu_model_and_cross_run_reject(self):
        disabled, enabled, left, right = self.factory.pair("vstar")
        cases = []
        forged = copy.deepcopy(right)
        forged["hardware"]["gpu_uuids"] = ["f9f7acc2-e6da-475f-a647-f41653eed61e"]
        cases.append((enabled, forged))
        forged = copy.deepcopy(right)
        forged["selected_partition"]["ordinals"] = [9]
        cases.append((enabled, forged))
        forged = copy.deepcopy(right)
        forged["config"]["loaded"]["quick_gate"] = 0.8
        cases.append((enabled, forged))
        forged = copy.deepcopy(right)
        forged["artifacts"]["processor"]["sha256"] = "0" * 64
        cases.append((enabled, forged))
        forged_row = copy.deepcopy(enabled)
        forged_row["_eg_run_fingerprint"] = "0" * 64
        cases.append((forged_row, right))
        for row, manifest in cases:
            self.assert_rejected("vstar", disabled, row, left, manifest)

        forged_left = copy.deepcopy(left)
        forged_right = copy.deepcopy(right)
        files = [{"path": "forged.py", "sha256": "0" * 64}]
        for manifest in (forged_left, forged_right):
            manifest["code"]["manifest"] = {"files": copy.deepcopy(files)}
            manifest["code"]["manifest_sha256"] = canonical_sha256({"files": files})
            manifest["code"]["revision"] = canonical_sha256(files)
        forged_disabled = copy.deepcopy(disabled)
        forged_enabled = copy.deepcopy(enabled)
        for row, manifest in (
            (forged_disabled, forged_left), (forged_enabled, forged_right),
        ):
            row["_eg_code_revision"] = manifest["code"]["revision"]
            row["_eg_run_fingerprint"] = canonical_sha256(manifest)
        self.assert_rejected(
            "vstar", forged_disabled, forged_enabled, forged_left, forged_right,
        )

        with tempfile.TemporaryDirectory() as other_directory:
            _, foreign_enabled, _, foreign_right = CombinedPairFactory(
                other_directory
            ).pair("vstar")
            self.assert_rejected(
                "vstar", disabled, foreign_enabled, left, foreign_right,
            )

    def test_physical_source_mutation_rejects_and_is_not_rewritten(self):
        disabled, enabled, left, right = self.factory.pair("vstar")
        path = self.factory.expand.image_path
        original = path.read_bytes()
        try:
            Image.new("RGB", (400, 300), (9, 8, 7)).save(path)
            mutated = path.read_bytes()
            self.assertNotEqual(mutated, original)
            self.assert_rejected("vstar", disabled, enabled, left, right)
            self.assertEqual(path.read_bytes(), mutated)
        finally:
            path.write_bytes(original)

    def test_evaluator_poison_and_labels_never_enter_dtos_or_change_decisions(self):
        disabled, enabled, left, right = self.factory.pair("hr-bench_4k")
        _, clean_pairs, clean = self.validate(
            "hr-bench_4k", disabled, enabled, left, right,
        )
        poisoned_disabled = copy.deepcopy(disabled)
        poisoned_enabled = copy.deepcopy(enabled)
        poison = {
            "benchmark": "POISON_BENCHMARK", "resolution": "POISON_8K",
            "category": "POISON_CATEGORY", "evaluator_label": "POISON_LABEL",
        }
        poisoned_disabled.update(poison)
        poisoned_enabled.update(poison)
        _, poisoned_pairs, poisoned = self.validate(
            "hr-bench_4k", poisoned_disabled, poisoned_enabled, left, right,
        )
        encoded = _canonical({
            "p0": poisoned_pairs[0].p0,
            "candidates": poisoned_pairs[0].candidates,
            "decision": poisoned[0].decision.__dict__,
        })
        self.assertNotIn("POISON", encoded)
        self.assertEqual(clean[0].decision, poisoned[0].decision)
        self.assertEqual(clean.canonical_digest, poisoned.canonical_digest)
        self.assertEqual(clean_pairs[0].input_identity, poisoned_pairs[0].input_identity)

    def test_complete_partition_freezes_before_distinct_label_access_boundary(self):
        disabled, enabled, left, right = self.factory.pair("hr-bench_4k")

        class LabelTrap(dict):
            opened = False
            accesses = 0

            def get(self, key, default=None):
                if key == "answer":
                    type(self).accesses += 1
                    if not type(self).opened:
                        raise AssertionError("label accessed before frozen boundary")
                return super().get(key, default)

            def __getitem__(self, key):
                if key == "answer":
                    type(self).accesses += 1
                    if not type(self).opened:
                        raise AssertionError("label accessed before frozen boundary")
                return super().__getitem__(key)

        trapped_disabled = LabelTrap(disabled)
        trapped_enabled = LabelTrap(enabled)
        _, pairs, frozen = self.validate(
            "hr-bench_4k", trapped_disabled, trapped_enabled, left, right,
        )
        self.assertEqual(LabelTrap.accesses, 0)
        self.assertEqual(len(frozen), len(pairs))
        LabelTrap.opened = True
        self.assertEqual(trapped_enabled["answer"], enabled["answer"])
        self.assertGreater(LabelTrap.accesses, 0)

    def test_freeze_rejects_empty_raw_or_legacy_extracted_sequences(self):
        disabled, enabled, left, right = self.factory.pair("vstar")
        _, pairs, _ = self.validate("vstar", disabled, enabled, left, right)
        with (
            patch.object(phase6, "_FROZEN_COMBINED_INFERENCE_REVISION", FROZEN_REVISION),
            patch.object(phase4, "_support_prompt_sha256", return_value=PROMPT_SHA),
            self.assertRaises(ValueError),
        ):
            freeze_combined_decisions(
                "vstar", [], [], disabled_launch_manifest=left,
                enabled_launch_manifest=right,
            )
        with self.assertRaises(TypeError):
            freeze_combined_decisions(pairs)

    def test_frozen_batch_digest_is_recomputable_without_selector_or_labels(self):
        disabled, enabled, left, right = self.factory.pair("vstar")
        _, _, frozen = self.validate("vstar", disabled, enabled, left, right)
        with patch.object(
            phase6, "select_unified_state",
            side_effect=AssertionError("selector must not be reopened"),
        ):
            self.assertEqual(frozen.recompute_digest(), frozen.canonical_digest)
            frozen.verify_digest()
        forged_record = replace(
            frozen[0],
            _decision_json=_canonical({
                "action": "ZOOM", "status": "selected_candidate",
                "output": 0, "stability_gain": 0.9,
            }),
        )
        forged = replace(frozen, records=(forged_record,))
        self.assertNotEqual(forged.recompute_digest(), forged.canonical_digest)
        with self.assertRaises(ValueError):
            forged.verify_digest()


if __name__ == "__main__":
    unittest.main()
