import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cvsearch.eval.phase3_zoom_oracle as phase3
from cvsearch.eval.phase3_zoom_oracle import (
    PairExpectation,
    canonical_output_digest,
    score_dev_paths,
    score_paired_rows,
    validate_launch_pair,
)
from cvsearch.evidence_gap.answers import aggregate_hr_answers, aggregate_vstar_losses
from cvsearch.evidence_gap.method import load_method_config
from cvsearch.evidence_gap.provenance import canonical_sha256
from cvsearch.evidence_gap.types import (
    EVIDENCE_SUPPORT_TRANSFORM,
    EvidenceSupportResult,
    _validate_batch_plan,
    sanitize_evidence_requirements,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "reproduction/evidence_gap/configs"
DISABLED_CONFIG = CONFIG_ROOT / "dev_unified_zoom_disabled_gamma000_budget512.json"
ENABLED_CONFIG = CONFIG_ROOT / "dev_unified_zoom_oracle_gamma000_budget512.json"


HR_OPTIONS = [
    "A. cat\nB. dog\nC. bird\nD. fish",
    "A. dog\nB. cat\nC. fish\nD. bird",
    "A. bird\nB. fish\nC. cat\nD. dog",
    "A. fish\nB. bird\nC. dog\nD. cat",
]
P0_HR = ["A", "B", "C", "D"]
CANDIDATE_HR = ["B", "A", "D", "C"]


def _processor_fingerprint(checkpoint="frozen/qwen"):
    return {
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "attention_implementation": "flash_attention_2",
        "checkpoint_resolved": checkpoint,
        "checkpoint_snapshot": Path(checkpoint).name,
        "checkpoint_transformers_version": "4.41.2",
        "effective_padding_side": "left",
        "image_processor_class": (
            "transformers.models.qwen2_vl.image_processing_qwen2_vl_fast."
            "Qwen2VLImageProcessorFast"
        ),
        "image_processor_is_fast": True,
        "image_processor_max_pixels": 12_845_056,
        "image_processor_merge_size": 2,
        "image_processor_min_pixels": 3_136,
        "image_processor_patch_size": 14,
        "image_processor_temporal_patch_size": 2,
        "model_config_class": (
            "transformers.models.qwen2_5_vl.configuration_qwen2_5_vl."
            "Qwen2_5_VLConfig"
        ),
        "model_device": "cuda:0", "model_dtype": "torch.bfloat16",
        "model_type": "qwen2_5_vl",
        "processor_class": (
            "transformers.models.qwen2_5_vl.processing_qwen2_5_vl."
            "Qwen2_5_VLProcessor"
        ),
        "runtime_pillow_version": "11.0.0",
        "runtime_torch_version": "2.7.1+cu118",
        "runtime_transformers_version": "4.57.0",
        "tokenizer_class": (
            "transformers.models.qwen2.tokenization_qwen2_fast."
            "Qwen2TokenizerFast"
        ),
        "tokenizer_padding_side_default": "right",
        "wrapper_device": "cuda:0", "wrapper_dtype": "torch.bfloat16",
        "wrapper_flash_attention": True,
    }


def _model_manifest(checkpoint="frozen/qwen"):
    return {
        "artifacts": {
            "qwen": {
                "kind": "directory", "path": checkpoint, "sha256": "d" * 64,
            },
            "processor": {
                "kind": "processor_files", "path": checkpoint,
                "sha256": "e" * 64,
            },
        },
        "environment": {"packages": {
            "pillow": "11.0.0", "torch": "2.7.1+cu118",
            "transformers": "4.57.0",
        }},
    }


def _configs():
    return load_method_config(str(DISABLED_CONFIG)), load_method_config(str(ENABLED_CONFIG))


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _answer_record(answer_type, output, options):
    if answer_type == "option_list":
        record = aggregate_hr_answers(options, output)
        record.output = copy.deepcopy(output)
        record.selected_from = "cvsearch_anchor"
    else:
        losses = [0.8, 0.1] if output == 1 else [0.1, 0.9]
        record = aggregate_vstar_losses([losses])
        record.selected_from = "search"
    return record.to_dict()


def _merge_identity(crop):
    material = {
        "per_descriptor_crop_xyxy": [list(crop)],
        "merged_crop_xyxy": [list(crop)],
        "union_crop_xyxy": list(crop),
    }
    return dict(material, identity_sha256=hashlib.sha256(_canonical(material).encode()).hexdigest())


def _support_payload(plan, plan_hash, observation_name, p_yes):
    requirements = sanitize_evidence_requirements(({
        "kind": "target_detail", "target": "object",
        "requirements": ["presence", "visual_detail"],
    },))
    observation = plan[f"{observation_name}_observation"]
    identity = {
        key: observation[key] for key in (
            "canonical_keys", "renderer_identities", "rendered_mode",
            "rendered_size", "view_sha256",
        )
    }
    result = EvidenceSupportResult(
        requirements=requirements,
        requirement_set_id=EvidenceSupportResult.requirement_set_id_for(requirements),
        observation_identity=_canonical(identity),
        prompt_version=plan["prompt_version"],
        prompt_template_sha256=plan["prompt_template_sha256"],
        prompt_sha256=plan[f"{observation_name}_prompt_sha256"],
        processor_mode=plan["processor_mode"],
        processor_fingerprint_json=_canonical(plan["processor_fingerprint"]),
        checkpoint=plan["checkpoint"],
        yes_tokenization=(9454,), no_tokenization=(2753,),
        yes_token_id=9454, no_token_id=2753,
        p_yes_transform=EVIDENCE_SUPPORT_TRANSFORM,
        yes_logit=1.0, no_logit=0.0,
        p_yes=p_yes, p_no=1.0 - p_yes,
        support_avg=p_yes, support_min=p_yes,
        observation_mode=observation["rendered_mode"],
        observation_size=tuple(observation["rendered_size"]),
        view_sha256=observation["view_sha256"], elapsed_seconds=0.01,
        accounted_pixels=plan["pixels_per_logical_forward"],
        batch_plan_hash=plan_hash,
    )
    return result.to_dict()


def _plan(answer_type, options):
    source = {"mode": "RGB", "size": [100, 100], "pixel_sha256": "0" * 64}
    source_key = _canonical(source)
    bbox = [20, 20, 40, 40]
    current_key = _canonical({"bbox": bbox, "depth": 1, "render_level": 0})
    source_renderer = _canonical({
        "source_image_key": source_key, "renderer_kind": "fine",
        "bbox": bbox, "render_level": 0,
    })
    current_crop = [10, 10, 70, 70]
    candidate_crop = [25, 25, 55, 55]
    zoom_renderer = _canonical({
        "action": "P2C_ZOOM", "candidate_crop_xyxy": candidate_crop,
        "candidate_view_size": 112, "current_key": current_key,
        "render_policy": "native_coordinate_crop_view_size_div3",
        "source_image_key": source_key,
    })
    zoom_key = "zoom-" + hashlib.sha256(zoom_renderer.encode()).hexdigest()
    descriptor = {
        "canonical_key": current_key, "bbox_original": bbox, "depth": 1,
        "render_level": 0, "posterior_score": 0.9,
        "first_seen_ordinal": 0, "tree_scope": "main", "crop_origin": [0, 0],
        "source_image_key": source_key, "source": "fine", "renderer_kind": "fine",
        "renderer_identity": source_renderer,
    }
    mapping = {
        "descriptor_index": 0, "current_key": current_key, "zoom_key": zoom_key,
        "bbox_original": bbox, "depth": 1, "render_level": 0,
        "posterior_score": 0.9, "first_seen_ordinal": 0,
        "tree_scope": "main", "crop_origin": [0, 0],
        "source_image_key": source_key, "source": "fine", "renderer_kind": "fine",
        "source_renderer_identity": source_renderer,
        "zoom_renderer_identity": zoom_renderer,
        "current_crop_xyxy": current_crop, "candidate_crop_xyxy": candidate_crop,
    }
    current_observation = {
        "canonical_keys": [current_key], "renderer_identities": [source_renderer],
        "rendered_mode": "RGB", "rendered_size": [64, 64],
        "view_sha256": "1" * 64, "descriptors": [descriptor],
    }
    candidate_observation = {
        "canonical_keys": [zoom_key], "renderer_identities": [zoom_renderer],
        "rendered_mode": "RGB", "rendered_size": [48, 48],
        "view_sha256": "2" * 64,
        "descriptors": [{
            "canonical_key": zoom_key, "renderer_identity": zoom_renderer,
            "source_descriptor": descriptor,
        }],
    }
    requirements = sanitize_evidence_requirements(({
        "kind": "target_detail", "target": "object",
        "requirements": ["presence", "visual_detail"],
    },))
    answer_calls = 4 if answer_type == "option_list" else 1 + len(options)
    total_calls = 2 + answer_calls
    plan = {
        "schema_version": 1, "batch_kind": "p2c_post_anchor_coordinate_zoom",
        "answer_type": answer_type, "source_identity": source,
        "source_width": 100, "source_height": 100, "accounted_source_area": 10_000,
        "render_policy": "native_coordinate_crop_view_size_div3",
        "base_view_size": 336, "candidate_view_size": 112,
        "current_keys": [current_key], "zoom_keys": [zoom_key],
        "coordinate_mapping": [mapping],
        "current_merge_identity": _merge_identity(current_crop),
        "candidate_merge_identity": _merge_identity(candidate_crop),
        "current_observation": current_observation,
        "candidate_observation": candidate_observation,
        "candidate_answer_input_sha256": candidate_observation["view_sha256"],
        "q0_sha256": hashlib.sha256(b"q0").hexdigest(),
        "requirement_order": [item.requirement_id for item in requirements],
        "requirement_set_id": EvidenceSupportResult.requirement_set_id_for(requirements),
        "prompt_version": "qwen_answer_free_evidence_support_v1",
        "prompt_template_sha256": "0ba54d5ef5190f162d2036721693ae1af1a499f088e12a5106f8c9b692dfcb86",
        "current_prompt_sha256": "4" * 64, "candidate_prompt_sha256": "4" * 64,
        "processor_mode": "single_rendered_view_chat_left_padding_final_yes_no_logits",
        "processor_fingerprint": _processor_fingerprint(),
        "checkpoint": "frozen/qwen",
        "yes_tokenization": [9454], "no_tokenization": [2753],
        "yes_token_id": 9454, "no_token_id": 2753,
        "p_yes_transform": EVIDENCE_SUPPORT_TRANSFORM,
        "verifier_status": "disabled_same_checkpoint_unpromoted",
        "current_support_calls": 1, "candidate_support_calls": 1,
        "candidate_answer_calls": answer_calls, "candidate_option_count": len(options),
        "pixels_per_logical_forward": 10_000, "total_calls": total_calls,
        "total_pixels": total_calls * 10_000,
    }
    payload, plan_hash, full = _validate_batch_plan(plan)
    assert full
    return dict(payload, plan_hash=plan_hash)


def _budget(calls, pixels):
    return {
        "max_mllm_calls": 512, "max_processed_pixels": 10_000_000_000,
        "mllm_calls": calls, "processed_pixels": pixels,
    }


def _forced_step(index, answer, budget):
    return {
        "step": index, "action": "FORCED_RETURN", "gap_fallback_used": False,
        "elapsed_seconds": 0.0, "focus_key": None, "feasible_actions": [],
        "gaps": {}, "no_op_reason": "minimal_v1 has no certified-stop controller",
        "answer": copy.deepcopy(answer), "support_avg": 0.0, "support_min": 0.0,
        "certified": False, "budget": copy.deepcopy(budget),
    }


def _p0_trace_fields():
    return {
        "query_plan": {
            "main_query": "q0", "targets": ["object"],
            "augmented_queries": ["locate and inspect object"],
            "evidence_items": [{
                "kind": "target_detail", "target": "object",
                "requirements": ["presence", "visual_detail"],
            }],
            "global_scope_required": False, "fallback_used": True,
        },
        "candidate_ranks": [], "history": [], "final_boxes": [],
        "method_mode": "root_search_fallback", "cvsearch_search_mode": 1,
        "root_ans_conf": 0.1, "num_pop": [], "num_zoom_in": [],
        "num_zoom_out": [], "effective_ranking_query": "q0",
    }


def _row(benchmark, ordinal=0):
    disabled_config, enabled_config = _configs()
    answer_type = "logits_match" if benchmark == "vstar" else "option_list"
    options = ["cat", "dog"] if answer_type == "logits_match" else HR_OPTIONS
    p0 = 1 if answer_type == "logits_match" else P0_HR
    candidate = (
        {"winner": 0, "losses": [0.1, 0.9]}
        if answer_type == "logits_match" else CANDIDATE_HR
    )
    truth = 0 if answer_type == "logits_match" else CANDIDATE_HR
    p0_record = _answer_record(answer_type, p0, options)
    before = _budget(10, 10_000)
    plan = _plan(answer_type, options)
    after = _budget(
        before["mllm_calls"] + plan["total_calls"],
        before["processed_pixels"] + plan["total_pixels"],
    )
    base = {
        "question": "q0", "options": options, "answer_type": answer_type,
        "input_image": f"image/{ordinal}.jpg", "answer": copy.deepcopy(truth),
        "output": copy.deepcopy(p0), "_eg_ordinal": ordinal,
        "_eg_code_revision": "a" * 64, "_eg_run_fingerprint": "b" * 64,
    }
    disabled = copy.deepcopy(base)
    disabled["method_trace"] = {
        **_p0_trace_fields(),
        "steps": [_forced_step(0, p0_record, before)],
        "final_answer": copy.deepcopy(p0_record), "anchor_answer": copy.deepcopy(p0_record),
        "budget": before, "elapsed_seconds": 1.0, "termination": "FORCED_RETURN",
        "config_id": disabled_config["config_id"], "effective_config": disabled_config,
        "budget_interrupted": False, "pixel_accounting": disabled_config["pixel_accounting"],
        "anchor_state_score": 0.1, "selected_state_score": 0.1,
        "replacement_margin": 0.0, "support_status": "not_observed",
    }

    current_support = _support_payload(plan, plan["plan_hash"], "current", 0.4)
    candidate_support = _support_payload(plan, plan["plan_hash"], "candidate", 0.8)
    batch = {
        "status": "success", "admitted": True, "charged": True,
        "batch_plan": plan, "batch_plan_hash": plan["plan_hash"],
        "current_support": current_support, "candidate_support": candidate_support,
        "candidate_answer": copy.deepcopy(candidate), "failure_phase": None,
        "failure_reason": None, "exception_type": None,
        "executed_stages": (
            ["current_support", "candidate_support", "vstar_answer"]
            if answer_type == "logits_match" else
            ["current_support", "candidate_support", "hr_answer_0", "hr_answer_1",
             "hr_answer_2", "hr_answer_3"]
        ),
        "elapsed_seconds": 0.25, "ledger_before": before, "ledger_after": after,
        "verifier_status": "disabled_same_checkpoint_unpromoted",
        "verifier_avg": None, "verifier_min": None, "promotable": False,
    }
    p0_stability = copy.deepcopy(p0_record)
    if answer_type == "option_list":
        p0_stability = aggregate_hr_answers(options, p0).to_dict()
        p0_stability["output"] = copy.deepcopy(p0)
        p0_stability["selected_from"] = "cvsearch_raw"
        candidate_stability = aggregate_hr_answers(options, candidate).to_dict()
        producing_phase = "cvsearch_raw"
        cvsearch_raw = copy.deepcopy(p0)
    else:
        candidate_stability = aggregate_vstar_losses([candidate["losses"]]).to_dict()
        producing_phase = "search"
        cvsearch_raw = p0
    descriptor = copy.deepcopy(plan["current_observation"]["descriptors"])
    audit = {
        "p0_anchor": {
            "emitted_answer": copy.deepcopy(p0), "cvsearch_raw": cvsearch_raw,
            "producing_phase": producing_phase, "node_keys": plan["current_keys"],
            "support_view": descriptor,
        },
        "current_keys": plan["current_keys"], "zoom_keys": plan["zoom_keys"],
        "coordinate_mapping": plan["coordinate_mapping"],
        "render_policy": plan["render_policy"], "base_view_size": 336,
        "candidate_view_size": 112, "batch_result": batch,
        "current_gap_support": current_support, "candidate_gap_support": candidate_support,
        "uncertainty": p0_stability["uncertainty"],
        "uncalibrated_g_zoom_proxy": 0.6, "support_delta": 0.4,
        "support_proxy_status": "audit_only_uncalibrated",
        "p0_stability": p0_stability, "candidate_stability": candidate_stability,
        "feasible": True, "normalized_actual_cost": plan["total_calls"] / 512,
        "support_contract_status": "matched", "coverage_status": "not_observed",
        "verifier_status": "disabled_same_checkpoint_unpromoted",
        "verifier_avg": None, "verifier_min": None, "score_margin": None,
        "score_status": "unavailable_missing_verifier_coverage",
        "replacement_reason": "replacement_disabled_p2c",
    }
    focus = "zoom-aggregate-" + hashlib.sha256(
        json.dumps(plan["zoom_keys"], separators=(",", ":")).encode()
    ).hexdigest()
    zoom_step = {
        "step": 0, "action": "ZOOM", "gap_fallback_used": False,
        "elapsed_seconds": 0.0, "focus_key": focus, "feasible_actions": ["ZOOM"],
        "gaps": {"g_zoom_proxy_audit_only": 0.6}, "no_op_reason": None,
        "answer": copy.deepcopy(p0_record), "support_avg": 0.0, "support_min": 0.0,
        "certified": False, "budget": after, "zoom_audit": audit,
    }
    enabled = copy.deepcopy(base)
    enabled["_eg_run_fingerprint"] = "c" * 64
    enabled["method_trace"] = {
        **_p0_trace_fields(),
        "steps": [zoom_step, _forced_step(1, p0_record, after)],
        "final_answer": copy.deepcopy(p0_record), "anchor_answer": copy.deepcopy(p0_record),
        "budget": after, "elapsed_seconds": 1.5, "termination": "FORCED_RETURN",
        "config_id": enabled_config["config_id"], "effective_config": enabled_config,
        "budget_interrupted": False, "pixel_accounting": enabled_config["pixel_accounting"],
        "anchor_state_score": None, "selected_state_score": None,
        "replacement_margin": None, "support_status": "observed_answer_free_audit_only",
    }
    return disabled, enabled


def _refresh_plan_hash(row):
    audit = row["method_trace"]["steps"][0]["zoom_audit"]
    batch = audit["batch_result"]
    plan = batch["batch_plan"]
    plan.pop("plan_hash", None)
    plan_hash = hashlib.sha256(_canonical(plan).encode()).hexdigest()
    plan["plan_hash"] = plan_hash
    batch["batch_plan_hash"] = plan_hash
    for name in ("current_support", "candidate_support"):
        support = batch.get(name)
        if support is not None:
            support["batch_plan_hash"] = plan_hash
    return plan_hash


def _as_model_failed(enabled):
    row = copy.deepcopy(enabled)
    step = row["method_trace"]["steps"][0]
    audit = step["zoom_audit"]
    batch = audit["batch_result"]
    batch.update(
        status="model_failed", candidate_answer=None,
        candidate_support=None, failure_phase="candidate_support",
        failure_reason="failed candidate support", exception_type="RuntimeError",
        executed_stages=["current_support"],
    )
    audit.update(
        candidate_gap_support=None, uncalibrated_g_zoom_proxy=None,
        support_delta=None, candidate_stability=None, feasible=False,
        support_contract_status="not_observed", replacement_reason=None,
    )
    step.update(
        feasible_actions=[], gaps={}, no_op_reason="zoom_model_failed",
    )
    row["method_trace"]["support_status"] = "zoom_model_failed"
    return row


def _as_render_noop(enabled):
    row = copy.deepcopy(enabled)
    step = row["method_trace"]["steps"][0]
    audit = step["zoom_audit"]
    batch = audit["batch_result"]
    plan = batch["batch_plan"]
    plan["candidate_observation"]["view_sha256"] = plan["current_observation"]["view_sha256"]
    plan["candidate_answer_input_sha256"] = plan["candidate_observation"]["view_sha256"]
    _refresh_plan_hash(row)
    before = copy.deepcopy(batch["ledger_before"])
    batch.update(
        status="render_noop", admitted=False, charged=False,
        current_support=None, candidate_support=None, candidate_answer=None,
        failure_phase="preflight", failure_reason="aggregate_render_unchanged",
        exception_type=None, executed_stages=[], ledger_after=before,
    )
    audit.update(
        current_gap_support=None, candidate_gap_support=None,
        uncalibrated_g_zoom_proxy=None, support_delta=None,
        candidate_stability=None, feasible=False, normalized_actual_cost=0.0,
        support_contract_status="not_observed", replacement_reason=None,
    )
    step.update(
        feasible_actions=[], gaps={}, no_op_reason="zoom_render_noop", budget=before,
    )
    row["method_trace"].update(
        budget=before, support_status="zoom_render_noop",
    )
    row["method_trace"]["steps"][1]["budget"] = copy.deepcopy(before)
    return row


def _as_unavailable_noop(enabled):
    row = copy.deepcopy(enabled)
    step = row["method_trace"]["steps"][0]
    audit = step["zoom_audit"]
    before = copy.deepcopy(audit["batch_result"]["ledger_before"])
    audit.update(
        zoom_keys=[], coordinate_mapping=[], batch_result=None,
        current_gap_support=None, candidate_gap_support=None,
        uncalibrated_g_zoom_proxy=None, support_delta=None,
        candidate_stability=None, feasible=False, normalized_actual_cost=None,
        support_contract_status="not_observed", replacement_reason=None,
    )
    audit["p0_anchor"]["support_view"] = None
    step.update(
        focus_key=None, feasible_actions=[], gaps={},
        no_op_reason="zoom_p0_support_view_unavailable", budget=before,
    )
    row["method_trace"].update(
        budget=before, support_status="zoom_p0_support_view_unavailable",
    )
    row["method_trace"]["steps"][1]["budget"] = copy.deepcopy(before)
    return row


def _as_root_empty_noop(enabled):
    row = _as_unavailable_noop(enabled)
    step = row["method_trace"]["steps"][0]
    audit = step["zoom_audit"]
    audit["current_keys"] = []
    audit["p0_anchor"].update(
        producing_phase="root", cvsearch_raw=0, node_keys=[], support_view=[],
    )
    step["no_op_reason"] = "zoom_p0_support_view_empty"
    row["method_trace"]["support_status"] = "zoom_p0_support_view_empty"
    return row


class Phase3ZoomOracleTest(unittest.TestCase):
    def test_frozen_config_pair_changes_only_three_zoom_controls(self):
        disabled, enabled = _configs()
        changed = {key for key in enabled if enabled.get(key) != disabled.get(key)}
        self.assertEqual(changed, {
            "config_id", "p2c_zoom_enabled", "p2c_zoom_admission_mode",
        })
        self.assertFalse(disabled["p2c_zoom_enabled"])
        self.assertTrue(enabled["p2c_zoom_enabled"])
        self.assertFalse(enabled["p2c_zoom_replacement_enabled"])

    def test_hr_oracle_scores_raw_candidate_and_rejects_projection_substitution(self):
        disabled, enabled = _row("hr-bench_4k")
        expectation = PairExpectation(1, 4, 0, canonical_output_digest([disabled]))
        report = score_paired_rows(
            "hr-bench_4k", [disabled], [enabled], expectation,
            bootstrap_replicates=10_000, launch_manifest=_model_manifest(),
        )
        self.assertEqual(report["p0"]["correct"], 0)
        self.assertEqual(report["candidate_alone"]["correct"], 4)
        self.assertEqual(report["oracle"]["correct"], 4)
        self.assertFalse(report["audit"]["candidate_stability_output_scored"])

        corrupted = copy.deepcopy(enabled)
        corrupted["method_trace"]["steps"][0]["zoom_audit"][
            "candidate_stability"
        ]["output"] = copy.deepcopy(P0_HR)
        with self.assertRaises(ValueError):
            score_paired_rows(
                "hr-bench_4k", [disabled], [corrupted], expectation,
                bootstrap_replicates=10_000, launch_manifest=_model_manifest(),
            )

    def test_vstar_oracle_scores_raw_winner_and_losses(self):
        disabled, enabled = _row("vstar")
        report = score_paired_rows(
            "vstar", [disabled], [enabled],
            PairExpectation(1, 1, 0, canonical_output_digest([disabled])),
            bootstrap_replicates=10_000, launch_manifest=_model_manifest(),
        )
        self.assertEqual(report["candidate_alone"]["correct"], 1)
        self.assertEqual(report["oracle"]["corrected"], 1)
        self.assertEqual(report["candidate"]["status_counts"], {"success": 1})

        root = _as_root_empty_noop(enabled)
        self.assertNotEqual(
            root["method_trace"]["steps"][0]["zoom_audit"]["p0_anchor"][
                "cvsearch_raw"
            ],
            root["output"],
        )
        root_report = score_paired_rows(
            "vstar", [disabled], [root],
            PairExpectation(1, 1, 0, canonical_output_digest([disabled])),
            bootstrap_replicates=10_000, launch_manifest=_model_manifest(),
        )
        self.assertEqual(root_report["candidate"]["status_counts"], {"no_batch": 1})

    def test_exact_noop_model_failed_and_render_noop_are_infeasible(self):
        for name, transform, status in (
            ("unavailable", _as_unavailable_noop, "no_batch"),
            ("model_failed", _as_model_failed, "model_failed"),
            ("render_noop", _as_render_noop, "render_noop"),
        ):
            with self.subTest(name=name):
                disabled, enabled = _row("vstar")
                enabled = transform(enabled)
                report = score_paired_rows(
                    "vstar", [disabled], [enabled],
                    PairExpectation(1, 1, 0, canonical_output_digest([disabled])),
                    bootstrap_replicates=10_000, launch_manifest=_model_manifest(),
                )
                self.assertEqual(report["candidate"]["feasible_topics"], 0)
                self.assertEqual(report["candidate"]["status_counts"], {status: 1})

    def test_coherent_trace_geometry_plan_stage_p0_budget_and_identity_tampering_fails(self):
        disabled, enabled = _row("vstar")
        expectation = PairExpectation(1, 1, 0, canonical_output_digest([disabled]))
        score_paired_rows(
            "vstar", [disabled], [enabled], expectation,
            bootstrap_replicates=10_000, launch_manifest=_model_manifest(),
        )

        def geometry(row):
            plan = row["method_trace"]["steps"][0]["zoom_audit"]["batch_result"]["batch_plan"]
            plan["coordinate_mapping"][0]["bbox_original"] = [21, 20, 40, 40]
            _refresh_plan_hash(row)

        def budget(row):
            step = row["method_trace"]["steps"][0]
            batch = step["zoom_audit"]["batch_result"]
            batch["ledger_after"]["mllm_calls"] += 1
            step["budget"]["mllm_calls"] += 1
            row["method_trace"]["budget"]["mllm_calls"] += 1
            row["method_trace"]["steps"][1]["budget"]["mllm_calls"] += 1

        def plan_kind(row):
            plan = row["method_trace"]["steps"][0]["zoom_audit"][
                "batch_result"
            ]["batch_plan"]
            plan["batch_kind"] = "p2a_post_anchor_next"
            _refresh_plan_hash(row)

        def model_provenance(row):
            batch = row["method_trace"]["steps"][0]["zoom_audit"]["batch_result"]
            plan = batch["batch_plan"]
            plan["checkpoint"] = "forged/qwen"
            plan["processor_fingerprint"] = _processor_fingerprint("forged/qwen")
            for name in ("current_support", "candidate_support"):
                batch[name]["checkpoint"] = plan["checkpoint"]
                batch[name]["processor_fingerprint"] = copy.deepcopy(
                    plan["processor_fingerprint"]
                )
            _refresh_plan_hash(row)

        mutations = (
            ("config", lambda row: row["method_trace"]["effective_config"].__setitem__("quick_gate", 0.8)),
            ("ordinal", lambda row: row.__setitem__("_eg_ordinal", 9)),
            ("identity", lambda row: row.__setitem__("input_image", "other.jpg")),
            ("output", lambda row: row.__setitem__("output", 0)),
            ("p0", lambda row: row["method_trace"]["steps"][0]["zoom_audit"]["p0_anchor"].__setitem__("emitted_answer", 0)),
            ("step", lambda row: row["method_trace"]["steps"][0].__setitem__("action", "NEXT")),
            ("trace_extra", lambda row: row["method_trace"].__setitem__("forged", True)),
            ("query_plan", lambda row: row["method_trace"]["query_plan"].__setitem__("main_query", "forged")),
            ("trace_status", lambda row: row["method_trace"].__setitem__("support_status", "zoom_render_noop")),
            ("plan", plan_kind),
            ("model_provenance", model_provenance),
            ("stage", lambda row: row["method_trace"]["steps"][0]["zoom_audit"]["batch_result"].__setitem__("executed_stages", ["current_support"])),
            ("geometry", geometry),
            ("budget", budget),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                corrupted = copy.deepcopy(enabled)
                mutate(corrupted)
                with self.assertRaises((TypeError, ValueError)):
                    score_paired_rows(
                        "vstar", [disabled], [corrupted], expectation,
                        bootstrap_replicates=10_000,
                        launch_manifest=_model_manifest(),
                    )

        spoofed = _as_unavailable_noop(enabled)
        spoofed["method_trace"]["steps"][0]["no_op_reason"] = (
            "zoom_no_evidence_requirements"
        )
        spoofed["method_trace"]["support_status"] = "zoom_no_evidence_requirements"
        with self.assertRaises(ValueError):
            score_paired_rows(
                "vstar", [disabled], [spoofed], expectation,
                bootstrap_replicates=10_000, launch_manifest=_model_manifest(),
            )

    def test_launch_pair_binds_zoom_configs_and_nonconfig_identity(self):
        disabled, enabled = _row("vstar")
        disabled_config, enabled_config = _configs()
        common = {
            **_model_manifest(),
            "schema_version": 1, "benchmark": "vstar",
            "code": {"revision": "a" * 64},
            "selected_partition": {"ordinals": [0]},
            "hardware": {"gpu_uuids": ["GPU-test"]},
        }
        common["environment"]["python"] = "3.11"
        disabled_manifest = copy.deepcopy(common)
        disabled_manifest["config"] = {"loaded": disabled_config}
        enabled_manifest = copy.deepcopy(common)
        enabled_manifest["config"] = {"loaded": enabled_config}
        disabled["_eg_run_fingerprint"] = canonical_sha256(disabled_manifest)
        enabled["_eg_run_fingerprint"] = canonical_sha256(enabled_manifest)
        validate_launch_pair(
            [disabled], [enabled], disabled_manifest, enabled_manifest,
        )

        corrupted = copy.deepcopy(enabled_manifest)
        corrupted["hardware"]["gpu_uuids"] = ["GPU-other"]
        with self.assertRaises(ValueError):
            validate_launch_pair(
                [disabled], [enabled], disabled_manifest, corrupted,
            )
        corrupted = copy.deepcopy(enabled_manifest)
        corrupted["config"]["loaded"]["quick_gate"] = 0.8
        with self.assertRaises(ValueError):
            validate_launch_pair(
                [disabled], [enabled], disabled_manifest, corrupted,
            )

    def test_dev_report_binds_inputs_sidecars_evaluator_and_three_way_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            expected_jsonl = {}
            expected_sidecars = {}
            for benchmark in ("vstar", "hr-bench_4k", "hr-bench_8k"):
                pair = []
                expected_jsonl[benchmark] = {}
                expected_sidecars[benchmark] = {}
                for variant in ("disabled", "enabled"):
                    path = root / f"{benchmark}-{variant}.jsonl"
                    content = f"{benchmark}:{variant}\n".encode()
                    sidecar_content = f"{benchmark}:{variant}:manifest\n".encode()
                    path.write_bytes(content)
                    Path(f"{path}.launch-manifest.json").write_bytes(sidecar_content)
                    pair.append(path)
                    expected_jsonl[benchmark][variant] = hashlib.sha256(content).hexdigest()
                    expected_sidecars[benchmark][variant] = hashlib.sha256(sidecar_content).hexdigest()
                paths[benchmark] = tuple(pair)

            deltas = {
                "vstar": 0.0, "hr-bench_4k": 0.25, "hr-bench_8k": 0.25,
            }

            def fake_score(benchmark, *_args, **_kwargs):
                return {
                    "revision": "a" * 64,
                    "oracle": {"delta": deltas[benchmark]},
                }

            with (
                patch.object(phase3, "load_jsonl", return_value=[{}]),
                patch.object(phase3, "load_launch_manifest", return_value={}),
                patch.object(phase3, "validate_frozen_dev_identity"),
                patch.object(phase3, "validate_launch_pair", return_value={}),
                patch.object(phase3, "score_paired_rows", side_effect=fake_score),
            ):
                first = score_dev_paths(paths)
                second = score_dev_paths(paths)
                deltas["vstar"] = -0.01
                blocked = score_dev_paths(paths)

        identity = first["execution_identity"]
        self.assertEqual(identity["inference_revision"], "a" * 64)
        self.assertEqual(identity["input_jsonl_sha256"], expected_jsonl)
        self.assertEqual(identity["launch_manifest_sha256"], expected_sidecars)
        self.assertEqual(identity, second["execution_identity"])
        self.assertTrue(first["gate"]["p2d_warranted"])
        self.assertFalse(blocked["gate"]["p2d_warranted"])
        evaluator = identity["evaluator_revision"]
        self.assertEqual(evaluator["path"], "cvsearch/eval/phase3_zoom_oracle.py")
        self.assertEqual(
            evaluator["sha256"],
            hashlib.sha256(Path(phase3.__file__).read_bytes()).hexdigest(),
        )

    def test_dev_report_rejects_input_sidecar_or_evaluator_change_during_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for benchmark in ("vstar", "hr-bench_4k", "hr-bench_8k"):
                pair = []
                for variant in ("disabled", "enabled"):
                    path = root / f"{benchmark}-{variant}.jsonl"
                    path.write_text("row\n")
                    Path(f"{path}.launch-manifest.json").write_text("manifest\n")
                    pair.append(path)
                paths[benchmark] = tuple(pair)

            calls = 0

            def mutate_input(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    paths["vstar"][0].write_text("changed\n")
                return {"revision": "a" * 64, "oracle": {"delta": 0.1}}

            patches = (
                patch.object(phase3, "load_jsonl", return_value=[{}]),
                patch.object(phase3, "load_launch_manifest", return_value={}),
                patch.object(phase3, "validate_frozen_dev_identity"),
                patch.object(phase3, "validate_launch_pair", return_value={}),
            )
            with patches[0], patches[1], patches[2], patches[3], patch.object(
                phase3, "score_paired_rows", side_effect=mutate_input,
            ):
                with self.assertRaises(RuntimeError):
                    score_dev_paths(paths)

            calls = 0

            def mutate_sidecar(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    Path(
                        f"{paths['hr-bench_8k'][1]}.launch-manifest.json"
                    ).write_text("changed manifest\n")
                return {"revision": "a" * 64, "oracle": {"delta": 0.1}}

            patches = (
                patch.object(phase3, "load_jsonl", return_value=[{}]),
                patch.object(phase3, "load_launch_manifest", return_value={}),
                patch.object(phase3, "validate_frozen_dev_identity"),
                patch.object(phase3, "validate_launch_pair", return_value={}),
            )
            with patches[0], patches[1], patches[2], patches[3], patch.object(
                phase3, "score_paired_rows", side_effect=mutate_sidecar,
            ):
                with self.assertRaises(RuntimeError):
                    score_dev_paths(paths)

            with patch.object(phase3, "_EVALUATOR_SOURCE_SHA256_AT_IMPORT", "0" * 64):
                with self.assertRaises(RuntimeError):
                    phase3._evaluator_revision()


if __name__ == "__main__":
    unittest.main()
