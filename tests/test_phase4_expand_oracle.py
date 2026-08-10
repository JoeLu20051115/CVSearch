import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import cvsearch.eval.phase4_expand_oracle as phase4
from cvsearch.eval.phase4_expand_oracle import (
    DISABLED_CONFIG,
    ENABLED_CONFIG,
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
PROMPT_SHA = "4" * 64
HR_OPTIONS = [
    "A. cat\nB. dog\nC. bird\nD. fish",
    "A. dog\nB. cat\nC. fish\nD. bird",
    "A. bird\nB. fish\nC. cat\nD. dog",
    "A. fish\nB. bird\nC. dog\nD. cat",
]
P0_HR = ["A", "B", "C", "D"]
CANDIDATE_HR = ["B", "A", "D", "C"]


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


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
        "model_device": "cuda:0",
        "model_dtype": "torch.bfloat16",
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
        "wrapper_device": "cuda:0",
        "wrapper_dtype": "torch.bfloat16",
        "wrapper_flash_attention": True,
    }


def _configs():
    return load_method_config(str(DISABLED_CONFIG)), load_method_config(str(ENABLED_CONFIG))


def _source_identity(image):
    return {
        "mode": "RGB",
        "size": [image.width, image.height],
        "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }


def _descriptor(source, bbox, *, depth, posterior, ordinal):
    source_key = _canonical(source)
    key = _canonical({"bbox": list(bbox), "depth": depth, "render_level": 0})
    renderer = _canonical({
        "source_image_key": source_key,
        "renderer_kind": "fine",
        "bbox": list(bbox),
        "render_level": 0,
    })
    return {
        "canonical_key": key,
        "bbox_original": list(bbox),
        "depth": depth,
        "render_level": 0,
        "posterior_score": posterior,
        "first_seen_ordinal": ordinal,
        "tree_scope": "main",
        "crop_origin": [0, 0],
        "source_image_key": source_key,
        "source": "fine",
        "renderer_kind": "fine",
        "renderer_identity": renderer,
    }


def _requirements():
    return sanitize_evidence_requirements(({
        "kind": "target_detail",
        "target": "object",
        "requirements": ["presence", "visual_detail"],
    },))


def _support_payload(plan, plan_hash, observation_name, p_yes):
    requirements = _requirements()
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
        yes_tokenization=(9454,),
        no_tokenization=(2753,),
        yes_token_id=9454,
        no_token_id=2753,
        p_yes_transform=EVIDENCE_SUPPORT_TRANSFORM,
        yes_logit=1.0,
        no_logit=0.0,
        p_yes=p_yes,
        p_no=1.0 - p_yes,
        support_avg=p_yes,
        support_min=p_yes,
        observation_mode=observation["rendered_mode"],
        observation_size=tuple(observation["rendered_size"]),
        view_sha256=observation["view_sha256"],
        elapsed_seconds=0.01,
        accounted_pixels=plan["pixels_per_logical_forward"],
        batch_plan_hash=plan_hash,
    )
    return result.to_dict()


def _answer_record(answer_type, output, options):
    if answer_type == "option_list":
        record = aggregate_hr_answers(options, output)
        record.output = copy.deepcopy(output)
        record.selected_from = "cvsearch_anchor"
    else:
        losses = [0.8, 0.1] if output == 1 else [0.1, 0.8]
        record = aggregate_vstar_losses([losses])
        record.selected_from = "search"
    return record.to_dict()


def _budget(calls, pixels):
    return {
        "max_mllm_calls": 512,
        "max_processed_pixels": 10_000_000_000,
        "mllm_calls": calls,
        "processed_pixels": pixels,
    }


def _forced_step(index, answer, budget):
    return {
        "step": index,
        "action": "FORCED_RETURN",
        "gap_fallback_used": False,
        "elapsed_seconds": 0.0,
        "focus_key": None,
        "feasible_actions": [],
        "gaps": {},
        "no_op_reason": "minimal_v1 has no certified-stop controller",
        "answer": copy.deepcopy(answer),
        "support_avg": 0.0,
        "support_min": 0.0,
        "certified": False,
        "budget": copy.deepcopy(budget),
    }


def _p0_trace_fields():
    return {
        "query_plan": {
            "main_query": "q0",
            "targets": ["object"],
            "augmented_queries": ["locate and inspect object"],
            "evidence_items": [{
                "kind": "target_detail",
                "target": "object",
                "requirements": ["presence", "visual_detail"],
            }],
            "global_scope_required": False,
            "fallback_used": True,
        },
        "candidate_ranks": [],
        "history": [],
        "final_boxes": [],
        "method_mode": "root_search_fallback",
        "cvsearch_search_mode": 1,
        "root_ans_conf": 0.1,
        "num_pop": [],
        "num_zoom_in": [],
        "num_zoom_out": [],
        "effective_ranking_query": "q0",
    }


class PairFactory:
    def __init__(self, directory):
        self.root = Path(directory)
        self.image_path = self.root / "source.png"
        image = Image.new("RGB", (400, 300), (31, 73, 127))
        image.save(self.image_path)

    def manifest(self, *, config=None, benchmark="vstar", ordinal=0):
        payload = {
            "schema_version": 1,
            "benchmark": benchmark,
            "code": {"revision": "a" * 64},
            "selected_partition": {"ordinals": [ordinal]},
            "hardware": {"gpu_uuids": ["GPU-test"]},
            "artifacts": {
                "qwen": {"kind": "directory", "path": "frozen/qwen", "sha256": "d" * 64},
                "processor": {
                    "kind": "processor_files", "path": "frozen/qwen", "sha256": "e" * 64,
                },
                "source_images": {"files": [{
                    "path": str(self.image_path),
                    "sha256": hashlib.sha256(self.image_path.read_bytes()).hexdigest(),
                    "size": self.image_path.stat().st_size,
                }]},
            },
            "environment": {
                "python": "3.11",
                "packages": {
                    "pillow": "11.0.0",
                    "torch": "2.7.1+cu118",
                    "transformers": "4.57.0",
                },
            },
        }
        if config is not None:
            payload["config"] = {"loaded": copy.deepcopy(config)}
        return payload

    def pair(self, benchmark="vstar", ordinal=0):
        with Image.open(self.image_path) as opened:
            image = opened.convert("RGB")
        source = _source_identity(image)
        focus = _descriptor(source, (80, 100, 40, 40), depth=1, posterior=0.8, ordinal=0)
        context = _descriptor(source, (125, 100, 30, 30), depth=2, posterior=0.7, ordinal=1)
        focus_rectangle = phase4._rectangle(focus)
        context_rectangle = phase4._rectangle(context)
        edge_gap = 5.0 / 500.0
        selection = {
            "candidate": copy.deepcopy(context),
            "no_op_reason": None,
            "current_keys": [focus["canonical_key"]],
            "focus_descriptors": [copy.deepcopy(focus)],
            "focus_union_xyxy": focus_rectangle,
            "positive_outside_area": 900.0,
            "normalized_edge_gap": edge_gap,
            "rank_tuple": [
                edge_gap, False, -0.7, 1, context["canonical_key"],
            ],
        }
        answer_type = "logits_match" if benchmark == "vstar" else "option_list"
        options = ["cat", "dog"] if answer_type == "logits_match" else copy.deepcopy(HR_OPTIONS)
        p0 = 1 if answer_type == "logits_match" else copy.deepcopy(P0_HR)
        candidate = (
            {"winner": 0, "losses": [0.1, 0.9]}
            if answer_type == "logits_match" else copy.deepcopy(CANDIDATE_HR)
        )
        truth = 0 if answer_type == "logits_match" else copy.deepcopy(CANDIDATE_HR)
        composition, current_pixels, candidate_pixels = phase4._composition_material(
            image, [focus], context,
        )
        current_observation = {
            "canonical_keys": [focus["canonical_key"]],
            "renderer_identities": [focus["renderer_identity"]],
            **current_pixels,
            "descriptors": [copy.deepcopy(focus)],
        }
        candidate_observation = {
            "canonical_keys": [focus["canonical_key"], context["canonical_key"]],
            "renderer_identities": [focus["renderer_identity"], context["renderer_identity"]],
            **candidate_pixels,
            "descriptors": [copy.deepcopy(focus), copy.deepcopy(context)],
        }
        requirements = _requirements()
        config_disabled, config_enabled = _configs()
        options_hash, answer_hashes, answer_identity = phase4._expected_answer_identity(
            answer_type, "q0", options,
        )
        answer_calls = 4 if answer_type == "option_list" else 1 + len(options)
        total_calls = 2 + answer_calls
        area = image.width * image.height
        focus_crop = phase4._native_crop(focus, image.size)
        context_crop = phase4._native_crop(context, image.size)
        plan = {
            "schema_version": 1,
            "batch_kind": "p4a_post_anchor_expand_context",
            "answer_type": answer_type,
            "source_identity": source,
            "source_width": image.width,
            "source_height": image.height,
            "accounted_source_area": area,
            "selection_policy": phase4._SELECTION_POLICY,
            "composition_policy": phase4._COMPOSITION_POLICY,
            "base_view_size": 336,
            "patch_scale": None,
            "current_keys": [focus["canonical_key"]],
            "candidate_keys": [focus["canonical_key"], context["canonical_key"]],
            "focus_role": {
                "role": "focus",
                "canonical_keys": [focus["canonical_key"]],
                "renderer_identities": [focus["renderer_identity"]],
                "descriptors": [copy.deepcopy(focus)],
                "native_crops_xyxy": [focus_crop],
            },
            "context_role": {
                "role": "context",
                "canonical_key": context["canonical_key"],
                "renderer_identity": context["renderer_identity"],
                "descriptor": copy.deepcopy(context),
                "native_crop_xyxy": context_crop,
                "candidate_bbox_xyxy": context_rectangle,
                "focus_union_xyxy": focus_rectangle,
                "positive_outside_area": 900.0,
                "contained_by_focus_descriptor": False,
                "contains_complete_focus_union": False,
                "normalized_edge_gap": edge_gap,
                "rank_tuple": copy.deepcopy(selection["rank_tuple"]),
            },
            "focus_merge_identity": phase4._merge_identity([focus_crop]),
            "context_merge_identity": phase4._merge_identity([context_crop]),
            "composition_identity": composition,
            "current_observation": current_observation,
            "candidate_observation": candidate_observation,
            "candidate_answer_input_sha256": candidate_observation["view_sha256"],
            "q0": "q0",
            "options": copy.deepcopy(options),
            "options_sha256": options_hash,
            "answer_prompt_sha256": answer_hashes,
            "answer_call_identity_sha256": answer_identity,
            "q0_sha256": hashlib.sha256(b"q0").hexdigest(),
            "requirement_order": [item.requirement_id for item in requirements],
            "requirement_set_id": EvidenceSupportResult.requirement_set_id_for(requirements),
            "prompt_version": config_enabled["evidence_support_prompt_version"],
            "prompt_template_sha256": config_enabled["evidence_support_prompt_template_sha256"],
            "current_prompt_sha256": PROMPT_SHA,
            "candidate_prompt_sha256": PROMPT_SHA,
            "processor_mode": config_enabled["evidence_support_processor_mode"],
            "processor_fingerprint": _processor_fingerprint(),
            "checkpoint": "frozen/qwen",
            "yes_tokenization": [9454],
            "no_tokenization": [2753],
            "yes_token_id": 9454,
            "no_token_id": 2753,
            "p_yes_transform": EVIDENCE_SUPPORT_TRANSFORM,
            "verifier_status": "disabled_same_checkpoint_unpromoted",
            "current_support_calls": 1,
            "candidate_support_calls": 1,
            "candidate_answer_calls": answer_calls,
            "candidate_option_count": len(options),
            "pixels_per_logical_forward": area,
            "total_calls": total_calls,
            "total_pixels": total_calls * area,
        }
        payload, plan_hash, full = _validate_batch_plan(plan)
        assert full
        plan = dict(payload, plan_hash=plan_hash)
        current_support = _support_payload(plan, plan_hash, "current", 0.4)
        candidate_support = _support_payload(plan, plan_hash, "candidate", 0.8)
        p0_record = _answer_record(answer_type, p0, options)
        before = _budget(10, 10 * area)
        after = _budget(10 + total_calls, (10 + total_calls) * area)
        batch = {
            "status": "success",
            "admitted": True,
            "charged": True,
            "batch_plan": plan,
            "batch_plan_hash": plan_hash,
            "current_support": current_support,
            "candidate_support": candidate_support,
            "candidate_answer": copy.deepcopy(candidate),
            "failure_phase": None,
            "failure_reason": None,
            "exception_type": None,
            "executed_stages": (
                ["current_support", "candidate_support", "vstar_answer"]
                if answer_type == "logits_match" else
                ["current_support", "candidate_support", "hr_answer_0", "hr_answer_1",
                 "hr_answer_2", "hr_answer_3"]
            ),
            "elapsed_seconds": 0.25,
            "ledger_before": before,
            "ledger_after": after,
            "verifier_status": "disabled_same_checkpoint_unpromoted",
            "verifier_avg": None,
            "verifier_min": None,
            "promotable": False,
        }
        if answer_type == "option_list":
            p0_stability = aggregate_hr_answers(options, p0).to_dict()
            p0_stability["output"] = copy.deepcopy(p0)
            p0_stability["selected_from"] = "cvsearch_raw"
            candidate_stability = aggregate_hr_answers(options, candidate).to_dict()
            producing_phase = "cvsearch_raw"
        else:
            p0_stability = copy.deepcopy(p0_record)
            candidate_stability = aggregate_vstar_losses([candidate["losses"]]).to_dict()
            producing_phase = "search"
        audit = {
            "p0_anchor": {
                "emitted_answer": copy.deepcopy(p0),
                "cvsearch_raw": copy.deepcopy(p0),
                "producing_phase": producing_phase,
                "node_keys": [focus["canonical_key"]],
                "support_view": [copy.deepcopy(focus)],
            },
            "current_keys": [focus["canonical_key"]],
            "candidate_keys": [focus["canonical_key"], context["canonical_key"]],
            "selection_decision": copy.deepcopy(selection),
            "selection_decision_sha256": hashlib.sha256(
                _canonical(selection).encode()
            ).hexdigest(),
            "selection_policy": phase4._SELECTION_POLICY,
            "composition_policy": phase4._COMPOSITION_POLICY,
            "focus_role": copy.deepcopy(plan["focus_role"]),
            "context_role": copy.deepcopy(plan["context_role"]),
            "focus_merge_identity": copy.deepcopy(plan["focus_merge_identity"]),
            "context_merge_identity": copy.deepcopy(plan["context_merge_identity"]),
            "composition_identity": copy.deepcopy(plan["composition_identity"]),
            "batch_result": batch,
            "current_gap_support": current_support,
            "candidate_gap_support": candidate_support,
            "uncertainty": p0_stability["uncertainty"],
            "uncalibrated_g_expand_proxy": 0.6,
            "support_delta": 0.4,
            "support_proxy_status": "audit_only_uncalibrated",
            "p0_stability": p0_stability,
            "candidate_stability": candidate_stability,
            "feasible": True,
            "normalized_actual_cost": total_calls / 512,
            "support_contract_status": "matched",
            "coverage_status": "not_observed",
            "verifier_status": "disabled_same_checkpoint_unpromoted",
            "verifier_avg": None,
            "verifier_min": None,
            "score_margin": None,
            "score_status": "unavailable_missing_verifier_coverage",
            "replacement_reason": "replacement_disabled_p4a",
        }
        base = {
            "question": "q0",
            "options": copy.deepcopy(options),
            "answer_type": answer_type,
            "input_image": "source.png",
            "answer": truth,
            "output": copy.deepcopy(p0),
            "_eg_ordinal": ordinal,
            "_eg_code_revision": "a" * 64,
            "_eg_run_fingerprint": "b" * 64,
        }
        disabled = copy.deepcopy(base)
        disabled["method_trace"] = {
            **_p0_trace_fields(),
            "steps": [_forced_step(0, p0_record, before)],
            "final_answer": copy.deepcopy(p0_record),
            "anchor_answer": copy.deepcopy(p0_record),
            "budget": before,
            "elapsed_seconds": 1.0,
            "termination": "FORCED_RETURN",
            "config_id": config_disabled["config_id"],
            "effective_config": config_disabled,
            "budget_interrupted": False,
            "pixel_accounting": config_disabled["pixel_accounting"],
            "anchor_state_score": 0.1,
            "selected_state_score": 0.1,
            "replacement_margin": 0.0,
            "support_status": "not_observed",
        }
        expand_step = {
            "step": 0,
            "action": "EXPAND",
            "gap_fallback_used": False,
            "elapsed_seconds": 0.0,
            "focus_key": context["canonical_key"],
            "feasible_actions": ["EXPAND"],
            "gaps": {"g_expand_proxy_audit_only": 0.6},
            "no_op_reason": None,
            "answer": copy.deepcopy(p0_record),
            "support_avg": 0.0,
            "support_min": 0.0,
            "certified": False,
            "budget": after,
            "expand_audit": audit,
        }
        enabled = copy.deepcopy(base)
        enabled["_eg_run_fingerprint"] = "c" * 64
        enabled["method_trace"] = {
            **_p0_trace_fields(),
            "steps": [expand_step, _forced_step(1, p0_record, after)],
            "final_answer": copy.deepcopy(p0_record),
            "anchor_answer": copy.deepcopy(p0_record),
            "budget": after,
            "elapsed_seconds": 1.5,
            "termination": "FORCED_RETURN",
            "config_id": config_enabled["config_id"],
            "effective_config": config_enabled,
            "budget_interrupted": False,
            "pixel_accounting": config_enabled["pixel_accounting"],
            "anchor_state_score": None,
            "selected_state_score": None,
            "replacement_margin": None,
            "support_status": "observed_answer_free_audit_only",
        }
        manifest = self.manifest(
            config=config_enabled, benchmark=benchmark, ordinal=ordinal,
        )
        enabled["_eg_run_fingerprint"] = canonical_sha256(manifest)
        return disabled, enabled, manifest


def _batch(row):
    return row["method_trace"]["steps"][0]["expand_audit"]["batch_result"]


def _audit(row):
    return row["method_trace"]["steps"][0]["expand_audit"]


def _refresh_plan_hash(row):
    batch = _batch(row)
    plan = batch["batch_plan"]
    plan.pop("plan_hash", None)
    plan_hash = hashlib.sha256(_canonical(plan).encode()).hexdigest()
    plan["plan_hash"] = plan_hash
    batch["batch_plan_hash"] = plan_hash
    for name in ("current_support", "candidate_support"):
        if batch.get(name) is not None:
            batch[name]["batch_plan_hash"] = plan_hash


def _refresh_composition_hash(plan):
    composition = plan["composition_identity"]
    composition.pop("identity_sha256", None)
    composition["identity_sha256"] = hashlib.sha256(
        _canonical(composition).encode()
    ).hexdigest()


def _as_model_failed(row):
    row = copy.deepcopy(row)
    batch = _batch(row)
    batch.update(
        status="model_failed",
        candidate_answer=None,
        candidate_support=None,
        failure_phase="candidate_support",
        failure_reason="failed candidate support",
        exception_type="RuntimeError",
        executed_stages=["current_support"],
    )
    audit = _audit(row)
    audit.update(
        candidate_gap_support=None,
        uncalibrated_g_expand_proxy=None,
        support_delta=None,
        candidate_stability=None,
        feasible=False,
        support_contract_status="not_observed",
        replacement_reason=None,
    )
    step = row["method_trace"]["steps"][0]
    step.update(
        feasible_actions=[], gaps={}, no_op_reason="expand_model_failed",
    )
    row["method_trace"]["support_status"] = "expand_model_failed"
    return row


def _as_budget_rejected(row):
    row = copy.deepcopy(row)
    batch = _batch(row)
    before = copy.deepcopy(batch["ledger_before"])
    batch.update(
        status="budget_rejected",
        admitted=False,
        charged=False,
        current_support=None,
        candidate_support=None,
        candidate_answer=None,
        failure_phase="admission",
        failure_reason="mllm_calls budget exhausted before model execution",
        exception_type="BudgetExceeded",
        executed_stages=[],
        ledger_after=before,
    )
    audit = _audit(row)
    audit.update(
        current_gap_support=None,
        candidate_gap_support=None,
        uncalibrated_g_expand_proxy=None,
        support_delta=None,
        candidate_stability=None,
        feasible=False,
        normalized_actual_cost=0.0,
        support_contract_status="not_observed",
        replacement_reason=None,
    )
    step = row["method_trace"]["steps"][0]
    step.update(
        feasible_actions=[], gaps={}, no_op_reason="expand_budget_rejected",
        budget=copy.deepcopy(before),
    )
    row["method_trace"].update(
        budget=copy.deepcopy(before), support_status="expand_budget_rejected",
    )
    row["method_trace"]["steps"][1]["budget"] = copy.deepcopy(before)
    return row


class Phase4ExpandOracleTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.factory = PairFactory(self.directory.name)

    def score(self, benchmark, disabled, enabled, manifest, *, p0_correct=0):
        cycles = 1 if benchmark == "vstar" else 4
        expectation = PairExpectation(
            1, cycles, p0_correct, canonical_output_digest([disabled]),
        )
        with patch.object(phase4, "_support_prompt_sha256", return_value=PROMPT_SHA):
            return score_paired_rows(
                benchmark, [disabled], [enabled], expectation,
                launch_manifest=manifest, bootstrap_replicates=10_000,
            )

    def assert_rejected(self, benchmark, disabled, enabled, manifest, *, p0_correct=0):
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            self.score(
                benchmark, disabled, enabled, manifest, p0_correct=p0_correct,
            )

    def test_public_api_configs_and_coherent_hr_raw_oracle_tie_retain_p0(self):
        disabled_config, enabled_config = _configs()
        self.assertTrue(callable(score_paired_rows))
        self.assertTrue(callable(score_dev_paths))
        self.assertTrue(callable(validate_launch_pair))
        changed = {
            key for key in enabled_config
            if enabled_config.get(key) != disabled_config.get(key)
        }
        self.assertEqual(changed, {
            "config_id", "p4a_expand_enabled", "p4a_expand_admission_mode",
        })
        self.assertEqual(DISABLED_CONFIG.parent, ROOT / "reproduction/evidence_gap/configs")

        disabled, enabled, manifest = self.factory.pair("hr-bench_4k")
        raw_seen = []
        real_score_labels = phase4._score_labels

        def capture_raw(benchmark, pairs):
            raw_seen.append(copy.deepcopy(pairs[0]["candidate"]))
            return real_score_labels(benchmark, pairs)

        with patch.object(phase4, "_score_labels", side_effect=capture_raw):
            report = self.score("hr-bench_4k", disabled, enabled, manifest)
        self.assertEqual(raw_seen, [CANDIDATE_HR])
        self.assertEqual(report["p0"]["correct"], 0)
        self.assertEqual(report["candidate_alone"]["correct"], 4)
        self.assertEqual(report["oracle"]["correct"], 4)
        self.assertFalse(report["audit"]["candidate_stability_output_scored"])
        self.assertTrue(report["audit"]["raw_candidate_answer_scored"])

        tie_disabled = copy.deepcopy(disabled)
        tie_enabled = copy.deepcopy(enabled)
        tie_truth = ["A", "A", "D", "D"]
        tie_disabled["answer"] = copy.deepcopy(tie_truth)
        tie_enabled["answer"] = copy.deepcopy(tie_truth)
        tie = self.score(
            "hr-bench_4k", tie_disabled, tie_enabled, manifest, p0_correct=2,
        )
        self.assertEqual(tie["candidate_alone"]["correct"], 2)
        self.assertEqual(tie["oracle"]["correct"], 2)
        self.assertEqual(tie["oracle"]["selected_candidate_topics"], 0)
        self.assertEqual(tie["oracle"]["tie_rule"], "retain_p0")

    def test_coherent_vstar_scores_finite_raw_loss_argmin_and_frozen_bootstrap(self):
        disabled, enabled, manifest = self.factory.pair("vstar")
        report = self.score("vstar", disabled, enabled, manifest)
        self.assertEqual(report["candidate_alone"]["correct"], 1)
        self.assertEqual(report["oracle"]["corrected"], 1)
        self.assertEqual(report["bootstrap"]["replicates"], 10_000)
        self.assertEqual(report["candidate"]["status_counts"], {"success": 1})

        nonfinite = copy.deepcopy(enabled)
        _batch(nonfinite)["candidate_answer"]["losses"][0] = float("nan")
        self.assert_rejected("vstar", disabled, nonfinite, manifest)
        wrong_winner = copy.deepcopy(enabled)
        _batch(wrong_winner)["candidate_answer"]["winner"] = 1
        self.assert_rejected("vstar", disabled, wrong_winner, manifest)
        expectation = PairExpectation(1, 1, 0, canonical_output_digest([disabled]))
        with (
            patch.object(phase4, "_support_prompt_sha256", return_value=PROMPT_SHA),
            self.assertRaises(ValueError),
        ):
            score_paired_rows(
                "vstar", [disabled], [enabled], expectation,
                launch_manifest=manifest, bootstrap_replicates=9,
            )

    def test_selection_hash_geometry_crop_merge_and_source_tampering_fail_closed(self):
        disabled, enabled, manifest = self.factory.pair("vstar")

        mutations = []
        forged = copy.deepcopy(enabled)
        _audit(forged)["selection_decision_sha256"] = "f" * 64
        mutations.append(("selection_hash", forged))

        forged = copy.deepcopy(enabled)
        selection = _audit(forged)["selection_decision"]
        selection["normalized_edge_gap"] = 0.5
        selection["rank_tuple"][0] = 0.5
        _audit(forged)["selection_decision_sha256"] = hashlib.sha256(
            _canonical(selection).encode()
        ).hexdigest()
        plan = _batch(forged)["batch_plan"]
        plan["context_role"]["normalized_edge_gap"] = 0.5
        plan["context_role"]["rank_tuple"][0] = 0.5
        _refresh_plan_hash(forged)
        mutations.append(("synchronized_selection_geometry", forged))

        forged = copy.deepcopy(enabled)
        plan = _batch(forged)["batch_plan"]
        plan["focus_role"]["native_crops_xyxy"][0][0] += 1
        plan["focus_merge_identity"] = phase4._merge_identity(
            plan["focus_role"]["native_crops_xyxy"]
        )
        _audit(forged)["focus_role"] = copy.deepcopy(plan["focus_role"])
        _audit(forged)["focus_merge_identity"] = copy.deepcopy(plan["focus_merge_identity"])
        _refresh_plan_hash(forged)
        mutations.append(("synchronized_crop_merge", forged))

        for name, row in mutations:
            with self.subTest(name=name):
                self.assert_rejected("vstar", disabled, row, manifest)

        original = self.factory.image_path.read_bytes()
        try:
            Image.new("RGB", (400, 300), (9, 9, 9)).save(self.factory.image_path)
            forged_manifest = copy.deepcopy(manifest)
            entry = forged_manifest["artifacts"]["source_images"]["files"][0]
            entry["sha256"] = hashlib.sha256(self.factory.image_path.read_bytes()).hexdigest()
            entry["size"] = self.factory.image_path.stat().st_size
            self.assert_rejected("vstar", disabled, enabled, forged_manifest)
        finally:
            self.factory.image_path.write_bytes(original)

    def test_panel_and_composite_dependent_hash_forgery_is_rejected_by_rgb_replay(self):
        disabled, enabled, manifest = self.factory.pair("vstar")

        panel = copy.deepcopy(enabled)
        plan = _batch(panel)["batch_plan"]
        composition = plan["composition_identity"]
        composition["focus_panel_sha256"] = "f" * 64
        composition["focus_current_rectangle_sha256"] = "f" * 64
        composition["focus_candidate_rectangle_sha256"] = "f" * 64
        _refresh_composition_hash(plan)
        _audit(panel)["composition_identity"] = copy.deepcopy(composition)
        _refresh_plan_hash(panel)
        self.assert_rejected("vstar", disabled, panel, manifest)

        composite = copy.deepcopy(enabled)
        batch = _batch(composite)
        plan = batch["batch_plan"]
        forged_hash = "e" * 64
        plan["composition_identity"]["candidate_composite_sha256"] = forged_hash
        plan["candidate_observation"]["view_sha256"] = forged_hash
        plan["candidate_answer_input_sha256"] = forged_hash
        support = batch["candidate_support"]
        support["view_sha256"] = forged_hash
        support["observation_identity"]["view_sha256"] = forged_hash
        _refresh_composition_hash(plan)
        audit = _audit(composite)
        audit["composition_identity"] = copy.deepcopy(plan["composition_identity"])
        audit["candidate_gap_support"] = copy.deepcopy(support)
        _refresh_plan_hash(composite)
        self.assert_rejected("vstar", disabled, composite, manifest)

    def test_q0_options_prompt_and_call_identity_synchronized_forgery_is_rejected(self):
        disabled, enabled, manifest = self.factory.pair("vstar")

        q0 = copy.deepcopy(enabled)
        plan = _batch(q0)["batch_plan"]
        plan["q0"] = "forged q0"
        plan["q0_sha256"] = hashlib.sha256(b"forged q0").hexdigest()
        _, prompts, identity = phase4._expected_answer_identity(
            plan["answer_type"], plan["q0"], plan["options"],
        )
        plan["answer_prompt_sha256"] = prompts
        plan["answer_call_identity_sha256"] = identity
        _refresh_plan_hash(q0)
        self.assert_rejected("vstar", disabled, q0, manifest)

        options = copy.deepcopy(enabled)
        plan = _batch(options)["batch_plan"]
        plan["options"] = ["bird", "fish"]
        option_hash, prompts, identity = phase4._expected_answer_identity(
            plan["answer_type"], plan["q0"], plan["options"],
        )
        plan["options_sha256"] = option_hash
        plan["answer_prompt_sha256"] = prompts
        plan["answer_call_identity_sha256"] = identity
        _refresh_plan_hash(options)
        self.assert_rejected("vstar", disabled, options, manifest)

        prompt = copy.deepcopy(enabled)
        plan = _batch(prompt)["batch_plan"]
        plan["current_prompt_sha256"] = "f" * 64
        plan["candidate_prompt_sha256"] = "f" * 64
        _batch(prompt)["current_support"]["prompt_sha256"] = "f" * 64
        _batch(prompt)["candidate_support"]["prompt_sha256"] = "f" * 64
        _audit(prompt)["current_gap_support"] = copy.deepcopy(_batch(prompt)["current_support"])
        _audit(prompt)["candidate_gap_support"] = copy.deepcopy(_batch(prompt)["candidate_support"])
        _refresh_plan_hash(prompt)
        self.assert_rejected("vstar", disabled, prompt, manifest)

    def test_status_stage_ledger_budget_p0_output_and_trace_tampering(self):
        disabled, enabled, manifest = self.factory.pair("vstar")
        model_failed = self.score(
            "vstar", disabled, _as_model_failed(enabled), manifest,
        )
        self.assertEqual(model_failed["candidate"]["status_counts"], {"model_failed": 1})
        self.assertEqual(
            model_failed["candidate"]["no_op_reason_counts"],
            {"expand_model_failed": 1},
        )
        budget_rejected = self.score(
            "vstar", disabled, _as_budget_rejected(enabled), manifest,
        )
        self.assertEqual(
            budget_rejected["candidate"]["status_counts"], {"budget_rejected": 1},
        )
        self.assertEqual(
            budget_rejected["candidate"]["no_op_reason_counts"],
            {"expand_budget_rejected": 1},
        )

        def mutate_stage(row):
            _batch(row)["executed_stages"] = ["current_support"]

        def mutate_ledger(row):
            _batch(row)["ledger_after"]["mllm_calls"] += 1
            row["method_trace"]["steps"][0]["budget"]["mllm_calls"] += 1
            row["method_trace"]["steps"][1]["budget"]["mllm_calls"] += 1
            row["method_trace"]["budget"]["mllm_calls"] += 1

        attacks = (
            ("status", lambda row: _batch(row).__setitem__("status", "render_noop")),
            ("stage", mutate_stage),
            ("ledger", mutate_ledger),
            ("p0", lambda row: _audit(row)["p0_anchor"].__setitem__("emitted_answer", 0)),
            ("output", lambda row: row.__setitem__("output", 0)),
            ("action", lambda row: row["method_trace"]["steps"][0].__setitem__("action", "NEXT")),
            ("trace_extra", lambda row: row["method_trace"].__setitem__("forged", True)),
        )
        for name, mutate in attacks:
            with self.subTest(name=name):
                forged = copy.deepcopy(enabled)
                mutate(forged)
                self.assert_rejected("vstar", disabled, forged, manifest)

    def test_source_mutation_during_label_boundary_is_detected(self):
        disabled, enabled, manifest = self.factory.pair("vstar")
        original = self.factory.image_path.read_bytes()
        real_score = phase4._score_labels

        def mutate_source(benchmark, pairs):
            result = real_score(benchmark, pairs)
            Image.new("RGB", (400, 300), (1, 2, 3)).save(self.factory.image_path)
            return result

        try:
            with patch.object(phase4, "_score_labels", side_effect=mutate_source):
                self.assert_rejected("vstar", disabled, enabled, manifest)
        finally:
            self.factory.image_path.write_bytes(original)

    def test_launch_revision_fingerprint_gpu_config_and_partition_tampering(self):
        disabled, enabled, _ = self.factory.pair("vstar")
        disabled_config, enabled_config = _configs()
        disabled_manifest = self.factory.manifest(config=disabled_config)
        enabled_manifest = self.factory.manifest(config=enabled_config)
        disabled["_eg_run_fingerprint"] = canonical_sha256(disabled_manifest)
        enabled["_eg_run_fingerprint"] = canonical_sha256(enabled_manifest)
        validate_launch_pair(
            [disabled], [enabled], disabled_manifest, enabled_manifest,
        )

        cases = []
        forged = copy.deepcopy(enabled_manifest)
        forged["hardware"]["gpu_uuids"] = ["GPU-other"]
        cases.append(("gpu", enabled, forged))
        forged = copy.deepcopy(enabled_manifest)
        forged["code"]["revision"] = "f" * 64
        cases.append(("revision", enabled, forged))
        forged = copy.deepcopy(enabled_manifest)
        forged["selected_partition"]["ordinals"] = [9]
        cases.append(("partition", enabled, forged))
        forged = copy.deepcopy(enabled_manifest)
        forged["config"]["loaded"]["quick_gate"] = 0.8
        cases.append(("config", enabled, forged))
        fingerprint_row = copy.deepcopy(enabled)
        fingerprint_row["_eg_run_fingerprint"] = "0" * 64
        cases.append(("fingerprint", fingerprint_row, enabled_manifest))
        for name, row, manifest in cases:
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                validate_launch_pair(
                    [disabled], [row], disabled_manifest, manifest,
                )

    def test_programmatic_score_binds_enabled_manifest_config_revision_fingerprint_partition(self):
        disabled, enabled, manifest = self.factory.pair("vstar")
        cases = []

        forged_manifest = copy.deepcopy(manifest)
        forged_manifest["config"]["loaded"] = _configs()[0]
        forged_row = copy.deepcopy(enabled)
        forged_row["_eg_run_fingerprint"] = canonical_sha256(forged_manifest)
        cases.append(("config", forged_row, forged_manifest))

        forged_manifest = copy.deepcopy(manifest)
        forged_manifest["code"]["revision"] = "f" * 64
        forged_row = copy.deepcopy(enabled)
        forged_row["_eg_run_fingerprint"] = canonical_sha256(forged_manifest)
        cases.append(("revision", forged_row, forged_manifest))

        forged_manifest = copy.deepcopy(manifest)
        forged_manifest["selected_partition"]["ordinals"] = [9]
        forged_row = copy.deepcopy(enabled)
        forged_row["_eg_run_fingerprint"] = canonical_sha256(forged_manifest)
        cases.append(("partition", forged_row, forged_manifest))

        forged_row = copy.deepcopy(enabled)
        forged_row["_eg_run_fingerprint"] = "0" * 64
        cases.append(("fingerprint", forged_row, manifest))

        for name, row, forged_manifest in cases:
            with self.subTest(name=name):
                self.assert_rejected("vstar", disabled, row, forged_manifest)

    def test_bootstrap_phase_input_and_source_mutation_are_detected_post_scoring(self):
        disabled, enabled, manifest = self.factory.pair("vstar")
        real_bootstrap = phase4._bootstrap

        def mutate_row(benchmark, rows, replicates):
            result = real_bootstrap(benchmark, rows, replicates)
            enabled["question"] = "mutated during bootstrap"
            return result

        with patch.object(phase4, "_bootstrap", side_effect=mutate_row):
            self.assert_rejected("vstar", disabled, enabled, manifest)

        disabled, enabled, manifest = self.factory.pair("vstar")
        original = self.factory.image_path.read_bytes()

        def mutate_source(benchmark, rows, replicates):
            result = real_bootstrap(benchmark, rows, replicates)
            Image.new("RGB", (400, 300), (7, 8, 9)).save(self.factory.image_path)
            return result

        try:
            with patch.object(phase4, "_bootstrap", side_effect=mutate_source):
                self.assert_rejected("vstar", disabled, enabled, manifest)
        finally:
            self.factory.image_path.write_bytes(original)


class Phase4DevAndCliTest(unittest.TestCase):
    def _paths(self, root):
        paths = {}
        for benchmark in ("vstar", "hr-bench_4k", "hr-bench_8k"):
            pair = []
            for variant in ("disabled", "enabled"):
                path = root / f"{benchmark}-{variant}.jsonl"
                path.write_text(f"{benchmark}:{variant}\n", encoding="utf-8")
                Path(f"{path}.launch-manifest.json").write_text(
                    f"{benchmark}:{variant}:manifest\n", encoding="utf-8",
                )
                pair.append(path)
            paths[benchmark] = tuple(pair)
        return paths

    def test_dev_report_three_way_gate_identity_and_input_sidecar_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(Path(directory))
            deltas = {"vstar": 0.0, "hr-bench_4k": 0.25, "hr-bench_8k": 0.25}

            def fake_score(benchmark, *_args, **_kwargs):
                return {"revision": "a" * 64, "oracle": {"delta": deltas[benchmark]}}

            patches = (
                patch.object(phase4, "load_jsonl", return_value=[{}]),
                patch.object(phase4, "load_launch_manifest", return_value={}),
                patch.object(phase4, "validate_frozen_dev_identity"),
                patch.object(phase4, "validate_launch_pair", return_value={}),
                patch.object(phase4, "score_paired_rows", side_effect=fake_score),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                passing = score_dev_paths(paths)
                deltas["vstar"] = -0.01
                blocked = score_dev_paths(paths)
            self.assertTrue(passing["gate"]["p4a_warranted"])
            self.assertFalse(blocked["gate"]["p4a_warranted"])
            self.assertFalse(passing["gate"]["deployable_selector_claimed"])
            self.assertIn("launch_manifest_sha256", passing["execution_identity"])

            calls = 0

            def mutate_jsonl(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    paths["vstar"][0].write_text("mutated\n", encoding="utf-8")
                return {"revision": "a" * 64, "oracle": {"delta": 0.1}}

            patches = (
                patch.object(phase4, "load_jsonl", return_value=[{}]),
                patch.object(phase4, "load_launch_manifest", return_value={}),
                patch.object(phase4, "validate_frozen_dev_identity"),
                patch.object(phase4, "validate_launch_pair", return_value={}),
                patch.object(phase4, "score_paired_rows", side_effect=mutate_jsonl),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with self.assertRaises(RuntimeError):
                    score_dev_paths(paths)

        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(Path(directory))
            calls = 0

            def mutate_sidecar(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    Path(f"{paths['hr-bench_8k'][1]}.launch-manifest.json").write_text(
                        "mutated manifest\n", encoding="utf-8",
                    )
                return {"revision": "a" * 64, "oracle": {"delta": 0.1}}

            patches = (
                patch.object(phase4, "load_jsonl", return_value=[{}]),
                patch.object(phase4, "load_launch_manifest", return_value={}),
                patch.object(phase4, "validate_frozen_dev_identity"),
                patch.object(phase4, "validate_launch_pair", return_value={}),
                patch.object(phase4, "score_paired_rows", side_effect=mutate_sidecar),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with self.assertRaises(RuntimeError):
                    score_dev_paths(paths)

    def test_cli_atomic_write_and_failure_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._paths(root)
            output = root / "report.json"
            argv = [
                "--vstar-disabled", str(paths["vstar"][0]),
                "--vstar-enabled", str(paths["vstar"][1]),
                "--hr4-disabled", str(paths["hr-bench_4k"][0]),
                "--hr4-enabled", str(paths["hr-bench_4k"][1]),
                "--hr8-disabled", str(paths["hr-bench_8k"][0]),
                "--hr8-enabled", str(paths["hr-bench_8k"][1]),
                "--output", str(output),
            ]
            report = {"schema_version": 2, "ok": True}
            with patch.object(phase4, "score_dev_paths", return_value=report):
                self.assertEqual(phase4.main(argv), 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertEqual(list(root.glob(f".{output.name}.*.tmp")), [])

            output.unlink()
            with patch.object(phase4, "score_dev_paths", side_effect=ValueError("invalid")):
                with self.assertRaises(ValueError):
                    phase4.main(argv)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(f".{output.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
