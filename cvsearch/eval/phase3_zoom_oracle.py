"""Strict evaluator-only scorer for paired P2C coordinate-ZOOM artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cvsearch.eval.phase2_oracle import (
    BOOTSTRAP_REPLICATES,
    FROZEN_DEV_EXPECTATIONS,
    PairExpectation,
    _answer_output,
    _bootstrap,
    _budget,
    _canonical,
    _finite,
    _list,
    _mapping,
    _percentile,
    _runtime_report,
    _score_labels,
    _sha256,
    canonical_output_digest,
    load_jsonl,
    load_launch_manifest,
    validate_frozen_dev_identity,
)
from cvsearch.evidence_gap.answers import aggregate_hr_answers, aggregate_vstar_losses
from cvsearch.evidence_gap.method import load_method_config
from cvsearch.evidence_gap.provenance import canonical_sha256
from cvsearch.evidence_gap.types import (
    EvidenceRequirement,
    EvidenceSupportResult,
    ObservationBatchResult,
    sanitize_evidence_requirements,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "reproduction/evidence_gap/configs"
DISABLED_CONFIG = CONFIG_ROOT / "dev_unified_zoom_disabled_gamma000_budget512.json"
ENABLED_CONFIG = CONFIG_ROOT / "dev_unified_zoom_oracle_gamma000_budget512.json"
_HR_BENCHMARKS = frozenset({"hr-bench_4k", "hr-bench_8k"})
_EVALUATOR_SOURCE_PATH = Path(__file__).resolve()
_EVALUATOR_SOURCE_RELATIVE = _EVALUATOR_SOURCE_PATH.relative_to(ROOT).as_posix()
_EVALUATOR_SOURCE_SHA256_AT_IMPORT = hashlib.sha256(
    _EVALUATOR_SOURCE_PATH.read_bytes()
).hexdigest()

_SUPPORT_FIELDS = frozenset({
    "requirements", "requirement_order", "requirement_set_id",
    "observation_identity", "prompt_version", "prompt_template_sha256",
    "prompt_sha256", "processor_mode", "processor_fingerprint", "checkpoint",
    "yes_tokenization", "no_tokenization", "yes_token_id", "no_token_id",
    "p_yes_transform", "yes_logit", "no_logit", "p_yes", "p_no",
    "support_avg", "support_min", "aggregate_alias_note", "observation_mode",
    "observation_size", "view_sha256", "elapsed_seconds", "logical_calls",
    "accounted_pixels", "batch_plan_hash",
})
_BATCH_FIELDS = frozenset({
    "status", "admitted", "charged", "batch_plan", "batch_plan_hash",
    "current_support", "candidate_support", "candidate_answer", "failure_phase",
    "failure_reason", "exception_type", "executed_stages", "elapsed_seconds",
    "ledger_before", "ledger_after", "verifier_status", "verifier_avg",
    "verifier_min", "promotable",
})
_P0_FIELDS = frozenset({
    "emitted_answer", "cvsearch_raw", "producing_phase", "node_keys", "support_view",
})
_AUDIT_FIELDS = frozenset({
    "p0_anchor", "current_keys", "zoom_keys", "coordinate_mapping",
    "render_policy", "base_view_size", "candidate_view_size", "batch_result",
    "current_gap_support", "candidate_gap_support", "uncertainty",
    "uncalibrated_g_zoom_proxy", "support_delta", "support_proxy_status",
    "p0_stability", "candidate_stability", "feasible", "normalized_actual_cost",
    "support_contract_status", "coverage_status", "verifier_status",
    "verifier_avg", "verifier_min", "score_margin", "score_status",
    "replacement_reason",
})
_BASE_STEP_FIELDS = frozenset({
    "step", "action", "gap_fallback_used", "elapsed_seconds", "focus_key",
    "feasible_actions", "gaps", "no_op_reason", "answer", "support_avg",
    "support_min", "certified", "budget",
})
_TRACE_FIELDS = frozenset({
    "query_plan", "candidate_ranks", "steps", "history", "final_answer",
    "budget", "elapsed_seconds", "termination", "final_boxes", "method_mode",
    "config_id", "effective_config", "cvsearch_search_mode", "root_ans_conf",
    "num_pop", "num_zoom_in", "num_zoom_out", "budget_interrupted",
    "effective_ranking_query", "pixel_accounting", "anchor_answer",
    "anchor_state_score", "selected_state_score", "replacement_margin",
    "support_status",
})
_QUERY_PLAN_FIELDS = frozenset({
    "main_query", "targets", "augmented_queries", "evidence_items",
    "global_scope_required", "fallback_used",
})
_SHARED_P0_TRACE_FIELDS = (
    "query_plan", "candidate_ranks", "history", "final_boxes", "method_mode",
    "cvsearch_search_mode", "root_ans_conf", "num_pop", "num_zoom_in",
    "num_zoom_out", "effective_ranking_query",
)
_NO_BATCH_REASONS = frozenset({
    "zoom_invalid_evidence_requirements", "zoom_no_evidence_requirements",
    "zoom_p0_support_view_unavailable", "zoom_p0_support_view_empty",
    "zoom_p0_support_view_nonlocal", "zoom_batch_preflight_failed",
})


def _evaluator_revision() -> dict[str, str]:
    current = hashlib.sha256(_EVALUATOR_SOURCE_PATH.read_bytes()).hexdigest()
    if current != _EVALUATOR_SOURCE_SHA256_AT_IMPORT:
        raise RuntimeError("P2C evaluator source changed during evaluation")
    return {
        "kind": "source_content_sha256",
        "path": _EVALUATOR_SOURCE_RELATIVE,
        "sha256": current,
    }


def _validate_config_pair(disabled: Mapping[str, Any], enabled: Mapping[str, Any]) -> None:
    expected_disabled = load_method_config(str(DISABLED_CONFIG))
    expected_enabled = load_method_config(str(ENABLED_CONFIG))
    if dict(disabled) != expected_disabled or dict(enabled) != expected_enabled:
        raise ValueError("run config does not match the frozen Phase-3 P2C config")
    changed = {
        key for key in expected_enabled
        if expected_enabled.get(key) != expected_disabled.get(key)
    }
    if changed != {"config_id", "p2c_zoom_enabled", "p2c_zoom_admission_mode"}:
        raise ValueError("P2C configs violate the frozen sibling contract")


def _launch_model_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _mapping(manifest, "P2C launch manifest")
    artifacts = _mapping(manifest.get("artifacts"), "launch artifacts")
    qwen = _mapping(artifacts.get("qwen"), "Qwen launch artifact")
    processor = _mapping(artifacts.get("processor"), "processor launch artifact")
    checkpoint = qwen.get("path")
    if (
        not isinstance(checkpoint, str) or not checkpoint
        or processor.get("path") != checkpoint
        or qwen.get("kind") != "directory"
        or processor.get("kind") != "processor_files"
    ):
        raise ValueError("launch Qwen/processor artifacts do not share one checkpoint")
    _sha256(qwen.get("sha256"), "Qwen artifact digest")
    _sha256(processor.get("sha256"), "processor artifact digest")
    environment = _mapping(manifest.get("environment"), "launch environment")
    packages = _mapping(environment.get("packages"), "launch packages")
    versions = {}
    for name in ("pillow", "torch", "transformers"):
        value = packages.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"launch package {name} version is missing")
        versions[name] = value
    fingerprint = {
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
        "runtime_pillow_version": versions["pillow"],
        "runtime_torch_version": versions["torch"],
        "runtime_transformers_version": versions["transformers"],
        "tokenizer_class": (
            "transformers.models.qwen2.tokenization_qwen2_fast."
            "Qwen2TokenizerFast"
        ),
        "tokenizer_padding_side_default": "right",
        "wrapper_device": "cuda:0", "wrapper_dtype": "torch.bfloat16",
        "wrapper_flash_attention": True,
    }
    return {"checkpoint": checkpoint, "processor_fingerprint": fingerprint}


def validate_launch_pair(
    disabled_rows: Sequence[Mapping[str, Any]],
    enabled_rows: Sequence[Mapping[str, Any]],
    disabled_manifest: Mapping[str, Any], enabled_manifest: Mapping[str, Any],
) -> dict[str, str]:
    """Bind a disabled/P2C pair to one non-config launch identity."""
    disabled_manifest = dict(_mapping(disabled_manifest, "disabled launch manifest"))
    enabled_manifest = dict(_mapping(enabled_manifest, "enabled launch manifest"))
    if disabled_manifest.get("schema_version") != 1 or enabled_manifest.get("schema_version") != 1:
        raise ValueError("launch manifest schema version is invalid")
    disabled_config = _mapping(disabled_manifest.get("config"), "disabled manifest config")
    enabled_config = _mapping(enabled_manifest.get("config"), "enabled manifest config")
    _validate_config_pair(
        _mapping(disabled_config.get("loaded"), "disabled loaded config"),
        _mapping(enabled_config.get("loaded"), "enabled loaded config"),
    )
    disabled_model = _launch_model_contract(disabled_manifest)
    enabled_model = _launch_model_contract(enabled_manifest)
    if disabled_model != enabled_model:
        raise ValueError("disabled and P2C launch model contracts differ")
    disabled_identity = dict(disabled_manifest)
    enabled_identity = dict(enabled_manifest)
    disabled_identity.pop("config")
    enabled_identity.pop("config")
    if disabled_identity != enabled_identity:
        raise ValueError("disabled and P2C launch manifests differ in non-config identity")
    ordinals = [row.get("_eg_ordinal") for row in disabled_rows]
    if ordinals != [row.get("_eg_ordinal") for row in enabled_rows]:
        raise ValueError("launch pair row ordinals differ")
    partition = _mapping(disabled_manifest.get("selected_partition"), "selected partition")
    if partition.get("ordinals") != ordinals:
        raise ValueError("launch manifest partition differs from JSONL ordinals")
    code = _mapping(disabled_manifest.get("code"), "launch code")
    revision = _sha256(code.get("revision"), "launch code revision")
    for row in tuple(disabled_rows) + tuple(enabled_rows):
        if row.get("_eg_code_revision") != revision:
            raise ValueError("launch revision differs from JSONL")
    disabled_fingerprint = canonical_sha256(disabled_manifest)
    enabled_fingerprint = canonical_sha256(enabled_manifest)
    if any(row.get("_eg_run_fingerprint") != disabled_fingerprint for row in disabled_rows):
        raise ValueError("disabled JSONL fingerprint differs from launch manifest")
    if any(row.get("_eg_run_fingerprint") != enabled_fingerprint for row in enabled_rows):
        raise ValueError("P2C JSONL fingerprint differs from launch manifest")
    return {"disabled": disabled_fingerprint, "enabled": enabled_fingerprint}


def _support_result(value: Any, context: str) -> EvidenceSupportResult | None:
    if value is None:
        return None
    payload = dict(_mapping(value, context))
    if frozenset(payload) != _SUPPORT_FIELDS:
        raise ValueError(f"{context} has an invalid exact schema")
    raw_requirements = _list(payload["requirements"], f"{context}.requirements")
    requirements = []
    for item in raw_requirements:
        item = _mapping(item, f"{context}.requirement")
        if set(item) != {"requirement_id", "kind", "text"}:
            raise ValueError(f"{context} requirement schema is invalid")
        requirements.append(EvidenceRequirement(
            item["requirement_id"], item["kind"], item["text"],
        ))
    observation_identity = json.dumps(
        payload["observation_identity"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )
    fingerprint = json.dumps(
        payload["processor_fingerprint"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )
    result = EvidenceSupportResult(
        requirements=tuple(requirements),
        requirement_set_id=payload["requirement_set_id"],
        observation_identity=observation_identity,
        prompt_version=payload["prompt_version"],
        prompt_template_sha256=payload["prompt_template_sha256"],
        prompt_sha256=payload["prompt_sha256"],
        processor_mode=payload["processor_mode"],
        processor_fingerprint_json=fingerprint, checkpoint=payload["checkpoint"],
        yes_tokenization=tuple(payload["yes_tokenization"]),
        no_tokenization=tuple(payload["no_tokenization"]),
        yes_token_id=payload["yes_token_id"], no_token_id=payload["no_token_id"],
        p_yes_transform=payload["p_yes_transform"],
        yes_logit=payload["yes_logit"], no_logit=payload["no_logit"],
        p_yes=payload["p_yes"], p_no=payload["p_no"],
        support_avg=payload["support_avg"], support_min=payload["support_min"],
        observation_mode=payload["observation_mode"],
        observation_size=tuple(payload["observation_size"]),
        view_sha256=payload["view_sha256"], elapsed_seconds=payload["elapsed_seconds"],
        logical_calls=payload["logical_calls"],
        accounted_pixels=payload["accounted_pixels"],
        batch_plan_hash=payload["batch_plan_hash"],
    )
    if result.to_dict() != payload:
        raise ValueError(f"{context} is not the canonical support DTO")
    return result


def _batch_result(value: Any) -> ObservationBatchResult | None:
    if value is None:
        return None
    payload = dict(_mapping(value, "ZOOM batch result"))
    if frozenset(payload) != _BATCH_FIELDS:
        raise ValueError("ZOOM batch result has an invalid exact schema")
    result = ObservationBatchResult(
        status=payload["status"], batch_plan=payload["batch_plan"],
        admitted=payload["admitted"], charged=payload["charged"],
        ledger_before=payload["ledger_before"], ledger_after=payload["ledger_after"],
        current_support=_support_result(payload["current_support"], "current support"),
        candidate_support=_support_result(payload["candidate_support"], "candidate support"),
        candidate_answer=payload["candidate_answer"],
        failure_phase=payload["failure_phase"], failure_reason=payload["failure_reason"],
        exception_type=payload["exception_type"],
        executed_stages=tuple(payload["executed_stages"]),
        elapsed_seconds=payload["elapsed_seconds"],
    )
    if result.to_dict() != payload:
        raise ValueError("ZOOM batch result is not the canonical DTO")
    return result


def _validate_forced_step(
    step: Mapping[str, Any], *, index: int, answer: Mapping[str, Any],
    budget: Mapping[str, int],
) -> None:
    if frozenset(step) != _BASE_STEP_FIELDS:
        raise ValueError("FORCED_RETURN step has an invalid exact schema")
    if (
        step.get("step") != index or step.get("action") != "FORCED_RETURN"
        or step.get("gap_fallback_used") is not False
        or _finite(step.get("elapsed_seconds"), "forced elapsed", minimum=0.0) != 0.0
        or step.get("focus_key") is not None or step.get("feasible_actions") != []
        or step.get("gaps") != {}
        or step.get("no_op_reason") != "minimal_v1 has no certified-stop controller"
        or step.get("answer") != answer
        or _finite(step.get("support_avg"), "forced support_avg") != 0.0
        or _finite(step.get("support_min"), "forced support_min") != 0.0
        or step.get("certified") is not False
        or _budget(step.get("budget"), "forced budget") != budget
    ):
        raise ValueError("FORCED_RETURN step does not match its exact P0 trace")


def _query_requirements(
    trace: Mapping[str, Any], question: Any,
) -> tuple[tuple[EvidenceRequirement, ...], bool]:
    plan = _mapping(trace.get("query_plan"), "trace query plan")
    if frozenset(plan) != _QUERY_PLAN_FIELDS:
        raise ValueError("trace query plan has an invalid exact schema")
    if not isinstance(question, str) or plan.get("main_query") != question:
        raise ValueError("trace query plan main query differs from the input")
    for field in ("targets", "augmented_queries"):
        values = _list(plan.get(field), f"query plan {field}")
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"query plan {field} must contain nonempty strings")
    if (
        plan.get("global_scope_required") not in (True, False)
        or not isinstance(plan.get("global_scope_required"), bool)
        or plan.get("fallback_used") not in (True, False)
        or not isinstance(plan.get("fallback_used"), bool)
    ):
        raise TypeError("query plan scope/fallback fields must be booleans")
    evidence_items = _list(plan.get("evidence_items"), "query plan evidence items")
    try:
        return sanitize_evidence_requirements(evidence_items), False
    except (TypeError, ValueError):
        return (), True


def _validate_common_trace(
    trace: Mapping[str, Any], *, config: Mapping[str, Any], enabled: bool,
    output: Any, context: str,
) -> tuple[dict[str, int], float, Mapping[str, Any], Mapping[str, Any] | None]:
    if frozenset(trace) != _TRACE_FIELDS:
        raise ValueError(f"{context} MethodTrace has an invalid exact schema")
    if trace.get("effective_config") != config or trace.get("config_id") != config["config_id"]:
        raise ValueError(f"{context} config identity mismatch")
    if trace.get("termination") != "FORCED_RETURN":
        raise ValueError(f"{context} must terminate by forced return")
    if trace.get("budget_interrupted") is not False:
        raise ValueError(f"{context} budget interruption is ineligible")
    if trace.get("pixel_accounting") != config["pixel_accounting"]:
        raise ValueError(f"{context} pixel accounting mismatch")
    elapsed = _finite(trace.get("elapsed_seconds"), f"{context}.elapsed", minimum=0.0)
    budget = _budget(trace.get("budget"), f"{context}.budget")
    final_answer = _mapping(trace.get("final_answer"), f"{context}.final_answer")
    anchor_answer = _mapping(trace.get("anchor_answer"), f"{context}.anchor_answer")
    if final_answer != anchor_answer or _answer_output(final_answer, context) != output:
        raise ValueError(f"{context} final/anchor answer differs from emitted output")
    steps = _list(trace.get("steps"), f"{context}.steps")
    if not all(isinstance(step, Mapping) for step in steps):
        raise ValueError(f"{context}.steps must contain objects")
    if enabled:
        if len(steps) != 2 or [step.get("action") for step in steps] != ["ZOOM", "FORCED_RETURN"]:
            raise ValueError("P2C trace must contain exactly ZOOM then FORCED_RETURN")
        if any(trace.get(name) is not None for name in (
            "anchor_state_score", "selected_state_score", "replacement_margin",
        )):
            raise ValueError("P2C selector scores and replacement must remain disabled")
        zoom_step = steps[0]
        _validate_forced_step(steps[1], index=1, answer=final_answer, budget=budget)
    else:
        if len(steps) != 1 or steps[0].get("action") != "FORCED_RETURN":
            raise ValueError("disabled sibling must contain only FORCED_RETURN")
        anchor_score = _finite(trace.get("anchor_state_score"), "disabled anchor score")
        selected_score = _finite(trace.get("selected_state_score"), "disabled selected score")
        if (
            anchor_score != selected_score or trace.get("replacement_margin") != 0.0
            or trace.get("support_status") != "not_observed"
        ):
            raise ValueError("disabled selector/support trace is inconsistent")
        _validate_forced_step(steps[0], index=0, answer=final_answer, budget=budget)
        zoom_step = None
    return budget, elapsed, final_answer, zoom_step


def _canonical_p0_stability(
    benchmark: str, options: list[Any], p0_anchor: Mapping[str, Any],
    step_answer: Mapping[str, Any], audit: Mapping[str, Any],
) -> None:
    stability = _mapping(audit.get("p0_stability"), "ZOOM P0 stability")
    if benchmark in _HR_BENCHMARKS:
        raw = p0_anchor.get("cvsearch_raw")
        if not isinstance(raw, list) or len(raw) != 4 or not all(isinstance(item, str) for item in raw):
            raise ValueError("HR P0 raw output must contain four strings")
        expected = aggregate_hr_answers(options, raw)
        expected.output = copy.deepcopy(p0_anchor.get("emitted_answer"))
        expected.selected_from = "cvsearch_raw"
        if stability != expected.to_dict():
            raise ValueError("HR P0 stability is not canonical")
    elif stability != step_answer:
        raise ValueError("V* P0 stability must equal the exact P0 answer record")
    if _finite(audit.get("uncertainty"), "ZOOM uncertainty") != _finite(
        stability.get("uncertainty"), "ZOOM P0 uncertainty",
    ):
        raise ValueError("ZOOM uncertainty differs from P0 stability")


def _validate_p0_anchor(
    benchmark: str, options: list[Any], output: Any, audit: Mapping[str, Any],
    step_answer: Mapping[str, Any], batch: ObservationBatchResult | None,
    no_op_reason: Any,
) -> Mapping[str, Any]:
    anchor = _mapping(audit.get("p0_anchor"), "ZOOM P0 anchor")
    if frozenset(anchor) != _P0_FIELDS or anchor.get("emitted_answer") != output:
        raise ValueError("ZOOM P0 anchor schema/output is invalid")
    node_keys = _list(anchor.get("node_keys"), "ZOOM P0 node keys")
    if (
        any(not isinstance(key, str) or not key for key in node_keys)
        or len(node_keys) != len(set(node_keys))
        or audit.get("current_keys") != node_keys
    ):
        raise ValueError("ZOOM P0/current keys are invalid")
    support_value = anchor.get("support_view")
    if support_value is None:
        support = None
    else:
        support = _list(support_value, "ZOOM P0 support view")
        if [item.get("canonical_key") for item in support if isinstance(item, Mapping)] != node_keys:
            raise ValueError("ZOOM P0 support view differs from its keys")
        if len(support) != len(node_keys) or any(not isinstance(item, Mapping) for item in support):
            raise ValueError("ZOOM P0 support view descriptors are invalid")
    if benchmark in _HR_BENCHMARKS:
        if (
            anchor.get("producing_phase") != "cvsearch_raw"
            or anchor.get("cvsearch_raw") != output
            or not isinstance(output, list) or len(output) != 4
        ):
            raise ValueError("HR ZOOM P0 must bind exact cvsearch_raw")
    else:
        raw = anchor.get("cvsearch_raw")
        if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw < len(options):
            raise ValueError("V* cvsearch_raw is not a valid option index")
        phase = anchor.get("producing_phase")
        if phase == "root":
            # cvsearch_raw is the raw root response; emitted_answer is the
            # aggregated P0. They are intentionally distinct identities.
            if node_keys or support != []:
                raise ValueError("V* root P0 must bind the empty root view")
        elif phase == "search":
            if raw != output:
                raise ValueError("V* search P0 must bind exact cvsearch_raw")
        else:
            raise ValueError("V* P0 phase must be root or search")
    _canonical_p0_stability(benchmark, options, anchor, step_answer, audit)
    if batch is not None and "current_observation" in batch.batch_plan:
        expected = batch.batch_plan["current_observation"]["descriptors"]
        if support != expected or not support:
            raise ValueError("ZOOM batch is not bound to the exact local P0 descriptors")
    elif batch is None:
        if no_op_reason == "zoom_p0_support_view_unavailable" and support is not None:
            raise ValueError("unavailable P0 support must be null")
        if no_op_reason == "zoom_p0_support_view_empty" and (support != [] or node_keys):
            raise ValueError("empty P0 support no-op must bind an empty view")
        if no_op_reason == "zoom_p0_support_view_nonlocal" and (
            not support or not any(
                item.get("source") == "global" or item.get("renderer_kind") == "root"
                for item in support
            )
        ):
            raise ValueError("nonlocal P0 support no-op lacks a nonlocal descriptor")
        if no_op_reason == "zoom_batch_preflight_failed" and (
            not support or any(
                item.get("source") == "global" or item.get("renderer_kind") == "root"
                for item in support
            )
        ):
            raise ValueError("batch-preflight no-op lacks an eligible local P0 view")
    return anchor


def _validate_zoom_step(
    benchmark: str, row: Mapping[str, Any], trace: Mapping[str, Any],
    step: Mapping[str, Any], disabled_budget: Mapping[str, int],
    enabled_budget: Mapping[str, int], final_answer: Mapping[str, Any],
    config: Mapping[str, Any], requirements: tuple[EvidenceRequirement, ...],
    invalid_requirements: bool, model_contract: Mapping[str, Any],
) -> tuple[bool, Any, str, float | None, float]:
    if frozenset(step) != _BASE_STEP_FIELDS | {"zoom_audit"}:
        raise ValueError("ZOOM StepTrace has an invalid exact schema")
    if (
        step.get("step") != 0 or step.get("action") != "ZOOM"
        or step.get("gap_fallback_used") is not False
        or _finite(step.get("elapsed_seconds"), "ZOOM step elapsed", minimum=0.0) != 0.0
        or _finite(step.get("support_avg"), "ZOOM support_avg") != 0.0
        or _finite(step.get("support_min"), "ZOOM support_min") != 0.0
        or step.get("certified") is not False or step.get("answer") != final_answer
        or _budget(step.get("budget"), "ZOOM step budget") != enabled_budget
    ):
        raise ValueError("ZOOM StepTrace fixed P0 fields are invalid")
    audit = _mapping(step.get("zoom_audit"), "ZOOM audit")
    if frozenset(audit) != _AUDIT_FIELDS:
        raise ValueError("ZOOM audit has an invalid exact schema")
    if (
        audit.get("render_policy") != config["p2c_zoom_render_policy"]
        or audit.get("base_view_size") != 336
        or audit.get("candidate_view_size") != 112
        or audit.get("support_proxy_status") != "audit_only_uncalibrated"
        or audit.get("coverage_status") != "not_observed"
        or audit.get("verifier_status") != "disabled_same_checkpoint_unpromoted"
        or audit.get("verifier_avg") is not None or audit.get("verifier_min") is not None
        or audit.get("score_margin") is not None
        or audit.get("score_status") != "unavailable_missing_verifier_coverage"
    ):
        raise ValueError("ZOOM audit frozen policy/status fields are invalid")
    batch = _batch_result(audit.get("batch_result"))
    no_op_reason = step.get("no_op_reason")
    _validate_p0_anchor(
        benchmark, _list(row.get("options"), "options"), row.get("output"),
        audit, final_answer, batch, no_op_reason,
    )
    if batch is None:
        if no_op_reason == "zoom_invalid_evidence_requirements":
            requirements_match_reason = invalid_requirements
        elif no_op_reason == "zoom_no_evidence_requirements":
            requirements_match_reason = not invalid_requirements and not requirements
        else:
            requirements_match_reason = not invalid_requirements and bool(requirements)
        if (
            no_op_reason not in _NO_BATCH_REASONS
            or not requirements_match_reason
            or audit.get("zoom_keys") != [] or audit.get("coordinate_mapping") != []
            or step.get("focus_key") is not None or step.get("feasible_actions") != []
            or step.get("gaps") != {} or enabled_budget != disabled_budget
        ):
            raise ValueError("ZOOM no-batch trace is not an exact legal no-op")
        status = "no_batch"
        expected_cost = None
        plan = None
    else:
        plan = batch.batch_plan
        expected_order = [item.requirement_id for item in requirements]
        if plan.get("batch_kind") != "p2c_post_anchor_coordinate_zoom":
            raise ValueError("ZOOM batch plan kind is invalid")
        if invalid_requirements or not requirements:
            raise ValueError("ZOOM batch requires valid nonempty query requirements")
        if (
            plan.get("requirement_order") != expected_order
            or plan.get("requirement_set_id")
            != EvidenceSupportResult.requirement_set_id_for(requirements)
        ):
            raise ValueError("ZOOM batch requirements differ from the exact query plan")
        if (
            plan.get("checkpoint") != model_contract["checkpoint"]
            or plan.get("processor_fingerprint")
            != model_contract["processor_fingerprint"]
        ):
            raise ValueError("ZOOM batch model provenance differs from launch identity")
        if plan.get("q0_sha256") != hashlib.sha256(row["question"].encode()).hexdigest():
            raise ValueError("ZOOM plan q0 identity differs from the input")
        if plan.get("answer_type") != row.get("answer_type"):
            raise ValueError("ZOOM plan answer type differs from the input")
        if plan.get("candidate_option_count") != len(row["options"]):
            raise ValueError("ZOOM plan option count differs from the input")
        if (
            audit.get("current_keys") != plan.get("current_keys")
            or audit.get("zoom_keys") != plan.get("zoom_keys")
            or audit.get("coordinate_mapping") != plan.get("coordinate_mapping", [])
            or audit.get("render_policy") != plan.get("render_policy")
            or audit.get("base_view_size") != plan.get("base_view_size")
            or audit.get("candidate_view_size") != plan.get("candidate_view_size")
        ):
            raise ValueError("ZOOM audit identity/geometry differs from its plan")
        before = _budget(batch.to_dict()["ledger_before"], "ZOOM ledger before")
        after = _budget(batch.to_dict()["ledger_after"], "ZOOM ledger after")
        if before != disabled_budget or after != enabled_budget:
            raise ValueError("ZOOM batch ledgers are not paired to P0/final budgets")
        expected_cost = (after["mllm_calls"] - before["mllm_calls"]) / after["max_mllm_calls"]
        if batch.status == "success":
            matched = all((
                plan.get("prompt_version") == config["evidence_support_prompt_version"],
                plan.get("processor_mode") == config["evidence_support_processor_mode"],
                plan.get("prompt_template_sha256")
                == config["evidence_support_prompt_template_sha256"],
                plan.get("p_yes_transform")
                == config["evidence_support_probability_transform"],
            ))
            expected_contract = "matched" if matched else "mismatch"
        else:
            matched = False
            expected_contract = "not_observed"
        if audit.get("support_contract_status") != expected_contract:
            raise ValueError("ZOOM support contract status is not derived from the exact plan")
        expected_reason = (
            None if batch.status == "success" and matched
            else "zoom_support_contract_mismatch" if batch.status == "success"
            else f"zoom_{batch.status}"
        )
        if no_op_reason != expected_reason:
            raise ValueError("ZOOM no-op reason differs from its batch status")
        encoded = json.dumps(plan.get("zoom_keys", []), separators=(",", ":"), allow_nan=False)
        expected_focus = (
            None if not plan.get("zoom_keys") else
            "zoom-aggregate-" + hashlib.sha256(encoded.encode()).hexdigest()
        )
        if step.get("focus_key") != expected_focus:
            raise ValueError("ZOOM aggregate focus identity is invalid")
        status = "success" if matched and batch.status == "success" else (
            "support_mismatch" if batch.status == "success" else batch.status
        )

    current_payload = None if batch is None or batch.current_support is None else batch.current_support.to_dict()
    candidate_payload = None if batch is None or batch.candidate_support is None else batch.candidate_support.to_dict()
    if (
        audit.get("current_gap_support") != current_payload
        or audit.get("candidate_gap_support") != candidate_payload
        or audit.get("normalized_actual_cost") != expected_cost
    ):
        raise ValueError("ZOOM support/cost audit differs from its batch")
    feasible = status == "success"
    if audit.get("feasible") is not feasible:
        raise ValueError("ZOOM feasibility differs from the exact batch contract")
    expected_support_status = (
        "observed_answer_free_audit_only" if feasible else no_op_reason
    )
    if trace.get("support_status") != expected_support_status:
        raise ValueError("ZOOM trace support status differs from its exact outcome")
    if feasible:
        current_p = batch.current_support.p_yes
        candidate_p = batch.candidate_support.p_yes
        gap = 1.0 - current_p
        delta = candidate_p - current_p
        if (
            abs(_finite(audit.get("uncalibrated_g_zoom_proxy"), "ZOOM proxy") - gap) > 1e-12
            or abs(_finite(audit.get("support_delta"), "ZOOM support delta") - delta) > 1e-12
            or audit.get("replacement_reason") != "replacement_disabled_p2c"
            or step.get("feasible_actions") != ["ZOOM"]
            or step.get("gaps") != {"g_zoom_proxy_audit_only": gap}
        ):
            raise ValueError("feasible ZOOM measurements/trace are inconsistent")
        candidate = batch.candidate_answer
        if benchmark in _HR_BENCHMARKS:
            expected_stability = aggregate_hr_answers(row["options"], candidate).to_dict()
        else:
            expected_stability = aggregate_vstar_losses([candidate["losses"]]).to_dict()
        if audit.get("candidate_stability") != expected_stability:
            raise ValueError("ZOOM candidate projection is not canonical")
        return True, candidate, status, delta, batch.elapsed_seconds
    if (
        audit.get("uncalibrated_g_zoom_proxy") is not None
        or audit.get("support_delta") is not None
        or audit.get("candidate_stability") is not None
        or audit.get("replacement_reason") is not None
        or step.get("feasible_actions") != [] or step.get("gaps") != {}
    ):
        raise ValueError("infeasible ZOOM cannot synthesize candidate state")
    return False, None, status, None, 0.0 if batch is None else batch.elapsed_seconds


def _validate_pairs(
    benchmark: str, disabled_rows: Sequence[Mapping[str, Any]],
    enabled_rows: Sequence[Mapping[str, Any]], expectation: PairExpectation,
    model_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if benchmark != "vstar" and benchmark not in _HR_BENCHMARKS:
        raise ValueError("P2C scorer supports only V* and HR-4K/8K")
    if len(disabled_rows) != expectation.topics or len(enabled_rows) != expectation.topics:
        raise ValueError("paired topic count differs from the frozen expectation")
    disabled_config = load_method_config(str(DISABLED_CONFIG))
    enabled_config = load_method_config(str(ENABLED_CONFIG))
    _validate_config_pair(disabled_config, enabled_config)
    pairs = []
    previous = -1
    revisions = set()
    disabled_fingerprints = set()
    enabled_fingerprints = set()
    for disabled, enabled in zip(disabled_rows, enabled_rows):
        if not isinstance(disabled, Mapping) or not isinstance(enabled, Mapping):
            raise TypeError("paired JSONL rows must be objects")
        ordinal = disabled.get("_eg_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal <= previous:
            raise ValueError("disabled ordinals must be unique and increasing")
        previous = ordinal
        if enabled.get("_eg_ordinal") != ordinal:
            raise ValueError("paired ordinals differ")
        revision = _sha256(disabled.get("_eg_code_revision"), "disabled inference revision")
        if enabled.get("_eg_code_revision") != revision:
            raise ValueError("paired inference revisions differ")
        revisions.add(revision)
        disabled_fingerprints.add(_sha256(disabled.get("_eg_run_fingerprint"), "disabled fingerprint"))
        enabled_fingerprints.add(_sha256(enabled.get("_eg_run_fingerprint"), "enabled fingerprint"))
        for field in ("input_image", "question", "options", "answer_type", "answer"):
            if disabled.get(field) != enabled.get(field):
                raise ValueError(f"paired rows differ in exact {field}")
        answer_type = disabled.get("answer_type")
        expected_type = "logits_match" if benchmark == "vstar" else "option_list"
        if answer_type != expected_type:
            raise ValueError("benchmark and answer type disagree")
        output = disabled.get("output")
        if enabled.get("output") != output:
            raise ValueError("P2C emitted output differs from disabled P0")
        disabled_trace = _mapping(disabled.get("method_trace"), "disabled trace")
        enabled_trace = _mapping(enabled.get("method_trace"), "enabled trace")
        for field in _SHARED_P0_TRACE_FIELDS:
            if disabled_trace.get(field) != enabled_trace.get(field):
                raise ValueError(f"paired P0 traces differ in exact {field}")
        requirements, invalid_requirements = _query_requirements(
            enabled_trace, enabled.get("question"),
        )
        disabled_budget, disabled_elapsed, disabled_answer, _ = _validate_common_trace(
            disabled_trace, config=disabled_config, enabled=False,
            output=output, context=f"disabled row {ordinal}",
        )
        enabled_budget, enabled_elapsed, enabled_answer, zoom_step = _validate_common_trace(
            enabled_trace, config=enabled_config, enabled=True,
            output=output, context=f"enabled row {ordinal}",
        )
        if enabled_answer != disabled_answer:
            raise ValueError("P2C P0 answer record differs from the disabled sibling")
        feasible, candidate, status, support_delta, batch_elapsed = _validate_zoom_step(
            benchmark, enabled, enabled_trace, zoom_step, disabled_budget,
            enabled_budget, enabled_answer, enabled_config, requirements,
            invalid_requirements, model_contract,
        )
        pairs.append({
            "ordinal": ordinal, "disabled": disabled, "enabled": enabled,
            "p0": output, "candidate": candidate, "feasible": feasible,
            "status": status, "support_delta": support_delta,
            "batch_elapsed": batch_elapsed, "disabled_budget": disabled_budget,
            "enabled_budget": enabled_budget, "disabled_elapsed": disabled_elapsed,
            "enabled_elapsed": enabled_elapsed,
        })
    if len(revisions) != 1 or len(disabled_fingerprints) != 1 or len(enabled_fingerprints) != 1:
        raise ValueError("each paired variant must use one revision and run fingerprint")
    digest = canonical_output_digest(disabled_rows)
    if expectation.output_digest is not None and digest != expectation.output_digest:
        raise ValueError("disabled output digest differs from the frozen reference")
    return pairs


def score_paired_rows(
    benchmark: str, disabled_rows: Sequence[Mapping[str, Any]],
    enabled_rows: Sequence[Mapping[str, Any]], expectation: PairExpectation,
    *, launch_manifest: Mapping[str, Any],
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Validate all label-blind P2C invariants, then score raw candidates."""
    model_contract = _launch_model_contract(launch_manifest)
    pairs = _validate_pairs(
        benchmark, disabled_rows, enabled_rows, expectation, model_contract,
    )
    label_pairs = pairs
    if benchmark == "vstar":
        # ObservationBatchResult has already bound the raw winner to loss argmin.
        # Project that raw winner only at the reused Phase-2 label boundary; never
        # score the derived candidate_stability AnswerRecord.
        label_pairs = [
            dict(pair, candidate=pair["candidate"]["winner"])
            if pair["feasible"] else dict(pair)
            for pair in pairs
        ]
    rows = _score_labels(benchmark, label_pairs)  # Deliberate evaluator-label boundary.
    p0_correct = sum(row["p0_correct"] for row in rows)
    cycles = sum(row["total"] for row in rows)
    if p0_correct != expectation.p0_correct or cycles != expectation.cycles:
        raise ValueError("disabled score/cycles differ from the frozen reference")
    feasible = [row for row in rows if row["feasible"]]
    candidate_correct = sum(row["candidate_correct"] for row in feasible)
    candidate_total = sum(row["total"] for row in feasible)
    oracle_correct = sum(row["oracle_correct"] for row in rows)
    corrected = sum(max(0, row["oracle_correct"] - row["p0_correct"]) for row in rows)
    corrupted = sum(max(0, row["p0_correct"] - row["oracle_correct"]) for row in rows)
    support_deltas = [row["support_delta"] for row in feasible]
    disabled_digest = canonical_output_digest(disabled_rows)
    enabled_digest = canonical_output_digest(enabled_rows)
    return {
        "benchmark": benchmark,
        "sample": {"topics": len(rows), "cycles": cycles, "topic_unit": "input row; HR shuffles bound"},
        "revision": rows[0]["disabled"]["_eg_code_revision"],
        "output_digest": {
            "disabled": disabled_digest, "enabled": enabled_digest,
            "exact_match": disabled_digest == enabled_digest,
        },
        "p0": {"correct": p0_correct, "total": cycles, "accuracy": p0_correct / cycles},
        "candidate_alone": {
            "correct": candidate_correct, "total": candidate_total,
            "accuracy": None if not candidate_total else candidate_correct / candidate_total,
            "corrected_vs_p0": sum(max(0, row["candidate_correct"] - row["p0_correct"]) for row in feasible),
            "corrupted_vs_p0": sum(max(0, row["p0_correct"] - row["candidate_correct"]) for row in feasible),
        },
        "oracle": {
            "correct": oracle_correct, "total": cycles, "accuracy": oracle_correct / cycles,
            "delta": (oracle_correct - p0_correct) / cycles,
            "corrected": corrected, "corrupted": corrupted,
            "changed_topics": sum(row["changed"] for row in rows),
            "selected_candidate_topics": sum(row["selected_candidate"] for row in rows),
            "tie_rule": "retain_p0",
        },
        "candidate": {
            "feasible_topics": len(feasible), "infeasible_topics": len(rows) - len(feasible),
            "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        },
        "cost": {
            "disabled": _runtime_report(rows, "disabled"),
            "enabled": _runtime_report(rows, "enabled"),
            "zoom_batch_latency_s": {
                "total": sum(row["batch_elapsed"] for row in rows),
                "p50": statistics.median([row["batch_elapsed"] for row in rows]),
                "p95": _percentile([row["batch_elapsed"] for row in rows], 0.95),
            },
        },
        "support_delta": {
            "count": len(support_deltas),
            "mean": None if not support_deltas else statistics.fmean(support_deltas),
            "min": None if not support_deltas else min(support_deltas),
            "max": None if not support_deltas else max(support_deltas),
        },
        "audit": {
            "replacements": 0, "forced_returns": {"disabled": len(rows), "enabled": len(rows)},
            "labels_used_only_after_pair_trace_config_budget_validation": True,
            "candidate_stability_output_scored": False,
        },
        "bootstrap": _bootstrap(benchmark, rows, bootstrap_replicates),
    }


def score_dev_paths(paths: Mapping[str, tuple[str | Path, str | Path]]) -> dict[str, Any]:
    if set(paths) != set(FROZEN_DEV_EXPECTATIONS):
        raise ValueError("P2C dev scorer requires V*, HR-4K, and HR-8K")
    evaluator_before = _evaluator_revision()
    jsonl_paths = [Path(path) for pair in paths.values() for path in pair]
    manifest_paths = [Path(f"{path}.launch-manifest.json") for path in jsonl_paths]
    all_paths = jsonl_paths + manifest_paths
    snapshots = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in all_paths}
    reports = {}
    fingerprints = {}
    for benchmark, pair in paths.items():
        disabled_rows = load_jsonl(pair[0])
        enabled_rows = load_jsonl(pair[1])
        disabled_manifest = load_launch_manifest(pair[0])
        enabled_manifest = load_launch_manifest(pair[1])
        validate_frozen_dev_identity(benchmark, disabled_manifest, disabled_rows)
        validate_frozen_dev_identity(benchmark, enabled_manifest, enabled_rows)
        fingerprints[benchmark] = validate_launch_pair(
            disabled_rows, enabled_rows, disabled_manifest, enabled_manifest,
        )
        reports[benchmark] = score_paired_rows(
            benchmark, disabled_rows, enabled_rows, FROZEN_DEV_EXPECTATIONS[benchmark],
            launch_manifest=enabled_manifest,
        )
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in all_paths}
    evaluator_after = _evaluator_revision()
    if after != snapshots or evaluator_after != evaluator_before:
        raise RuntimeError("P2C evaluator inputs or source changed during scoring")
    revisions = {report["revision"] for report in reports.values()}
    if len(revisions) != 1:
        raise ValueError("P2C dev inputs must share one inference revision")

    def hashes(suffix: str) -> dict[str, dict[str, str]]:
        return {
            benchmark: {
                variant: snapshots[str(Path(f"{path}{suffix}"))]
                for variant, path in zip(("disabled", "enabled"), paths[benchmark])
            }
            for benchmark in FROZEN_DEV_EXPECTATIONS
        }

    hr4 = reports["hr-bench_4k"]["oracle"]["delta"]
    hr8 = reports["hr-bench_8k"]["oracle"]["delta"]
    vstar = reports["vstar"]["oracle"]["delta"]
    hr_gate = hr4 >= 0.0 and hr8 >= 0.0 and min(hr4, hr8) > 0.0
    vstar_gate = vstar >= 0.0
    return {
        "schema_version": 2,
        "task": "unified-evidence-gap-phase3-p2c-coordinate-zoom-oracle-dev",
        "execution_identity": {
            "inference_revision": next(iter(revisions)),
            "evaluator_revision": evaluator_before,
            "input_jsonl_sha256": hashes(""),
            "launch_manifest_sha256": hashes(".launch-manifest.json"),
        },
        "benchmarks": reports, "launch_fingerprints": fingerprints,
        "cross_resolution_inference": "independent samples; no shared-sample CI",
        "gate": {
            "hr4_nonnegative": hr4 >= 0.0, "hr8_nonnegative": hr8 >= 0.0,
            "hr_minimum_strictly_positive": min(hr4, hr8) > 0.0,
            "vstar_oracle_nonnegative": vstar_gate,
            "p2d_warranted": hr_gate and vstar_gate,
            "ci_changes_predeclared_gate": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("vstar", "hr4", "hr8"):
        parser.add_argument(f"--{flag}-disabled", required=True, type=Path)
        parser.add_argument(f"--{flag}-enabled", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = score_dev_paths({
        "vstar": (args.vstar_disabled, args.vstar_enabled),
        "hr-bench_4k": (args.hr4_disabled, args.hr4_enabled),
        "hr-bench_8k": (args.hr8_disabled, args.hr8_enabled),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
