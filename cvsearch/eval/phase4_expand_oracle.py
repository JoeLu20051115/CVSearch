"""Strict evaluator-only scorer for paired P4A spatial-context EXPAND artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import tempfile
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

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
from cvsearch.eval.phase3_zoom_oracle import (
    _batch_result,
    _launch_model_contract,
)
from cvsearch.evidence_gap.answers import aggregate_hr_answers, aggregate_vstar_losses
from cvsearch.evidence_gap.method import load_method_config
from cvsearch.evidence_gap.provenance import canonical_sha256
from cvsearch.evidence_gap.types import (
    AnswerRecord,
    EvidenceRequirement,
    EvidenceSupportResult,
    ObservationBatchResult,
    _validate_batch_plan,
    sanitize_evidence_requirements,
)
from cvsearch.models.utils import (
    expand2square,
    merge_bbox_list,
    union_all_bboxes,
    visualize_bbox_and_arrow,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "reproduction/evidence_gap/configs"
DISABLED_CONFIG = (
    CONFIG_ROOT / "dev_unified_expand_context_disabled_gamma000_budget512.json"
)
ENABLED_CONFIG = (
    CONFIG_ROOT / "dev_unified_expand_context_oracle_gamma000_budget512.json"
)
_HR_BENCHMARKS = frozenset({"hr-bench_4k", "hr-bench_8k"})

_SELECTION_POLICY = "nearest_spatial_native_cvsearch_context_v1"
_COMPOSITION_POLICY = "focus_top_blank_or_context_bottom_native_pixels_v1"
_FROZEN_BASE_VIEW_SIZE = 336
_FROZEN_SCALE_SIZE = 672
_FROZEN_PATCH_SCALE = 1.2
_FROZEN_INFERENCE_REVISION = (
    "c13460ba0982b0f0a9ca62473d0d8cafddc4224fb1f171fd7779390bde856732"
)
_FROZEN_ARTIFACT_KINDS = {
    "annotation_file": "file",
    "ic_examples": "file",
    "processor": "processor_files",
    "qwen": "directory",
    "sam": "file",
    "source_images": "selected_source_images",
    "spacy": "directory",
}
_FROZEN_ENVIRONMENT_PACKAGES = frozenset({
    "accelerate", "einops", "flash-attn", "hydra-core", "iopath",
    "matplotlib", "networkx", "numpy", "pillow", "safetensors",
    "scikit-image", "scikit-learn", "scipy", "sentence-transformers",
    "spacy", "torch", "torchvision", "transformers",
})
_FROZEN_BACKGROUND_RGB = [122, 116, 104]
_ANSWER_SUFFIX = "Answer the option letter directly."
_SUPPORT_TEMPLATE = (
    "Assess only whether the displayed visual observation provides sufficient visible "
    "evidence for the question and every ordered evidence requirement. Do not answer "
    "the question and do not infer missing evidence.\n"
    "Question: {question}\n"
    "Ordered evidence requirements:\n{requirements}\n"
    "Does this observation provide sufficient visible evidence for the complete "
    "requirement set? Answer Yes or No."
)

_P0_FIELDS = frozenset({
    "emitted_answer", "cvsearch_raw", "producing_phase", "node_keys", "support_view",
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
_SELECTION_FIELDS = frozenset({
    "candidate", "no_op_reason", "current_keys", "focus_descriptors",
    "focus_union_xyxy", "positive_outside_area", "normalized_edge_gap",
    "rank_tuple",
})
_DESCRIPTOR_FIELDS = frozenset({
    "canonical_key", "bbox_original", "depth", "render_level", "posterior_score",
    "first_seen_ordinal", "tree_scope", "crop_origin", "source_image_key",
    "source", "renderer_kind", "renderer_identity",
})
_AUDIT_FIELDS = frozenset({
    "p0_anchor", "current_keys", "candidate_keys", "selection_decision",
    "selection_decision_sha256",
    "selection_policy", "composition_policy", "focus_role", "context_role",
    "focus_merge_identity", "context_merge_identity", "composition_identity",
    "batch_result", "current_gap_support", "candidate_gap_support", "uncertainty",
    "uncalibrated_g_expand_proxy", "support_delta", "support_proxy_status",
    "p0_stability", "candidate_stability", "feasible", "normalized_actual_cost",
    "support_contract_status", "coverage_status", "verifier_status", "verifier_avg",
    "verifier_min", "score_margin", "score_status", "replacement_reason",
})
_PRESELECTION_NOOPS = frozenset({
    "expand_invalid_evidence_requirements", "expand_no_evidence_requirements",
    "expand_p0_focus_unavailable", "expand_p0_focus_empty",
    "expand_p0_focus_nonlocal",
})
_SELECTION_NOOPS = frozenset({
    "expand_duplicate_context", "expand_context_adds_no_new_area",
    "expand_no_spatially_eligible_unvisited_context",
})
_ANSWER_RECORD_FIELDS = frozenset(AnswerRecord().to_dict())

_EVALUATOR_DEPENDENCY_PATHS = tuple(
    ROOT / relative for relative in (
        "cvsearch/eval/phase2_oracle.py",
        "cvsearch/eval/phase3_zoom_oracle.py",
        "cvsearch/eval/phase4_expand_oracle.py",
        "cvsearch/evidence_gap/answers.py",
        "cvsearch/evidence_gap/method.py",
        "cvsearch/evidence_gap/provenance.py",
        "cvsearch/evidence_gap/types.py",
        "cvsearch/models/utils.py",
        DISABLED_CONFIG.relative_to(ROOT).as_posix(),
        ENABLED_CONFIG.relative_to(ROOT).as_posix(),
    )
)
_EVALUATOR_DEPENDENCY_HASHES_AT_IMPORT = {
    path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in _EVALUATOR_DEPENDENCY_PATHS
}


def _evaluator_revision() -> dict[str, Any]:
    files = [{"path": path.relative_to(ROOT).as_posix(),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in _EVALUATOR_DEPENDENCY_PATHS]
    current = {item["path"]: item["sha256"] for item in files}
    if current != _EVALUATOR_DEPENDENCY_HASHES_AT_IMPORT:
        raise RuntimeError("P4A evaluator dependency source changed during evaluation")
    return {
        "kind": "source_manifest_sha256",
        "files": files,
        "sha256": hashlib.sha256(_canonical(files)).hexdigest(),
    }


def _validate_config_pair(disabled: Mapping[str, Any], enabled: Mapping[str, Any]) -> None:
    expected_disabled = load_method_config(str(DISABLED_CONFIG))
    expected_enabled = load_method_config(str(ENABLED_CONFIG))
    if dict(disabled) != expected_disabled or dict(enabled) != expected_enabled:
        raise ValueError("run config does not match the frozen Phase-4 P4A config")
    changed = {
        key for key in expected_enabled
        if expected_enabled.get(key) != expected_disabled.get(key)
    }
    if changed != {"config_id", "p4a_expand_enabled", "p4a_expand_admission_mode"}:
        raise ValueError("P4A configs violate the frozen sibling contract")


def _validate_manifest_artifact(
    name: str, value: Any, expected_kind: str,
) -> dict[str, Any]:
    artifact = dict(_mapping(value, f"launch {name} artifact"))
    expected_fields = {"kind", "files", "sha256"}
    if name != "source_images":
        expected_fields.add("path")
    if set(artifact) != expected_fields or artifact.get("kind") != expected_kind:
        raise ValueError(f"launch {name} artifact has an invalid exact schema")
    if name != "source_images" and (
        not isinstance(artifact.get("path"), str) or not artifact["path"]
    ):
        raise ValueError(f"launch {name} artifact path is invalid")
    files = _list(artifact.get("files"), f"launch {name} artifact files")
    if not files:
        raise ValueError(f"launch {name} artifact files must not be empty")
    for index, value in enumerate(files):
        entry = _mapping(value, f"launch {name} artifact file[{index}]")
        if not {"path", "sha256", "size"}.issubset(entry) or not set(entry).issubset({
            "path", "sha256", "size", "resolved_path", "symlink_target",
        }):
            raise ValueError(f"launch {name} artifact file schema is invalid")
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            raise ValueError(f"launch {name} artifact file path is invalid")
        _sha256(entry.get("sha256"), f"launch {name} artifact file digest")
        size = entry.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"launch {name} artifact file size is invalid")
        symlink_fields = {"resolved_path", "symlink_target"}.intersection(entry)
        if symlink_fields and symlink_fields != {"resolved_path", "symlink_target"}:
            raise ValueError(f"launch {name} artifact symlink identity is incomplete")
    material = {"kind": expected_kind, "files": files}
    if artifact.get("sha256") != canonical_sha256(material):
        raise ValueError(f"launch {name} artifact aggregate digest is invalid")
    return artifact


def _validate_frozen_launch_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(_mapping(manifest, "P4A launch manifest"))
    if set(manifest) != {
        "schema_version", "benchmark", "code", "config", "selected_partition",
        "artifacts", "environment", "hardware",
    } or manifest.get("schema_version") != 1:
        raise ValueError("P4A launch manifest has an invalid exact schema")
    code = dict(_mapping(manifest.get("code"), "P4A launch code"))
    if set(code) != {"revision", "manifest", "manifest_sha256"}:
        raise ValueError("P4A launch code identity has an invalid exact schema")
    code_manifest = dict(_mapping(code.get("manifest"), "P4A code manifest"))
    if set(code_manifest) != {"files"}:
        raise ValueError("P4A code manifest has an invalid exact schema")
    code_files = _list(code_manifest.get("files"), "P4A code manifest files")
    if not code_files or any(
        set(_mapping(item, "P4A code file")) != {"path", "sha256"}
        or not isinstance(item.get("path"), str) or not item.get("path")
        or _sha256(item.get("sha256"), "P4A code file digest") != item.get("sha256")
        for item in code_files
    ):
        raise ValueError("P4A code manifest files are invalid")
    revision = _sha256(code.get("revision"), "P4A launch revision")
    if (
        code.get("manifest_sha256") != canonical_sha256(code_manifest)
        or revision != canonical_sha256(code_files)
        or revision != _FROZEN_INFERENCE_REVISION
    ):
        raise ValueError("P4A launch code does not match the frozen reviewed revision")
    artifacts = dict(_mapping(manifest.get("artifacts"), "P4A launch artifacts"))
    if set(artifacts) != set(_FROZEN_ARTIFACT_KINDS):
        raise ValueError("P4A launch artifacts differ from the exact rerank-disabled set")
    validated_artifacts = {
        name: _validate_manifest_artifact(name, artifacts[name], kind)
        for name, kind in _FROZEN_ARTIFACT_KINDS.items()
    }
    hardware = dict(_mapping(manifest.get("hardware"), "P4A launch hardware"))
    if set(hardware) != {"gpu_uuids"}:
        raise ValueError("P4A launch hardware has an invalid exact schema")
    gpu_uuids = _list(hardware.get("gpu_uuids"), "P4A GPU UUIDs")
    if len(gpu_uuids) != 1 or not isinstance(gpu_uuids[0], str):
        raise ValueError("P4A launch must bind exactly one physical GPU UUID")
    try:
        parsed_uuid = uuid.UUID(gpu_uuids[0])
    except (AttributeError, ValueError) as error:
        raise ValueError("P4A physical GPU UUID is malformed") from error
    if str(parsed_uuid) != gpu_uuids[0].lower():
        raise ValueError("P4A physical GPU UUID is not canonical")
    environment = dict(_mapping(manifest.get("environment"), "P4A launch environment"))
    if set(environment) != {
        "python", "python_implementation", "platform", "executable", "packages",
        "torch_cuda", "cudnn", "cuda_visible_devices",
    }:
        raise ValueError("P4A launch environment has an invalid exact schema")
    for name in ("python", "python_implementation", "platform", "executable", "torch_cuda"):
        if not isinstance(environment.get(name), str) or not environment[name]:
            raise ValueError(f"P4A launch environment {name} is invalid")
    if isinstance(environment.get("cudnn"), bool) or not isinstance(environment.get("cudnn"), int):
        raise ValueError("P4A launch cuDNN version is invalid")
    if not isinstance(environment.get("cuda_visible_devices"), str):
        raise ValueError("P4A launch CUDA_VISIBLE_DEVICES is invalid")
    packages = dict(_mapping(environment.get("packages"), "P4A launch packages"))
    if set(packages) != _FROZEN_ENVIRONMENT_PACKAGES or any(
        value is not None and (not isinstance(value, str) or not value)
        for value in packages.values()
    ):
        raise ValueError("P4A launch package manifest is invalid")
    config = dict(_mapping(manifest.get("config"), "P4A launch config"))
    if set(config) != {"loaded", "loaded_sha256", "source"}:
        raise ValueError("P4A launch config has an invalid exact schema")
    loaded = dict(_mapping(config.get("loaded"), "P4A loaded config"))
    if config.get("loaded_sha256") != canonical_sha256(loaded):
        raise ValueError("P4A loaded config digest is invalid")
    config_source = _validate_manifest_artifact("config_source", config.get("source"), "file")
    config_files = config_source["files"]
    config_path = Path(config_source["path"])
    if (
        len(config_files) != 1 or config_path.is_symlink() or not config_path.is_file()
        or config_path.name != config_files[0]["path"]
        or config_path.stat().st_size != config_files[0]["size"]
        or _strict_sha256_file(config_path) != config_files[0]["sha256"]
    ):
        raise ValueError("P4A frozen config source bytes are invalid")
    try:
        source_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("P4A frozen config source is not UTF-8 JSON") from error
    if source_config != loaded:
        raise ValueError("P4A loaded config differs from its frozen JSON source")
    partition = dict(_mapping(manifest.get("selected_partition"), "P4A partition"))
    if set(partition) != {
        "ordinals", "rows_sha256", "rows", "split", "split_seed",
        "num_chunks", "chunk_idx",
    }:
        raise ValueError("P4A selected partition has an invalid exact schema")
    _sha256(partition.get("rows_sha256"), "P4A partition row digest")
    ordinals = _list(partition.get("ordinals"), "P4A partition ordinals")
    if (
        not ordinals or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in ordinals
        )
        or ordinals != sorted(set(ordinals))
        or isinstance(partition.get("rows"), bool)
        or partition.get("rows") != len(ordinals)
        or not isinstance(partition.get("split"), str) or not partition["split"]
        or isinstance(partition.get("split_seed"), bool)
        or not isinstance(partition.get("split_seed"), int)
        or partition.get("num_chunks") != 1 or partition.get("chunk_idx") != 0
    ):
        raise ValueError("P4A selected partition values are invalid")
    return dict(manifest, artifacts=validated_artifacts)


def validate_launch_pair(
    disabled_rows: Sequence[Mapping[str, Any]],
    enabled_rows: Sequence[Mapping[str, Any]],
    disabled_manifest: Mapping[str, Any],
    enabled_manifest: Mapping[str, Any],
) -> dict[str, str]:
    """Bind disabled/P4A artifacts to one exact non-config launch identity."""
    disabled_manifest = _validate_frozen_launch_manifest(disabled_manifest)
    enabled_manifest = _validate_frozen_launch_manifest(enabled_manifest)
    disabled_config = _mapping(disabled_manifest.get("config"), "disabled manifest config")
    enabled_config = _mapping(enabled_manifest.get("config"), "enabled manifest config")
    _validate_config_pair(
        _mapping(disabled_config.get("loaded"), "disabled loaded config"),
        _mapping(enabled_config.get("loaded"), "enabled loaded config"),
    )
    disabled_model = _launch_model_contract(disabled_manifest)
    enabled_model = _launch_model_contract(enabled_manifest)
    if disabled_model != enabled_model:
        raise ValueError("disabled and P4A launch model contracts differ")
    disabled_identity = dict(disabled_manifest)
    enabled_identity = dict(enabled_manifest)
    disabled_identity.pop("config")
    enabled_identity.pop("config")
    if disabled_identity != enabled_identity:
        raise ValueError("disabled and P4A launch manifests differ in non-config identity")
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
        raise ValueError("P4A JSONL fingerprint differs from launch manifest")
    return {"disabled": disabled_fingerprint, "enabled": enabled_fingerprint}


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
    for field in ("global_scope_required", "fallback_used"):
        if not isinstance(plan.get(field), bool):
            raise TypeError(f"query plan {field} must be boolean")
    try:
        return sanitize_evidence_requirements(
            _list(plan.get("evidence_items"), "query plan evidence items")
        ), False
    except (TypeError, ValueError):
        return (), True


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


def _validate_answer_record(
    benchmark: str, options: Sequence[str], output: Any, value: Any,
) -> dict[str, Any]:
    record = dict(_mapping(value, "P0 answer record"))
    if frozenset(record) != _ANSWER_RECORD_FIELDS:
        raise ValueError("P0 answer record has an invalid exact schema")
    if benchmark in _HR_BENCHMARKS:
        raw = _list(record.get("raw_outputs"), "HR P0 raw outputs")
        if len(raw) != 4 or not all(isinstance(item, str) for item in raw):
            raise ValueError("HR P0 raw outputs must contain four strings")
        expected = aggregate_hr_answers(list(options), raw)
        expected.output = copy.deepcopy(output)
        expected.selected_from = "cvsearch_anchor"
    else:
        raw_rows = _list(record.get("raw_outputs"), "V* P0 raw loss rows")
        loss_rows = [_list(row, "V* P0 raw loss row") for row in raw_rows]
        if len(loss_rows) != 1 or len(loss_rows[0]) != len(options):
            raise ValueError("V* P0 must contain one loss row over the exact options")
        expected = aggregate_vstar_losses(loss_rows)
        if record.get("selected_from") not in {"root", "search"}:
            raise ValueError("V* P0 answer source is not canonical")
        expected.selected_from = record["selected_from"]
    payload = expected.to_dict()
    if payload != record or record.get("output") != output:
        raise ValueError("P0 answer record is not phase-specifically canonical")
    return payload


def _validate_common_trace(
    trace: Mapping[str, Any], *, config: Mapping[str, Any], enabled: bool,
    benchmark: str, options: Sequence[str], output: Any, context: str,
) -> tuple[dict[str, int], float, Mapping[str, Any], Mapping[str, Any] | None]:
    if frozenset(trace) != _TRACE_FIELDS:
        raise ValueError(f"{context} MethodTrace has an invalid exact schema")
    if trace.get("effective_config") != config or trace.get("config_id") != config["config_id"]:
        raise ValueError(f"{context} config identity mismatch")
    if trace.get("termination") != "FORCED_RETURN" or trace.get("budget_interrupted") is not False:
        raise ValueError(f"{context} must terminate by uninterrupted forced return")
    if trace.get("pixel_accounting") != config["pixel_accounting"]:
        raise ValueError(f"{context} pixel accounting mismatch")
    elapsed = _finite(trace.get("elapsed_seconds"), f"{context}.elapsed", minimum=0.0)
    budget = _budget(trace.get("budget"), f"{context}.budget")
    final_answer = _validate_answer_record(
        benchmark, options, output, trace.get("final_answer"),
    )
    anchor_answer = _validate_answer_record(
        benchmark, options, output, trace.get("anchor_answer"),
    )
    if final_answer != anchor_answer or _answer_output(final_answer, context) != output:
        raise ValueError(f"{context} final/anchor answer differs from emitted output")
    steps = _list(trace.get("steps"), f"{context}.steps")
    if not all(isinstance(step, Mapping) for step in steps):
        raise ValueError(f"{context}.steps must contain objects")
    if enabled:
        if len(steps) != 2 or [step.get("action") for step in steps] != ["EXPAND", "FORCED_RETURN"]:
            raise ValueError("P4A trace must contain exactly EXPAND then FORCED_RETURN")
        if any(trace.get(name) is not None for name in (
            "anchor_state_score", "selected_state_score", "replacement_margin",
        )):
            raise ValueError("P4A selector scores and replacement must remain disabled")
        action_step = steps[0]
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
        action_step = None
    return budget, elapsed, final_answer, action_step


def _observation_sha256(image: Image.Image) -> str:
    payload = image.mode.encode("utf-8") + b"\x00"
    payload += f"{image.width}x{image.height}".encode("ascii") + b"\x00"
    payload += image.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _source_identity(image: Image.Image) -> dict[str, Any]:
    return {
        "mode": image.mode,
        "size": [image.width, image.height],
        "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }


def _strict_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source_image(
    row: Mapping[str, Any], manifest: Mapping[str, Any],
) -> tuple[Image.Image, dict[str, str]]:
    input_image = row.get("input_image")
    if not isinstance(input_image, str) or not input_image:
        raise ValueError("row input_image must be a nonempty path string")
    artifacts = _mapping(manifest.get("artifacts"), "launch artifacts")
    source_artifact = _mapping(artifacts.get("source_images"), "source image artifact")
    files = _list(source_artifact.get("files"), "source image artifact files")
    normalized = input_image.replace("\\", "/")
    candidates = []
    for item in files:
        entry = _mapping(item, "source image file entry")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("source image artifact path is invalid")
        path_normalized = raw_path.replace("\\", "/")
        if (
            path_normalized == normalized
            or path_normalized.endswith("/" + normalized.lstrip("/"))
        ):
            candidates.append(entry)
    if len(candidates) != 1:
        raise ValueError("row input image does not resolve uniquely in launch artifacts")
    entry = candidates[0]
    path = Path(entry["path"])
    expected_file_sha = _sha256(entry.get("sha256"), "source image artifact digest")
    if not path.is_file() or _strict_sha256_file(path) != expected_file_sha:
        raise ValueError("source image artifact bytes differ from the launch manifest")
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    return image, {
        "input_image": input_image,
        "artifact_path": str(path),
        "artifact_file_sha256": expected_file_sha,
        "rgb_pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }


def _canonical_source_key(source: Mapping[str, Any]) -> str:
    return json.dumps(
        source, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _validate_descriptor(
    value: Any, *, source_key: str, source_size: Sequence[int], context: str,
) -> dict[str, Any]:
    descriptor = dict(_mapping(value, context))
    if frozenset(descriptor) != _DESCRIPTOR_FIELDS:
        raise ValueError(f"{context} descriptor has an invalid exact schema")
    bbox = _list(descriptor.get("bbox_original"), f"{context}.bbox_original")
    if len(bbox) != 4:
        raise ValueError(f"{context} bbox must contain four coordinates")
    numbers = [_finite(item, f"{context}.bbox[{index}]") for index, item in enumerate(bbox)]
    x, y, width, height = numbers
    if (
        x < 0 or y < 0 or width <= 0 or height <= 0
        or x + width > source_size[0] or y + height > source_size[1]
    ):
        raise ValueError(f"{context} bbox is outside the RGB source")
    for name in ("depth", "render_level", "first_seen_ordinal"):
        item = descriptor.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{context}.{name} must be a non-negative integer")
    posterior = descriptor.get("posterior_score")
    if posterior is not None:
        _finite(posterior, f"{context}.posterior_score")
    crop_origin = _list(descriptor.get("crop_origin"), f"{context}.crop_origin")
    if len(crop_origin) != 2:
        raise ValueError(f"{context} crop origin must contain two coordinates")
    for index, item in enumerate(crop_origin):
        _finite(item, f"{context}.crop_origin[{index}]")
    if descriptor.get("tree_scope") not in {"main", "cropped"}:
        raise ValueError(f"{context} tree scope is invalid")
    if descriptor.get("source") not in {None, "fast", "fine", "fine_fallback"}:
        raise ValueError(f"{context} is not a local native descriptor")
    renderer_kind = "fast" if descriptor["source"] == "fast" else "fine"
    if descriptor.get("renderer_kind") != renderer_kind:
        raise ValueError(f"{context} renderer kind is not native")
    if descriptor.get("source_image_key") != source_key:
        raise ValueError(f"{context} source identity differs from trusted RGB")
    expected_key = json.dumps(
        {
            "bbox": bbox,
            "depth": descriptor["depth"],
            "render_level": descriptor["render_level"],
        },
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    expected_renderer = json.dumps(
        {
            "source_image_key": source_key,
            "renderer_kind": renderer_kind,
            "bbox": bbox,
            "render_level": descriptor["render_level"],
        },
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    if descriptor.get("canonical_key") != expected_key:
        raise ValueError(f"{context} canonical key differs from geometry")
    if descriptor.get("renderer_identity") != expected_renderer:
        raise ValueError(f"{context} renderer identity differs from geometry")
    return descriptor


def _rectangle(descriptor: Mapping[str, Any]) -> list[float]:
    x, y, width, height = (float(value) for value in descriptor["bbox_original"])
    return [x, y, x + width, y + height]


def _intersection_union_area(
    rectangle: Sequence[float], focus_rectangles: Sequence[Sequence[float]],
) -> float:
    intersections = []
    for focus in focus_rectangles:
        overlap = [
            max(rectangle[0], focus[0]), max(rectangle[1], focus[1]),
            min(rectangle[2], focus[2]), min(rectangle[3], focus[3]),
        ]
        if overlap[0] < overlap[2] and overlap[1] < overlap[3]:
            intersections.append(overlap)
    if not intersections:
        return 0.0
    xs = sorted({float(value) for item in intersections for value in item[::2]})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted(
            (float(item[1]), float(item[3])) for item in intersections
            if float(item[0]) < right and left < float(item[2])
        )
        if not intervals:
            continue
        start, end = intervals[0]
        covered = 0.0
        for top, bottom in intervals[1:]:
            if top > end:
                covered += end - start
                start, end = top, bottom
            else:
                end = max(end, bottom)
        area += (right - left) * (covered + end - start)
    return area


def _validate_selection_decision(
    value: Any, *, source: Mapping[str, Any], expected_current_keys: Sequence[str],
) -> dict[str, Any]:
    selection = dict(_mapping(value, "EXPAND selection decision"))
    if frozenset(selection) != _SELECTION_FIELDS:
        raise ValueError("EXPAND selection decision has an invalid exact schema")
    current_keys = _list(selection.get("current_keys"), "EXPAND selection current keys")
    if current_keys != list(expected_current_keys):
        raise ValueError("EXPAND selection current keys differ from the P0 anchor")
    source_size = _list(source.get("size"), "trusted RGB source size")
    source_key = _canonical_source_key(source)
    focus_values = _list(selection.get("focus_descriptors"), "EXPAND selection focus")
    focus = [
        _validate_descriptor(
            item, source_key=source_key, source_size=source_size,
            context=f"EXPAND selection focus[{index}]",
        )
        for index, item in enumerate(focus_values)
    ]
    if [item["canonical_key"] for item in focus] != current_keys:
        raise ValueError("EXPAND selection focus ordering differs from current keys")
    if len({item["renderer_identity"] for item in focus}) != len(focus):
        raise ValueError("EXPAND selection focus renderer identities are not unique")
    candidate_value = selection.get("candidate")
    no_op_reason = selection.get("no_op_reason")
    measurements = (
        selection.get("focus_union_xyxy"), selection.get("positive_outside_area"),
        selection.get("normalized_edge_gap"), selection.get("rank_tuple"),
    )
    if candidate_value is None:
        if no_op_reason not in _SELECTION_NOOPS or any(item is not None for item in measurements):
            raise ValueError("EXPAND selection no-op is not canonical")
        if not focus:
            raise ValueError("EXPAND selection no-op lost its exact focus descriptors")
        return selection
    if no_op_reason is not None or not focus or any(item is None for item in measurements):
        raise ValueError("successful EXPAND selection is incomplete")
    candidate = _validate_descriptor(
        candidate_value, source_key=source_key, source_size=source_size,
        context="EXPAND selection context",
    )
    if (
        candidate["canonical_key"] in current_keys
        or candidate["renderer_identity"] in {item["renderer_identity"] for item in focus}
    ):
        raise ValueError("EXPAND selection context duplicates the focus")
    rectangles = [_rectangle(item) for item in focus]
    expected_union = [
        min(item[0] for item in rectangles), min(item[1] for item in rectangles),
        max(item[2] for item in rectangles), max(item[3] for item in rectangles),
    ]
    context_rectangle = _rectangle(candidate)
    context_area = (
        (context_rectangle[2] - context_rectangle[0])
        * (context_rectangle[3] - context_rectangle[1])
    )
    outside = context_area - _intersection_union_area(context_rectangle, rectangles)
    contained = any(
        context_rectangle[0] >= item[0] and context_rectangle[1] >= item[1]
        and context_rectangle[2] <= item[2] and context_rectangle[3] <= item[3]
        for item in rectangles
    )
    contains_all = all(
        context_rectangle[0] <= item[0] and context_rectangle[1] <= item[1]
        and context_rectangle[2] >= item[2] and context_rectangle[3] >= item[3]
        for item in rectangles
    )
    edge_gap = min(
        math.hypot(
            max(item[0] - context_rectangle[2], context_rectangle[0] - item[2], 0.0),
            max(item[1] - context_rectangle[3], context_rectangle[1] - item[3], 0.0),
        )
        for item in rectangles
    ) / math.hypot(*source_size)
    expected_rank = [
        edge_gap, candidate["posterior_score"] is None,
        0.0 if candidate["posterior_score"] is None else -candidate["posterior_score"],
        candidate["first_seen_ordinal"], candidate["canonical_key"],
    ]
    rank = _list(selection.get("rank_tuple"), "EXPAND selection rank tuple")
    if (
        len(rank) != 5
        or isinstance(rank[0], bool) or not isinstance(rank[0], (int, float))
        or not math.isfinite(float(rank[0]))
        or type(rank[1]) is not bool
        or isinstance(rank[2], bool) or not isinstance(rank[2], (int, float))
        or not math.isfinite(float(rank[2]))
        or isinstance(rank[3], bool) or not isinstance(rank[3], int)
        or rank[3] < 0 or not isinstance(rank[4], str) or not rank[4]
    ):
        raise ValueError("EXPAND selection rank tuple types are invalid")
    if (
        selection["focus_union_xyxy"] != expected_union
        or abs(_finite(selection["positive_outside_area"], "EXPAND outside area") - outside) > 1e-9
        or abs(_finite(selection["normalized_edge_gap"], "EXPAND edge gap") - edge_gap) > 1e-12
        or rank != expected_rank
        or outside <= 0.0 or contained or contains_all
    ):
        raise ValueError("EXPAND selection spatial geometry/rank is not canonical")
    return selection


def _native_crop(
    descriptor: Mapping[str, Any], source_size: Sequence[int],
) -> list[int]:
    bbox = descriptor["bbox_original"]
    object_width = math.ceil(float(bbox[2]))
    object_height = math.ceil(float(bbox[3]))
    center_x = int(float(bbox[0]) + float(bbox[2]) / 2)
    center_y = int(float(bbox[1]) + float(bbox[3]) / 2)
    patch_size = (
        _FROZEN_BASE_VIEW_SIZE // 3
        if descriptor["source"] == "fast" else _FROZEN_BASE_VIEW_SIZE
    )
    patch_width = max(object_width, patch_size)
    patch_height = max(object_height, patch_size)
    if descriptor["source"] != "fast":
        patch_width = int(patch_width * _FROZEN_PATCH_SCALE)
        patch_height = int(patch_height * _FROZEN_PATCH_SCALE)
    left = max(0, center_x - patch_width // 2)
    top = max(0, center_y - patch_height // 2)
    return [
        left, top, min(left + patch_width, source_size[0]),
        min(top + patch_height, source_size[1]),
    ]


def _merge_identity(crops: Sequence[Sequence[int]]) -> dict[str, Any]:
    per_descriptor = [list(item) for item in crops]
    merged = [list(item) for item in merge_bbox_list(copy.deepcopy(per_descriptor), threshold=0)]
    union = union_all_bboxes(merged)
    if union is None:
        raise ValueError("EXPAND merge identity requires at least one crop")
    material = {
        "per_descriptor_crop_xyxy": per_descriptor,
        "merged_crop_xyxy": merged,
        "union_crop_xyxy": list(union),
    }
    return dict(
        material,
        identity_sha256=hashlib.sha256(_canonical(material)).hexdigest(),
    )


def _render_native_panel(
    source: Image.Image, descriptors: Sequence[Mapping[str, Any]],
    background: Sequence[int],
) -> Image.Image:
    square, left, top = expand2square(source, tuple(background))
    crops = [_native_crop(item, source.size) for item in descriptors]
    merged = merge_bbox_list(copy.deepcopy(crops), threshold=0)
    color_boxes = []
    for crop in merged:
        square_box = [crop[0] + left, crop[1] + top, crop[2] + left, crop[3] + top]
        color_boxes.append(visualize_bbox_and_arrow(
            square, square_box, "red", square.width // 120, xyxy=True,
        ))
    union = union_all_bboxes(color_boxes)
    if union is None:
        raise ValueError("EXPAND native renderer has no union crop")
    cropped = square.crop(union)
    ratio = max(
        1.0,
        min(_FROZEN_SCALE_SIZE / cropped.width, _FROZEN_SCALE_SIZE / cropped.height),
    )
    return cropped.resize((int(cropped.width * ratio), int(cropped.height * ratio)))


@lru_cache(maxsize=256)
def _support_prompt_sha256(
    checkpoint: str, q0: str, requirement_material: tuple[tuple[str, str], ...],
) -> str:
    """Recompute the exact Qwen chat prompt from trusted q0/requirements."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint)
    lines = "\n".join(
        f"{index + 1}. [{requirement_id}] {text}"
        for index, (requirement_id, text) in enumerate(requirement_material)
    )
    support_text = _SUPPORT_TEMPLATE.format(question=q0, requirements=lines)
    message = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": support_text},
        ],
    }]
    prompt = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Qwen processor did not produce one support prompt")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _composition_material(
    source: Image.Image, focus: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    focus_panel = _render_native_panel(source, focus, _FROZEN_BACKGROUND_RGB)
    context_panel = _render_native_panel(source, [context], _FROZEN_BACKGROUND_RGB)
    separator = 8
    canvas_size = (
        max(focus_panel.width, context_panel.width),
        focus_panel.height + separator + context_panel.height,
    )
    context_offset = (0, focus_panel.height + separator)
    background = tuple(_FROZEN_BACKGROUND_RGB)
    blank_panel = Image.new("RGB", context_panel.size, background)
    current = Image.new("RGB", canvas_size, background)
    candidate = Image.new("RGB", canvas_size, background)
    current.paste(focus_panel, (0, 0))
    candidate.paste(focus_panel, (0, 0))
    candidate.paste(context_panel, context_offset)
    composition = {
        "policy": _COMPOSITION_POLICY,
        "background_rgb": list(background),
        "separator_width": separator,
        "canvas_size": list(canvas_size),
        "focus_offset_xy": [0, 0],
        "context_offset_xy": list(context_offset),
        "focus_size": list(focus_panel.size),
        "context_size": list(context_panel.size),
        "focus_panel_sha256": _observation_sha256(focus_panel),
        "context_panel_sha256": _observation_sha256(context_panel),
        "blank_panel_sha256": _observation_sha256(blank_panel),
        "focus_current_rectangle_sha256": _observation_sha256(
            current.crop((0, 0, focus_panel.width, focus_panel.height))
        ),
        "focus_candidate_rectangle_sha256": _observation_sha256(
            candidate.crop((0, 0, focus_panel.width, focus_panel.height))
        ),
        "current_context_slot_sha256": _observation_sha256(current.crop((
            context_offset[0], context_offset[1],
            context_offset[0] + context_panel.width,
            context_offset[1] + context_panel.height,
        ))),
        "candidate_context_slot_sha256": _observation_sha256(candidate.crop((
            context_offset[0], context_offset[1],
            context_offset[0] + context_panel.width,
            context_offset[1] + context_panel.height,
        ))),
        "current_composite_sha256": _observation_sha256(current),
        "candidate_composite_sha256": _observation_sha256(candidate),
        "pixel_delta_region_xyxy": [
            0, context_offset[1], context_panel.width,
            context_offset[1] + context_panel.height,
        ],
    }
    composition["identity_sha256"] = hashlib.sha256(_canonical(composition)).hexdigest()
    current_meta = {
        "rendered_mode": current.mode,
        "rendered_size": list(current.size),
        "view_sha256": _observation_sha256(current),
    }
    candidate_meta = {
        "rendered_mode": candidate.mode,
        "rendered_size": list(candidate.size),
        "view_sha256": _observation_sha256(candidate),
    }
    return composition, current_meta, candidate_meta


def _expected_answer_identity(
    answer_type: str, q0: str, options: Sequence[str],
) -> tuple[str, list[str], str]:
    options_payload = list(options)
    options_hash = hashlib.sha256(_canonical(options_payload)).hexdigest()
    q0_hash = hashlib.sha256(q0.encode("utf-8")).hexdigest()
    if answer_type == "option_list":
        prompt_hashes = [
            hashlib.sha256((q0 + "\n" + option + _ANSWER_SUFFIX).encode("utf-8")).hexdigest()
            for option in options_payload
        ]
    else:
        prompt_hashes = [hashlib.sha256(_canonical({
            "q0": q0, "options": options_payload,
        })).hexdigest()]
    identity = {
        "answer_type": answer_type,
        "q0_sha256": q0_hash,
        "options_sha256": options_hash,
        "answer_prompt_sha256": prompt_hashes,
    }
    return options_hash, prompt_hashes, hashlib.sha256(_canonical(identity)).hexdigest()


def _validate_expand_plan(
    plan_value: Mapping[str, Any], *, source_image: Image.Image,
    row: Mapping[str, Any], requirements: tuple[EvidenceRequirement, ...],
    selection: Mapping[str, Any], config: Mapping[str, Any],
    model_contract: Mapping[str, Any], allow_render_noop: bool,
) -> tuple[dict[str, Any], str]:
    plan, plan_hash, is_full = _validate_batch_plan(
        plan_value, allow_expand_noop=allow_render_noop,
    )
    if not is_full or plan.get("batch_kind") != "p4a_post_anchor_expand_context":
        raise ValueError("EXPAND charged/render plan must use the exact full P4A schema")
    source = _source_identity(source_image)
    if plan.get("source_identity") != source:
        raise ValueError("EXPAND plan source identity differs from trusted launch RGB")
    if (
        plan.get("source_width") != source_image.width
        or plan.get("source_height") != source_image.height
        or plan.get("accounted_source_area") != source_image.width * source_image.height
        or plan.get("pixels_per_logical_forward") != source_image.width * source_image.height
    ):
        raise ValueError("EXPAND source geometry/accounting differs from trusted RGB")
    patch_scale = plan.get("patch_scale")
    if (
        plan.get("selection_policy") != _SELECTION_POLICY
        or plan.get("selection_policy") != config["p4a_expand_selection_policy"]
        or plan.get("composition_policy") != _COMPOSITION_POLICY
        or plan.get("base_view_size") != _FROZEN_BASE_VIEW_SIZE
        or type(patch_scale) is not float
        or not math.isfinite(patch_scale)
        or patch_scale != _FROZEN_PATCH_SCALE
    ):
        raise ValueError("EXPAND frozen selection/render policy differs")
    question = row.get("question")
    options = row.get("options")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("trusted q0 must be a nonempty string")
    if not isinstance(options, list) or not options or not all(
        isinstance(option, str) and option for option in options
    ):
        raise ValueError("trusted ordered options must be nonempty strings")
    if plan.get("q0") != question or plan.get("options") != options:
        raise ValueError("EXPAND q0/options differ from trusted paired inputs")
    q0_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
    options_hash, answer_hashes, answer_identity = _expected_answer_identity(
        row["answer_type"], question, options,
    )
    if (
        plan.get("q0_sha256") != q0_hash
        or plan.get("options_sha256") != options_hash
        or plan.get("answer_prompt_sha256") != answer_hashes
        or plan.get("answer_call_identity_sha256") != answer_identity
        or plan.get("answer_type") != row.get("answer_type")
        or plan.get("candidate_option_count") != len(options)
    ):
        raise ValueError("EXPAND answer-call identity is not trusted-input-derived")
    expected_order = [item.requirement_id for item in requirements]
    if (
        plan.get("requirement_order") != expected_order
        or plan.get("requirement_set_id")
        != EvidenceSupportResult.requirement_set_id_for(requirements)
    ):
        raise ValueError("EXPAND requirements differ from the trusted query plan")
    requirement_material = tuple((item.requirement_id, item.text) for item in requirements)
    prompt_hash = _support_prompt_sha256(
        model_contract["checkpoint"], question, requirement_material,
    )
    if (
        plan.get("prompt_version") != config["evidence_support_prompt_version"]
        or plan.get("processor_mode") != config["evidence_support_processor_mode"]
        or plan.get("prompt_template_sha256")
        != config["evidence_support_prompt_template_sha256"]
        or plan.get("p_yes_transform")
        != config["evidence_support_probability_transform"]
        or plan.get("current_prompt_sha256") != prompt_hash
        or plan.get("candidate_prompt_sha256") != prompt_hash
        or plan.get("checkpoint") != model_contract["checkpoint"]
        or plan.get("processor_fingerprint") != model_contract["processor_fingerprint"]
    ):
        raise ValueError("EXPAND support/model provenance is not launch-derived")

    focus = selection["focus_descriptors"]
    context = selection["candidate"]
    if context is None:
        raise ValueError("EXPAND full plan cannot follow a no-candidate selection")
    current_keys = [item["canonical_key"] for item in focus]
    candidate_keys = current_keys + [context["canonical_key"]]
    if plan.get("current_keys") != current_keys or plan.get("candidate_keys") != candidate_keys:
        raise ValueError("EXPAND plan keys differ from the frozen selection")
    focus_role = {
        "role": "focus",
        "canonical_keys": current_keys,
        "renderer_identities": [item["renderer_identity"] for item in focus],
        "descriptors": copy.deepcopy(focus),
        "native_crops_xyxy": [_native_crop(item, source_image.size) for item in focus],
    }
    rectangle = _rectangle(context)
    context_role = {
        "role": "context",
        "canonical_key": context["canonical_key"],
        "renderer_identity": context["renderer_identity"],
        "descriptor": copy.deepcopy(context),
        "native_crop_xyxy": _native_crop(context, source_image.size),
        "candidate_bbox_xyxy": rectangle,
        "focus_union_xyxy": selection["focus_union_xyxy"],
        "positive_outside_area": selection["positive_outside_area"],
        "contained_by_focus_descriptor": False,
        "contains_complete_focus_union": False,
        "normalized_edge_gap": selection["normalized_edge_gap"],
        "rank_tuple": selection["rank_tuple"],
    }
    if (
        _canonical(plan.get("focus_role")) != _canonical(focus_role)
        or _canonical(plan.get("context_role")) != _canonical(context_role)
    ):
        raise ValueError("EXPAND role/geometry material differs from frozen selection")
    if (
        plan.get("focus_merge_identity") != _merge_identity(focus_role["native_crops_xyxy"])
        or plan.get("context_merge_identity")
        != _merge_identity([context_role["native_crop_xyxy"]])
    ):
        raise ValueError("EXPAND native crop merge/union identity is not recomputable")
    composition, current_pixels, candidate_pixels = _composition_material(
        source_image, focus, context,
    )
    if plan.get("composition_identity") != composition:
        raise ValueError("EXPAND composition is not exactly reproducible from source RGB")
    current_observation = {
        "canonical_keys": current_keys,
        "renderer_identities": focus_role["renderer_identities"],
        **current_pixels,
        "descriptors": copy.deepcopy(focus),
    }
    candidate_observation = {
        "canonical_keys": candidate_keys,
        "renderer_identities": (
            focus_role["renderer_identities"] + [context["renderer_identity"]]
        ),
        **candidate_pixels,
        "descriptors": copy.deepcopy(focus) + [copy.deepcopy(context)],
    }
    if (
        plan.get("current_observation") != current_observation
        or plan.get("candidate_observation") != candidate_observation
        or plan.get("candidate_answer_input_sha256")
        != candidate_observation["view_sha256"]
    ):
        raise ValueError("EXPAND rendered observation/answer bindings are not exact")
    return plan, plan_hash


def _canonical_p0_stability(
    benchmark: str, options: list[Any], anchor: Mapping[str, Any],
    step_answer: Mapping[str, Any], audit: Mapping[str, Any],
) -> None:
    stability = _mapping(audit.get("p0_stability"), "EXPAND P0 stability")
    if benchmark in _HR_BENCHMARKS:
        raw = anchor.get("cvsearch_raw")
        if not isinstance(raw, list) or len(raw) != 4 or not all(
            isinstance(item, str) for item in raw
        ):
            raise ValueError("HR P0 raw output must contain four strings")
        expected = aggregate_hr_answers(options, raw)
        expected.output = copy.deepcopy(anchor.get("emitted_answer"))
        expected.selected_from = "cvsearch_raw"
        if stability != expected.to_dict():
            raise ValueError("HR EXPAND P0 stability is not canonical")
    elif stability != step_answer:
        raise ValueError("V* EXPAND P0 stability must equal the exact P0 answer record")
    if _finite(audit.get("uncertainty"), "EXPAND uncertainty") != _finite(
        stability.get("uncertainty"), "EXPAND P0 uncertainty",
    ):
        raise ValueError("EXPAND uncertainty differs from canonical P0 stability")


def _validate_root_descriptor(
    value: Any, *, source: Mapping[str, Any], context: str,
) -> dict[str, Any]:
    descriptor = dict(_mapping(value, context))
    if frozenset(descriptor) != _DESCRIPTOR_FIELDS:
        raise ValueError(f"{context} root descriptor has an invalid exact schema")
    source_key = _canonical_source_key(source)
    bbox = [0, 0, source["size"][0], source["size"][1]]
    expected = {
        "canonical_key": json.dumps(
            {"bbox": bbox, "depth": 0, "render_level": 0},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ),
        "bbox_original": bbox,
        "depth": 0,
        "render_level": 0,
        "posterior_score": None,
        "first_seen_ordinal": 0,
        "tree_scope": "main",
        "crop_origin": [0, 0],
        "source_image_key": source_key,
        "source": "global",
        "renderer_kind": "root",
        "renderer_identity": json.dumps(
            {"renderer_kind": "root", "source_image_key": source_key},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ),
    }
    if descriptor != expected:
        raise ValueError(f"{context} root descriptor is not canonical")
    return descriptor


def _validate_p0_anchor(
    benchmark: str, row: Mapping[str, Any], audit: Mapping[str, Any],
    step_answer: Mapping[str, Any], selection: Mapping[str, Any] | None,
    no_op_reason: Any, source: Mapping[str, Any],
) -> Mapping[str, Any]:
    anchor = _mapping(audit.get("p0_anchor"), "EXPAND P0 anchor")
    if frozenset(anchor) != _P0_FIELDS or anchor.get("emitted_answer") != row.get("output"):
        raise ValueError("EXPAND P0 anchor schema/output is invalid")
    keys = _list(anchor.get("node_keys"), "EXPAND P0 node keys")
    if (
        any(not isinstance(key, str) or not key for key in keys)
        or len(keys) != len(set(keys))
        or audit.get("current_keys") != keys
    ):
        raise ValueError("EXPAND P0/current keys are invalid")
    support_value = anchor.get("support_view")
    support = None if support_value is None else _list(
        support_value, "EXPAND P0 support view",
    )
    if support is not None and (
        len(support) != len(keys)
        or any(not isinstance(item, Mapping) for item in support)
        or [item.get("canonical_key") for item in support] != keys
    ):
        raise ValueError("EXPAND P0 support descriptors differ from node keys")
    if support is not None:
        validated_support = []
        for index, descriptor in enumerate(support):
            if descriptor.get("source") == "global" or descriptor.get("renderer_kind") == "root":
                validated_support.append(_validate_root_descriptor(
                    descriptor, source=source, context=f"EXPAND P0 support[{index}]",
                ))
            else:
                validated_support.append(_validate_descriptor(
                    descriptor, source_key=_canonical_source_key(source),
                    source_size=source["size"], context=f"EXPAND P0 support[{index}]",
                ))
        if (
            [item["canonical_key"] for item in validated_support] != keys
            or len({item["renderer_identity"] for item in validated_support})
            != len(validated_support)
        ):
            raise ValueError("EXPAND P0 support descriptor identities are not exact and unique")
    options = _list(row.get("options"), "options")
    output = row.get("output")
    if benchmark in _HR_BENCHMARKS:
        if (
            anchor.get("producing_phase") != "cvsearch_raw"
            or anchor.get("cvsearch_raw") != output
            or not isinstance(output, list) or len(output) != 4
        ):
            raise ValueError("HR EXPAND P0 must bind exact cvsearch_raw")
    else:
        raw = anchor.get("cvsearch_raw")
        if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw < len(options):
            raise ValueError("V* cvsearch_raw is not a valid option index")
        if anchor.get("producing_phase") == "root":
            if keys or support != []:
                raise ValueError("V* root P0 must bind the empty root view")
        elif anchor.get("producing_phase") == "search":
            if raw != output:
                raise ValueError("V* search P0 must bind exact cvsearch_raw")
        else:
            raise ValueError("V* P0 phase must be root or search")
        if step_answer.get("selected_from") != anchor.get("producing_phase"):
            raise ValueError("V* P0 answer source differs from its producing phase")
    _canonical_p0_stability(benchmark, options, anchor, step_answer, audit)
    if selection is not None:
        if support != selection["focus_descriptors"]:
            raise ValueError("EXPAND selection focus differs from exact P0 support")
    elif no_op_reason == "expand_p0_focus_unavailable" and support is not None:
        raise ValueError("unavailable EXPAND focus must be null")
    elif no_op_reason == "expand_p0_focus_empty" and (support != [] or keys):
        raise ValueError("empty EXPAND focus must bind an empty view")
    elif no_op_reason == "expand_p0_focus_nonlocal" and (
        not support or not any(
            item.get("source") == "global" or item.get("renderer_kind") == "root"
            for item in support
        )
    ):
        raise ValueError("nonlocal EXPAND no-op lacks a nonlocal P0 descriptor")
    return anchor


def _validate_candidate_answer(
    benchmark: str, row: Mapping[str, Any], value: Any,
) -> tuple[Any, Mapping[str, Any]]:
    options = _list(row.get("options"), "candidate options")
    if benchmark in _HR_BENCHMARKS:
        candidate = _list(value, "HR raw EXPAND candidate")
        if len(candidate) != 4 or not all(isinstance(item, str) for item in candidate):
            raise ValueError("HR raw EXPAND candidate must contain four strings")
        return candidate, aggregate_hr_answers(options, candidate).to_dict()
    candidate = _mapping(value, "V* raw EXPAND candidate")
    if set(candidate) != {"winner", "losses"}:
        raise ValueError("V* raw EXPAND candidate has an invalid exact schema")
    winner = candidate.get("winner")
    if isinstance(winner, bool) or not isinstance(winner, int) or not 0 <= winner < len(options):
        raise ValueError("V* raw EXPAND winner is outside the option range")
    losses = _list(candidate.get("losses"), "V* EXPAND losses")
    if len(losses) != len(options):
        raise ValueError("V* EXPAND losses do not match ordered options")
    finite_losses = [
        _finite(loss, f"V* EXPAND loss[{index}]") for index, loss in enumerate(losses)
    ]
    expected_winner = min(range(len(finite_losses)), key=finite_losses.__getitem__)
    if winner != expected_winner:
        raise ValueError("V* EXPAND winner is not the finite-loss argmin")
    canonical_candidate = {"winner": winner, "losses": finite_losses}
    return canonical_candidate, aggregate_vstar_losses([finite_losses]).to_dict()


def _validate_expand_step(
    benchmark: str, row: Mapping[str, Any], trace: Mapping[str, Any],
    step: Mapping[str, Any], disabled_budget: Mapping[str, int],
    enabled_budget: Mapping[str, int], final_answer: Mapping[str, Any],
    config: Mapping[str, Any], requirements: tuple[EvidenceRequirement, ...],
    invalid_requirements: bool, model_contract: Mapping[str, Any],
    source_image: Image.Image,
) -> tuple[bool, Any, str, float | None, float]:
    if frozenset(step) != _BASE_STEP_FIELDS | {"expand_audit"}:
        raise ValueError("EXPAND StepTrace has an invalid exact schema")
    if (
        step.get("step") != 0 or step.get("action") != "EXPAND"
        or step.get("gap_fallback_used") is not False
        or _finite(step.get("elapsed_seconds"), "EXPAND step elapsed", minimum=0.0) != 0.0
        or _finite(step.get("support_avg"), "EXPAND support_avg") != 0.0
        or _finite(step.get("support_min"), "EXPAND support_min") != 0.0
        or step.get("certified") is not False or step.get("answer") != final_answer
        or _budget(step.get("budget"), "EXPAND step budget") != enabled_budget
    ):
        raise ValueError("EXPAND StepTrace fixed P0 fields are invalid")
    audit = _mapping(step.get("expand_audit"), "EXPAND audit")
    if frozenset(audit) != _AUDIT_FIELDS:
        raise ValueError("EXPAND audit has an invalid exact schema")
    if (
        audit.get("selection_policy") != config["p4a_expand_selection_policy"]
        or audit.get("selection_policy") != _SELECTION_POLICY
        or audit.get("composition_policy") != _COMPOSITION_POLICY
        or audit.get("support_proxy_status") != "audit_only_uncalibrated"
        or audit.get("coverage_status") != "not_observed"
        or audit.get("verifier_status") != "disabled_same_checkpoint_unpromoted"
        or audit.get("verifier_avg") is not None or audit.get("verifier_min") is not None
        or audit.get("score_margin") is not None
        or audit.get("score_status") != "unavailable_missing_verifier_coverage"
    ):
        raise ValueError("EXPAND audit frozen policy/status fields are invalid")
    source = _source_identity(source_image)
    selection_value = audit.get("selection_decision")
    selection = None
    if selection_value is not None:
        selection = _validate_selection_decision(
            selection_value, source=source,
            expected_current_keys=_list(audit.get("current_keys"), "EXPAND current keys"),
        )
    expected_selection_hash = hashlib.sha256(_canonical(selection_value)).hexdigest()
    if audit.get("selection_decision_sha256") != expected_selection_hash:
        raise ValueError("EXPAND selection decision hash is not canonical")
    no_op_reason = step.get("no_op_reason")
    _validate_p0_anchor(
        benchmark, row, audit, final_answer, selection, no_op_reason, source,
    )
    candidate_keys = _list(audit.get("candidate_keys"), "EXPAND candidate keys")
    batch = _batch_result(audit.get("batch_result"))
    for field in (
        "focus_role", "context_role", "focus_merge_identity",
        "context_merge_identity", "composition_identity",
    ):
        projected = None if batch is None else batch.batch_plan.get(field)
        if audit.get(field) != projected:
            raise ValueError(f"EXPAND audit {field} differs from its batch plan")

    if selection is None:
        if (
            no_op_reason not in _PRESELECTION_NOOPS or batch is not None
            or candidate_keys or step.get("focus_key") is not None
            or step.get("feasible_actions") != [] or step.get("gaps") != {}
            or enabled_budget != disabled_budget
        ):
            raise ValueError("EXPAND preselection no-op is not exact")
        if no_op_reason == "expand_invalid_evidence_requirements":
            legal = invalid_requirements
        elif no_op_reason == "expand_no_evidence_requirements":
            legal = not invalid_requirements and not requirements
        else:
            legal = not invalid_requirements and bool(requirements)
        if not legal:
            raise ValueError("EXPAND preselection reason contradicts query requirements")
        status = "no_batch"
        expected_cost = None
    elif selection["candidate"] is None:
        if (
            no_op_reason != selection["no_op_reason"] or batch is not None
            or candidate_keys or step.get("focus_key") is not None
            or step.get("feasible_actions") != [] or step.get("gaps") != {}
            or enabled_budget != disabled_budget
            or invalid_requirements or not requirements
        ):
            raise ValueError("EXPAND selection no-candidate trace is not exact")
        status = "no_spatial_candidate"
        expected_cost = None
    else:
        candidate_key = selection["candidate"]["canonical_key"]
        expected_keys = list(selection["current_keys"]) + [candidate_key]
        if candidate_keys != expected_keys or step.get("focus_key") != candidate_key:
            raise ValueError("EXPAND selected context differs from trace candidate identity")
        if batch is None:
            if (
                no_op_reason != "expand_batch_preflight_failed"
                or step.get("feasible_actions") != [] or step.get("gaps") != {}
                or enabled_budget != disabled_budget
                or invalid_requirements or not requirements
            ):
                raise ValueError("EXPAND batch preflight failure is not exact")
            status = "preflight_failed"
            expected_cost = None
        else:
            _validate_expand_plan(
                batch.batch_plan, source_image=source_image, row=row,
                requirements=requirements, selection=selection, config=config,
                model_contract=model_contract,
                allow_render_noop=batch.status == "render_noop",
            )
            before = _budget(batch.to_dict()["ledger_before"], "EXPAND ledger before")
            after = _budget(batch.to_dict()["ledger_after"], "EXPAND ledger after")
            if (
                before != disabled_budget or after != enabled_budget
                or before["max_mllm_calls"] != config["max_mllm_calls"]
                or before["max_processed_pixels"] != config["max_processed_pixels"]
            ):
                raise ValueError("EXPAND batch ledgers are not paired to P0/final budgets")
            maximum = after["max_mllm_calls"]
            expected_cost = 0.0 if maximum <= 0 else (
                after["mllm_calls"] - before["mllm_calls"]
            ) / maximum
            if batch.status == "no_requirements":
                raise ValueError("main P4A trace cannot call a batch without requirements")
            if batch.status == "success":
                expected_contract = "matched"
                expected_reason = None
            else:
                expected_contract = "not_observed"
                expected_reason = f"expand_{batch.status}"
            if audit.get("support_contract_status") != expected_contract:
                raise ValueError("EXPAND support contract status is not plan-derived")
            if no_op_reason != expected_reason:
                raise ValueError("EXPAND no-op reason differs from exact batch status")
            if batch.status == "render_noop" and (
                batch.failure_phase != "preflight"
                or batch.failure_reason != "aggregate_render_unchanged"
                or batch.exception_type is not None
            ):
                raise ValueError("EXPAND render-noop failure taxonomy is invalid")
            if batch.status == "budget_rejected" and (
                batch.failure_phase != "admission"
                or batch.exception_type != "BudgetExceeded"
            ):
                raise ValueError("EXPAND budget rejection taxonomy is invalid")
            if batch.status == "budget_rejected":
                calls_overflow = (
                    before["mllm_calls"] + batch.batch_plan["total_calls"]
                    > before["max_mllm_calls"]
                )
                pixels_overflow = (
                    before["processed_pixels"] + batch.batch_plan["total_pixels"]
                    > before["max_processed_pixels"]
                )
                expected_failure = (
                    "mllm_calls budget exhausted before model execution"
                    if calls_overflow else
                    "processed_pixels budget exhausted before model execution"
                )
                if not (calls_overflow or pixels_overflow) or batch.failure_reason != expected_failure:
                    raise ValueError("EXPAND budget rejection is not caused by an exact overflow")
            status = batch.status

    expected_contract = "matched" if batch is not None and batch.status == "success" else "not_observed"
    if audit.get("support_contract_status") != expected_contract:
        raise ValueError("EXPAND support contract status is not outcome-derived")

    current_payload = None if batch is None or batch.current_support is None else batch.current_support.to_dict()
    candidate_payload = None if batch is None or batch.candidate_support is None else batch.candidate_support.to_dict()
    if (
        audit.get("current_gap_support") != current_payload
        or audit.get("candidate_gap_support") != candidate_payload
        or audit.get("normalized_actual_cost") != expected_cost
    ):
        raise ValueError("EXPAND support/cost audit differs from its atomic batch")
    feasible = batch is not None and batch.status == "success"
    if audit.get("feasible") is not feasible:
        raise ValueError("EXPAND feasibility differs from exact batch contract")
    expected_trace_status = "observed_answer_free_audit_only" if feasible else no_op_reason
    if trace.get("support_status") != expected_trace_status:
        raise ValueError("EXPAND trace support status differs from exact outcome")
    if feasible:
        assert batch is not None
        current_p = batch.current_support.p_yes
        candidate_p = batch.candidate_support.p_yes
        gap = 1.0 - current_p
        delta = candidate_p - current_p
        raw_candidate, expected_stability = _validate_candidate_answer(
            benchmark, row, batch.candidate_answer,
        )
        if (
            abs(_finite(audit.get("uncalibrated_g_expand_proxy"), "EXPAND proxy") - gap) > 1e-12
            or abs(_finite(audit.get("support_delta"), "EXPAND support delta") - delta) > 1e-12
            or audit.get("candidate_stability") != expected_stability
            or audit.get("replacement_reason") != "replacement_disabled_p4a"
            or step.get("feasible_actions") != ["EXPAND"]
            or step.get("gaps") != {"g_expand_proxy_audit_only": gap}
        ):
            raise ValueError("feasible EXPAND measurements/candidate are not canonical")
        return True, raw_candidate, status, delta, batch.elapsed_seconds
    if (
        audit.get("uncalibrated_g_expand_proxy") is not None
        or audit.get("support_delta") is not None
        or audit.get("candidate_stability") is not None
        or audit.get("replacement_reason") is not None
        or step.get("feasible_actions") != [] or step.get("gaps") != {}
    ):
        raise ValueError("infeasible EXPAND cannot synthesize candidate state")
    return False, None, status, None, 0.0 if batch is None else batch.elapsed_seconds


def _validate_pairs(
    benchmark: str, disabled_rows: Sequence[Mapping[str, Any]],
    enabled_rows: Sequence[Mapping[str, Any]], expectation: PairExpectation,
    model_contract: Mapping[str, Any], launch_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if benchmark != "vstar" and benchmark not in _HR_BENCHMARKS:
        raise ValueError("P4A scorer supports only V* and HR-4K/8K")
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
        disabled_fingerprints.add(
            _sha256(disabled.get("_eg_run_fingerprint"), "disabled fingerprint")
        )
        enabled_fingerprints.add(
            _sha256(enabled.get("_eg_run_fingerprint"), "enabled fingerprint")
        )
        for field in ("input_image", "question", "options", "answer_type"):
            if disabled.get(field) != enabled.get(field):
                raise ValueError(f"paired rows differ in exact {field}")
        expected_type = "logits_match" if benchmark == "vstar" else "option_list"
        if disabled.get("answer_type") != expected_type:
            raise ValueError("benchmark and answer type disagree")
        output = disabled.get("output")
        if enabled.get("output") != output:
            raise ValueError("P4A emitted output differs from disabled P0")
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
            benchmark=benchmark, options=disabled.get("options"), output=output,
            context=f"disabled row {ordinal}",
        )
        enabled_budget, enabled_elapsed, enabled_answer, expand_step = _validate_common_trace(
            enabled_trace, config=enabled_config, enabled=True,
            benchmark=benchmark, options=enabled.get("options"), output=output,
            context=f"enabled row {ordinal}",
        )
        if enabled_answer != disabled_answer:
            raise ValueError("P4A P0 answer record differs from disabled sibling")
        source_image, source_audit = _resolve_source_image(enabled, launch_manifest)
        feasible, candidate, status, support_delta, batch_elapsed = _validate_expand_step(
            benchmark, enabled, enabled_trace, expand_step, disabled_budget,
            enabled_budget, enabled_answer, enabled_config, requirements,
            invalid_requirements, model_contract, source_image,
        )
        pairs.append({
            "ordinal": ordinal,
            "disabled": disabled,
            "enabled": enabled,
            "p0": output,
            "candidate": candidate,
            "feasible": feasible,
            "status": status,
            "no_op_reason": expand_step.get("no_op_reason"),
            "support_delta": support_delta,
            "batch_elapsed": batch_elapsed,
            "disabled_budget": disabled_budget,
            "enabled_budget": enabled_budget,
            "disabled_elapsed": disabled_elapsed,
            "enabled_elapsed": enabled_elapsed,
            "source_audit": source_audit,
        })
    if len(revisions) != 1 or len(disabled_fingerprints) != 1 or len(enabled_fingerprints) != 1:
        raise ValueError("each paired variant must use one revision and run fingerprint")
    digest = canonical_output_digest(disabled_rows)
    if expectation.output_digest is not None and digest != expectation.output_digest:
        raise ValueError("disabled output digest differs from the frozen reference")
    return pairs


def _memory_snapshot(
    disabled_rows: Sequence[Mapping[str, Any]],
    enabled_rows: Sequence[Mapping[str, Any]],
    disabled_manifest: Mapping[str, Any], enabled_manifest: Mapping[str, Any],
    *, include_labels: bool,
) -> str:
    def project(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if include_labels:
            return [dict(row) for row in rows]
        return [{key: row[key] for key in row if key != "answer"} for row in rows]

    return hashlib.sha256(_canonical({
        "disabled_rows": project(disabled_rows),
        "enabled_rows": project(enabled_rows),
        "disabled_launch_manifest": disabled_manifest,
        "enabled_launch_manifest": enabled_manifest,
    })).hexdigest()


def _annotation_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _mapping(
        _mapping(manifest.get("artifacts"), "P4A launch artifacts").get("annotation_file"),
        "P4A annotation artifact",
    )
    files = _list(artifact.get("files"), "P4A annotation artifact files")
    if len(files) != 1:
        raise ValueError("P4A annotation artifact must contain exactly one file")
    entry = _mapping(files[0], "P4A annotation artifact file")
    path = Path(artifact["path"])
    if path.is_symlink() or not path.is_file() or path.name != entry.get("path"):
        raise ValueError("P4A annotation artifact path is not the exact regular file")
    size = path.stat().st_size
    digest = _strict_sha256_file(path)
    if size != entry.get("size") or digest != entry.get("sha256"):
        raise ValueError("P4A annotation artifact bytes differ from the launch manifest")
    return {
        "path": str(path),
        "file_sha256": digest,
        "size": size,
        "manifest_sha256": artifact["sha256"],
    }


def _score_with_trusted_annotation(
    benchmark: str, pairs: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any], annotation_identity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(annotation_identity["path"])
    raw = path.read_bytes()
    if (
        len(raw) != annotation_identity["size"]
        or hashlib.sha256(raw).hexdigest() != annotation_identity["file_sha256"]
    ):
        raise RuntimeError("P4A annotation artifact changed before label binding")
    try:
        annotations = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("P4A annotation artifact is not strict UTF-8 JSON") from error
    if not isinstance(annotations, list):
        raise TypeError("P4A annotation artifact must contain a list")
    selected = []
    trusted_pairs = []
    for pair in pairs:
        ordinal = pair["ordinal"]
        if ordinal >= len(annotations):
            raise ValueError("P4A annotation ordinal is outside the trusted artifact")
        annotation = dict(_mapping(annotations[ordinal], "P4A trusted annotation row"))
        selected.append({"ordinal": ordinal, "annotation": annotation})
        for side in ("disabled", "enabled"):
            row = pair[side]
            for field in ("input_image", "question", "options", "answer_type"):
                if row.get(field) != annotation.get(field):
                    raise ValueError(f"P4A {side} row differs from trusted annotation {field}")
        trusted = dict(pair)
        trusted_disabled = dict(pair["disabled"])
        if benchmark in _HR_BENCHMARKS:
            label = _list(annotation.get("answer"), "trusted HR answer")
            if (
                len(label) != 4 or any(
                    not isinstance(item, str) or len(item) != 1
                    or item not in frozenset("ABCD") for item in label
                )
                or pair["disabled"].get("answer") != label
                or pair["enabled"].get("answer") != label
            ):
                raise ValueError("paired HR labels differ from the trusted annotation")
            trusted_disabled["answer"] = copy.deepcopy(label)
        elif "answer" in annotation or "answer" in pair["disabled"] or "answer" in pair["enabled"]:
            raise ValueError("V* must use the annotation-defined first-option label rule")
        trusted["disabled"] = trusted_disabled
        if benchmark == "vstar" and trusted["feasible"]:
            trusted["candidate"] = trusted["candidate"]["winner"]
        trusted_pairs.append(trusted)
    partition = _mapping(manifest.get("selected_partition"), "P4A selected partition")
    selected_sha256 = canonical_sha256(selected)
    if partition.get("rows_sha256") != selected_sha256:
        raise ValueError("P4A selected annotation rows differ from the launch partition")
    return _score_labels(benchmark, trusted_pairs), {
        **dict(annotation_identity),
        "selected_rows_sha256": selected_sha256,
        "selected_ordinals": [pair["ordinal"] for pair in pairs],
    }


def score_paired_rows(
    benchmark: str, disabled_rows: Sequence[Mapping[str, Any]],
    enabled_rows: Sequence[Mapping[str, Any]], expectation: PairExpectation,
    *, disabled_launch_manifest: Mapping[str, Any],
    enabled_launch_manifest: Mapping[str, Any],
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Validate every label-blind P4A invariant, then score raw candidates."""
    evaluator_before = _evaluator_revision()
    memory_before = _memory_snapshot(
        disabled_rows, enabled_rows, disabled_launch_manifest,
        enabled_launch_manifest, include_labels=False,
    )
    validate_launch_pair(
        disabled_rows, enabled_rows, disabled_launch_manifest,
        enabled_launch_manifest,
    )
    if (
        disabled_launch_manifest.get("benchmark") != benchmark
        or enabled_launch_manifest.get("benchmark") != benchmark
    ):
        raise ValueError("P4A launch benchmark differs from the requested benchmark")
    annotation_identity = _annotation_identity(enabled_launch_manifest)
    model_contract = _launch_model_contract(enabled_launch_manifest)
    pairs = _validate_pairs(
        benchmark, disabled_rows, enabled_rows, expectation,
        model_contract, enabled_launch_manifest,
    )
    if (
        _memory_snapshot(
            disabled_rows, enabled_rows, disabled_launch_manifest,
            enabled_launch_manifest, include_labels=False,
        ) != memory_before
        or _evaluator_revision() != evaluator_before
    ):
        raise RuntimeError("P4A label-blind inputs or evaluator changed during validation")
    # Deliberate unique evaluator-label boundary: trusted annotation bytes are opened here.
    label_memory_before = _memory_snapshot(
        disabled_rows, enabled_rows, disabled_launch_manifest,
        enabled_launch_manifest, include_labels=True,
    )
    rows, annotation_audit = _score_with_trusted_annotation(
        benchmark, pairs, enabled_launch_manifest, annotation_identity,
    )
    if _evaluator_revision() != evaluator_before:
        raise RuntimeError("P4A evaluator dependencies changed during label scoring")
    for pair in pairs:
        source = pair["source_audit"]
        if _strict_sha256_file(Path(source["artifact_path"])) != source["artifact_file_sha256"]:
            raise RuntimeError("P4A source image artifact changed during scoring")
    p0_correct = sum(row["p0_correct"] for row in rows)
    cycles = sum(row["total"] for row in rows)
    if p0_correct != expectation.p0_correct or cycles != expectation.cycles:
        raise ValueError("disabled score/cycles differ from the frozen reference")
    feasible = [row for row in rows if row["feasible"]]
    candidate_correct = sum(row["candidate_correct"] for row in feasible)
    candidate_total = sum(row["total"] for row in feasible)
    oracle_correct = sum(row["oracle_correct"] for row in rows)
    corrected_cycles = sum(max(0, row["oracle_correct"] - row["p0_correct"]) for row in rows)
    corrupted_cycles = sum(max(0, row["p0_correct"] - row["oracle_correct"]) for row in rows)
    support_deltas = [row["support_delta"] for row in feasible]
    source_material = [row["source_audit"] for row in rows]
    input_material = [{
        "ordinal": row["ordinal"],
        "input_image": row["disabled"]["input_image"],
        "question": row["disabled"]["question"],
        "options": row["disabled"]["options"],
        "answer_type": row["disabled"]["answer_type"],
    } for row in rows]
    disabled_digest = canonical_output_digest(disabled_rows)
    enabled_digest = canonical_output_digest(enabled_rows)
    report = {
        "benchmark": benchmark,
        "sample": {
            "topics": len(rows), "cycles": cycles,
            "topic_unit": "input row; HR four shuffles bound as one topic",
        },
        "revision": rows[0]["disabled"]["_eg_code_revision"],
        "output_digest": {
            "disabled": disabled_digest,
            "enabled": enabled_digest,
            "exact_match": disabled_digest == enabled_digest,
        },
        "input_identity": {
            "paired_inputs_sha256": hashlib.sha256(_canonical(input_material)).hexdigest(),
            "source_artifacts_sha256": hashlib.sha256(_canonical(source_material)).hexdigest(),
            "sources": source_material,
            "annotation": annotation_audit,
        },
        "p0": {"correct": p0_correct, "total": cycles, "accuracy": p0_correct / cycles},
        "candidate_alone": {
            "correct": candidate_correct,
            "total": candidate_total,
            "accuracy": None if not candidate_total else candidate_correct / candidate_total,
            "corrected_topics_vs_p0": sum(
                row["candidate_correct"] > row["p0_correct"] for row in feasible
            ),
            "corrupted_topics_vs_p0": sum(
                row["candidate_correct"] < row["p0_correct"] for row in feasible
            ),
            "corrected_cycles_vs_p0": sum(
                max(0, row["candidate_correct"] - row["p0_correct"]) for row in feasible
            ),
            "corrupted_cycles_vs_p0": sum(
                max(0, row["p0_correct"] - row["candidate_correct"]) for row in feasible
            ),
        },
        "oracle": {
            "correct": oracle_correct,
            "total": cycles,
            "accuracy": oracle_correct / cycles,
            "delta": (oracle_correct - p0_correct) / cycles,
            "corrected_topics": sum(
                row["oracle_correct"] > row["p0_correct"] for row in rows
            ),
            "corrupted_topics": sum(
                row["oracle_correct"] < row["p0_correct"] for row in rows
            ),
            "corrected_cycles": corrected_cycles,
            "corrupted_cycles": corrupted_cycles,
            "changed_topics": sum(row["changed"] for row in rows),
            "selected_candidate_topics": sum(row["selected_candidate"] for row in rows),
            "tie_rule": "retain_p0",
        },
        "candidate": {
            "feasible_topics": len(feasible),
            "infeasible_topics": len(rows) - len(feasible),
            "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
            "no_op_reason_counts": dict(sorted(Counter(
                row["no_op_reason"] for row in rows
                if row["no_op_reason"] is not None
            ).items())),
        },
        "cost": {
            "disabled": _runtime_report(rows, "disabled"),
            "enabled": _runtime_report(rows, "enabled"),
            "expand_batch_latency_s": {
                "total": sum(row["batch_elapsed"] for row in rows),
                "p50": statistics.median([row["batch_elapsed"] for row in rows]),
                "p95": _percentile([row["batch_elapsed"] for row in rows], 0.95),
            },
        },
        "support_delta": {
            "count": len(support_deltas),
            "mean": None if not support_deltas else statistics.fmean(support_deltas),
            "min": None if not support_deltas else min(support_deltas),
            "p50": None if not support_deltas else statistics.median(support_deltas),
            "p95": None if not support_deltas else _percentile(support_deltas, 0.95),
            "max": None if not support_deltas else max(support_deltas),
            "values_by_ordinal": [
                {"ordinal": row["ordinal"], "delta": row["support_delta"]}
                for row in feasible
            ],
        },
        "audit": {
            "replacements": 0,
            "forced_returns": {"disabled": len(rows), "enabled": len(rows)},
            "labels_bound_only_after_complete_pair_trace_pixel_validation": True,
            "candidate_stability_output_scored": False,
            "raw_candidate_answer_scored": True,
            "global_candidate_pool_reconstruction": (
                "trusted_frozen_runtime_revision_not_row_reconstructed"
            ),
        },
        "evaluator_revision": evaluator_before,
        "bootstrap": _bootstrap(benchmark, rows, bootstrap_replicates),
    }
    if (
        _memory_snapshot(
            disabled_rows, enabled_rows, disabled_launch_manifest,
            enabled_launch_manifest, include_labels=True,
        ) != label_memory_before
        or _evaluator_revision() != evaluator_before
    ):
        raise RuntimeError("P4A evaluator inputs or source changed during report construction")
    if _strict_sha256_file(Path(annotation_audit["path"])) != annotation_audit["file_sha256"]:
        raise RuntimeError("P4A annotation artifact changed during report construction")
    for pair in pairs:
        source = pair["source_audit"]
        if _strict_sha256_file(Path(source["artifact_path"])) != source["artifact_file_sha256"]:
            raise RuntimeError("P4A source image artifact changed during report construction")
    return report


def score_dev_paths(paths: Mapping[str, tuple[str | Path, str | Path]]) -> dict[str, Any]:
    if set(paths) != set(FROZEN_DEV_EXPECTATIONS):
        raise ValueError("P4A dev scorer requires V*, HR-4K, and HR-8K")
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
            benchmark, disabled_rows, enabled_rows,
            FROZEN_DEV_EXPECTATIONS[benchmark],
            disabled_launch_manifest=disabled_manifest,
            enabled_launch_manifest=enabled_manifest,
        )
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in all_paths}
    evaluator_after = _evaluator_revision()
    if after != snapshots or evaluator_after != evaluator_before:
        raise RuntimeError("P4A evaluator inputs or source changed during dev scoring")
    revisions = {report["revision"] for report in reports.values()}
    if len(revisions) != 1:
        raise ValueError("P4A dev inputs must share one inference revision")

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
    hr_gate = hr4 > 0.0 and hr8 > 0.0
    vstar_gate = vstar >= 0.0
    return {
        "schema_version": 2,
        "task": "unified-evidence-gap-phase4-p4a-spatial-context-expand-oracle-dev",
        "execution_identity": {
            "inference_revision": next(iter(revisions)),
            "evaluator_revision": evaluator_before,
            "input_jsonl_sha256": hashes(""),
            "launch_manifest_sha256": hashes(".launch-manifest.json"),
        },
        "benchmarks": reports,
        "launch_fingerprints": fingerprints,
        "cross_resolution_inference": "independent samples; no shared-sample CI",
        "gate": {
            "hr4_raw_oracle_strictly_positive": hr4 > 0.0,
            "hr8_raw_oracle_strictly_positive": hr8 > 0.0,
            "vstar_raw_oracle_nonnegative": vstar_gate,
            "p4a_warranted": hr_gate and vstar_gate,
            "candidate_generation_headroom_only": True,
            "deployable_selector_claimed": False,
            "ci_changes_predeclared_gate": False,
        },
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
    _write_json_atomic(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
