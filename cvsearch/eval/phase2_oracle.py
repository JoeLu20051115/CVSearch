"""Strict evaluator-only scorer for paired disabled/P2A NEXT artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cvsearch.evidence_gap.answers import official_letter
from cvsearch.evidence_gap.method import load_method_config
from cvsearch.evidence_gap.provenance import canonical_sha256


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "reproduction/evidence_gap/configs"
ENABLED_CONFIG = CONFIG_ROOT / "dev_unified_next_oracle_gamma000_budget512.json"
DISABLED_CONFIG = CONFIG_ROOT / "dev_unified_next_disabled_gamma000_budget512.json"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_INDICES = (249, 9749)
BOOTSTRAP_PREFIX = "p2a-next-v1:260810"
_SHA256_CHARS = frozenset("0123456789abcdef")
_HR_BENCHMARKS = frozenset({"hr-bench_4k", "hr-bench_8k"})


@dataclass(frozen=True)
class PairExpectation:
    topics: int
    cycles: int
    p0_correct: int
    output_digest: str | None

    def __post_init__(self) -> None:
        for name in ("topics", "cycles", "p0_correct"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.topics < 1 or self.cycles < self.topics or self.p0_correct > self.cycles:
            raise ValueError("invalid pair expectation counts")
        if self.output_digest is not None:
            _sha256(self.output_digest, "expected output digest")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(char not in _SHA256_CHARS for char in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA256")
    return value


FROZEN_DEV_EXPECTATIONS = {
    "vstar": PairExpectation(
        37, 37, 32,
        "1ac383b2dd64f29023d9d36e50e6986854611eacaae9d8ead3b1b8b9a93a8ba7",
    ),
    "hr-bench_4k": PairExpectation(39, 156, 106, None),
    "hr-bench_8k": PairExpectation(28, 112, 90, None),
}
FROZEN_FULL_VSTAR_EXPECTATION = PairExpectation(191, 191, 171, None)
FROZEN_DEV_ORDINAL_SHA256 = {
    "vstar": "dbbb9d17f1e241b366d1ddb02593f8918a3bf08b6580ccd9b809efee91718712",
    "hr-bench_4k": "21dac8e13a98d5f73377375beba4d382c4f970db32271779e090e74592862e56",
    "hr-bench_8k": "fc42e4cce529b95b45ee97da6802e70fbfb2f73faf827fdd2a2e934281d47372",
}


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    return value


def _finite(value: Any, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{context} must be finite and at least {minimum}")
    return result


def canonical_output_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    material = []
    for row in rows:
        ordinal = row.get("_eg_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("output digest ordinal must be a non-negative integer")
        material.append({"ordinal": ordinal, "output": row.get("output")})
    return hashlib.sha256(_canonical(material)).hexdigest()


def _budget(value: Any, context: str) -> dict[str, int]:
    budget = _mapping(value, context)
    expected = {
        "max_mllm_calls", "max_processed_pixels", "mllm_calls", "processed_pixels",
    }
    if set(budget) != expected:
        raise ValueError(f"{context} has an invalid schema")
    result = {}
    for key in expected:
        item = budget[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{context}.{key} must be a non-negative integer")
        result[key] = item
    if result["max_mllm_calls"] != 512 or result["max_processed_pixels"] != 10_000_000_000:
        raise ValueError(f"{context} does not use the frozen limits")
    if (
        result["mllm_calls"] > result["max_mllm_calls"]
        or result["processed_pixels"] > result["max_processed_pixels"]
    ):
        raise ValueError(f"{context} exceeds its declared limits")
    return result


def _answer_output(value: Any, context: str) -> Any:
    return _mapping(value, context).get("output")


def _validate_config_pair(disabled: Mapping[str, Any], enabled: Mapping[str, Any]) -> None:
    expected_disabled = load_method_config(str(DISABLED_CONFIG))
    expected_enabled = load_method_config(str(ENABLED_CONFIG))
    if dict(disabled) != expected_disabled or dict(enabled) != expected_enabled:
        raise ValueError("run effective config does not match the frozen Phase-2 config")
    changed = {
        key for key in expected_enabled
        if expected_enabled.get(key) != expected_disabled.get(key)
    }
    if changed != {"config_id", "next_enabled", "next_admission_mode"}:
        raise ValueError("disabled and enabled configs violate the frozen sibling contract")


def _validate_common_trace(
    trace: Mapping[str, Any], *, config: Mapping[str, Any], enabled: bool,
    output: Any, context: str,
) -> tuple[dict[str, int], float, list[Mapping[str, Any]]]:
    if trace.get("effective_config") != config or trace.get("config_id") != config["config_id"]:
        raise ValueError(f"{context} config identity mismatch")
    if trace.get("termination") != "FORCED_RETURN":
        raise ValueError(f"{context} must terminate by forced return")
    steps = _list(trace.get("steps"), f"{context}.steps")
    if not steps or any(not isinstance(step, Mapping) for step in steps):
        raise ValueError(f"{context}.steps must contain trace objects")
    if steps[-1].get("action") != "FORCED_RETURN":
        raise ValueError(f"{context} final step must be forced return")
    if _answer_output(trace.get("final_answer"), f"{context}.final_answer") != output:
        raise ValueError(f"{context} final answer differs from emitted output")
    if _answer_output(trace.get("anchor_answer"), f"{context}.anchor_answer") != output:
        raise ValueError(f"{context} anchor answer differs from emitted output")
    if trace.get("budget_interrupted") is not False:
        raise ValueError(f"{context} budget interruption is not eligible for the gate")
    if trace.get("pixel_accounting") != config["pixel_accounting"]:
        raise ValueError(f"{context} pixel accounting mismatch")
    elapsed = _finite(trace.get("elapsed_seconds"), f"{context}.elapsed_seconds", minimum=0.0)
    budget = _budget(trace.get("budget"), f"{context}.budget")
    next_steps = [step for step in steps if step.get("action") == "NEXT"]
    if enabled:
        if len(next_steps) != 1:
            raise ValueError(f"{context} must contain exactly one NEXT audit")
        if any(value is not None for value in (
            trace.get("anchor_state_score"), trace.get("selected_state_score"),
            trace.get("replacement_margin"),
        )):
            raise ValueError(f"{context} selector scores must remain disabled")
    elif next_steps:
        raise ValueError(f"{context} disabled sibling must not contain NEXT")
    elif trace.get("replacement_margin") != 0.0 or trace.get("support_status") != "not_observed":
        raise ValueError(f"{context} disabled selector/support status mismatch")
    return budget, elapsed, next_steps


def _validate_batch_plan(batch: Mapping[str, Any], before: Mapping[str, int],
                         after: Mapping[str, int], answer_type: str) -> None:
    plan = _mapping(batch.get("batch_plan"), "NEXT batch plan")
    plan_hash = _sha256(plan.get("plan_hash"), "NEXT batch plan hash")
    without_hash = dict(plan)
    without_hash.pop("plan_hash")
    if hashlib.sha256(_canonical(without_hash)).hexdigest() != plan_hash:
        raise ValueError("NEXT batch plan hash does not match its contents")
    if batch.get("batch_plan_hash") != plan_hash:
        raise ValueError("NEXT batch result plan hash mismatch")
    if plan.get("answer_type") != answer_type:
        raise ValueError("NEXT batch answer type mismatch")
    for total, ledger_key in (("total_calls", "mllm_calls"), ("total_pixels", "processed_pixels")):
        value = plan.get(total)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"NEXT batch {total} must be positive")
        if after[ledger_key] - before[ledger_key] != value:
            raise ValueError(f"NEXT batch {total} does not match ledger delta")


def _validate_next(
    step: Mapping[str, Any], disabled_budget: Mapping[str, int],
    enabled_budget: Mapping[str, int], answer_type: str,
) -> tuple[bool, Any, str, float | None, float]:
    audit = _mapping(step.get("next_audit"), "NEXT audit")
    if audit.get("coverage_status") != "not_observed":
        raise ValueError("NEXT coverage must remain unobserved")
    if audit.get("verifier_status") != "disabled_same_checkpoint_unpromoted":
        raise ValueError("NEXT verifier must remain disabled")
    if audit.get("verifier_avg") is not None or audit.get("verifier_min") is not None:
        raise ValueError("NEXT verifier scores must remain null")
    if audit.get("score_margin") is not None or audit.get("score_status") != "unavailable_missing_verifier_coverage":
        raise ValueError("NEXT selector score must remain unavailable")
    feasible = audit.get("feasible")
    if not isinstance(feasible, bool):
        raise TypeError("NEXT feasibility must be boolean")
    batch_value = audit.get("batch_result")
    if not feasible:
        if step.get("feasible_actions") != [] or not isinstance(step.get("no_op_reason"), str):
            raise ValueError("infeasible NEXT must retain its exact no-op status")
        if (
            audit.get("replacement_reason") is not None
            or audit.get("g_next") is not None
            or audit.get("support_delta") is not None
            or audit.get("candidate_stability") is not None
            or audit.get("normalized_actual_cost") is not None
            or audit.get("current_gap_support") is not None
            or audit.get("candidate_gap_support") is not None
            or audit.get("support_contract_status") != "not_observed"
        ):
            raise ValueError("infeasible NEXT cannot synthesize candidate measurements")
        status = "no_batch"
        if batch_value is None:
            if enabled_budget != disabled_budget:
                raise ValueError("no-batch NEXT changed the P0 budget")
        else:
            batch = _mapping(batch_value, "infeasible NEXT batch")
            status = batch.get("status")
            if status not in {"budget_rejected", "model_failed", "no_requirements", "success"}:
                raise ValueError("invalid infeasible NEXT batch status")
            if (
                batch.get("promotable") is not False
                or batch.get("verifier_status") != "disabled_same_checkpoint_unpromoted"
                or batch.get("verifier_avg") is not None or batch.get("verifier_min") is not None
            ):
                raise ValueError("infeasible NEXT batch has invalid verifier provenance")
            before = _budget(batch.get("ledger_before"), "infeasible NEXT ledger before")
            after = _budget(batch.get("ledger_after"), "infeasible NEXT ledger after")
            if before != disabled_budget or after != enabled_budget:
                raise ValueError("infeasible NEXT batch is not paired to P0/final budgets")
            expected_state = {
                "budget_rejected": (False, False),
                "no_requirements": (False, False),
                "model_failed": (True, True),
                "success": (True, True),
            }[status]
            if (batch.get("admitted"), batch.get("charged")) != expected_state:
                raise ValueError("infeasible NEXT batch admission/charge state is invalid")
            if status in {"budget_rejected", "no_requirements"}:
                if before != after or batch.get("candidate_answer") is not None:
                    raise ValueError("uncharged NEXT batch changed budget or exposed an answer")
            else:
                _validate_batch_plan(batch, before, after, answer_type)
                if status == "model_failed" and batch.get("candidate_answer") is not None:
                    raise ValueError("failed NEXT batch cannot expose a candidate answer")
                if status == "success" and batch.get("candidate_answer") is None:
                    raise ValueError("support-mismatched success lost its raw candidate answer")
        return False, None, str(status), None, 0.0

    if step.get("feasible_actions") != ["NEXT"] or step.get("no_op_reason") is not None:
        raise ValueError("feasible NEXT action trace is inconsistent")
    if audit.get("support_contract_status") != "matched":
        raise ValueError("feasible NEXT support contract must match")
    if audit.get("replacement_reason") != "replacement_disabled_p2a":
        raise ValueError("feasible NEXT replacement must remain disabled")
    batch = _mapping(batch_value, "feasible NEXT batch")
    if (
        batch.get("status") != "success" or batch.get("admitted") is not True
        or batch.get("charged") is not True or batch.get("promotable") is not False
        or batch.get("verifier_status") != "disabled_same_checkpoint_unpromoted"
        or batch.get("verifier_avg") is not None or batch.get("verifier_min") is not None
    ):
        raise ValueError("feasible NEXT batch status/provenance is invalid")
    before = _budget(batch.get("ledger_before"), "NEXT ledger before")
    after = _budget(batch.get("ledger_after"), "NEXT ledger after")
    if before != disabled_budget or after != enabled_budget:
        raise ValueError("NEXT batch ledger is not paired to exact P0/final budgets")
    _validate_batch_plan(batch, before, after, answer_type)
    current = _mapping(batch.get("current_support"), "current support")
    candidate = _mapping(batch.get("candidate_support"), "candidate support")
    current_p = _finite(current.get("p_yes"), "current support p_yes")
    candidate_p = _finite(candidate.get("p_yes"), "candidate support p_yes")
    if not 0.0 <= current_p <= 1.0 or not 0.0 <= candidate_p <= 1.0:
        raise ValueError("support probabilities must be in [0, 1]")
    support_delta = _finite(audit.get("support_delta"), "NEXT support delta")
    if abs(support_delta - (candidate_p - current_p)) > 1e-12:
        raise ValueError("NEXT support delta does not match support records")
    batch_elapsed = _finite(batch.get("elapsed_seconds"), "NEXT batch latency", minimum=0.0)
    answer = batch.get("candidate_answer")
    if answer_type == "logits_match":
        candidate_answer = _mapping(answer, "V* candidate answer")
        if set(candidate_answer) != {"winner", "losses"}:
            raise ValueError("V* candidate answer schema is invalid")
        winner = candidate_answer["winner"]
        losses = _list(candidate_answer["losses"], "V* losses")
        if isinstance(winner, bool) or not isinstance(winner, int) or not losses:
            raise ValueError("V* winner/losses are invalid")
        numeric_losses = [_finite(loss, "V* loss") for loss in losses]
        if not 0 <= winner < len(losses) or winner != min(range(len(losses)), key=numeric_losses.__getitem__):
            raise ValueError("V* winner must equal exact loss argmin")
        answer = winner
    else:
        raw = _list(answer, "HR candidate raw outputs")
        if len(raw) != 4 or any(not isinstance(value, str) for value in raw):
            raise ValueError("HR candidate answer must contain four exact raw strings")
        answer = list(raw)
    return True, answer, "success", support_delta, batch_elapsed


def _validate_pairs(
    benchmark: str, disabled_rows: Sequence[Mapping[str, Any]],
    enabled_rows: Sequence[Mapping[str, Any]], expectation: PairExpectation,
) -> list[dict[str, Any]]:
    if benchmark != "vstar" and benchmark not in _HR_BENCHMARKS:
        raise ValueError("Phase-2 oracle supports only V* and the two HR resolutions")
    if len(disabled_rows) != expectation.topics or len(enabled_rows) != expectation.topics:
        raise ValueError("paired topic count does not match the frozen expectation")
    disabled_config = load_method_config(str(DISABLED_CONFIG))
    enabled_config = load_method_config(str(ENABLED_CONFIG))
    _validate_config_pair(disabled_config, enabled_config)
    pairs = []
    previous = -1
    revisions = set()
    disabled_fingerprints = set()
    enabled_fingerprints = set()
    for index, (disabled, enabled) in enumerate(zip(disabled_rows, enabled_rows)):
        if not isinstance(disabled, Mapping) or not isinstance(enabled, Mapping):
            raise TypeError("paired JSONL rows must be objects")
        ordinal = disabled.get("_eg_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal <= previous:
            raise ValueError("disabled rows must be in unique increasing ordinal order")
        previous = ordinal
        if enabled.get("_eg_ordinal") != ordinal:
            raise ValueError("disabled and enabled rows differ in ordinal order")
        revision = _sha256(disabled.get("_eg_code_revision"), "disabled code revision")
        if enabled.get("_eg_code_revision") != revision:
            raise ValueError("paired rows differ in exact code revision")
        revisions.add(revision)
        disabled_fingerprints.add(_sha256(disabled.get("_eg_run_fingerprint"), "disabled run fingerprint"))
        enabled_fingerprints.add(_sha256(enabled.get("_eg_run_fingerprint"), "enabled run fingerprint"))
        for field in ("input_image", "question", "options", "answer_type", "answer"):
            if disabled.get(field) != enabled.get(field):
                raise ValueError(f"paired rows differ in exact {field} input")
        answer_type = disabled.get("answer_type")
        expected_type = "logits_match" if benchmark == "vstar" else "option_list"
        if answer_type != expected_type:
            raise ValueError("benchmark and answer_type disagree")
        output = disabled.get("output")
        if enabled.get("output") != output:
            raise ValueError("P2A emitted output differs from disabled P0")
        disabled_trace = _mapping(disabled.get("method_trace"), "disabled trace")
        enabled_trace = _mapping(enabled.get("method_trace"), "enabled trace")
        disabled_budget, disabled_elapsed, _ = _validate_common_trace(
            disabled_trace, config=disabled_config, enabled=False,
            output=output, context=f"disabled row {ordinal}",
        )
        enabled_budget, enabled_elapsed, next_steps = _validate_common_trace(
            enabled_trace, config=enabled_config, enabled=True,
            output=output, context=f"enabled row {ordinal}",
        )
        audit = _mapping(next_steps[0].get("next_audit"), "NEXT audit")
        p0_anchor = _mapping(audit.get("p0_anchor"), "NEXT P0 anchor")
        if set(p0_anchor) != {
            "emitted_answer", "cvsearch_raw", "producing_phase", "node_keys", "support_view",
        }:
            raise ValueError("NEXT P0 anchor schema is invalid")
        if p0_anchor.get("emitted_answer") != output:
            raise ValueError("P2A output differs from its immutable P0 anchor")
        producing_phase = p0_anchor.get("producing_phase")
        node_keys = _list(p0_anchor.get("node_keys"), "NEXT P0 node keys")
        if (
            any(not isinstance(key, str) or not key for key in node_keys)
            or len(node_keys) != len(set(node_keys))
        ):
            raise ValueError("NEXT P0 node keys must be unique nonempty strings")
        support_value = p0_anchor.get("support_view")
        if support_value is None:
            if not (
                benchmark == "vstar"
                and p0_anchor.get("producing_phase") == "search"
                and audit.get("feasible") is False
                and audit.get("batch_result") is None
                and next_steps[0].get("no_op_reason")
                == "next_p0_support_view_unavailable"
            ):
                raise ValueError("NEXT P0 support view may be unavailable only on its exact no-op")
            support_view = None
        else:
            support_view = _list(support_value, "NEXT P0 support view")
            if any(not isinstance(item, Mapping) for item in support_view):
                raise ValueError("NEXT P0 support view must contain descriptor objects")
            if [item.get("canonical_key") for item in support_view] != node_keys:
                raise ValueError("NEXT P0 support view does not match its node keys")
        raw = p0_anchor.get("cvsearch_raw")
        if benchmark in _HR_BENCHMARKS:
            if producing_phase != "cvsearch_raw" or raw != output:
                raise ValueError("HR NEXT P0 must bind the exact cvsearch_raw producer")
        else:
            options = _list(disabled.get("options"), "V* options")
            if (
                isinstance(raw, bool) or not isinstance(raw, int)
                or not 0 <= raw < len(options)
            ):
                raise ValueError("V* cvsearch_raw must be a valid option index")
            if producing_phase == "root":
                if node_keys or support_view != []:
                    raise ValueError("V* root P0 must bind the empty root support view")
            elif producing_phase == "search":
                if raw != output or not node_keys:
                    raise ValueError("V* search P0 must bind its exact searched view and raw output")
            else:
                raise ValueError("V* NEXT P0 producing phase must be root or search")
        feasible, candidate, status, support_delta, batch_elapsed = _validate_next(
            next_steps[0], disabled_budget, enabled_budget, answer_type,
        )
        pairs.append({
            "ordinal": ordinal, "disabled": disabled, "enabled": enabled,
            "p0": output, "candidate": candidate, "feasible": feasible,
            "status": status, "support_delta": support_delta,
            "batch_elapsed": batch_elapsed,
            "disabled_budget": disabled_budget, "enabled_budget": enabled_budget,
            "disabled_elapsed": disabled_elapsed, "enabled_elapsed": enabled_elapsed,
        })
    if len(revisions) != 1 or len(disabled_fingerprints) != 1 or len(enabled_fingerprints) != 1:
        raise ValueError("each paired variant must use one exact revision and run fingerprint")
    digest = canonical_output_digest(disabled_rows)
    if expectation.output_digest is not None and digest != expectation.output_digest:
        raise ValueError("disabled output digest does not match the frozen reference")
    return pairs


def validate_launch_pair(
    disabled_rows: Sequence[Mapping[str, Any]],
    enabled_rows: Sequence[Mapping[str, Any]],
    disabled_manifest: Mapping[str, Any],
    enabled_manifest: Mapping[str, Any],
) -> dict[str, str]:
    """Bind paired rows to sidecars with one shared non-config launch identity."""
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
    disabled_identity = dict(disabled_manifest)
    enabled_identity = dict(enabled_manifest)
    disabled_identity.pop("config")
    enabled_identity.pop("config")
    if disabled_identity != enabled_identity:
        raise ValueError("disabled and enabled launch manifests differ in non-config identity")
    ordinals = [row.get("_eg_ordinal") for row in disabled_rows]
    if ordinals != [row.get("_eg_ordinal") for row in enabled_rows]:
        raise ValueError("launch pair row ordinals differ")
    partition = _mapping(disabled_manifest.get("selected_partition"), "selected partition")
    if partition.get("ordinals") != ordinals:
        raise ValueError("launch manifest selected partition differs from JSONL ordinals")
    code = _mapping(disabled_manifest.get("code"), "launch code")
    revision = _sha256(code.get("revision"), "launch code revision")
    for row in tuple(disabled_rows) + tuple(enabled_rows):
        if row.get("_eg_code_revision") != revision:
            raise ValueError("launch manifest code revision differs from JSONL")
    disabled_fingerprint = canonical_sha256(disabled_manifest)
    enabled_fingerprint = canonical_sha256(enabled_manifest)
    if any(row.get("_eg_run_fingerprint") != disabled_fingerprint for row in disabled_rows):
        raise ValueError("disabled JSONL fingerprint does not match launch manifest")
    if any(row.get("_eg_run_fingerprint") != enabled_fingerprint for row in enabled_rows):
        raise ValueError("enabled JSONL fingerprint does not match launch manifest")
    return {"disabled": disabled_fingerprint, "enabled": enabled_fingerprint}


def validate_frozen_dev_identity(
    benchmark: str, manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if benchmark not in FROZEN_DEV_EXPECTATIONS:
        raise ValueError("unknown frozen development benchmark")
    manifest = _mapping(manifest, "development launch manifest")
    if manifest.get("benchmark") != benchmark:
        raise ValueError("launch manifest benchmark differs from requested benchmark")
    partition = _mapping(manifest.get("selected_partition"), "selected development partition")
    ordinals = partition.get("ordinals")
    if not isinstance(ordinals, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in ordinals
    ):
        raise ValueError("development partition ordinals are invalid")
    row_ordinals = [row.get("_eg_ordinal") for row in rows]
    expected_topics = FROZEN_DEV_EXPECTATIONS[benchmark].topics
    if (
        ordinals != row_ordinals
        or len(ordinals) != expected_topics
        or partition.get("rows") != expected_topics
    ):
        raise ValueError("development partition rows or ordinals do not match JSONL")
    ordinal_hash = hashlib.sha256(
        ",".join(str(value) for value in ordinals).encode("utf-8")
    ).hexdigest()
    if ordinal_hash != FROZEN_DEV_ORDINAL_SHA256[benchmark]:
        raise ValueError("development partition ordinal hash is not frozen")
    if (
        partition.get("split") != "dev"
        or partition.get("split_seed") != 260809
        or partition.get("num_chunks") != 1
        or partition.get("chunk_idx") != 0
    ):
        raise ValueError("development split, seed, or chunk identity is not frozen")


def validate_frozen_full_vstar_identity(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
) -> None:
    manifest = _mapping(manifest, "full V* launch manifest")
    if manifest.get("benchmark") != "vstar":
        raise ValueError("full V* launch manifest benchmark is not vstar")
    partition = _mapping(manifest.get("selected_partition"), "full V* partition")
    ordinals = partition.get("ordinals")
    expected_ordinals = list(range(191))
    if (
        ordinals != expected_ordinals
        or [row.get("_eg_ordinal") for row in rows] != expected_ordinals
        or partition.get("rows") != 191
    ):
        raise ValueError("full V* partition must contain exact ordinals 0 through 190")
    if (
        partition.get("split") != "all"
        or partition.get("split_seed") != 260809
        or partition.get("num_chunks") != 1
        or partition.get("chunk_idx") != 0
    ):
        raise ValueError("full V* split, seed, or chunk identity is not frozen")


def _hr_correct(label: Any, output: Any) -> tuple[int, tuple[str | None, ...]]:
    truth = _list(label, "HR labels")
    raw = _list(output, "HR raw outputs")
    if len(truth) != 4 or len(raw) != 4 or any(item not in "ABCD" for item in truth):
        raise ValueError("HR labels and raw outputs must contain four cycles")
    if any(not isinstance(item, str) for item in raw):
        raise ValueError("HR raw outputs must be strings")
    parsed = tuple(official_letter(item) for item in raw)
    return sum(want == got for want, got in zip(truth, parsed)), parsed


def _score_labels(benchmark: str, pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for pair in pairs:
        row = pair["disabled"]
        if benchmark == "vstar":
            p0 = pair["p0"]
            if isinstance(p0, bool) or not isinstance(p0, int):
                raise ValueError("V* P0 output must be a plain integer")
            p0_correct = int(p0 == 0)
            candidate_correct = None
            changed = False
            if pair["feasible"]:
                candidate = pair["candidate"]
                candidate_correct = int(candidate == 0)
                changed = candidate != p0
            oracle_correct = max(p0_correct, candidate_correct or 0)
            selected_candidate = candidate_correct is not None and candidate_correct > p0_correct
            total = 1
        else:
            p0_correct, p0_parsed = _hr_correct(row.get("answer"), pair["p0"])
            candidate_correct = None
            changed = False
            if pair["feasible"]:
                candidate_correct, candidate_parsed = _hr_correct(
                    row.get("answer"), pair["candidate"],
                )
                changed = candidate_parsed != p0_parsed
            selected_candidate = candidate_correct is not None and candidate_correct > p0_correct
            oracle_correct = candidate_correct if selected_candidate else p0_correct
            total = 4
        scored.append(dict(
            pair, p0_correct=p0_correct, candidate_correct=candidate_correct,
            oracle_correct=oracle_correct, selected_candidate=selected_candidate,
            changed=changed, total=total,
        ))
    return scored


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _runtime_report(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    calls = [row[f"{prefix}_budget"]["mllm_calls"] for row in rows]
    pixels = [row[f"{prefix}_budget"]["processed_pixels"] for row in rows]
    latency = [row[f"{prefix}_elapsed"] for row in rows]
    return {
        "calls": sum(calls), "pixels": sum(pixels),
        "latency_p50_s": statistics.median(latency),
        "latency_p95_s": _percentile(latency, 0.95),
    }


def bootstrap_draw_index(benchmark: str, replicate: int, draw: int, topics: int) -> int:
    if not benchmark or not isinstance(benchmark, str):
        raise ValueError("bootstrap benchmark must be a nonempty string")
    for name, value in (("replicate", replicate), ("draw", draw)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"bootstrap {name} must be a non-negative integer")
    if isinstance(topics, bool) or not isinstance(topics, int) or topics < 1:
        raise ValueError("bootstrap topics must be a positive integer")
    material = f"{BOOTSTRAP_PREFIX}:{benchmark}:{replicate}:{draw}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % topics


def _bootstrap(benchmark: str, rows: Sequence[Mapping[str, Any]], replicates: int) -> dict[str, Any]:
    if replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("Phase-2 bootstrap is frozen at exactly 10,000 replicates")
    count = len(rows)
    deltas = [row["oracle_correct"] - row["p0_correct"] for row in rows]
    totals = [row["total"] for row in rows]
    if len(set(totals)) != 1:
        raise ValueError("bootstrap topics must use one benchmark scoring unit")
    denominator = count * totals[0]
    samples = []
    for replicate in range(replicates):
        improvement = 0
        for draw in range(count):
            index = bootstrap_draw_index(benchmark, replicate, draw, count)
            improvement += deltas[index]
        samples.append(improvement / denominator)
    samples.sort()
    point = sum(deltas) / denominator
    return {
        "unit": "topic",
        "replicates": replicates,
        "draw_rule": 'SHA256("p2a-next-v1:260810:{benchmark}:{replicate}:{draw}").digest()[:8] interpreted big-endian mod N',
        "index_rule": list(BOOTSTRAP_INDICES),
        "delta_point_estimate": point,
        "delta_ci95": [samples[BOOTSTRAP_INDICES[0]], samples[BOOTSTRAP_INDICES[1]]],
    }


def score_paired_rows(
    benchmark: str, disabled_rows: Sequence[Mapping[str, Any]],
    enabled_rows: Sequence[Mapping[str, Any]], expectation: PairExpectation,
    *, bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Validate label-blind run invariants, then score the frozen labels."""
    pairs = _validate_pairs(benchmark, disabled_rows, enabled_rows, expectation)
    rows = _score_labels(benchmark, pairs)  # Deliberate label boundary.
    p0_correct = sum(row["p0_correct"] for row in rows)
    if p0_correct != expectation.p0_correct:
        raise ValueError("disabled score does not match the frozen reference")
    cycles = sum(row["total"] for row in rows)
    if cycles != expectation.cycles:
        raise ValueError("scoring cycles do not match the frozen expectation")
    feasible = [row for row in rows if row["feasible"]]
    candidate_correct = sum(row["candidate_correct"] for row in feasible)
    candidate_total = sum(row["total"] for row in feasible)
    oracle_correct = sum(row["oracle_correct"] for row in rows)
    corrected = sum(max(0, row["oracle_correct"] - row["p0_correct"]) for row in rows)
    corrupted = sum(max(0, row["p0_correct"] - row["oracle_correct"]) for row in rows)
    support_deltas = [row["support_delta"] for row in feasible]
    revision = rows[0]["disabled"]["_eg_code_revision"]
    disabled_digest = canonical_output_digest(disabled_rows)
    enabled_digest = canonical_output_digest(enabled_rows)
    report = {
        "benchmark": benchmark,
        "sample": {"topics": len(rows), "cycles": cycles, "topic_unit": "input row; all HR shuffles bound"},
        "revision": revision,
        "output_digest": {"disabled": disabled_digest, "enabled": enabled_digest, "exact_match": disabled_digest == enabled_digest},
        "p0": {"correct": p0_correct, "total": cycles, "accuracy": p0_correct / cycles},
        "candidate_alone": {
            "correct": candidate_correct, "total": candidate_total,
            "accuracy": None if not candidate_total else candidate_correct / candidate_total,
            "corrected_vs_p0": sum(
                max(0, row["candidate_correct"] - row["p0_correct"]) for row in feasible
            ),
            "corrupted_vs_p0": sum(
                max(0, row["p0_correct"] - row["candidate_correct"]) for row in feasible
            ),
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
            "next_batch_latency_s": {
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
            "replacements": 0,
            "forced_returns": {
                "disabled": len(rows), "enabled": len(rows),
            },
            "labels_used_only_after_pair_trace_config_budget_validation": True,
            "candidate_stability_output_scored": False,
        },
        "bootstrap": _bootstrap(benchmark, rows, bootstrap_replicates),
    }
    return report


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    raw = source.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"JSONL must be nonempty and newline-terminated: {source}")
    rows = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise ValueError(f"blank JSONL line {line_number}: {source}")
        try:
            row = json.loads(
                line, parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
            _canonical(row)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid JSONL line {line_number}: {source}") from error
        if not isinstance(row, dict):
            raise TypeError(f"JSONL line {line_number} must be an object")
        rows.append(row)
    return rows


def load_launch_manifest(answer_path: str | Path) -> dict[str, Any]:
    path = Path(f"{Path(answer_path)}.launch-manifest.json")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required launch manifest is missing: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
        _canonical(value)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid launch manifest: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"launch manifest must be an object: {path}")
    return value


def score_dev_paths(paths: Mapping[str, tuple[str | Path, str | Path]]) -> dict[str, Any]:
    if set(paths) != set(FROZEN_DEV_EXPECTATIONS):
        raise ValueError("dev scorer requires exactly V*, HR-4K, and HR-8K")
    artifact_paths = [Path(path) for pair in paths.values() for path in pair]
    artifact_paths.extend(
        Path(f"{path}.launch-manifest.json") for path in tuple(artifact_paths)
    )
    snapshots = {
        str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in artifact_paths
    }
    reports = {}
    launch_fingerprints = {}
    for benchmark, pair in paths.items():
        disabled_rows = load_jsonl(pair[0])
        enabled_rows = load_jsonl(pair[1])
        disabled_manifest = load_launch_manifest(pair[0])
        enabled_manifest = load_launch_manifest(pair[1])
        validate_frozen_dev_identity(benchmark, disabled_manifest, disabled_rows)
        validate_frozen_dev_identity(benchmark, enabled_manifest, enabled_rows)
        launch_fingerprints[benchmark] = validate_launch_pair(
            disabled_rows, enabled_rows, disabled_manifest, enabled_manifest,
        )
        reports[benchmark] = score_paired_rows(
            benchmark, disabled_rows, enabled_rows,
            FROZEN_DEV_EXPECTATIONS[benchmark],
        )
    after = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in snapshots
    }
    if after != snapshots:
        raise RuntimeError("oracle scorer input artifacts changed during evaluation")
    hr4 = reports["hr-bench_4k"]["oracle"]["delta"]
    hr8 = reports["hr-bench_8k"]["oracle"]["delta"]
    gate = hr4 >= 0.0 and hr8 >= 0.0 and min(hr4, hr8) > 0.0
    return {
        "schema_version": 1,
        "task": "unified-evidence-gap-phase2-next-oracle-dev",
        "benchmarks": reports,
        "launch_fingerprints": launch_fingerprints,
        "cross_resolution_inference": "independent samples; no shared-sample CI",
        "gate": {
            "hr4_nonnegative": hr4 >= 0.0,
            "hr8_nonnegative": hr8 >= 0.0,
            "minimum_strictly_positive": min(hr4, hr8) > 0.0,
            "p2b_warranted": gate,
            "full_vstar_required_if_warranted": gate,
            "ci_changes_predeclared_gate": False,
        },
    }


def score_full_vstar_paths(
    disabled_path: str | Path, enabled_path: str | Path,
) -> dict[str, Any]:
    paths = [
        Path(disabled_path), Path(enabled_path),
        Path(f"{disabled_path}.launch-manifest.json"),
        Path(f"{enabled_path}.launch-manifest.json"),
    ]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    disabled_rows = load_jsonl(disabled_path)
    enabled_rows = load_jsonl(enabled_path)
    disabled_manifest = load_launch_manifest(disabled_path)
    enabled_manifest = load_launch_manifest(enabled_path)
    validate_frozen_full_vstar_identity(disabled_manifest, disabled_rows)
    validate_frozen_full_vstar_identity(enabled_manifest, enabled_rows)
    fingerprints = validate_launch_pair(
        disabled_rows, enabled_rows,
        disabled_manifest, enabled_manifest,
    )
    report = score_paired_rows(
        "vstar", disabled_rows, enabled_rows, FROZEN_FULL_VSTAR_EXPECTATION,
    )
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if after != before:
        raise RuntimeError("full V* oracle scorer inputs changed during evaluation")
    if report["output_digest"]["exact_match"] is not True or report["p0"]["correct"] != 171:
        raise ValueError("full V* P2A does not exactly preserve the 171/191 P0 anchor")
    return {
        "launch_fingerprints": fingerprints,
        "report": report,
        "gate_0_6_disabled_anchor": "171/191",
        "p2a_emitted_preservation": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for benchmark, flag in (
        ("vstar", "vstar"), ("hr-bench_4k", "hr4"), ("hr-bench_8k", "hr8"),
    ):
        parser.add_argument(f"--{flag}-disabled", required=True, type=Path)
        parser.add_argument(f"--{flag}-enabled", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vstar-full-disabled", type=Path)
    parser.add_argument("--vstar-full-enabled", type=Path)
    args = parser.parse_args(argv)
    if (args.vstar_full_disabled is None) != (args.vstar_full_enabled is None):
        parser.error("full V* disabled/enabled paths must be supplied together")
    report = score_dev_paths({
        "vstar": (args.vstar_disabled, args.vstar_enabled),
        "hr-bench_4k": (args.hr4_disabled, args.hr4_enabled),
        "hr-bench_8k": (args.hr8_disabled, args.hr8_enabled),
    })
    if args.vstar_full_disabled is not None:
        report["full_vstar"] = score_full_vstar_paths(
            args.vstar_full_disabled, args.vstar_full_enabled,
        )
        report["gate"]["full_vstar_preserved"] = True
    else:
        report["gate"]["full_vstar_preserved"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
