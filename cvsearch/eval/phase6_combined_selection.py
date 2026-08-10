"""Strict label-blind validation and selection for combined ZOOM/EXPAND runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cvsearch.eval.phase3_zoom_oracle as phase3
import cvsearch.eval.phase4_expand_oracle as phase4
import cvsearch.eval.phase5_unified_selector as phase5
from cvsearch.eval.phase5_unified_selector import SelectionDecision, select_unified_state
from cvsearch.evidence_gap.method import load_method_config
from cvsearch.evidence_gap.provenance import canonical_sha256


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "reproduction/evidence_gap/configs"
DISABLED_CONFIG = CONFIG_ROOT / (
    "dev_unified_zoom_expand_disabled_gamma000_budget512.json"
)
ENABLED_CONFIG = CONFIG_ROOT / (
    "dev_unified_zoom_expand_observation_gamma000_budget512.json"
)
SELECTOR_ID = "phase5-action-stability-ege025-zgt050-v2"
SELECTOR_SOURCE_SHA256 = (
    "a51ce9fbc358115b5cbd0adc6fa978333218fa365ce2a2dc0e94ec1a4b24eeae"
)
REVIEWED_ADMISSION_RULES = (
    phase5.ActionAdmissionRule("EXPAND", ">=", 0.25),
    phase5.ActionAdmissionRule("ZOOM", ">", 0.50),
)
TIE_ORDER = ("EXPAND", "ZOOM")
_FROZEN_COMBINED_INFERENCE_REVISION = (
    "6475b02d1abcf8fc55ee7ac5854d6857b159dcb7ba659b33c156a03377be3457"
)
_SUPPORTED_BENCHMARKS = frozenset({"vstar", "hr-bench_4k", "hr-bench_8k"})
_INPUT_FIELDS = ("input_image", "question", "options", "answer_type")


def _strict_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _snapshot(value: Any) -> str:
    return _strict_json(value)


@dataclass(frozen=True)
class CombinedLaunchProvenance:
    benchmark: str
    revision: str
    disabled_run_fingerprint: str
    enabled_run_fingerprint: str
    disabled_manifest_sha256: str
    enabled_manifest_sha256: str
    partition_rows_sha256: str
    partition_ordinals: tuple[int, ...]
    gpu_uuid: str


@dataclass(frozen=True)
class ExtractedCombinedPair:
    ordinal: int
    provenance: CombinedLaunchProvenance
    _input_identity_json: str
    _p0_json: str
    _candidate_jsons: tuple[str, str]

    @property
    def input_identity(self) -> dict[str, Any]:
        return json.loads(self._input_identity_json)

    @property
    def p0(self) -> dict[str, Any]:
        return json.loads(self._p0_json)

    @property
    def candidates(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return json.loads(self._candidate_jsons[0]), json.loads(self._candidate_jsons[1])

    @property
    def extracted_digest(self) -> str:
        return canonical_sha256({
            "ordinal": self.ordinal,
            "provenance": asdict(self.provenance),
            "input_identity": self.input_identity,
            "p0": self.p0,
            "candidates": list(self.candidates),
        })


@dataclass(frozen=True)
class FrozenCombinedDecision:
    ordinal: int
    provenance: CombinedLaunchProvenance
    extracted_digest: str
    _input_identity_json: str
    _p0_json: str
    _candidate_jsons: tuple[str, str]
    _decision_json: str

    @property
    def input_identity(self) -> dict[str, Any]:
        return json.loads(self._input_identity_json)

    @property
    def p0(self) -> dict[str, Any]:
        return json.loads(self._p0_json)

    @property
    def candidates(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return json.loads(self._candidate_jsons[0]), json.loads(self._candidate_jsons[1])

    @property
    def decision(self) -> SelectionDecision:
        return SelectionDecision(**json.loads(self._decision_json))

    def digest_material(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "provenance": asdict(self.provenance),
            "extracted_digest": self.extracted_digest,
            "input_identity": self.input_identity,
            "p0": self.p0,
            "candidates": list(self.candidates),
            "decision": asdict(self.decision),
        }


@dataclass(frozen=True)
class FrozenCombinedDecisionBatch(Sequence[FrozenCombinedDecision]):
    records: tuple[FrozenCombinedDecision, ...]
    selector_id: str
    selector_source_sha256: str
    admission_rules: tuple[phase5.ActionAdmissionRule, ...]
    tie_order: tuple[str, str]
    canonical_digest: str

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def __iter__(self) -> Iterator[FrozenCombinedDecision]:
        return iter(self.records)

    def digest_material(self) -> dict[str, Any]:
        return {
            "selector_id": self.selector_id,
            "selector_source_sha256": self.selector_source_sha256,
            "admission_rules": [asdict(rule) for rule in self.admission_rules],
            "tie_order": list(self.tie_order),
            "records": [record.digest_material() for record in self.records],
        }

    def recompute_digest(self) -> str:
        return canonical_sha256(self.digest_material())

    def verify_digest(self) -> None:
        if self.recompute_digest() != self.canonical_digest:
            raise ValueError("frozen combined decision digest does not match its records")


def _validate_config_pair(
    disabled: Mapping[str, Any], enabled: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_disabled = load_method_config(DISABLED_CONFIG)
    expected_enabled = load_method_config(ENABLED_CONFIG)
    if dict(disabled) != expected_disabled or dict(enabled) != expected_enabled:
        raise ValueError("combined run config does not match the reviewed siblings")
    changed = {
        key for key in expected_enabled
        if expected_enabled.get(key) != expected_disabled.get(key)
    }
    if changed != {
        "config_id", "p2c_zoom_enabled", "p2c_zoom_admission_mode",
        "p4a_expand_enabled", "p4a_expand_admission_mode",
    }:
        raise ValueError("combined configs violate the five-field sibling contract")
    return expected_disabled, expected_enabled


def validate_combined_launch_pair(
    disabled_rows: Sequence[Mapping[str, Any]],
    enabled_rows: Sequence[Mapping[str, Any]],
    disabled_manifest: Mapping[str, Any],
    enabled_manifest: Mapping[str, Any],
    *, expected_inference_revision: str | None = None,
) -> CombinedLaunchProvenance:
    """Bind one paired combined partition to the reviewed one-pass revision."""
    disabled_manifest = phase4._validate_frozen_launch_manifest(
        disabled_manifest,
        expected_inference_revision=(
            _FROZEN_COMBINED_INFERENCE_REVISION
            if expected_inference_revision is None else expected_inference_revision
        ),
    )
    enabled_manifest = phase4._validate_frozen_launch_manifest(
        enabled_manifest,
        expected_inference_revision=(
            _FROZEN_COMBINED_INFERENCE_REVISION
            if expected_inference_revision is None else expected_inference_revision
        ),
    )
    if disabled_manifest.get("benchmark") not in _SUPPORTED_BENCHMARKS:
        raise ValueError("combined launch benchmark is unsupported")
    if enabled_manifest.get("benchmark") != disabled_manifest.get("benchmark"):
        raise ValueError("combined launch benchmarks differ")
    disabled_config = phase3._mapping(
        disabled_manifest.get("config"), "disabled combined manifest config",
    )
    enabled_config = phase3._mapping(
        enabled_manifest.get("config"), "enabled combined manifest config",
    )
    _validate_config_pair(
        phase3._mapping(disabled_config.get("loaded"), "disabled combined config"),
        phase3._mapping(enabled_config.get("loaded"), "enabled combined config"),
    )
    if phase3._launch_model_contract(disabled_manifest) != phase3._launch_model_contract(
        enabled_manifest
    ):
        raise ValueError("combined launch model contracts differ")
    left_identity = dict(disabled_manifest)
    right_identity = dict(enabled_manifest)
    left_identity.pop("config")
    right_identity.pop("config")
    if left_identity != right_identity:
        raise ValueError("combined launch manifests differ outside reviewed configs")

    ordinals = [row.get("_eg_ordinal") for row in disabled_rows]
    if not ordinals or ordinals != sorted(set(ordinals)):
        raise ValueError("combined disabled ordinals must be nonempty, unique, and sorted")
    if ordinals != [row.get("_eg_ordinal") for row in enabled_rows]:
        raise ValueError("combined launch row ordinals differ")
    partition = phase3._mapping(
        disabled_manifest.get("selected_partition"), "combined selected partition",
    )
    if partition.get("ordinals") != ordinals or partition.get("rows") != len(ordinals):
        raise ValueError("combined launch partition differs from JSONL rows")
    code = phase3._mapping(disabled_manifest.get("code"), "combined launch code")
    revision = phase3._sha256(code.get("revision"), "combined launch revision")
    for row in tuple(disabled_rows) + tuple(enabled_rows):
        if row.get("_eg_code_revision") != revision:
            raise ValueError("combined row revision differs from launch revision")
    disabled_fingerprint = canonical_sha256(disabled_manifest)
    enabled_fingerprint = canonical_sha256(enabled_manifest)
    if any(
        row.get("_eg_run_fingerprint") != disabled_fingerprint
        for row in disabled_rows
    ):
        raise ValueError("combined disabled fingerprint differs from its manifest")
    if any(
        row.get("_eg_run_fingerprint") != enabled_fingerprint
        for row in enabled_rows
    ):
        raise ValueError("combined enabled fingerprint differs from its manifest")
    gpu_uuids = phase3._list(
        phase3._mapping(disabled_manifest.get("hardware"), "combined hardware").get(
            "gpu_uuids"
        ),
        "combined GPU UUIDs",
    )
    return CombinedLaunchProvenance(
        benchmark=disabled_manifest["benchmark"],
        revision=revision,
        disabled_run_fingerprint=disabled_fingerprint,
        enabled_run_fingerprint=enabled_fingerprint,
        disabled_manifest_sha256=canonical_sha256(disabled_manifest),
        enabled_manifest_sha256=canonical_sha256(enabled_manifest),
        partition_rows_sha256=phase3._sha256(
            partition.get("rows_sha256"), "combined partition row digest",
        ),
        partition_ordinals=tuple(ordinals),
        gpu_uuid=gpu_uuids[0],
    )


def _validate_enabled_trace(
    trace: Mapping[str, Any], *, config: Mapping[str, Any], benchmark: str,
    options: Sequence[str], output: Any, context: str,
) -> tuple[dict[str, int], float, Mapping[str, Any], list[Mapping[str, Any]]]:
    if frozenset(trace) != phase4._TRACE_FIELDS:
        raise ValueError(f"{context} MethodTrace has an invalid exact schema")
    if trace.get("effective_config") != config or trace.get("config_id") != config["config_id"]:
        raise ValueError(f"{context} config identity mismatch")
    if trace.get("termination") != "FORCED_RETURN" or trace.get("budget_interrupted") is not False:
        raise ValueError(f"{context} must terminate by uninterrupted forced return")
    if trace.get("pixel_accounting") != config["pixel_accounting"]:
        raise ValueError(f"{context} pixel accounting mismatch")
    elapsed = phase3._finite(trace.get("elapsed_seconds"), f"{context}.elapsed", minimum=0.0)
    budget = phase3._budget(trace.get("budget"), f"{context}.budget")
    final_answer = phase4._validate_answer_record(
        benchmark, options, output, trace.get("final_answer"),
    )
    anchor_answer = phase4._validate_answer_record(
        benchmark, options, output, trace.get("anchor_answer"),
    )
    if final_answer != anchor_answer or phase3._answer_output(final_answer, context) != output:
        raise ValueError(f"{context} final/anchor answer differs from emitted P0")
    steps = phase3._list(trace.get("steps"), f"{context}.steps")
    if (
        len(steps) != 3
        or not all(isinstance(step, Mapping) for step in steps)
        or [step.get("action") for step in steps] != ["ZOOM", "EXPAND", "FORCED_RETURN"]
        or [step.get("step") for step in steps] != [0, 1, 2]
    ):
        raise ValueError("combined trace must be exactly ZOOM, EXPAND, FORCED_RETURN")
    if any(trace.get(name) is not None for name in (
        "anchor_state_score", "selected_state_score", "replacement_margin",
    )):
        raise ValueError("combined selector/replacement fields must remain disabled")
    phase4._validate_forced_step(
        steps[2], index=2, answer=final_answer, budget=budget,
    )
    return budget, elapsed, final_answer, steps


def _confidence(stability: Mapping[str, Any], context: str) -> float:
    value = stability.get("confidence")
    confidence = phase3._finite(value, context, minimum=0.0)
    if confidence > 1.0:
        raise ValueError(f"{context} must be in [0, 1]")
    return confidence


def _candidate_dto(action: str, audit: Mapping[str, Any]) -> dict[str, Any]:
    feasible = audit.get("feasible")
    if type(feasible) is not bool:
        raise TypeError(f"{action} feasibility must be an exact bool")
    stability = audit.get("candidate_stability")
    if not feasible:
        if stability is not None:
            raise ValueError(f"infeasible {action} cannot expose candidate stability")
        return {
            "action": action, "feasible": False,
            "output": None, "candidate_stability": None,
        }
    stability = phase3._mapping(stability, f"{action} candidate stability")
    return {
        "action": action,
        "feasible": True,
        "output": json.loads(_snapshot(stability.get("output"))),
        "candidate_stability": {
            "confidence": _confidence(stability, f"{action} candidate confidence"),
        },
    }


def validate_and_extract_combined_pairs(
    benchmark: str,
    disabled_rows: Sequence[Mapping[str, Any]],
    enabled_rows: Sequence[Mapping[str, Any]],
    *, disabled_launch_manifest: Mapping[str, Any],
    enabled_launch_manifest: Mapping[str, Any],
    expected_inference_revision: str | None = None,
) -> tuple[ExtractedCombinedPair, ...]:
    """Validate one complete raw partition and extract exact selector DTOs."""
    if benchmark not in _SUPPORTED_BENCHMARKS:
        raise ValueError("combined selector supports only V* and HR-4K/8K")
    provenance = validate_combined_launch_pair(
        disabled_rows, enabled_rows,
        disabled_launch_manifest, enabled_launch_manifest,
        expected_inference_revision=expected_inference_revision,
    )
    if provenance.benchmark != benchmark:
        raise ValueError("requested benchmark differs from combined launch")
    disabled_config, enabled_config = _validate_config_pair(
        phase3._mapping(
            phase3._mapping(disabled_launch_manifest.get("config"), "disabled config").get(
                "loaded"
            ),
            "disabled loaded config",
        ),
        phase3._mapping(
            phase3._mapping(enabled_launch_manifest.get("config"), "enabled config").get(
                "loaded"
            ),
            "enabled loaded config",
        ),
    )
    model_contract = phase3._launch_model_contract(enabled_launch_manifest)
    result = []
    for disabled, enabled in zip(disabled_rows, enabled_rows):
        ordinal = disabled.get("_eg_ordinal")
        for field in _INPUT_FIELDS:
            if disabled.get(field) != enabled.get(field):
                raise ValueError(f"combined paired rows differ in exact {field}")
        expected_type = "logits_match" if benchmark == "vstar" else "option_list"
        if disabled.get("answer_type") != expected_type:
            raise ValueError("combined benchmark and answer type disagree")
        output = disabled.get("output")
        if enabled.get("output") != output:
            raise ValueError("combined enabled public output differs from disabled P0")
        disabled_trace = phase3._mapping(disabled.get("method_trace"), "disabled trace")
        enabled_trace = phase3._mapping(enabled.get("method_trace"), "combined trace")
        for field in phase4._SHARED_P0_TRACE_FIELDS:
            if disabled_trace.get(field) != enabled_trace.get(field):
                raise ValueError(f"combined paired P0 traces differ in exact {field}")
        options = phase3._list(enabled.get("options"), "combined options")
        disabled_budget, _, disabled_answer, _ = phase4._validate_common_trace(
            disabled_trace, config=disabled_config, enabled=False,
            benchmark=benchmark, options=options, output=output,
            context=f"disabled row {ordinal}",
        )
        enabled_budget, _, enabled_answer, steps = _validate_enabled_trace(
            enabled_trace, config=enabled_config, benchmark=benchmark,
            options=options, output=output, context=f"combined row {ordinal}",
        )
        if enabled_answer != disabled_answer:
            raise ValueError("combined exact P0 answer differs from disabled sibling")
        if any(step.get("answer") != enabled_answer for step in steps[:2]):
            raise ValueError("combined action answers must remain exact P0")
        requirements, invalid_requirements = phase4._query_requirements(
            enabled_trace, enabled.get("question"),
        )
        zoom_after = phase3._budget(steps[0].get("budget"), "combined ZOOM budget")
        phase3._validate_zoom_step(
            benchmark, enabled, enabled_trace, steps[0], disabled_budget,
            zoom_after, enabled_answer, enabled_config, requirements,
            invalid_requirements, model_contract, step_index=0,
            check_trace_support_status=False,
        )
        source_image, _ = phase4._resolve_source_image(
            enabled, enabled_launch_manifest,
        )
        phase4._validate_expand_step(
            benchmark, enabled, enabled_trace, steps[1], zoom_after,
            enabled_budget, enabled_answer, enabled_config, requirements,
            invalid_requirements, model_contract, source_image,
            step_index=1, check_trace_support_status=True,
        )
        zoom_audit = phase3._mapping(steps[0].get("zoom_audit"), "combined ZOOM audit")
        expand_audit = phase3._mapping(
            steps[1].get("expand_audit"), "combined EXPAND audit",
        )
        if (
            zoom_audit.get("p0_anchor") != expand_audit.get("p0_anchor")
            or zoom_audit.get("p0_stability") != expand_audit.get("p0_stability")
        ):
            raise ValueError("combined actions do not share one exact P0 anchor/stability")
        p0_stability = phase3._mapping(
            zoom_audit.get("p0_stability"), "combined P0 stability",
        )
        if p0_stability.get("output") != output:
            raise ValueError("combined canonical P0 stability output differs from P0")
        p0 = {
            "action": "P0",
            "output": json.loads(_snapshot(output)),
            "p0_stability": {
                "confidence": _confidence(p0_stability, "combined P0 confidence"),
            },
        }
        candidates = (
            _candidate_dto("ZOOM", zoom_audit),
            _candidate_dto("EXPAND", expand_audit),
        )
        input_identity = {
            "ordinal": ordinal,
            **{
                field: json.loads(_snapshot(enabled.get(field)))
                for field in _INPUT_FIELDS
            },
        }
        input_identity_json = _snapshot(input_identity)
        p0_json = _snapshot(p0)
        candidate_jsons = tuple(_snapshot(item) for item in candidates)
        result.append(ExtractedCombinedPair(
            ordinal=ordinal,
            provenance=provenance,
            _input_identity_json=input_identity_json,
            _p0_json=p0_json,
            _candidate_jsons=candidate_jsons,
        ))
    return tuple(result)


def freeze_combined_decisions(
    benchmark: str,
    disabled_rows: Sequence[Mapping[str, Any]],
    enabled_rows: Sequence[Mapping[str, Any]],
    *, disabled_launch_manifest: Mapping[str, Any],
    enabled_launch_manifest: Mapping[str, Any],
    expected_inference_revision: str | None = None,
) -> FrozenCombinedDecisionBatch:
    """Validate raw rows and immediately freeze every label-blind decision."""
    pairs = validate_and_extract_combined_pairs(
        benchmark, disabled_rows, enabled_rows,
        disabled_launch_manifest=disabled_launch_manifest,
        enabled_launch_manifest=enabled_launch_manifest,
        expected_inference_revision=expected_inference_revision,
    )
    if phase5.ACTION_ADMISSION_RULES != REVIEWED_ADMISSION_RULES:
        raise RuntimeError("Phase-5 selector admission rules changed")
    if tuple(phase5._ACTION_NAME_ORDER) != TIE_ORDER:
        raise RuntimeError("Phase-5 selector tie order changed")
    selector_source = hashlib.sha256(Path(phase5.__file__).read_bytes()).hexdigest()
    if selector_source != SELECTOR_SOURCE_SHA256:
        raise RuntimeError("Phase-5 selector source changed")
    if not pairs:
        raise ValueError("combined decision partition must not be empty")
    provenance = pairs[0].provenance
    if (
        any(pair.provenance != provenance for pair in pairs)
        or tuple(pair.ordinal for pair in pairs) != provenance.partition_ordinals
    ):
        raise ValueError("combined decisions must cover one exact validated partition")
    records = []
    previous = -1
    for pair in pairs:
        if pair.ordinal <= previous:
            raise ValueError("combined decision ordinals must be increasing")
        previous = pair.ordinal
        decision = select_unified_state(pair.p0, list(pair.candidates))
        records.append(FrozenCombinedDecision(
            ordinal=pair.ordinal,
            provenance=pair.provenance,
            extracted_digest=pair.extracted_digest,
            _input_identity_json=pair._input_identity_json,
            _p0_json=pair._p0_json,
            _candidate_jsons=pair._candidate_jsons,
            _decision_json=_snapshot(asdict(decision)),
        ))
    records_tuple = tuple(records)
    material = {
        "selector_id": SELECTOR_ID,
        "selector_source_sha256": selector_source,
        "admission_rules": [asdict(rule) for rule in REVIEWED_ADMISSION_RULES],
        "tie_order": list(TIE_ORDER),
        "records": [record.digest_material() for record in records_tuple],
    }
    batch = FrozenCombinedDecisionBatch(
        records=records_tuple,
        selector_id=SELECTOR_ID,
        selector_source_sha256=selector_source,
        admission_rules=REVIEWED_ADMISSION_RULES,
        tie_order=TIE_ORDER,
        canonical_digest=canonical_sha256(material),
    )
    batch.verify_digest()
    return batch
