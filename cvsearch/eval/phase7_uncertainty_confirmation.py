"""Pure label-blind admission for deferred search and confirmed actions."""

from __future__ import annotations

import math
import re
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cvsearch.eval.phase3_zoom_oracle as phase3
import cvsearch.eval.phase4_expand_oracle as phase4
from cvsearch.evidence_gap.answers import aggregate_hr_answers, aggregate_vstar_losses
from cvsearch.evidence_gap.method import build_query_plan, load_method_config
from cvsearch.evidence_gap.provenance import canonical_sha256


SEARCH_STABILITY_GAIN_THRESHOLD = 0.75
CONFIRMATION_THRESHOLDS = frozenset({0.25, 0.50, 0.75})
_ACTION_NAMES = frozenset({"EXPAND", "ZOOM"})
_P0_FIELDS = frozenset({"action", "output", "p0_stability"})
_CANDIDATE_FIELDS = frozenset({
    "action", "feasible", "output", "stability", "view_sha256",
})
_CONFIRMATION_FIELDS = frozenset({
    "action", "feasible", "output", "stability", "aggregate_stability",
    "view_sha256", "prompt_sha256",
})
_FORBIDDEN_KEY_FRAGMENTS = (
    "answer", "benchmark", "category", "correct", "evaluator",
    "groundtruth", "label", "ordinal", "question", "resolution", "target",
    "truth",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_ROOT = _ROOT / "reproduction/evidence_gap/configs"
BASE_SEARCH_CONFIG = _CONFIG_ROOT / (
    "dev_unified_zoom_expand_disabled_gamma000_budget512.json"
)
DEFERRED_SEARCH_CONFIG = _CONFIG_ROOT / (
    "dev_deferred_search_gate080_gamma000_budget512.json"
)
_SUPPORTED_BENCHMARKS = frozenset({"vstar", "hr-bench_4k", "hr-bench_8k"})
_INPUT_FIELDS = ("input_image", "question", "options", "answer_type")
_HISTORY_FIELDS = frozenset({
    "step", "answer", "support_avg", "support_min", "cost",
    "has_unvisited_branch", "state",
})


@dataclass(frozen=True)
class ConfirmationDecision:
    action: str
    status: str
    output: Any
    stability_gain: float | None


@dataclass(frozen=True)
class SearchLaunchProvenance:
    benchmark: str
    revision: str
    base_run_fingerprint: str
    search_run_fingerprint: str
    base_manifest_sha256: str
    search_manifest_sha256: str
    partition_rows_sha256: str
    partition_ordinals: tuple[int, ...]
    gpu_uuid: str


@dataclass(frozen=True)
class ExtractedSearchPair:
    ordinal: int
    provenance: SearchLaunchProvenance
    _input_identity_json: str
    _p0_json: str
    _candidate_json: str

    @property
    def input_identity(self) -> dict[str, Any]:
        return json.loads(self._input_identity_json)

    @property
    def p0(self) -> dict[str, Any]:
        return json.loads(self._p0_json)

    @property
    def candidate(self) -> dict[str, Any]:
        return json.loads(self._candidate_json)

    @property
    def extracted_digest(self) -> str:
        return canonical_sha256({
            "ordinal": self.ordinal,
            "provenance": asdict(self.provenance),
            "input_identity": self.input_identity,
            "p0": self.p0,
            "candidate": self.candidate,
        })


@dataclass(frozen=True)
class FrozenSearchDecision:
    ordinal: int
    provenance: SearchLaunchProvenance
    extracted_digest: str
    _input_identity_json: str
    _p0_json: str
    _candidate_json: str
    _decision_json: str

    @property
    def input_identity(self) -> dict[str, Any]:
        return json.loads(self._input_identity_json)

    @property
    def p0(self) -> dict[str, Any]:
        return json.loads(self._p0_json)

    @property
    def candidate(self) -> dict[str, Any]:
        return json.loads(self._candidate_json)

    @property
    def decision(self) -> ConfirmationDecision:
        return ConfirmationDecision(**json.loads(self._decision_json))

    def digest_material(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "provenance": asdict(self.provenance),
            "extracted_digest": self.extracted_digest,
            "input_identity": self.input_identity,
            "p0": self.p0,
            "candidate": self.candidate,
            "decision": asdict(self.decision),
        }


@dataclass(frozen=True)
class FrozenSearchDecisionBatch(Sequence[FrozenSearchDecision]):
    records: tuple[FrozenSearchDecision, ...]
    selector_source_sha256: str
    threshold: float
    canonical_digest: str

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def __iter__(self) -> Iterator[FrozenSearchDecision]:
        return iter(self.records)

    def digest_material(self) -> dict[str, Any]:
        return {
            "selector_source_sha256": self.selector_source_sha256,
            "threshold": self.threshold,
            "records": [record.digest_material() for record in self.records],
        }

    def recompute_digest(self) -> str:
        return canonical_sha256(self.digest_material())

    def verify_digest(self) -> None:
        if self.recompute_digest() != self.canonical_digest:
            raise ValueError("frozen SEARCH decision digest does not match its records")


def _snapshot_json(value: Any, active: set[int] | None = None) -> Any:
    active = set() if active is None else active
    value_type = type(value)
    if value_type is dict:
        identity = id(value)
        if identity in active:
            raise ValueError("confirmation DTO must not contain a reference cycle")
        active.add(identity)
        try:
            result = {}
            for key, nested in value.items():
                if type(key) is not str:
                    raise TypeError("confirmation DTO keys must be exact strings")
                joined = re.sub(r"[^a-z0-9]", "", key.casefold())
                if any(fragment in joined for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                    raise ValueError("confirmation DTO contains forbidden metadata")
                result[key] = _snapshot_json(nested, active)
            return result
        finally:
            active.remove(identity)
    if value_type is list:
        identity = id(value)
        if identity in active:
            raise ValueError("confirmation DTO must not contain a reference cycle")
        active.add(identity)
        try:
            return [_snapshot_json(item, active) for item in value]
        finally:
            active.remove(identity)
    if value is None or value_type in (str, int, bool):
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("confirmation DTO numbers must be finite")
        return value
    raise TypeError("confirmation DTO must contain only exact JSON builtins")


def _exact_mapping(value: Any, fields: frozenset[str], name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact dict")
    if set(value) != fields:
        raise ValueError(f"{name} has an invalid exact schema")
    return value


def _exact_snapshot(
    value: Any, fields: frozenset[str], name: str,
) -> dict[str, Any]:
    _exact_mapping(value, fields, name)
    return _snapshot_json(value)


def _confidence(value: Any, name: str) -> float:
    stability = _exact_mapping(value, frozenset({"confidence"}), name)
    confidence = stability["confidence"]
    if (
        type(confidence) not in (int, float)
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError(f"{name} confidence must be finite and in [0, 1]")
    return float(confidence)


def _sha256(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _p0(value: Any) -> tuple[dict[str, Any], float]:
    p0 = _exact_snapshot(value, _P0_FIELDS, "P0 state")
    if p0["action"] != "P0" or p0["output"] is None:
        raise ValueError("P0 state is not canonical")
    return p0, _confidence(p0["p0_stability"], "P0 stability")


def _candidate(value: Any, allowed_actions: frozenset[str]) -> tuple[dict[str, Any], float | None]:
    candidate = _exact_snapshot(value, _CANDIDATE_FIELDS, "candidate state")
    if candidate["action"] not in allowed_actions:
        raise ValueError("candidate action is not canonical")
    if type(candidate["feasible"]) is not bool:
        raise TypeError("candidate feasible must be an exact bool")
    if not candidate["feasible"]:
        if any(candidate[key] is not None for key in ("output", "stability", "view_sha256")):
            raise ValueError("infeasible candidate must have null observation fields")
        return candidate, None
    if candidate["output"] is None:
        raise ValueError("feasible candidate must contain an output")
    _sha256(candidate["view_sha256"], "candidate view")
    return candidate, _confidence(candidate["stability"], "candidate stability")


def _confirmation(value: Any, action: str) -> tuple[dict[str, Any], float | None]:
    confirmation = _exact_snapshot(
        value, _CONFIRMATION_FIELDS, "confirmation state",
    )
    if confirmation["action"] != action:
        raise ValueError("confirmation action differs from its candidate")
    if type(confirmation["feasible"]) is not bool:
        raise TypeError("confirmation feasible must be an exact bool")
    if not confirmation["feasible"]:
        if any(confirmation[key] is not None for key in (
            "output", "stability", "aggregate_stability", "view_sha256",
            "prompt_sha256",
        )):
            raise ValueError("infeasible confirmation must have null observation fields")
        return confirmation, None
    if confirmation["output"] is None:
        raise ValueError("feasible confirmation must contain an output")
    _confidence(confirmation["stability"], "confirmation stability")
    aggregate = _confidence(
        confirmation["aggregate_stability"], "aggregate stability",
    )
    _sha256(confirmation["view_sha256"], "confirmation view")
    prompts = confirmation["prompt_sha256"]
    if type(prompts) is not list or not prompts:
        raise ValueError("confirmation prompt hashes must be a nonempty list")
    for index, prompt in enumerate(prompts):
        _sha256(prompt, f"confirmation prompt {index}")
    if len(set(prompts)) != len(prompts):
        raise ValueError("confirmation prompt hashes must be distinct")
    return confirmation, aggregate


def _retained(p0: dict[str, Any]) -> ConfirmationDecision:
    return ConfirmationDecision(
        action="P0", status="retained_p0", output=p0["output"],
        stability_gain=None,
    )


def confirm_search_candidate(p0: dict[str, Any], candidate: dict[str, Any]) -> ConfirmationDecision:
    """Admit a paired gate-0.8 SEARCH state only at the frozen high-gain boundary."""
    p0, p0_confidence = _p0(p0)
    candidate, candidate_confidence = _candidate(candidate, frozenset({"SEARCH"}))
    if not candidate["feasible"]:
        return _retained(p0)
    if candidate["output"] == p0["output"]:
        return _retained(p0)
    if candidate_confidence is None:
        raise AssertionError("feasible candidate lost its stability")
    gain = candidate_confidence - p0_confidence
    if gain < SEARCH_STABILITY_GAIN_THRESHOLD:
        return _retained(p0)
    return ConfirmationDecision(
        action="SEARCH", status="selected_deferred_search",
        output=candidate["output"], stability_gain=gain,
    )


def confirm_action_candidate(
    p0: dict[str, Any], candidate: dict[str, Any], confirmation: dict[str, Any],
    *, threshold: float,
) -> ConfirmationDecision:
    """Admit one uncertain action only after a distinct view confirms it."""
    if type(threshold) not in (int, float) or float(threshold) not in CONFIRMATION_THRESHOLDS:
        raise ValueError("confirmation threshold must be one frozen coarse value")
    threshold = float(threshold)
    p0, p0_confidence = _p0(p0)
    candidate, _ = _candidate(candidate, _ACTION_NAMES)
    confirmation, aggregate_confidence = _confirmation(
        confirmation, candidate["action"],
    )
    if not candidate["feasible"] or not confirmation["feasible"]:
        return _retained(p0)
    if candidate["output"] == p0["output"]:
        return _retained(p0)
    if candidate["output"] != confirmation["output"]:
        return _retained(p0)
    if candidate["view_sha256"] == confirmation["view_sha256"]:
        return _retained(p0)
    if aggregate_confidence is None:
        raise AssertionError("feasible confirmation lost aggregate stability")
    gain = aggregate_confidence - p0_confidence
    if aggregate_confidence < threshold or gain <= 0.0:
        return _retained(p0)
    return ConfirmationDecision(
        action=candidate["action"], status="selected_confirmed_action",
        output=confirmation["output"], stability_gain=gain,
    )


def _strict_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _validate_search_config_pair(
    base: Mapping[str, Any], search: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_base = load_method_config(BASE_SEARCH_CONFIG)
    expected_search = load_method_config(DEFERRED_SEARCH_CONFIG)
    if dict(base) != expected_base or dict(search) != expected_search:
        raise ValueError("SEARCH run config does not match the reviewed siblings")
    if set(expected_base) != set(expected_search):
        raise ValueError("SEARCH sibling config schemas differ")
    changed = {
        key for key in expected_search
        if expected_search[key] != expected_base[key]
    }
    if changed != {"config_id", "quick_gate"}:
        raise ValueError("SEARCH configs violate the two-field sibling contract")
    if expected_base["quick_gate"] != 0.6 or expected_search["quick_gate"] != 0.8:
        raise ValueError("SEARCH quick gates differ from the reviewed boundary")
    return expected_base, expected_search


def validate_search_launch_pair(
    base_rows: Sequence[Mapping[str, Any]],
    search_rows: Sequence[Mapping[str, Any]],
    base_manifest: Mapping[str, Any],
    search_manifest: Mapping[str, Any],
) -> SearchLaunchProvenance:
    """Bind a gate-0.6/gate-0.8 pair before extracting selector inputs."""
    raw_base_config = phase3._mapping(
        phase3._mapping(base_manifest.get("config"), "base SEARCH config").get(
            "loaded"
        ),
        "base SEARCH loaded config",
    )
    raw_search_config = phase3._mapping(
        phase3._mapping(search_manifest.get("config"), "deferred SEARCH config").get(
            "loaded"
        ),
        "deferred SEARCH loaded config",
    )
    _validate_search_config_pair(raw_base_config, raw_search_config)

    raw_code = phase3._mapping(base_manifest.get("code"), "base SEARCH code")
    revision = phase3._sha256(raw_code.get("revision"), "base SEARCH revision")
    base_manifest = phase4._validate_frozen_launch_manifest(
        base_manifest, expected_inference_revision=revision,
    )
    search_manifest = phase4._validate_frozen_launch_manifest(
        search_manifest, expected_inference_revision=revision,
    )
    benchmark = base_manifest.get("benchmark")
    if benchmark not in _SUPPORTED_BENCHMARKS:
        raise ValueError("SEARCH launch benchmark is unsupported")
    if search_manifest.get("benchmark") != benchmark:
        raise ValueError("SEARCH launch benchmarks differ")
    if phase3._launch_model_contract(base_manifest) != phase3._launch_model_contract(
        search_manifest
    ):
        raise ValueError("SEARCH launch model contracts differ")

    base_identity = dict(base_manifest)
    search_identity = dict(search_manifest)
    base_identity.pop("config")
    search_identity.pop("config")
    if base_identity != search_identity:
        raise ValueError("SEARCH launch manifests differ outside reviewed configs")

    ordinals = [row.get("_eg_ordinal") for row in base_rows]
    if not ordinals or ordinals != sorted(set(ordinals)):
        raise ValueError("base SEARCH ordinals must be nonempty, unique, and sorted")
    if ordinals != [row.get("_eg_ordinal") for row in search_rows]:
        raise ValueError("SEARCH paired row ordinals differ")
    partition = phase3._mapping(
        base_manifest.get("selected_partition"), "SEARCH selected partition",
    )
    if partition.get("ordinals") != ordinals or partition.get("rows") != len(ordinals):
        raise ValueError("SEARCH launch partition differs from JSONL rows")
    for row in tuple(base_rows) + tuple(search_rows):
        if row.get("_eg_code_revision") != revision:
            raise ValueError("SEARCH row revision differs from launch revision")
    base_fingerprint = canonical_sha256(base_manifest)
    search_fingerprint = canonical_sha256(search_manifest)
    if base_fingerprint == search_fingerprint:
        raise ValueError("SEARCH sibling run fingerprints must be distinct")
    if any(row.get("_eg_run_fingerprint") != base_fingerprint for row in base_rows):
        raise ValueError("base SEARCH fingerprint differs from its manifest")
    if any(
        row.get("_eg_run_fingerprint") != search_fingerprint
        for row in search_rows
    ):
        raise ValueError("deferred SEARCH fingerprint differs from its manifest")
    gpu_uuids = phase3._list(
        phase3._mapping(base_manifest.get("hardware"), "SEARCH hardware").get(
            "gpu_uuids"
        ),
        "SEARCH GPU UUIDs",
    )
    return SearchLaunchProvenance(
        benchmark=benchmark,
        revision=revision,
        base_run_fingerprint=base_fingerprint,
        search_run_fingerprint=search_fingerprint,
        base_manifest_sha256=canonical_sha256(base_manifest),
        search_manifest_sha256=canonical_sha256(search_manifest),
        partition_rows_sha256=phase3._sha256(
            partition.get("rows_sha256"), "SEARCH partition row digest",
        ),
        partition_ordinals=tuple(ordinals),
        gpu_uuid=gpu_uuids[0],
    )


def _canonical_history_answer(
    benchmark: str, options: Sequence[str], value: Any, phase: str,
) -> dict[str, Any]:
    record = dict(phase3._mapping(value, f"{phase} history answer"))
    if frozenset(record) != phase4._ANSWER_RECORD_FIELDS:
        raise ValueError(f"{phase} history answer has an invalid exact schema")
    if benchmark == "vstar":
        raw_rows = phase3._list(record.get("raw_outputs"), "V* history losses")
        rows = [phase3._list(row, "V* history loss row") for row in raw_rows]
        if len(rows) != 1 or len(rows[0]) != len(options):
            raise ValueError("V* history answer does not cover the exact options")
        expected = aggregate_vstar_losses(rows)
    else:
        raw = phase3._list(record.get("raw_outputs"), "HR history outputs")
        if len(raw) != 4 or not all(isinstance(item, str) for item in raw):
            raise ValueError("HR history answer must contain four raw outputs")
        expected = aggregate_hr_answers(list(options), raw)
        expected.output = json.loads(_strict_json(record.get("output")))
    expected.selected_from = phase
    if expected.to_dict() != record:
        raise ValueError(f"{phase} history answer is not canonical")
    return record


def _validated_history(
    benchmark: str, options: Sequence[str], trace: Mapping[str, Any], context: str,
) -> list[dict[str, Any]]:
    raw_history = phase3._list(trace.get("history"), f"{context} history")
    if len(raw_history) not in (1, 2):
        raise ValueError(f"{context} history must contain root and optional search")
    result = []
    previous_cost = -1
    for index, raw in enumerate(raw_history):
        item = dict(phase3._mapping(raw, f"{context} history[{index}]"))
        if frozenset(item) != _HISTORY_FIELDS or item.get("step") != index:
            raise ValueError(f"{context} history record has an invalid exact schema")
        if (
            phase3._finite(item.get("support_avg"), "history support_avg") != 0.0
            or phase3._finite(item.get("support_min"), "history support_min") != 0.0
            or type(item.get("has_unvisited_branch")) is not bool
            or item["has_unvisited_branch"]
            or type(item.get("state")) is not dict
            or item["state"]
            or isinstance(item.get("cost"), bool)
            or not isinstance(item.get("cost"), int)
            or item["cost"] <= previous_cost
        ):
            raise ValueError(f"{context} history bookkeeping is not canonical")
        previous_cost = item["cost"]
        phase = "root" if index == 0 else "search"
        item["answer"] = _canonical_history_answer(
            benchmark, options, item.get("answer"), phase,
        )
        result.append(item)
    return result


def _answer_confidence(answer: Mapping[str, Any], context: str) -> float:
    value = answer.get("confidence")
    confidence = phase3._finite(value, context, minimum=0.0)
    if confidence > 1.0:
        raise ValueError(f"{context} must be in [0, 1]")
    return confidence


def _validate_query_plan(
    row: Mapping[str, Any], trace: Mapping[str, Any], *, search_observed: bool,
) -> dict[str, Any]:
    policy = {
        field: json.loads(_strict_json(row.get(field)))
        for field in _INPUT_FIELDS
    }
    plan = dict(phase3._mapping(trace.get("query_plan"), "SEARCH query plan"))
    initial = build_query_plan(policy, ()).to_dict()
    if plan == initial:
        return plan
    if not search_observed:
        raise ValueError("SEARCH-free query plan differs from its canonical initial plan")
    targets = plan.get("targets")
    if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
        raise ValueError("post-search query plan targets are invalid")
    expected = build_query_plan(policy, targets).to_dict()
    expected["evidence_items"].append({
        "kind": "runtime_ranking_context",
        "query_source": "main_query_plus_current_visual_cue",
        "planned_augmented_queries_used": False,
    })
    if plan != expected:
        raise ValueError("post-search query plan is not the canonical runtime plan")
    return plan


def validate_and_extract_search_pairs(
    benchmark: str,
    base_rows: Sequence[Mapping[str, Any]],
    search_rows: Sequence[Mapping[str, Any]],
    *, base_launch_manifest: Mapping[str, Any],
    search_launch_manifest: Mapping[str, Any],
) -> tuple[ExtractedSearchPair, ...]:
    """Validate one complete pair and project label-blind SEARCH DTOs."""
    if benchmark not in _SUPPORTED_BENCHMARKS:
        raise ValueError("deferred SEARCH supports only V* and HR-4K/8K")
    provenance = validate_search_launch_pair(
        base_rows, search_rows, base_launch_manifest, search_launch_manifest,
    )
    if provenance.benchmark != benchmark:
        raise ValueError("requested benchmark differs from SEARCH launch")
    base_config, search_config = _validate_search_config_pair(
        phase3._mapping(
            phase3._mapping(base_launch_manifest.get("config"), "base config").get(
                "loaded"
            ),
            "base loaded config",
        ),
        phase3._mapping(
            phase3._mapping(search_launch_manifest.get("config"), "search config").get(
                "loaded"
            ),
            "search loaded config",
        ),
    )
    result = []
    for base, search in zip(base_rows, search_rows):
        ordinal = base.get("_eg_ordinal")
        for field in _INPUT_FIELDS:
            if base.get(field) != search.get(field):
                raise ValueError(f"SEARCH paired rows differ in exact {field}")
        expected_type = "logits_match" if benchmark == "vstar" else "option_list"
        if base.get("answer_type") != expected_type:
            raise ValueError("SEARCH benchmark and answer type disagree")
        options = phase3._list(base.get("options"), "SEARCH options")
        base_trace = phase3._mapping(base.get("method_trace"), "base SEARCH trace")
        search_trace = phase3._mapping(
            search.get("method_trace"), "deferred SEARCH trace",
        )
        _, _, base_answer, _ = phase4._validate_common_trace(
            base_trace, config=base_config, enabled=False,
            benchmark=benchmark, options=options, output=base.get("output"),
            context=f"base SEARCH row {ordinal}",
        )
        _, _, search_answer, _ = phase4._validate_common_trace(
            search_trace, config=search_config, enabled=False,
            benchmark=benchmark, options=options, output=search.get("output"),
            context=f"deferred SEARCH row {ordinal}",
        )
        for field in (
            "candidate_ranks", "method_mode", "root_ans_conf",
            "effective_ranking_query", "pixel_accounting",
        ):
            if base_trace.get(field) != search_trace.get(field):
                raise ValueError(f"SEARCH paired traces differ in exact {field}")
        if base.get("root_ans_conf") != search.get("root_ans_conf"):
            raise ValueError("SEARCH paired rows differ in root confidence")
        if "root_ans_conf" in base and base.get("root_ans_conf") != base_trace.get(
            "root_ans_conf"
        ):
            raise ValueError("SEARCH row and trace root confidence differ")

        base_history = _validated_history(
            benchmark, options, base_trace, f"base SEARCH row {ordinal}",
        )
        search_history = _validated_history(
            benchmark, options, search_trace, f"deferred SEARCH row {ordinal}",
        )
        base_plan = _validate_query_plan(
            base, base_trace, search_observed=len(base_history) == 2,
        )
        search_plan = _validate_query_plan(
            search, search_trace, search_observed=len(search_history) == 2,
        )
        if base_history[0] != search_history[0]:
            raise ValueError("SEARCH paired root observations differ")
        if len(base_history) == 2:
            if (
                search_history != base_history
                or search_plan != base_plan
                or search_answer != base_answer
                or search.get("output") != base.get("output")
                or search_trace.get("final_boxes") != base_trace.get("final_boxes")
                or search_trace.get("budget") != base_trace.get("budget")
            ):
                raise ValueError("existing base SEARCH changed under the deferred gate")
            feasible = False
        elif len(search_history) == 1:
            if search_answer != base_answer or search.get("output") != base.get("output"):
                raise ValueError("SEARCH-free sibling changed its P0 answer")
            feasible = False
        else:
            feasible = True

        p0 = {
            "action": "P0",
            "output": json.loads(_strict_json(base.get("output"))),
            "p0_stability": {
                "confidence": _answer_confidence(
                    base_answer, "base SEARCH confidence",
                ),
            },
        }
        if feasible:
            search_state = {
                "input_image": search.get("input_image"),
                "query_plan": search_trace.get("query_plan"),
                "search_observation": search_history[1],
                "final_boxes": search_trace.get("final_boxes"),
            }
            candidate = {
                "action": "SEARCH", "feasible": True,
                "output": json.loads(_strict_json(search.get("output"))),
                "stability": {
                    "confidence": _answer_confidence(
                        search_answer, "deferred SEARCH confidence",
                    ),
                },
                "view_sha256": canonical_sha256(search_state),
            }
        else:
            candidate = {
                "action": "SEARCH", "feasible": False, "output": None,
                "stability": None, "view_sha256": None,
            }
        input_identity = {
            "ordinal": ordinal,
            **{
                field: json.loads(_strict_json(base.get(field)))
                for field in _INPUT_FIELDS
            },
        }
        result.append(ExtractedSearchPair(
            ordinal=ordinal,
            provenance=provenance,
            _input_identity_json=_strict_json(input_identity),
            _p0_json=_strict_json(p0),
            _candidate_json=_strict_json(candidate),
        ))
    return tuple(result)


def freeze_search_decisions(
    benchmark: str,
    base_rows: Sequence[Mapping[str, Any]],
    search_rows: Sequence[Mapping[str, Any]],
    *, base_launch_manifest: Mapping[str, Any],
    search_launch_manifest: Mapping[str, Any],
) -> FrozenSearchDecisionBatch:
    """Freeze every SEARCH decision before any label-bearing scorer runs."""
    pairs = validate_and_extract_search_pairs(
        benchmark, base_rows, search_rows,
        base_launch_manifest=base_launch_manifest,
        search_launch_manifest=search_launch_manifest,
    )
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records = tuple(FrozenSearchDecision(
        ordinal=pair.ordinal,
        provenance=pair.provenance,
        extracted_digest=pair.extracted_digest,
        _input_identity_json=pair._input_identity_json,
        _p0_json=pair._p0_json,
        _candidate_json=pair._candidate_json,
        _decision_json=_strict_json(asdict(confirm_search_candidate(
            pair.p0, pair.candidate,
        ))),
    ) for pair in pairs)
    material = {
        "selector_source_sha256": source_sha256,
        "threshold": SEARCH_STABILITY_GAIN_THRESHOLD,
        "records": [record.digest_material() for record in records],
    }
    batch = FrozenSearchDecisionBatch(
        records=records,
        selector_source_sha256=source_sha256,
        threshold=SEARCH_STABILITY_GAIN_THRESHOLD,
        canonical_digest=canonical_sha256(material),
    )
    batch.verify_digest()
    return batch
