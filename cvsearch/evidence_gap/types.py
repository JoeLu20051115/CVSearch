"""Pure state contracts for query-aware evidence-gap search."""

import copy
from dataclasses import dataclass, field, is_dataclass, replace as dataclass_replace
import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any


ZOOM = "ZOOM"
SPLIT = "SPLIT"
EXPAND = "EXPAND"
NEXT = "NEXT"
BACKTRACK = "BACKTRACK"
CERTIFIED_STOP = "CERTIFIED_STOP"
FORCED_RETURN = "FORCED_RETURN"
ACTION_VALUES = (ZOOM, SPLIT, EXPAND, NEXT, BACKTRACK, CERTIFIED_STOP, FORCED_RETURN)
EVIDENCE_SUPPORT_TRANSFORM = (
    "v1:p_yes=softmax(final_position_two_logits[Yes,No],dim=-1,"
    "preserve_model_dtype,no_float32_cast)[0];no_legacy_2p_minus_1"
)


@dataclass(frozen=True)
class EvidenceRequirement:
    """One canonical, answer-free visual requirement."""

    requirement_id: str
    kind: str
    text: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) for value in (
            self.requirement_id, self.kind, self.text,
        )):
            raise TypeError("evidence requirement fields must be strings")
        if self.kind not in {
            "target_detail", "relation_context", "coverage", "question_evidence",
        }:
            raise ValueError("evidence requirement kind is not answer-free")
        if not self.text or any(
            unicodedata.category(character).startswith("C") for character in self.text
        ):
            raise ValueError("evidence requirement text must be nonempty and control-free")
        if self.requirement_id != _requirement_id(self.kind, self.text):
            raise ValueError("evidence requirement identity does not match its content")

    def to_dict(self) -> dict[str, str]:
        return {
            "requirement_id": self.requirement_id,
            "kind": self.kind,
            "text": self.text,
        }


def _normalized_nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must be nonempty")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError(f"{name} must not contain Unicode category C characters")
    return normalized


def _requirement_id(kind: str, text: str) -> str:
    payload = json.dumps(
        {"kind": kind, "text": text}, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )
    return "req-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sanitize_evidence_requirements(items: Any) -> tuple[EvidenceRequirement, ...]:
    """Validate the planner's exact schemas and freeze their visible meaning."""
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise TypeError("evidence requirements must be a sequence")
    requirements: list[EvidenceRequirement] = []
    seen: set[str] = set()
    seen_items: set[str] = set()

    def register_item(payload: Mapping[str, Any]) -> None:
        identity = json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        if identity in seen_items:
            raise ValueError("duplicate evidence items are not allowed")
        seen_items.add(identity)

    def add(kind: str, text: str) -> None:
        requirement_id = _requirement_id(kind, text)
        if requirement_id in seen:
            raise ValueError("duplicate evidence requirements are not allowed")
        seen.add(requirement_id)
        requirements.append(EvidenceRequirement(requirement_id, kind, text))

    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError("each evidence item must be a mapping")
        kind = item.get("kind")
        if kind == "target_detail":
            if set(item) != {"kind", "target", "requirements"}:
                raise ValueError("target_detail has an invalid or contaminated schema")
            target = _normalized_nonempty_text(item["target"], "target")
            values = item["requirements"]
            if not isinstance(values, list) or values != ["presence", "visual_detail"]:
                raise ValueError(
                    "target_detail requirements must be exactly ['presence', 'visual_detail']"
                )
            register_item({
                "kind": kind, "target": target,
                "requirements": ["presence", "visual_detail"],
            })
            add(kind, f"presence and visual detail of {target}")
        elif kind == "relation_context":
            if set(item) != {"kind", "targets"}:
                raise ValueError("relation_context has an invalid or contaminated schema")
            values = item["targets"]
            if not isinstance(values, list):
                raise TypeError("relation targets must be a list")
            targets = tuple(_normalized_nonempty_text(value, "relation target") for value in values)
            if not targets:
                raise ValueError("relation targets must be nonempty")
            if len({target.casefold() for target in targets}) != len(targets):
                raise ValueError("relation targets must not contain duplicates")
            register_item({"kind": kind, "targets": list(targets)})
            add(kind, f"relation context among {' and '.join(targets)}")
        elif kind in {"coverage", "question_evidence"}:
            if set(item) != {"kind", "requirement"}:
                raise ValueError(f"{kind} has an invalid or contaminated schema")
            value = _normalized_nonempty_text(item["requirement"], "requirement")
            allowed = "global_scope" if kind == "coverage" else "visual_detail"
            if value != allowed:
                raise ValueError(f"{kind} requirement is not answer-free")
            text = "global scope coverage" if kind == "coverage" else "question visual detail"
            register_item({"kind": kind, "requirement": value})
            add(kind, text)
        elif kind == "runtime_ranking_context":
            if set(item) != {
                "kind", "query_source", "planned_augmented_queries_used",
            }:
                raise ValueError("runtime_ranking_context has an invalid or contaminated schema")
            if item["query_source"] != "main_query_plus_current_visual_cue":
                raise ValueError("runtime query_source does not match the frozen audit value")
            if item["planned_augmented_queries_used"] is not False:
                raise ValueError("planned_augmented_queries_used must remain false")
            register_item({
                "kind": kind,
                "query_source": item["query_source"],
                "planned_augmented_queries_used": False,
            })
        else:
            raise ValueError("unknown evidence requirement kind")
    return tuple(requirements)


@dataclass(frozen=True)
class EvidenceSupportResult:
    """Immutable provenance for one aggregate requirement-set Yes/No forward."""

    requirements: tuple[EvidenceRequirement, ...]
    requirement_set_id: str
    observation_identity: str
    prompt_version: str
    prompt_template_sha256: str
    prompt_sha256: str
    processor_mode: str
    processor_fingerprint_json: str = field(repr=False)
    checkpoint: str
    yes_tokenization: tuple[int, ...]
    no_tokenization: tuple[int, ...]
    yes_token_id: int
    no_token_id: int
    p_yes_transform: str
    yes_logit: float
    no_logit: float
    p_yes: float
    p_no: float
    support_avg: float
    support_min: float
    observation_mode: str
    observation_size: tuple[int, int]
    view_sha256: str
    elapsed_seconds: float
    logical_calls: int = 1
    accounted_pixels: int | None = None
    batch_plan_hash: str | None = None

    @staticmethod
    def requirement_set_id_for(requirements: tuple[EvidenceRequirement, ...]) -> str:
        if not isinstance(requirements, tuple) or not all(
            isinstance(item, EvidenceRequirement) for item in requirements
        ):
            raise TypeError("requirements must be an immutable EvidenceRequirement tuple")
        payload = json.dumps(
            [item.to_dict() for item in requirements], sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        )
        return "reqset-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __post_init__(self) -> None:
        if not self.requirements:
            raise ValueError("support results require at least one requirement")
        if self.requirement_set_id != self.requirement_set_id_for(self.requirements):
            raise ValueError("requirement_set_id does not match requirements")
        for name in (
            "prompt_version", "prompt_template_sha256", "prompt_sha256",
            "processor_mode", "checkpoint", "p_yes_transform", "observation_mode",
            "view_sha256",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        try:
            identity = json.loads(
                self.observation_identity,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite constant {value}")
                ),
            )
            fingerprint = json.loads(
                self.processor_fingerprint_json,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite constant {value}")
                ),
            )
            _json_safe(identity)
            _json_safe(fingerprint)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("support provenance must be strict JSON") from error
        if not isinstance(self.yes_tokenization, tuple) or not isinstance(
            self.no_tokenization, tuple
        ):
            raise TypeError("Yes/No tokenizations must be tuples")
        if self.logical_calls != 1:
            raise ValueError("aggregate support uses exactly one logical call")
        for name in ("yes_logit", "no_logit", "p_yes", "p_no", "elapsed_seconds"):
            value = _finite_number(getattr(self, name), name)
            if name in {"p_yes", "p_no"} and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if abs((self.p_yes + self.p_no) - 1.0) > 1e-5:
            raise ValueError("support probabilities must be normalized")
        if self.support_avg != self.p_yes or self.support_min != self.p_yes:
            raise ValueError("support avg/min are aggregate p_yes aliases")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        if (
            not isinstance(self.observation_size, tuple)
            or len(self.observation_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
                   for value in self.observation_size)
        ):
            raise ValueError("observation_size must contain two positive integers")
        if self.accounted_pixels is not None:
            _integral(self.accounted_pixels, "accounted_pixels")

    @property
    def processor_fingerprint(self) -> Any:
        return json.loads(self.processor_fingerprint_json)

    def with_batch_accounting(
        self, *, accounted_pixels: int, batch_plan_hash: str,
    ) -> "EvidenceSupportResult":
        return dataclass_replace(
            self, accounted_pixels=_integral(accounted_pixels, "accounted_pixels"),
            batch_plan_hash=batch_plan_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirements": [item.to_dict() for item in self.requirements],
            "requirement_order": [item.requirement_id for item in self.requirements],
            "requirement_set_id": self.requirement_set_id,
            "observation_identity": json.loads(self.observation_identity),
            "prompt_version": self.prompt_version,
            "prompt_template_sha256": self.prompt_template_sha256,
            "prompt_sha256": self.prompt_sha256,
            "processor_mode": self.processor_mode,
            "processor_fingerprint": json.loads(self.processor_fingerprint_json),
            "checkpoint": self.checkpoint,
            "yes_tokenization": list(self.yes_tokenization),
            "no_tokenization": list(self.no_tokenization),
            "yes_token_id": self.yes_token_id,
            "no_token_id": self.no_token_id,
            "p_yes_transform": self.p_yes_transform,
            "yes_logit": _json_safe(self.yes_logit),
            "no_logit": _json_safe(self.no_logit),
            "p_yes": _json_safe(self.p_yes),
            "p_no": _json_safe(self.p_no),
            "support_avg": _json_safe(self.support_avg),
            "support_min": _json_safe(self.support_min),
            "aggregate_alias_note": "support_avg=support_min=p_yes; not per-item support",
            "observation_mode": self.observation_mode,
            "observation_size": list(self.observation_size),
            "view_sha256": self.view_sha256,
            "elapsed_seconds": _json_safe(self.elapsed_seconds),
            "logical_calls": self.logical_calls,
            "accounted_pixels": self.accounted_pixels,
            "batch_plan_hash": self.batch_plan_hash,
        }


_ZERO_BATCH_PLAN_FIELDS = frozenset({
    "schema_version", "batch_kind", "answer_type", "source_identity",
    "q0_sha256", "requirement_order", "requirement_set_id", "verifier_status",
    "current_support_calls", "candidate_support_calls", "candidate_answer_calls",
    "candidate_option_count", "pixels_per_logical_forward", "total_calls",
    "total_pixels",
})
_FULL_BATCH_PLAN_FIELDS = _ZERO_BATCH_PLAN_FIELDS | frozenset({
    "source_width", "source_height", "accounted_source_area",
    "current_observation", "candidate_observation", "prompt_version",
    "prompt_template_sha256", "current_prompt_sha256", "candidate_prompt_sha256",
    "processor_mode", "processor_fingerprint", "checkpoint", "yes_tokenization",
    "no_tokenization", "yes_token_id", "no_token_id", "p_yes_transform",
})
_LEDGER_FIELDS = frozenset({
    "max_mllm_calls", "max_processed_pixels", "mllm_calls", "processed_pixels",
})


def _sha256_text(value: Any, name: str, *, prefix: str = "") -> str:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError(f"{name} must be a SHA-256 identifier")
    digest = value[len(prefix):]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a SHA-256 identifier")
    return value


def _batch_observation_identity(value: Any, name: str) -> str:
    if not isinstance(value, dict) or set(value) != {
        "canonical_keys", "renderer_identities", "rendered_mode", "rendered_size",
        "view_sha256", "descriptors",
    }:
        raise ValueError(f"{name} observation has an invalid exact schema")
    canonical_keys = value["canonical_keys"]
    renderer_ids = value["renderer_identities"]
    descriptors = value["descriptors"]
    if not all(isinstance(items, list) for items in (canonical_keys, renderer_ids, descriptors)):
        raise TypeError(f"{name} observation sequences must be lists")
    if len(canonical_keys) != len(renderer_ids) or len(canonical_keys) != len(descriptors):
        raise ValueError(f"{name} observation descriptor counts do not match")
    if not all(isinstance(item, str) and item for item in canonical_keys + renderer_ids):
        raise ValueError(f"{name} observation identities must be nonempty strings")
    for key, renderer_id, descriptor in zip(canonical_keys, renderer_ids, descriptors):
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("canonical_key") != key
            or descriptor.get("renderer_identity") != renderer_id
        ):
            raise ValueError(f"{name} observation descriptor identity does not match")
    mode = value["rendered_mode"]
    size = value["rendered_size"]
    if not isinstance(mode, str) or not mode:
        raise ValueError(f"{name} rendered mode must be nonempty")
    if (
        not isinstance(size, list) or len(size) != 2
        or any(isinstance(item, bool) or not isinstance(item, Integral) or item <= 0 for item in size)
    ):
        raise ValueError(f"{name} rendered size must contain two positive integers")
    _sha256_text(value["view_sha256"], f"{name} view_sha256")
    identity = {
        "canonical_keys": canonical_keys,
        "renderer_identities": renderer_ids,
        "rendered_mode": mode,
        "rendered_size": size,
        "view_sha256": value["view_sha256"],
    }
    return json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _validate_batch_plan(plan: Mapping[str, Any]) -> tuple[dict[str, Any], str, bool]:
    if not isinstance(plan, Mapping):
        raise TypeError("batch plan must be a mapping")
    supplied_hash = plan.get("plan_hash")
    payload = _json_safe(dict(plan))
    payload.pop("plan_hash", None)
    fields = frozenset(payload)
    if fields not in {_ZERO_BATCH_PLAN_FIELDS, _FULL_BATCH_PLAN_FIELDS}:
        raise ValueError("batch plan does not match an exact plan schema")
    is_full = fields == _FULL_BATCH_PLAN_FIELDS
    if payload["schema_version"] != 1 or payload["batch_kind"] != "p2a_post_anchor_next":
        raise ValueError("batch plan version or kind is invalid")
    if payload["answer_type"] not in {"option_list", "logits_match"}:
        raise ValueError("batch plan answer_type is invalid")
    if payload["verifier_status"] != "disabled_same_checkpoint_unpromoted":
        raise ValueError("batch plan verifier status is invalid")
    source = payload["source_identity"]
    if not isinstance(source, dict) or set(source) != {"mode", "size", "pixel_sha256"}:
        raise ValueError("batch plan source identity schema is invalid")
    if source["mode"] != "RGB":
        raise ValueError("batch plan source must be RGB")
    if (
        not isinstance(source["size"], list) or len(source["size"]) != 2
        or any(isinstance(item, bool) or not isinstance(item, Integral) or item <= 0
               for item in source["size"])
    ):
        raise ValueError("batch plan source size is invalid")
    _sha256_text(source["pixel_sha256"], "source pixel_sha256")
    _sha256_text(payload["q0_sha256"], "q0_sha256")
    _sha256_text(payload["requirement_set_id"], "requirement_set_id", prefix="reqset-")
    order = payload["requirement_order"]
    if (
        not isinstance(order, list)
        or not all(isinstance(item, str) and item.startswith("req-") for item in order)
        or len(set(order)) != len(order)
    ):
        raise ValueError("batch plan requirement order is invalid")
    count_names = (
        "current_support_calls", "candidate_support_calls", "candidate_answer_calls",
        "candidate_option_count", "pixels_per_logical_forward", "total_calls",
        "total_pixels",
    )
    counts = {name: _integral(payload[name], name) for name in count_names}
    area = int(source["size"][0] * source["size"][1])
    if counts["pixels_per_logical_forward"] != area:
        raise ValueError("batch plan pixels per logical forward must equal source RGB area")
    if is_full:
        if not order:
            raise ValueError("a charged batch plan requires evidence requirements")
        width = _integral(payload["source_width"], "source_width")
        height = _integral(payload["source_height"], "source_height")
        accounted_area = _integral(payload["accounted_source_area"], "accounted_source_area")
        if [width, height] != source["size"] or accounted_area != area:
            raise ValueError("batch plan source geometry is inconsistent")
        if counts["current_support_calls"] != 1 or counts["candidate_support_calls"] != 1:
            raise ValueError("batch plan must contain exactly two aggregate support calls")
        option_count = counts["candidate_option_count"]
        if payload["answer_type"] == "option_list":
            if option_count != 4 or counts["candidate_answer_calls"] != 4:
                raise ValueError("HR batch plan must contain four ordered answer calls")
        elif option_count <= 0 or counts["candidate_answer_calls"] != 1 + option_count:
            raise ValueError("V* batch plan answer calls must equal one plus option count")
        expected_calls = 2 + counts["candidate_answer_calls"]
        if counts["total_calls"] != expected_calls:
            raise ValueError("batch plan total calls are inconsistent")
        if counts["total_pixels"] != expected_calls * area:
            raise ValueError("batch plan total pixels are inconsistent")
        _batch_observation_identity(payload["current_observation"], "current")
        _batch_observation_identity(payload["candidate_observation"], "candidate")
        for name in (
            "prompt_version", "processor_mode", "checkpoint", "p_yes_transform",
        ):
            if not isinstance(payload[name], str) or not payload[name]:
                raise ValueError(f"batch plan {name} must be nonempty")
        for name in (
            "prompt_template_sha256", "current_prompt_sha256", "candidate_prompt_sha256",
        ):
            _sha256_text(payload[name], name)
        if not isinstance(payload["processor_fingerprint"], dict):
            raise TypeError("batch plan processor fingerprint must be an object")
        if payload["yes_tokenization"] != [9454] or payload["no_tokenization"] != [2753]:
            raise ValueError("batch plan Yes/No tokenizations must be exact frozen single tokens")
        if payload["yes_token_id"] != 9454 or payload["no_token_id"] != 2753:
            raise ValueError("batch plan Yes/No token IDs are invalid")
        if payload["p_yes_transform"] != EVIDENCE_SUPPORT_TRANSFORM:
            raise ValueError("batch plan support transform is not frozen")
    elif any(counts[name] != 0 for name in (
        "current_support_calls", "candidate_support_calls", "candidate_answer_calls",
        "candidate_option_count", "total_calls", "total_pixels",
    )):
        raise ValueError("zero batch plan costs must all be zero")

    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )
    plan_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if supplied_hash is not None and supplied_hash != plan_hash:
        raise ValueError("supplied batch plan hash does not match the exact plan")
    return payload, plan_hash, is_full


def _validate_batch_ledger(value: Mapping[str, Any], name: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _LEDGER_FIELDS:
        raise ValueError(f"{name} ledger has an invalid exact schema")
    result = {field: _integral(value[field], f"{name}.{field}") for field in _LEDGER_FIELDS}
    if result["mllm_calls"] > result["max_mllm_calls"]:
        raise ValueError(f"{name} call ledger exceeds its maximum")
    if result["processed_pixels"] > result["max_processed_pixels"]:
        raise ValueError(f"{name} pixel ledger exceeds its maximum")
    return result


def _validate_plan_support(
    support: EvidenceSupportResult, plan: Mapping[str, Any], plan_hash: str,
    observation_name: str,
) -> None:
    if not isinstance(support, EvidenceSupportResult):
        raise TypeError("batch support must be an EvidenceSupportResult")
    if [item.requirement_id for item in support.requirements] != plan["requirement_order"]:
        raise ValueError("batch support requirement order does not match plan")
    if support.requirement_set_id != plan["requirement_set_id"]:
        raise ValueError("batch support requirement set does not match plan")
    observation = plan[f"{observation_name}_observation"]
    if support.observation_identity != _batch_observation_identity(
        observation, observation_name,
    ):
        raise ValueError("batch support observation identity does not match plan")
    if (
        support.observation_mode != observation["rendered_mode"]
        or list(support.observation_size) != observation["rendered_size"]
        or support.view_sha256 != observation["view_sha256"]
    ):
        raise ValueError("batch support rendered view does not match plan")
    prompt_hash_name = f"{observation_name}_prompt_sha256"
    if (
        support.prompt_version != plan["prompt_version"]
        or support.prompt_template_sha256 != plan["prompt_template_sha256"]
        or support.prompt_sha256 != plan[prompt_hash_name]
        or support.processor_mode != plan["processor_mode"]
        or support.processor_fingerprint != plan["processor_fingerprint"]
        or support.checkpoint != plan["checkpoint"]
        or list(support.yes_tokenization) != plan["yes_tokenization"]
        or list(support.no_tokenization) != plan["no_tokenization"]
        or support.yes_token_id != plan["yes_token_id"]
        or support.no_token_id != plan["no_token_id"]
        or support.p_yes_transform != plan["p_yes_transform"]
    ):
        raise ValueError("batch support prompt or processor provenance does not match plan")
    if support.accounted_pixels != plan["pixels_per_logical_forward"]:
        raise ValueError("batch support accounted pixels do not match plan")
    if support.batch_plan_hash != plan_hash:
        raise ValueError("batch support plan hash does not match exact plan hash")


@dataclass(frozen=True, init=False)
class ObservationBatchResult:
    """Strict snapshot of one all-or-none post-anchor P2A observation plan."""

    status: str
    admitted: bool
    charged: bool
    current_support: EvidenceSupportResult | None
    candidate_support: EvidenceSupportResult | None
    failure_phase: str | None
    failure_reason: str | None
    exception_type: str | None
    executed_stages: tuple[str, ...]
    elapsed_seconds: float
    verifier_status: str
    verifier_avg: None
    verifier_min: None
    promotable: bool
    _batch_plan_json: str = field(repr=False)
    _candidate_answer_json: str | None = field(repr=False)
    _ledger_before_json: str = field(repr=False)
    _ledger_after_json: str = field(repr=False)

    def __init__(
        self, *, status: str, batch_plan: Mapping[str, Any], admitted: bool,
        charged: bool, ledger_before: Mapping[str, Any], ledger_after: Mapping[str, Any],
        current_support: EvidenceSupportResult | None = None,
        candidate_support: EvidenceSupportResult | None = None,
        candidate_answer: Any = None, failure_phase: str | None = None,
        failure_reason: str | None = None, exception_type: str | None = None,
        executed_stages: tuple[str, ...] = (), elapsed_seconds: float = 0.0,
    ) -> None:
        if status not in {"success", "no_requirements", "budget_rejected", "model_failed"}:
            raise ValueError("invalid observation batch status")
        if not isinstance(admitted, bool) or not isinstance(charged, bool):
            raise TypeError("admitted and charged must be booleans")
        expected_state = {
            "success": (True, True),
            "model_failed": (True, True),
            "no_requirements": (False, False),
            "budget_rejected": (False, False),
        }[status]
        if (admitted, charged) != expected_state:
            raise ValueError("status requires a consistent admitted and charged state")
        if status == "success" and candidate_answer is None:
            raise ValueError("successful observation batch requires a candidate answer")
        if status != "success" and candidate_answer is not None:
            raise ValueError("unsuccessful observation batch cannot expose a candidate answer")
        if not isinstance(executed_stages, tuple) or not all(
            isinstance(stage, str) and stage for stage in executed_stages
        ):
            raise TypeError("executed_stages must be a tuple of nonempty strings")
        elapsed = _finite_number(elapsed_seconds, "elapsed_seconds")
        if elapsed < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        plan_without_hash, plan_hash, is_full_plan = _validate_batch_plan(batch_plan)
        if status in {"success", "model_failed"} and not is_full_plan:
            raise ValueError("charged result status requires a full nonzero batch plan")
        if status == "no_requirements" and is_full_plan:
            raise ValueError("no-requirements status requires a zero batch plan")
        if status == "success" and (
            current_support is None or candidate_support is None
        ):
            raise ValueError("successful result requires both matching support records")
        if status in {"no_requirements", "budget_rejected"} and (
            current_support is not None or candidate_support is not None
        ):
            raise ValueError("uncharged result cannot retain support records")
        if candidate_support is not None and current_support is None:
            raise ValueError("candidate support cannot exist without current support")
        if status == "success":
            if failure_phase is not None or failure_reason is not None or exception_type is not None:
                raise ValueError("successful result cannot contain failure provenance")
        else:
            if not isinstance(failure_phase, str) or not failure_phase:
                raise ValueError("unsuccessful result requires a failure phase")
            if not isinstance(failure_reason, str) or not failure_reason:
                raise ValueError("unsuccessful result requires a failure reason")
            if status == "model_failed" and (
                not isinstance(exception_type, str) or not exception_type
            ):
                raise ValueError("model_failed result requires an exception type")

        before = _validate_batch_ledger(ledger_before, "before")
        after = _validate_batch_ledger(ledger_after, "after")
        if (
            before["max_mllm_calls"] != after["max_mllm_calls"]
            or before["max_processed_pixels"] != after["max_processed_pixels"]
        ):
            raise ValueError("batch ledger limits must remain unchanged")
        expected_calls = plan_without_hash["total_calls"] if charged else 0
        expected_pixels = plan_without_hash["total_pixels"] if charged else 0
        if (
            after["mllm_calls"] - before["mllm_calls"] != expected_calls
            or after["processed_pixels"] - before["processed_pixels"] != expected_pixels
        ):
            raise ValueError("batch ledger delta does not match exact plan totals")

        if current_support is not None:
            _validate_plan_support(
                current_support, plan_without_hash, plan_hash, "current",
            )
        if candidate_support is not None:
            _validate_plan_support(
                candidate_support, plan_without_hash, plan_hash, "candidate",
            )

        candidate_value = None if candidate_answer is None else _json_safe(candidate_answer)
        if status == "success" and plan_without_hash["answer_type"] == "option_list":
            if (
                not isinstance(candidate_value, list) or len(candidate_value) != 4
                or not all(isinstance(item, str) for item in candidate_value)
            ):
                raise ValueError("successful HR candidate answer must contain four strings")
        elif status == "success":
            option_count = plan_without_hash["candidate_option_count"]
            if not isinstance(candidate_value, dict) or set(candidate_value) != {"winner", "losses"}:
                raise ValueError("successful V* candidate answer schema is invalid")
            winner = candidate_value["winner"]
            losses = candidate_value["losses"]
            if (
                isinstance(winner, bool) or not isinstance(winner, Integral)
                or not isinstance(losses, list) or len(losses) != option_count
                or not all(isinstance(loss, Real) and not isinstance(loss, bool)
                           and math.isfinite(float(loss)) for loss in losses)
            ):
                raise ValueError("successful V* candidate answer values are invalid")
            if winner != min(range(option_count), key=losses.__getitem__):
                raise ValueError("successful V* winner must equal the loss argmin")

        plan_payload = dict(plan_without_hash, plan_hash=plan_hash)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "admitted", bool(admitted))
        object.__setattr__(self, "charged", bool(charged))
        object.__setattr__(self, "current_support", current_support)
        object.__setattr__(self, "candidate_support", candidate_support)
        object.__setattr__(self, "failure_phase", failure_phase)
        object.__setattr__(self, "failure_reason", failure_reason)
        object.__setattr__(self, "exception_type", exception_type)
        object.__setattr__(self, "executed_stages", executed_stages)
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "verifier_status", "disabled_same_checkpoint_unpromoted")
        object.__setattr__(self, "verifier_avg", None)
        object.__setattr__(self, "verifier_min", None)
        object.__setattr__(self, "promotable", False)
        object.__setattr__(self, "_batch_plan_json", json.dumps(
            plan_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ))
        candidate_snapshot = None if candidate_value is None else json.dumps(
            candidate_value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        object.__setattr__(self, "_candidate_answer_json", candidate_snapshot)
        object.__setattr__(self, "_ledger_before_json", json.dumps(
            before, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ))
        object.__setattr__(self, "_ledger_after_json", json.dumps(
            after, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ))

    @property
    def batch_plan(self) -> dict[str, Any]:
        return json.loads(self._batch_plan_json)

    @property
    def batch_plan_hash(self) -> str:
        return self.batch_plan["plan_hash"]

    @property
    def candidate_answer(self) -> Any:
        if self._candidate_answer_json is None:
            return None
        return json.loads(self._candidate_answer_json)

    def to_dict(self) -> dict[str, Any]:
        def support_payload(value: EvidenceSupportResult | None) -> Any:
            if value is None:
                return None
            return value.to_dict()

        return {
            "status": self.status,
            "admitted": self.admitted,
            "charged": self.charged,
            "batch_plan": self.batch_plan,
            "batch_plan_hash": self.batch_plan_hash,
            "current_support": support_payload(self.current_support),
            "candidate_support": support_payload(self.candidate_support),
            "candidate_answer": self.candidate_answer,
            "failure_phase": self.failure_phase,
            "failure_reason": self.failure_reason,
            "exception_type": self.exception_type,
            "executed_stages": list(self.executed_stages),
            "elapsed_seconds": _json_safe(self.elapsed_seconds),
            "ledger_before": json.loads(self._ledger_before_json),
            "ledger_after": json.loads(self._ledger_after_json),
            "verifier_status": self.verifier_status,
            "verifier_avg": None,
            "verifier_min": None,
            "promotable": False,
        }


class BudgetExceeded(RuntimeError):
    pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if type(value).__module__ == "numpy" and type(value).__name__ == "bool":
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON values must be finite")
        return number
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if is_dataclass(value):
        return _json_safe(value.to_dict())
    raise TypeError(f"not JSON-safe: {type(value).__name__}")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _optional_finite_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, name)


def _integral(value: Any, name: str) -> int:
    if isinstance(value, bool) or type(value).__name__ == "bool":
        raise TypeError(f"{name} must be a finite integer")
    if isinstance(value, Integral):
        integer = int(value)
    else:
        number = _finite_number(value, name)
        if not number.is_integer():
            raise ValueError(f"{name} must be an integer")
        integer = int(number)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def canonical_key(bbox: Any, depth: int, render_level: int) -> str:
    try:
        coordinates = tuple(bbox)
    except TypeError as error:
        raise TypeError("bbox must contain four numeric coordinates") from error
    if len(coordinates) != 4:
        raise ValueError("bbox must contain exactly four coordinates")
    rounded = tuple(int(round(_finite_number(value, "bbox coordinate"))) for value in coordinates)
    canonical_depth = _integral(depth, "depth")
    canonical_render_level = _integral(render_level, "render_level")
    return f"{rounded[0]}:{rounded[1]}:{rounded[2]}:{rounded[3]}:d{canonical_depth}:r{canonical_render_level}"


@dataclass
class QueryPlan:
    main_query: str = ""
    targets: tuple[str, ...] = ()
    augmented_queries: tuple[str, ...] = ()
    evidence_items: tuple[Any, ...] = ()
    global_scope_required: bool = False
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "main_query": _json_safe(self.main_query),
            "targets": _json_safe(self.targets),
            "augmented_queries": _json_safe(self.augmented_queries),
            "evidence_items": _json_safe(self.evidence_items),
            "global_scope_required": _json_safe(self.global_scope_required),
            "fallback_used": _json_safe(self.fallback_used),
        }


@dataclass(frozen=True)
class CandidateScore:
    main: float = 0.0
    augmented: float = 0.0
    complexity: float = 0.0
    edge_density: float = 0.0
    relevance: float = 0.0
    visual: float = 0.0
    rank: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "main": _json_safe(self.main),
            "augmented": _json_safe(self.augmented),
            "complexity": _json_safe(self.complexity),
            "edge_density": _json_safe(self.edge_density),
            "relevance": _json_safe(self.relevance),
            "visual": _json_safe(self.visual),
            "rank": _json_safe(self.rank),
        }


@dataclass
class SearchCandidate:
    key: str = ""
    bbox: tuple[int | float, int | float, int | float, int | float] = (0, 0, 0, 0)
    parent_key: str | None = None
    child_keys: tuple[str, ...] = ()
    depth: int = 0
    source: str = ""
    render_level: int = 0
    score: CandidateScore = field(default_factory=CandidateScore)
    node: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": _json_safe(self.key),
            "bbox": _json_safe(self.bbox),
            "parent_key": _json_safe(self.parent_key),
            "child_keys": _json_safe(self.child_keys),
            "depth": _json_safe(self.depth),
            "source": _json_safe(self.source),
            "render_level": _json_safe(self.render_level),
            "score": self.score.to_dict(),
        }


@dataclass
class BudgetLedger:
    max_mllm_calls: int
    max_processed_pixels: int | float
    mllm_calls: int = 0
    processed_pixels: int | float = 0

    def __post_init__(self) -> None:
        self.max_mllm_calls = self._count(self.max_mllm_calls, "max_mllm_calls")
        self.mllm_calls = self._count(self.mllm_calls, "mllm_calls")
        self.max_processed_pixels = self._nonnegative(self.max_processed_pixels, "max_processed_pixels")
        self.processed_pixels = self._nonnegative(self.processed_pixels, "processed_pixels")
        if self.mllm_calls > self.max_mllm_calls:
            raise BudgetExceeded("mllm_calls already exceeds max_mllm_calls")
        if self.processed_pixels > self.max_processed_pixels:
            raise BudgetExceeded("processed_pixels already exceeds max_processed_pixels")

    @staticmethod
    def _nonnegative(value: int | float, name: str) -> int | float:
        number = _finite_number(value, name)
        if number < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @classmethod
    def _count(cls, value: int | float, name: str) -> int:
        number = cls._nonnegative(value, name)
        if not float(number).is_integer():
            raise ValueError(f"{name} must be an integer count")
        return int(number)

    def consume(self, kind: str, amount: int | float) -> None:
        if kind == "mllm_calls":
            increment = self._count(amount, kind)
            current = self.mllm_calls
            limit = self.max_mllm_calls
        elif kind == "processed_pixels":
            increment = self._nonnegative(amount, kind)
            current = self.processed_pixels
            limit = self.max_processed_pixels
        else:
            raise ValueError(f"unknown budget kind: {kind}")
        if current + increment > limit:
            raise BudgetExceeded(f"{kind}: {current}+{increment}>{limit}")
        if kind == "mllm_calls":
            self.mllm_calls = int(current + increment)
        else:
            self.processed_pixels = current + increment

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_mllm_calls": _json_safe(self.max_mllm_calls),
            "max_processed_pixels": _json_safe(self.max_processed_pixels),
            "mllm_calls": _json_safe(self.mllm_calls),
            "processed_pixels": _json_safe(self.processed_pixels),
        }


@dataclass
class AnswerRecord:
    output: Any = None
    canonical_answer: Any = None
    raw_outputs: tuple[Any, ...] = ()
    groups: dict[str, Any] = field(default_factory=dict)
    frequency: float = 0.0
    margin: float = 0.0
    confidence: float = 0.0
    uncertainty: float = 1.0
    losses: tuple[float, ...] = ()
    selected_from: str = ""
    aggregation_available: bool | None = None
    aggregation_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": _json_safe(self.output),
            "canonical_answer": _json_safe(self.canonical_answer),
            "raw_outputs": _json_safe(self.raw_outputs),
            "groups": _json_safe(self.groups),
            "frequency": _json_safe(self.frequency),
            "margin": _json_safe(self.margin),
            "confidence": _json_safe(self.confidence),
            "uncertainty": _json_safe(self.uncertainty),
            "losses": _json_safe(self.losses),
            "selected_from": _json_safe(self.selected_from),
            "aggregation_available": _json_safe(self.aggregation_available),
            "aggregation_reason": _json_safe(self.aggregation_reason),
        }


@dataclass(frozen=True, init=False)
class P0Anchor:
    """Immutable capture of the exact emitted P0 and its producing view."""

    producing_phase: str
    node_keys: tuple[str, ...]
    _emitted_answer_json: str = field(repr=False)
    _cvsearch_raw_json: str = field(repr=False)
    _support_view: tuple[Any, ...] | None = field(repr=False, compare=False)

    def __init__(
        self,
        emitted_answer: Any,
        cvsearch_raw: Any,
        producing_phase: str,
        node_keys: tuple[str, ...],
        support_view: tuple[Any, ...] | None,
    ) -> None:
        allowed_phases = {
            "quick", "fast", "root", "search", "cvsearch_raw", "cvsearch_anchor", "response",
        }
        if producing_phase not in allowed_phases:
            raise ValueError("producing_phase must name a stable P0 production phase")
        if not isinstance(node_keys, tuple) or not all(
            isinstance(key, str) and key for key in node_keys
        ):
            raise TypeError("node_keys must be a tuple of nonempty canonical keys")
        if len(set(node_keys)) != len(node_keys):
            raise ValueError("node_keys must not contain duplicates")

        emitted_snapshot = json.dumps(
            _json_safe(emitted_answer), ensure_ascii=False, separators=(",", ":"),
            allow_nan=False,
        )
        raw_snapshot = json.dumps(
            _json_safe(cvsearch_raw), ensure_ascii=False, separators=(",", ":"),
            allow_nan=False,
        )

        frozen_view = None
        if support_view is not None:
            if not isinstance(support_view, tuple):
                raise TypeError("support_view must be an immutable tuple or None")
            frozen_view = copy.deepcopy(tuple(support_view))
            view_keys = tuple(getattr(item, "canonical_key", None) for item in frozen_view)
            if view_keys != node_keys:
                raise ValueError("support view keys must exactly match node_keys")
            for item in frozen_view:
                dataclass_parameters = getattr(type(item), "__dataclass_params__", None)
                if (
                    not is_dataclass(item)
                    or dataclass_parameters is None
                    or not dataclass_parameters.frozen
                    or not callable(getattr(item, "to_dict", None))
                ):
                    raise TypeError("support_view must contain immutable render descriptors")
                _json_safe(item.to_dict())

        object.__setattr__(self, "producing_phase", producing_phase)
        object.__setattr__(self, "node_keys", tuple(node_keys))
        object.__setattr__(self, "_emitted_answer_json", emitted_snapshot)
        object.__setattr__(self, "_cvsearch_raw_json", raw_snapshot)
        object.__setattr__(self, "_support_view", frozen_view)

    @property
    def emitted_answer(self) -> Any:
        return json.loads(self._emitted_answer_json)

    @property
    def cvsearch_raw(self) -> Any:
        return json.loads(self._cvsearch_raw_json)

    @property
    def support_view(self) -> tuple[Any, ...] | None:
        if self._support_view is None:
            return None
        return copy.deepcopy(self._support_view)

    def to_dict(self) -> dict[str, Any]:
        return {
            "emitted_answer": json.loads(self._emitted_answer_json),
            "cvsearch_raw": json.loads(self._cvsearch_raw_json),
            "producing_phase": self.producing_phase,
            "node_keys": _json_safe(self.node_keys),
            "support_view": None if self._support_view is None else [
                _json_safe(item.to_dict()) for item in self._support_view
            ],
        }


@dataclass
class HistoryRecord:
    step: int = 0
    answer: AnswerRecord = field(default_factory=AnswerRecord)
    support_avg: float = 0.0
    support_min: float = 0.0
    cost: int | float = 0
    has_unvisited_branch: bool = False
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": _json_safe(self.step),
            "answer": self.answer.to_dict(),
            "support_avg": _json_safe(self.support_avg),
            "support_min": _json_safe(self.support_min),
            "cost": _json_safe(self.cost),
            "has_unvisited_branch": _json_safe(self.has_unvisited_branch),
            "state": _json_safe(self.state),
        }


@dataclass(frozen=True)
class NextAudit:
    """Strict, observation-only audit for one unified P2A NEXT attempt."""

    p0_anchor: P0Anchor
    current_keys: tuple[str, ...]
    candidate_keys: tuple[str, ...]
    batch_result: ObservationBatchResult | None
    uncertainty: float
    g_next: float | None
    support_delta: float | None
    p0_stability: AnswerRecord
    candidate_stability: AnswerRecord | None
    feasible: bool
    normalized_actual_cost: float | None
    support_contract_status: str
    coverage_status: str = "not_observed"
    verifier_status: str = "disabled_same_checkpoint_unpromoted"
    verifier_avg: None = None
    verifier_min: None = None
    score_margin: None = None
    score_status: str = "unavailable_missing_verifier_coverage"
    replacement_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.p0_anchor, P0Anchor):
            raise TypeError("NEXT audit requires a P0Anchor")
        for name in ("current_keys", "candidate_keys"):
            keys = getattr(self, name)
            if not isinstance(keys, tuple) or not all(
                isinstance(key, str) and key for key in keys
            ) or len(keys) != len(set(keys)):
                raise ValueError(f"{name} must be unique nonempty canonical keys")
        if self.current_keys != self.p0_anchor.node_keys:
            raise ValueError("NEXT current keys must match the exact P0 anchor")
        uncertainty = _finite_number(self.uncertainty, "uncertainty")
        if not 0.0 <= uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0, 1]")
        if not isinstance(self.p0_stability, AnswerRecord):
            raise TypeError("NEXT audit requires P0 stability")
        if _json_safe(self.p0_stability.output) != _json_safe(
            self.p0_anchor.emitted_answer
        ):
            raise ValueError("P0 stability output must preserve the exact emitted anchor")
        if abs(uncertainty - _finite_number(
            self.p0_stability.uncertainty, "p0_stability.uncertainty"
        )) > 1e-12:
            raise ValueError("NEXT uncertainty must match immutable P0 stability")
        if self.candidate_stability is not None and not isinstance(
            self.candidate_stability, AnswerRecord
        ):
            raise TypeError("candidate_stability must be an AnswerRecord or None")
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be boolean")
        if self.support_contract_status not in {"matched", "not_observed", "mismatch"}:
            raise ValueError("invalid support contract status")
        if self.coverage_status != "not_observed":
            raise ValueError("P2A coverage must remain unobserved")
        if self.verifier_status != "disabled_same_checkpoint_unpromoted":
            raise ValueError("P2A verifier must remain disabled")
        if self.verifier_avg is not None or self.verifier_min is not None:
            raise ValueError("P2A verifier scores must remain null")
        if self.score_margin is not None or self.score_status != (
            "unavailable_missing_verifier_coverage"
        ):
            raise ValueError("P2A selector score must remain unavailable")
        cost = _optional_finite_number(
            self.normalized_actual_cost, "normalized_actual_cost"
        )
        if cost is not None and not 0.0 <= cost <= 1.0:
            raise ValueError("normalized_actual_cost must be in [0, 1]")
        gap = _optional_finite_number(self.g_next, "g_next")
        delta = _optional_finite_number(self.support_delta, "support_delta")
        if gap is not None and not 0.0 <= gap <= 1.0:
            raise ValueError("g_next must be in [0, 1]")
        if delta is not None and not -1.0 <= delta <= 1.0:
            raise ValueError("support_delta must be in [-1, 1]")
        if self.batch_result is not None and not isinstance(
            self.batch_result, ObservationBatchResult
        ):
            raise TypeError("batch_result must be an ObservationBatchResult or None")
        if self.batch_result is None:
            if cost is not None:
                raise ValueError("NEXT without a batch cannot report actual batch cost")
        else:
            batch_payload = self.batch_result.to_dict()
            before = batch_payload["ledger_before"]
            after = batch_payload["ledger_after"]
            maximum = after["max_mllm_calls"]
            expected_cost = 0.0 if maximum <= 0 else (
                after["mllm_calls"] - before["mllm_calls"]
            ) / maximum
            if cost is None or abs(cost - expected_cost) > 1e-12:
                raise ValueError("normalized actual cost must match the batch ledger delta")
            if len(self.candidate_keys) != 1:
                raise ValueError("an attempted NEXT batch requires exactly one candidate key")
            plan = self.batch_result.batch_plan
            if "current_observation" in plan and (
                plan["current_observation"]["canonical_keys"] != list(self.current_keys)
                or plan["candidate_observation"]["canonical_keys"]
                != list(self.candidate_keys)
            ):
                raise ValueError("NEXT keys must match the exact batch observations")
            if self.batch_result.status == "success":
                if self.support_contract_status not in {"matched", "mismatch"}:
                    raise ValueError("successful batch requires an explicit support contract audit")
            elif self.support_contract_status != "not_observed":
                raise ValueError("unsuccessful batch cannot claim a support contract match")

        success = (
            self.batch_result is not None
            and self.batch_result.status == "success"
            and self.support_contract_status == "matched"
        )
        if success:
            if (
                len(self.candidate_keys) != 1
                or self.batch_result.current_support is None
                or self.batch_result.candidate_support is None
                or gap is None
                or delta is None
                or self.candidate_stability is None
                or not self.feasible
                or cost is None
                or self.replacement_reason != "replacement_disabled_p2a"
            ):
                raise ValueError("successful NEXT audit is incomplete")
            expected_gap = 1.0 - self.batch_result.current_support.p_yes
            expected_delta = (
                self.batch_result.candidate_support.p_yes
                - self.batch_result.current_support.p_yes
            )
            if abs(gap - expected_gap) > 1e-12 or abs(delta - expected_delta) > 1e-12:
                raise ValueError("NEXT gap support values do not match the atomic batch")
            candidate_answer = self.batch_result.candidate_answer
            if self.batch_result.batch_plan["answer_type"] == "option_list":
                if tuple(candidate_answer) != tuple(self.candidate_stability.raw_outputs):
                    raise ValueError("HR stability must retain all four raw candidate shuffles")
            elif (
                self.candidate_stability.output != candidate_answer["winner"]
                or tuple(self.candidate_stability.losses) != tuple(candidate_answer["losses"])
            ):
                raise ValueError("V* stability must retain the exact loss argmin material")
        elif (
            gap is not None
            or delta is not None
            or self.candidate_stability is not None
            or self.feasible
            or self.replacement_reason is not None
        ):
            raise ValueError("unsuccessful NEXT audit cannot synthesize measurements")

    @property
    def current_support(self) -> EvidenceSupportResult | None:
        return None if self.batch_result is None else self.batch_result.current_support

    @property
    def candidate_support(self) -> EvidenceSupportResult | None:
        return None if self.batch_result is None else self.batch_result.candidate_support

    def to_dict(self) -> dict[str, Any]:
        return {
            "p0_anchor": self.p0_anchor.to_dict(),
            "current_keys": _json_safe(self.current_keys),
            "candidate_keys": _json_safe(self.candidate_keys),
            "batch_result": None if self.batch_result is None else self.batch_result.to_dict(),
            "current_gap_support": (
                None if self.current_support is None else self.current_support.to_dict()
            ),
            "candidate_gap_support": (
                None if self.candidate_support is None else self.candidate_support.to_dict()
            ),
            "uncertainty": _json_safe(self.uncertainty),
            "g_next": _json_safe(self.g_next),
            "support_delta": _json_safe(self.support_delta),
            "p0_stability": self.p0_stability.to_dict(),
            "candidate_stability": (
                None if self.candidate_stability is None
                else self.candidate_stability.to_dict()
            ),
            "feasible": self.feasible,
            "normalized_actual_cost": _json_safe(self.normalized_actual_cost),
            "support_contract_status": self.support_contract_status,
            "coverage_status": self.coverage_status,
            "verifier_status": self.verifier_status,
            "verifier_avg": None,
            "verifier_min": None,
            "score_margin": None,
            "score_status": self.score_status,
            "replacement_reason": self.replacement_reason,
        }


@dataclass
class StepTrace:
    step: int = 0
    action: str = ""
    gap_fallback_used: bool = False
    elapsed_seconds: float = 0.0
    focus_key: str | None = None
    feasible_actions: tuple[str, ...] = ()
    gaps: dict[str, float] = field(default_factory=dict)
    no_op_reason: str | None = None
    answer: AnswerRecord | None = None
    support_avg: float = 0.0
    support_min: float = 0.0
    certified: bool = False
    budget: BudgetLedger | None = None
    next_audit: NextAudit | None = None

    def __post_init__(self) -> None:
        if self.next_audit is None:
            return
        if not isinstance(self.next_audit, NextAudit):
            raise TypeError("next_audit must be a NextAudit")
        if self.action != NEXT:
            raise ValueError("NEXT audit can only be attached to action=NEXT")
        expected_actions = (NEXT,) if self.next_audit.feasible else ()
        if self.feasible_actions != expected_actions:
            raise ValueError("NEXT feasible actions do not match its strict audit")
        expected_gaps = (
            {"g_next": self.next_audit.g_next}
            if self.next_audit.g_next is not None else {}
        )
        if self.gaps != expected_gaps:
            raise ValueError("NEXT StepTrace gaps do not match its strict audit")
        if self.next_audit.feasible:
            if self.no_op_reason is not None:
                raise ValueError("successful NEXT cannot report a no-op reason")
        elif not isinstance(self.no_op_reason, str) or not self.no_op_reason:
            raise ValueError("unsuccessful NEXT requires a no-op reason")
        if self.answer is None or _json_safe(self.answer.output) != _json_safe(
            self.next_audit.p0_anchor.emitted_answer
        ):
            raise ValueError("NEXT StepTrace must retain the exact P0 answer")
        if self.budget is None:
            raise ValueError("NEXT StepTrace requires its post-attempt budget")
        if self.next_audit.batch_result is not None:
            expected_budget = self.next_audit.batch_result.to_dict()["ledger_after"]
            if self.budget.to_dict() != expected_budget:
                raise ValueError("NEXT StepTrace budget must match the batch ledger")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "step": _json_safe(self.step),
            "action": _json_safe(self.action),
            "gap_fallback_used": _json_safe(self.gap_fallback_used),
            "elapsed_seconds": _json_safe(self.elapsed_seconds),
            "focus_key": _json_safe(self.focus_key),
            "feasible_actions": _json_safe(self.feasible_actions),
            "gaps": _json_safe(self.gaps),
            "no_op_reason": _json_safe(self.no_op_reason),
            "answer": None if self.answer is None else self.answer.to_dict(),
            "support_avg": _json_safe(self.support_avg),
            "support_min": _json_safe(self.support_min),
            "certified": _json_safe(self.certified),
            "budget": None if self.budget is None else self.budget.to_dict(),
        }
        if self.next_audit is not None:
            if not isinstance(self.next_audit, NextAudit):
                raise TypeError("next_audit must be a NextAudit")
            payload["next_audit"] = self.next_audit.to_dict()
        return payload


@dataclass
class MethodTrace:
    query_plan: QueryPlan | None = None
    candidate_ranks: list[Any] = field(default_factory=list)
    steps: list[StepTrace] = field(default_factory=list)
    history: list[HistoryRecord] = field(default_factory=list)
    final_answer: AnswerRecord | None = None
    budget: BudgetLedger | None = None
    elapsed_seconds: float = 0.0
    termination: str | None = None
    final_boxes: tuple[tuple[int | float, int | float, int | float, int | float], ...] = ()
    method_mode: str | None = None
    config_id: str | None = None
    effective_config: dict[str, Any] = field(default_factory=dict)
    cvsearch_search_mode: int | None = None
    root_ans_conf: float | None = None
    num_pop: list[Any] = field(default_factory=list)
    num_zoom_in: list[Any] = field(default_factory=list)
    num_zoom_out: list[Any] = field(default_factory=list)
    budget_interrupted: bool = False
    effective_ranking_query: str | None = None
    pixel_accounting: str | None = None
    anchor_answer: AnswerRecord | None = None
    anchor_state_score: float | None = None
    selected_state_score: float | None = None
    replacement_margin: float | None = None
    support_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_plan": None if self.query_plan is None else self.query_plan.to_dict(),
            "candidate_ranks": _json_safe(self.candidate_ranks),
            "steps": _json_safe(self.steps),
            "history": _json_safe(self.history),
            "final_answer": None if self.final_answer is None else self.final_answer.to_dict(),
            "budget": None if self.budget is None else self.budget.to_dict(),
            "elapsed_seconds": _json_safe(self.elapsed_seconds),
            "termination": _json_safe(self.termination),
            "final_boxes": _json_safe(self.final_boxes),
            "method_mode": _json_safe(self.method_mode),
            "config_id": _json_safe(self.config_id),
            "effective_config": _json_safe(self.effective_config),
            "cvsearch_search_mode": _json_safe(self.cvsearch_search_mode),
            "root_ans_conf": _json_safe(self.root_ans_conf),
            "num_pop": _json_safe(self.num_pop),
            "num_zoom_in": _json_safe(self.num_zoom_in),
            "num_zoom_out": _json_safe(self.num_zoom_out),
            "budget_interrupted": _json_safe(self.budget_interrupted),
            "effective_ranking_query": _json_safe(self.effective_ranking_query),
            "pixel_accounting": _json_safe(self.pixel_accounting),
            "anchor_answer": None if self.anchor_answer is None else self.anchor_answer.to_dict(),
            "anchor_state_score": _optional_finite_number(
                self.anchor_state_score, "anchor_state_score"
            ),
            "selected_state_score": _optional_finite_number(
                self.selected_state_score, "selected_state_score"
            ),
            "replacement_margin": _optional_finite_number(
                self.replacement_margin, "replacement_margin"
            ),
            "support_status": _json_safe(self.support_status),
        }
