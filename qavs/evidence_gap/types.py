"""Pure state contracts for query-aware evidence-gap search."""

from dataclasses import dataclass, field, is_dataclass, replace as dataclass_replace
import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any


EVIDENCE_SUPPORT_TRANSFORM = (
    "v1:p_yes=softmax(final_position_two_logits[Yes,No],dim=-1,"
    "preserve_model_dtype,no_float32_cast)[0];no_legacy_2p_minus_1"
)


# One ULP at unit magnitude for the frozen bfloat16 probability transform.
EVIDENCE_SUPPORT_NORMALIZATION_TOLERANCE = 2 ** -7


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
        if abs((self.p_yes + self.p_no) - 1.0) > EVIDENCE_SUPPORT_NORMALIZATION_TOLERANCE:
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
