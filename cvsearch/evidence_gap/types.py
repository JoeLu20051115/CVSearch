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

from cvsearch.models.utils import merge_bbox_list, union_all_bboxes
from .search_state import ExpandDecision


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


def _validate_support_token_contract(payload: Mapping[str, Any]) -> None:
    """Keep Qwen's frozen IDs exact while admitting the pinned InternVL adapter."""
    if not isinstance(payload, Mapping):
        raise TypeError("support token contract must be a mapping")
    fingerprint = payload.get("processor_fingerprint")
    if type(fingerprint) is not dict:
        raise TypeError("support token contract requires a processor fingerprint")
    model_type = fingerprint.get("model_type")
    expected_by_model = {
        "qwen2_5_vl": (9454, 2753),
        "qwen3_vl": (9454, 2753),
        "internvl_chat": (9583, 2917),
    }
    if model_type not in expected_by_model:
        raise ValueError(f"unsupported support token model_type: {model_type!r}")
    yes_id, no_id = expected_by_model[model_type]
    if (
        payload.get("yes_tokenization") != [yes_id]
        or payload.get("no_tokenization") != [no_id]
        or payload.get("yes_token_id") != yes_id
        or payload.get("no_token_id") != no_id
    ):
        raise ValueError(
            f"batch plan Yes/No token contract is invalid for {model_type}"
        )


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
_FULL_ZOOM_BATCH_PLAN_FIELDS = _FULL_BATCH_PLAN_FIELDS | frozenset({
    "render_policy", "base_view_size", "candidate_view_size", "current_keys",
    "zoom_keys", "coordinate_mapping", "current_merge_identity",
    "candidate_merge_identity", "candidate_answer_input_sha256",
})
_FULL_EXPAND_BATCH_PLAN_FIELDS = _FULL_BATCH_PLAN_FIELDS | frozenset({
    "selection_policy", "composition_policy", "base_view_size", "patch_scale",
    "current_keys", "candidate_keys", "focus_role", "context_role",
    "focus_merge_identity", "context_merge_identity", "composition_identity",
    "candidate_answer_input_sha256", "q0", "options", "options_sha256",
    "answer_prompt_sha256", "answer_call_identity_sha256",
})
_ZOOM_MAPPING_FIELDS = frozenset({
    "descriptor_index", "current_key", "zoom_key", "bbox_original", "depth",
    "render_level", "posterior_score", "first_seen_ordinal", "tree_scope",
    "crop_origin", "source_image_key", "source", "renderer_kind",
    "source_renderer_identity", "zoom_renderer_identity", "current_crop_xyxy",
    "candidate_crop_xyxy",
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


def _validate_zoom_crop(value: Any, name: str, source_size: list[int]) -> list[int]:
    if (
        not isinstance(value, list) or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, Integral) for item in value)
    ):
        raise ValueError(f"{name} must be an integer xyxy crop")
    crop = [int(item) for item in value]
    if not (
        0 <= crop[0] < crop[2] <= source_size[0]
        and 0 <= crop[1] < crop[3] <= source_size[1]
    ):
        raise ValueError(f"{name} is outside the source image")
    return crop


def _validate_zoom_merge_identity(
    value: Any, name: str, expected_crops: list[list[int]], source_size: list[int],
) -> None:
    fields = {
        "per_descriptor_crop_xyxy", "merged_crop_xyxy", "union_crop_xyxy",
        "identity_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} has an invalid exact schema")
    if value["per_descriptor_crop_xyxy"] != expected_crops:
        raise ValueError(f"{name} descriptor crops do not match coordinate mapping")
    merged = value["merged_crop_xyxy"]
    if not isinstance(merged, list) or not merged:
        raise ValueError(f"{name} merged crops must be nonempty")
    for index, crop in enumerate(merged):
        _validate_zoom_crop(crop, f"{name}.merged_crop_xyxy[{index}]", source_size)
    union = _validate_zoom_crop(
        value["union_crop_xyxy"], f"{name}.union_crop_xyxy", source_size,
    )
    expected_merged = [list(crop) for crop in merge_bbox_list(
        copy.deepcopy(expected_crops), threshold=0,
    )]
    expected_union_raw = union_all_bboxes(expected_merged)
    if expected_union_raw is None:
        raise ValueError(f"{name} expected union must be nonempty")
    expected_union = list(expected_union_raw)
    if merged != expected_merged or union != expected_union:
        raise ValueError(f"{name} does not match the native merge and union geometry")
    identity_payload = {
        "per_descriptor_crop_xyxy": expected_crops,
        "merged_crop_xyxy": merged,
        "union_crop_xyxy": value["union_crop_xyxy"],
    }
    encoded = json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    expected_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if value["identity_sha256"] != expected_hash:
        raise ValueError(f"{name} identity hash does not match its crops")


def _validate_zoom_plan(
    payload: dict[str, Any], *, require_observation_change: bool,
) -> None:
    if payload["render_policy"] != "native_coordinate_crop_view_size_div3":
        raise ValueError("ZOOM batch render policy is not frozen")
    base_size = _integral(payload["base_view_size"], "base_view_size")
    candidate_size = _integral(payload["candidate_view_size"], "candidate_view_size")
    if base_size < 4 or candidate_size != base_size // 3 or candidate_size <= 0:
        raise ValueError("ZOOM batch view sizes do not match the divided-by-three policy")
    current_keys = payload["current_keys"]
    zoom_keys = payload["zoom_keys"]
    mapping = payload["coordinate_mapping"]
    if not all(isinstance(items, list) for items in (current_keys, zoom_keys, mapping)):
        raise TypeError("ZOOM batch keys and mapping must be lists")
    if (
        not current_keys or len(current_keys) != len(zoom_keys)
        or len(current_keys) != len(mapping)
        or len(set(current_keys)) != len(current_keys)
        or len(set(zoom_keys)) != len(zoom_keys)
        or not all(isinstance(key, str) and key for key in current_keys + zoom_keys)
    ):
        raise ValueError("ZOOM batch requires one ordered zoom key per current key")
    source = payload["source_identity"]
    source_size = source["size"]
    source_key = json.dumps(
        source, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    current_crops: list[list[int]] = []
    candidate_crops: list[list[int]] = []
    source_descriptors: list[dict[str, Any]] = []
    zoom_renderer_ids: list[str] = []
    strict_shrink_observed = False
    for index, item in enumerate(mapping):
        if not isinstance(item, dict) or set(item) != _ZOOM_MAPPING_FIELDS:
            raise ValueError("ZOOM coordinate mapping has an invalid exact schema")
        if item["descriptor_index"] != index:
            raise ValueError("ZOOM coordinate mapping order is not one-to-one")
        if item["current_key"] != current_keys[index] or item["zoom_key"] != zoom_keys[index]:
            raise ValueError("ZOOM coordinate mapping keys do not match ordered keys")
        bbox = item["bbox_original"]
        if (
            not isinstance(bbox, list) or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, Real)
                   or not math.isfinite(float(value)) for value in bbox)
        ):
            raise ValueError("ZOOM source bbox must contain four finite numbers")
        x, y, width, height = (float(value) for value in bbox)
        if (
            x < 0 or y < 0 or width <= 0 or height <= 0
            or x + width > source_size[0] or y + height > source_size[1]
        ):
            raise ValueError("ZOOM source bbox is outside the source image")
        depth = _integral(item["depth"], "ZOOM descriptor depth")
        render_level = _integral(item["render_level"], "ZOOM descriptor render_level")
        _integral(item["first_seen_ordinal"], "ZOOM descriptor first_seen_ordinal")
        posterior = item["posterior_score"]
        if posterior is not None:
            _finite_number(posterior, "ZOOM descriptor posterior_score")
        if item["tree_scope"] not in {"main", "cropped"}:
            raise ValueError("ZOOM descriptor tree scope is invalid")
        origin = item["crop_origin"]
        if (
            not isinstance(origin, list) or len(origin) != 2
            or any(isinstance(value, bool) or not isinstance(value, Real)
                   or not math.isfinite(float(value)) for value in origin)
        ):
            raise ValueError("ZOOM descriptor crop origin is invalid")
        if item["source_image_key"] != source_key:
            raise ValueError("ZOOM descriptor source identity does not match the RGB source")
        if item["source"] not in {None, "fast", "fine", "fine_fallback"}:
            raise ValueError("ZOOM descriptor is not a local native CVSearch view")
        expected_kind = "fast" if item["source"] == "fast" else "fine"
        if item["renderer_kind"] != expected_kind:
            raise ValueError("ZOOM descriptor renderer kind does not match its source")
        expected_current_key = json.dumps(
            {"bbox": bbox, "depth": depth, "render_level": render_level},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        if item["current_key"] != expected_current_key:
            raise ValueError("ZOOM current key does not match original geometry")
        renderer_payload = {
            "source_image_key": source_key,
            "renderer_kind": expected_kind,
            "bbox": bbox,
            "render_level": render_level,
        }
        expected_source_renderer = json.dumps(
            renderer_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        if item["source_renderer_identity"] != expected_source_renderer:
            raise ValueError("ZOOM source renderer identity does not match original geometry")
        current_crop = _validate_zoom_crop(
            item["current_crop_xyxy"], "current_crop_xyxy", source_size,
        )
        candidate_crop = _validate_zoom_crop(
            item["candidate_crop_xyxy"], "candidate_crop_xyxy", source_size,
        )
        if not (
            current_crop[0] <= candidate_crop[0]
            and current_crop[1] <= candidate_crop[1]
            and candidate_crop[2] <= current_crop[2]
            and candidate_crop[3] <= current_crop[3]
        ):
            raise ValueError("ZOOM candidate crop must be a subset of its current crop")
        strict_shrink_observed = strict_shrink_observed or candidate_crop != current_crop
        zoom_payload = {
            "action": "P2C_ZOOM",
            "candidate_crop_xyxy": candidate_crop,
            "candidate_view_size": candidate_size,
            "current_key": item["current_key"],
            "render_policy": payload["render_policy"],
            "source_image_key": source_key,
        }
        expected_zoom_renderer = json.dumps(
            zoom_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        expected_zoom_key = "zoom-" + hashlib.sha256(
            expected_zoom_renderer.encode("utf-8")
        ).hexdigest()
        if (
            item["zoom_renderer_identity"] != expected_zoom_renderer
            or item["zoom_key"] != expected_zoom_key
        ):
            raise ValueError("ZOOM render-level identity does not match its crop")
        source_descriptor = {
            "canonical_key": item["current_key"],
            "bbox_original": bbox,
            "depth": depth,
            "render_level": render_level,
            "posterior_score": posterior,
            "first_seen_ordinal": item["first_seen_ordinal"],
            "tree_scope": item["tree_scope"],
            "crop_origin": origin,
            "source_image_key": source_key,
            "source": item["source"],
            "renderer_kind": expected_kind,
            "renderer_identity": expected_source_renderer,
        }
        source_descriptors.append(source_descriptor)
        zoom_renderer_ids.append(expected_zoom_renderer)
        current_crops.append(current_crop)
        candidate_crops.append(candidate_crop)
    _validate_zoom_merge_identity(
        payload["current_merge_identity"], "current_merge_identity",
        current_crops, source_size,
    )
    _validate_zoom_merge_identity(
        payload["candidate_merge_identity"], "candidate_merge_identity",
        candidate_crops, source_size,
    )
    current_observation = payload["current_observation"]
    candidate_observation = payload["candidate_observation"]
    if (
        current_observation["canonical_keys"] != current_keys
        or current_observation["renderer_identities"]
        != [item["source_renderer_identity"] for item in mapping]
        or current_observation["descriptors"] != source_descriptors
    ):
        raise ValueError("ZOOM current observation is not bound to P0 descriptors")
    expected_candidate_descriptors = [
        {
            "canonical_key": zoom_key,
            "renderer_identity": renderer_id,
            "source_descriptor": source_descriptor,
        }
        for zoom_key, renderer_id, source_descriptor in zip(
            zoom_keys, zoom_renderer_ids, source_descriptors,
        )
    ]
    if (
        candidate_observation["canonical_keys"] != zoom_keys
        or candidate_observation["renderer_identities"] != zoom_renderer_ids
        or candidate_observation["descriptors"] != expected_candidate_descriptors
    ):
        raise ValueError("ZOOM candidate observation is not a render-only derivative")
    if payload["candidate_answer_input_sha256"] != candidate_observation["view_sha256"]:
        raise ValueError("ZOOM candidate support and answer input hashes must match")
    if require_observation_change and not strict_shrink_observed:
        raise ValueError("ZOOM requires at least one strictly smaller candidate crop")
    if (
        require_observation_change
        and current_observation["view_sha256"] == candidate_observation["view_sha256"]
    ):
        raise ValueError("ZOOM requires distinct current and candidate rendered hashes")


def _expand_native_crop(
    descriptor: Mapping[str, Any], *, source_size: list[int], base_view_size: int,
    patch_scale: float | None,
) -> list[int]:
    bbox = descriptor["bbox_original"]
    object_width = math.ceil(float(bbox[2]))
    object_height = math.ceil(float(bbox[3]))
    center_x = int(float(bbox[0]) + float(bbox[2]) / 2)
    center_y = int(float(bbox[1]) + float(bbox[3]) / 2)
    patch_size = base_view_size // 3 if descriptor["source"] == "fast" else base_view_size
    scale = None if descriptor["source"] == "fast" else patch_scale
    patch_width = max(object_width, patch_size)
    patch_height = max(object_height, patch_size)
    if scale is not None:
        patch_width = int(patch_width * scale)
        patch_height = int(patch_height * scale)
    left = max(0, center_x - patch_width // 2)
    top = max(0, center_y - patch_height // 2)
    return [
        left, top, min(left + patch_width, source_size[0]),
        min(top + patch_height, source_size[1]),
    ]


def _validate_expand_descriptor(
    descriptor: Any, *, source_key: str, source_size: list[int], name: str,
) -> dict[str, Any]:
    fields = {
        "canonical_key", "bbox_original", "depth", "render_level", "posterior_score",
        "first_seen_ordinal", "tree_scope", "crop_origin", "source_image_key",
        "source", "renderer_kind", "renderer_identity",
    }
    if not isinstance(descriptor, dict) or set(descriptor) != fields:
        raise ValueError(f"{name} descriptor has an invalid exact schema")
    bbox = descriptor["bbox_original"]
    if (
        not isinstance(bbox, list) or len(bbox) != 4
        or any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(float(value)) for value in bbox)
    ):
        raise ValueError(f"{name} descriptor bbox must contain four finite numbers")
    x, y, width, height = (float(value) for value in bbox)
    if (
        x < 0 or y < 0 or width <= 0 or height <= 0
        or x + width > source_size[0] or y + height > source_size[1]
    ):
        raise ValueError(f"{name} descriptor bbox is outside the source")
    depth = _integral(descriptor["depth"], f"{name} depth")
    render_level = _integral(descriptor["render_level"], f"{name} render_level")
    _integral(descriptor["first_seen_ordinal"], f"{name} first_seen_ordinal")
    posterior = descriptor["posterior_score"]
    if posterior is not None:
        _finite_number(posterior, f"{name} posterior_score")
    if descriptor["tree_scope"] not in {"main", "cropped"}:
        raise ValueError(f"{name} tree_scope is invalid")
    origin = descriptor["crop_origin"]
    if (
        not isinstance(origin, list) or len(origin) != 2
        or any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(float(value)) for value in origin)
    ):
        raise ValueError(f"{name} crop_origin is invalid")
    if descriptor["source_image_key"] != source_key:
        raise ValueError(f"{name} source identity differs from the RGB source")
    if descriptor["source"] not in {None, "fast", "fine", "fine_fallback"}:
        raise ValueError(f"{name} is not a local native descriptor")
    expected_kind = "fast" if descriptor["source"] == "fast" else "fine"
    if descriptor["renderer_kind"] != expected_kind:
        raise ValueError(f"{name} renderer kind does not match source")
    expected_key = json.dumps(
        {"bbox": bbox, "depth": depth, "render_level": render_level},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    renderer_payload = {
        "source_image_key": source_key, "renderer_kind": expected_kind,
        "bbox": bbox, "render_level": render_level,
    }
    expected_renderer = json.dumps(
        renderer_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    if descriptor["canonical_key"] != expected_key:
        raise ValueError(f"{name} canonical key does not match geometry")
    if descriptor["renderer_identity"] != expected_renderer:
        raise ValueError(f"{name} renderer identity does not match geometry")
    return descriptor


def _expand_union_area(rectangles: Sequence[Sequence[float]]) -> float:
    xs = sorted({float(value) for rectangle in rectangles for value in rectangle[::2]})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted(
            (float(rectangle[1]), float(rectangle[3])) for rectangle in rectangles
            if float(rectangle[0]) < right and left < float(rectangle[2])
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


def _validate_expand_plan(
    payload: dict[str, Any], *, require_observation_change: bool,
) -> None:
    if payload["selection_policy"] != "nearest_spatial_native_cvsearch_context_v1":
        raise ValueError("EXPAND selection policy is not frozen")
    if payload["composition_policy"] != "focus_top_blank_or_context_bottom_native_pixels_v1":
        raise ValueError("EXPAND composition policy is not frozen")
    base_size = _integral(payload["base_view_size"], "EXPAND base_view_size")
    if base_size < 4:
        raise ValueError("EXPAND base view size must be at least four")
    patch_scale = payload["patch_scale"]
    if patch_scale is not None:
        patch_scale = _finite_number(patch_scale, "EXPAND patch_scale")
        if patch_scale <= 0:
            raise ValueError("EXPAND patch_scale must be positive or null")
    current_keys = payload["current_keys"]
    candidate_keys = payload["candidate_keys"]
    if (
        not isinstance(current_keys, list) or not current_keys
        or not all(isinstance(key, str) and key for key in current_keys)
        or len(current_keys) != len(set(current_keys))
        or not isinstance(candidate_keys, list)
        or candidate_keys[:-1] != current_keys
        or len(candidate_keys) != len(current_keys) + 1
        or len(candidate_keys) != len(set(candidate_keys))
    ):
        raise ValueError("EXPAND candidate keys must append exactly one context key")
    source = payload["source_identity"]
    source_size = source["size"]
    source_key = json.dumps(
        source, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    focus_role = payload["focus_role"]
    focus_fields = {
        "role", "canonical_keys", "renderer_identities", "descriptors",
        "native_crops_xyxy",
    }
    if not isinstance(focus_role, dict) or set(focus_role) != focus_fields:
        raise ValueError("EXPAND focus role has an invalid exact schema")
    if focus_role["role"] != "focus" or focus_role["canonical_keys"] != current_keys:
        raise ValueError("EXPAND focus role does not preserve P0 ordering")
    descriptors = focus_role["descriptors"]
    renderer_ids = focus_role["renderer_identities"]
    crops = focus_role["native_crops_xyxy"]
    if not all(isinstance(value, list) for value in (descriptors, renderer_ids, crops)):
        raise TypeError("EXPAND focus role sequences must be lists")
    if not (len(descriptors) == len(renderer_ids) == len(crops) == len(current_keys)):
        raise ValueError("EXPAND focus role descriptor counts differ")
    for index, descriptor in enumerate(descriptors):
        _validate_expand_descriptor(
            descriptor, source_key=source_key, source_size=source_size,
            name=f"focus[{index}]",
        )
        if (
            descriptor["canonical_key"] != current_keys[index]
            or descriptor["renderer_identity"] != renderer_ids[index]
        ):
            raise ValueError("EXPAND focus role descriptor identities differ")
        expected_crop = _expand_native_crop(
            descriptor, source_size=source_size, base_view_size=base_size,
            patch_scale=patch_scale,
        )
        if _validate_zoom_crop(crops[index], "focus native crop", source_size) != expected_crop:
            raise ValueError("EXPAND focus native crop is not recomputable")
    if len(set(renderer_ids)) != len(renderer_ids):
        raise ValueError("EXPAND focus renderer identities must be unique")

    context_role = payload["context_role"]
    context_fields = {
        "role", "canonical_key", "renderer_identity", "descriptor",
        "native_crop_xyxy", "candidate_bbox_xyxy", "focus_union_xyxy",
        "positive_outside_area", "contained_by_focus_descriptor",
        "contains_complete_focus_union", "normalized_edge_gap", "rank_tuple",
    }
    if not isinstance(context_role, dict) or set(context_role) != context_fields:
        raise ValueError("EXPAND context role has an invalid exact schema")
    if context_role["role"] != "context":
        raise ValueError("EXPAND appended descriptor must retain context role")
    context = _validate_expand_descriptor(
        context_role["descriptor"], source_key=source_key, source_size=source_size,
        name="context",
    )
    if (
        context_role["canonical_key"] != candidate_keys[-1]
        or context["canonical_key"] != candidate_keys[-1]
        or context_role["renderer_identity"] != context["renderer_identity"]
        or context["renderer_identity"] in renderer_ids
    ):
        raise ValueError("EXPAND context identity is duplicate or mismatched")
    expected_context_crop = _expand_native_crop(
        context, source_size=source_size, base_view_size=base_size,
        patch_scale=patch_scale,
    )
    if _validate_zoom_crop(
        context_role["native_crop_xyxy"], "context native crop", source_size,
    ) != expected_context_crop:
        raise ValueError("EXPAND context native crop is not recomputable")
    focus_rectangles = [
        [float(item["bbox_original"][0]), float(item["bbox_original"][1]),
         float(item["bbox_original"][0]) + float(item["bbox_original"][2]),
         float(item["bbox_original"][1]) + float(item["bbox_original"][3])]
        for item in descriptors
    ]
    expected_union = [
        min(item[0] for item in focus_rectangles),
        min(item[1] for item in focus_rectangles),
        max(item[2] for item in focus_rectangles),
        max(item[3] for item in focus_rectangles),
    ]
    bbox = context["bbox_original"]
    context_rectangle = [
        float(bbox[0]), float(bbox[1]), float(bbox[0]) + float(bbox[2]),
        float(bbox[1]) + float(bbox[3]),
    ]
    if (
        context_role["focus_union_xyxy"] != expected_union
        or context_role["candidate_bbox_xyxy"] != context_rectangle
    ):
        raise ValueError("EXPAND spatial rectangles do not match descriptors")
    intersections = []
    for item in focus_rectangles:
        overlap = [
            max(item[0], context_rectangle[0]), max(item[1], context_rectangle[1]),
            min(item[2], context_rectangle[2]), min(item[3], context_rectangle[3]),
        ]
        if overlap[0] < overlap[2] and overlap[1] < overlap[3]:
            intersections.append(overlap)
    context_area = float(bbox[2]) * float(bbox[3])
    outside_area = context_area - _expand_union_area(intersections)
    contained = any(
        context_rectangle[0] >= item[0] and context_rectangle[1] >= item[1]
        and context_rectangle[2] <= item[2] and context_rectangle[3] <= item[3]
        for item in focus_rectangles
    )
    contains_all = all(
        context_rectangle[0] <= item[0] and context_rectangle[1] <= item[1]
        and context_rectangle[2] >= item[2] and context_rectangle[3] >= item[3]
        for item in focus_rectangles
    )
    if (
        context_role["contained_by_focus_descriptor"] is not False
        or context_role["contains_complete_focus_union"] is not False
        or contained or contains_all or outside_area <= 0
        or abs(_finite_number(
            context_role["positive_outside_area"], "positive_outside_area",
        ) - outside_area) > 1e-9
    ):
        raise ValueError("EXPAND context does not add eligible outside area")
    edge_gap = min(
        math.hypot(
            max(item[0] - context_rectangle[2], context_rectangle[0] - item[2], 0.0),
            max(item[1] - context_rectangle[3], context_rectangle[1] - item[3], 0.0),
        ) for item in focus_rectangles
    ) / math.hypot(source_size[0], source_size[1])
    if abs(_finite_number(
        context_role["normalized_edge_gap"], "normalized_edge_gap",
    ) - edge_gap) > 1e-12:
        raise ValueError("EXPAND normalized edge gap is not recomputable")
    expected_rank = [
        edge_gap, context["posterior_score"] is None,
        0.0 if context["posterior_score"] is None else -context["posterior_score"],
        context["first_seen_ordinal"], context["canonical_key"],
    ]
    if context_role["rank_tuple"] != expected_rank:
        raise ValueError("EXPAND deterministic rank tuple is invalid")
    _validate_zoom_merge_identity(
        payload["focus_merge_identity"], "focus_merge_identity", crops, source_size,
    )
    _validate_zoom_merge_identity(
        payload["context_merge_identity"], "context_merge_identity",
        [context_role["native_crop_xyxy"]], source_size,
    )
    current_observation = payload["current_observation"]
    candidate_observation = payload["candidate_observation"]
    if (
        current_observation["canonical_keys"] != current_keys
        or current_observation["renderer_identities"] != renderer_ids
        or current_observation["descriptors"] != descriptors
        or candidate_observation["canonical_keys"] != candidate_keys
        or candidate_observation["renderer_identities"] != renderer_ids + [context["renderer_identity"]]
        or candidate_observation["descriptors"] != descriptors + [context]
    ):
        raise ValueError("EXPAND observations do not preserve focus then append context")
    composition = payload["composition_identity"]
    composition_fields = {
        "policy", "background_rgb", "separator_width", "canvas_size",
        "focus_offset_xy", "context_offset_xy", "focus_size", "context_size",
        "focus_panel_sha256", "context_panel_sha256", "blank_panel_sha256",
        "focus_current_rectangle_sha256", "focus_candidate_rectangle_sha256",
        "current_context_slot_sha256", "candidate_context_slot_sha256",
        "current_composite_sha256", "candidate_composite_sha256",
        "pixel_delta_region_xyxy", "identity_sha256",
    }
    if not isinstance(composition, dict) or set(composition) != composition_fields:
        raise ValueError("EXPAND composition identity has an invalid exact schema")
    if composition["policy"] != payload["composition_policy"]:
        raise ValueError("EXPAND composition policy identities differ")
    background = composition["background_rgb"]
    if (
        not isinstance(background, list) or len(background) != 3
        or any(isinstance(value, bool) or not isinstance(value, Integral)
               or not 0 <= value <= 255 for value in background)
    ):
        raise ValueError("EXPAND background RGB is invalid")
    focus_size = composition["focus_size"]
    context_size = composition["context_size"]
    canvas_size = composition["canvas_size"]
    for name, size in (("focus", focus_size), ("context", context_size), ("canvas", canvas_size)):
        if (
            not isinstance(size, list) or len(size) != 2
            or any(isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
                   for value in size)
        ):
            raise ValueError(f"EXPAND {name} size is invalid")
    if (
        composition["separator_width"] != 8
        or composition["focus_offset_xy"] != [0, 0]
        or composition["context_offset_xy"] != [0, focus_size[1] + 8]
        or canvas_size != [max(focus_size[0], context_size[0]),
                           focus_size[1] + 8 + context_size[1]]
        or composition["pixel_delta_region_xyxy"] != [
            0, focus_size[1] + 8, context_size[0],
            focus_size[1] + 8 + context_size[1],
        ]
    ):
        raise ValueError("EXPAND fixed canvas geometry is invalid")
    for name in composition_fields - {
        "policy", "background_rgb", "separator_width", "canvas_size",
        "focus_offset_xy", "context_offset_xy", "focus_size", "context_size",
        "pixel_delta_region_xyxy",
    }:
        _sha256_text(composition[name], f"composition.{name}")
    if (
        composition["focus_current_rectangle_sha256"] != composition["focus_panel_sha256"]
        or composition["focus_candidate_rectangle_sha256"] != composition["focus_panel_sha256"]
        or composition["current_context_slot_sha256"] != composition["blank_panel_sha256"]
        or composition["candidate_context_slot_sha256"] != composition["context_panel_sha256"]
        or composition["current_composite_sha256"] != current_observation["view_sha256"]
        or composition["candidate_composite_sha256"] != candidate_observation["view_sha256"]
        or current_observation["rendered_size"] != canvas_size
        or candidate_observation["rendered_size"] != canvas_size
    ):
        raise ValueError("EXPAND composition hashes do not bind its observations")
    composition_payload = dict(composition)
    supplied_composition_hash = composition_payload.pop("identity_sha256")
    encoded_composition = json.dumps(
        composition_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    if supplied_composition_hash != hashlib.sha256(
        encoded_composition.encode("utf-8")
    ).hexdigest():
        raise ValueError("EXPAND composition identity hash is invalid")
    if payload["candidate_answer_input_sha256"] != candidate_observation["view_sha256"]:
        raise ValueError("EXPAND candidate answer input differs from candidate composite")
    options = payload["options"]
    if payload["answer_type"] == "option_single":
        if (
            not isinstance(options, str) or not options.strip()
            or payload["candidate_option_count"] != 1
        ):
            raise ValueError("EXPAND single-choice options must be one nonempty string")
    elif (
        not isinstance(options, list) or not options
        or not all(isinstance(option, str) and option for option in options)
        or len(options) != payload["candidate_option_count"]
    ):
        raise ValueError("EXPAND options are not exact ordered strings")
    options_encoded = json.dumps(
        options, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )
    if payload["options_sha256"] != hashlib.sha256(
        options_encoded.encode("utf-8")
    ).hexdigest():
        raise ValueError("EXPAND options hash is invalid")
    q0 = payload["q0"]
    if (
        not isinstance(q0, str) or not q0.strip()
        or payload["q0_sha256"] != hashlib.sha256(q0.encode("utf-8")).hexdigest()
    ):
        raise ValueError("EXPAND q0 does not match its trusted hash")
    answer_hashes = payload["answer_prompt_sha256"]
    expected_hash_count = 4 if payload["answer_type"] == "option_list" else 1
    if not isinstance(answer_hashes, list) or len(answer_hashes) != expected_hash_count:
        raise ValueError("EXPAND answer prompt hash cardinality is invalid")
    for index, value in enumerate(answer_hashes):
        _sha256_text(value, f"answer_prompt_sha256[{index}]")
    if payload["answer_type"] == "option_list":
        expected_answer_hashes = [
            hashlib.sha256((
                q0 + "\n" + option + "Answer the option letter directly."
            ).encode("utf-8")).hexdigest()
            for option in options
        ]
    elif payload["answer_type"] == "option_single":
        prompt = (
            q0 + " Options:\n" + options + "\n"
            "Select the best answer to the above multiple-choice question based on "
            "the image. Respond with only the letter of the correct option.\n"
            "The best answer is:"
        )
        expected_answer_hashes = [
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        ]
    else:
        answer_payload = {"q0": q0, "options": options}
        answer_encoded = json.dumps(
            answer_payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        expected_answer_hashes = [
            hashlib.sha256(answer_encoded.encode("utf-8")).hexdigest()
        ]
    if answer_hashes != expected_answer_hashes:
        raise ValueError("EXPAND answer prompt hashes are not scorer-recomputable")
    call_identity = {
        "answer_type": payload["answer_type"], "q0_sha256": payload["q0_sha256"],
        "options_sha256": payload["options_sha256"],
        "answer_prompt_sha256": answer_hashes,
    }
    encoded_call = json.dumps(
        call_identity, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    if payload["answer_call_identity_sha256"] != hashlib.sha256(
        encoded_call.encode("utf-8")
    ).hexdigest():
        raise ValueError("EXPAND answer-call identity hash is invalid")
    if require_observation_change and (
        current_observation["view_sha256"] == candidate_observation["view_sha256"]
    ):
        raise ValueError("EXPAND requires distinct blank and context composites")


def _validate_batch_plan(
    plan: Mapping[str, Any], *, allow_zoom_noop: bool = False,
    allow_expand_noop: bool = False,
) -> tuple[dict[str, Any], str, bool]:
    if not isinstance(plan, Mapping):
        raise TypeError("batch plan must be a mapping")
    supplied_hash = plan.get("plan_hash")
    payload = _json_safe(dict(plan))
    payload.pop("plan_hash", None)
    fields = frozenset(payload)
    if fields not in {
        _ZERO_BATCH_PLAN_FIELDS, _FULL_BATCH_PLAN_FIELDS, _FULL_ZOOM_BATCH_PLAN_FIELDS,
        _FULL_EXPAND_BATCH_PLAN_FIELDS,
    }:
        raise ValueError("batch plan does not match an exact plan schema")
    is_full = fields in {
        _FULL_BATCH_PLAN_FIELDS, _FULL_ZOOM_BATCH_PLAN_FIELDS,
        _FULL_EXPAND_BATCH_PLAN_FIELDS,
    }
    batch_kind = payload["batch_kind"]
    if payload["schema_version"] != 1 or batch_kind not in {
        "p2a_post_anchor_next", "p2c_post_anchor_coordinate_zoom",
        "p4a_post_anchor_expand_context",
    }:
        raise ValueError("batch plan version or kind is invalid")
    if (
        (batch_kind == "p2a_post_anchor_next" and fields == _FULL_ZOOM_BATCH_PLAN_FIELDS)
        or (
            batch_kind == "p2c_post_anchor_coordinate_zoom"
            and is_full and fields != _FULL_ZOOM_BATCH_PLAN_FIELDS
        )
        or (
            batch_kind == "p4a_post_anchor_expand_context"
            and is_full and fields != _FULL_EXPAND_BATCH_PLAN_FIELDS
        )
        or (
            batch_kind != "p4a_post_anchor_expand_context"
            and fields == _FULL_EXPAND_BATCH_PLAN_FIELDS
        )
    ):
        raise ValueError("batch plan kind does not match its exact schema")
    if payload["answer_type"] not in {
        "option_list", "option_single", "logits_match",
    }:
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
        elif payload["answer_type"] == "option_single":
            if option_count != 1 or counts["candidate_answer_calls"] != 1:
                raise ValueError("single-choice batch plan must contain one answer call")
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
        _validate_support_token_contract(payload)
        if payload["p_yes_transform"] != EVIDENCE_SUPPORT_TRANSFORM:
            raise ValueError("batch plan support transform is not frozen")
        if batch_kind == "p2c_post_anchor_coordinate_zoom":
            _validate_zoom_plan(
                payload, require_observation_change=not allow_zoom_noop,
            )
        if batch_kind == "p4a_post_anchor_expand_context":
            _validate_expand_plan(
                payload, require_observation_change=not allow_expand_noop,
            )
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
    """Strict snapshot of one all-or-none post-anchor observation plan."""

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
        if status not in {
            "success", "no_requirements", "render_noop", "budget_rejected", "model_failed",
        }:
            raise ValueError("invalid observation batch status")
        if not isinstance(admitted, bool) or not isinstance(charged, bool):
            raise TypeError("admitted and charged must be booleans")
        expected_state = {
            "success": (True, True),
            "model_failed": (True, True),
            "no_requirements": (False, False),
            "render_noop": (False, False),
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
        plan_without_hash, plan_hash, is_full_plan = _validate_batch_plan(
            batch_plan, allow_zoom_noop=status == "render_noop",
            allow_expand_noop=status == "render_noop",
        )
        if status in {"success", "model_failed"} and not is_full_plan:
            raise ValueError("charged result status requires a full nonzero batch plan")
        if status == "no_requirements" and is_full_plan:
            raise ValueError("no-requirements status requires a zero batch plan")
        if status == "render_noop" and (
            not is_full_plan
            or plan_without_hash["batch_kind"] not in {
                "p2c_post_anchor_coordinate_zoom", "p4a_post_anchor_expand_context",
            }
        ):
            raise ValueError("render-noop status requires a full P2C ZOOM batch plan")
        expected_stages = (
            ("current_support", "candidate_support", *(
                f"hr_answer_{index}" for index in range(4)
            ))
            if plan_without_hash["answer_type"] == "option_list"
            else (
                "current_support", "candidate_support", "treebench_answer",
            ) if plan_without_hash["answer_type"] == "option_single"
            else ("current_support", "candidate_support", "vstar_answer")
        )
        if status == "success" and executed_stages != expected_stages:
            raise ValueError("successful batch executed_stages must be the exact full plan")
        if status == "model_failed":
            if failure_phase == "result_validation":
                expected_prefix = expected_stages
            elif failure_phase in expected_stages:
                expected_prefix = expected_stages[:expected_stages.index(failure_phase)]
            else:
                raise ValueError("model_failed failure_phase is not an exact plan stage")
            if executed_stages != expected_prefix:
                raise ValueError("model_failed executed_stages must be the exact plan prefix")
        elif status != "success" and executed_stages:
            raise ValueError("uncharged batch executed_stages must be empty")
        if status in {"success", "model_failed"} and (
            (current_support is not None) != ("current_support" in executed_stages)
            or (candidate_support is not None) != (
                "candidate_support" in executed_stages
            )
        ):
            raise ValueError("support records must match the exact executed_stages prefix")
        if (
            plan_without_hash["batch_kind"] == "p2c_post_anchor_coordinate_zoom"
            and is_full_plan
        ):
            mapping = plan_without_hash["coordinate_mapping"]
            crop_changed = any(
                item["current_crop_xyxy"] != item["candidate_crop_xyxy"]
                for item in mapping
            )
            view_changed = (
                plan_without_hash["current_observation"]["view_sha256"]
                != plan_without_hash["candidate_observation"]["view_sha256"]
            )
            if status == "render_noop" and crop_changed and view_changed:
                raise ValueError("render-noop status contradicts changed ZOOM observations")
            if status in {"success", "model_failed"} and not (
                crop_changed and view_changed
            ):
                raise ValueError("charged ZOOM requires changed crops and rendered RGB")
        if (
            plan_without_hash["batch_kind"] == "p4a_post_anchor_expand_context"
            and is_full_plan
        ):
            view_changed = (
                plan_without_hash["current_observation"]["view_sha256"]
                != plan_without_hash["candidate_observation"]["view_sha256"]
            )
            if status == "render_noop" and view_changed:
                raise ValueError("render-noop status contradicts changed EXPAND composites")
            if status in {"success", "model_failed"} and not view_changed:
                raise ValueError("charged EXPAND requires changed rendered RGB")
        if status == "success" and (
            current_support is None or candidate_support is None
        ):
            raise ValueError("successful result requires both matching support records")
        if status in {"no_requirements", "render_noop", "budget_rejected"} and (
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
        elif status == "success" and plan_without_hash["answer_type"] == "option_single":
            if not isinstance(candidate_value, str):
                raise ValueError("successful single-choice candidate answer must be a string")
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
    _expected_p0_stability_json: str = field(repr=False, compare=False)
    _p0_options: tuple[str, ...] | None = field(
        default=None, repr=False, compare=False,
    )
    _candidate_options: tuple[str, ...] | None = field(
        default=None, repr=False, compare=False,
    )
    coverage_status: str = "not_observed"
    verifier_status: str = "disabled_same_checkpoint_unpromoted"
    verifier_avg: None = None
    verifier_min: None = None
    score_margin: None = None
    score_status: str = "unavailable_missing_verifier_coverage"
    replacement_reason: str | None = None
    _p0_stability_snapshot_json: str = field(
        init=False, repr=False, compare=False,
    )
    _candidate_stability_snapshot_json: str | None = field(
        init=False, repr=False, compare=False,
    )

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
        p0_stability_json = json.dumps(
            self.p0_stability.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        if (
            not isinstance(self._expected_p0_stability_json, str)
            or p0_stability_json != self._expected_p0_stability_json
        ):
            raise ValueError("P0 stability does not match its trusted snapshot")
        if _json_safe(self.p0_stability.output) != _json_safe(
            self.p0_anchor.emitted_answer
        ):
            raise ValueError("P0 stability output must preserve the exact emitted anchor")
        if abs(uncertainty - _finite_number(
            self.p0_stability.uncertainty, "p0_stability.uncertainty"
        )) > 1e-12:
            raise ValueError("NEXT uncertainty must match immutable P0 stability")
        if self.p0_anchor.producing_phase == "cvsearch_raw":
            p0_raw = self.p0_anchor.cvsearch_raw
            if (
                not isinstance(self._p0_options, tuple)
                or not self._p0_options
                or not isinstance(p0_raw, list)
                or len(self._p0_options) != len(p0_raw)
                or not all(
                    isinstance(option, str) and option
                    for option in self._p0_options
                )
                or not all(isinstance(raw, str) for raw in p0_raw)
            ):
                raise ValueError("HR P0 stability requires exact hidden option blocks")
            from .answers import aggregate_hr_answers
            expected_p0_stability = aggregate_hr_answers(
                list(self._p0_options), p0_raw,
            )
            expected_p0_stability.output = self.p0_anchor.emitted_answer
            expected_p0_stability.selected_from = "cvsearch_raw"
            if self.p0_stability.to_dict() != expected_p0_stability.to_dict():
                raise ValueError("P0 stability is not the canonical recomputation")
        elif self._p0_options is not None:
            raise ValueError("non-HR P0 stability cannot retain option blocks")
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
            if self.support_contract_status != "not_observed":
                raise ValueError("NEXT without a batch must remain not_observed")
            if self._candidate_options is not None:
                raise ValueError("NEXT without candidate stability cannot retain options")
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
                if (
                    not isinstance(self._candidate_options, tuple)
                    or len(self._candidate_options)
                    != self.batch_result.batch_plan["candidate_option_count"]
                    or not all(
                        isinstance(option, str) and option
                        for option in self._candidate_options
                    )
                ):
                    raise ValueError("HR stability requires the exact hidden option blocks")
                from .answers import aggregate_hr_answers
                expected_stability = aggregate_hr_answers(
                    list(self._candidate_options), list(candidate_answer),
                )
            else:
                if self._candidate_options is not None:
                    raise ValueError("V* stability does not retain HR option blocks")
                from .answers import aggregate_vstar_losses
                expected_stability = aggregate_vstar_losses(
                    [candidate_answer["losses"]]
                )
            if self.candidate_stability.to_dict() != expected_stability.to_dict():
                raise ValueError("candidate stability is not the canonical recomputation")
        elif (
            gap is not None
            or delta is not None
            or self.candidate_stability is not None
            or self.feasible
            or self.replacement_reason is not None
        ):
            raise ValueError("unsuccessful NEXT audit cannot synthesize measurements")
        elif self._candidate_options is not None:
            raise ValueError("unsuccessful NEXT audit cannot retain candidate options")

        p0_snapshot = json.dumps(
            self.p0_stability.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        candidate_snapshot = None if self.candidate_stability is None else json.dumps(
            self.candidate_stability.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        object.__setattr__(self, "_p0_stability_snapshot_json", p0_snapshot)
        object.__setattr__(self, "_candidate_stability_snapshot_json", candidate_snapshot)

    @property
    def current_support(self) -> EvidenceSupportResult | None:
        return None if self.batch_result is None else self.batch_result.current_support

    @property
    def candidate_support(self) -> EvidenceSupportResult | None:
        return None if self.batch_result is None else self.batch_result.candidate_support

    def to_dict(self) -> dict[str, Any]:
        current_p0 = json.dumps(
            self.p0_stability.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        current_candidate = None if self.candidate_stability is None else json.dumps(
            self.candidate_stability.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        if (
            current_p0 != self._expected_p0_stability_json
            or current_p0 != self._p0_stability_snapshot_json
            or current_candidate != self._candidate_stability_snapshot_json
        ):
            raise ValueError("NEXT stability material was mutated after construction")
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


@dataclass(frozen=True)
class ZoomAudit:
    """Strict, observation-only audit for one unified P2C coordinate ZOOM."""

    p0_anchor: P0Anchor
    current_keys: tuple[str, ...]
    zoom_keys: tuple[str, ...]
    batch_result: ObservationBatchResult | None
    render_policy: str
    base_view_size: int
    candidate_view_size: int
    uncertainty: float
    uncalibrated_g_zoom_proxy: float | None
    support_delta: float | None
    p0_stability: AnswerRecord
    candidate_stability: AnswerRecord | None
    feasible: bool
    normalized_actual_cost: float | None
    support_contract_status: str
    _expected_p0_stability_json: str = field(repr=False, compare=False)
    _expected_p0_record_json: str = field(repr=False, compare=False)
    _p0_options: tuple[str, ...] | None = field(
        default=None, repr=False, compare=False,
    )
    _candidate_options: tuple[str, ...] | None = field(
        default=None, repr=False, compare=False,
    )
    support_proxy_status: str = "audit_only_uncalibrated"
    coverage_status: str = "not_observed"
    verifier_status: str = "disabled_same_checkpoint_unpromoted"
    verifier_avg: None = None
    verifier_min: None = None
    score_margin: None = None
    score_status: str = "unavailable_missing_verifier_coverage"
    replacement_reason: str | None = None
    _p0_stability_snapshot_json: str = field(
        init=False, repr=False, compare=False,
    )
    _candidate_stability_snapshot_json: str | None = field(
        init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.p0_anchor, P0Anchor):
            raise TypeError("ZOOM audit requires a P0Anchor")
        for name in ("current_keys", "zoom_keys"):
            keys = getattr(self, name)
            if (
                not isinstance(keys, tuple)
                or not all(isinstance(key, str) and key for key in keys)
                or len(keys) != len(set(keys))
            ):
                raise ValueError(f"{name} must be unique nonempty keys")
        if self.current_keys != self.p0_anchor.node_keys:
            raise ValueError("ZOOM current keys must match the exact P0 anchor")
        if self.render_policy != "native_coordinate_crop_view_size_div3":
            raise ValueError("ZOOM audit render policy is not frozen")
        base_size = _integral(self.base_view_size, "base_view_size")
        candidate_size = _integral(self.candidate_view_size, "candidate_view_size")
        if base_size < 4 or candidate_size != base_size // 3 or candidate_size <= 0:
            raise ValueError("ZOOM audit view sizes do not match the render policy")
        uncertainty = _finite_number(self.uncertainty, "uncertainty")
        if not 0.0 <= uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0, 1]")
        if not isinstance(self.p0_stability, AnswerRecord):
            raise TypeError("ZOOM audit requires P0 stability")
        p0_stability_json = json.dumps(
            self.p0_stability.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        if (
            not isinstance(self._expected_p0_stability_json, str)
            or p0_stability_json != self._expected_p0_stability_json
        ):
            raise ValueError("ZOOM P0 stability does not match its trusted snapshot")
        try:
            expected_p0_record = json.loads(self._expected_p0_record_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("ZOOM expected P0 record must be canonical JSON") from error
        if (
            not isinstance(expected_p0_record, dict)
            or json.dumps(
                expected_p0_record, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ) != self._expected_p0_record_json
        ):
            raise ValueError("ZOOM expected P0 record must be canonical JSON")
        if _json_safe(self.p0_stability.output) != _json_safe(
            self.p0_anchor.emitted_answer
        ):
            raise ValueError("ZOOM P0 stability must preserve the exact emitted anchor")
        if abs(uncertainty - _finite_number(
            self.p0_stability.uncertainty, "p0_stability.uncertainty",
        )) > 1e-12:
            raise ValueError("ZOOM uncertainty must match immutable P0 stability")
        if self.p0_anchor.producing_phase == "cvsearch_raw":
            p0_raw = self.p0_anchor.cvsearch_raw
            if (
                not isinstance(self._p0_options, tuple) or not self._p0_options
                or not isinstance(p0_raw, list)
                or len(self._p0_options) != len(p0_raw)
                or not all(isinstance(option, str) and option for option in self._p0_options)
                or not all(isinstance(raw, str) for raw in p0_raw)
            ):
                raise ValueError("HR ZOOM P0 stability requires exact hidden option blocks")
            from .answers import aggregate_hr_answers
            expected_p0 = aggregate_hr_answers(list(self._p0_options), p0_raw)
            expected_p0.output = self.p0_anchor.emitted_answer
            expected_p0.selected_from = "cvsearch_raw"
            if self.p0_stability.to_dict() != expected_p0.to_dict():
                raise ValueError("ZOOM P0 stability is not the canonical recomputation")
        elif self._p0_options is not None:
            raise ValueError("non-HR ZOOM P0 stability cannot retain option blocks")
        if self.candidate_stability is not None and not isinstance(
            self.candidate_stability, AnswerRecord
        ):
            raise TypeError("candidate_stability must be an AnswerRecord or None")
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be boolean")
        if self.support_contract_status not in {"matched", "not_observed", "mismatch"}:
            raise ValueError("invalid ZOOM support contract status")
        if self.support_proxy_status != "audit_only_uncalibrated":
            raise ValueError("ZOOM support proxy must remain explicitly audit-only")
        if self.coverage_status != "not_observed":
            raise ValueError("P2C coverage must remain unobserved")
        if self.verifier_status != "disabled_same_checkpoint_unpromoted":
            raise ValueError("P2C verifier must remain disabled")
        if self.verifier_avg is not None or self.verifier_min is not None:
            raise ValueError("P2C verifier scores must remain null")
        if self.score_margin is not None or self.score_status != (
            "unavailable_missing_verifier_coverage"
        ):
            raise ValueError("P2C selector score must remain unavailable")
        cost = _optional_finite_number(
            self.normalized_actual_cost, "normalized_actual_cost",
        )
        if cost is not None and not 0.0 <= cost <= 1.0:
            raise ValueError("normalized_actual_cost must be in [0, 1]")
        gap = _optional_finite_number(
            self.uncalibrated_g_zoom_proxy, "uncalibrated_g_zoom_proxy",
        )
        delta = _optional_finite_number(self.support_delta, "support_delta")
        if gap is not None and not 0.0 <= gap <= 1.0:
            raise ValueError("uncalibrated_g_zoom_proxy must be in [0, 1]")
        if delta is not None and not -1.0 <= delta <= 1.0:
            raise ValueError("support_delta must be in [-1, 1]")
        if self.batch_result is not None and not isinstance(
            self.batch_result, ObservationBatchResult
        ):
            raise TypeError("batch_result must be an ObservationBatchResult or None")
        if self.batch_result is None:
            if self.zoom_keys or cost is not None:
                raise ValueError("ZOOM without a batch cannot retain zoom keys or cost")
            if self.support_contract_status != "not_observed":
                raise ValueError("ZOOM without a batch must remain not_observed")
        else:
            plan = self.batch_result.batch_plan
            if plan["batch_kind"] != "p2c_post_anchor_coordinate_zoom":
                raise ValueError("ZOOM audit requires a P2C batch")
            if "current_observation" in plan:
                if (
                    plan["current_keys"] != list(self.current_keys)
                    or plan["zoom_keys"] != list(self.zoom_keys)
                    or plan["render_policy"] != self.render_policy
                    or plan["base_view_size"] != base_size
                    or plan["candidate_view_size"] != candidate_size
                ):
                    raise ValueError("ZOOM audit does not match its exact batch plan")
            batch_payload = self.batch_result.to_dict()
            before = batch_payload["ledger_before"]
            after = batch_payload["ledger_after"]
            maximum = after["max_mllm_calls"]
            expected_cost = 0.0 if maximum <= 0 else (
                after["mllm_calls"] - before["mllm_calls"]
            ) / maximum
            if cost is None or abs(cost - expected_cost) > 1e-12:
                raise ValueError("ZOOM normalized cost must match its batch ledger")
            if self.batch_result.status == "success":
                if self.support_contract_status not in {"matched", "mismatch"}:
                    raise ValueError("successful ZOOM requires a support contract audit")
            elif self.support_contract_status != "not_observed":
                raise ValueError("unsuccessful ZOOM cannot claim a support contract match")

        success = (
            self.batch_result is not None
            and self.batch_result.status == "success"
            and self.support_contract_status == "matched"
        )
        if success:
            if (
                len(self.zoom_keys) != len(self.current_keys)
                or not self.zoom_keys
                or gap is None or delta is None
                or self.candidate_stability is None or not self.feasible
                or self.replacement_reason != "replacement_disabled_p2c"
            ):
                raise ValueError("successful ZOOM audit is incomplete")
            current_support = self.batch_result.current_support
            candidate_support = self.batch_result.candidate_support
            if current_support is None or candidate_support is None:
                raise ValueError("successful ZOOM audit lost support results")
            if (
                abs(gap - (1.0 - current_support.p_yes)) > 1e-12
                or abs(delta - (candidate_support.p_yes - current_support.p_yes)) > 1e-12
            ):
                raise ValueError("ZOOM support proxy does not match the atomic batch")
            candidate_answer = self.batch_result.candidate_answer
            if self.batch_result.batch_plan["answer_type"] == "option_list":
                if (
                    not isinstance(self._candidate_options, tuple)
                    or len(self._candidate_options) != 4
                ):
                    raise ValueError("HR ZOOM stability requires exact option blocks")
                from .answers import aggregate_hr_answers
                expected_candidate = aggregate_hr_answers(
                    list(self._candidate_options), list(candidate_answer),
                )
            elif self.batch_result.batch_plan["answer_type"] == "option_single":
                if self._candidate_options is not None:
                    raise ValueError("single-choice ZOOM cannot retain HR options")
                from .answers import aggregate_single_choice
                expected_candidate = aggregate_single_choice(candidate_answer)
            else:
                if self._candidate_options is not None:
                    raise ValueError("V* ZOOM stability cannot retain HR options")
                from .answers import aggregate_vstar_losses
                expected_candidate = aggregate_vstar_losses(
                    [candidate_answer["losses"]],
                )
            if self.candidate_stability.to_dict() != expected_candidate.to_dict():
                raise ValueError("ZOOM candidate stability is not canonical")
        elif (
            gap is not None or delta is not None or self.candidate_stability is not None
            or self.feasible or self.replacement_reason is not None
        ):
            raise ValueError("unsuccessful ZOOM audit cannot synthesize measurements")
        elif self._candidate_options is not None:
            raise ValueError("unsuccessful ZOOM audit cannot retain candidate options")

        candidate_json = None if self.candidate_stability is None else json.dumps(
            self.candidate_stability.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        object.__setattr__(self, "_p0_stability_snapshot_json", p0_stability_json)
        object.__setattr__(self, "_candidate_stability_snapshot_json", candidate_json)

    @property
    def current_support(self) -> EvidenceSupportResult | None:
        return None if self.batch_result is None else self.batch_result.current_support

    @property
    def candidate_support(self) -> EvidenceSupportResult | None:
        return None if self.batch_result is None else self.batch_result.candidate_support

    @property
    def coordinate_mapping(self) -> list[dict[str, Any]]:
        if self.batch_result is None:
            return []
        return copy.deepcopy(self.batch_result.batch_plan.get("coordinate_mapping", []))

    @property
    def focus_key(self) -> str | None:
        if not self.zoom_keys:
            return None
        encoded = json.dumps(list(self.zoom_keys), separators=(",", ":"), allow_nan=False)
        return "zoom-aggregate-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        current_p0 = json.dumps(
            self.p0_stability.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        current_candidate = None if self.candidate_stability is None else json.dumps(
            self.candidate_stability.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        if (
            current_p0 != self._expected_p0_stability_json
            or current_p0 != self._p0_stability_snapshot_json
            or current_candidate != self._candidate_stability_snapshot_json
        ):
            raise ValueError("ZOOM stability material was mutated after construction")
        return {
            "p0_anchor": self.p0_anchor.to_dict(),
            "current_keys": _json_safe(self.current_keys),
            "zoom_keys": _json_safe(self.zoom_keys),
            "coordinate_mapping": _json_safe(self.coordinate_mapping),
            "render_policy": self.render_policy,
            "base_view_size": self.base_view_size,
            "candidate_view_size": self.candidate_view_size,
            "batch_result": None if self.batch_result is None else self.batch_result.to_dict(),
            "current_gap_support": (
                None if self.current_support is None else self.current_support.to_dict()
            ),
            "candidate_gap_support": (
                None if self.candidate_support is None else self.candidate_support.to_dict()
            ),
            "uncertainty": _json_safe(self.uncertainty),
            "uncalibrated_g_zoom_proxy": _json_safe(self.uncalibrated_g_zoom_proxy),
            "support_delta": _json_safe(self.support_delta),
            "support_proxy_status": self.support_proxy_status,
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


@dataclass(frozen=True)
class ExpandAudit:
    """Strict, observation-only audit for one P4A spatial-context EXPAND."""

    p0_anchor: P0Anchor
    current_keys: tuple[str, ...]
    candidate_keys: tuple[str, ...]
    selection_decision: ExpandDecision | None
    batch_result: ObservationBatchResult | None
    selection_policy: str
    composition_policy: str
    uncertainty: float
    uncalibrated_g_expand_proxy: float | None
    support_delta: float | None
    p0_stability: AnswerRecord
    candidate_stability: AnswerRecord | None
    feasible: bool
    normalized_actual_cost: float | None
    support_contract_status: str
    _expected_p0_stability_json: str = field(repr=False, compare=False)
    _expected_p0_record_json: str = field(repr=False, compare=False)
    _expected_selection_json: str = field(repr=False, compare=False)
    _p0_options: tuple[str, ...] | None = field(default=None, repr=False, compare=False)
    _candidate_options: tuple[str, ...] | None = field(default=None, repr=False, compare=False)
    support_proxy_status: str = "audit_only_uncalibrated"
    coverage_status: str = "not_observed"
    verifier_status: str = "disabled_same_checkpoint_unpromoted"
    verifier_avg: None = None
    verifier_min: None = None
    score_margin: None = None
    score_status: str = "unavailable_missing_verifier_coverage"
    replacement_reason: str | None = None
    _p0_stability_snapshot_json: str = field(init=False, repr=False, compare=False)
    _candidate_stability_snapshot_json: str | None = field(
        init=False, repr=False, compare=False,
    )
    _selection_snapshot_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.p0_anchor, P0Anchor):
            raise TypeError("EXPAND audit requires a P0Anchor")
        for name in ("current_keys", "candidate_keys"):
            keys = getattr(self, name)
            if (
                not isinstance(keys, tuple)
                or not all(isinstance(key, str) and key for key in keys)
                or len(keys) != len(set(keys))
            ):
                raise ValueError(f"{name} must be unique nonempty keys")
        if self.current_keys != self.p0_anchor.node_keys:
            raise ValueError("EXPAND current keys must match the exact P0 anchor")
        if self.selection_decision is not None and type(
            self.selection_decision
        ) is not ExpandDecision:
            raise TypeError("EXPAND selection decision must be canonical or null")
        selection_json = json.dumps(
            None if self.selection_decision is None
            else self.selection_decision.to_dict(),
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
        if selection_json != self._expected_selection_json:
            raise ValueError("EXPAND selection differs from its trusted snapshot")
        if self.selection_decision is None:
            if self.candidate_keys or self.batch_result is not None:
                raise ValueError("EXPAND preselection state cannot retain candidate material")
        else:
            decision = self.selection_decision
            if decision.current_keys != self.current_keys:
                raise ValueError("EXPAND selection current keys differ from exact P0")
            expected_focus = [
                descriptor.to_dict() for descriptor in decision.focus_descriptors
            ]
            if (
                not expected_focus
                or self.p0_anchor.to_dict()["support_view"] != expected_focus
            ):
                raise ValueError("EXPAND selection focus differs from exact P0 support")
            if decision.candidate is None:
                if self.candidate_keys or self.batch_result is not None:
                    raise ValueError("EXPAND no-candidate selection retained candidate state")
            elif self.candidate_keys != self.current_keys + (
                decision.candidate.canonical_key,
            ):
                raise ValueError("EXPAND selected context differs from candidate keys")
        if self.selection_policy != "nearest_spatial_native_cvsearch_context_v1":
            raise ValueError("EXPAND audit selection policy is not frozen")
        if self.composition_policy != "focus_top_blank_or_context_bottom_native_pixels_v1":
            raise ValueError("EXPAND audit composition policy is not frozen")
        uncertainty = _finite_number(self.uncertainty, "uncertainty")
        if not 0.0 <= uncertainty <= 1.0:
            raise ValueError("EXPAND uncertainty must be in [0, 1]")
        if not isinstance(self.p0_stability, AnswerRecord):
            raise TypeError("EXPAND audit requires P0 stability")
        p0_stability_json = json.dumps(
            self.p0_stability.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        if p0_stability_json != self._expected_p0_stability_json:
            raise ValueError("EXPAND P0 stability does not match its trusted snapshot")
        try:
            expected_p0_record = json.loads(self._expected_p0_record_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("EXPAND expected P0 record must be canonical JSON") from error
        if json.dumps(
            expected_p0_record, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ) != self._expected_p0_record_json:
            raise ValueError("EXPAND expected P0 record must be canonical JSON")
        if _json_safe(self.p0_stability.output) != _json_safe(self.p0_anchor.emitted_answer):
            raise ValueError("EXPAND P0 stability must preserve the exact emitted anchor")
        if abs(uncertainty - _finite_number(
            self.p0_stability.uncertainty, "p0_stability.uncertainty",
        )) > 1e-12:
            raise ValueError("EXPAND uncertainty must match immutable P0 stability")
        if self.p0_anchor.producing_phase == "cvsearch_raw":
            p0_raw = self.p0_anchor.cvsearch_raw
            if (
                not isinstance(self._p0_options, tuple) or not self._p0_options
                or not isinstance(p0_raw, list)
                or len(self._p0_options) != len(p0_raw)
                or not all(isinstance(option, str) and option for option in self._p0_options)
                or not all(isinstance(raw, str) for raw in p0_raw)
            ):
                raise ValueError("HR EXPAND P0 stability requires exact option blocks")
            from .answers import aggregate_hr_answers
            expected_p0 = aggregate_hr_answers(list(self._p0_options), p0_raw)
            expected_p0.output = self.p0_anchor.emitted_answer
            expected_p0.selected_from = "cvsearch_raw"
            if self.p0_stability.to_dict() != expected_p0.to_dict():
                raise ValueError("EXPAND P0 stability is not canonical")
        elif self._p0_options is not None:
            raise ValueError("non-HR EXPAND P0 stability cannot retain options")
        if self.candidate_stability is not None and not isinstance(
            self.candidate_stability, AnswerRecord,
        ):
            raise TypeError("EXPAND candidate_stability must be an AnswerRecord or None")
        if not isinstance(self.feasible, bool):
            raise TypeError("EXPAND feasible must be boolean")
        if self.support_contract_status not in {"matched", "not_observed", "mismatch"}:
            raise ValueError("invalid EXPAND support contract status")
        if self.support_proxy_status != "audit_only_uncalibrated":
            raise ValueError("EXPAND support proxy must remain audit-only")
        if self.coverage_status != "not_observed":
            raise ValueError("P4A coverage must remain unobserved")
        if (
            self.verifier_status != "disabled_same_checkpoint_unpromoted"
            or self.verifier_avg is not None or self.verifier_min is not None
            or self.score_margin is not None
            or self.score_status != "unavailable_missing_verifier_coverage"
        ):
            raise ValueError("P4A verifier/selector must remain unavailable")
        cost = _optional_finite_number(
            self.normalized_actual_cost, "normalized_actual_cost",
        )
        if cost is not None and not 0.0 <= cost <= 1.0:
            raise ValueError("EXPAND normalized cost must be in [0, 1]")
        gap = _optional_finite_number(
            self.uncalibrated_g_expand_proxy, "uncalibrated_g_expand_proxy",
        )
        delta = _optional_finite_number(self.support_delta, "support_delta")
        if gap is not None and not 0.0 <= gap <= 1.0:
            raise ValueError("EXPAND gap proxy must be in [0, 1]")
        if delta is not None and not -1.0 <= delta <= 1.0:
            raise ValueError("EXPAND support delta must be in [-1, 1]")
        if self.batch_result is not None and not isinstance(
            self.batch_result, ObservationBatchResult,
        ):
            raise TypeError("EXPAND batch_result must be an ObservationBatchResult or None")
        if self.batch_result is None:
            if cost is not None:
                raise ValueError("EXPAND without a batch cannot report cost")
            if self.support_contract_status != "not_observed":
                raise ValueError("EXPAND without a batch must remain not_observed")
            if self.candidate_keys and self.candidate_keys[:-1] != self.current_keys:
                raise ValueError("EXPAND preflight candidate must append to exact P0 keys")
        else:
            plan = self.batch_result.batch_plan
            if plan["batch_kind"] != "p4a_post_anchor_expand_context":
                raise ValueError("EXPAND audit requires a P4A batch")
            if "current_observation" in plan and (
                plan["current_keys"] != list(self.current_keys)
                or plan["candidate_keys"] != list(self.candidate_keys)
                or plan["selection_policy"] != self.selection_policy
                or plan["composition_policy"] != self.composition_policy
            ):
                raise ValueError("EXPAND audit differs from its exact batch plan")
            if self.selection_decision is None or self.selection_decision.candidate is None:
                raise ValueError("EXPAND batch requires an independently frozen selection")
            decision = self.selection_decision
            if "current_observation" in plan and (
                plan["focus_role"]["descriptors"]
                != [item.to_dict() for item in decision.focus_descriptors]
                or plan["context_role"]["descriptor"]
                != decision.candidate.to_dict()
                or plan["context_role"]["focus_union_xyxy"]
                != list(decision.focus_union_xyxy)
                or plan["context_role"]["positive_outside_area"]
                != decision.positive_outside_area
                or plan["context_role"]["normalized_edge_gap"]
                != decision.normalized_edge_gap
                or plan["context_role"]["rank_tuple"] != list(decision.rank_tuple)
            ):
                raise ValueError("EXPAND batch differs from frozen selection provenance")
            batch_payload = self.batch_result.to_dict()
            before = batch_payload["ledger_before"]
            after = batch_payload["ledger_after"]
            maximum = after["max_mllm_calls"]
            expected_cost = 0.0 if maximum <= 0 else (
                after["mllm_calls"] - before["mllm_calls"]
            ) / maximum
            if cost is None or abs(cost - expected_cost) > 1e-12:
                raise ValueError("EXPAND normalized cost differs from batch ledger")
            if self.batch_result.status == "success":
                if self.support_contract_status not in {"matched", "mismatch"}:
                    raise ValueError("successful EXPAND requires support contract audit")
            elif self.support_contract_status != "not_observed":
                raise ValueError("unsuccessful EXPAND cannot claim support match")

        success = (
            self.batch_result is not None and self.batch_result.status == "success"
            and self.support_contract_status == "matched"
        )
        if success:
            if (
                self.candidate_keys[:-1] != self.current_keys
                or len(self.candidate_keys) != len(self.current_keys) + 1
                or gap is None or delta is None or self.candidate_stability is None
                or not self.feasible
                or self.replacement_reason != "replacement_disabled_p4a"
            ):
                raise ValueError("successful EXPAND audit is incomplete")
            current_support = self.batch_result.current_support
            candidate_support = self.batch_result.candidate_support
            if current_support is None or candidate_support is None:
                raise ValueError("successful EXPAND audit lost support results")
            if (
                abs(gap - (1.0 - current_support.p_yes)) > 1e-12
                or abs(delta - (candidate_support.p_yes - current_support.p_yes)) > 1e-12
            ):
                raise ValueError("EXPAND support proxy differs from atomic batch")
            candidate_answer = self.batch_result.candidate_answer
            if self.batch_result.batch_plan["answer_type"] == "option_list":
                if not isinstance(self._candidate_options, tuple) or len(self._candidate_options) != 4:
                    raise ValueError("HR EXPAND stability requires exact options")
                from .answers import aggregate_hr_answers
                expected_candidate = aggregate_hr_answers(
                    list(self._candidate_options), list(candidate_answer),
                )
            elif self.batch_result.batch_plan["answer_type"] == "option_single":
                if self._candidate_options is not None:
                    raise ValueError("single-choice EXPAND cannot retain HR options")
                from .answers import aggregate_single_choice
                expected_candidate = aggregate_single_choice(candidate_answer)
            else:
                if self._candidate_options is not None:
                    raise ValueError("V* EXPAND stability cannot retain HR options")
                from .answers import aggregate_vstar_losses
                expected_candidate = aggregate_vstar_losses([candidate_answer["losses"]])
            if self.candidate_stability.to_dict() != expected_candidate.to_dict():
                raise ValueError("EXPAND candidate stability is not canonical")
        elif (
            gap is not None or delta is not None or self.candidate_stability is not None
            or self.feasible or self.replacement_reason is not None
        ):
            raise ValueError("unsuccessful EXPAND audit cannot synthesize measurements")
        elif self._candidate_options is not None:
            raise ValueError("unsuccessful EXPAND audit cannot retain candidate options")

        candidate_json = None if self.candidate_stability is None else json.dumps(
            self.candidate_stability.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        object.__setattr__(self, "_p0_stability_snapshot_json", p0_stability_json)
        object.__setattr__(self, "_candidate_stability_snapshot_json", candidate_json)
        object.__setattr__(self, "_selection_snapshot_json", selection_json)

    @property
    def current_support(self) -> EvidenceSupportResult | None:
        return None if self.batch_result is None else self.batch_result.current_support

    @property
    def candidate_support(self) -> EvidenceSupportResult | None:
        return None if self.batch_result is None else self.batch_result.candidate_support

    @property
    def focus_key(self) -> str | None:
        if len(self.candidate_keys) != len(self.current_keys) + 1:
            return None
        return self.candidate_keys[-1]

    def _plan_material(self, name: str, default: Any) -> Any:
        if self.batch_result is None:
            return copy.deepcopy(default)
        return copy.deepcopy(self.batch_result.batch_plan.get(name, default))

    def to_dict(self) -> dict[str, Any]:
        current_p0 = json.dumps(
            self.p0_stability.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        current_candidate = None if self.candidate_stability is None else json.dumps(
            self.candidate_stability.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        current_selection = json.dumps(
            None if self.selection_decision is None
            else self.selection_decision.to_dict(),
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
        if (
            current_p0 != self._expected_p0_stability_json
            or current_p0 != self._p0_stability_snapshot_json
            or current_candidate != self._candidate_stability_snapshot_json
            or current_selection != self._expected_selection_json
            or current_selection != self._selection_snapshot_json
        ):
            raise ValueError("EXPAND immutable audit material changed after construction")
        return {
            "p0_anchor": self.p0_anchor.to_dict(),
            "current_keys": _json_safe(self.current_keys),
            "candidate_keys": _json_safe(self.candidate_keys),
            "selection_decision": (
                None if self.selection_decision is None
                else self.selection_decision.to_dict()
            ),
            "selection_decision_sha256": hashlib.sha256(
                current_selection.encode("utf-8")
            ).hexdigest(),
            "selection_policy": self.selection_policy,
            "composition_policy": self.composition_policy,
            "focus_role": self._plan_material("focus_role", None),
            "context_role": self._plan_material("context_role", None),
            "focus_merge_identity": self._plan_material("focus_merge_identity", None),
            "context_merge_identity": self._plan_material("context_merge_identity", None),
            "composition_identity": self._plan_material("composition_identity", None),
            "batch_result": None if self.batch_result is None else self.batch_result.to_dict(),
            "current_gap_support": (
                None if self.current_support is None else self.current_support.to_dict()
            ),
            "candidate_gap_support": (
                None if self.candidate_support is None else self.candidate_support.to_dict()
            ),
            "uncertainty": _json_safe(self.uncertainty),
            "uncalibrated_g_expand_proxy": _json_safe(self.uncalibrated_g_expand_proxy),
            "support_delta": _json_safe(self.support_delta),
            "support_proxy_status": self.support_proxy_status,
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


def _split_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return value.lower()


def _split_path(value: Any, name: str) -> tuple[int, ...]:
    if (
        type(value) is not tuple or not 1 <= len(value) <= 2
        or any(type(index) is not int or not 0 <= index < 4 for index in value)
    ):
        raise ValueError(f"{name} must identify a depth-one or depth-two split patch")
    return value


def _split_xyxy(
    value: Any, name: str, source_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    if (
        type(value) is not tuple or len(value) != 4
        or any(type(coordinate) is not int for coordinate in value)
    ):
        raise TypeError(f"{name} must be an exact integer XYXY tuple")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= source_size[0] and 0 <= y0 < y1 <= source_size[1]):
        raise ValueError(f"{name} is outside the source image")
    return value


@dataclass(frozen=True)
class SplitProbeObservation:
    """One answer-free support probe used to navigate depth-two candidates."""

    patch_path: tuple[int, ...]
    target_box_xyxy: tuple[int, int, int, int]
    source_size: tuple[int, int]
    render_sha256: str
    raw_support: float
    ranking_score: float
    mllm_calls: int
    processed_pixels: int

    def __post_init__(self) -> None:
        path = _split_path(self.patch_path, "split probe path")
        if len(path) != 2:
            raise ValueError("split probe must bind one depth-two patch")
        if (
            type(self.source_size) is not tuple or len(self.source_size) != 2
            or any(type(value) is not int or value <= 0 for value in self.source_size)
        ):
            raise ValueError("split probe source size must contain positive integers")
        _split_xyxy(self.target_box_xyxy, "split probe box", self.source_size)
        _split_sha256(self.render_sha256, "split probe render hash")
        for value, name in (
            (self.raw_support, "split probe support"),
            (self.ranking_score, "split probe ranking score"),
        ):
            number = _finite_number(value, name)
            if not 0 <= number <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if _integral(self.mllm_calls, "split probe MLLM calls") != 1:
            raise ValueError("split probe must consume exactly one support call")
        if _integral(self.processed_pixels, "split probe pixels") <= 0:
            raise ValueError("split probe pixels must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "patch_path": _json_safe(self.patch_path),
            "target_box_xyxy": _json_safe(self.target_box_xyxy),
            "source_size": _json_safe(self.source_size),
            "render_sha256": self.render_sha256,
            "raw_support": _json_safe(self.raw_support),
            "ranking_score": _json_safe(self.ranking_score),
            "mllm_calls": self.mllm_calls,
            "processed_pixels": self.processed_pixels,
        }


@dataclass(frozen=True)
class SplitViewObservation:
    """One answer-bearing native view of a ranked split child."""

    role: str
    patch_path: tuple[int, ...]
    target_box_xyxy: tuple[int, int, int, int]
    crop_xyxy: tuple[int, int, int, int]
    source_size: tuple[int, int]
    render_sha256: str
    raw_support: float
    answer: Any
    mllm_calls: int
    processed_pixels: int
    _answer_snapshot_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.role not in {"tight", "medium", "context"}:
            raise ValueError("split view role must be tight, medium, or context")
        _split_path(self.patch_path, "split patch path")
        if (
            type(self.source_size) is not tuple or len(self.source_size) != 2
            or any(type(value) is not int or value <= 0 for value in self.source_size)
        ):
            raise ValueError("split source size must contain two positive integers")
        target = _split_xyxy(
            self.target_box_xyxy, "split target box", self.source_size,
        )
        crop = _split_xyxy(self.crop_xyxy, "split crop", self.source_size)
        if not (
            crop[0] <= target[0] and crop[1] <= target[1]
            and target[2] <= crop[2] and target[3] <= crop[3]
        ):
            raise ValueError("split crop must contain its target child")
        if self.role == "tight" and crop != target:
            raise ValueError("tight split crop must equal its target child")
        if self.role in {"medium", "context"} and crop == target:
            raise ValueError("context-preserving split crop must differ from target")
        _split_sha256(self.render_sha256, "split render hash")
        support = _finite_number(self.raw_support, "split raw support")
        if not 0 <= support <= 1:
            raise ValueError("split raw support must be in [0, 1]")
        calls = _integral(self.mllm_calls, "split MLLM calls")
        pixels = _integral(self.processed_pixels, "split processed pixels")
        if calls <= 0 or pixels <= 0:
            raise ValueError("split view costs must be positive")
        answer_json = json.dumps(
            _json_safe(self.answer), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        object.__setattr__(self, "_answer_snapshot_json", answer_json)

    def to_dict(self) -> dict[str, Any]:
        current_answer = json.dumps(
            _json_safe(self.answer), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        if current_answer != self._answer_snapshot_json:
            raise ValueError("split view answer was mutated after construction")
        return {
            "role": self.role,
            "patch_path": _json_safe(self.patch_path),
            "target_box_xyxy": _json_safe(self.target_box_xyxy),
            "crop_xyxy": _json_safe(self.crop_xyxy),
            "source_size": _json_safe(self.source_size),
            "render_sha256": self.render_sha256,
            "raw_support": _json_safe(self.raw_support),
            "answer": json.loads(self._answer_snapshot_json),
            "mllm_calls": self.mllm_calls,
            "processed_pixels": self.processed_pixels,
        }


@dataclass(frozen=True)
class SplitBranchObservation:
    """Two- or three-scale observation plus its complete sibling ranking."""

    visit_index: int
    selected_sibling_rank: int
    ranked_sibling_paths: tuple[tuple[int, ...], ...]
    ranked_sibling_boxes: tuple[tuple[int, int, int, int], ...]
    ranked_sibling_scores: tuple[float, ...]
    tight_view: SplitViewObservation
    context_view: SplitViewObservation
    backtracked: bool
    medium_view: SplitViewObservation | None = None

    def __post_init__(self) -> None:
        visit_index = _integral(self.visit_index, "split visit index")
        if visit_index not in {0, 1, 2, 3, 4, 5}:
            raise ValueError("split visit index must be in [0, 5]")
        selected_rank = _integral(
            self.selected_sibling_rank, "selected split sibling rank",
        )
        if not 0 <= selected_rank < 4:
            raise ValueError("selected split sibling rank must be in [0, 3]")
        if not isinstance(self.backtracked, bool) or (
            visit_index == 0 and self.backtracked
        ):
            raise ValueError("the initial split branch cannot be a backtrack")
        if not isinstance(self.tight_view, SplitViewObservation) or not isinstance(
            self.context_view, SplitViewObservation
        ):
            raise TypeError("split branch requires exact tight and context views")
        if self.tight_view.role != "tight" or self.context_view.role != "context":
            raise ValueError("split branch view roles are reversed")
        if (visit_index < 4) != (self.medium_view is None):
            raise ValueError("only Stage 3b rescue branches require a medium view")
        views = [self.tight_view, self.context_view]
        if self.medium_view is not None:
            if not isinstance(self.medium_view, SplitViewObservation):
                raise TypeError("split branch medium view must be exact")
            if self.medium_view.role != "medium":
                raise ValueError("split branch medium view role is invalid")
            views.append(self.medium_view)
        if any(
            view.patch_path != self.tight_view.patch_path
            or view.target_box_xyxy != self.tight_view.target_box_xyxy
            or view.source_size != self.tight_view.source_size
            for view in views[1:]
        ):
            raise ValueError("split branch views must bind the same target child")
        if len({view.render_sha256.lower() for view in views}) != len(views):
            raise ValueError("split branch views must have distinct render hashes")
        sequences = (
            self.ranked_sibling_paths,
            self.ranked_sibling_boxes,
            self.ranked_sibling_scores,
        )
        if any(type(value) is not tuple or len(value) != 4 for value in sequences):
            raise ValueError("split branch must retain all four ranked siblings")
        paths = tuple(
            _split_path(path, f"split sibling path {index}")
            for index, path in enumerate(self.ranked_sibling_paths)
        )
        if len(set(paths)) != 4 or len({path[:-1] for path in paths}) != 1:
            raise ValueError("split sibling paths must be unique children of one parent")
        boxes = tuple(
            _split_xyxy(
                box, f"split sibling box {index}", self.tight_view.source_size,
            )
            for index, box in enumerate(self.ranked_sibling_boxes)
        )
        if len(set(boxes)) != 4:
            raise ValueError("split sibling boxes must be unique")
        scores = tuple(
            _finite_number(score, f"split sibling score {index}")
            for index, score in enumerate(self.ranked_sibling_scores)
        )
        if tuple(sorted(scores, reverse=True)) != scores:
            raise ValueError("split sibling scores must retain descending rank order")
        if self.tight_view.patch_path != paths[selected_rank]:
            raise ValueError("observed split path must match its sibling rank")
        path_index = paths.index(self.tight_view.patch_path)
        if boxes[path_index] != self.tight_view.target_box_xyxy:
            raise ValueError("observed split box differs from its ranked sibling box")

    @property
    def observed_path(self) -> tuple[int, ...]:
        return self.tight_view.patch_path

    @property
    def focus_key(self) -> str:
        return "p" + ".".join(map(str, self.observed_path))

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        result = {
            "visit_index": self.visit_index,
            "selected_sibling_rank": self.selected_sibling_rank,
            "observed_path": _json_safe(self.observed_path),
            "ranked_siblings": [
                {"path": _json_safe(path), "box": _json_safe(box), "score": score}
                for path, box, score in zip(
                    self.ranked_sibling_paths,
                    self.ranked_sibling_boxes,
                    self.ranked_sibling_scores,
                )
            ],
            "tight_view": self.tight_view.to_dict(),
            "context_view": self.context_view.to_dict(),
            "backtracked": self.backtracked,
        }
        if self.medium_view is not None:
            result["medium_view"] = self.medium_view.to_dict()
        return result


@dataclass(frozen=True)
class SplitSearchAudit:
    """Immutable candidate-only audit for bounded Stage 3 split search."""

    p0_anchor: P0Anchor
    p0_stability: AnswerRecord
    branches: tuple[SplitBranchObservation, ...]
    root_ranked_paths: tuple[tuple[int, ...], ...]
    root_ranked_boxes: tuple[tuple[int, int, int, int], ...]
    root_ranked_scores: tuple[float, ...]
    rank_sha256: str
    query_sha256: str
    render_policy: str
    ledger_before: BudgetLedger
    ledger_after: BudgetLedger
    screening_probes: tuple[SplitProbeObservation, ...] = ()
    no_op_reason: str | None = None
    _p0_snapshot_json: str = field(init=False, repr=False, compare=False)
    _ledger_before_snapshot_json: str = field(init=False, repr=False, compare=False)
    _ledger_after_snapshot_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.p0_anchor, P0Anchor):
            raise TypeError("split audit requires a P0Anchor")
        if not isinstance(self.p0_stability, AnswerRecord):
            raise TypeError("split audit requires P0 stability")
        if _json_safe(self.p0_stability.output) != _json_safe(
            self.p0_anchor.emitted_answer
        ):
            raise ValueError("split P0 stability differs from its exact anchor")
        if type(self.branches) is not tuple or len(self.branches) > 6 or not all(
            isinstance(branch, SplitBranchObservation) for branch in self.branches
        ):
            raise ValueError("split audit permits at most six exact branches")
        if tuple(branch.visit_index for branch in self.branches) != tuple(
            range(len(self.branches))
        ):
            raise ValueError("split audit branches must retain visit order")
        if len({branch.observed_path for branch in self.branches}) != len(self.branches):
            raise ValueError("split audit cannot revisit a branch")
        if type(self.screening_probes) is not tuple or not all(
            isinstance(probe, SplitProbeObservation)
            for probe in self.screening_probes
        ):
            raise TypeError("split screening probes must be an exact tuple")
        if self.screening_probes and len(self.screening_probes) not in {8, 16}:
            raise ValueError("split screening must cover eight or sixteen leaves")
        if len({probe.patch_path for probe in self.screening_probes}) != len(
            self.screening_probes
        ):
            raise ValueError("split screening cannot probe a path twice")
        hashes = [
            view.render_sha256.lower()
            for branch in self.branches
            for view in (
                branch.tight_view, branch.medium_view, branch.context_view,
            )
            if view is not None
        ]
        if len(hashes) != len(set(hashes)):
            raise ValueError("split audit render hashes must be globally distinct")
        root_sequences = (
            self.root_ranked_paths, self.root_ranked_boxes, self.root_ranked_scores,
        )
        if self.branches:
            if any(type(value) is not tuple or len(value) != 4 for value in root_sequences):
                raise ValueError("split audit must retain all four ranked root children")
            root_paths = tuple(
                _split_path(path, f"split root path {index}")
                for index, path in enumerate(self.root_ranked_paths)
            )
            if any(len(path) != 1 for path in root_paths) or len(set(root_paths)) != 4:
                raise ValueError("split root ranking must contain four depth-one paths")
            source_size = self.branches[0].tight_view.source_size
            root_boxes = tuple(
                _split_xyxy(box, f"split root box {index}", source_size)
                for index, box in enumerate(self.root_ranked_boxes)
            )
            if len(set(root_boxes)) != 4:
                raise ValueError("split root ranking boxes must be unique")
            root_scores = tuple(
                _finite_number(score, f"split root score {index}")
                for index, score in enumerate(self.root_ranked_scores)
            )
            if tuple(sorted(root_scores, reverse=True)) != root_scores:
                raise ValueError("split root scores must retain descending rank order")
            if self.screening_probes:
                probe_roots = {probe.patch_path[:1] for probe in self.screening_probes}
                expected_roots = (
                    set(root_paths) if len(self.screening_probes) == 16
                    else set(root_paths[:2])
                )
                if probe_roots != expected_roots or any(
                    sum(probe.patch_path[:1] == root for probe in self.screening_probes) != 4
                    for root in expected_roots
                ):
                    raise ValueError("split screening does not cover its ranked roots")
                probes_by_path = {
                    probe.patch_path: probe for probe in self.screening_probes
                }
                for branch in self.branches:
                    probe = probes_by_path.get(branch.observed_path)
                    if probe is None:
                        raise ValueError("observed branch was not support-screened")
                    if (
                        probe.render_sha256.lower()
                        != branch.tight_view.render_sha256.lower()
                        or probe.raw_support != branch.tight_view.raw_support
                    ):
                        raise ValueError("observed tight view differs from its support probe")
            else:
                for branch in self.branches:
                    if branch.observed_path[0] != root_paths[branch.visit_index][0]:
                        raise ValueError("split branch does not descend from its ranked root")
        elif any(root_sequences):
            raise ValueError("empty split audit cannot expose a root ranking")
        if not self.branches and self.screening_probes:
            raise ValueError("empty split audit cannot expose screening probes")
        _split_sha256(self.rank_sha256, "split rank hash")
        _split_sha256(self.query_sha256, "split query hash")
        if self.render_policy not in {
            "native_2x2_overlap_two_scale_depth2_v1",
            "native_2x2_overlap_support_screen_two_scale_depth2_v2",
            "native_2x2_overlap_support_screen_three_scale_all_roots_depth2_v3",
        }:
            raise ValueError("split render policy is not frozen")
        screened_policy = self.render_policy != (
            "native_2x2_overlap_two_scale_depth2_v1"
        )
        if self.branches and screened_policy != bool(self.screening_probes):
            raise ValueError("split render policy differs from screening provenance")
        stage3b_policy = self.render_policy == (
            "native_2x2_overlap_support_screen_three_scale_all_roots_depth2_v3"
        )
        if self.branches and stage3b_policy != (
            len(self.branches) == 6 and len(self.screening_probes) == 16
        ):
            raise ValueError("Stage 3b policy requires an exact all-root cascade")
        if not isinstance(self.ledger_before, BudgetLedger) or not isinstance(
            self.ledger_after, BudgetLedger
        ):
            raise TypeError("split audit requires before and after budget ledgers")
        before = self.ledger_before.to_dict()
        after = self.ledger_after.to_dict()
        if (
            before["max_mllm_calls"] != after["max_mllm_calls"]
            or before["max_processed_pixels"] != after["max_processed_pixels"]
        ):
            raise ValueError("split budget limits changed during observation")
        expected_calls = sum(
            view.mllm_calls
            for branch in self.branches
            for view in (
                branch.tight_view, branch.medium_view, branch.context_view,
            )
            if view is not None
        ) + sum(probe.mllm_calls for probe in self.screening_probes)
        expected_pixels = sum(
            view.processed_pixels
            for branch in self.branches
            for view in (
                branch.tight_view, branch.medium_view, branch.context_view,
            )
            if view is not None
        ) + sum(probe.processed_pixels for probe in self.screening_probes)
        if (
            after["mllm_calls"] - before["mllm_calls"] != expected_calls
            or after["processed_pixels"] - before["processed_pixels"] != expected_pixels
        ):
            raise ValueError("split budget delta does not match observed views")
        if self.branches:
            if self.no_op_reason is not None:
                raise ValueError("observed split branches cannot report a no-op")
        elif not isinstance(self.no_op_reason, str) or not self.no_op_reason.startswith(
            "split_"
        ):
            raise ValueError("empty split audit requires an exact split no-op reason")
        snapshots = (
            json.dumps(
                self.p0_stability.to_dict(), sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ),
            json.dumps(before, sort_keys=True, separators=(",", ":"), allow_nan=False),
            json.dumps(after, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )
        object.__setattr__(self, "_p0_snapshot_json", snapshots[0])
        object.__setattr__(self, "_ledger_before_snapshot_json", snapshots[1])
        object.__setattr__(self, "_ledger_after_snapshot_json", snapshots[2])

    @property
    def focus_key(self) -> str | None:
        return None if not self.branches else self.branches[0].focus_key

    def to_dict(self) -> dict[str, Any]:
        current = (
            json.dumps(
                self.p0_stability.to_dict(), sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ),
            json.dumps(
                self.ledger_before.to_dict(), sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ),
            json.dumps(
                self.ledger_after.to_dict(), sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ),
        )
        expected = (
            self._p0_snapshot_json,
            self._ledger_before_snapshot_json,
            self._ledger_after_snapshot_json,
        )
        if current != expected:
            raise ValueError("split audit nested material was mutated after construction")
        stage3b_policy = self.render_policy == (
            "native_2x2_overlap_support_screen_three_scale_all_roots_depth2_v3"
        )
        return {
            "p0_anchor": self.p0_anchor.to_dict(),
            "p0_stability": json.loads(self._p0_snapshot_json),
            "root_ranked_siblings": [
                {"path": _json_safe(path), "box": _json_safe(box), "score": score}
                for path, box, score in zip(
                    self.root_ranked_paths,
                    self.root_ranked_boxes,
                    self.root_ranked_scores,
                )
            ],
            "branches": [branch.to_dict() for branch in self.branches],
            "screening_probes": [
                probe.to_dict() for probe in self.screening_probes
            ],
            "rank_sha256": self.rank_sha256,
            "query_sha256": self.query_sha256,
            "render_policy": self.render_policy,
            "max_depth": 2,
            "max_observed_branches": 6 if stage3b_policy else 4,
            "max_screening_probes": 16 if stage3b_policy else 8,
            "ledger_before": json.loads(self._ledger_before_snapshot_json),
            "ledger_after": json.loads(self._ledger_after_snapshot_json),
            "no_op_reason": self.no_op_reason,
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
    zoom_audit: ZoomAudit | None = None
    expand_audit: ExpandAudit | None = None
    split_search_audit: SplitSearchAudit | None = None
    _answer_snapshot_json: str | None = field(
        default=None, repr=False, compare=False,
    )
    _zoom_answer_snapshot_json: str | None = field(
        default=None, init=False, repr=False, compare=False,
    )
    _expand_answer_snapshot_json: str | None = field(
        default=None, init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        if sum(
            audit is not None
            for audit in (
                self.next_audit, self.zoom_audit, self.expand_audit,
                self.split_search_audit,
            )
        ) > 1:
            raise ValueError("a step cannot contain multiple observation audits")
        if self.split_search_audit is not None:
            audit = self.split_search_audit
            if not isinstance(audit, SplitSearchAudit):
                raise TypeError("split_search_audit must be a SplitSearchAudit")
            if self.gap_fallback_used is not False or self.certified is not False:
                raise ValueError("SPLIT StepTrace fixed boolean fields must remain false")
            for value, name in (
                (self.elapsed_seconds, "elapsed_seconds"),
                (self.support_avg, "support_avg"),
                (self.support_min, "support_min"),
            ):
                if _finite_number(value, name) != 0.0:
                    raise ValueError(f"SPLIT StepTrace {name} must remain zero")
            if self.action != SPLIT:
                raise ValueError("SPLIT audit can only be attached to action=SPLIT")
            if self.focus_key != audit.focus_key:
                raise ValueError("SPLIT focus key must match its first observed branch")
            expected_actions = (SPLIT,) if audit.branches else ()
            if self.feasible_actions != expected_actions:
                raise ValueError("SPLIT feasible actions do not match its strict audit")
            if self.gaps:
                raise ValueError("SPLIT StepTrace cannot expose uncalibrated gap scores")
            if self.no_op_reason != audit.no_op_reason:
                raise ValueError("SPLIT no-op reason differs from its audit")
            if self.answer is None or self.answer.to_dict() != self.split_search_audit.p0_stability.to_dict():
                raise ValueError("SPLIT StepTrace must retain the exact P0 answer record")
            if self.budget is None or self.budget.to_dict() != audit.ledger_after.to_dict():
                raise ValueError("SPLIT StepTrace budget must match the audit ledger")
            audit.to_dict()
            return
        if self.expand_audit is not None:
            if not isinstance(self.expand_audit, ExpandAudit):
                raise TypeError("expand_audit must be an ExpandAudit")
            if self.gap_fallback_used is not False or self.certified is not False:
                raise ValueError("EXPAND StepTrace fixed boolean fields must remain false")
            for value, name in (
                (self.elapsed_seconds, "elapsed_seconds"),
                (self.support_avg, "support_avg"),
                (self.support_min, "support_min"),
            ):
                if _finite_number(value, name) != 0.0:
                    raise ValueError(f"EXPAND StepTrace {name} must remain zero")
            if self.action != EXPAND:
                raise ValueError("EXPAND audit can only be attached to action=EXPAND")
            if self.focus_key != self.expand_audit.focus_key:
                raise ValueError("EXPAND focus key must match its context candidate")
            expected_actions = (EXPAND,) if self.expand_audit.feasible else ()
            if self.feasible_actions != expected_actions:
                raise ValueError("EXPAND feasible actions do not match its strict audit")
            expected_gaps = (
                {
                    "g_expand_proxy_audit_only":
                    self.expand_audit.uncalibrated_g_expand_proxy,
                }
                if self.expand_audit.uncalibrated_g_expand_proxy is not None else {}
            )
            if self.gaps != expected_gaps:
                raise ValueError("EXPAND StepTrace gaps do not match its audit-only proxy")
            batch = self.expand_audit.batch_result
            if self.expand_audit.feasible:
                if self.no_op_reason is not None:
                    raise ValueError("successful EXPAND cannot report a no-op reason")
            elif batch is None:
                early_reasons = {
                    "expand_invalid_evidence_requirements",
                    "expand_no_evidence_requirements",
                    "expand_p0_focus_unavailable",
                    "expand_p0_focus_empty",
                    "expand_p0_focus_nonlocal",
                }
                decision = self.expand_audit.selection_decision
                if decision is None:
                    if self.no_op_reason not in early_reasons:
                        raise ValueError("EXPAND preselection no-op reason is not exact")
                elif decision.candidate is None:
                    expected_reason = decision.no_op_reason.value
                    if self.no_op_reason != expected_reason:
                        raise ValueError("EXPAND selection no-op differs from its decision")
                elif self.no_op_reason != "expand_batch_preflight_failed":
                    raise ValueError("EXPAND selected context without batch must be preflight failure")
            else:
                expected_reason = (
                    "expand_support_contract_mismatch"
                    if batch.status == "success" else f"expand_{batch.status}"
                )
                if self.no_op_reason != expected_reason:
                    raise ValueError("EXPAND no-op reason does not match its batch status")
            if self.answer is None or _json_safe(self.answer.output) != _json_safe(
                self.expand_audit.p0_anchor.emitted_answer
            ):
                raise ValueError("EXPAND StepTrace must retain the exact P0 answer")
            answer_snapshot = json.dumps(
                self.answer.to_dict(), sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            )
            if answer_snapshot != self.expand_audit._expected_p0_record_json:
                raise ValueError("EXPAND StepTrace answer snapshot differs from exact P0 record")
            if self._expand_answer_snapshot_json is None:
                self._expand_answer_snapshot_json = answer_snapshot
            elif answer_snapshot != self._expand_answer_snapshot_json:
                raise ValueError("EXPAND StepTrace answer snapshot changed")
            if self.budget is None:
                raise ValueError("EXPAND StepTrace requires its post-attempt budget")
            if batch is not None:
                expected_budget = batch.to_dict()["ledger_after"]
                if self.budget.to_dict() != expected_budget:
                    raise ValueError("EXPAND StepTrace budget must match the batch ledger")
            return
        if self.zoom_audit is not None:
            if not isinstance(self.zoom_audit, ZoomAudit):
                raise TypeError("zoom_audit must be a ZoomAudit")
            if self.gap_fallback_used is not False or self.certified is not False:
                raise ValueError("ZOOM StepTrace fixed boolean fields must remain false")
            for value, name in (
                (self.elapsed_seconds, "elapsed_seconds"),
                (self.support_avg, "support_avg"),
                (self.support_min, "support_min"),
            ):
                if _finite_number(value, name) != 0.0:
                    raise ValueError(f"ZOOM StepTrace {name} must remain zero")
            if self.action != ZOOM:
                raise ValueError("ZOOM audit can only be attached to action=ZOOM")
            if self.focus_key != self.zoom_audit.focus_key:
                raise ValueError("ZOOM focus key must match its aggregate render identity")
            expected_actions = (ZOOM,) if self.zoom_audit.feasible else ()
            if self.feasible_actions != expected_actions:
                raise ValueError("ZOOM feasible actions do not match its strict audit")
            expected_gaps = (
                {"g_zoom_proxy_audit_only": self.zoom_audit.uncalibrated_g_zoom_proxy}
                if self.zoom_audit.uncalibrated_g_zoom_proxy is not None else {}
            )
            if self.gaps != expected_gaps:
                raise ValueError("ZOOM StepTrace gaps do not match its audit-only proxy")
            batch = self.zoom_audit.batch_result
            if self.zoom_audit.feasible:
                if self.no_op_reason is not None:
                    raise ValueError("successful ZOOM cannot report a no-op reason")
            elif batch is None:
                if self.no_op_reason not in {
                    "zoom_invalid_evidence_requirements",
                    "zoom_no_evidence_requirements",
                    "zoom_p0_support_view_unavailable",
                    "zoom_p0_support_view_empty",
                    "zoom_p0_support_view_nonlocal",
                    "zoom_batch_preflight_failed",
                }:
                    raise ValueError("ZOOM no-op reason is not an exact no-batch status")
            else:
                expected_reason = (
                    "zoom_support_contract_mismatch"
                    if batch.status == "success"
                    else f"zoom_{batch.status}"
                )
                if self.no_op_reason != expected_reason:
                    raise ValueError("ZOOM no-op reason does not match its batch status")
            if self.answer is None or _json_safe(self.answer.output) != _json_safe(
                self.zoom_audit.p0_anchor.emitted_answer
            ):
                raise ValueError("ZOOM StepTrace must retain the exact P0 answer")
            answer_snapshot = json.dumps(
                self.answer.to_dict(), sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            )
            if answer_snapshot != self.zoom_audit._expected_p0_record_json:
                raise ValueError("ZOOM StepTrace answer snapshot differs from exact P0 record")
            if self._zoom_answer_snapshot_json is None:
                self._zoom_answer_snapshot_json = answer_snapshot
            elif answer_snapshot != self._zoom_answer_snapshot_json:
                raise ValueError("ZOOM StepTrace answer snapshot changed")
            if self.budget is None:
                raise ValueError("ZOOM StepTrace requires its post-attempt budget")
            if batch is not None:
                expected_budget = batch.to_dict()["ledger_after"]
                if self.budget.to_dict() != expected_budget:
                    raise ValueError("ZOOM StepTrace budget must match the batch ledger")
            return
        if self.next_audit is None:
            return
        if not isinstance(self.next_audit, NextAudit):
            raise TypeError("next_audit must be a NextAudit")
        if self.action != NEXT:
            raise ValueError("NEXT audit can only be attached to action=NEXT")
        expected_focus = (
            self.next_audit.candidate_keys[0]
            if len(self.next_audit.candidate_keys) == 1 else None
        )
        if self.focus_key != expected_focus:
            raise ValueError("NEXT focus key must match the sole candidate key")
        expected_actions = (NEXT,) if self.next_audit.feasible else ()
        if self.feasible_actions != expected_actions:
            raise ValueError("NEXT feasible actions do not match its strict audit")
        expected_gaps = (
            {"g_next": self.next_audit.g_next}
            if self.next_audit.g_next is not None else {}
        )
        if self.gaps != expected_gaps:
            raise ValueError("NEXT StepTrace gaps do not match its strict audit")
        batch = self.next_audit.batch_result
        if self.next_audit.feasible:
            if self.no_op_reason is not None:
                raise ValueError("successful NEXT cannot report a no-op reason")
        else:
            if batch is None:
                expected_reasons = {
                    "next_invalid_evidence_requirements",
                    "next_no_evidence_requirements",
                    "next_p0_support_view_unavailable",
                    "next_queue_empty",
                    "next_no_valid_candidates",
                    "next_all_candidates_visited",
                    "next_all_observations_visited",
                    "next_batch_preflight_failed",
                }
                if self.no_op_reason not in expected_reasons:
                    raise ValueError("NEXT no-op reason is not an exact no-batch status")
                expected_candidate_count = (
                    1 if self.no_op_reason == "next_batch_preflight_failed" else 0
                )
                if len(self.next_audit.candidate_keys) != expected_candidate_count:
                    raise ValueError("NEXT no-batch status contradicts candidate selection")
            else:
                expected_reason = (
                    "next_support_contract_mismatch"
                    if batch.status == "success"
                    else f"next_{batch.status}"
                )
                if self.no_op_reason != expected_reason:
                    raise ValueError("NEXT no-op reason does not match its batch status")
        if self.answer is None or _json_safe(self.answer.output) != _json_safe(
            self.next_audit.p0_anchor.emitted_answer
        ):
            raise ValueError("NEXT StepTrace must retain the exact P0 answer")
        answer_snapshot = json.dumps(
            self.answer.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        if self._answer_snapshot_json is None:
            self._answer_snapshot_json = answer_snapshot
        elif answer_snapshot != self._answer_snapshot_json:
            raise ValueError("NEXT StepTrace answer snapshot changed")
        if self.budget is None:
            raise ValueError("NEXT StepTrace requires its post-attempt budget")
        if self.next_audit.batch_result is not None:
            expected_budget = self.next_audit.batch_result.to_dict()["ledger_after"]
            if self.budget.to_dict() != expected_budget:
                raise ValueError("NEXT StepTrace budget must match the batch ledger")

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
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
        if self.zoom_audit is not None:
            if not isinstance(self.zoom_audit, ZoomAudit):
                raise TypeError("zoom_audit must be a ZoomAudit")
            payload["zoom_audit"] = self.zoom_audit.to_dict()
        if self.expand_audit is not None:
            if not isinstance(self.expand_audit, ExpandAudit):
                raise TypeError("expand_audit must be an ExpandAudit")
            payload["expand_audit"] = self.expand_audit.to_dict()
        if self.split_search_audit is not None:
            if not isinstance(self.split_search_audit, SplitSearchAudit):
                raise TypeError("split_search_audit must be a SplitSearchAudit")
            payload["split_search_audit"] = self.split_search_audit.to_dict()
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
