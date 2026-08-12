"""Leak-free composition for the first executable evidence-gap runtime."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from PIL import Image

from cvsearch.models.utils import merge_bbox_list, union_all_bboxes

from .answers import (
    aggregate_hr_answers,
    aggregate_single_choice,
    aggregate_vstar_losses,
)
from .fusion import soft_fuse_hr
from .input import POLICY_FIELDS
from .policy import select_root_or_search
from .ranking import ConservativeQueryRanker, QueryAwareNodeRanker
from .search_state import (
    ExpandDecision,
    NextCandidate,
    SearchStateCollector,
    _canonical_key as _task2_canonical_key,
    _renderer_identity as _task2_renderer_identity,
    _renderer_kind as _task2_renderer_kind,
    _source_identity as _task2_source_identity,
    _source_key as _task2_source_key,
)
from .state import EvidenceStateScore, score_state, select_state
from .types import (
    EXPAND,
    FORCED_RETURN,
    NEXT,
    ZOOM,
    AnswerRecord,
    BudgetExceeded,
    BudgetLedger,
    EvidenceSupportResult,
    ExpandAudit,
    HistoryRecord,
    MethodTrace,
    NextAudit,
    ObservationBatchResult,
    P0Anchor,
    QueryPlan,
    StepTrace,
    ZoomAudit,
    _validate_batch_plan,
    _validate_plan_support,
    sanitize_evidence_requirements,
)


MINIMAL_V1: dict[str, Any] = {
    "config_id": "minimal_v1",
    "mode": "rerank_only",
    "rerank_enabled": False,
    "ranking_mode": "cvsearch",
    "ranking_rho": 0.0,
    "ranking_max_displacement": 0,
    "beta": 0.6,
    "alpha": 0.65,
    "visual_lambda": 0.5,
    "quick_gate": 0.8,
    "root_fallback_tolerance": 0.05,
    "enable_zoom": False,
    "enable_split": False,
    "enable_expand": False,
    "enable_certified_stop": False,
    "hr_fusion_mode": "off",
    "hr_fusion_gamma": 0.0,
    "max_mllm_calls": 256,
    "max_processed_pixels": 10_000_000_000,
    "pixel_accounting": "source_image_area_per_logical_forward_approximation",
}

ADAPTIVE_RANK_CONFIG_KEYS = (
    "detail_alpha_discount",
    "context_alpha_gain",
)

CONTEXT_VISUAL_CONFIG_KEYS = (
    "context_visual_discount",
)

APPEARANCE_DESCRIPTOR_CONFIG_KEYS = (
    "appearance_descriptor_visual_relief",
)

NEXT_CONFIG_KEYS = (
    "next_enabled",
    "next_admission_mode",
    "next_replacement_enabled",
    "evidence_support_prompt_version",
    "evidence_support_processor_mode",
    "evidence_support_prompt_template_sha256",
    "evidence_support_probability_transform",
)
ZOOM_OBSERVATION_CONFIG_KEYS = (
    "p2c_zoom_enabled",
    "p2c_zoom_admission_mode",
    "p2c_zoom_replacement_enabled",
    "p2c_zoom_render_policy",
)
EXPAND_OBSERVATION_CONFIG_KEYS = (
    "p4a_expand_enabled",
    "p4a_expand_admission_mode",
    "p4a_expand_replacement_enabled",
    "p4a_expand_selection_policy",
)
_NEXT_SUPPORT_CONTRACT = {
    "evidence_support_prompt_version": "qwen_answer_free_evidence_support_v1",
    "evidence_support_processor_mode": (
        "single_rendered_view_chat_left_padding_final_yes_no_logits"
    ),
    "evidence_support_prompt_template_sha256": (
        "0ba54d5ef5190f162d2036721693ae1af1a499f088e12a5106f8c9b692dfcb86"
    ),
    "evidence_support_probability_transform": (
        "v1:p_yes=softmax(final_position_two_logits[Yes,No],dim=-1,"
        "preserve_model_dtype,no_float32_cast)[0];no_legacy_2p_minus_1"
    ),
}

_GLOBAL_SCOPE = re.compile(
    r"\b(how many|number of|total count|throughout|all|every|none|no|without|absent|only|unique|each)\b",
    re.IGNORECASE,
)
_RELATION = re.compile(
    r"\b(beside|between|behind|in front of|left of|right of|side of|relative to|near|next to|above|below)\b",
    re.IGNORECASE,
)
_HR_DISALLOWED_QUERY = re.compile(
    r"\b(compar(?:e|ed|ing)(?:\s+(?:to|with))?|comparison|in\s+relation\s+to|"
    r"relative\s+position|positions?|locat(?:e|ed|ion)|where|directions?|orientation|"
    r"left|right|upper|lower|top|bottom|clockwise|counterclockwise|horizontal|vertical|diagonal|"
    r"front|rear|north(?:ern)?|south(?:ern)?|east(?:ern)?|west(?:ern)?|sides?|"
    r"same|different|both|versus|vs|than|larger|smaller|greater|less|fewer|"
    r"higher|taller|shorter|counts?|sum|average|total|arithmetic|calculat(?:e|ed|ion|ing)|"
    r"plus|minus|subtract(?:ed|ion|ing)?|multipl(?:y|ied|ication|ying)|product\s+of|"
    r"divid(?:e|ed|ing)|difference|ratio|percentage|maps?|countries|country|"
    r"adjacent|nearest|closest|facing|most\s+common|quantity|add(?:ed|ing)?|"
    r"sizes?|length|width|height|languages?|fonts?|styles?|ages?|who|when|why)\b",
    re.IGNORECASE,
)
_HR_COORDINATED_TARGET = re.compile(r"\b(?:and|or|versus|vs)\b|[,;/]", re.IGNORECASE)
_HR_LOCAL_ENTITY_TARGETS = frozenset(("object", "product", "sign", "signboard"))
_HR_READABLE_TARGET = re.compile(
    r"\b(signs?|labels?|posters?|banners?|notices?|billboards?|screens?|displays?|"
    r"plaques?|boards?|papers?|pages?|books?|newspapers?|magazines?|cards?|tags?|"
    r"logos?|packages?|packaging|bottles?)\b",
    re.IGNORECASE,
)
_HR_FROZEN_COMPLEX_LOCAL_TARGETS = frozenset((
    (
        "tell me the shape of the signboard attached to the building?",
        "signboard attached to the building",
    ),
))

_TARGET_DETAIL_KEYS = frozenset(("kind", "target", "requirements"))
_RUNTIME_CONTEXT_KEYS = frozenset(
    ("kind", "query_source", "planned_augmented_queries_used")
)


def _strict_json(value: Any, name: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be strict JSON-safe") from error


def _single_choice_question(question: str, options: str) -> str:
    return (
        question + " Options:\n" + options + "\n"
        "Select the best answer to the above multiple-choice question based on "
        "the image. Respond with only the letter of the correct option.\n"
        "The best answer is:"
    )


def _outputs_agree(left: Any, right: Any) -> bool:
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("evaluator output must be strict JSON-safe") from error


def _policy_copy(annotation: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(annotation, Mapping):
        raise TypeError("policy_annotation must be a mapping")
    if set(annotation) != set(POLICY_FIELDS):
        raise ValueError(f"policy_annotation must contain exactly {list(POLICY_FIELDS)}")
    result = copy.deepcopy({key: annotation[key] for key in POLICY_FIELDS})
    if not isinstance(result["question"], str) or not result["question"].strip():
        raise ValueError("question must be a nonempty string")
    if not isinstance(result["answer_type"], str) or not result["answer_type"].strip():
        raise ValueError("answer_type must be a nonempty string")
    if not isinstance(result["input_image"], str) or not result["input_image"].strip():
        raise ValueError("input_image must be a nonempty string")
    _strict_json(result, "policy_annotation")
    return result


def build_query_plan(policy_annotation: Mapping[str, Any], targets: Sequence[str]) -> QueryPlan:
    """Build the deterministic, answer-free planner used by ``minimal_v1``."""
    if not isinstance(policy_annotation, Mapping):
        raise TypeError("policy_annotation must be a mapping")
    if set(policy_annotation) != set(POLICY_FIELDS):
        raise ValueError(f"policy_annotation must contain exactly {list(POLICY_FIELDS)}")
    question = policy_annotation["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
        raise TypeError("targets must be a sequence of strings")
    normalized_targets: list[str] = []
    seen: set[str] = set()
    for value in targets:
        if not isinstance(value, str):
            raise TypeError("targets must contain only strings")
        target = " ".join(value.split())
        if not target:
            continue
        key = target.casefold()
        if key not in seen:
            normalized_targets.append(target)
            seen.add(key)

    augmented = tuple(f"locate and inspect {target}" for target in normalized_targets)
    evidence_items: list[dict[str, Any]] = [
        {"kind": "target_detail", "target": target, "requirements": ["presence", "visual_detail"]}
        for target in normalized_targets
    ]
    if len(normalized_targets) > 1 or _RELATION.search(question):
        evidence_items.append({"kind": "relation_context", "targets": list(normalized_targets)})
    global_scope_required = bool(_GLOBAL_SCOPE.search(question))
    if global_scope_required:
        evidence_items.append({"kind": "coverage", "requirement": "global_scope"})
    if not evidence_items:
        evidence_items.append({"kind": "question_evidence", "requirement": "visual_detail"})

    plan = QueryPlan(
        main_query=question,
        targets=tuple(normalized_targets),
        augmented_queries=augmented,
        evidence_items=tuple(evidence_items),
        global_scope_required=global_scope_required,
        fallback_used=True,
    )
    _strict_json(plan.to_dict(), "query plan")
    return plan


def _finite_weight(config: dict[str, Any], name: str) -> None:
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    config[name] = float(value)


def _nonnegative_integer(config: dict[str, Any], name: str) -> None:
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _runtime_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _capture_runtime_diagnostics(trace: MethodTrace, runtime_annotation: Mapping[str, Any]) -> None:
    search_mode = runtime_annotation.get("search_mode")
    if search_mode is not None:
        if isinstance(search_mode, bool) or not isinstance(search_mode, int) or search_mode not in {0, 1, 2, 3}:
            raise ValueError("CVSearch search_mode must be one of 0, 1, 2, 3")
        trace.cvsearch_search_mode = search_mode
    root_confidence = runtime_annotation.get("root_ans_conf")
    if root_confidence is not None:
        trace.root_ans_conf = _runtime_number(root_confidence, "root_ans_conf")
    for key in ("num_pop", "num_zoom_in", "num_zoom_out"):
        value = runtime_annotation.get(key, [])
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"CVSearch {key} must be a sequence")
        snapshot = copy.deepcopy(list(value))
        _strict_json(snapshot, key)
        setattr(trace, key, snapshot)


def _matches_observation_runtime_profile(
    config: Mapping[str, Any], *, adaptive_rank_supplied: bool,
) -> bool:
    """Accept the legacy observer or the exact frozen Phase-1 ranking runtime."""
    actions_enabled = bool(
        config.get("p2c_zoom_enabled") or config.get("p4a_expand_enabled")
    )
    common = (
        config["root_fallback_tolerance"] == 0.05
        and not any(config[name] for name in (
            "enable_zoom", "enable_split", "enable_expand", "enable_certified_stop",
        ))
        and config["hr_fusion_mode"] == "off"
        and config["hr_fusion_gamma"] == 0.0
        and config["max_mllm_calls"] == 512
        and config["max_processed_pixels"] == 10_000_000_000
    )
    legacy = (
        config["mode"] == "root_search_fallback"
        and (not actions_enabled or config["quick_gate"] == 0.6)
        and not config["rerank_enabled"]
        and config["ranking_mode"] == "cvsearch"
        and config["ranking_rho"] == 0.0
        and config["ranking_max_displacement"] == 0
    )
    frozen_phase1_v1_to_v4 = (
        adaptive_rank_supplied
        and config["mode"] == "rerank_only"
        and config["rerank_enabled"]
        and config["ranking_mode"] == "query_linear"
        and config["ranking_rho"] == 0.0
        and config["ranking_max_displacement"] == 0
        and config["beta"] == 1.0
        and config["alpha"] == 0.25
        and config["visual_lambda"] == 1.0
        and config["detail_alpha_discount"] == 0.15
        and config["context_alpha_gain"] == 0.45
        and config.get("context_visual_discount") in {None, 1.0}
        and config.get("appearance_descriptor_visual_relief") in {None, 1.0}
        and config["quick_gate"] == 0.8
    )
    frozen_phase1_v5 = (
        adaptive_rank_supplied
        and config["mode"] == "rerank_only"
        and config["rerank_enabled"]
        and config["ranking_mode"] == "query_linear"
        and config["ranking_rho"] == 0.0
        and config["ranking_max_displacement"] == 0
        and config["beta"] == 0.6
        and config["alpha"] == 0.6
        and config["visual_lambda"] == 1.0
        and config["detail_alpha_discount"] == 0.15
        and config["context_alpha_gain"] == 0.15
        and config.get("context_visual_discount") == 0.5
        and config.get("appearance_descriptor_visual_relief") == 1.0
        and config["quick_gate"] == 0.8
    )
    frozen_phase1_v6 = (
        adaptive_rank_supplied
        and config["mode"] == "rerank_only"
        and config["rerank_enabled"]
        and config["ranking_mode"] == "query_linear"
        and config["ranking_rho"] == 0.0
        and config["ranking_max_displacement"] == 0
        and config["beta"] == 0.6
        and config["alpha"] == 0.65
        and config["visual_lambda"] == 0.7
        and config["detail_alpha_discount"] == 0.15
        and config["context_alpha_gain"] == 0.15
        and config.get("context_visual_discount") == 0.25
        and config.get("appearance_descriptor_visual_relief") == 1.0
        and config["quick_gate"] == 0.8
    )
    return common and (
        legacy or frozen_phase1_v1_to_v4 or frozen_phase1_v5
        or frozen_phase1_v6
    )


def load_method_config(config: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
    """Load a strict minimal-v1 config with only the narrow zoom gate available."""
    if isinstance(config, Mapping):
        supplied = copy.deepcopy(dict(config))
    elif isinstance(config, (str, os.PathLike)):
        value = os.fspath(config)
        if value == "minimal_v1":
            supplied = {}
        else:
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open("r", encoding="utf-8") as handle:
                supplied = json.load(handle)
            if not isinstance(supplied, dict):
                raise ValueError("config JSON must contain an object")
    else:
        raise TypeError("config must be a mapping, minimal_v1, or JSON path")
    supplied_next_keys = set(supplied).intersection(NEXT_CONFIG_KEYS)
    if supplied_next_keys and supplied_next_keys != set(NEXT_CONFIG_KEYS):
        raise ValueError("NEXT config extension must be supplied as an all-or-none group")
    supplied_adaptive_rank_keys = set(supplied).intersection(ADAPTIVE_RANK_CONFIG_KEYS)
    if (
        supplied_adaptive_rank_keys
        and supplied_adaptive_rank_keys != set(ADAPTIVE_RANK_CONFIG_KEYS)
    ):
        raise ValueError(
            "adaptive ranking config extension must be supplied as an all-or-none group"
        )
    supplied_context_visual_keys = set(supplied).intersection(
        CONTEXT_VISUAL_CONFIG_KEYS
    )
    if supplied_context_visual_keys and (
        supplied_adaptive_rank_keys != set(ADAPTIVE_RANK_CONFIG_KEYS)
    ):
        raise ValueError(
            "context visual adaptation requires the complete adaptive ranking extension"
        )
    supplied_attribute_descriptor_keys = set(supplied).intersection(
        APPEARANCE_DESCRIPTOR_CONFIG_KEYS
    )
    if supplied_attribute_descriptor_keys and (
        supplied_context_visual_keys != set(CONTEXT_VISUAL_CONFIG_KEYS)
    ):
        raise ValueError(
            "attribute descriptor detail requires context visual adaptation"
        )
    supplied_zoom_observation_keys = set(supplied).intersection(
        ZOOM_OBSERVATION_CONFIG_KEYS
    )
    if (
        supplied_zoom_observation_keys
        and supplied_zoom_observation_keys != set(ZOOM_OBSERVATION_CONFIG_KEYS)
    ):
        raise ValueError(
            "P2C ZOOM observation config extension must be supplied as an all-or-none group"
        )
    supplied_expand_observation_keys = set(supplied).intersection(
        EXPAND_OBSERVATION_CONFIG_KEYS
    )
    if (
        supplied_expand_observation_keys
        and supplied_expand_observation_keys != set(EXPAND_OBSERVATION_CONFIG_KEYS)
    ):
        raise ValueError(
            "P4A EXPAND observation config extension must be supplied as an all-or-none group"
        )
    unknown = (
        set(supplied)
        - set(MINIMAL_V1)
        - set(ADAPTIVE_RANK_CONFIG_KEYS)
        - set(CONTEXT_VISUAL_CONFIG_KEYS)
        - set(APPEARANCE_DESCRIPTOR_CONFIG_KEYS)
        - set(NEXT_CONFIG_KEYS)
        - set(ZOOM_OBSERVATION_CONFIG_KEYS)
        - set(EXPAND_OBSERVATION_CONFIG_KEYS)
    )
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    legacy_query_linear = (
        "ranking_mode" not in supplied and supplied.get("rerank_enabled") is True
    )
    result = copy.deepcopy(MINIMAL_V1)
    result.update(supplied)
    if legacy_query_linear:
        result["ranking_mode"] = "query_linear"
    if not isinstance(result["config_id"], str) or re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", result["config_id"]
    ) is None:
        raise ValueError("config_id must be a nonempty version-safe identifier")
    if result["mode"] not in {"rerank_only", "root_search_fallback"}:
        raise ValueError("mode must be rerank_only or root_search_fallback")
    for name in (
        "rerank_enabled", "enable_zoom", "enable_split", "enable_expand", "enable_certified_stop"
    ):
        if not isinstance(result[name], bool):
            raise TypeError(f"{name} must be boolean")
    if not isinstance(result["ranking_mode"], str) or result["ranking_mode"] not in {
        "cvsearch", "query_linear", "conservative_rrf"
    }:
        raise ValueError("ranking_mode must be cvsearch, query_linear, or conservative_rrf")
    ranking_rho = _runtime_number(result["ranking_rho"], "ranking_rho")
    if not 0.0 <= ranking_rho <= 1.0:
        raise ValueError("ranking_rho must be in [0, 1]")
    result["ranking_rho"] = ranking_rho
    _nonnegative_integer(result, "ranking_max_displacement")
    ranking_mode = result["ranking_mode"]
    if ranking_mode == "cvsearch":
        if result["rerank_enabled"]:
            raise ValueError("cvsearch ranking requires reranking disabled")
        if ranking_rho != 0.0 or result["ranking_max_displacement"] != 0:
            raise ValueError("cvsearch ranking requires zero rho and displacement")
    elif ranking_mode == "query_linear":
        if not result["rerank_enabled"]:
            raise ValueError("query_linear ranking requires reranking enabled")
        if ranking_rho != 0.0 or result["ranking_max_displacement"] != 0:
            raise ValueError("query_linear ranking requires zero rho and displacement")
    elif not result["rerank_enabled"]:
        raise ValueError("conservative_rrf ranking requires reranking enabled")
    if any(result[name] for name in ("enable_split", "enable_expand", "enable_certified_stop")):
        raise ValueError("minimal_v1 cannot enable split, expand, or certified stop")
    if result["enable_zoom"] and (
        result["mode"] != "root_search_fallback" or result["rerank_enabled"]
    ):
        raise ValueError("zoom requires root_search_fallback with reranking disabled")
    if result["hr_fusion_mode"] not in {"off", "global_soft"}:
        raise ValueError("hr_fusion_mode must be off or global_soft")
    gamma = _runtime_number(result["hr_fusion_gamma"], "hr_fusion_gamma")
    if gamma < 0.0:
        raise ValueError("hr_fusion_gamma must be non-negative")
    result["hr_fusion_gamma"] = gamma
    if result["hr_fusion_mode"] == "off" and result["hr_fusion_gamma"] != 0.0:
        raise ValueError("disabled HR fusion requires zero gamma")
    for name in ("beta", "alpha", "visual_lambda", "quick_gate", "root_fallback_tolerance"):
        _finite_weight(result, name)
    for name in supplied_adaptive_rank_keys:
        _finite_weight(result, name)
    for name in supplied_context_visual_keys:
        _finite_weight(result, name)
    for name in supplied_attribute_descriptor_keys:
        _finite_weight(result, name)
    if supplied_adaptive_rank_keys and (
        not result["rerank_enabled"] or result["ranking_mode"] != "query_linear"
    ):
        raise ValueError("adaptive ranking weights require query_linear reranking")
    if supplied_context_visual_keys and (
        not result["rerank_enabled"] or result["ranking_mode"] != "query_linear"
    ):
        raise ValueError("context visual adaptation requires query_linear reranking")
    for name in ("max_mllm_calls", "max_processed_pixels"):
        _nonnegative_integer(result, name)
    if result["pixel_accounting"] != "source_image_area_per_logical_forward_approximation":
        raise ValueError("pixel_accounting is fixed for minimal_v1")
    if supplied_next_keys:
        if not isinstance(result["next_enabled"], bool):
            raise TypeError("next_enabled must be boolean")
        if not isinstance(result["next_replacement_enabled"], bool):
            raise TypeError("next_replacement_enabled must be boolean")
        expected_admission = "all_feasible" if result["next_enabled"] else "disabled"
        if result["next_admission_mode"] != expected_admission:
            raise ValueError("NEXT admission mode does not match enabled state")
        if result["next_replacement_enabled"]:
            raise ValueError("P2A replacement must remain disabled")
        for name, expected in _NEXT_SUPPORT_CONTRACT.items():
            if result[name] != expected:
                raise ValueError(f"{name} does not match the frozen support contract")
        if result["next_enabled"] and (
            result["mode"] != "root_search_fallback"
            or result["quick_gate"] != 0.6
            or result["root_fallback_tolerance"] != 0.05
            or result["rerank_enabled"]
            or result["ranking_mode"] != "cvsearch"
            or result["ranking_rho"] != 0.0
            or result["ranking_max_displacement"] != 0
            or any(result[name] for name in (
                "enable_zoom", "enable_split", "enable_expand", "enable_certified_stop",
            ))
            or result["hr_fusion_mode"] != "off"
            or result["hr_fusion_gamma"] != 0.0
            or result["max_mllm_calls"] != 512
            or result["max_processed_pixels"] != 10_000_000_000
        ):
            raise ValueError("enabled NEXT requires the frozen unified P2A config")
    if supplied_zoom_observation_keys:
        if not isinstance(result["p2c_zoom_enabled"], bool):
            raise TypeError("p2c_zoom_enabled must be boolean")
        if not isinstance(result["p2c_zoom_replacement_enabled"], bool):
            raise TypeError("p2c_zoom_replacement_enabled must be boolean")
        expected_admission = (
            "all_feasible" if result["p2c_zoom_enabled"] else "disabled"
        )
        if result["p2c_zoom_admission_mode"] != expected_admission:
            raise ValueError("P2C ZOOM admission mode does not match enabled state")
        if result["p2c_zoom_replacement_enabled"]:
            raise ValueError("P2C ZOOM replacement must remain disabled")
        if result["p2c_zoom_render_policy"] != (
            "native_coordinate_crop_view_size_div3"
        ):
            raise ValueError("P2C ZOOM render policy is not frozen")
        if not supplied_next_keys or result["next_enabled"]:
            raise ValueError("P2C ZOOM requires the frozen support contract with NEXT off")
        if not _matches_observation_runtime_profile(
            result,
            adaptive_rank_supplied=(
                supplied_adaptive_rank_keys == set(ADAPTIVE_RANK_CONFIG_KEYS)
            ),
        ):
            raise ValueError("P2C ZOOM requires the frozen unified observation config")
    if supplied_expand_observation_keys:
        if not isinstance(result["p4a_expand_enabled"], bool):
            raise TypeError("p4a_expand_enabled must be boolean")
        if not isinstance(result["p4a_expand_replacement_enabled"], bool):
            raise TypeError("p4a_expand_replacement_enabled must be boolean")
        expected_admission = (
            "all_feasible" if result["p4a_expand_enabled"] else "disabled"
        )
        if result["p4a_expand_admission_mode"] != expected_admission:
            raise ValueError("P4A EXPAND admission mode does not match enabled state")
        if result["p4a_expand_replacement_enabled"]:
            raise ValueError("P4A EXPAND replacement must remain disabled")
        if result["p4a_expand_selection_policy"] != (
            "nearest_spatial_native_cvsearch_context_v1"
        ):
            raise ValueError("P4A EXPAND selection policy is not frozen")
        if (
            not supplied_next_keys or result["next_enabled"]
            or not supplied_zoom_observation_keys
            or (
                result["p2c_zoom_enabled"]
                and not result["p4a_expand_enabled"]
            )
        ):
            raise ValueError("P4A EXPAND requires frozen NEXT/ZOOM groups")
        if not _matches_observation_runtime_profile(
            result,
            adaptive_rank_supplied=(
                supplied_adaptive_rank_keys == set(ADAPTIVE_RANK_CONFIG_KEYS)
            ),
        ):
            raise ValueError("P4A EXPAND requires the frozen unified observation config")
    _strict_json(result, "method config")
    return result


class _BudgetedZoomModel:
    """Pre-charge public Qwen calls using source-image pixels as an approximation."""

    def __init__(self, model: Any, ledger: BudgetLedger, *, answer_reserve_calls: int = 0,
                 answer_image_loader: Any = None, answer_type: str | None = None):
        self._model = model
        self._ledger = ledger
        self._base_view_size = getattr(model, "view_size", None)
        self._answer_reserve_calls = answer_reserve_calls
        self._answer_reserve_pixels: int | None = None if answer_reserve_calls else 0
        self._answer_image_loader = answer_image_loader
        self._answer_type = answer_type
        self._answer_started = False
        self._answer_reserve_invalid = False
        self._free_form_started = False
        self._free_form_remaining = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    @staticmethod
    def _pixels(image: Any) -> int:
        if not isinstance(image, Image.Image):
            raise TypeError("visual model calls require a PIL image")
        return int(image.width * image.height)

    def _consume_actual(self, calls: int, pixels: int) -> None:
        """Atomically validate both dimensions before mutating the real ledger."""
        if self._ledger.mllm_calls + calls > self._ledger.max_mllm_calls:
            raise BudgetExceeded("mllm_calls budget exhausted before model execution")
        if self._ledger.processed_pixels + pixels > self._ledger.max_processed_pixels:
            raise BudgetExceeded("processed_pixels budget exhausted before model execution")
        self._ledger.consume("mllm_calls", calls)
        self._ledger.consume("processed_pixels", pixels)

    @staticmethod
    def _observation_sha256(image: Image.Image) -> str:
        payload = image.mode.encode("utf-8") + b"\x00"
        payload += f"{image.width}x{image.height}".encode("ascii") + b"\x00"
        payload += image.tobytes()
        return hashlib.sha256(payload).hexdigest()

    def _validate_post_anchor_support_view(
        self, source_image: Image.Image, source_image_key: str,
        support_view: tuple[Any, ...] | None,
    ) -> tuple[tuple[Any, ...], list[str], list[str], list[dict[str, Any]]]:
        if support_view is None:
            raise ValueError("support view is unavailable")
        if not isinstance(support_view, tuple):
            raise TypeError("support view must be an immutable tuple")
        nodes: list[Any] = []
        descriptors: list[dict[str, Any]] = []
        canonical_keys: list[str] = []
        renderer_ids: list[str] = []
        seen_keys: set[str] = set()
        seen_renderer_ids: set[str] = set()
        for descriptor in support_view:
            if type(descriptor) is not NextCandidate:
                raise TypeError("support descriptor must be an exact Task-2 NextCandidate")
            if descriptor.source_image_key != source_image_key:
                raise ValueError("support descriptor source identity does not match the RGB source")
            if descriptor.source not in {None, "global", "fast", "fine", "fine_fallback"}:
                raise ValueError("support descriptor source is not native CVSearch provenance")
            if descriptor.tree_scope not in {"main", "cropped"}:
                raise ValueError("support descriptor tree scope is invalid")
            bbox = tuple(descriptor.bbox_original)
            if len(bbox) != 4 or any(
                isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(float(value)) for value in bbox
            ):
                raise ValueError("support descriptor bbox must contain four finite numbers")
            x, y, width, height = (float(value) for value in bbox)
            if (
                width <= 0 or height <= 0 or x < 0 or y < 0
                or x + width > source_image.width or y + height > source_image.height
            ):
                raise ValueError("support descriptor bbox is outside the RGB source")
            if any(
                isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
                for value in (
                    descriptor.depth, descriptor.render_level, descriptor.first_seen_ordinal,
                )
            ):
                raise ValueError("support descriptor integer provenance is invalid")
            if (
                descriptor.posterior_score is not None
                and (
                    isinstance(descriptor.posterior_score, bool)
                    or not isinstance(descriptor.posterior_score, Real)
                    or not math.isfinite(float(descriptor.posterior_score))
                )
            ):
                raise ValueError("support descriptor posterior must be finite or null")
            if (
                not isinstance(descriptor.crop_origin, tuple)
                or len(descriptor.crop_origin) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(float(value)) for value in descriptor.crop_origin
                )
            ):
                raise ValueError("support descriptor crop origin must be finite")
            expected_key = _task2_canonical_key(
                bbox, descriptor.depth, descriptor.render_level,
            )
            expected_kind = _task2_renderer_kind(descriptor.source)
            expected_renderer_id = _task2_renderer_identity(
                source_image_key, bbox, descriptor.render_level, expected_kind,
            )
            if descriptor.canonical_key != expected_key:
                raise ValueError("support descriptor canonical key does not match geometry")
            if descriptor.renderer_kind != expected_kind:
                raise ValueError("support descriptor renderer kind does not match source")
            if descriptor.renderer_identity != expected_renderer_id:
                raise ValueError("support descriptor renderer identity does not match geometry")
            if expected_key in seen_keys or expected_renderer_id in seen_renderer_ids:
                raise ValueError("support view contains a duplicate descriptor identity")
            seen_keys.add(expected_key)
            seen_renderer_ids.add(expected_renderer_id)
            expected_snapshot = {
                "canonical_key": expected_key,
                "bbox_original": list(bbox),
                "depth": descriptor.depth,
                "render_level": descriptor.render_level,
                "posterior_score": descriptor.posterior_score,
                "first_seen_ordinal": descriptor.first_seen_ordinal,
                "tree_scope": descriptor.tree_scope,
                "crop_origin": list(descriptor.crop_origin),
                "source_image_key": source_image_key,
                "source": descriptor.source,
                "renderer_kind": expected_kind,
                "renderer_identity": expected_renderer_id,
            }
            snapshot = copy.deepcopy(descriptor.to_dict())
            _strict_json(snapshot, "support descriptor")
            if snapshot != expected_snapshot:
                raise ValueError("support descriptor snapshot does not match its frozen fields")
            render_node = descriptor.render_node
            try:
                render_source = render_node.state.original_image_pil
                render_bbox = tuple(render_node.state.bbox)
            except (AttributeError, TypeError) as error:
                raise ValueError("support descriptor render adapter is malformed") from error
            if (
                not isinstance(render_source, Image.Image)
                or _task2_source_key(_task2_source_identity(render_source)) != source_image_key
                or render_bbox != bbox
            ):
                raise ValueError("support descriptor render adapter does not match the RGB source")
            nodes.append(render_node)
            canonical_keys.append(expected_key)
            renderer_ids.append(expected_renderer_id)
            descriptors.append(snapshot)
        return tuple(nodes), canonical_keys, renderer_ids, descriptors

    def _render_post_anchor_support_view(
        self, source_image: Image.Image,
        validated_view: tuple[tuple[Any, ...], list[str], list[str], list[dict[str, Any]]],
    ) -> tuple[Image.Image, tuple[Any, ...], str, dict[str, Any]]:
        nodes, canonical_keys, renderer_ids, descriptors = validated_view
        renderer = getattr(self._model, "process_nodes_to_image_list", None)
        if not callable(renderer):
            raise ValueError("raw model must expose the frozen Qwen renderer")
        rendered_views = renderer(list(nodes), source_image, root_anyres=True)
        if not isinstance(rendered_views, (list, tuple)) or not rendered_views:
            raise ValueError("Qwen renderer must return at least one view")
        if not all(isinstance(view, Image.Image) for view in rendered_views):
            raise TypeError("Qwen renderer returned a non-image view")
        rendered = rendered_views[0] if len(rendered_views) == 1 else rendered_views[-1]
        view_sha256 = self._observation_sha256(rendered)
        identity_payload = {
            "canonical_keys": canonical_keys,
            "renderer_identities": renderer_ids,
            "rendered_mode": rendered.mode,
            "rendered_size": [rendered.width, rendered.height],
            "view_sha256": view_sha256,
        }
        identity = json.dumps(
            identity_payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        metadata = dict(identity_payload, descriptors=descriptors)
        return rendered, nodes, identity, metadata

    @staticmethod
    def _post_anchor_plan_hash(plan: Mapping[str, Any]) -> str:
        payload = copy.deepcopy(dict(plan))
        payload.pop("plan_hash", None)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def post_anchor_observation_batch(
        self, *, source_image: Image.Image, q0: str, query_plan: QueryPlan,
        current_support_view: tuple[Any, ...] | None,
        candidate_support_view: tuple[Any, ...] | None,
        answer_type: str, options: Any,
    ) -> ObservationBatchResult:
        """Atomically charge two aggregate supports plus one candidate answer."""
        started = time.perf_counter()
        ledger_before = copy.deepcopy(self._ledger.to_dict())
        if not isinstance(source_image, Image.Image) or source_image.mode != "RGB":
            raise TypeError("source_image must be an RGB PIL image")
        if not isinstance(q0, str) or not q0.strip():
            raise ValueError("q0 must be a nonempty string")
        if not isinstance(query_plan, QueryPlan):
            raise TypeError("query_plan must be a QueryPlan")
        if query_plan.main_query != q0:
            raise ValueError("q0 must exactly match QueryPlan.main_query")
        requirements = sanitize_evidence_requirements(query_plan.evidence_items)
        source_area = self._pixels(source_image)
        source_identity = _task2_source_identity(source_image)
        source_image_key = _task2_source_key(source_identity)
        empty_plan = {
            "schema_version": 1,
            "batch_kind": "p2a_post_anchor_next",
            "answer_type": answer_type,
            "source_identity": source_identity,
            "q0_sha256": hashlib.sha256(q0.encode("utf-8")).hexdigest(),
            "requirement_order": [item.requirement_id for item in requirements],
            "requirement_set_id": EvidenceSupportResult.requirement_set_id_for(requirements),
            "verifier_status": "disabled_same_checkpoint_unpromoted",
            "current_support_calls": 0,
            "candidate_support_calls": 0,
            "candidate_answer_calls": 0,
            "candidate_option_count": 0,
            "pixels_per_logical_forward": source_area,
            "total_calls": 0,
            "total_pixels": 0,
        }
        if not requirements:
            return ObservationBatchResult(
                status="no_requirements", batch_plan=empty_plan, admitted=False, charged=False,
                ledger_before=ledger_before, ledger_after=self._ledger.to_dict(),
                failure_phase="preflight", failure_reason="no_requirements",
                elapsed_seconds=time.perf_counter() - started,
            )
        if self._answer_reserve_calls and not self._answer_started:
            return ObservationBatchResult(
                status="budget_rejected", batch_plan=empty_plan, admitted=False, charged=False,
                ledger_before=ledger_before, ledger_after=self._ledger.to_dict(),
                failure_phase="preflight", failure_reason="public_answer_reserve_unconsumed",
                elapsed_seconds=time.perf_counter() - started,
            )
        if answer_type not in {"option_list", "logits_match"}:
            raise ValueError("post-anchor batch supports only HR and V* answer types")
        if isinstance(options, (str, bytes)) or not isinstance(options, Sequence):
            raise TypeError("options must be a sequence of strings")
        frozen_options = tuple(options)
        if not all(isinstance(option, str) and option for option in frozen_options):
            raise ValueError("options must contain nonempty strings")
        if answer_type == "option_list" and len(frozen_options) != 4:
            raise ValueError("HR post-anchor answer requires exactly four option blocks")
        if answer_type == "logits_match" and not frozen_options:
            raise ValueError("V* post-anchor answer requires nonempty options")
        support_call = getattr(self._model, "evidence_support", None)
        prepare_support = getattr(self._model, "_prepare_evidence_support", None)
        if not callable(support_call) or not callable(prepare_support):
            raise ValueError("raw model must expose answer-free support preflight and execution")
        answer_method_name = (
            "free_form_using_nodes" if answer_type == "option_list"
            else "multiple_choices_with_losses"
        )
        if not callable(getattr(self._model, answer_method_name, None)):
            raise ValueError("raw model is missing the candidate-answer method")
        if not callable(getattr(self._model, "process_nodes_to_image_list", None)):
            raise ValueError("raw model must expose the frozen Qwen renderer")
        current_validated = self._validate_post_anchor_support_view(
            source_image, source_image_key, current_support_view,
        )
        candidate_validated = self._validate_post_anchor_support_view(
            source_image, source_image_key, candidate_support_view,
        )
        prepared = prepare_support(q0, requirements)
        if prepared is None:
            raise AssertionError("nonempty requirements must produce support provenance")
        current_rendered, current_nodes, current_identity, current_metadata = (
            self._render_post_anchor_support_view(source_image, current_validated)
        )
        candidate_rendered, candidate_nodes, candidate_identity, candidate_metadata = (
            self._render_post_anchor_support_view(source_image, candidate_validated)
        )
        answer_calls = 4 if answer_type == "option_list" else 1 + len(frozen_options)
        total_calls = 2 + answer_calls
        total_pixels = total_calls * source_area
        plan = {
            "schema_version": 1,
            "batch_kind": "p2a_post_anchor_next",
            "answer_type": answer_type,
            "source_identity": source_identity,
            "source_width": source_image.width,
            "source_height": source_image.height,
            "accounted_source_area": source_area,
            "current_observation": current_metadata,
            "candidate_observation": candidate_metadata,
            "q0_sha256": hashlib.sha256(q0.encode("utf-8")).hexdigest(),
            "requirement_order": [item.requirement_id for item in requirements],
            "requirement_set_id": EvidenceSupportResult.requirement_set_id_for(requirements),
            "prompt_version": prepared["prompt_version"],
            "prompt_template_sha256": prepared["prompt_template_sha256"],
            "current_prompt_sha256": prepared["prompt_sha256"],
            "candidate_prompt_sha256": prepared["prompt_sha256"],
            "processor_mode": prepared["processor_mode"],
            "processor_fingerprint": prepared["fingerprint"],
            "checkpoint": prepared["checkpoint"],
            "yes_tokenization": list(prepared["yes_tokens"]),
            "no_tokenization": list(prepared["no_tokens"]),
            "yes_token_id": prepared["yes_id"],
            "no_token_id": prepared["no_id"],
            "p_yes_transform": prepared["p_yes_transform"],
            "verifier_status": "disabled_same_checkpoint_unpromoted",
            "current_support_calls": 1,
            "candidate_support_calls": 1,
            "candidate_answer_calls": answer_calls,
            "candidate_option_count": len(frozen_options),
            "pixels_per_logical_forward": source_area,
            "total_calls": total_calls,
            "total_pixels": total_pixels,
        }
        _strict_json(plan, "post-anchor observation plan")
        plan_hash = self._post_anchor_plan_hash(plan)
        try:
            self._consume_actual(total_calls, total_pixels)
        except BudgetExceeded as error:
            return ObservationBatchResult(
                status="budget_rejected", batch_plan=plan, admitted=False, charged=False,
                ledger_before=ledger_before, ledger_after=self._ledger.to_dict(),
                failure_phase="admission", failure_reason=str(error),
                exception_type=type(error).__name__,
                elapsed_seconds=time.perf_counter() - started,
            )

        executed: list[str] = []
        current_support = None
        candidate_support = None
        phase = "current_support"
        try:
            current_support = support_call(
                question=q0, requirements=requirements,
                rendered_observation=current_rendered,
                observation_identity=current_identity,
            )
            if not isinstance(current_support, EvidenceSupportResult):
                raise TypeError("current support returned an invalid result")
            current_support = current_support.with_batch_accounting(
                accounted_pixels=source_area, batch_plan_hash=plan_hash,
            )
            executed.append(phase)
            phase = "candidate_support"
            candidate_support = support_call(
                question=q0, requirements=requirements,
                rendered_observation=candidate_rendered,
                observation_identity=candidate_identity,
            )
            if not isinstance(candidate_support, EvidenceSupportResult):
                raise TypeError("candidate support returned an invalid result")
            candidate_support = candidate_support.with_batch_accounting(
                accounted_pixels=source_area, batch_plan_hash=plan_hash,
            )
            executed.append(phase)
            if answer_type == "option_list":
                raw_outputs = []
                for index, option_block in enumerate(frozen_options):
                    phase = f"hr_answer_{index}"
                    question_input = q0 + "\n" + option_block + "Answer the option letter directly."
                    raw_output = self._model.free_form_using_nodes(
                        source_image, question_input, list(candidate_nodes)
                    )
                    if not isinstance(raw_output, str):
                        raise TypeError("HR candidate outputs must be strings")
                    raw_outputs.append(raw_output)
                    executed.append(phase)
                candidate_answer: Any = raw_outputs
            else:
                phase = "vstar_answer"
                winner, losses = self._model.multiple_choices_with_losses(
                    source_image, q0, frozen_options, list(candidate_nodes)
                )
                if isinstance(winner, bool) or not isinstance(winner, Integral):
                    raise ValueError("V* winner must be an option index")
                winner = int(winner)
                if not 0 <= winner < len(frozen_options):
                    raise ValueError("V* winner is outside the option range")
                if not isinstance(losses, (list, tuple)) or len(losses) != len(frozen_options):
                    raise ValueError("V* losses must match every option")
                finite_losses = tuple(_runtime_number(loss, "V* option loss") for loss in losses)
                if winner != min(range(len(finite_losses)), key=finite_losses.__getitem__):
                    raise ValueError("V* winner must equal the finite-loss argmin")
                executed.append(phase)
                candidate_answer = {"winner": winner, "losses": finite_losses}
        except Exception as error:
            return ObservationBatchResult(
                status="model_failed", batch_plan=plan, admitted=True, charged=True,
                ledger_before=ledger_before, ledger_after=self._ledger.to_dict(),
                current_support=current_support, candidate_support=candidate_support,
                failure_phase=phase, failure_reason=str(error),
                exception_type=type(error).__name__, executed_stages=tuple(executed),
                elapsed_seconds=time.perf_counter() - started,
            )
        return ObservationBatchResult(
            status="success", batch_plan=plan, admitted=True, charged=True,
            ledger_before=ledger_before, ledger_after=self._ledger.to_dict(),
            current_support=current_support, candidate_support=candidate_support,
            candidate_answer=candidate_answer, executed_stages=tuple(executed),
            elapsed_seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _zoom_merge_identity(crops: Sequence[Sequence[int]]) -> dict[str, Any]:
        per_descriptor = [list(crop) for crop in crops]
        merged = [list(crop) for crop in merge_bbox_list(
            [list(crop) for crop in per_descriptor], threshold=0,
        )]
        union = union_all_bboxes(merged)
        if union is None:
            raise ValueError("coordinate ZOOM requires at least one crop")
        identity_payload = {
            "per_descriptor_crop_xyxy": per_descriptor,
            "merged_crop_xyxy": merged,
            "union_crop_xyxy": list(union),
        }
        encoded = json.dumps(
            identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        return dict(
            identity_payload,
            identity_sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        )

    @staticmethod
    def _freeze_rgb(image: Any, name: str) -> Image.Image:
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise TypeError(f"{name} must be an RGB PIL image")
        return Image.frombytes("RGB", image.size, image.tobytes())

    def expand_context_observation_batch(
        self, *, source_image: Image.Image, q0: str, query_plan: QueryPlan,
        current_support_view: tuple[Any, ...] | None,
        expand_decision: ExpandDecision, answer_type: str, options: Any,
        composition_policy: str,
    ) -> ObservationBatchResult:
        """Observe exact P0 focus plus one pure spatial context panel."""
        started = time.perf_counter()
        ledger_before = copy.deepcopy(self._ledger.to_dict())
        source_image = self._freeze_rgb(source_image, "source_image")
        if not isinstance(q0, str) or not q0.strip():
            raise ValueError("q0 must be a nonempty string")
        if not isinstance(query_plan, QueryPlan) or query_plan.main_query != q0:
            raise ValueError("q0 must exactly match QueryPlan.main_query")
        if composition_policy != "focus_top_blank_or_context_bottom_native_pixels_v1":
            raise ValueError("EXPAND composition policy is not frozen")
        requirements = sanitize_evidence_requirements(query_plan.evidence_items)
        source_area = self._pixels(source_image)
        source_identity = _task2_source_identity(source_image)
        source_image_key = _task2_source_key(source_identity)
        empty_plan = {
            "schema_version": 1,
            "batch_kind": "p4a_post_anchor_expand_context",
            "answer_type": answer_type,
            "source_identity": source_identity,
            "q0_sha256": hashlib.sha256(q0.encode("utf-8")).hexdigest(),
            "requirement_order": [item.requirement_id for item in requirements],
            "requirement_set_id": EvidenceSupportResult.requirement_set_id_for(requirements),
            "verifier_status": "disabled_same_checkpoint_unpromoted",
            "current_support_calls": 0,
            "candidate_support_calls": 0,
            "candidate_answer_calls": 0,
            "candidate_option_count": 0,
            "pixels_per_logical_forward": source_area,
            "total_calls": 0,
            "total_pixels": 0,
        }
        if not requirements:
            return ObservationBatchResult(
                status="no_requirements", batch_plan=empty_plan,
                admitted=False, charged=False, ledger_before=ledger_before,
                ledger_after=self._ledger.to_dict(), failure_phase="preflight",
                failure_reason="no_requirements",
                elapsed_seconds=time.perf_counter() - started,
            )
        if self._answer_reserve_calls and not self._answer_started:
            return ObservationBatchResult(
                status="budget_rejected", batch_plan=empty_plan,
                admitted=False, charged=False, ledger_before=ledger_before,
                ledger_after=self._ledger.to_dict(), failure_phase="preflight",
                failure_reason="public_answer_reserve_unconsumed",
                elapsed_seconds=time.perf_counter() - started,
            )
        if answer_type not in {"option_list", "option_single", "logits_match"}:
            raise ValueError("EXPAND answer type is unsupported")
        if answer_type == "option_single":
            if not isinstance(options, str) or not options.strip():
                raise ValueError("single-choice options must be a nonempty string")
            frozen_options: str | tuple[str, ...] = options
        else:
            if isinstance(options, (str, bytes)) or not isinstance(options, Sequence):
                raise TypeError("options must be a sequence of strings")
            frozen_options = tuple(options)
            if not all(isinstance(option, str) and option for option in frozen_options):
                raise ValueError("options must contain nonempty strings")
        if answer_type == "option_list" and len(frozen_options) != 4:
            raise ValueError("HR EXPAND requires exactly four option blocks")
        if answer_type == "logits_match" and not frozen_options:
            raise ValueError("V* EXPAND requires nonempty options")
        if type(expand_decision) is not ExpandDecision or expand_decision.candidate is None:
            raise ValueError("EXPAND batch requires one successful pure spatial decision")
        support_call = getattr(self._model, "evidence_support", None)
        prepare_support = getattr(self._model, "_prepare_evidence_support", None)
        renderer = getattr(self._model, "process_nodes_to_image_list", None)
        get_patch = getattr(self._model, "get_patch", None)
        answer_method_name = (
            "free_form_using_nodes" if answer_type in {"option_list", "option_single"}
            else "multiple_choices_with_losses"
        )
        if not callable(support_call) or not callable(prepare_support):
            raise ValueError("raw model must expose answer-free support")
        if not callable(renderer) or not callable(get_patch):
            raise ValueError("raw model must expose the native EXPAND renderer")
        if not callable(getattr(self._model, answer_method_name, None)):
            raise ValueError("raw model is missing the candidate-answer method")

        focus_validated = self._validate_post_anchor_support_view(
            source_image, source_image_key, current_support_view,
        )
        context_validated = self._validate_post_anchor_support_view(
            source_image, source_image_key, (expand_decision.candidate,),
        )
        focus_nodes, current_keys, focus_renderer_ids, focus_descriptors = focus_validated
        context_nodes, context_keys, context_renderer_ids, context_descriptors = context_validated
        if (
            not focus_nodes or tuple(current_keys) != expand_decision.current_keys
            or tuple(focus_descriptors) != tuple(
                descriptor.to_dict()
                for descriptor in expand_decision.focus_descriptors
            )
        ):
            raise ValueError("EXPAND requires the exact nonempty P0 focus bundle")
        if any(
            descriptor["source"] == "global" or descriptor["renderer_kind"] == "root"
            for descriptor in focus_descriptors
        ):
            raise ValueError("EXPAND cannot preserve a global or root P0 focus")
        if (
            len(context_nodes) != 1 or context_keys[0] in current_keys
            or context_renderer_ids[0] in focus_renderer_ids
        ):
            raise ValueError("EXPAND context descriptor is duplicate")
        base_view_size = self.base_view_size
        patch_scale_value = getattr(self._model, "patch_scale", None)
        patch_scale = (
            None if patch_scale_value is None
            else _runtime_number(patch_scale_value, "EXPAND patch_scale")
        )
        background = getattr(self._model, "background_color", None)
        if (
            not isinstance(background, (list, tuple)) or len(background) != 3
            or any(isinstance(value, bool) or not isinstance(value, Integral)
                   or not 0 <= int(value) <= 255 for value in background)
        ):
            raise ValueError("raw model must expose a frozen RGB background color")
        background_rgb = tuple(int(value) for value in background)

        def native_crop(descriptor: Mapping[str, Any]) -> list[int]:
            patch_size = (
                base_view_size // 3 if descriptor["source"] == "fast"
                else base_view_size
            )
            scale = None if descriptor["source"] == "fast" else patch_scale
            raw_crop = get_patch(
                descriptor["bbox_original"], source_image.width, source_image.height,
                patch_size=patch_size, patch_scale=scale,
            )
            if (
                isinstance(raw_crop, (str, bytes)) or not isinstance(raw_crop, Sequence)
                or len(raw_crop) != 4
                or any(isinstance(value, bool) or not isinstance(value, Integral)
                       for value in raw_crop)
            ):
                raise ValueError("EXPAND native crop must be integer xyxy")
            crop = [int(value) for value in raw_crop]
            if not (
                0 <= crop[0] < crop[2] <= source_image.width
                and 0 <= crop[1] < crop[3] <= source_image.height
            ):
                raise ValueError("EXPAND native crop is outside the RGB source")
            return crop

        focus_crops = [native_crop(descriptor) for descriptor in focus_descriptors]
        context_crop = native_crop(context_descriptors[0])

        def render(nodes: tuple[Any, ...], name: str) -> Image.Image:
            self._model.view_size = base_view_size
            rendered_views = renderer(list(nodes), source_image.copy(), root_anyres=True)
            if not isinstance(rendered_views, (list, tuple)) or not rendered_views:
                raise ValueError("native Qwen renderer must return at least one view")
            rendered = rendered_views[0] if len(rendered_views) == 1 else rendered_views[-1]
            return self._freeze_rgb(rendered, name)

        try:
            focus_panel = render(focus_nodes, "EXPAND focus panel")
            context_panel = render(context_nodes, "EXPAND context panel")
        finally:
            self._model.view_size = base_view_size
        separator = 8
        canvas_size = (
            max(focus_panel.width, context_panel.width),
            focus_panel.height + separator + context_panel.height,
        )
        context_offset = (0, focus_panel.height + separator)
        blank_panel = Image.new("RGB", context_panel.size, background_rgb)
        current_rendered = Image.new("RGB", canvas_size, background_rgb)
        candidate_rendered = Image.new("RGB", canvas_size, background_rgb)
        current_rendered.paste(focus_panel, (0, 0))
        candidate_rendered.paste(focus_panel, (0, 0))
        candidate_rendered.paste(context_panel, context_offset)

        focus_hash = self._observation_sha256(focus_panel)
        context_hash = self._observation_sha256(context_panel)
        blank_hash = self._observation_sha256(blank_panel)
        current_hash = self._observation_sha256(current_rendered)
        candidate_hash = self._observation_sha256(candidate_rendered)
        composition = {
            "policy": composition_policy,
            "background_rgb": list(background_rgb),
            "separator_width": separator,
            "canvas_size": list(canvas_size),
            "focus_offset_xy": [0, 0],
            "context_offset_xy": list(context_offset),
            "focus_size": list(focus_panel.size),
            "context_size": list(context_panel.size),
            "focus_panel_sha256": focus_hash,
            "context_panel_sha256": context_hash,
            "blank_panel_sha256": blank_hash,
            "focus_current_rectangle_sha256": self._observation_sha256(
                current_rendered.crop((0, 0, focus_panel.width, focus_panel.height))
            ),
            "focus_candidate_rectangle_sha256": self._observation_sha256(
                candidate_rendered.crop((0, 0, focus_panel.width, focus_panel.height))
            ),
            "current_context_slot_sha256": self._observation_sha256(
                current_rendered.crop((
                    context_offset[0], context_offset[1],
                    context_offset[0] + context_panel.width,
                    context_offset[1] + context_panel.height,
                ))
            ),
            "candidate_context_slot_sha256": self._observation_sha256(
                candidate_rendered.crop((
                    context_offset[0], context_offset[1],
                    context_offset[0] + context_panel.width,
                    context_offset[1] + context_panel.height,
                ))
            ),
            "current_composite_sha256": current_hash,
            "candidate_composite_sha256": candidate_hash,
            "pixel_delta_region_xyxy": [
                0, context_offset[1], context_panel.width,
                context_offset[1] + context_panel.height,
            ],
        }
        composition_encoded = json.dumps(
            composition, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        composition["identity_sha256"] = hashlib.sha256(
            composition_encoded.encode("utf-8")
        ).hexdigest()

        candidate_keys = current_keys + context_keys
        candidate_renderer_ids = focus_renderer_ids + context_renderer_ids
        candidate_descriptors = focus_descriptors + context_descriptors

        def observation(
            rendered: Image.Image, keys: Sequence[str], renderer_ids: Sequence[str],
            descriptors: Sequence[Mapping[str, Any]],
        ) -> tuple[str, dict[str, Any]]:
            metadata = {
                "canonical_keys": list(keys),
                "renderer_identities": list(renderer_ids),
                "rendered_mode": rendered.mode,
                "rendered_size": [rendered.width, rendered.height],
                "view_sha256": self._observation_sha256(rendered),
                "descriptors": copy.deepcopy(list(descriptors)),
            }
            identity = dict(metadata)
            identity.pop("descriptors")
            return json.dumps(
                identity, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ), metadata

        current_identity, current_metadata = observation(
            current_rendered, current_keys, focus_renderer_ids, focus_descriptors,
        )
        candidate_identity, candidate_metadata = observation(
            candidate_rendered, candidate_keys, candidate_renderer_ids,
            candidate_descriptors,
        )
        prepared = prepare_support(q0, requirements)
        if prepared is None:
            raise AssertionError("nonempty requirements must produce support provenance")
        option_count = 1 if answer_type == "option_single" else len(frozen_options)
        answer_calls = (
            4 if answer_type == "option_list"
            else 1 if answer_type == "option_single" else 1 + option_count
        )
        total_calls = 2 + answer_calls
        total_pixels = total_calls * source_area
        options_payload = (
            frozen_options if answer_type == "option_single"
            else list(frozen_options)
        )
        options_encoded = json.dumps(
            options_payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        options_hash = hashlib.sha256(options_encoded.encode("utf-8")).hexdigest()
        if answer_type == "option_list":
            answer_prompts = [
                q0 + "\n" + option + "Answer the option letter directly."
                for option in frozen_options
            ]
            answer_prompt_hashes = [
                hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                for prompt in answer_prompts
            ]
        elif answer_type == "option_single":
            answer_prompts = [_single_choice_question(q0, frozen_options)]
            answer_prompt_hashes = [
                hashlib.sha256(answer_prompts[0].encode("utf-8")).hexdigest()
            ]
        else:
            call_payload = {"q0": q0, "options": options_payload}
            call_encoded = json.dumps(
                call_payload, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            )
            answer_prompt_hashes = [hashlib.sha256(call_encoded.encode("utf-8")).hexdigest()]
        q0_hash = hashlib.sha256(q0.encode("utf-8")).hexdigest()
        call_identity = {
            "answer_type": answer_type, "q0_sha256": q0_hash,
            "options_sha256": options_hash,
            "answer_prompt_sha256": answer_prompt_hashes,
        }
        call_identity_encoded = json.dumps(
            call_identity, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        context_descriptor = context_descriptors[0]
        x, y, width, height = (
            float(value) for value in context_descriptor["bbox_original"]
        )
        plan = {
            "schema_version": 1,
            "batch_kind": "p4a_post_anchor_expand_context",
            "answer_type": answer_type,
            "source_identity": source_identity,
            "source_width": source_image.width,
            "source_height": source_image.height,
            "accounted_source_area": source_area,
            "selection_policy": "nearest_spatial_native_cvsearch_context_v1",
            "composition_policy": composition_policy,
            "base_view_size": base_view_size,
            "patch_scale": patch_scale,
            "current_keys": current_keys,
            "candidate_keys": candidate_keys,
            "focus_role": {
                "role": "focus", "canonical_keys": current_keys,
                "renderer_identities": focus_renderer_ids,
                "descriptors": focus_descriptors,
                "native_crops_xyxy": focus_crops,
            },
            "context_role": {
                "role": "context", "canonical_key": context_keys[0],
                "renderer_identity": context_renderer_ids[0],
                "descriptor": context_descriptor,
                "native_crop_xyxy": context_crop,
                "candidate_bbox_xyxy": [x, y, x + width, y + height],
                "focus_union_xyxy": list(expand_decision.focus_union_xyxy),
                "positive_outside_area": expand_decision.positive_outside_area,
                "contained_by_focus_descriptor": False,
                "contains_complete_focus_union": False,
                "normalized_edge_gap": expand_decision.normalized_edge_gap,
                "rank_tuple": list(expand_decision.rank_tuple),
            },
            "focus_merge_identity": self._zoom_merge_identity(focus_crops),
            "context_merge_identity": self._zoom_merge_identity([context_crop]),
            "composition_identity": composition,
            "current_observation": current_metadata,
            "candidate_observation": candidate_metadata,
            "candidate_answer_input_sha256": candidate_hash,
            "q0": q0,
            "options": options_payload,
            "options_sha256": options_hash,
            "answer_prompt_sha256": answer_prompt_hashes,
            "answer_call_identity_sha256": hashlib.sha256(
                call_identity_encoded.encode("utf-8")
            ).hexdigest(),
            "q0_sha256": q0_hash,
            "requirement_order": [item.requirement_id for item in requirements],
            "requirement_set_id": EvidenceSupportResult.requirement_set_id_for(requirements),
            "prompt_version": prepared["prompt_version"],
            "prompt_template_sha256": prepared["prompt_template_sha256"],
            "current_prompt_sha256": prepared["prompt_sha256"],
            "candidate_prompt_sha256": prepared["prompt_sha256"],
            "processor_mode": prepared["processor_mode"],
            "processor_fingerprint": prepared["fingerprint"],
            "checkpoint": prepared["checkpoint"],
            "yes_tokenization": list(prepared["yes_tokens"]),
            "no_tokenization": list(prepared["no_tokens"]),
            "yes_token_id": prepared["yes_id"],
            "no_token_id": prepared["no_id"],
            "p_yes_transform": prepared["p_yes_transform"],
            "verifier_status": "disabled_same_checkpoint_unpromoted",
            "current_support_calls": 1,
            "candidate_support_calls": 1,
            "candidate_answer_calls": answer_calls,
            "candidate_option_count": option_count,
            "pixels_per_logical_forward": source_area,
            "total_calls": total_calls,
            "total_pixels": total_pixels,
        }
        _strict_json(plan, "EXPAND observation plan")
        plan, plan_hash, is_full_plan = _validate_batch_plan(
            plan, allow_expand_noop=current_hash == candidate_hash,
        )
        if not is_full_plan:
            raise AssertionError("EXPAND producer emitted a zero batch plan")
        if current_hash == candidate_hash:
            return ObservationBatchResult(
                status="render_noop", batch_plan=plan, admitted=False, charged=False,
                ledger_before=ledger_before, ledger_after=self._ledger.to_dict(),
                failure_phase="preflight", failure_reason="aggregate_render_unchanged",
                elapsed_seconds=time.perf_counter() - started,
            )
        try:
            self._consume_actual(total_calls, total_pixels)
        except BudgetExceeded as error:
            return ObservationBatchResult(
                status="budget_rejected", batch_plan=plan, admitted=False, charged=False,
                ledger_before=ledger_before, ledger_after=self._ledger.to_dict(),
                failure_phase="admission", failure_reason=str(error),
                exception_type=type(error).__name__,
                elapsed_seconds=time.perf_counter() - started,
            )

        def materialize(image: Image.Image) -> Image.Image:
            return Image.frombytes("RGB", image.size, image.tobytes())

        executed: list[str] = []
        current_support = None
        candidate_support = None
        phase = "current_support"
        try:
            self._model.view_size = base_view_size
            observed_support = support_call(
                question=q0, requirements=requirements,
                rendered_observation=materialize(current_rendered),
                observation_identity=current_identity,
            )
            if not isinstance(observed_support, EvidenceSupportResult):
                raise TypeError("current support returned an invalid result")
            observed_support = observed_support.with_batch_accounting(
                accounted_pixels=source_area, batch_plan_hash=plan_hash,
            )
            _validate_plan_support(observed_support, plan, plan_hash, "current")
            current_support = observed_support
            executed.append(phase)
            phase = "candidate_support"
            self._model.view_size = base_view_size
            observed_support = support_call(
                question=q0, requirements=requirements,
                rendered_observation=materialize(candidate_rendered),
                observation_identity=candidate_identity,
            )
            if not isinstance(observed_support, EvidenceSupportResult):
                raise TypeError("candidate support returned an invalid result")
            observed_support = observed_support.with_batch_accounting(
                accounted_pixels=source_area, batch_plan_hash=plan_hash,
            )
            _validate_plan_support(observed_support, plan, plan_hash, "candidate")
            candidate_support = observed_support
            executed.append(phase)
            if answer_type == "option_list":
                raw_outputs = []
                for index, prompt in enumerate(answer_prompts):
                    phase = f"hr_answer_{index}"
                    self._model.view_size = base_view_size
                    raw_output = self._model.free_form_using_nodes(
                        materialize(candidate_rendered), prompt, [],
                    )
                    if not isinstance(raw_output, str):
                        raise TypeError("HR candidate outputs must be strings")
                    raw_outputs.append(raw_output)
                    executed.append(phase)
                candidate_answer: Any = raw_outputs
            elif answer_type == "option_single":
                phase = "treebench_answer"
                self._model.view_size = base_view_size
                candidate_answer = self._model.free_form_using_nodes(
                    materialize(candidate_rendered), answer_prompts[0], [],
                )
                if not isinstance(candidate_answer, str):
                    raise TypeError("single-choice candidate output must be a string")
                executed.append(phase)
            else:
                phase = "vstar_answer"
                self._model.view_size = base_view_size
                winner, losses = self._model.multiple_choices_with_losses(
                    materialize(candidate_rendered), q0, frozen_options, [],
                )
                if isinstance(winner, bool) or not isinstance(winner, Integral):
                    raise ValueError("V* winner must be an option index")
                winner = int(winner)
                if not 0 <= winner < len(frozen_options):
                    raise ValueError("V* winner is outside the option range")
                if not isinstance(losses, (list, tuple)) or len(losses) != len(frozen_options):
                    raise ValueError("V* losses must match every option")
                finite_losses = tuple(
                    _runtime_number(loss, "V* option loss") for loss in losses
                )
                if winner != min(range(len(finite_losses)), key=finite_losses.__getitem__):
                    raise ValueError("V* winner must equal the finite-loss argmin")
                executed.append(phase)
                candidate_answer = {"winner": winner, "losses": finite_losses}
            phase = "result_validation"
            return ObservationBatchResult(
                status="success", batch_plan=plan, admitted=True, charged=True,
                ledger_before=ledger_before, ledger_after=self._ledger.to_dict(),
                current_support=current_support, candidate_support=candidate_support,
                candidate_answer=candidate_answer, executed_stages=tuple(executed),
                elapsed_seconds=time.perf_counter() - started,
            )
        except Exception as error:
            return ObservationBatchResult(
                status="model_failed", batch_plan=plan, admitted=True, charged=True,
                ledger_before=ledger_before, ledger_after=self._ledger.to_dict(),
                current_support=current_support, candidate_support=candidate_support,
                failure_phase=phase, failure_reason=str(error) or type(error).__name__,
                exception_type=type(error).__name__, executed_stages=tuple(executed),
                elapsed_seconds=time.perf_counter() - started,
            )
        finally:
            self._model.view_size = base_view_size

    def coordinate_zoom_observation_batch(
        self, *, source_image: Image.Image, q0: str, query_plan: QueryPlan,
        current_support_view: tuple[Any, ...] | None, answer_type: str,
        options: Any, render_policy: str,
    ) -> ObservationBatchResult:
        """Observe one tighter render of the exact local P0 coordinates."""
        started = time.perf_counter()
        ledger_before = copy.deepcopy(self._ledger.to_dict())
        source_image = self._freeze_rgb(source_image, "source_image")
        if not isinstance(q0, str) or not q0.strip():
            raise ValueError("q0 must be a nonempty string")
        if not isinstance(query_plan, QueryPlan) or query_plan.main_query != q0:
            raise ValueError("q0 must exactly match QueryPlan.main_query")
        if render_policy != "native_coordinate_crop_view_size_div3":
            raise ValueError("coordinate ZOOM render policy is not frozen")
        requirements = sanitize_evidence_requirements(query_plan.evidence_items)
        source_area = self._pixels(source_image)
        source_identity = _task2_source_identity(source_image)
        source_image_key = _task2_source_key(source_identity)
        empty_plan = {
            "schema_version": 1,
            "batch_kind": "p2c_post_anchor_coordinate_zoom",
            "answer_type": answer_type,
            "source_identity": source_identity,
            "q0_sha256": hashlib.sha256(q0.encode("utf-8")).hexdigest(),
            "requirement_order": [item.requirement_id for item in requirements],
            "requirement_set_id": EvidenceSupportResult.requirement_set_id_for(requirements),
            "verifier_status": "disabled_same_checkpoint_unpromoted",
            "current_support_calls": 0,
            "candidate_support_calls": 0,
            "candidate_answer_calls": 0,
            "candidate_option_count": 0,
            "pixels_per_logical_forward": source_area,
            "total_calls": 0,
            "total_pixels": 0,
        }
        if not requirements:
            return ObservationBatchResult(
                status="no_requirements", batch_plan=empty_plan,
                admitted=False, charged=False, ledger_before=ledger_before,
                ledger_after=self._ledger.to_dict(), failure_phase="preflight",
                failure_reason="no_requirements",
                elapsed_seconds=time.perf_counter() - started,
            )
        if self._answer_reserve_calls and not self._answer_started:
            return ObservationBatchResult(
                status="budget_rejected", batch_plan=empty_plan,
                admitted=False, charged=False, ledger_before=ledger_before,
                ledger_after=self._ledger.to_dict(), failure_phase="preflight",
                failure_reason="public_answer_reserve_unconsumed",
                elapsed_seconds=time.perf_counter() - started,
            )
        if answer_type not in {"option_list", "option_single", "logits_match"}:
            raise ValueError("coordinate ZOOM answer type is unsupported")
        if answer_type == "option_single":
            if not isinstance(options, str) or not options.strip():
                raise ValueError("single-choice options must be a nonempty string")
            frozen_options: str | tuple[str, ...] = options
        else:
            if isinstance(options, (str, bytes)) or not isinstance(options, Sequence):
                raise TypeError("options must be a sequence of strings")
            frozen_options = tuple(options)
            if not all(isinstance(option, str) and option for option in frozen_options):
                raise ValueError("options must contain nonempty strings")
        if answer_type == "option_list" and len(frozen_options) != 4:
            raise ValueError("HR coordinate ZOOM requires exactly four option blocks")
        if answer_type == "logits_match" and not frozen_options:
            raise ValueError("V* coordinate ZOOM requires nonempty options")
        support_call = getattr(self._model, "evidence_support", None)
        prepare_support = getattr(self._model, "_prepare_evidence_support", None)
        renderer = getattr(self._model, "process_nodes_to_image_list", None)
        get_patch = getattr(self._model, "get_patch", None)
        answer_method_name = (
            "free_form_using_nodes" if answer_type in {"option_list", "option_single"}
            else "multiple_choices_with_losses"
        )
        if not callable(support_call) or not callable(prepare_support):
            raise ValueError("raw model must expose answer-free support")
        if not callable(renderer) or not callable(get_patch):
            raise ValueError("raw model must expose the native coordinate crop renderer")
        if not callable(getattr(self._model, answer_method_name, None)):
            raise ValueError("raw model is missing the candidate-answer method")

        validated = self._validate_post_anchor_support_view(
            source_image, source_image_key, current_support_view,
        )
        nodes, current_keys, source_renderer_ids, descriptors = validated
        if not nodes:
            raise ValueError("coordinate ZOOM requires a nonempty local P0 support view")
        if any(
            descriptor["source"] == "global" or descriptor["renderer_kind"] == "root"
            for descriptor in descriptors
        ):
            raise ValueError("coordinate ZOOM cannot render global or root P0 views")
        base_view_size = self.base_view_size
        candidate_view_size = base_view_size // 3
        if candidate_view_size <= 0:
            raise ValueError("coordinate ZOOM candidate view size must be positive")

        def crop_for(descriptor: Mapping[str, Any], view_size: int) -> list[int]:
            source = descriptor["source"]
            patch_size = view_size // 3 if source == "fast" else view_size
            patch_scale = None if source == "fast" else getattr(
                self._model, "patch_scale", None,
            )
            raw_crop = get_patch(
                descriptor["bbox_original"], source_image.width, source_image.height,
                patch_size=patch_size, patch_scale=patch_scale,
            )
            if (
                isinstance(raw_crop, (str, bytes))
                or not isinstance(raw_crop, Sequence) or len(raw_crop) != 4
                or any(isinstance(value, bool) or not isinstance(value, Integral)
                       for value in raw_crop)
            ):
                raise ValueError("native coordinate crop must be integer xyxy")
            crop = [int(value) for value in raw_crop]
            if not (
                0 <= crop[0] < crop[2] <= source_image.width
                and 0 <= crop[1] < crop[3] <= source_image.height
            ):
                raise ValueError("native coordinate crop is outside the RGB source")
            return crop

        current_crops = [crop_for(descriptor, base_view_size) for descriptor in descriptors]
        candidate_crops = [
            crop_for(descriptor, candidate_view_size) for descriptor in descriptors
        ]
        coordinate_mapping: list[dict[str, Any]] = []
        zoom_keys: list[str] = []
        zoom_renderer_ids: list[str] = []
        candidate_descriptors: list[dict[str, Any]] = []
        for index, (
            descriptor, current_crop, candidate_crop,
        ) in enumerate(zip(descriptors, current_crops, candidate_crops)):
            zoom_payload = {
                "action": "P2C_ZOOM",
                "candidate_crop_xyxy": candidate_crop,
                "candidate_view_size": candidate_view_size,
                "current_key": descriptor["canonical_key"],
                "render_policy": render_policy,
                "source_image_key": source_image_key,
            }
            zoom_renderer_identity = json.dumps(
                zoom_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
            zoom_key = "zoom-" + hashlib.sha256(
                zoom_renderer_identity.encode("utf-8")
            ).hexdigest()
            zoom_keys.append(zoom_key)
            zoom_renderer_ids.append(zoom_renderer_identity)
            candidate_descriptors.append({
                "canonical_key": zoom_key,
                "renderer_identity": zoom_renderer_identity,
                "source_descriptor": copy.deepcopy(descriptor),
            })
            coordinate_mapping.append({
                "descriptor_index": index,
                "current_key": descriptor["canonical_key"],
                "zoom_key": zoom_key,
                "bbox_original": copy.deepcopy(descriptor["bbox_original"]),
                "depth": descriptor["depth"],
                "render_level": descriptor["render_level"],
                "posterior_score": descriptor["posterior_score"],
                "first_seen_ordinal": descriptor["first_seen_ordinal"],
                "tree_scope": descriptor["tree_scope"],
                "crop_origin": copy.deepcopy(descriptor["crop_origin"]),
                "source_image_key": descriptor["source_image_key"],
                "source": descriptor["source"],
                "renderer_kind": descriptor["renderer_kind"],
                "source_renderer_identity": descriptor["renderer_identity"],
                "zoom_renderer_identity": zoom_renderer_identity,
                "current_crop_xyxy": current_crop,
                "candidate_crop_xyxy": candidate_crop,
            })

        def render_at(view_size: int, name: str) -> Image.Image:
            self._model.view_size = view_size
            rendered_views = renderer(list(nodes), source_image.copy(), root_anyres=True)
            if not isinstance(rendered_views, (list, tuple)) or not rendered_views:
                raise ValueError("native Qwen renderer must return at least one view")
            rendered = rendered_views[0] if len(rendered_views) == 1 else rendered_views[-1]
            return self._freeze_rgb(rendered, name)

        try:
            current_rendered = render_at(base_view_size, "current rendered observation")
            candidate_rendered = render_at(
                candidate_view_size, "candidate rendered observation",
            )
        finally:
            self._model.view_size = base_view_size

        def observation(
            rendered: Image.Image, keys: Sequence[str], renderer_ids: Sequence[str],
            observation_descriptors: Sequence[Mapping[str, Any]],
        ) -> tuple[str, dict[str, Any]]:
            view_hash = self._observation_sha256(rendered)
            identity_payload = {
                "canonical_keys": list(keys),
                "renderer_identities": list(renderer_ids),
                "rendered_mode": rendered.mode,
                "rendered_size": [rendered.width, rendered.height],
                "view_sha256": view_hash,
            }
            identity = json.dumps(
                identity_payload, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            )
            return identity, dict(
                identity_payload,
                descriptors=copy.deepcopy(list(observation_descriptors)),
            )

        current_identity, current_metadata = observation(
            current_rendered, current_keys, source_renderer_ids, descriptors,
        )
        candidate_identity, candidate_metadata = observation(
            candidate_rendered, zoom_keys, zoom_renderer_ids, candidate_descriptors,
        )
        prepared = prepare_support(q0, requirements)
        if prepared is None:
            raise AssertionError("nonempty requirements must produce support provenance")
        option_count = 1 if answer_type == "option_single" else len(frozen_options)
        answer_calls = (
            4 if answer_type == "option_list"
            else 1 if answer_type == "option_single" else 1 + option_count
        )
        total_calls = 2 + answer_calls
        total_pixels = total_calls * source_area
        plan = {
            "schema_version": 1,
            "batch_kind": "p2c_post_anchor_coordinate_zoom",
            "answer_type": answer_type,
            "source_identity": source_identity,
            "source_width": source_image.width,
            "source_height": source_image.height,
            "accounted_source_area": source_area,
            "render_policy": render_policy,
            "base_view_size": base_view_size,
            "candidate_view_size": candidate_view_size,
            "current_keys": current_keys,
            "zoom_keys": zoom_keys,
            "coordinate_mapping": coordinate_mapping,
            "current_merge_identity": self._zoom_merge_identity(current_crops),
            "candidate_merge_identity": self._zoom_merge_identity(candidate_crops),
            "current_observation": current_metadata,
            "candidate_observation": candidate_metadata,
            "candidate_answer_input_sha256": candidate_metadata["view_sha256"],
            "q0_sha256": hashlib.sha256(q0.encode("utf-8")).hexdigest(),
            "requirement_order": [item.requirement_id for item in requirements],
            "requirement_set_id": EvidenceSupportResult.requirement_set_id_for(requirements),
            "prompt_version": prepared["prompt_version"],
            "prompt_template_sha256": prepared["prompt_template_sha256"],
            "current_prompt_sha256": prepared["prompt_sha256"],
            "candidate_prompt_sha256": prepared["prompt_sha256"],
            "processor_mode": prepared["processor_mode"],
            "processor_fingerprint": prepared["fingerprint"],
            "checkpoint": prepared["checkpoint"],
            "yes_tokenization": list(prepared["yes_tokens"]),
            "no_tokenization": list(prepared["no_tokens"]),
            "yes_token_id": prepared["yes_id"],
            "no_token_id": prepared["no_id"],
            "p_yes_transform": prepared["p_yes_transform"],
            "verifier_status": "disabled_same_checkpoint_unpromoted",
            "current_support_calls": 1,
            "candidate_support_calls": 1,
            "candidate_answer_calls": answer_calls,
            "candidate_option_count": option_count,
            "pixels_per_logical_forward": source_area,
            "total_calls": total_calls,
            "total_pixels": total_pixels,
        }
        _strict_json(plan, "coordinate ZOOM observation plan")
        changed_crop = any(
            left != right for left, right in zip(current_crops, candidate_crops)
        )
        changed_view = (
            current_metadata["view_sha256"] != candidate_metadata["view_sha256"]
        )
        plan, plan_hash, is_full_plan = _validate_batch_plan(
            plan, allow_zoom_noop=not (changed_crop and changed_view),
        )
        if not is_full_plan:
            raise AssertionError("coordinate ZOOM producer emitted a zero batch plan")
        if not changed_crop or not changed_view:
            return ObservationBatchResult(
                status="render_noop", batch_plan=plan, admitted=False, charged=False,
                ledger_before=ledger_before, ledger_after=self._ledger.to_dict(),
                failure_phase="preflight", failure_reason=(
                    "descriptor_crops_unchanged" if not changed_crop
                    else "aggregate_render_unchanged"
                ), elapsed_seconds=time.perf_counter() - started,
            )
        try:
            self._consume_actual(total_calls, total_pixels)
        except BudgetExceeded as error:
            return ObservationBatchResult(
                status="budget_rejected", batch_plan=plan, admitted=False, charged=False,
                ledger_before=ledger_before, ledger_after=self._ledger.to_dict(),
                failure_phase="admission", failure_reason=str(error),
                exception_type=type(error).__name__,
                elapsed_seconds=time.perf_counter() - started,
            )

        def materialize(image: Image.Image) -> Image.Image:
            return Image.frombytes("RGB", image.size, image.tobytes())

        executed: list[str] = []
        current_support = None
        candidate_support = None
        phase = "current_support"
        try:
            self._model.view_size = base_view_size
            observed_support = support_call(
                question=q0, requirements=requirements,
                rendered_observation=materialize(current_rendered),
                observation_identity=current_identity,
            )
            if not isinstance(observed_support, EvidenceSupportResult):
                raise TypeError("current support returned an invalid result")
            observed_support = observed_support.with_batch_accounting(
                accounted_pixels=source_area, batch_plan_hash=plan_hash,
            )
            _validate_plan_support(observed_support, plan, plan_hash, "current")
            current_support = observed_support
            executed.append(phase)
            phase = "candidate_support"
            self._model.view_size = base_view_size
            observed_support = support_call(
                question=q0, requirements=requirements,
                rendered_observation=materialize(candidate_rendered),
                observation_identity=candidate_identity,
            )
            if not isinstance(observed_support, EvidenceSupportResult):
                raise TypeError("candidate support returned an invalid result")
            observed_support = observed_support.with_batch_accounting(
                accounted_pixels=source_area, batch_plan_hash=plan_hash,
            )
            _validate_plan_support(observed_support, plan, plan_hash, "candidate")
            candidate_support = observed_support
            executed.append(phase)
            if answer_type == "option_list":
                raw_outputs = []
                for index, option_block in enumerate(frozen_options):
                    phase = f"hr_answer_{index}"
                    self._model.view_size = base_view_size
                    question_input = q0 + "\n" + option_block + (
                        "Answer the option letter directly."
                    )
                    raw_output = self._model.free_form_using_nodes(
                        materialize(candidate_rendered), question_input, [],
                    )
                    if not isinstance(raw_output, str):
                        raise TypeError("HR candidate outputs must be strings")
                    raw_outputs.append(raw_output)
                    executed.append(phase)
                candidate_answer: Any = raw_outputs
            elif answer_type == "option_single":
                phase = "treebench_answer"
                self._model.view_size = base_view_size
                candidate_answer = self._model.free_form_using_nodes(
                    materialize(candidate_rendered),
                    _single_choice_question(q0, frozen_options), [],
                )
                if not isinstance(candidate_answer, str):
                    raise TypeError("single-choice candidate output must be a string")
                executed.append(phase)
            else:
                phase = "vstar_answer"
                self._model.view_size = base_view_size
                winner, losses = self._model.multiple_choices_with_losses(
                    materialize(candidate_rendered), q0, frozen_options, [],
                )
                if isinstance(winner, bool) or not isinstance(winner, Integral):
                    raise ValueError("V* winner must be an option index")
                winner = int(winner)
                if not 0 <= winner < len(frozen_options):
                    raise ValueError("V* winner is outside the option range")
                if not isinstance(losses, (list, tuple)) or len(losses) != len(frozen_options):
                    raise ValueError("V* losses must match every option")
                finite_losses = tuple(
                    _runtime_number(loss, "V* option loss") for loss in losses
                )
                if winner != min(
                    range(len(finite_losses)), key=finite_losses.__getitem__,
                ):
                    raise ValueError("V* winner must equal the finite-loss argmin")
                executed.append(phase)
                candidate_answer = {"winner": winner, "losses": finite_losses}
            phase = "result_validation"
            return ObservationBatchResult(
                status="success", batch_plan=plan, admitted=True, charged=True,
                ledger_before=ledger_before, ledger_after=self._ledger.to_dict(),
                current_support=current_support, candidate_support=candidate_support,
                candidate_answer=candidate_answer, executed_stages=tuple(executed),
                elapsed_seconds=time.perf_counter() - started,
            )
        except Exception as error:
            return ObservationBatchResult(
                status="model_failed", batch_plan=plan, admitted=True, charged=True,
                ledger_before=ledger_before, ledger_after=self._ledger.to_dict(),
                current_support=current_support, candidate_support=candidate_support,
                failure_phase=phase, failure_reason=(
                    str(error) or type(error).__name__
                ),
                exception_type=type(error).__name__, executed_stages=tuple(executed),
                elapsed_seconds=time.perf_counter() - started,
            )
        finally:
            self._model.view_size = base_view_size

    def _ensure_answer_reserve(self) -> None:
        if not self._answer_reserve_calls or self._answer_reserve_pixels is not None:
            return
        if self._ledger.mllm_calls + self._answer_reserve_calls > self._ledger.max_mllm_calls:
            self._answer_reserve_invalid = True
            raise BudgetExceeded("complete answer does not fit the mllm_calls budget")
        if self._answer_image_loader is None:
            raise AssertionError("answer reserve requires a source-image loader")
        source_image = self._answer_image_loader()
        self._answer_reserve_pixels = self._answer_reserve_calls * self._pixels(source_image)
        if self._ledger.processed_pixels + self._answer_reserve_pixels > self._ledger.max_processed_pixels:
            self._answer_reserve_invalid = True
            raise BudgetExceeded("complete answer does not fit the processed_pixels budget")

    @property
    def answer_reserve_available(self) -> bool:
        return (
            bool(self._answer_reserve_calls)
            and not self._answer_started
            and not self._answer_reserve_invalid
        )

    @property
    def base_view_size(self) -> int:
        value = self._base_view_size
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 4:
            raise ValueError("zoom model view_size must be an integer of at least four")
        return int(value)

    def zoom_render_identities(self, image: Image.Image, node: Any,
                               render_levels: Sequence[int]) -> tuple[tuple[Any, ...], ...]:
        get_patch = getattr(self._model, "get_patch", None)
        if not callable(get_patch):
            raise ValueError("zoom model must expose label-free crop geometry")
        identities = []
        for level in render_levels:
            patch = get_patch(
                node.state.bbox,
                image.width,
                image.height,
                patch_size=level // 3,
                patch_scale=None,
            )
            identities.append(tuple(patch))
        return tuple(identities)

    def zoom_loss_rows(self, image: Image.Image, question: str, options: Any, node: Any,
                       render_levels: Sequence[int]) -> tuple[list[tuple[Any, Any]], str | None]:
        levels = tuple(render_levels)
        total_calls = len(levels) * self._choice_calls(options)
        total_pixels = total_calls * self._pixels(image)
        if self._ledger.mllm_calls + total_calls > self._ledger.max_mllm_calls:
            return [], "calls_budget_insufficient"
        if self._ledger.processed_pixels + total_pixels > self._ledger.max_processed_pixels:
            return [], "pixels_budget_insufficient"
        self._consume_actual(total_calls, total_pixels)
        rows: list[tuple[Any, Any]] = []
        try:
            for level in levels:
                self._model.view_size = level
                rows.append(self._model.multiple_choices_with_losses(
                    image, question, options, [node]
                ))
        finally:
            self._model.view_size = self.base_view_size
        return rows, None

    def _charge_search(self, calls: int, image: Image.Image | None = None) -> None:
        pixels = 0 if image is None else calls * self._pixels(image)
        self._ensure_answer_reserve()
        reserved_calls = self._answer_reserve_calls if not self._answer_started else 0
        reserved_pixels = int(self._answer_reserve_pixels or 0) if not self._answer_started else 0
        if self._ledger.mllm_calls + calls + reserved_calls > self._ledger.max_mllm_calls:
            raise BudgetExceeded("mllm_calls search admission would consume the final-answer reserve")
        if self._ledger.processed_pixels + pixels + reserved_pixels > self._ledger.max_processed_pixels:
            raise BudgetExceeded("processed_pixels search admission would consume the final-answer reserve")
        self._consume_actual(calls, pixels)

    def _charge_answer(self, calls: int, image: Image.Image) -> None:
        pixels = calls * self._pixels(image)
        if not self._answer_reserve_calls:
            self._consume_actual(calls, pixels)
            return
        self._ensure_answer_reserve()
        if self._answer_started:
            raise BudgetExceeded("the reserved final-answer operation was already consumed")
        if calls != self._answer_reserve_calls or pixels != self._answer_reserve_pixels:
            raise BudgetExceeded("final-answer operation does not match its reserved capacity")
        self._consume_actual(calls, pixels)
        self._answer_started = True

    def _charge_hr_terminal_batch(self) -> None:
        if self._answer_image_loader is None:
            raise AssertionError("HR terminal batch requires a source-image loader")
        source_image = self._answer_image_loader()
        self._consume_actual(4, 4 * self._pixels(source_image))

    def generate_visual_cues_using_ic(self, ic_examples: Any, question: str) -> Any:
        self._charge_search(1)
        return self._model.generate_visual_cues_using_ic(ic_examples, question)

    def get_confidence_value(self, nodes: Any, image_pil: Image.Image, *args: Any, **kwargs: Any) -> Any:
        self._charge_search(1, image_pil)
        return self._model.get_confidence_value(nodes, image_pil, *args, **kwargs)

    def free_form_using_nodes(self, image_pil: Image.Image, question: str, searched_nodes: Any, *args: Any,
                              **kwargs: Any) -> Any:
        if self._answer_type == "option_list":
            if not self._free_form_started:
                if self._answer_reserve_calls:
                    if self._answer_reserve_calls != 4:
                        raise AssertionError("reserved HR answer must contain four calls")
                    self._charge_answer(4, image_pil)
                else:
                    self._charge_hr_terminal_batch()
                self._free_form_started = True
                self._free_form_remaining = 4
            if self._free_form_remaining <= 0:
                raise BudgetExceeded("the HR terminal batch was already consumed")
            result = self._model.free_form_using_nodes(
                image_pil, question, searched_nodes, *args, **kwargs
            )
            self._free_form_remaining -= 1
            return result
        self._charge_search(1, image_pil)
        return self._model.free_form_using_nodes(image_pil, question, searched_nodes, *args, **kwargs)

    def free_form_batch(self, image_pil: Image.Image,
                        requests: Sequence[tuple[str, Any]]) -> list[Any]:
        batch = list(requests)
        if not batch:
            raise ValueError("free-form batch must be nonempty")
        if self._answer_reserve_calls:
            self._charge_answer(len(batch), image_pil)
            if self._answer_type == "option_list":
                self._free_form_started = True
                self._free_form_remaining = 0
        else:
            self._charge_search(len(batch), image_pil)
        return [
            self._model.free_form_using_nodes(image_pil, question, searched_nodes)
            for question, searched_nodes in batch
        ]

    @staticmethod
    def _choice_calls(options: Any) -> int:
        if not isinstance(options, Sequence) or isinstance(options, (str, bytes)) or not options:
            raise ValueError("options must be a nonempty sequence")
        return 1 + len(options)

    def multiple_choices_inference(self, image_pil: Image.Image, question: str, options: Any,
                                   searched_nodes: Any = None) -> Any:
        self._charge_answer(self._choice_calls(options), image_pil)
        return self._model.multiple_choices_inference(image_pil, question, options, searched_nodes)

    def multiple_choices_with_losses(self, image_pil: Image.Image, question: str, options: Any,
                                     searched_nodes: Any = None) -> Any:
        self._charge_answer(self._choice_calls(options), image_pil)
        return self._model.multiple_choices_with_losses(image_pil, question, options, searched_nodes)


def _image_for(policy: Mapping[str, Any], image_folder: str | None) -> Image.Image:
    path = policy["input_image"]
    if image_folder is not None:
        path = os.path.join(image_folder, path)
    with Image.open(path) as image:
        return image.convert("RGB")


def _as_answer_record(policy: Mapping[str, Any], raw: Any) -> AnswerRecord:
    answer_type = policy["answer_type"]
    if answer_type == "option_list":
        if not isinstance(policy["options"], list) or not isinstance(raw, list):
            raise ValueError("HR option_list requires option-block and output lists")
        return aggregate_hr_answers(policy["options"], raw)
    if answer_type == "option_single":
        return aggregate_single_choice(raw)
    return AnswerRecord(output=copy.deepcopy(raw), canonical_answer=copy.deepcopy(raw))


def _audit_score(answer: AnswerRecord, ledger: BudgetLedger) -> float:
    """Record the frozen state score without treating unavailable support as evidence."""
    if not isinstance(answer, AnswerRecord):
        raise TypeError("answer must be an AnswerRecord")
    if ledger.max_mllm_calls <= 0:
        normalized_cost = 0.0
    else:
        normalized_cost = ledger.mllm_calls / ledger.max_mllm_calls
    features = EvidenceStateScore(
        uncertainty=answer.uncertainty,
        support_avg=0.0,
        support_min=0.0,
        coverage=0.0,
        normalized_cost=normalized_cost,
    )
    return score_state(features)


def _root_answer(policy: Mapping[str, Any], model: _BudgetedZoomModel,
                 image_folder: str | None) -> AnswerRecord:
    image = _image_for(policy, image_folder)
    if policy["answer_type"] == "logits_match":
        winner, losses = model.multiple_choices_with_losses(
            image, policy["question"], policy["options"], []
        )
        record = aggregate_vstar_losses([losses])
        if record.output != winner:
            raise ValueError("root winner disagrees with option losses")
        return record
    if policy["answer_type"] == "option_list":
        requests = [
            (
                f"{policy['question']}\n{option_block}Answer the option letter directly.",
                [],
            )
            for option_block in policy["options"]
        ]
        raw = model.free_form_batch(image, requests)
        return aggregate_hr_answers(policy["options"], raw)
    raise ValueError("root_search_fallback currently supports only V* and HR-Bench answer types")


def _node_boxes(nodes: Sequence[Any]) -> tuple[tuple[int | float, ...], ...]:
    boxes: list[tuple[int | float, ...]] = []
    for node in nodes:
        try:
            box = tuple(node.state.bbox)
        except (AttributeError, TypeError):
            continue
        if len(box) == 4:
            boxes.append(box)
    return tuple(boxes)


def _zoom_node_ineligibility(node: Any, image: Image.Image, base_view_size: int) -> str | None:
    if getattr(node, "is_root", None) is not False:
        return "search_node_is_root"
    if getattr(node, "search_source", None) != "fast":
        return "search_node_is_not_fast"
    try:
        bbox = tuple(node.state.bbox)
    except (AttributeError, TypeError):
        return "search_node_bbox_is_invalid"
    if len(bbox) != 4 or any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value))
        for value in bbox
    ):
        return "search_node_bbox_is_invalid"
    x, y, width, height = (float(value) for value in bbox)
    if (
        x < 0 or y < 0 or width <= 0 or height <= 0
        or x + width > image.width or y + height > image.height
    ):
        return "search_node_bbox_is_invalid"
    if max(width, height) >= base_view_size:
        return "search_node_is_not_tighter_than_base_view"
    return None


def _normalized_query_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return " ".join(value.split()).casefold()


def _hr_local_perceptual_question_allowed(question: str, target: str) -> bool:
    """Match only direct questions about one frozen local visual attribute."""
    if (
        _GLOBAL_SCOPE.search(target)
        or _RELATION.search(target)
        or _HR_DISALLOWED_QUERY.search(target)
        or _HR_COORDINATED_TARGET.search(target)
    ):
        return False
    target_core = re.sub(r"^(?:the|a|an)\s+", "", target)
    if not target_core or (
        target_core not in _HR_LOCAL_ENTITY_TARGETS
        and (question, target_core) not in _HR_FROZEN_COMPLEX_LOCAL_TARGETS
    ):
        return False
    target_pattern = rf"(?:the\s+)?{re.escape(target_core)}"
    visual_attribute = r"(?:colou?rs?|shapes?|materials?|textures?|patterns?)"
    text_attribute = r"(?:texts?|words?|inscriptions?)"
    surface_verb = r"(?:written|displayed|visible|shown|printed|inscribed)"
    patterns = (
        rf"(?:what|which)\s+{visual_attribute}\??",
        rf"(?:what|which)\s+{visual_attribute}\s+(?:is|are)\s+{target_pattern}\??",
        rf"(?:what|which)\s+{visual_attribute}\s+(?:does|do)\s+{target_pattern}\s+have\??",
        rf"what\s+(?:is|are)\s+the\s+{visual_attribute}\s+of\s+{target_pattern}\??",
        rf"what\s+(?:is|are)\s+{target_pattern}(?:'s)?\s+{visual_attribute}\??",
        rf"(?:what|which)\s+material\s+(?:is|are)\s+{target_pattern}\s+made\s+(?:of|from)\??",
        rf"what\s+(?:is|are)\s+{target_pattern}\s+made\s+(?:of|from)\??",
        rf"(?:what|which)\s+pattern\s+(?:is|are)\s+(?:visible\s+)?(?:on|in)\s+{target_pattern}\??",
        rf"(?:what|which)\s+{text_attribute}\s+(?:is|are)\s+{surface_verb}\s+(?:on|in)\s+{target_pattern}\??",
        rf"(?:what|which)\s+{text_attribute}\s+(?:is|are)\s+(?:on|in)\s+{target_pattern}\??",
        rf"what\s+(?:is|are)\s+the\s+{text_attribute}\s+(?:on|in)\s+{target_pattern}\??",
        rf"what\s+(?:is|are)\s+(?:written|printed|inscribed)\s+(?:on|in)\s+{target_pattern}\??",
        rf"what's\s+the\s+{text_attribute}\s+{surface_verb}\s+(?:on|in)\s+{target_pattern}\s+in\s+the\s+image\??",
        rf"tell\s+me\s+the\s+{visual_attribute}\s+of\s+{target_pattern}\??",
    )
    if any(re.fullmatch(pattern, question) for pattern in patterns):
        return True
    return bool(
        _HR_READABLE_TARGET.search(target_core)
        and re.fullmatch(
            rf"what\s+(?:does|do)\s+{target_pattern}\s+(?:read|say)\??", question
        )
    )


def _hr_semantic_projection_allowed(
    answer_type: Any,
    question: Any,
    plan: QueryPlan | None,
    answer: AnswerRecord,
) -> bool:
    """Fail closed unless a stable HR answer belongs to the frozen local family."""
    if answer_type != "option_list" or not isinstance(answer, AnswerRecord):
        return False
    if answer.aggregation_available is not True:
        return False
    for value, threshold in ((answer.frequency, 0.75), (answer.margin, 0.5)):
        if isinstance(value, bool) or not isinstance(value, Real):
            return False
        number = float(value)
        if not math.isfinite(number) or number < threshold:
            return False
    if not isinstance(plan, QueryPlan) or plan.global_scope_required is not False:
        return False
    normalized_question = _normalized_query_text(question)
    if (
        normalized_question is None
        or _normalized_query_text(plan.main_query) != normalized_question
        or not isinstance(plan.targets, tuple)
        or len(plan.targets) != 1
    ):
        return False
    normalized_target = _normalized_query_text(plan.targets[0])
    if normalized_target is None or not isinstance(plan.evidence_items, tuple):
        return False

    target_detail_seen = False
    runtime_context_seen = False
    for item in plan.evidence_items:
        if not isinstance(item, Mapping):
            return False
        if item.get("kind") == "target_detail":
            if (
                target_detail_seen
                or set(item) != _TARGET_DETAIL_KEYS
                or _normalized_query_text(item.get("target")) != normalized_target
                or item.get("requirements") != ["presence", "visual_detail"]
            ):
                return False
            target_detail_seen = True
        elif item.get("kind") == "runtime_ranking_context":
            if (
                runtime_context_seen
                or set(item) != _RUNTIME_CONTEXT_KEYS
                or item.get("query_source") != "main_query_plus_current_visual_cue"
                or item.get("planned_augmented_queries_used") is not False
            ):
                return False
            runtime_context_seen = True
        else:
            return False
    if not target_detail_seen:
        return False
    if (
        _GLOBAL_SCOPE.search(normalized_question)
        or _RELATION.search(normalized_question)
        or _HR_DISALLOWED_QUERY.search(normalized_question)
    ):
        return False
    return _hr_local_perceptual_question_allowed(normalized_question, normalized_target)


def _zoom_plan_is_single_local_color_detail(plan: QueryPlan | None, question: str) -> bool:
    """Admit only the one local color template validated by the GPU ablation."""
    if not isinstance(plan, QueryPlan) or plan.global_scope_required is not False:
        return False
    normalized_question = _normalized_query_text(question)
    if normalized_question is None or _normalized_query_text(plan.main_query) != normalized_question:
        return False
    if not isinstance(plan.targets, tuple) or len(plan.targets) != 1:
        return False
    normalized_target = _normalized_query_text(plan.targets[0])
    if normalized_target is None:
        return False
    if not isinstance(plan.evidence_items, tuple):
        return False

    semantic_items: list[Mapping[str, Any]] = []
    runtime_context_seen = False
    for item in plan.evidence_items:
        if not isinstance(item, Mapping):
            return False
        if item.get("kind") == "runtime_ranking_context":
            if (
                runtime_context_seen
                or set(item) != _RUNTIME_CONTEXT_KEYS
                or item.get("query_source") != "main_query_plus_current_visual_cue"
                or item.get("planned_augmented_queries_used") is not False
            ):
                return False
            runtime_context_seen = True
        else:
            semantic_items.append(item)
    if len(semantic_items) != 1:
        return False
    target_detail = semantic_items[0]
    if (
        set(target_detail) != _TARGET_DETAIL_KEYS
        or target_detail.get("kind") != "target_detail"
        or _normalized_query_text(target_detail.get("target")) != normalized_target
        or target_detail.get("requirements") != ["presence", "visual_detail"]
    ):
        return False
    return normalized_question == f"what is the color of the {normalized_target}?"


def _ranker(config: Mapping[str, Any], scorer: Any, node_ranker: Any) -> Any:
    if not config["rerank_enabled"]:
        if node_ranker is not None or scorer is not None:
            raise ValueError("ranker/scorer supplied while rerank_enabled is false")
        return None
    if node_ranker is not None:
        if scorer is not None:
            raise ValueError("supply scorer or node_ranker, not both")
        base_ranker = node_ranker
    else:
        if scorer is None:
            raise ValueError("rerank_enabled requires an injected scorer or node_ranker")
        base_ranker = QueryAwareNodeRanker(
            scorer,
            beta=config["beta"],
            alpha=config["alpha"],
            visual_lambda=config["visual_lambda"],
            detail_alpha_discount=config.get("detail_alpha_discount", 0.0),
            context_alpha_gain=config.get("context_alpha_gain", 0.0),
            context_visual_discount=config.get("context_visual_discount"),
            appearance_descriptor_visual_relief=config.get(
                "appearance_descriptor_visual_relief"
            ),
        )
    if config["ranking_mode"] == "conservative_rrf":
        return ConservativeQueryRanker(
            base_ranker,
            rho=config["ranking_rho"],
            max_displacement=config["ranking_max_displacement"],
        )
    return base_ranker


def _collector_p0_bundle(
    collector: SearchStateCollector,
) -> tuple[tuple[str, ...], tuple[NextCandidate, ...] | None]:
    snapshots = [
        snapshot for snapshot in collector.to_dict()["snapshots"]
        if snapshot.get("event") == "p0_selected"
    ]
    if len(snapshots) != 1:
        return (), None
    selected = snapshots[0].get("selected_keys")
    if (
        not isinstance(selected, list)
        or not all(isinstance(key, str) and key for key in selected)
        or len(selected) != len(set(selected))
    ):
        return (), None
    keys = tuple(selected)
    return keys, collector.support_view(keys)


def _answer_observation_keys(
    nodes: Any, source_image: Image.Image,
) -> tuple[str, ...] | None:
    if not isinstance(nodes, (list, tuple)):
        return None
    source_key = _task2_source_key(_task2_source_identity(source_image))
    keys: list[str] = []
    for node in nodes:
        try:
            bbox_value = tuple(node.state.bbox)
            node_source = node.state.original_image_pil
        except (AttributeError, TypeError):
            return None
        if len(bbox_value) != 4 or not isinstance(node_source, Image.Image):
            return None
        if _task2_source_key(_task2_source_identity(node_source)) != source_key:
            return None
        bbox: list[int | float] = []
        for value in bbox_value:
            if (
                isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                return None
            number = float(value)
            bbox.append(int(number) if number.is_integer() else number)
        x, y, width, height = (float(value) for value in bbox)
        if (
            x < 0 or y < 0 or width <= 0 or height <= 0
            or x + width > source_image.width or y + height > source_image.height
        ):
            return None
        depth = getattr(node, "depth", 0)
        if isinstance(depth, bool) or not isinstance(depth, Integral) or int(depth) < 0:
            return None
        keys.append(_task2_canonical_key(tuple(bbox), int(depth), 0))
    if len(keys) != len(set(keys)):
        return None
    return tuple(keys)


def _bound_p0_observation_view(
    *, p0_keys: tuple[str, ...], p0_view: tuple[NextCandidate, ...] | None,
    answer_observations: Sequence[tuple[str, tuple[str, ...] | None, Any, Any]],
    raw_response: Any,
) -> tuple[NextCandidate, ...] | None:
    if p0_view is None or len(answer_observations) != 1:
        return None
    phase, observed_keys, observed_raw, observed_search_mode = answer_observations[-1]
    phase_matches = (
        (phase == "quick" and observed_search_mode == 0)
        or (
            phase == "search"
            and isinstance(observed_search_mode, Integral)
            and not isinstance(observed_search_mode, bool)
            and int(observed_search_mode) in {1, 2, 3}
        )
    )
    if (
        not phase_matches
        or observed_keys is None
        or observed_keys != p0_keys
        or not _outputs_agree(observed_raw, raw_response)
    ):
        return None
    return p0_view


def _support_contract_matches(
    config: Mapping[str, Any], result: ObservationBatchResult,
) -> bool:
    if result.status != "success":
        return False
    plan = result.batch_plan
    return (
        plan.get("prompt_version") == config["evidence_support_prompt_version"]
        and plan.get("processor_mode") == config["evidence_support_processor_mode"]
        and plan.get("prompt_template_sha256")
        == config["evidence_support_prompt_template_sha256"]
        and plan.get("p_yes_transform")
        == config["evidence_support_probability_transform"]
    )


def _normalized_batch_actual_cost(result: ObservationBatchResult | None) -> float | None:
    if result is None:
        return None
    payload = result.to_dict()
    before = payload["ledger_before"]
    after = payload["ledger_after"]
    maximum = after["max_mllm_calls"]
    if maximum <= 0:
        return 0.0
    return (after["mllm_calls"] - before["mllm_calls"]) / maximum


def get_evidence_gap_response(
    *,
    sam_model: Any,
    zoom_model: Any,
    nlp_model: Any,
    policy_annotation: Mapping[str, Any],
    original_annotation: Mapping[str, Any],
    ic_examples: Any,
    decomposed_question_template: str,
    config: str | os.PathLike[str] | Mapping[str, Any],
    image_folder: str | None = None,
    cvsearch_fn: Any = None,
    planner: Any = build_query_plan,
    policy_callback: Any = None,
    scorer: Any = None,
    node_ranker: Any = None,
    targets: Sequence[str] = (),
    answer_record_factory: Any = None,
) -> tuple[Any, MethodTrace]:
    """Run the honest minimal method without reading evaluator-only annotation fields."""
    # ``original_annotation`` is intentionally untouched; restoration is a separate boundary.
    policy = _policy_copy(policy_annotation)
    method_config = load_method_config(config)
    if not callable(planner):
        raise TypeError("planner must be callable")
    query_plan = planner(_policy_copy(policy), copy.deepcopy(targets))
    if not isinstance(query_plan, QueryPlan):
        raise TypeError("planner must return QueryPlan")
    _strict_json(query_plan.to_dict(), "query plan")
    if policy_callback is not None:
        if not callable(policy_callback):
            raise TypeError("policy_callback must be callable")
        callback_result = policy_callback(_policy_copy(policy))
        if callback_result is not None:
            _strict_json(callback_result, "policy callback result")

    ledger = BudgetLedger(
        max_mllm_calls=method_config["max_mllm_calls"],
        max_processed_pixels=method_config["max_processed_pixels"],
    )
    answer_reserve_calls = 0
    if method_config["mode"] == "rerank_only":
        if policy["answer_type"] == "logits_match":
            answer_reserve_calls = _BudgetedZoomModel._choice_calls(policy["options"])
        elif policy["answer_type"] == "option_list":
            if not isinstance(policy["options"], list) or len(policy["options"]) != 4:
                raise ValueError("HR rerank_only requires exactly four option blocks")
            answer_reserve_calls = 4
    budgeted_model = _BudgetedZoomModel(
        zoom_model,
        ledger,
        answer_reserve_calls=answer_reserve_calls,
        answer_image_loader=lambda: _image_for(policy, image_folder),
        answer_type=policy["answer_type"],
    )
    ranker = _ranker(method_config, scorer, node_ranker)
    effective_ranking_query = "cvsearch_default_order"
    if method_config["rerank_enabled"]:
        effective_ranking_query = (
            "main_query_plus_planned_augmented_queries"
            if query_plan.augmented_queries else "main_query_plus_current_visual_cue"
        )
    trace = MethodTrace(
        query_plan=copy.deepcopy(query_plan),
        method_mode=method_config["mode"],
        config_id=method_config["config_id"],
        effective_config=copy.deepcopy(method_config),
        effective_ranking_query=effective_ranking_query,
        pixel_accounting=method_config["pixel_accounting"],
    )
    runtime_annotation = _policy_copy(policy)
    observations: list[tuple[str, tuple[Any, ...], AnswerRecord, Any]] = []
    next_enabled = method_config.get("next_enabled") is True
    p2c_zoom_enabled = method_config.get("p2c_zoom_enabled") is True
    p4a_expand_enabled = method_config.get("p4a_expand_enabled") is True
    observation_action_enabled = next_enabled or p2c_zoom_enabled or p4a_expand_enabled
    next_source_image = (
        _image_for(policy, image_folder) if observation_action_enabled else None
    )
    next_collector = (
        SearchStateCollector(next_source_image) if next_source_image is not None else None
    )
    next_answer_observations: list[
        tuple[str, tuple[str, ...] | None, Any, Any]
    ] = []

    def observe(phase: str, nodes: list[Any], raw_answer: Any) -> None:
        if phase not in {"quick", "search"}:
            raise ValueError(f"unknown answer observation phase: {phase}")
        raw_snapshot = copy.deepcopy(raw_answer)
        if answer_record_factory is None:
            record = _as_answer_record(_policy_copy(policy), raw_snapshot)
        else:
            record = answer_record_factory(_policy_copy(policy), phase, raw_snapshot)
            if not isinstance(record, AnswerRecord):
                raise TypeError("answer_record_factory must return AnswerRecord")
            record = copy.deepcopy(record)
        record.selected_from = "root" if phase == "quick" else "search"
        _strict_json(record.to_dict(), "answer observation")
        observations.append((phase, tuple(nodes), record, raw_snapshot))
        if next_source_image is not None:
            next_answer_observations.append((
                phase,
                _answer_observation_keys(nodes, next_source_image),
                copy.deepcopy(raw_snapshot),
                copy.deepcopy(runtime_annotation.get("search_mode")),
            ))

    if cvsearch_fn is None:
        try:
            from CVSearch import get_cvsearch_response as cvsearch_fn
        except ImportError as error:
            raise ImportError("pass cvsearch_fn or import with the cvsearch directory on sys.path") from error

    started = time.perf_counter()
    root_record: AnswerRecord | None = None
    root_cost = 0
    budget_interrupted = False
    interrupted_record: AnswerRecord | None = None
    if method_config["mode"] == "root_search_fallback":
        root_record = _root_answer(_policy_copy(policy), budgeted_model, image_folder)
        root_record.selected_from = "root"
        _strict_json(root_record.to_dict(), "root answer")
        root_cost = ledger.mllm_calls

    cvsearch_kwargs = {
        "sam_model": sam_model,
        "zoom_model": budgeted_model,
        "nlp_model": nlp_model,
        "annotation": runtime_annotation,
        "ic_examples": copy.deepcopy(ic_examples),
        "decomposed_question_template": decomposed_question_template,
        "answering_confidence_threshold_upper": 0.9,
        "answering_confidence_threshold_lower": 0.0,
        "fast_threshold": method_config["quick_gate"],
        "pop_limit": lambda max_depth: max_depth * 3,
        "threshold_descrease": [0.05, 0.1, 0.2],
        "image_folder": image_folder,
        "search_mode": True,
        "enable_parent_verification": False,
        "node_ranker": ranker,
        "answer_observer": observe,
        "method_trace": trace,
        "search_state_sink": next_collector,
    }
    try:
        raw_response = cvsearch_fn(**cvsearch_kwargs)
    except BudgetExceeded:
        budget_interrupted = True
        if observations:
            _, _, interrupted_record, interrupted_raw = observations[-1]
            interrupted_record = copy.deepcopy(interrupted_record)
            interrupted_record.output = copy.deepcopy(interrupted_raw)
            raw_response = copy.deepcopy(interrupted_raw)
        elif root_record is not None:
            interrupted_record = copy.deepcopy(root_record)
            raw_response = copy.deepcopy(root_record.output)
        elif budgeted_model.answer_reserve_available:
            interrupted_record = _root_answer(_policy_copy(policy), budgeted_model, image_folder)
            interrupted_record.selected_from = "root"
            _strict_json(interrupted_record.to_dict(), "reserved root answer")
            raw_response = copy.deepcopy(interrupted_record.output)
        else:
            raise

    runtime_targets = runtime_annotation.get("targets")
    if not targets and planner is build_query_plan and isinstance(runtime_targets, (list, tuple)):
        runtime_plan = build_query_plan(_policy_copy(policy), runtime_targets)
        runtime_plan.evidence_items = tuple(runtime_plan.evidence_items) + ({
            "kind": "runtime_ranking_context",
            "query_source": "main_query_plus_current_visual_cue",
            "planned_augmented_queries_used": False,
        },)
        trace.query_plan = runtime_plan

    final_record: AnswerRecord
    final_observation_source: str
    output: Any
    if method_config["mode"] == "rerank_only":
        output = copy.deepcopy(raw_response)
        if interrupted_record is not None:
            final_record = copy.deepcopy(interrupted_record)
        else:
            final_record = copy.deepcopy(observations[-1][2]) if observations else _as_answer_record(policy, output)
        final_record.output = copy.deepcopy(output)
        if not final_record.selected_from:
            final_record.selected_from = "response"
        final_observation_source = final_record.selected_from
        if budget_interrupted:
            trace.history.append(HistoryRecord(
                step=0, answer=copy.deepcopy(final_record), cost=ledger.mllm_calls
            ))
    else:
        search_records = [item for item in observations if item[0] == "search"]
        if budget_interrupted:
            search_record = copy.deepcopy(interrupted_record)
            if search_record is not None and search_record.selected_from != "search":
                search_record = None
        elif search_records and policy["answer_type"] == "logits_match":
            _, nodes, _, raw_observed = search_records[-1]
            image = _image_for(policy, image_folder)
            try:
                winner, losses = budgeted_model.multiple_choices_with_losses(
                    image, policy["question"], policy["options"], list(nodes)
                )
            except BudgetExceeded:
                budget_interrupted = True
                search_record = None
            else:
                if winner != raw_observed:
                    raise ValueError("searched-node loss winner disagrees with CVSearch output")
                search_record = aggregate_vstar_losses([losses])
                search_record.selected_from = "search"
        elif search_records:
            _, _, observed_record, _ = search_records[-1]
            search_record = copy.deepcopy(observed_record)
            search_record.selected_from = "search"
        else:
            search_record = None
        if root_record is None:
            raise AssertionError("root fallback mode requires a root answer")
        if budget_interrupted and interrupted_record is not None:
            final_record = copy.deepcopy(interrupted_record)
        elif search_record is None:
            final_record = copy.deepcopy(root_record)
        elif policy["answer_type"] == "option_list" and (
            root_record.aggregation_available is False
            or search_record.aggregation_available is False
        ):
            final_record = copy.deepcopy(search_record)
            final_record.selected_from = "search"
        else:
            final_record = select_root_or_search(
                root_record, search_record, method_config["root_fallback_tolerance"]
            )
        final_observation_source = final_record.selected_from
        if policy["answer_type"] == "option_list":
            if method_config["hr_fusion_mode"] == "global_soft":
                final_record = soft_fuse_hr(
                    policy["options"], raw_response, final_record,
                    method_config["hr_fusion_gamma"],
                )
            else:
                final_record.output = copy.deepcopy(raw_response)
                final_record.selected_from = "cvsearch_anchor"
        output = copy.deepcopy(final_record.output)
        trace.history.append(HistoryRecord(step=0, answer=copy.deepcopy(root_record), cost=root_cost))
        if search_record is not None and (
            not budget_interrupted or final_observation_source == "search"
        ):
            trace.history.append(HistoryRecord(step=1, answer=copy.deepcopy(search_record), cost=ledger.mllm_calls))
        elif budget_interrupted and interrupted_record is not None and interrupted_record is not root_record:
            if not _outputs_agree(interrupted_record.output, root_record.output):
                trace.history.append(HistoryRecord(
                    step=1, answer=copy.deepcopy(interrupted_record), cost=ledger.mllm_calls
                ))

    if next_enabled:
        if next_collector is None or next_source_image is None:
            raise AssertionError("enabled NEXT requires its RGB source collector")

        # These three independent snapshots, not P0Anchor thawing, are the sole
        # source of the evaluator-facing return after the observation attempt.
        p0_output_snapshot = copy.deepcopy(output)
        p0_record_snapshot = copy.deepcopy(final_record)
        p0_raw_snapshot = copy.deepcopy(raw_response)
        if policy["answer_type"] == "option_list":
            p0_stability_snapshot = aggregate_hr_answers(
                policy["options"], p0_raw_snapshot,
            )
            p0_stability_snapshot.output = copy.deepcopy(p0_output_snapshot)
            p0_stability_snapshot.selected_from = "cvsearch_raw"
        else:
            p0_stability_snapshot = copy.deepcopy(p0_record_snapshot)
        p0_keys, collected_p0_view = _collector_p0_bundle(next_collector)
        if (
            policy["answer_type"] == "logits_match"
            and final_observation_source == "root"
            and not budget_interrupted
        ):
            producing_phase = "root"
            p0_keys = ()
            p0_support_view: tuple[NextCandidate, ...] | None = ()
        else:
            producing_phase = (
                "cvsearch_raw" if policy["answer_type"] == "option_list" else "search"
            )
            p0_support_view = None if budget_interrupted else _bound_p0_observation_view(
                p0_keys=p0_keys,
                p0_view=collected_p0_view,
                answer_observations=next_answer_observations,
                raw_response=p0_raw_snapshot,
            )
        p0_anchor = P0Anchor(
            emitted_answer=p0_output_snapshot,
            cvsearch_raw=p0_raw_snapshot,
            producing_phase=producing_phase,
            node_keys=p0_keys,
            support_view=p0_support_view,
        )

        next_no_op_reason: str | None = None
        candidate_keys: tuple[str, ...] = ()
        batch_result: ObservationBatchResult | None = None
        support_contract_status = "not_observed"
        candidate_stability: AnswerRecord | None = None
        g_next: float | None = None
        support_delta: float | None = None
        replacement_reason: str | None = None
        feasible = False

        try:
            requirements = sanitize_evidence_requirements(trace.query_plan.evidence_items)
        except (TypeError, ValueError):
            requirements = ()
            next_no_op_reason = "next_invalid_evidence_requirements"
        if next_no_op_reason is None and not requirements:
            next_no_op_reason = "next_no_evidence_requirements"
        if next_no_op_reason is None and p0_support_view is None:
            next_no_op_reason = "next_p0_support_view_unavailable"

        candidate: NextCandidate | None = None
        if next_no_op_reason is None:
            decision = next_collector.next_candidate()
            if decision.candidate is None:
                next_no_op_reason = decision.no_op_reason.value
            else:
                candidate = decision.candidate
                candidate_keys = (candidate.canonical_key,)

        if candidate is not None:
            try:
                batch_result = budgeted_model.post_anchor_observation_batch(
                    source_image=next_source_image,
                    q0=policy["question"],
                    query_plan=trace.query_plan,
                    current_support_view=p0_support_view,
                    candidate_support_view=(candidate,),
                    answer_type=policy["answer_type"],
                    options=policy["options"],
                )
            except (TypeError, ValueError):
                next_no_op_reason = "next_batch_preflight_failed"
            else:
                if batch_result.status == "success":
                    if _support_contract_matches(method_config, batch_result):
                        support_contract_status = "matched"
                        current_support = batch_result.current_support
                        candidate_support = batch_result.candidate_support
                        if current_support is None or candidate_support is None:
                            raise AssertionError("successful batch lost its support results")
                        g_next = 1.0 - current_support.p_yes
                        support_delta = candidate_support.p_yes - current_support.p_yes
                        if policy["answer_type"] == "option_list":
                            candidate_stability = aggregate_hr_answers(
                                policy["options"], batch_result.candidate_answer,
                            )
                        elif policy["answer_type"] == "option_single":
                            candidate_stability = aggregate_single_choice(
                                batch_result.candidate_answer,
                            )
                        else:
                            candidate_answer = batch_result.candidate_answer
                            candidate_stability = aggregate_vstar_losses(
                                [candidate_answer["losses"]]
                            )
                            if candidate_stability.output != candidate_answer["winner"]:
                                raise ValueError(
                                    "NEXT V* stability winner disagrees with exact loss argmin"
                                )
                        replacement_reason = "replacement_disabled_p2a"
                        feasible = True
                    else:
                        support_contract_status = "mismatch"
                        next_no_op_reason = "next_support_contract_mismatch"
                else:
                    next_no_op_reason = f"next_{batch_result.status}"

        next_audit = NextAudit(
            p0_anchor=p0_anchor,
            current_keys=p0_keys,
            candidate_keys=candidate_keys,
            batch_result=batch_result,
            uncertainty=p0_stability_snapshot.uncertainty,
            g_next=g_next,
            support_delta=support_delta,
            p0_stability=copy.deepcopy(p0_stability_snapshot),
            candidate_stability=candidate_stability,
            feasible=feasible,
            normalized_actual_cost=_normalized_batch_actual_cost(batch_result),
            support_contract_status=support_contract_status,
            _expected_p0_stability_json=json.dumps(
                p0_stability_snapshot.to_dict(), sort_keys=True,
                separators=(",", ":"), ensure_ascii=False, allow_nan=False,
            ),
            _p0_options=(
                tuple(policy["options"])
                if policy["answer_type"] == "option_list"
                else None
            ),
            _candidate_options=(
                tuple(policy["options"])
                if candidate_stability is not None
                and policy["answer_type"] == "option_list"
                else None
            ),
            replacement_reason=replacement_reason,
        )
        trace.steps.append(StepTrace(
            step=len(trace.steps),
            action=NEXT,
            focus_key=None if candidate is None else candidate.canonical_key,
            feasible_actions=(NEXT,) if feasible else (),
            gaps={} if g_next is None else {"g_next": g_next},
            no_op_reason=next_no_op_reason,
            answer=copy.deepcopy(p0_record_snapshot),
            budget=copy.deepcopy(ledger),
            next_audit=next_audit,
        ))
        trace.support_status = (
            "observed_answer_free" if feasible else next_no_op_reason
        )

        output = copy.deepcopy(p0_output_snapshot)
        final_record = copy.deepcopy(p0_record_snapshot)
        raw_response = copy.deepcopy(p0_raw_snapshot)

    if p2c_zoom_enabled:
        if next_collector is None or next_source_image is None:
            raise AssertionError("enabled P2C ZOOM requires its RGB source collector")

        p0_output_snapshot = copy.deepcopy(output)
        p0_record_snapshot = copy.deepcopy(final_record)
        p0_raw_snapshot = copy.deepcopy(raw_response)
        if policy["answer_type"] == "option_list":
            p0_stability_snapshot = aggregate_hr_answers(
                policy["options"], p0_raw_snapshot,
            )
            p0_stability_snapshot.output = copy.deepcopy(p0_output_snapshot)
            p0_stability_snapshot.selected_from = "cvsearch_raw"
        else:
            p0_stability_snapshot = copy.deepcopy(p0_record_snapshot)
        p0_keys, collected_p0_view = _collector_p0_bundle(next_collector)
        if (
            policy["answer_type"] == "logits_match"
            and final_observation_source == "root"
            and not budget_interrupted
        ):
            producing_phase = "root"
            p0_keys = ()
            p0_support_view: tuple[NextCandidate, ...] | None = ()
        else:
            producing_phase = (
                "cvsearch_raw" if policy["answer_type"] == "option_list" else "search"
            )
            p0_support_view = None if budget_interrupted else _bound_p0_observation_view(
                p0_keys=p0_keys,
                p0_view=collected_p0_view,
                answer_observations=next_answer_observations,
                raw_response=p0_raw_snapshot,
            )
        p0_anchor = P0Anchor(
            emitted_answer=p0_output_snapshot,
            cvsearch_raw=p0_raw_snapshot,
            producing_phase=producing_phase,
            node_keys=p0_keys,
            support_view=p0_support_view,
        )

        zoom_no_op_reason: str | None = None
        batch_result: ObservationBatchResult | None = None
        zoom_keys: tuple[str, ...] = ()
        support_contract_status = "not_observed"
        candidate_stability: AnswerRecord | None = None
        g_zoom_proxy: float | None = None
        support_delta: float | None = None
        replacement_reason: str | None = None
        feasible = False
        base_view_size = budgeted_model.base_view_size
        candidate_view_size = base_view_size // 3

        try:
            requirements = sanitize_evidence_requirements(trace.query_plan.evidence_items)
        except (TypeError, ValueError):
            requirements = ()
            zoom_no_op_reason = "zoom_invalid_evidence_requirements"
        if zoom_no_op_reason is None and not requirements:
            zoom_no_op_reason = "zoom_no_evidence_requirements"
        if zoom_no_op_reason is None and p0_support_view is None:
            zoom_no_op_reason = "zoom_p0_support_view_unavailable"
        if zoom_no_op_reason is None and not p0_support_view:
            zoom_no_op_reason = "zoom_p0_support_view_empty"
        if zoom_no_op_reason is None and any(
            descriptor.source == "global" or descriptor.renderer_kind == "root"
            for descriptor in p0_support_view
        ):
            zoom_no_op_reason = "zoom_p0_support_view_nonlocal"

        if zoom_no_op_reason is None:
            try:
                batch_result = budgeted_model.coordinate_zoom_observation_batch(
                    source_image=next_source_image,
                    q0=policy["question"],
                    query_plan=trace.query_plan,
                    current_support_view=p0_support_view,
                    answer_type=policy["answer_type"],
                    options=policy["options"],
                    render_policy=method_config["p2c_zoom_render_policy"],
                )
            except Exception:
                zoom_no_op_reason = "zoom_batch_preflight_failed"
            else:
                plan = batch_result.batch_plan
                if "zoom_keys" in plan:
                    zoom_keys = tuple(plan["zoom_keys"])
                if batch_result.status == "success":
                    if _support_contract_matches(method_config, batch_result):
                        support_contract_status = "matched"
                        current_support = batch_result.current_support
                        candidate_support = batch_result.candidate_support
                        if current_support is None or candidate_support is None:
                            raise AssertionError("successful ZOOM lost support results")
                        g_zoom_proxy = 1.0 - current_support.p_yes
                        support_delta = candidate_support.p_yes - current_support.p_yes
                        if policy["answer_type"] == "option_list":
                            candidate_stability = aggregate_hr_answers(
                                policy["options"], batch_result.candidate_answer,
                            )
                        elif policy["answer_type"] == "option_single":
                            candidate_stability = aggregate_single_choice(
                                batch_result.candidate_answer,
                            )
                        else:
                            candidate_answer = batch_result.candidate_answer
                            candidate_stability = aggregate_vstar_losses(
                                [candidate_answer["losses"]],
                            )
                            if candidate_stability.output != candidate_answer["winner"]:
                                raise ValueError(
                                    "P2C ZOOM V* stability winner disagrees with loss argmin"
                                )
                        replacement_reason = "replacement_disabled_p2c"
                        feasible = True
                    else:
                        support_contract_status = "mismatch"
                        zoom_no_op_reason = "zoom_support_contract_mismatch"
                else:
                    zoom_no_op_reason = f"zoom_{batch_result.status}"

        zoom_audit = ZoomAudit(
            p0_anchor=p0_anchor,
            current_keys=p0_keys,
            zoom_keys=zoom_keys,
            batch_result=batch_result,
            render_policy=method_config["p2c_zoom_render_policy"],
            base_view_size=base_view_size,
            candidate_view_size=candidate_view_size,
            uncertainty=p0_stability_snapshot.uncertainty,
            uncalibrated_g_zoom_proxy=g_zoom_proxy,
            support_delta=support_delta,
            p0_stability=copy.deepcopy(p0_stability_snapshot),
            candidate_stability=candidate_stability,
            feasible=feasible,
            normalized_actual_cost=_normalized_batch_actual_cost(batch_result),
            support_contract_status=support_contract_status,
            _expected_p0_stability_json=json.dumps(
                p0_stability_snapshot.to_dict(), sort_keys=True,
                separators=(",", ":"), ensure_ascii=False, allow_nan=False,
            ),
            _expected_p0_record_json=json.dumps(
                p0_record_snapshot.to_dict(), sort_keys=True,
                separators=(",", ":"), ensure_ascii=False, allow_nan=False,
            ),
            _p0_options=(
                tuple(policy["options"])
                if policy["answer_type"] == "option_list" else None
            ),
            _candidate_options=(
                tuple(policy["options"])
                if candidate_stability is not None
                and policy["answer_type"] == "option_list" else None
            ),
            replacement_reason=replacement_reason,
        )
        trace.steps.append(StepTrace(
            step=len(trace.steps),
            action=ZOOM,
            focus_key=zoom_audit.focus_key,
            feasible_actions=(ZOOM,) if feasible else (),
            gaps=(
                {} if g_zoom_proxy is None
                else {"g_zoom_proxy_audit_only": g_zoom_proxy}
            ),
            no_op_reason=zoom_no_op_reason,
            answer=copy.deepcopy(p0_record_snapshot),
            budget=copy.deepcopy(ledger),
            zoom_audit=zoom_audit,
        ))
        trace.support_status = (
            "observed_answer_free_audit_only" if feasible else zoom_no_op_reason
        )

        output = copy.deepcopy(p0_output_snapshot)
        final_record = copy.deepcopy(p0_record_snapshot)
        raw_response = copy.deepcopy(p0_raw_snapshot)

    if p4a_expand_enabled:
        if next_collector is None or next_source_image is None:
            raise AssertionError("enabled P4A EXPAND requires its RGB source collector")

        p0_output_snapshot = copy.deepcopy(output)
        p0_record_snapshot = copy.deepcopy(final_record)
        p0_raw_snapshot = copy.deepcopy(raw_response)
        if policy["answer_type"] == "option_list":
            p0_stability_snapshot = aggregate_hr_answers(
                policy["options"], p0_raw_snapshot,
            )
            p0_stability_snapshot.output = copy.deepcopy(p0_output_snapshot)
            p0_stability_snapshot.selected_from = "cvsearch_raw"
        else:
            p0_stability_snapshot = copy.deepcopy(p0_record_snapshot)
        p0_keys, collected_p0_view = _collector_p0_bundle(next_collector)
        if (
            policy["answer_type"] == "logits_match"
            and final_observation_source == "root"
            and not budget_interrupted
        ):
            producing_phase = "root"
            p0_keys = ()
            p0_support_view: tuple[NextCandidate, ...] | None = ()
        else:
            producing_phase = (
                "cvsearch_raw" if policy["answer_type"] == "option_list" else "search"
            )
            p0_support_view = None if budget_interrupted else _bound_p0_observation_view(
                p0_keys=p0_keys,
                p0_view=collected_p0_view,
                answer_observations=next_answer_observations,
                raw_response=p0_raw_snapshot,
            )
        p0_anchor = P0Anchor(
            emitted_answer=p0_output_snapshot,
            cvsearch_raw=p0_raw_snapshot,
            producing_phase=producing_phase,
            node_keys=p0_keys,
            support_view=p0_support_view,
        )

        expand_no_op_reason: str | None = None
        candidate_keys: tuple[str, ...] = ()
        batch_result: ObservationBatchResult | None = None
        support_contract_status = "not_observed"
        candidate_stability: AnswerRecord | None = None
        g_expand_proxy: float | None = None
        support_delta: float | None = None
        replacement_reason: str | None = None
        feasible = False

        try:
            requirements = sanitize_evidence_requirements(trace.query_plan.evidence_items)
        except (TypeError, ValueError):
            requirements = ()
            expand_no_op_reason = "expand_invalid_evidence_requirements"
        if expand_no_op_reason is None and not requirements:
            expand_no_op_reason = "expand_no_evidence_requirements"
        if expand_no_op_reason is None and p0_support_view is None:
            expand_no_op_reason = "expand_p0_focus_unavailable"
        if expand_no_op_reason is None and not p0_support_view:
            expand_no_op_reason = "expand_p0_focus_empty"
        if expand_no_op_reason is None and any(
            descriptor.source == "global" or descriptor.renderer_kind == "root"
            for descriptor in p0_support_view
        ):
            expand_no_op_reason = "expand_p0_focus_nonlocal"

        candidate: NextCandidate | None = None
        decision: ExpandDecision | None = None
        if expand_no_op_reason is None:
            collector_before = json.dumps(
                next_collector.to_dict(), sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            )
            decision = next_collector.peek_expand_candidate(p0_keys)
            collector_after = json.dumps(
                next_collector.to_dict(), sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            )
            if collector_before != collector_after:
                raise AssertionError("EXPAND candidate peek mutated CVSearch collector state")
            if decision.candidate is None:
                expand_no_op_reason = decision.no_op_reason.value
            else:
                candidate = decision.candidate
                candidate_keys = p0_keys + (candidate.canonical_key,)

        if candidate is not None and decision is not None:
            try:
                batch_result = budgeted_model.expand_context_observation_batch(
                    source_image=next_source_image,
                    q0=policy["question"],
                    query_plan=trace.query_plan,
                    current_support_view=p0_support_view,
                    expand_decision=decision,
                    answer_type=policy["answer_type"],
                    options=policy["options"],
                    composition_policy=(
                        "focus_top_blank_or_context_bottom_native_pixels_v1"
                    ),
                )
            except Exception:
                expand_no_op_reason = "expand_batch_preflight_failed"
            else:
                if batch_result.status == "success":
                    if _support_contract_matches(method_config, batch_result):
                        support_contract_status = "matched"
                        current_support = batch_result.current_support
                        candidate_support = batch_result.candidate_support
                        if current_support is None or candidate_support is None:
                            raise AssertionError("successful EXPAND lost support results")
                        g_expand_proxy = 1.0 - current_support.p_yes
                        support_delta = candidate_support.p_yes - current_support.p_yes
                        if policy["answer_type"] == "option_list":
                            candidate_stability = aggregate_hr_answers(
                                policy["options"], batch_result.candidate_answer,
                            )
                        elif policy["answer_type"] == "option_single":
                            candidate_stability = aggregate_single_choice(
                                batch_result.candidate_answer,
                            )
                        else:
                            candidate_answer = batch_result.candidate_answer
                            candidate_stability = aggregate_vstar_losses(
                                [candidate_answer["losses"]],
                            )
                            if candidate_stability.output != candidate_answer["winner"]:
                                raise ValueError(
                                    "P4A EXPAND V* stability winner disagrees with loss argmin"
                                )
                        replacement_reason = "replacement_disabled_p4a"
                        feasible = True
                    else:
                        support_contract_status = "mismatch"
                        expand_no_op_reason = "expand_support_contract_mismatch"
                else:
                    expand_no_op_reason = f"expand_{batch_result.status}"

        expand_audit = ExpandAudit(
            p0_anchor=p0_anchor,
            current_keys=p0_keys,
            candidate_keys=candidate_keys,
            selection_decision=decision,
            batch_result=batch_result,
            selection_policy=method_config["p4a_expand_selection_policy"],
            composition_policy="focus_top_blank_or_context_bottom_native_pixels_v1",
            uncertainty=p0_stability_snapshot.uncertainty,
            uncalibrated_g_expand_proxy=g_expand_proxy,
            support_delta=support_delta,
            p0_stability=copy.deepcopy(p0_stability_snapshot),
            candidate_stability=candidate_stability,
            feasible=feasible,
            normalized_actual_cost=_normalized_batch_actual_cost(batch_result),
            support_contract_status=support_contract_status,
            _expected_p0_stability_json=json.dumps(
                p0_stability_snapshot.to_dict(), sort_keys=True,
                separators=(",", ":"), ensure_ascii=False, allow_nan=False,
            ),
            _expected_p0_record_json=json.dumps(
                p0_record_snapshot.to_dict(), sort_keys=True,
                separators=(",", ":"), ensure_ascii=False, allow_nan=False,
            ),
            _expected_selection_json=json.dumps(
                None if decision is None else decision.to_dict(), sort_keys=True,
                separators=(",", ":"), ensure_ascii=False, allow_nan=False,
            ),
            _p0_options=(
                tuple(policy["options"])
                if policy["answer_type"] == "option_list" else None
            ),
            _candidate_options=(
                tuple(policy["options"])
                if candidate_stability is not None
                and policy["answer_type"] == "option_list" else None
            ),
            replacement_reason=replacement_reason,
        )
        trace.steps.append(StepTrace(
            step=len(trace.steps),
            action=EXPAND,
            focus_key=expand_audit.focus_key,
            feasible_actions=(EXPAND,) if feasible else (),
            gaps=(
                {} if g_expand_proxy is None
                else {"g_expand_proxy_audit_only": g_expand_proxy}
            ),
            no_op_reason=expand_no_op_reason,
            answer=copy.deepcopy(p0_record_snapshot),
            budget=copy.deepcopy(ledger),
            expand_audit=expand_audit,
        ))
        trace.support_status = (
            "observed_answer_free_audit_only" if feasible else expand_no_op_reason
        )

        output = copy.deepcopy(p0_output_snapshot)
        final_record = copy.deepcopy(p0_record_snapshot)
        raw_response = copy.deepcopy(p0_raw_snapshot)

    zoom_final_boxes: tuple[tuple[int | float, ...], ...] | None = None
    if method_config["enable_zoom"]:
        render_levels: tuple[int, ...] = ()
        render_identities: tuple[tuple[Any, ...], ...] = ()
        zoom_feasible: tuple[str, ...] = ()
        zoom_no_op_reason: str | None = None
        zoom_node: Any = None
        if policy["answer_type"] != "logits_match":
            zoom_no_op_reason = "answer_type_is_not_logits_match"
        elif not _zoom_plan_is_single_local_color_detail(trace.query_plan, policy["question"]):
            zoom_no_op_reason = "query_plan_is_not_single_target_detail"
        elif budget_interrupted:
            zoom_no_op_reason = "search_observation_is_incomplete"
        elif not search_records or len(search_records[-1][1]) != 1:
            zoom_no_op_reason = "requires_exactly_one_complete_search_node"
        else:
            zoom_node = search_records[-1][1][0]
            image = _image_for(policy, image_folder)
            base_view_size = budgeted_model.base_view_size
            render_levels = (base_view_size // 3, base_view_size // 4)
            zoom_no_op_reason = _zoom_node_ineligibility(zoom_node, image, base_view_size)
            if zoom_no_op_reason is None:
                render_identities = budgeted_model.zoom_render_identities(
                    image, zoom_node, render_levels
                )
                if len(set(render_identities)) < 2:
                    zoom_no_op_reason = "render_levels_not_distinct"
            if zoom_no_op_reason is None:
                rendered, zoom_no_op_reason = budgeted_model.zoom_loss_rows(
                    image, policy["question"], policy["options"], zoom_node, render_levels
                )
                if zoom_no_op_reason is None:
                    zoom_feasible = (ZOOM,)
                    loss_rows = []
                    winners = []
                    for winner, losses in rendered:
                        level_record = aggregate_vstar_losses([losses])
                        if level_record.output != winner:
                            raise ValueError("zoom winner disagrees with option losses")
                        winners.append(winner)
                        loss_rows.append(losses)
                    if len(set(winners)) != 1:
                        zoom_no_op_reason = "render_level_winners_disagree"
                    else:
                        final_record = aggregate_vstar_losses(loss_rows)
                        final_record.selected_from = "zoom"
                        output = copy.deepcopy(final_record.output)
                        zoom_final_boxes = _node_boxes((zoom_node,))

        render_label = ",".join(str(level) for level in render_levels) or "not_applicable"
        zoom_state = {
            "action": ZOOM,
            "render_levels": list(render_levels),
            "render_identities": [list(identity) for identity in render_identities],
            "feasible": bool(zoom_feasible),
            "no_op_reason": zoom_no_op_reason,
        }
        trace.history.append(HistoryRecord(
            step=2,
            answer=copy.deepcopy(final_record),
            cost=ledger.mllm_calls,
            state=zoom_state,
        ))
        trace.steps.append(StepTrace(
            step=0,
            action=ZOOM,
            focus_key=f"render_levels={render_label}",
            feasible_actions=zoom_feasible,
            no_op_reason=zoom_no_op_reason,
            answer=copy.deepcopy(final_record),
            budget=copy.deepcopy(ledger),
        ))

    boxes = runtime_annotation.get("searched_bbox", ())
    try:
        final_boxes = tuple(tuple(box) for box in boxes)
    except TypeError as error:
        raise ValueError("searched_bbox must be a sequence of boxes") from error
    if not final_boxes and final_observation_source in {"root", "search", "response"}:
        matching = [item for item in observations if item[2].selected_from == final_observation_source]
        if matching:
            final_boxes = _node_boxes(matching[-1][1])
    if final_record.selected_from == "zoom":
        final_boxes = zoom_final_boxes or ()
    elif final_observation_source == "root":
        final_boxes = ()

    trace.final_answer = copy.deepcopy(final_record)
    trace.anchor_answer = copy.deepcopy(trace.final_answer)
    if next_enabled or p2c_zoom_enabled or p4a_expand_enabled:
        trace.anchor_state_score = None
        trace.selected_state_score = None
        trace.replacement_margin = None
    else:
        audit_score = _audit_score(trace.final_answer, ledger)
        trace.anchor_state_score = audit_score
        trace.selected_state_score = audit_score
        trace.replacement_margin = 0.0
        trace.support_status = "not_observed"
    trace.final_boxes = final_boxes
    trace.budget = copy.deepcopy(ledger)
    trace.elapsed_seconds = time.perf_counter() - started
    trace.termination = FORCED_RETURN
    trace.budget_interrupted = budget_interrupted
    _capture_runtime_diagnostics(trace, runtime_annotation)
    trace.steps.append(StepTrace(
        step=len(trace.steps),
        action=FORCED_RETURN,
        no_op_reason="minimal_v1 has no certified-stop controller",
        answer=copy.deepcopy(final_record),
        budget=copy.deepcopy(ledger),
    ))
    if not _outputs_agree(output, trace.final_answer.output):
        raise ValueError("emitted output and trace.final_answer.output disagree")
    _strict_json(trace.to_dict(), "method trace")
    return output, trace


def compose_output_record(original_annotation: Mapping[str, Any], output: Any,
                          trace: MethodTrace) -> dict[str, Any]:
    """Restore evaluator fields only after policy/model work has completed."""
    if not isinstance(original_annotation, Mapping):
        raise TypeError("original_annotation must be a mapping")
    if not isinstance(trace, MethodTrace):
        raise TypeError("trace must be MethodTrace")
    if trace.final_answer is not None and not _outputs_agree(output, trace.final_answer.output):
        raise ValueError("emitted output and trace.final_answer.output disagree")
    if (
        "method_trace" in original_annotation
        or "_eg_ordinal" in original_annotation
        or "_eg_run_fingerprint" in original_annotation
        or "_eg_code_revision" in original_annotation
    ):
        raise ValueError("original annotation contains a reserved evidence-gap field")
    record = copy.deepcopy(dict(original_annotation))
    record["output"] = copy.deepcopy(output)
    trace_payload = trace.to_dict()
    record["method_trace"] = trace_payload
    if trace.cvsearch_search_mode is not None:
        record["search_mode"] = trace_payload["cvsearch_search_mode"]
    if trace.root_ans_conf is not None:
        record["root_ans_conf"] = trace_payload["root_ans_conf"]
    for key in ("num_pop", "num_zoom_in", "num_zoom_out"):
        record[key] = copy.deepcopy(trace_payload[key])
    _strict_json(record, "output record")
    return record
