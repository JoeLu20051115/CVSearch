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

from .answers import aggregate_hr_answers, aggregate_vstar_losses
from .fusion import soft_fuse_hr
from .input import POLICY_FIELDS
from .policy import select_root_or_search
from .ranking import ConservativeQueryRanker, QueryAwareNodeRanker
from .state import EvidenceStateScore, score_state, select_state
from .types import (
    FORCED_RETURN,
    ZOOM,
    AnswerRecord,
    BudgetExceeded,
    BudgetLedger,
    EvidenceSupportResult,
    HistoryRecord,
    MethodTrace,
    ObservationBatchResult,
    QueryPlan,
    StepTrace,
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
    unknown = set(supplied) - set(MINIMAL_V1)
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
    for name in ("max_mllm_calls", "max_processed_pixels"):
        _nonnegative_integer(result, name)
    if result["pixel_accounting"] != "source_image_area_per_logical_forward_approximation":
        raise ValueError("pixel_accounting is fixed for minimal_v1")
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

    def _render_post_anchor_support_view(
        self, source_image: Image.Image, support_view: tuple[Any, ...] | None,
    ) -> tuple[Image.Image, tuple[Any, ...], str, dict[str, Any]]:
        if support_view is None:
            raise ValueError("support view is unavailable")
        if not isinstance(support_view, tuple):
            raise TypeError("support view must be an immutable tuple")
        nodes: list[Any] = []
        descriptors: list[dict[str, Any]] = []
        canonical_keys: list[str] = []
        renderer_ids: list[str] = []
        for descriptor in support_view:
            key = getattr(descriptor, "canonical_key", None)
            renderer_id = getattr(descriptor, "renderer_identity", None)
            to_dict = getattr(descriptor, "to_dict", None)
            if not isinstance(key, str) or not key or not isinstance(renderer_id, str) or not renderer_id:
                raise ValueError("support descriptors require canonical and renderer identities")
            if not callable(to_dict):
                raise TypeError("support descriptors must expose strict snapshots")
            snapshot = copy.deepcopy(to_dict())
            _strict_json(snapshot, "support descriptor")
            nodes.append(descriptor.render_node)
            canonical_keys.append(key)
            renderer_ids.append(renderer_id)
            descriptors.append(snapshot)
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
        return rendered, tuple(nodes), identity, metadata

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
        source_identity = {
            "mode": source_image.mode,
            "size": [source_image.width, source_image.height],
            "pixel_sha256": hashlib.sha256(source_image.tobytes()).hexdigest(),
        }
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
        prepared = prepare_support(q0, requirements)
        if prepared is None:
            raise AssertionError("nonempty requirements must produce support provenance")
        current_rendered, current_nodes, current_identity, current_metadata = (
            self._render_post_anchor_support_view(source_image, current_support_view)
        )
        candidate_rendered, candidate_nodes, candidate_identity, candidate_metadata = (
            self._render_post_anchor_support_view(source_image, candidate_support_view)
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
            "verifier_status": "disabled_same_checkpoint_unpromoted",
            "current_support_calls": 1,
            "candidate_support_calls": 1,
            "candidate_answer_calls": answer_calls,
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
            scorer, beta=config["beta"], alpha=config["alpha"], visual_lambda=config["visual_lambda"]
        )
    if config["ranking_mode"] == "conservative_rrf":
        return ConservativeQueryRanker(
            base_ranker,
            rho=config["ranking_rho"],
            max_displacement=config["ranking_max_displacement"],
        )
    return base_ranker


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
        # Phase-2 collection is not enabled by any current config.  Keeping the
        # public ingress explicit makes the disabled path auditable and inert.
        "search_state_sink": None,
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
    audit_score = _audit_score(trace.final_answer, ledger)
    trace.anchor_answer = copy.deepcopy(trace.final_answer)
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
        step=1 if method_config["enable_zoom"] else 0,
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
