"""Standalone orchestration for independent query-aware adaptive search."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from PIL import Image

from qavs.evidence_gap.input import sanitize_annotation
from qavs.evidence_gap.pdf_controller import (
    InitialAssessmentBudgetExceeded,
    PDFTreeController,
    RouteDirective,
)
from qavs.evidence_gap.pdf_runtime import (
    CachedGeneratorObservations,
    CachedOptionLabelLosses,
    PDFStateEvaluator,
    TreeActionAdapter,
    TreeCatalog,
    build_pdf_query_plan,
    generate_text_only_response,
    question_kind_from_plan,
    query_required_roles,
    wrapper_yes_no_probability,
)
from qavs.evidence_gap.pdf_types import SearchStateRecord
from qavs.evidence_gap.provenance import canonical_sha256
from qavs.evidence_gap.query_profile import (
    EXPAND_REASONS,
    primary_expand_reason,
)
from qavs.evidence_gap.ranking import QueryAwareNodeRanker

from .config import GlobalGateConfig, IndependentSearchConfig
from .binary import NegativeCoverage
from .decision import (
    cumulative_phase_records,
    observe_global,
    select_accepted_hypothesis,
    select_global_verifier_challenger,
    source_image_identity,
)
from .frontend import materialize_frontend
from .grounding import TargetInstanceRegistry
from .semantics import OptionCatalog, build_option_catalog


@dataclass(frozen=True)
class DeltaRouteResult:
    directive: RouteDirective
    delta: float | None
    domain: tuple[str, ...]
    top_key: str
    support: float
    winning_label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.directive.action,
            "reason": self.directive.reason,
            "delta": self.delta,
            "domain": list(self.domain),
            "top_key": self.top_key,
            "support": self.support,
            "winning_label": self.winning_label,
        }


class SupportDeltaRouter:
    """Route from consecutive same-answer support changes, never from a trend test."""

    def __init__(self, *, patience: int, min_delta: float) -> None:
        if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
            raise ValueError("delta patience must be a positive integer")
        if (
            isinstance(min_delta, bool)
            or not isinstance(min_delta, (int, float))
            or not math.isfinite(float(min_delta))
            or float(min_delta) < 0.0
        ):
            raise ValueError("minimum delta must be finite and non-negative")
        self.patience = patience
        self.min_delta = float(min_delta)
        self._domain: tuple[str, ...] | None = None
        self._top_key: str | None = None
        self._support: float | None = None
        self._declines = 0
        self._insufficient = 0
        self._switches = 0

    def observe(
        self,
        *,
        domain: tuple[str, ...],
        top_key: str,
        support: float,
        winning_label: str,
    ) -> DeltaRouteResult:
        if not domain or any(not isinstance(item, str) or not item for item in domain):
            raise ValueError("delta domain must be a nonempty text tuple")
        if not isinstance(top_key, str) or not top_key:
            raise ValueError("delta top key must be nonempty")
        if (
            isinstance(support, bool) or not isinstance(support, (int, float))
            or not math.isfinite(float(support)) or not 0.0 <= float(support) <= 1.0
        ):
            raise ValueError("delta support must be a probability")
        if winning_label not in {"Support", "Refute", "Insufficient"}:
            raise ValueError("delta winning label is invalid")
        support = float(support)
        same_domain = self._domain == domain
        switched = same_domain and self._top_key not in {None, top_key}
        delta = (
            support - self._support
            if same_domain and not switched and self._support is not None else None
        )
        if not same_domain:
            self._switches = 0
            self._declines = 0
            self._insufficient = 0
        elif switched:
            self._switches += 1
            self._declines = 0
            self._insufficient = 0
        self._domain = domain
        self._top_key = top_key
        self._support = support

        if winning_label == "Refute":
            directive = RouteDirective("backtrack", "explicit_refute")
        else:
            self._insufficient = (
                self._insufficient + 1 if winning_label == "Insufficient" else 0
            )
            if delta is not None and delta < -self.min_delta:
                self._declines += 1
            elif delta is not None:
                self._declines = 0
            if self._switches >= self.patience:
                directive = RouteDirective("backtrack", "answer_switch_patience")
            elif self._insufficient >= self.patience:
                directive = RouteDirective("backtrack", "insufficient_patience")
            elif self._declines >= self.patience:
                directive = RouteDirective("backtrack", "consecutive_support_decline")
            elif delta is not None and delta > self.min_delta:
                directive = RouteDirective("continue", "same_answer_support_improved")
            else:
                directive = RouteDirective("continue", "complementary_evidence_needed")
        return DeltaRouteResult(
            directive=directive,
            delta=delta,
            domain=domain,
            top_key=top_key,
            support=support,
            winning_label=winning_label,
        )


def _strict_json(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be strict JSON") from error


def _state_dict(state: SearchStateRecord, offset: int) -> dict[str, Any]:
    payload = state.to_dict()
    payload["state_id"] += offset
    return payload


def _step(
    step: Any,
    offset: int,
    *,
    primary_expand_reason: str | None = None,
    include_restoration_provenance: bool = False,
) -> dict[str, Any]:
    is_expand = step.action.value == "EXPAND"
    if is_expand and primary_expand_reason not in EXPAND_REASONS:
        raise ValueError("EXPAND step requires one primary reason")
    if not is_expand and primary_expand_reason is not None:
        raise ValueError("non-EXPAND step cannot retain a primary reason")
    payload = {
        "action": step.action.value,
        "status": step.status,
        "before": _state_dict(step.before, offset),
        "after": _state_dict(step.after, offset),
        "reason": step.reason,
    }
    if is_expand:
        payload["primary_expand_reason"] = primary_expand_reason
    if include_restoration_provenance and step.action.value == "BACKTRACK":
        if step.restored_assessment_state_id is None:
            raise ValueError("deferred-gap BACKTRACK requires its source assessment")
        payload["restored_assessment_state_id"] = step.restored_assessment_state_id + offset
    return payload


def _budget_from_config(config: IndependentSearchConfig) -> dict[str, int]:
    return {
        "remaining_steps": config.budget.max_steps,
        "remaining_model_calls": config.budget.max_model_calls,
        "remaining_pixels": config.budget.max_processed_pixels,
    }


def _budget_from_state(state: SearchStateRecord) -> dict[str, int]:
    return {
        "remaining_steps": state.remaining_steps,
        "remaining_model_calls": state.remaining_model_calls,
        "remaining_pixels": state.remaining_pixels,
    }


def _phase_config(
    config: IndependentSearchConfig, budget: Mapping[str, int],
) -> Any:
    base = config.to_pdf_config()
    payload = base.to_dict()
    payload["budget"] = {
        "max_steps": budget["remaining_steps"],
        "max_model_calls": budget["remaining_model_calls"],
        "max_processed_pixels": budget["remaining_pixels"],
    }
    return type(base).from_mapping(payload)


def _record_state(value: Mapping[str, Any]) -> SearchStateRecord:
    payload = dict(value)
    for name in (
        "focus_keys", "path_keys", "context_keys", "visited_keys",
        "observation_keys",
    ):
        payload[name] = tuple(payload[name])
    return SearchStateRecord(**payload)


def _zoom_level(state: SearchStateRecord) -> int:
    prefix = f"{state.path_keys[-1]}@zoom"
    return max((
        int(item[len(prefix):])
        for item in state.observation_keys
        if item.startswith(prefix) and item[len(prefix):].isdigit()
    ), default=0)


def _effective_geometry(
    adapter: TreeActionAdapter, state: SearchStateRecord, verifier_policy: str,
) -> list[float]:
    return list(adapter.effective_geometry(state))


def _fit_size(width: int, height: int, max_edge: int) -> tuple[int, int]:
    edge = max(width, height)
    if edge <= max_edge:
        return width, height
    scale = max_edge / edge
    return max(1, round(width * scale)), max(1, round(height * scale))


def _rendered_scale(
    adapter: TreeActionAdapter,
    state: SearchStateRecord,
    verifier_policy: str,
    geometry: Sequence[float],
) -> float:
    if verifier_policy == "focus_only_target_detail_v1":
        width, height = adapter.render_local_verifier_view(state).size
    elif state.path_keys[-1] == adapter.catalog.root_key:
        width, height = _fit_size(*adapter.render_state(state)[0].size, 2048)
    elif _zoom_level(state):
        width, height = _fit_size(*adapter.render_state(state)[0].size, 2048)
    else:
        width, height = _fit_size(round(geometry[2]), round(geometry[3]), 2048)
    return float(min(width / geometry[2], height / geometry[3]))


def _recorded_rendered_scale(
    record: Mapping[str, Any], state: SearchStateRecord,
    geometry: Sequence[float], root_key: str,
) -> float:
    policy = record["verifier_view_policy"]
    if policy == "focus_only_target_detail_v1":
        width, height = record["verifier_view_size"]
    elif state.path_keys[-1] == root_key or _zoom_level(state):
        width, height = _fit_size(*record["answer_view_size"], 2048)
    else:
        width, height = _fit_size(round(geometry[2]), round(geometry[3]), 2048)
    return float(min(width / geometry[2], height / geometry[3]))


def _render_sha256(view: Image.Image) -> str:
    rgb = view.convert("RGB")
    header = f"RGB:{rgb.width}x{rgb.height}:".encode("ascii")
    return hashlib.sha256(header + rgb.tobytes()).hexdigest()


def _v3_action(state: SearchStateRecord, root_key: str) -> str:
    if state.path_keys[-1] == root_key:
        return "GLOBAL"
    observation = state.observation_keys[-1]
    if "@zoom" in observation:
        return "ZOOM"
    if observation.startswith("context:"):
        return "EXPAND"
    if len(state.path_keys) > 2:
        return "SPLIT"
    return "NEXT"


def _enrich_records(
    records: Sequence[Mapping[str, Any]],
    adapter: TreeActionAdapter,
    *,
    state_offset: int,
) -> list[dict[str, Any]]:
    source_identity = source_image_identity(adapter.image)
    enriched = []
    for value in records:
        record = copy.deepcopy(value)
        render_contract = record.pop("_render_contract", None)
        state = _record_state(record["state"])
        policy = record["verifier_view_policy"]
        geometry = (
            list(render_contract["effective_geometry"])
            if isinstance(render_contract, Mapping) else
            _effective_geometry(adapter, state, policy)
        )
        rendered_scale = (
            _recorded_rendered_scale(
                record, state, geometry, state.path_keys[0],
            )
            if isinstance(render_contract, Mapping) else
            _rendered_scale(adapter, state, policy, geometry)
        )
        record["state"].update({
            "state_id": state.state_id + state_offset,
            "source_image_identity": source_identity,
            "effective_geometry": geometry,
            "rendered_scale": rendered_scale,
            "branch_context": list(state.path_keys) + [
                f"context:{key}" for key in state.context_keys
            ],
        })
        if "option_support" in record:
            render_sha256 = (
                render_contract["render_sha256"]
                if isinstance(render_contract, Mapping) else
                _render_sha256(
                    adapter.render_local_verifier_view(state)
                    if policy == "focus_only_target_detail_v1" else
                    adapter.render_verifier_view(state)
                )
            )
            record["state"].update({
                "action": _v3_action(state, adapter.catalog.root_key),
                "render_sha256": render_sha256,
            })
        enriched.append(_strict_json(record, "state evaluation"))
    return enriched


def _initial_candidate_state(
    adapter: TreeActionAdapter,
    budget: Mapping[str, int],
) -> SearchStateRecord:
    candidates = adapter.queue.ranked_remaining((adapter.catalog.root_key,))
    if not candidates:
        raise ValueError("Revision-3 local search requires a ranked non-root candidate")
    key = candidates[0].canonical_key
    path = adapter.catalog.path_to(key)
    return SearchStateRecord(
        state_id=1,
        focus_keys=(key,),
        path_keys=path,
        context_keys=(),
        visited_keys=tuple(dict.fromkeys((adapter.catalog.root_key,) + path)),
        observation_keys=(f"{adapter.catalog.root_key}@root", f"{key}@base"),
        remaining_steps=budget["remaining_steps"],
        remaining_model_calls=budget["remaining_model_calls"],
        remaining_pixels=budget["remaining_pixels"],
    )


def _negative_candidate_coverage(
    records: Sequence[Mapping[str, Any]],
    candidate_keys: Sequence[str],
) -> NegativeCoverage:
    """Return candidate completeness plus deduplicated checked-region coverage."""
    expected = tuple(dict.fromkeys(candidate_keys))
    if not expected:
        return NegativeCoverage(False, 0.0)
    visited: set[str] = set()
    source_size: tuple[float, float] | None = None
    rectangles: set[tuple[float, float, float, float]] = set()
    for record in records:
        state = record.get("state") if isinstance(record, Mapping) else None
        keys = state.get("visited_keys") if isinstance(state, Mapping) else None
        if isinstance(keys, list):
            visited.update(key for key in keys if isinstance(key, str))
        if not isinstance(state, Mapping) or state.get("action") == "GLOBAL":
            continue
        source = state.get("source_image_identity")
        size = source.get("size") if isinstance(source, Mapping) else None
        geometry = state.get("effective_geometry")
        values = (
            size + geometry
            if isinstance(size, list) and isinstance(geometry, list) else []
        )
        if (
            not isinstance(size, list) or len(size) != 2
            or not isinstance(geometry, list) or len(geometry) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) for value in values)
        ):
            continue
        width, height = (float(value) for value in size)
        if width <= 0.0 or height <= 0.0:
            continue
        current_size = (width, height)
        if source_size is None:
            source_size = current_size
        elif source_size != current_size:
            return NegativeCoverage(False, 0.0)
        x, y, box_width, box_height = (float(value) for value in geometry)
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(width, x + box_width), min(height, y + box_height)
        if x1 > x0 and y1 > y0:
            rectangles.add((x0, y0, x1, y1))
    union_area = 0.0
    if source_size is not None and rectangles:
        x_edges = sorted({edge for box in rectangles for edge in (box[0], box[2])})
        for left, right in zip(x_edges, x_edges[1:]):
            intervals = sorted(
                (y0, y1) for x0, y0, x1, y1 in rectangles
                if x0 < right and x1 > left
            )
            covered_y = 0.0
            if intervals:
                start, end = intervals[0]
                for next_start, next_end in intervals[1:]:
                    if next_start > end:
                        covered_y += end - start
                        start, end = next_start, next_end
                    else:
                        end = max(end, next_end)
                covered_y += end - start
            union_area += (right - left) * covered_y
    ratio = (
        union_area / (source_size[0] * source_size[1])
        if source_size is not None else 0.0
    )
    return NegativeCoverage(set(expected).issubset(visited), min(1.0, ratio))


def _binary_no(output: Any, catalog: OptionCatalog | None) -> bool:
    return (
        catalog is not None
        and catalog.answer_type == "yes_no"
        and output == "no"
    )


def _candidate_pool(
    *,
    phase: str,
    catalog: TreeCatalog,
    adapter: TreeActionAdapter,
    final_state: SearchStateRecord,
    proposal_keys: Sequence[str],
    recovered_keys: Sequence[str],
    active_top_k: int,
) -> dict[str, Any]:
    remaining = [
        item.canonical_key
        for item in adapter.queue.ranked_remaining(final_state.visited_keys)
    ]
    return {
        "phase": phase,
        "root_key": catalog.root_key,
        "root_retained": catalog.root_key in catalog,
        "proposal_keys": list(proposal_keys),
        "recovered_keys": list(recovered_keys),
        "active_keys": remaining[:active_top_k],
        "reserve_keys": remaining[active_top_k:],
        "visited_keys": list(final_state.visited_keys),
        "remaining_ranked_keys": remaining,
    }


def _recovery_initial_state(
    *,
    catalog: TreeCatalog,
    adapter: TreeActionAdapter,
    previous: SearchStateRecord,
    recovered_keys: Sequence[str],
    budget: Mapping[str, int],
) -> SearchStateRecord:
    if not recovered_keys:
        raise ValueError("recovery phase requires at least one recovered key")
    visited = set(previous.visited_keys)
    frontier = set()
    for key in recovered_keys:
        path = catalog.path_to(key)
        if len(path) > 1 and path[-2] in visited:
            parent = path[-2]
            adapter.reveal_children(parent)
            frontier.add(key)
    start = next((
        item.canonical_key
        for item in adapter.queue.ranked_remaining(previous.visited_keys)
        if item.canonical_key in frontier
    ), None)
    if start is None:
        raise ValueError("recovered frontier is not reachable from the ranked queue")
    path = catalog.path_to(start)
    return SearchStateRecord(
        state_id=0,
        focus_keys=(start,),
        path_keys=path,
        context_keys=(),
        visited_keys=tuple(dict.fromkeys(previous.visited_keys + path)),
        observation_keys=previous.observation_keys + (f"{start}@base",),
        remaining_steps=budget["remaining_steps"],
        remaining_model_calls=budget["remaining_model_calls"],
        remaining_pixels=budget["remaining_pixels"],
    )


def _run_phase(
    *,
    phase: str,
    collector: Any,
    image: Image.Image,
    query_plan: Any,
    policy: Mapping[str, Any],
    config: IndependentSearchConfig,
    budget: Mapping[str, int],
    state_offset: int,
    proposal_keys: Sequence[str],
    recovered_keys: Sequence[str],
    generator_model: Any,
    verifier_model: Any,
    clip_scorer: Any,
    generator_checkpoint_sha256: str,
    verifier_checkpoint_sha256: str,
    option_catalog: OptionCatalog | None,
    target_registry: TargetInstanceRegistry | None,
    shared_records: Sequence[Mapping[str, Any]],
    question_kind: str,
    required_roles: tuple[str, ...],
    timing_origin: float,
    conditional_losses: CachedOptionLabelLosses | None = None,
    previous_state: SearchStateRecord | None = None,
    pair_scoring: Any | None = None,
) -> dict[str, Any]:
    catalog = TreeCatalog.from_collector(collector, image)
    if target_registry is not None:
        target_registry.add_candidates(catalog.candidate_records())
    ranking = config.ranking
    ranker = QueryAwareNodeRanker(
        clip_scorer,
        alpha=ranking.alpha if config.modules.query_ranking else 0.0,
        beta=ranking.beta,
        visual_lambda=ranking.visual_lambda,
        top_k_augmented=ranking.top_k_augmented,
    )
    adapter = TreeActionAdapter(
        catalog,
        image,
        query_plan,
        ranker,
        min_zoom_factor=(
            config.observation_geometry.min_zoom_factor
            if config.observation_geometry is not None else 0.4
        ),
        context_max_normalized_gap=(
            config.observation_geometry.context_max_normalized_gap
            if config.observation_geometry is not None else 1.0
        ),
        target_box_provider=(
            target_registry.target_bbox_for_state
            if target_registry is not None else None
        ),
    )
    evaluator = PDFStateEvaluator(
        generator_model=generator_model,
        adapter=adapter,
        policy_annotation=policy,
        query_plan=query_plan,
        verifier_probability=(
            None if config.schema_version == 3 else
            lambda view, prompt: wrapper_yes_no_probability(
                verifier_model, view, prompt,
            )
        ),
        verifier_checkpoint_sha256=verifier_checkpoint_sha256,
        generator_checkpoint_sha256=generator_checkpoint_sha256,
        option_catalog=(option_catalog if config.schema_version == 3 else None),
        conditional_losses=conditional_losses,
        target_registry=target_registry,
        grounding_conditional_losses=conditional_losses,
        grounding_threshold=(
            config.verification.grounding_threshold
            if config.verification is not None else None
        ),
        prior_bundle_records=(
            shared_records if config.schema_version == 3 else ()
        ),
        state_id_offset=(state_offset if config.schema_version == 3 else 0),
        defer_gaps=config.schema_version == 3,
        pair_scoring=pair_scoring,
    )
    initial_state = None
    if previous_state is not None:
        initial_state = _recovery_initial_state(
            catalog=catalog,
            adapter=adapter,
            previous=previous_state,
            recovered_keys=recovered_keys,
            budget=budget,
        )
    elif config.schema_version == 3:
        initial_state = _initial_candidate_state(adapter, budget)

    delta_router = SupportDeltaRouter(
        patience=config.controller.stall_patience,
        min_delta=config.controller.min_progress,
    )
    route_events: list[dict[str, Any]] = []

    def assess_with_timing(state: SearchStateRecord) -> Any:
        assessment = evaluator(state)
        if config.schema_version == 3:
            elapsed = time.perf_counter() - timing_origin
            if elapsed <= 0.0:
                raise AssertionError("assessment prefix timing must be positive")
            evaluator.records[-1]["prefix_elapsed_seconds"] = elapsed
        return assessment

    assess_with_timing.adapter = adapter
    assess_with_timing.estimate_cost = evaluator.estimate_cost

    def route_after_assessment(
        state: SearchStateRecord, assessment: Any,
    ) -> RouteDirective:
        current_records = _enrich_records(
            evaluator.records, adapter, state_offset=state_offset,
        )
        cumulative = cumulative_phase_records(
            shared_records, current_records, root_key=catalog.root_key,
        )
        decision = select_accepted_hypothesis(
            cumulative,
            root_state_id=0,
            acceptance=config.acceptance,
            validation_enabled=config.modules.evidence_validation,
            root_fallback_answer=cumulative[0]["answer"]["output"],
            option_catalog=option_catalog,
            question_kind=question_kind,
            required_roles=required_roles,
            negative_coverage=_negative_candidate_coverage(
                cumulative,
                tuple(proposal_keys) + tuple(recovered_keys),
            ),
            negative_coverage_for_records=lambda prefix: _negative_candidate_coverage(
                prefix, tuple(proposal_keys) + tuple(recovered_keys),
            ),
        )
        if decision.accepted_hypothesis is not None:
            route_events.append({
                "state_id": state.state_id + state_offset,
                "action": "accept",
                "reason": "paper_acceptance_gate",
                "delta": None,
            })
            return RouteDirective("accept", "paper_acceptance_gate")
        record = current_records[-1]
        vector = (
            record["evidence_bundle"]["option_support"]
            if "evidence_bundle" in record else record["option_support"]
        )
        options = vector["options"]
        top = min(
            options,
            key=lambda item: (-item["normalized_support"], item["key"]),
        )
        domain = (
            ("bundle", question_kind) + required_roles
            if "evidence_bundle" in record else tuple(record["state"]["path_keys"])
        )
        routed = delta_router.observe(
            domain=domain,
            top_key=top["key"],
            support=top["raw_support"],
            winning_label=top["distribution"]["winner"],
        )
        event = routed.to_dict()
        event["state_id"] = state.state_id + state_offset
        route_events.append(event)
        return routed.directive

    try:
        result = PDFTreeController(_phase_config(config, budget)).run(
            root_key=catalog.root_key,
            initial_state=initial_state,
            assess=assess_with_timing,
            feasible=adapter.feasible,
            execute=adapter.execute,
            branch_available=adapter.branch_available,
            route_after_assessment=(
                route_after_assessment if config.schema_version == 3 else None
            ),
        )
    except InitialAssessmentBudgetExceeded as error:
        if previous_state is None:
            raise
        final_state = error.state
        records: list[dict[str, Any]] = []
        budget_after = _budget_from_state(final_state)
        phase_trace = {
            "phase": phase,
            "termination": "INITIAL_ASSESSMENT_BUDGET_EXCEEDED",
            "selected_history_state_id": None,
            "assessment_count": 0,
            "budget_before": dict(budget),
            "budget_after": budget_after,
            "final_state": _state_dict(final_state, state_offset),
            "steps": [],
            "joint_ranking": copy.deepcopy(adapter.ranking_details),
            "state_evaluations": [],
        }
        if config.schema_version == 3:
            phase_trace["route_events"] = route_events
        pool = _candidate_pool(
            phase=phase,
            catalog=catalog,
            adapter=adapter,
            final_state=final_state,
            proposal_keys=proposal_keys,
            recovered_keys=recovered_keys,
            active_top_k=config.proposals.active_top_k,
        )
        return {
            "trace": phase_trace,
            "pool": pool,
            "records": records,
            "budget_after": budget_after,
            "pool_exhausted": not pool["remaining_ranked_keys"],
            "next_state_offset": state_offset + final_state.state_id + 1,
            "steps": (),
            "candidate_count": adapter.queue.candidate_count,
            "final_state": final_state,
        }
    evaluator.finalize_gap_trace()
    records = _enrich_records(
        evaluator.records, adapter, state_offset=state_offset,
    )
    record_by_state_id = {
        record["state"]["state_id"]: record for record in records
    }
    for step in result.steps:
        if step.action.value == "BACKTRACK" and step.restored_assessment_state_id is not None:
            source_id = step.restored_assessment_state_id + state_offset
            if source_id not in record_by_state_id:
                raise ValueError("BACKTRACK restoration references an unknown assessment")
            record_by_state_id[step.after.state_id + state_offset] = record_by_state_id[source_id]

    def action_expand_reason(step: Any) -> str | None:
        if step.action.value != "EXPAND":
            return None
        state_id = step.before.state_id + state_offset
        record = record_by_state_id.get(state_id)
        gap = record.get("gap") if isinstance(record, Mapping) else None
        scores = gap.get("scores") if isinstance(gap, Mapping) else None
        if not isinstance(scores, Mapping):
            raise ValueError("EXPAND action lacks its action-time gap scores")
        return primary_expand_reason(query_plan, scores)
    final_state = result.final_state
    budget_after = _budget_from_state(final_state)
    phase_trace = {
        "phase": phase,
        "termination": result.termination.value,
        "selected_history_state_id": result.selected_history_state_id + state_offset,
        "assessment_count": result.assessment_count,
        "budget_before": dict(budget),
        "budget_after": budget_after,
        "final_state": _state_dict(final_state, state_offset),
        "steps": [
            _step(
                item,
                state_offset,
                primary_expand_reason=action_expand_reason(item),
                include_restoration_provenance=bool(evaluator.deferred_gap_state_ids),
            )
            for item in result.steps
        ],
        "joint_ranking": copy.deepcopy(adapter.ranking_details),
        "state_evaluations": copy.deepcopy(records),
    }
    if config.schema_version == 3:
        phase_trace["route_events"] = route_events
    pool = _candidate_pool(
        phase=phase,
        catalog=catalog,
        adapter=adapter,
        final_state=final_state,
        proposal_keys=proposal_keys,
        recovered_keys=recovered_keys,
        active_top_k=config.proposals.active_top_k,
    )
    return {
        "trace": phase_trace,
        "pool": pool,
        "records": records,
        "budget_after": budget_after,
        "pool_exhausted": not pool["remaining_ranked_keys"],
        "next_state_offset": state_offset + final_state.state_id + 1,
        "steps": result.steps,
        "candidate_count": adapter.queue.candidate_count,
        "final_state": final_state,
    }


def _append_unique_phase_records(
    cumulative: list[dict[str, Any]],
    phase_records: Sequence[dict[str, Any]],
    *,
    root_key: str,
) -> None:
    cumulative[:] = cumulative_phase_records(
        cumulative, phase_records, root_key=root_key,
    )


def _execution_provenance(
    image: Image.Image, generator_checkpoint_sha256: str,
    verifier_checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "source_image_identity": source_image_identity(image),
        "generator_checkpoint_sha256": generator_checkpoint_sha256,
        "verifier_checkpoint_sha256": verifier_checkpoint_sha256,
    }


def _option_catalog_trace(catalog: OptionCatalog) -> dict[str, Any]:
    entries = [
        {"key": item.key, "text": item.text, "output": item.output}
        for item in catalog.entries
    ]
    return {
        "answer_type": catalog.answer_type,
        "identity_sha256": catalog.identity_sha256,
        "projection_sha256": canonical_sha256([
            {"key": item["key"], "output": item["output"]}
            for item in entries
        ]),
        "entries": entries,
    }


def _global_evidence_record(
    global_observation: Any,
    image: Image.Image,
    root_key: str,
    prefix_elapsed_seconds: float,
) -> dict[str, Any]:
    vector = global_observation.option_support
    if not isinstance(vector, Mapping):
        raise ValueError("Revision-3 search requires the immutable global option vector")
    answer = global_observation.answer_record or {
        "output": global_observation.output,
        "canonical_answer": global_observation.output,
        "frequency": global_observation.consistency,
        "aggregation_available": True,
    }
    answer = copy.deepcopy(answer)
    answer["output"] = global_observation.output
    return _strict_json({
        "prefix_elapsed_seconds": prefix_elapsed_seconds,
        "state": {
            "state_id": 0,
            "focus_keys": [root_key],
            "path_keys": [root_key],
            "context_keys": [],
            "visited_keys": [root_key],
            "observation_keys": [f"{root_key}@root"],
            "source_image_identity": source_image_identity(image),
            "effective_geometry": [0.0, 0.0, float(image.width), float(image.height)],
            "rendered_scale": 1.0,
            "branch_context": [root_key],
            "action": "GLOBAL",
            "render_sha256": _render_sha256(image),
            "positive_support_eligible": global_observation.accepted,
        },
        "answer": answer,
        "option_support": copy.deepcopy(dict(vector)),
        "assessment": {
            "uncertainty": 1.0 - global_observation.consistency,
            "support_avg": global_observation.support,
            "support_min": global_observation.support,
            "model_calls": global_observation.cost.model_calls,
            "processed_pixels": global_observation.cost.processed_pixels,
        },
        "verifier_view_policy": "immutable_full_image_v1",
    }, "global evidence record")


def _direct_trace(
    global_observation: Any,
    execution_provenance: Mapping[str, Any],
    config: IndependentSearchConfig,
    option_catalog: OptionCatalog | None,
    *,
    selected_output: Any | None = None,
    source: str = "validated_full_image",
    reason: str = "global_sufficient_and_accepted",
) -> dict[str, Any]:
    output = (
        global_observation.output
        if selected_output is None else _strict_json(
            selected_output, "selected direct output",
        )
    )
    trace = {
        "schema_version": config.schema_version,
        "method": config.method,
        "mode": "direct",
        "execution_provenance": copy.deepcopy(dict(execution_provenance)),
        "global_observation": global_observation.to_dict(),
        "validated_full_image": global_observation.validation_dict(),
        "sam_proposals": None,
        "candidate_pool": [],
        "scan_recover": [],
        "evidence_ledger": {
            "schema_version": config.schema_version,
            "aggregation": config.acceptance.aggregation,
            "root_state_id": None,
            "events": [],
            "groups": [],
        },
        "accepted_hypothesis": None,
        "controller": {"phases": []},
        "module_activity": {
            "global_observation": True,
            "sam": False,
            "planner": False,
            "phase_count": 0,
            "state_evaluations": 0,
            "actions": {"changed": {}, "no_op": {}},
        },
        "final_decision": {
            "source": source,
            "reason": reason,
            "output_sha256": canonical_sha256(output),
        },
    }
    if config.schema_version == 3:
        if option_catalog is None:
            raise AssertionError("schema 3 direct trace lacks an option catalog")
        trace.update({
            "option_catalog": _option_catalog_trace(option_catalog),
            "global_evidence_record": None,
            "query_plan": None,
            "question_kind": None,
            "required_roles": [],
            "target_registry": None,
        })
    return trace


def run_independent_sample(
    *,
    original_annotation: Mapping[str, Any],
    image_folder: str | Path,
    ic_examples: Any,
    config: IndependentSearchConfig,
    sam_model: Any,
    generator_model: Any,
    verifier_model: Any,
    nlp_model: Any,
    clip_scorer: Any,
    generator_checkpoint_sha256: str,
    verifier_checkpoint_sha256: str,
    generator_family: str,
) -> tuple[Any, dict[str, Any]]:
    """Run independent roles concurrently on supported single-device backends."""
    from qavs.evidence_gap.parallel_scoring import parallel_model_scoring

    with parallel_model_scoring(generator_model, verifier_model) as pair_scoring:
        return _run_independent_sample(
            original_annotation=original_annotation, image_folder=image_folder,
            ic_examples=ic_examples, config=config, sam_model=sam_model,
            generator_model=generator_model, verifier_model=verifier_model,
            nlp_model=nlp_model, clip_scorer=clip_scorer,
            generator_checkpoint_sha256=generator_checkpoint_sha256,
            verifier_checkpoint_sha256=verifier_checkpoint_sha256,
            generator_family=generator_family, pair_scoring=pair_scoring,
        )


def _run_independent_sample(
    *,
    original_annotation: Mapping[str, Any],
    image_folder: str | Path,
    ic_examples: Any,
    config: IndependentSearchConfig,
    sam_model: Any,
    generator_model: Any,
    verifier_model: Any,
    nlp_model: Any,
    clip_scorer: Any,
    generator_checkpoint_sha256: str,
    verifier_checkpoint_sha256: str,
    generator_family: str,
    pair_scoring: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run global validation, proposal search, and at most one recovery phase."""
    timing_origin = time.perf_counter()
    if not isinstance(config, IndependentSearchConfig):
        raise TypeError("config must be IndependentSearchConfig")
    generator_model = CachedGeneratorObservations(generator_model)
    policy = sanitize_annotation(original_annotation)
    option_catalog = (
        build_option_catalog(policy) if config.schema_version == 3 else None
    )
    image_value = Path(policy["input_image"]).expanduser()
    image_path = image_value if image_value.is_absolute() else Path(image_folder) / image_value
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")

    conditional_losses = (
        CachedOptionLabelLosses(verifier_model) if config.schema_version == 3 else None
    )
    global_observation = observe_global(
        model=generator_model,
        verifier_model=verifier_model,
        policy=policy,
        image=image,
        direct_threshold=config.direct_threshold,
        gate=config.global_gate,
        generator_checkpoint_sha256=generator_checkpoint_sha256,
        verifier_checkpoint_sha256=verifier_checkpoint_sha256,
        option_catalog=(option_catalog if config.schema_version == 3 else None),
        conditional_losses=conditional_losses,
        pair_scoring=pair_scoring,
    )
    global_prefix_elapsed_seconds = time.perf_counter() - timing_origin
    execution_provenance = _execution_provenance(
        image, generator_checkpoint_sha256, verifier_checkpoint_sha256,
    )
    global_verifier_output = None
    if (
        config.schema_version == 3
        and config.acceptance.aggregation.startswith("global_verifier_then_")
    ):
        if option_catalog is None:
            raise AssertionError("schema 3 hybrid lacks an option catalog")
        challenger_gate = GlobalGateConfig(
            min_support=config.acceptance.global_verifier_min_support,
            min_margin=config.acceptance.global_verifier_min_margin,
            min_consistency=0.0,
        )
        global_verifier_output = select_global_verifier_challenger(
            global_observation.option_support,
            catalog=option_catalog,
            gate=challenger_gate,
        )
    if global_verifier_output is not None and not _binary_no(
        global_verifier_output, option_catalog,
    ):
        output = _strict_json(
            global_verifier_output, "global verifier challenger output",
        )
        return output, _strict_json(
            _direct_trace(
                global_observation, execution_provenance, config,
                option_catalog, selected_output=output,
                source="accepted_global_verifier",
                reason="global_verifier_support_and_margin",
            ),
            "independent global verifier trace",
        )
    if global_observation.direct_stop and not _binary_no(
        global_observation.output, option_catalog,
    ):
        output = _strict_json(global_observation.output, "independent direct output")
        return output, _strict_json(
            _direct_trace(
                global_observation, execution_provenance, config, option_catalog,
            ),
            "independent direct trace",
        )

    proposal_frontend = materialize_frontend(
        image=image,
        question=policy["question"],
        ic_examples=ic_examples,
        sam_model=sam_model,
        zoom_model=generator_model,
        nlp_model=nlp_model,
        proposal_config=config.proposals,
    )
    recovery_events: list[dict[str, Any]] = []
    recovered_keys: tuple[str, ...] = ()
    immediate_recovery = not proposal_frontend.coverage.adequate
    if immediate_recovery:
        trigger = (
            "empty_proposals"
            if proposal_frontend.coverage.valid_count == 0
            else "coverage_failed"
        )
        recovery = proposal_frontend.recovery.materialize(
            proposal_frontend.collector, trigger=trigger,
        )
        recovery_events.append(recovery.to_dict())
        recovered_keys = recovery.added_keys

    query_plan = build_pdf_query_plan(
        policy,
        proposal_frontend.targets,
        generator=lambda prompt: generate_text_only_response(generator_model, prompt),
    )
    question_kind = (
        question_kind_from_plan(query_plan)
        if config.schema_version == 3 else "attribute"
    )
    required_roles = (
        query_required_roles(query_plan)
        if config.schema_version == 3 else ()
    )
    target_registry = None
    initial_catalog = None
    if config.schema_version == 3:
        if config.verification is None or option_catalog is None:
            raise AssertionError("schema 3 config lacks verifier settings")
        initial_catalog = TreeCatalog.from_collector(
            proposal_frontend.collector, proposal_frontend.image,
        )
        target_registry = TargetInstanceRegistry.from_frontend(
            image=proposal_frontend.image,
            targets=query_plan.targets,
            candidates=initial_catalog.candidate_records(),
            sam_model=sam_model,
            target_instance_iou=config.verification.target_instance_iou,
        )

    budget = _budget_from_config(config)
    state_offset = 0
    phase_traces: list[dict[str, Any]] = []
    pool_snapshots: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = (
        [_global_evidence_record(
            global_observation,
            proposal_frontend.image,
            initial_catalog.root_key,
            global_prefix_elapsed_seconds,
        )]
        if initial_catalog is not None else []
    )
    all_steps = []
    candidate_count = 0

    first = _run_phase(
        phase="recovery" if immediate_recovery else "proposals",
        conditional_losses=conditional_losses,
        pair_scoring=pair_scoring,
        collector=proposal_frontend.collector,
        image=proposal_frontend.image,
        query_plan=query_plan,
        policy=policy,
        config=config,
        budget=budget,
        state_offset=state_offset,
        proposal_keys=proposal_frontend.proposal_keys,
        recovered_keys=recovered_keys,
        generator_model=generator_model,
        verifier_model=verifier_model,
        clip_scorer=clip_scorer,
        generator_checkpoint_sha256=generator_checkpoint_sha256,
        verifier_checkpoint_sha256=verifier_checkpoint_sha256,
        option_catalog=option_catalog,
        target_registry=target_registry,
        shared_records=evidence_records,
        question_kind=question_kind,
        required_roles=required_roles,
        timing_origin=timing_origin,
    )
    phase_traces.append(first["trace"])
    pool_snapshots.append(first["pool"])
    _append_unique_phase_records(
        evidence_records, first["records"], root_key=first["pool"]["root_key"],
    )
    all_steps.extend(first["steps"])
    candidate_count = first["candidate_count"]
    decision = select_accepted_hypothesis(
        evidence_records,
        root_state_id=0,
        acceptance=config.acceptance,
        validation_enabled=config.modules.evidence_validation,
        root_fallback_answer=global_observation.output,
        option_catalog=(option_catalog if config.schema_version == 3 else None),
        question_kind=question_kind,
        required_roles=required_roles,
        negative_coverage=_negative_candidate_coverage(
            evidence_records,
            proposal_frontend.proposal_keys + recovered_keys,
        ),
        negative_coverage_for_records=lambda prefix: _negative_candidate_coverage(
            prefix, proposal_frontend.proposal_keys + recovered_keys,
        ),
    )

    budget = first["budget_after"]
    state_offset = first["next_state_offset"]
    can_recover = (
        not immediate_recovery
        and decision.accepted_hypothesis is None
        and first["trace"]["termination"] != "ACCEPTED_STOP"
        and first["pool_exhausted"]
        and all(value > 0 for value in budget.values())
    )
    if can_recover:
        recovery = proposal_frontend.recovery.materialize(
            proposal_frontend.collector, trigger="pool_exhausted",
        )
        recovery_events.append(recovery.to_dict())
        recovered_keys = tuple(dict.fromkeys(recovered_keys + recovery.added_keys))
        if recovery.added_keys:
            second = _run_phase(
                phase="recovery",
                conditional_losses=conditional_losses,
                pair_scoring=pair_scoring,
                collector=proposal_frontend.collector,
                image=proposal_frontend.image,
                query_plan=query_plan,
                policy=policy,
                config=config,
                budget=budget,
                state_offset=state_offset,
                proposal_keys=proposal_frontend.proposal_keys,
                recovered_keys=recovered_keys,
                generator_model=generator_model,
                verifier_model=verifier_model,
                clip_scorer=clip_scorer,
                generator_checkpoint_sha256=generator_checkpoint_sha256,
                verifier_checkpoint_sha256=verifier_checkpoint_sha256,
                option_catalog=option_catalog,
                target_registry=target_registry,
                shared_records=evidence_records,
                question_kind=question_kind,
                required_roles=required_roles,
                timing_origin=timing_origin,
                previous_state=first["final_state"],
            )
            phase_traces.append(second["trace"])
            pool_snapshots.append(second["pool"])
            _append_unique_phase_records(
                evidence_records,
                second["records"], root_key=second["pool"]["root_key"],
            )
            all_steps.extend(second["steps"])
            candidate_count = second["candidate_count"]

    decision = select_accepted_hypothesis(
        evidence_records,
        root_state_id=0,
        acceptance=config.acceptance,
        validation_enabled=config.modules.evidence_validation,
        root_fallback_answer=global_observation.output,
        option_catalog=(option_catalog if config.schema_version == 3 else None),
        question_kind=question_kind,
        required_roles=required_roles,
        negative_coverage=_negative_candidate_coverage(
            evidence_records,
            proposal_frontend.proposal_keys + recovered_keys,
        ),
        negative_coverage_for_records=lambda prefix: _negative_candidate_coverage(
            prefix, proposal_frontend.proposal_keys + recovered_keys,
        ),
    )
    output = _strict_json(decision.answer, "independent search output")
    changed = Counter(
        item.action.value for item in all_steps if item.status == "changed"
    )
    no_op = Counter(
        item.action.value for item in all_steps if item.status == "no_op"
    )
    trace = {
        "schema_version": config.schema_version,
        "method": config.method,
        "mode": "search",
        "execution_provenance": execution_provenance,
        "global_observation": global_observation.to_dict(),
        "validated_full_image": global_observation.validation_dict(),
        "sam_proposals": proposal_frontend.proposal_trace(),
        "candidate_pool": pool_snapshots,
        "scan_recover": recovery_events,
        "evidence_ledger": decision.ledger_trace,
        "accepted_hypothesis": (
            None
            if decision.accepted_hypothesis is None
            else decision.accepted_hypothesis.to_dict()
        ),
        "controller": {"phases": phase_traces},
        "module_activity": {
            "global_observation": True,
            "sam": True,
            "planner": True,
            "query_ranking": config.modules.query_ranking,
            "adaptive_observation": config.modules.adaptive_observation,
            "evidence_validation": config.modules.evidence_validation,
            "recovery": bool(recovery_events),
            "candidate_count": candidate_count,
            "phase_count": len(phase_traces),
            "state_evaluations": sum(
                phase["assessment_count"] for phase in phase_traces
            ),
            "actions": {
                "changed": dict(sorted(changed.items())),
                "no_op": dict(sorted(no_op.items())),
            },
        },
        "final_decision": {
            "source": decision.source,
            "reason": (
                "accepted_cumulative_local_evidence"
                if decision.accepted_hypothesis is not None
                else "no_accepted_local_hypothesis"
            ),
            "output_sha256": canonical_sha256(output),
        },
    }
    if config.schema_version == 3:
        if target_registry is None or option_catalog is None:
            raise AssertionError("schema 3 search lacks a target registry")
        trace.update({
            "option_catalog": _option_catalog_trace(option_catalog),
            "global_evidence_record": copy.deepcopy(evidence_records[0]),
            "query_plan": query_plan.to_dict(),
            "question_kind": question_kind,
            "required_roles": list(required_roles),
            "target_registry": target_registry.to_dict(),
        })
    return output, _strict_json(trace, "independent search trace")
