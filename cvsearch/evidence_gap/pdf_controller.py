"""Pure control primitives for PDF-faithful tree search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence

from .pdf_types import (
    ActionName,
    CandidateDescriptor,
    EvidenceGapScores,
    PDFSearchConfig,
    SearchStateRecord,
    TerminationType,
)


@dataclass(frozen=True)
class _FrozenQueueEntry:
    candidate: CandidateDescriptor
    rank_score: float
    discovery_ordinal: int


class FrozenRankedQueue:
    """Candidate-preserving queue ordered only by frozen joint-rank scores."""

    def __init__(self) -> None:
        self._entries: dict[str, _FrozenQueueEntry] = {}
        self._popped: set[str] = set()
        self._popped_scores: list[float] = []
        self._sibling_groups = 0
        self._native_first_choice_changes = 0

    @property
    def candidate_count(self) -> int:
        return len(self._entries)

    @property
    def native_first_choice_changes(self) -> int:
        return self._native_first_choice_changes

    @property
    def sibling_groups(self) -> int:
        return self._sibling_groups

    @property
    def popped_scores(self) -> tuple[float, ...]:
        return tuple(self._popped_scores)

    def add_sibling_group(
        self,
        native_candidates: Sequence[CandidateDescriptor],
        ranked_keys: Sequence[str],
        score_by_key: Mapping[str, float],
    ) -> None:
        native = list(native_candidates)
        ranked = list(ranked_keys)
        if not native:
            raise ValueError("sibling group must contain at least one candidate")
        if not all(isinstance(item, CandidateDescriptor) for item in native):
            raise TypeError("native candidates must be CandidateDescriptor values")
        native_keys = [item.canonical_key for item in native]
        if len(native_keys) != len(set(native_keys)):
            raise ValueError("native candidates must not contain duplicates")
        if any(key in self._entries for key in native_keys):
            raise ValueError("candidate is already present in the frozen queue")
        sibling_groups = {item.sibling_group for item in native}
        if len(sibling_groups) != 1:
            raise ValueError("all candidates must belong to one sibling group")
        if len(ranked) != len(set(ranked)) or set(ranked) != set(native_keys):
            raise ValueError("ranked keys must preserve the exact candidate set")
        if not isinstance(score_by_key, Mapping) or set(score_by_key) != set(native_keys):
            raise ValueError("score_by_key must cover the exact candidate set")

        scores: dict[str, float] = {}
        for key in native_keys:
            value = score_by_key[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("joint rank scores must be finite numbers")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("joint rank scores must be finite values in [0, 1]")
            scores[key] = value
        ordered_scores = [scores[key] for key in ranked]
        if any(left < right for left, right in zip(ordered_scores, ordered_scores[1:])):
            raise ValueError("ranked keys must be non-increasing in joint rank score")

        if ranked[0] != native_keys[0]:
            self._native_first_choice_changes += 1
        start = len(self._entries)
        for offset, candidate in enumerate(native):
            self._entries[candidate.canonical_key] = _FrozenQueueEntry(
                candidate=candidate,
                rank_score=scores[candidate.canonical_key],
                discovery_ordinal=start + offset,
            )
        self._sibling_groups += 1

    def ranked_remaining(
        self, excluded_keys: Sequence[str] = (),
    ) -> tuple[CandidateDescriptor, ...]:
        excluded = set(excluded_keys)
        if not all(isinstance(key, str) for key in excluded):
            raise TypeError("excluded keys must be strings")
        remaining = [
            entry for key, entry in self._entries.items()
            if key not in self._popped and key not in excluded
        ]
        remaining.sort(
            key=lambda entry: (
                -entry.rank_score,
                entry.discovery_ordinal,
                entry.candidate.canonical_key,
            ),
        )
        return tuple(entry.candidate for entry in remaining)

    def pop_next(self, excluded_keys: Sequence[str] = ()) -> CandidateDescriptor | None:
        remaining = self.ranked_remaining(excluded_keys)
        if not remaining:
            return None
        candidate = remaining[0]
        selected = self._entries[candidate.canonical_key]
        key = candidate.canonical_key
        self._popped.add(key)
        self._popped_scores.append(selected.rank_score)
        return candidate

    def score(self, key: str) -> float:
        try:
            return self._entries[key].rank_score
        except KeyError as error:
            raise KeyError(f"unknown frozen queue candidate: {key}") from error


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite probability")
    return result


def _cost(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _strict_json(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be strict JSON") from error


@dataclass(frozen=True)
class StateAssessment:
    answer: Any
    uncertainty: float
    gaps: EvidenceGapScores
    support_avg: float
    support_min: float
    verifier_independent: bool
    aggregation_available: bool
    model_calls: int
    processed_pixels: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer", _strict_json(self.answer, "answer"))
        object.__setattr__(self, "uncertainty", _probability(self.uncertainty, "uncertainty"))
        if not isinstance(self.gaps, EvidenceGapScores):
            raise TypeError("gaps must be EvidenceGapScores")
        object.__setattr__(self, "support_avg", _probability(self.support_avg, "support_avg"))
        object.__setattr__(self, "support_min", _probability(self.support_min, "support_min"))
        if type(self.verifier_independent) is not bool:
            raise TypeError("verifier_independent must be a boolean")
        if type(self.aggregation_available) is not bool:
            raise TypeError("aggregation_available must be a boolean")
        _cost(self.model_calls, "model_calls")
        _cost(self.processed_pixels, "processed_pixels")


@dataclass(frozen=True)
class ActionOutcome:
    status: str
    path_keys: tuple[str, ...] | None = None
    focus_keys: tuple[str, ...] | None = None
    context_keys: tuple[str, ...] | None = None
    visited_keys: tuple[str, ...] | None = None
    observation_keys: tuple[str, ...] | None = None
    model_calls: int = 0
    processed_pixels: int = 0
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"changed", "no_op"}:
            raise ValueError("action outcome status must be changed or no_op")
        fields = (
            self.path_keys, self.focus_keys, self.context_keys,
            self.visited_keys, self.observation_keys,
        )
        if self.status == "changed":
            if any(value is None for value in fields) or self.reason is not None:
                raise ValueError("changed action requires complete state fields and no reason")
        else:
            if any(value is not None for value in fields):
                raise ValueError("no-op action cannot contain state fields")
            if not isinstance(self.reason, str) or not self.reason:
                raise ValueError("no-op action requires a reason")
        _cost(self.model_calls, "model_calls")
        _cost(self.processed_pixels, "processed_pixels")

    @classmethod
    def changed(
        cls, *, path_keys: tuple[str, ...], focus_keys: tuple[str, ...],
        context_keys: tuple[str, ...], visited_keys: tuple[str, ...],
        observation_keys: tuple[str, ...], model_calls: int = 0,
        processed_pixels: int = 0,
    ) -> "ActionOutcome":
        return cls(
            "changed", path_keys, focus_keys, context_keys, visited_keys,
            observation_keys, model_calls, processed_pixels,
        )

    @classmethod
    def no_op(cls, reason: str) -> "ActionOutcome":
        return cls("no_op", reason=reason)


@dataclass(frozen=True)
class ControllerStep:
    action: ActionName
    status: str
    before: SearchStateRecord
    after: SearchStateRecord
    reason: str | None = None


@dataclass(frozen=True)
class _HistorySnapshot:
    state: SearchStateRecord
    assessment: StateAssessment


@dataclass(frozen=True)
class ControllerResult:
    termination: TerminationType
    answer: Any
    final_state: SearchStateRecord
    selected_history_state_id: int
    steps: tuple[ControllerStep, ...]
    assessment_count: int


_GAP_ACTIONS = (
    ActionName.ZOOM, ActionName.SPLIT, ActionName.EXPAND, ActionName.NEXT,
)


class PDFTreeController:
    """Run root-to-leaf evidence search with uncertainty and bounded fallback."""

    def __init__(self, config: PDFSearchConfig):
        if not isinstance(config, PDFSearchConfig):
            raise TypeError("config must be PDFSearchConfig")
        self.config = config

    @staticmethod
    def _gap(assessment: StateAssessment, action: ActionName) -> float:
        return getattr(assessment.gaps, action.value.lower())

    def _certified(self, assessment: StateAssessment) -> bool:
        controller = self.config.controller
        return (
            self.config.modules.certified_stop
            and assessment.verifier_independent
            and assessment.aggregation_available
            and assessment.uncertainty <= 1.0 - controller.min_answer_confidence
            and max(assessment.gaps.to_dict().values()) <= controller.max_gap_for_stop
            and assessment.support_avg >= controller.min_support_avg
            and assessment.support_min >= controller.min_support_min
        )

    @staticmethod
    def _quality(snapshot: _HistorySnapshot) -> tuple[float, float, float, float, int]:
        assessment = snapshot.assessment
        return (
            assessment.support_min,
            assessment.support_avg,
            -assessment.uncertainty,
            -max(assessment.gaps.to_dict().values()),
            -snapshot.state.state_id,
        )

    @staticmethod
    def _progress(previous: StateAssessment, current: StateAssessment) -> float:
        return (
            (previous.uncertainty - current.uncertainty)
            + (current.support_min - previous.support_min)
            + (current.support_avg - previous.support_avg)
        ) / 3.0

    def run(
        self,
        *,
        root_key: str,
        assess: Callable[[SearchStateRecord], StateAssessment],
        feasible: Callable[[SearchStateRecord], Mapping[ActionName, bool]],
        execute: Callable[[ActionName, SearchStateRecord], ActionOutcome],
        branch_available: Callable[[SearchStateRecord], bool],
    ) -> ControllerResult:
        if not isinstance(root_key, str) or not root_key:
            raise ValueError("root_key must be nonempty")
        for callback, name in (
            (assess, "assess"), (feasible, "feasible"),
            (execute, "execute"), (branch_available, "branch_available"),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")

        budget = self.config.budget
        state = SearchStateRecord(
            state_id=0,
            focus_keys=(root_key,),
            path_keys=(root_key,),
            context_keys=(),
            visited_keys=(root_key,),
            observation_keys=(f"{root_key}@root",),
            remaining_steps=budget.max_steps,
            remaining_model_calls=budget.max_model_calls,
            remaining_pixels=budget.max_processed_pixels,
        )
        history: list[_HistorySnapshot] = []
        steps: list[ControllerStep] = []
        failed_signatures: set[tuple[tuple[str, ...], ActionName]] = set()
        used_backtrack_states: set[int] = set()
        previous: StateAssessment | None = None
        stall_count = 0
        next_state_id = 1

        def force_return(current: SearchStateRecord) -> ControllerResult:
            selected = max(history, key=self._quality)
            return ControllerResult(
                termination=TerminationType.FORCED_RETURN,
                answer=_strict_json(selected.assessment.answer, "final answer"),
                final_state=current,
                selected_history_state_id=selected.state.state_id,
                steps=tuple(steps),
                assessment_count=len(history),
            )

        def backtrack(current: SearchStateRecord) -> SearchStateRecord | None:
            nonlocal next_state_id, previous, stall_count
            candidates = [
                item for item in history[:-1]
                if item.state.state_id not in used_backtrack_states
                and branch_available(item.state)
            ]
            if not candidates or current.remaining_steps == 0:
                return None
            selected = max(candidates, key=self._quality)
            used_backtrack_states.add(selected.state.state_id)
            restored = SearchStateRecord(
                state_id=next_state_id,
                focus_keys=selected.state.focus_keys,
                path_keys=selected.state.path_keys,
                context_keys=selected.state.context_keys,
                visited_keys=tuple(dict.fromkeys(current.visited_keys + selected.state.visited_keys)),
                observation_keys=selected.state.observation_keys,
                remaining_steps=current.remaining_steps - 1,
                remaining_model_calls=current.remaining_model_calls,
                remaining_pixels=current.remaining_pixels,
            )
            next_state_id += 1
            steps.append(ControllerStep(ActionName.BACKTRACK, "changed", current, restored))
            previous = None
            stall_count = 0
            return restored

        while True:
            current_assessment = assess(state)
            if not isinstance(current_assessment, StateAssessment):
                raise TypeError("assess must return StateAssessment")
            if (
                current_assessment.model_calls > state.remaining_model_calls
                or current_assessment.processed_pixels > state.remaining_pixels
            ):
                raise ValueError("state assessment exceeded remaining budget")
            state = SearchStateRecord(
                state_id=state.state_id,
                focus_keys=state.focus_keys,
                path_keys=state.path_keys,
                context_keys=state.context_keys,
                visited_keys=state.visited_keys,
                observation_keys=state.observation_keys,
                remaining_steps=state.remaining_steps,
                remaining_model_calls=state.remaining_model_calls - current_assessment.model_calls,
                remaining_pixels=state.remaining_pixels - current_assessment.processed_pixels,
            )
            history.append(_HistorySnapshot(state, current_assessment))

            if (
                state.remaining_steps == 0
                or state.remaining_model_calls == 0
                or state.remaining_pixels == 0
            ):
                return force_return(state)
            if self._certified(current_assessment):
                return ControllerResult(
                    termination=TerminationType.CERTIFIED_STOP,
                    answer=_strict_json(current_assessment.answer, "final answer"),
                    final_state=state,
                    selected_history_state_id=state.state_id,
                    steps=tuple(steps),
                    assessment_count=len(history),
                )

            if previous is not None:
                if self._progress(previous, current_assessment) < self.config.controller.min_progress:
                    stall_count += 1
                else:
                    stall_count = 0
            if stall_count >= self.config.controller.stall_patience:
                restored = backtrack(state) if self.config.modules.backtracking else None
                if restored is not None:
                    state = restored
                    continue

            raw_feasible = feasible(state)
            if not isinstance(raw_feasible, Mapping) or set(raw_feasible) != set(_GAP_ACTIONS):
                raise ValueError("feasible must return exactly the four evidence-gap actions")
            if any(type(value) is not bool for value in raw_feasible.values()):
                raise TypeError("feasibility values must be booleans")
            enabled = {
                ActionName.ZOOM: self.config.modules.zoom,
                ActionName.SPLIT: self.config.modules.split,
                ActionName.EXPAND: self.config.modules.expand,
                ActionName.NEXT: self.config.modules.next,
            }
            signature = state.observation_keys
            ordered = sorted(
                _GAP_ACTIONS,
                key=lambda action: (-self._gap(current_assessment, action), _GAP_ACTIONS.index(action)),
            )
            changed = False
            for action in ordered:
                if (
                    not enabled[action] or not raw_feasible[action]
                    or (signature, action) in failed_signatures
                ):
                    continue
                outcome = execute(action, state)
                if not isinstance(outcome, ActionOutcome):
                    raise TypeError("execute must return ActionOutcome")
                if outcome.status == "no_op":
                    failed_signatures.add((signature, action))
                    steps.append(ControllerStep(action, "no_op", state, state, outcome.reason))
                    continue
                assert outcome.path_keys is not None
                assert outcome.focus_keys is not None
                assert outcome.context_keys is not None
                assert outcome.visited_keys is not None
                assert outcome.observation_keys is not None
                if (
                    outcome.model_calls > state.remaining_model_calls
                    or outcome.processed_pixels > state.remaining_pixels
                ):
                    raise ValueError("action exceeded remaining budget")
                if (
                    outcome.path_keys == state.path_keys
                    and outcome.context_keys == state.context_keys
                    and outcome.observation_keys == state.observation_keys
                ):
                    raise ValueError("changed action did not change path, context, or observation")
                after = SearchStateRecord(
                    state_id=next_state_id,
                    focus_keys=outcome.focus_keys,
                    path_keys=outcome.path_keys,
                    context_keys=outcome.context_keys,
                    visited_keys=outcome.visited_keys,
                    observation_keys=outcome.observation_keys,
                    remaining_steps=state.remaining_steps - 1,
                    remaining_model_calls=state.remaining_model_calls - outcome.model_calls,
                    remaining_pixels=state.remaining_pixels - outcome.processed_pixels,
                )
                next_state_id += 1
                steps.append(ControllerStep(action, "changed", state, after))
                previous = current_assessment
                state = after
                changed = True
                break
            if changed:
                continue

            restored = backtrack(state) if self.config.modules.backtracking else None
            if restored is not None:
                state = restored
                continue
            return force_return(state)
