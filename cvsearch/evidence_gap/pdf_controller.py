"""Pure control primitives for PDF-faithful tree search."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .pdf_types import CandidateDescriptor


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

    def pop_next(self) -> CandidateDescriptor | None:
        remaining = [
            entry for key, entry in self._entries.items()
            if key not in self._popped
        ]
        if not remaining:
            return None
        selected = min(
            remaining,
            key=lambda entry: (
                -entry.rank_score,
                entry.discovery_ordinal,
                entry.candidate.canonical_key,
            ),
        )
        key = selected.candidate.canonical_key
        self._popped.add(key)
        self._popped_scores.append(selected.rank_score)
        return selected.candidate
