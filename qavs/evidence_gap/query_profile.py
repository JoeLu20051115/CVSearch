"""Answer-free question demands used only to adapt candidate ranking."""

from __future__ import annotations

import re
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


_DETAIL = re.compile(
    r"\b(?:colou?r|shape|material|texture|pattern|words?|text|letters?|numbers?|read|written)\b",
    re.IGNORECASE,
)
_CONTEXT = re.compile(
    r"\b(?:left|right|above|below|behind|front|between|beside|near|next\s+to|relative|side)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryProfile:
    detail_demand: float
    context_demand: float

    def to_dict(self) -> dict[str, float]:
        return {
            "detail_demand": self.detail_demand,
            "context_demand": self.context_demand,
        }


EXPAND_REASONS = frozenset({
    "EXPAND-REL",
    "EXPAND-COUNT",
    "EXPAND-COMP",
    "EXPAND-CONTEXT",
})


def primary_expand_reason(query_plan: Any, gap_scores: Any) -> str:
    """Choose one answer-free, mutually exclusive reason for EXPAND."""
    if isinstance(query_plan, str):
        question_kind = query_plan
    else:
        from .pdf_runtime import question_kind_from_plan

        question_kind = question_kind_from_plan(query_plan)
    if question_kind not in {
        "relation", "count", "comparison", "attribute", "coverage",
    }:
        raise ValueError("question kind is unsupported for EXPAND")
    if isinstance(gap_scores, Mapping):
        expand_gap = gap_scores.get("expand")
    else:
        expand_gap = getattr(gap_scores, "expand", None)
    if (
        isinstance(expand_gap, bool)
        or not isinstance(expand_gap, (int, float))
        or not math.isfinite(float(expand_gap))
        or not 0.0 <= float(expand_gap) <= 1.0
    ):
        raise ValueError("expand gap must be a finite probability")
    return {
        "relation": "EXPAND-REL",
        "count": "EXPAND-COUNT",
        "comparison": "EXPAND-COMP",
        "attribute": "EXPAND-CONTEXT",
        "coverage": "EXPAND-CONTEXT",
    }[question_kind]


def _weight(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def adaptive_alpha(
    base_alpha: Any,
    profile: QueryProfile,
    detail_discount: Any,
    context_gain: Any,
) -> float:
    """Return the bounded query-specific relevance weight."""
    if not isinstance(profile, QueryProfile):
        raise TypeError("profile must be a QueryProfile")
    value = (
        _weight(base_alpha, "base_alpha")
        + _weight(context_gain, "context_gain") * profile.context_demand
        - _weight(detail_discount, "detail_discount") * profile.detail_demand
    )
    return min(1.0, max(0.0, value))


def infer_query_profile(question: Any, augmented_queries: Any) -> QueryProfile:
    """Infer two binary demands without labels, options, answers, or target boxes."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    if not question.strip():
        raise ValueError("question must be nonempty")
    if isinstance(augmented_queries, (str, bytes)) or not isinstance(
        augmented_queries, Sequence
    ):
        raise TypeError("augmented_queries must be a sequence")
    normalized = []
    for query in augmented_queries:
        if not isinstance(query, str):
            raise TypeError("augmented queries must be strings")
        query = " ".join(query.split())
        if not query:
            raise ValueError("augmented queries must be nonempty")
        normalized.append(query.casefold())
    return QueryProfile(
        detail_demand=1.0 if _DETAIL.search(question) else 0.0,
        context_demand=(
            1.0 if _CONTEXT.search(question) or len(set(normalized)) >= 2 else 0.0
        ),
    )
