"""Optional appearance demand layered over immutable Phase-1 v1 profiles."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .query_profile import infer_query_profile


_APPEARANCE_DESCRIPTOR = re.compile(
    r"\b(?:black|white|red|blue|green|yellow|brown|orange|purple|pink|"
    r"gr[ae]y|silver|gold(?:en)?|beige|cyan|magenta|maroon|navy|teal|"
    r"with\s+(?:a\s+|an\s+|the\s+)?|wearing|holding|carrying)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryProfileV4:
    detail_demand: float
    context_demand: float
    appearance_demand: float

    def to_dict(self) -> dict[str, float]:
        return {
            "detail_demand": self.detail_demand,
            "context_demand": self.context_demand,
            "appearance_demand": self.appearance_demand,
        }


def _weight(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("appearance_descriptor_visual_relief must be in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("appearance_descriptor_visual_relief must be in [0, 1]")
    return result


def infer_query_profile_v4(
    question: Any,
    augmented_queries: Any,
    *,
    appearance_descriptor_visual_relief: Any,
) -> QueryProfileV4:
    """Detect visible target qualifiers without changing relevance demand."""
    base = infer_query_profile(question, augmented_queries)
    relief = _weight(appearance_descriptor_visual_relief)
    appearance = relief if _APPEARANCE_DESCRIPTOR.search(question) else 0.0
    return QueryProfileV4(
        detail_demand=base.detail_demand,
        context_demand=base.context_demand,
        appearance_demand=appearance,
    )


__all__ = ["QueryProfileV4", "infer_query_profile_v4"]
