"""Optional continuous descriptor demand layered over immutable Phase-1 v1."""

from __future__ import annotations

import math
import re
from typing import Any

from .query_profile import QueryProfile, infer_query_profile


_ATTRIBUTE_DESCRIPTOR = re.compile(
    r"\b(?:black|white|red|blue|green|yellow|brown|orange|purple|pink|"
    r"gr[ae]y|silver|gold(?:en)?|beige|cyan|magenta|maroon|navy|teal)\b",
    re.IGNORECASE,
)


def _weight(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("attribute_descriptor_detail_weight must be in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("attribute_descriptor_detail_weight must be in [0, 1]")
    return result


def infer_query_profile_v3(
    question: Any,
    augmented_queries: Any,
    *,
    attribute_descriptor_detail_weight: Any,
) -> QueryProfile:
    """Add partial detail demand for visible attribute descriptors in relations."""
    base = infer_query_profile(question, augmented_queries)
    weight = _weight(attribute_descriptor_detail_weight)
    if base.detail_demand > 0.0 or _ATTRIBUTE_DESCRIPTOR.search(question) is None:
        return base
    return QueryProfile(
        detail_demand=weight,
        context_demand=base.context_demand,
    )


__all__ = ["infer_query_profile_v3"]
