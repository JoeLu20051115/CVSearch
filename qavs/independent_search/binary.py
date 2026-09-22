"""Fail-closed evidence gate for binary negative answers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any


@dataclass(frozen=True)
class StrictNoConfig:
    min_coverage: float = 0.80
    min_refute_probability: float = 0.70
    max_yes_support_exclusive: float = 0.70
    min_distinct_refute_views: int = 2

    def __post_init__(self) -> None:
        for name in (
            "min_coverage",
            "min_refute_probability",
            "max_yes_support_exclusive",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be a finite probability")
            object.__setattr__(self, name, float(value))
        if (
            isinstance(self.min_distinct_refute_views, bool)
            or not isinstance(self.min_distinct_refute_views, int)
            or self.min_distinct_refute_views < 2
        ):
            raise ValueError("min_distinct_refute_views must be at least two")


@dataclass(frozen=True)
class NegativeGateDecision:
    accepted: bool
    status: str
    coverage: float
    eligible_state_ids: tuple[int, ...]
    distinct_refute_views: int
    max_yes_support: float
    context_refute_view: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "status": self.status,
            "coverage": self.coverage,
            "eligible_state_ids": list(self.eligible_state_ids),
            "distinct_refute_views": self.distinct_refute_views,
            "max_yes_support": self.max_yes_support,
            "context_refute_view": self.context_refute_view,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class NegativeCoverage:
    """Evidence that all relevant candidates and enough image area were checked."""

    all_relevant_visited: bool
    region_union_ratio: float

    def __post_init__(self) -> None:
        if not isinstance(self.all_relevant_visited, bool):
            raise TypeError("all_relevant_visited must be bool")
        value = self.region_union_ratio
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError("region_union_ratio must be a finite probability")
        object.__setattr__(self, "region_union_ratio", float(value))


def _probabilities(option: Any) -> tuple[str, dict[str, float]] | None:
    if not isinstance(option, Mapping) or option.get("key") != "yes":
        return None
    distribution = option.get("distribution")
    if not isinstance(distribution, Mapping):
        return None
    labels = distribution.get("labels")
    values = distribution.get("probabilities")
    if (
        not isinstance(labels, list)
        or labels != ["Support", "Refute", "Insufficient"]
        or not isinstance(values, list)
        or len(values) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in values
        )
        or distribution.get("winner") not in labels
    ):
        return None
    return distribution["winner"], {
        label: float(value) for label, value in zip(labels, values, strict=True)
    }


def _valid_view(record: Any) -> tuple[int, str, tuple[float, ...], str, dict[str, float]] | None:
    if not isinstance(record, Mapping):
        return None
    state = record.get("state")
    vector = record.get("option_support")
    if (
        not isinstance(state, Mapping)
        or not isinstance(vector, Mapping)
        or vector.get("valid") is not True
    ):
        return None
    state_id = state.get("state_id")
    action = state.get("action")
    geometry = state.get("effective_geometry")
    render_sha256 = state.get("render_sha256")
    if (
        isinstance(state_id, bool)
        or not isinstance(state_id, int)
        or state_id < 0
        or action not in {"GLOBAL", "BASE", "NEXT", "ZOOM", "SPLIT", "EXPAND", "RECOVER"}
        or not isinstance(geometry, list)
        or len(geometry) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in geometry
        )
        or re.fullmatch(r"[0-9a-f]{64}", str(render_sha256)) is None
    ):
        return None
    if action != "GLOBAL":
        grounding = record.get("grounding")
        grounding_record = (
            grounding.get("record") if isinstance(grounding, Mapping) else None
        )
        if not isinstance(grounding_record, Mapping) or grounding_record.get("valid") is not True:
            return None
    options = vector.get("options")
    if not isinstance(options, list):
        return None
    yes = next((_probabilities(option) for option in options if isinstance(option, Mapping) and option.get("key") == "yes"), None)
    if yes is None:
        return None
    winner, probabilities = yes
    return (
        state_id,
        action,
        tuple(float(value) for value in geometry),
        winner,
        probabilities,
    )


def evaluate_negative_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    coverage: NegativeCoverage,
    config: StrictNoConfig,
) -> NegativeGateDecision:
    """Accept No only after covered, grounded, geometrically distinct Refute views."""
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence")
    if not isinstance(config, StrictNoConfig):
        raise TypeError("config must be StrictNoConfig")
    if not isinstance(coverage, NegativeCoverage):
        raise TypeError("coverage must be NegativeCoverage")

    views = [view for record in records if (view := _valid_view(record)) is not None]
    max_yes_support = max(
        (view[4]["Support"] for view in views),
        default=0.0,
    )
    refutes = [
        view for view in views
        if view[3] == "Refute"
        and view[4]["Refute"] >= config.min_refute_probability
    ]
    geometries = {view[2] for view in refutes}
    context_refute = any(view[1] in {"GLOBAL", "EXPAND"} for view in refutes)
    reasons = []
    if not coverage.all_relevant_visited:
        reasons.append("candidate_coverage")
    if coverage.region_union_ratio < config.min_coverage:
        reasons.append("spatial_coverage")
    if len(geometries) < config.min_distinct_refute_views:
        reasons.append("distinct_refute_views")
    if not context_refute:
        reasons.append("context_refute_view")
    if max_yes_support >= config.max_yes_support_exclusive:
        reasons.append("yes_support")
    accepted = not reasons
    return NegativeGateDecision(
        accepted=accepted,
        status="Refute" if accepted else "Insufficient",
        coverage=coverage.region_union_ratio,
        eligible_state_ids=tuple(sorted(view[0] for view in refutes)),
        distinct_refute_views=len(geometries),
        max_yes_support=max_yes_support,
        context_refute_view=context_refute,
        reasons=tuple(reasons),
    )
