"""Permutation-invariant Vstar admission for a confirmed BACKTRACK recovery."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from cvsearch.eval.phase7_uncertainty_confirmation import _p0, _snapshot_json
from cvsearch.eval.phase12_generated_query_lazy_search import (
    project_lazy_candidate,
)


B8_FROZEN_B5_RULE = "p0.6666666666666666-c0.05-g-0.1"
_B5_DECISION_FIELDS = frozenset({
    "action", "status", "output", "confidence", "confidence_gain",
    "path_consensus",
})


@dataclass(frozen=True)
class VstarConfirmedBacktrackDecision:
    action: str
    status: str
    output: Any


def select_vstar_confirmed_backtrack(
    p0: dict[str, Any], options: Sequence[str], observations: Any,
    b5_decision: Mapping[str, Any],
) -> VstarConfirmedBacktrackDecision:
    """Select only the exact P0, candidate, candidate recovery trajectory."""
    p0, p0_confidence = _p0(p0)
    retained = VstarConfirmedBacktrackDecision(
        "P0", "retained_p0", _snapshot_json(p0["output"]),
    )
    try:
        if type(b5_decision) is not dict or set(b5_decision) != _B5_DECISION_FIELDS:
            return retained
        frozen = _snapshot_json(b5_decision)
        confidence = frozen["confidence"]
        gain = frozen["confidence_gain"]
        if (
            frozen["action"] != "LAZY"
            or frozen["status"] != "selected_generated_query_lazy_search"
            or type(confidence) not in (int, float)
            or not math.isfinite(float(confidence))
            or not 0.05 <= float(confidence) <= 1.0
            or type(gain) not in (int, float)
            or not math.isfinite(float(gain))
            or not -0.1 <= float(gain) <= 1.0
            or frozen["path_consensus"] != 2 / 3
            or type(observations) is not list or len(observations) != 3
        ):
            return retained
        projection = project_lazy_candidate(
            "logits_match", options, observations,
        )
        winners = tuple(projection["view_outputs"])
        candidate = frozen["output"]
        if (
            not projection["feasible"]
            or projection["output"] != candidate
            or not math.isclose(
                float(confidence), float(projection["confidence"]),
                rel_tol=0.0, abs_tol=1e-12,
            )
            or not math.isclose(
                float(gain), float(confidence) - p0_confidence,
                rel_tol=0.0, abs_tol=1e-12,
            )
            or winners[0] != p0["output"]
            or winners[1] != candidate
            or winners[2] != candidate
            or candidate == p0["output"]
        ):
            return retained
    except (TypeError, ValueError):
        return retained
    return VstarConfirmedBacktrackDecision(
        "VSTAR_BACKTRACK", "selected_confirmed_backtrack",
        _snapshot_json(candidate),
    )
