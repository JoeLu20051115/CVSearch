"""Generated-query ranking with lazy split and disagreement backtracking."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Sequence

from cvsearch.evidence_gap.answers import (
    aggregate_hr_answers,
    aggregate_vstar_losses,
)
from cvsearch.eval.phase7_uncertainty_confirmation import (
    _confidence,
    _exact_snapshot,
    _p0,
    _sha256,
    _snapshot_json,
)
from cvsearch.eval.phase9_dense_evidence_search import _percentiles


LAZY_GRID = 2
LAZY_OVERLAP_FRACTION = 0.125
LAZY_QUERY_BETA = 0.5
LAZY_RELEVANCE_WEIGHT = 0.70
LAZY_VISUAL_FEATURE_WEIGHT = 0.5
LAZY_MIN_PATH_CONSENSUS = frozenset({2 / 3, 1.0})
LAZY_MIN_CONFIDENCES = frozenset({0.5, 0.75})
LAZY_MIN_GAINS = frozenset({0.0, 0.1, 0.25})
_LOC_LINE = re.compile(r"^\s*(?:(?:[-*]|\d+[.)])\s*)?LOC\s*:\s*(.+?)\s*$", re.I)
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?|\d+", re.UNICODE)
_LOCALIZATION_GENERIC_WORDS = frozenset({
    "area", "evidence", "image", "location", "object", "region", "visual",
})
_QUESTION_FUNCTION_WORDS = frozenset({
    "a", "an", "are", "do", "does", "how", "is", "of", "the", "what",
    "when", "where", "which", "who", "why",
})
_CANDIDATE_FIELDS = frozenset({
    "action", "feasible", "output", "stability", "path_consensus",
    "view_sha256", "rank_sha256", "query_sha256",
})


@dataclass(frozen=True)
class AdaptivePatch:
    path: tuple[int, ...]
    box: tuple[int, int, int, int]

    @property
    def identity(self) -> str:
        return "root" if not self.path else "p" + ".".join(map(str, self.path))


@dataclass(frozen=True)
class RankedAdaptivePatch:
    patch: AdaptivePatch
    relevance: float
    edge_density: float
    variance: float
    relevance_percentile: float
    edge_percentile: float
    variance_percentile: float
    visual_information: float
    score: float


@dataclass(frozen=True)
class LazyDecision:
    action: str
    status: str
    output: Any
    confidence: float | None
    confidence_gain: float | None
    path_consensus: float | None


def parse_localization_queries(raw: str) -> tuple[str, ...]:
    if not isinstance(raw, str):
        raise TypeError("localization response must be text")
    result = []
    seen = set()
    for line in raw.splitlines():
        match = _LOC_LINE.fullmatch(line)
        if match is None:
            continue
        phrase = " ".join(match.group(1).split())
        key = phrase.casefold()
        if phrase and key not in seen:
            seen.add(key)
            result.append(phrase)
    if not result:
        raise ValueError("localization response contains no LOC phrases")
    return tuple(result[:4])


def sanitize_localization_queries(
    question: str, queries: Sequence[str],
) -> tuple[str, ...]:
    """Remove every generated semantic token not licensed by the original q0."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("localization sanitizer question must be nonempty")
    if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
        raise TypeError("localization sanitizer queries must be a sequence")
    question_tokens = _WORD.findall(question)
    allowed = {token.casefold() for token in question_tokens}
    result = []
    seen = set()
    for query in queries:
        if not isinstance(query, str):
            raise TypeError("localization query must be text")
        tokens = _WORD.findall(query)
        retained = [
            token for token in tokens
            if token.casefold() in allowed
            or token.casefold() in _LOCALIZATION_GENERIC_WORDS
        ]
        has_question_content = any(
            token.casefold() in allowed
            and token.casefold() not in _QUESTION_FUNCTION_WORDS
            for token in retained
        )
        cleaned = " ".join(retained)
        key = cleaned.casefold()
        if cleaned and has_question_content and key not in seen:
            seen.add(key)
            result.append(cleaned)
    if not result:
        fallback = [
            token for token in question_tokens
            if token.casefold() not in _QUESTION_FUNCTION_WORDS
        ]
        if not fallback:
            fallback = question_tokens
        cleaned = " ".join(fallback)
        if not cleaned:
            raise ValueError("localization sanitizer produced no q0-licensed phrase")
        result.append(cleaned)
    return tuple(result[:4])


def _finite_sequence(values: Any, count: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"{name} must align with every patch")
    result = []
    for value in values:
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must contain finite values")
        result.append(float(value))
    return tuple(result)


def fuse_query_relevance(
    main_scores: Any, augmented_scores: Any, *, beta: float = LAZY_QUERY_BETA,
) -> list[float]:
    if (
        isinstance(beta, bool) or not isinstance(beta, (int, float))
        or not math.isfinite(float(beta)) or not 0 <= float(beta) <= 1
    ):
        raise ValueError("query beta must be finite in [0, 1]")
    if not isinstance(main_scores, (list, tuple)) or not main_scores:
        raise ValueError("main query scores must be nonempty")
    main = _finite_sequence(main_scores, len(main_scores), "main query scores")
    if not isinstance(augmented_scores, (list, tuple)) or len(augmented_scores) != len(main):
        raise ValueError("augmentation scores must align with every patch")
    result = []
    for index, row in enumerate(augmented_scores):
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError(f"augmentation scores {index} must be nonempty")
        values = _finite_sequence(row, len(row), f"augmentation scores {index}")
        top = sorted(values, reverse=True)[:3]
        aug = math.fsum(top) / len(top)
        result.append(float(beta) * main[index] + (1 - float(beta)) * aug)
    return result


def generate_lazy_children(parent: AdaptivePatch) -> tuple[AdaptivePatch, ...]:
    if type(parent) is not AdaptivePatch:
        raise TypeError("lazy parent must be an exact AdaptivePatch")
    x0, y0, x1, y1 = parent.box
    if not (0 <= x0 < x1 and 0 <= y0 < y1):
        raise ValueError("lazy parent box is invalid")
    width, height = x1 - x0, y1 - y0
    children = []
    for row in range(LAZY_GRID):
        base_y0 = y0 + math.floor(row * height / LAZY_GRID)
        base_y1 = y0 + math.ceil((row + 1) * height / LAZY_GRID)
        pad_y = (base_y1 - base_y0) * LAZY_OVERLAP_FRACTION
        child_y0 = max(y0, math.floor(base_y0 - pad_y))
        child_y1 = min(y1, math.ceil(base_y1 + pad_y))
        for col in range(LAZY_GRID):
            base_x0 = x0 + math.floor(col * width / LAZY_GRID)
            base_x1 = x0 + math.ceil((col + 1) * width / LAZY_GRID)
            pad_x = (base_x1 - base_x0) * LAZY_OVERLAP_FRACTION
            child_x0 = max(x0, math.floor(base_x0 - pad_x))
            child_x1 = min(x1, math.ceil(base_x1 + pad_x))
            index = row * LAZY_GRID + col
            children.append(AdaptivePatch(
                parent.path + (index,),
                (child_x0, child_y0, child_x1, child_y1),
            ))
    if len({child.box for child in children}) != LAZY_GRID ** 2:
        raise ValueError("lazy split produced duplicate patches")
    return tuple(children)


def rank_lazy_siblings(
    patches: Sequence[AdaptivePatch], relevance: Any, edge_density: Any,
    variance: Any,
) -> tuple[RankedAdaptivePatch, ...]:
    if not isinstance(patches, (list, tuple)) or not patches:
        raise ValueError("lazy ranking requires nonempty siblings")
    patches = tuple(patches)
    if not all(type(patch) is AdaptivePatch for patch in patches):
        raise TypeError("lazy ranking patches must be exact AdaptivePatch values")
    count = len(patches)
    relevance = _finite_sequence(relevance, count, "lazy relevance")
    edge_density = _finite_sequence(edge_density, count, "lazy edge density")
    variance = _finite_sequence(variance, count, "lazy variance")
    rel_pct = _percentiles(relevance)
    edge_pct = _percentiles(edge_density)
    var_pct = _percentiles(variance)
    ranked = []
    for index, patch in enumerate(patches):
        visual = (
            LAZY_VISUAL_FEATURE_WEIGHT * var_pct[index]
            + (1 - LAZY_VISUAL_FEATURE_WEIGHT) * edge_pct[index]
        )
        score = (
            LAZY_RELEVANCE_WEIGHT * rel_pct[index]
            + (1 - LAZY_RELEVANCE_WEIGHT) * visual
        )
        ranked.append(RankedAdaptivePatch(
            patch, relevance[index], edge_density[index], variance[index],
            rel_pct[index], edge_pct[index], var_pct[index], visual, score,
        ))
    ranked.sort(key=lambda item: (
        -item.score, -item.relevance_percentile, item.patch.path,
    ))
    return tuple(ranked)


def choose_backtrack_patch(
    ranked_roots: Sequence[AdaptivePatch], focus: AdaptivePatch, *,
    visited: set[str] | frozenset[str],
) -> AdaptivePatch:
    if type(focus) is not AdaptivePatch or not focus.path:
        raise ValueError("backtrack focus must belong to a root branch")
    if not isinstance(ranked_roots, (list, tuple)) or not isinstance(visited, (set, frozenset)):
        raise TypeError("backtrack roots and visited identities are invalid")
    for patch in ranked_roots:
        if (
            type(patch) is AdaptivePatch and len(patch.path) == 1
            and patch.path[0] != focus.path[0] and patch.identity not in visited
        ):
            return patch
    raise ValueError("no unvisited root branch is available for backtrack")


def lazy_view_record(answer_type: str, options: Sequence[str], observation: Any):
    if answer_type == "logits_match":
        if not isinstance(observation, Mapping) or set(observation) != {"winner", "losses"}:
            raise ValueError("lazy V* observation has an invalid exact schema")
        winner, losses = observation["winner"], observation["losses"]
        if (
            type(winner) is not int or not 0 <= winner < len(options)
            or type(losses) is not list or len(losses) != len(options)
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) for value in losses
            )
        ):
            raise ValueError("lazy V* observation has invalid dimensions")
        finite = [float(value) for value in losses]
        if winner != min(range(len(finite)), key=finite.__getitem__):
            raise ValueError("lazy V* winner disagrees with losses")
        return aggregate_vstar_losses([finite])
    if answer_type == "option_list":
        if (
            type(observation) is not list or len(observation) != 4
            or not all(isinstance(value, str) for value in observation)
        ):
            raise ValueError("lazy HR observation requires four strings")
        return aggregate_hr_answers(list(options), observation)
    raise ValueError("lazy answer type is unsupported")


def project_lazy_candidate(
    answer_type: str, options: Sequence[str], observations: Any,
) -> dict[str, Any]:
    if type(observations) is not list or len(observations) not in (2, 3):
        raise ValueError("lazy projection requires two or three views")
    records = [lazy_view_record(answer_type, options, item) for item in observations]
    canonical = [record.canonical_answer for record in records]
    counts = Counter(value for value in canonical if value is not None)
    winner, count = (None, 0) if not counts else min(
        counts.items(), key=lambda item: (-item[1], str(item[0])),
    )
    required = 2
    majority = [
        record for record in records
        if winner is not None and record.canonical_answer == winner
        and record.aggregation_available is not False
    ]
    feasible = count >= required and len(majority) == count
    path_consensus = count / len(records) if feasible else 0.0
    confidence = min(
        [path_consensus] + [float(record.confidence) for record in majority],
    ) if feasible else 0.0
    return {
        "feasible": feasible,
        "output": _snapshot_json(majority[0].output) if feasible else None,
        "confidence": confidence,
        "canonical_answer": _snapshot_json(winner),
        "path_consensus": path_consensus,
        "view_outputs": _snapshot_json(canonical),
        "aggregation_available": all(
            record.aggregation_available is not False for record in records
        ),
    }


def lazy_rule_key(
    min_path_consensus: float, min_confidence: float, min_gain: float,
) -> str:
    if (
        type(min_path_consensus) not in (int, float)
        or float(min_path_consensus) not in LAZY_MIN_PATH_CONSENSUS
        or type(min_confidence) not in (int, float)
        or float(min_confidence) not in LAZY_MIN_CONFIDENCES
        or type(min_gain) not in (int, float)
        or float(min_gain) not in LAZY_MIN_GAINS
    ):
        raise ValueError("lazy rule key must use the frozen grid")
    return f"p{float(min_path_consensus)}-c{float(min_confidence)}-g{float(min_gain)}"


def all_lazy_rule_keys() -> tuple[str, ...]:
    return tuple(
        lazy_rule_key(path, confidence, gain)
        for path in sorted(LAZY_MIN_PATH_CONSENSUS)
        for confidence in sorted(LAZY_MIN_CONFIDENCES)
        for gain in sorted(LAZY_MIN_GAINS)
    )


def _candidate(value: Any) -> tuple[dict[str, Any], float | None]:
    candidate = _exact_snapshot(value, _CANDIDATE_FIELDS, "lazy candidate")
    if candidate["action"] != "LAZY" or type(candidate["feasible"]) is not bool:
        raise ValueError("lazy candidate action or feasibility is invalid")
    _sha256(candidate["rank_sha256"], "lazy rank hash")
    _sha256(candidate["query_sha256"], "lazy query hash")
    hashes = candidate["view_sha256"]
    if type(hashes) is not list or len(hashes) > 3:
        raise ValueError("lazy candidate view hashes are invalid")
    for index, value in enumerate(hashes):
        _sha256(value, f"lazy view hash {index}")
    if len(set(hashes)) != len(hashes):
        raise ValueError("lazy view hashes must be distinct")
    if not candidate["feasible"]:
        if any(candidate[field] is not None for field in ("output", "stability", "path_consensus")):
            raise ValueError("infeasible lazy candidate exposes a projection")
        return candidate, None
    if len(hashes) not in (2, 3):
        raise ValueError("feasible lazy candidate must bind two or three views")
    if candidate["output"] is None:
        raise ValueError("feasible lazy candidate lacks an output")
    confidence = _confidence(candidate["stability"], "lazy stability")
    if candidate["path_consensus"] not in LAZY_MIN_PATH_CONSENSUS:
        raise ValueError("lazy candidate has invalid path consensus")
    return candidate, confidence


def select_lazy_candidate(
    p0: dict[str, Any], candidate: dict[str, Any], *,
    min_path_consensus: float, min_confidence: float, min_gain: float,
) -> LazyDecision:
    lazy_rule_key(min_path_consensus, min_confidence, min_gain)
    p0, p0_confidence = _p0(p0)
    candidate, candidate_confidence = _candidate(candidate)
    retained = LazyDecision(
        "P0", "retained_p0", _snapshot_json(p0["output"]), None, None, None,
    )
    if not candidate["feasible"] or candidate["output"] == p0["output"]:
        return retained
    if candidate_confidence is None:
        raise AssertionError("feasible lazy candidate lost confidence")
    gain = Decimal(str(candidate_confidence)) - Decimal(str(p0_confidence))
    if (
        Decimal(str(candidate["path_consensus"])) < Decimal(str(min_path_consensus))
        or Decimal(str(candidate_confidence)) < Decimal(str(min_confidence))
        or gain < Decimal(str(min_gain))
    ):
        return retained
    return LazyDecision(
        "LAZY", "selected_generated_query_lazy_search",
        _snapshot_json(candidate["output"]), candidate_confidence, float(gain),
        float(candidate["path_consensus"]),
    )
