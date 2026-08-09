"""Pure state contracts for query-aware evidence-gap search."""

from dataclasses import dataclass, field, is_dataclass
import math
from numbers import Integral, Real
from typing import Any


ZOOM = "ZOOM"
SPLIT = "SPLIT"
EXPAND = "EXPAND"
NEXT = "NEXT"
BACKTRACK = "BACKTRACK"
CERTIFIED_STOP = "CERTIFIED_STOP"
FORCED_RETURN = "FORCED_RETURN"
ACTION_VALUES = (ZOOM, SPLIT, EXPAND, NEXT, BACKTRACK, CERTIFIED_STOP, FORCED_RETURN)


class BudgetExceeded(RuntimeError):
    pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if type(value).__module__ == "numpy" and type(value).__name__ == "bool":
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON values must be finite")
        return number
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if is_dataclass(value):
        return _json_safe(value.to_dict())
    raise TypeError(f"not JSON-safe: {type(value).__name__}")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _integral(value: Any, name: str) -> int:
    if isinstance(value, bool) or type(value).__name__ == "bool":
        raise TypeError(f"{name} must be a finite integer")
    if isinstance(value, Integral):
        integer = int(value)
    else:
        number = _finite_number(value, name)
        if not number.is_integer():
            raise ValueError(f"{name} must be an integer")
        integer = int(number)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def canonical_key(bbox: Any, depth: int, render_level: int) -> str:
    try:
        coordinates = tuple(bbox)
    except TypeError as error:
        raise TypeError("bbox must contain four numeric coordinates") from error
    if len(coordinates) != 4:
        raise ValueError("bbox must contain exactly four coordinates")
    rounded = tuple(int(round(_finite_number(value, "bbox coordinate"))) for value in coordinates)
    canonical_depth = _integral(depth, "depth")
    canonical_render_level = _integral(render_level, "render_level")
    return f"{rounded[0]}:{rounded[1]}:{rounded[2]}:{rounded[3]}:d{canonical_depth}:r{canonical_render_level}"


@dataclass
class QueryPlan:
    main_query: str = ""
    targets: tuple[str, ...] = ()
    augmented_queries: tuple[str, ...] = ()
    evidence_items: tuple[Any, ...] = ()
    global_scope_required: bool = False
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "main_query": _json_safe(self.main_query),
            "targets": _json_safe(self.targets),
            "augmented_queries": _json_safe(self.augmented_queries),
            "evidence_items": _json_safe(self.evidence_items),
            "global_scope_required": _json_safe(self.global_scope_required),
            "fallback_used": _json_safe(self.fallback_used),
        }


@dataclass(frozen=True)
class CandidateScore:
    main: float = 0.0
    augmented: float = 0.0
    complexity: float = 0.0
    edge_density: float = 0.0
    relevance: float = 0.0
    visual: float = 0.0
    rank: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "main": _json_safe(self.main),
            "augmented": _json_safe(self.augmented),
            "complexity": _json_safe(self.complexity),
            "edge_density": _json_safe(self.edge_density),
            "relevance": _json_safe(self.relevance),
            "visual": _json_safe(self.visual),
            "rank": _json_safe(self.rank),
        }


@dataclass
class SearchCandidate:
    key: str = ""
    bbox: tuple[int | float, int | float, int | float, int | float] = (0, 0, 0, 0)
    parent_key: str | None = None
    child_keys: tuple[str, ...] = ()
    depth: int = 0
    source: str = ""
    render_level: int = 0
    score: CandidateScore = field(default_factory=CandidateScore)
    node: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": _json_safe(self.key),
            "bbox": _json_safe(self.bbox),
            "parent_key": _json_safe(self.parent_key),
            "child_keys": _json_safe(self.child_keys),
            "depth": _json_safe(self.depth),
            "source": _json_safe(self.source),
            "render_level": _json_safe(self.render_level),
            "score": self.score.to_dict(),
        }


@dataclass
class BudgetLedger:
    max_mllm_calls: int
    max_processed_pixels: int | float
    mllm_calls: int = 0
    processed_pixels: int | float = 0

    def __post_init__(self) -> None:
        self.max_mllm_calls = self._count(self.max_mllm_calls, "max_mllm_calls")
        self.mllm_calls = self._count(self.mllm_calls, "mllm_calls")
        self.max_processed_pixels = self._nonnegative(self.max_processed_pixels, "max_processed_pixels")
        self.processed_pixels = self._nonnegative(self.processed_pixels, "processed_pixels")
        if self.mllm_calls > self.max_mllm_calls:
            raise BudgetExceeded("mllm_calls already exceeds max_mllm_calls")
        if self.processed_pixels > self.max_processed_pixels:
            raise BudgetExceeded("processed_pixels already exceeds max_processed_pixels")

    @staticmethod
    def _nonnegative(value: int | float, name: str) -> int | float:
        number = _finite_number(value, name)
        if number < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @classmethod
    def _count(cls, value: int | float, name: str) -> int:
        number = cls._nonnegative(value, name)
        if not float(number).is_integer():
            raise ValueError(f"{name} must be an integer count")
        return int(number)

    def consume(self, kind: str, amount: int | float) -> None:
        if kind == "mllm_calls":
            increment = self._count(amount, kind)
            current = self.mllm_calls
            limit = self.max_mllm_calls
        elif kind == "processed_pixels":
            increment = self._nonnegative(amount, kind)
            current = self.processed_pixels
            limit = self.max_processed_pixels
        else:
            raise ValueError(f"unknown budget kind: {kind}")
        if current + increment > limit:
            raise BudgetExceeded(f"{kind}: {current}+{increment}>{limit}")
        if kind == "mllm_calls":
            self.mllm_calls = int(current + increment)
        else:
            self.processed_pixels = current + increment

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_mllm_calls": _json_safe(self.max_mllm_calls),
            "max_processed_pixels": _json_safe(self.max_processed_pixels),
            "mllm_calls": _json_safe(self.mllm_calls),
            "processed_pixels": _json_safe(self.processed_pixels),
        }


@dataclass
class AnswerRecord:
    output: Any = None
    canonical_answer: Any = None
    raw_outputs: tuple[Any, ...] = ()
    groups: dict[str, Any] = field(default_factory=dict)
    frequency: float = 0.0
    margin: float = 0.0
    confidence: float = 0.0
    uncertainty: float = 1.0
    losses: tuple[float, ...] = ()
    selected_from: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": _json_safe(self.output),
            "canonical_answer": _json_safe(self.canonical_answer),
            "raw_outputs": _json_safe(self.raw_outputs),
            "groups": _json_safe(self.groups),
            "frequency": _json_safe(self.frequency),
            "margin": _json_safe(self.margin),
            "confidence": _json_safe(self.confidence),
            "uncertainty": _json_safe(self.uncertainty),
            "losses": _json_safe(self.losses),
            "selected_from": _json_safe(self.selected_from),
        }


@dataclass
class HistoryRecord:
    step: int = 0
    answer: AnswerRecord = field(default_factory=AnswerRecord)
    support_avg: float = 0.0
    support_min: float = 0.0
    cost: int | float = 0
    has_unvisited_branch: bool = False
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": _json_safe(self.step),
            "answer": self.answer.to_dict(),
            "support_avg": _json_safe(self.support_avg),
            "support_min": _json_safe(self.support_min),
            "cost": _json_safe(self.cost),
            "has_unvisited_branch": _json_safe(self.has_unvisited_branch),
            "state": _json_safe(self.state),
        }


@dataclass
class StepTrace:
    step: int = 0
    action: str = ""
    gap_fallback_used: bool = False
    elapsed_seconds: float = 0.0
    focus_key: str | None = None
    feasible_actions: tuple[str, ...] = ()
    gaps: dict[str, float] = field(default_factory=dict)
    no_op_reason: str | None = None
    answer: AnswerRecord | None = None
    support_avg: float = 0.0
    support_min: float = 0.0
    certified: bool = False
    budget: BudgetLedger | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": _json_safe(self.step),
            "action": _json_safe(self.action),
            "gap_fallback_used": _json_safe(self.gap_fallback_used),
            "elapsed_seconds": _json_safe(self.elapsed_seconds),
            "focus_key": _json_safe(self.focus_key),
            "feasible_actions": _json_safe(self.feasible_actions),
            "gaps": _json_safe(self.gaps),
            "no_op_reason": _json_safe(self.no_op_reason),
            "answer": None if self.answer is None else self.answer.to_dict(),
            "support_avg": _json_safe(self.support_avg),
            "support_min": _json_safe(self.support_min),
            "certified": _json_safe(self.certified),
            "budget": None if self.budget is None else self.budget.to_dict(),
        }


@dataclass
class MethodTrace:
    query_plan: QueryPlan | None = None
    candidate_ranks: list[Any] = field(default_factory=list)
    steps: list[StepTrace] = field(default_factory=list)
    history: list[HistoryRecord] = field(default_factory=list)
    final_answer: AnswerRecord | None = None
    budget: BudgetLedger | None = None
    elapsed_seconds: float = 0.0
    termination: str | None = None
    final_boxes: tuple[tuple[int | float, int | float, int | float, int | float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_plan": None if self.query_plan is None else self.query_plan.to_dict(),
            "candidate_ranks": _json_safe(self.candidate_ranks),
            "steps": _json_safe(self.steps),
            "history": _json_safe(self.history),
            "final_answer": None if self.final_answer is None else self.final_answer.to_dict(),
            "budget": None if self.budget is None else self.budget.to_dict(),
            "elapsed_seconds": _json_safe(self.elapsed_seconds),
            "termination": _json_safe(self.termination),
            "final_boxes": _json_safe(self.final_boxes),
        }
