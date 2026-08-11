"""Strict immutable types for the PDF-faithful evidence-gap controller."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from typing import Any


class ActionName(str, Enum):
    ZOOM = "ZOOM"
    SPLIT = "SPLIT"
    EXPAND = "EXPAND"
    NEXT = "NEXT"
    BACKTRACK = "BACKTRACK"


class TerminationType(str, Enum):
    CERTIFIED_STOP = "CERTIFIED_STOP"
    FORCED_RETURN = "FORCED_RETURN"


def _exact_keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != expected:
        raise ValueError(f"{name} must contain exact keys {sorted(expected)}")
    return value


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True)
class RankingConfig:
    alpha: float
    beta: float
    visual_lambda: float
    top_k_augmented: int
    main_query: bool
    augmented_query: bool
    complexity: bool
    edge_density: bool

    @classmethod
    def from_mapping(cls, value: Any) -> "RankingConfig":
        data = _exact_keys(value, {
            "alpha", "beta", "visual_lambda", "top_k_augmented",
            "main_query", "augmented_query", "complexity", "edge_density",
        }, "ranking")
        return cls(
            alpha=_probability(data["alpha"], "alpha"),
            beta=_probability(data["beta"], "beta"),
            visual_lambda=_probability(data["visual_lambda"], "visual_lambda"),
            top_k_augmented=_positive_integer(data["top_k_augmented"], "top_k_augmented"),
            main_query=_boolean(data["main_query"], "main_query"),
            augmented_query=_boolean(data["augmented_query"], "augmented_query"),
            complexity=_boolean(data["complexity"], "complexity"),
            edge_density=_boolean(data["edge_density"], "edge_density"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModuleConfig:
    structured_planner: bool
    zoom: bool
    split: bool
    expand: bool
    next: bool
    reanswer: bool
    independent_verifier: bool
    backtracking: bool
    certified_stop: bool

    @classmethod
    def from_mapping(cls, value: Any) -> "ModuleConfig":
        names = {
            "structured_planner", "zoom", "split", "expand", "next",
            "reanswer", "independent_verifier", "backtracking", "certified_stop",
        }
        data = _exact_keys(value, names, "modules")
        return cls(**{name: _boolean(data[name], name) for name in names})

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetConfig:
    max_steps: int
    max_model_calls: int
    max_processed_pixels: int

    @classmethod
    def from_mapping(cls, value: Any) -> "BudgetConfig":
        data = _exact_keys(
            value, {"max_steps", "max_model_calls", "max_processed_pixels"}, "budget"
        )
        return cls(**{
            name: _positive_integer(data[name], name)
            for name in ("max_steps", "max_model_calls", "max_processed_pixels")
        })

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ControllerConfig:
    stall_patience: int
    min_progress: float
    max_gap_for_stop: float
    min_answer_confidence: float
    min_support_avg: float
    min_support_min: float

    @classmethod
    def from_mapping(cls, value: Any) -> "ControllerConfig":
        names = {
            "stall_patience", "min_progress", "max_gap_for_stop",
            "min_answer_confidence", "min_support_avg", "min_support_min",
        }
        data = _exact_keys(value, names, "controller")
        return cls(
            stall_patience=_positive_integer(data["stall_patience"], "stall_patience"),
            **{
                name: _probability(data[name], name)
                for name in names if name != "stall_patience"
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PDFSearchConfig:
    schema_version: int
    method: str
    profile: str
    ranking: RankingConfig
    modules: ModuleConfig
    budget: BudgetConfig
    controller: ControllerConfig

    @classmethod
    def from_mapping(cls, value: Any) -> "PDFSearchConfig":
        data = _exact_keys(
            value,
            {"schema_version", "method", "profile", "ranking", "modules", "budget", "controller"},
            "PDF search config",
        )
        if data["schema_version"] != 1:
            raise ValueError("schema_version must be 1")
        if data["method"] != "pdf_faithful_v1":
            raise ValueError("method must be pdf_faithful_v1")
        if data["profile"] not in {"full", "ablation"}:
            raise ValueError("profile must be full or ablation")
        ranking = RankingConfig.from_mapping(data["ranking"])
        modules = ModuleConfig.from_mapping(data["modules"])
        if data["profile"] == "full":
            if ranking.top_k_augmented != 3:
                raise ValueError("full profile requires top_k_augmented=3")
            for name, enabled in modules.to_dict().items():
                if not enabled:
                    raise ValueError(f"full profile requires module {name}")
            for name in ("main_query", "augmented_query", "complexity", "edge_density"):
                if not getattr(ranking, name):
                    raise ValueError(f"full profile requires ranking component {name}")
        return cls(
            schema_version=1,
            method="pdf_faithful_v1",
            profile=data["profile"],
            ranking=ranking,
            modules=modules,
            budget=BudgetConfig.from_mapping(data["budget"]),
            controller=ControllerConfig.from_mapping(data["controller"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "profile": self.profile,
            "ranking": self.ranking.to_dict(),
            "modules": self.modules.to_dict(),
            "budget": self.budget.to_dict(),
            "controller": self.controller.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class EvidenceGapScores:
    zoom: float
    split: float
    expand: float
    next: float

    def __post_init__(self) -> None:
        for name in ("zoom", "split", "expand", "next"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateDescriptor:
    canonical_key: str
    sibling_group: str
    native_ordinal: int
    bbox_original: tuple[float, float, float, float]
    depth: int
    render_level: int

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_key, str) or not self.canonical_key:
            raise ValueError("canonical_key must be nonempty")
        if not isinstance(self.sibling_group, str) or not self.sibling_group:
            raise ValueError("sibling_group must be nonempty")
        _nonnegative_integer(self.native_ordinal, "native_ordinal")
        _nonnegative_integer(self.depth, "depth")
        _nonnegative_integer(self.render_level, "render_level")
        if not isinstance(self.bbox_original, tuple) or len(self.bbox_original) != 4:
            raise TypeError("bbox_original must be a four-value tuple")
        bbox = tuple(float(value) for value in self.bbox_original)
        if not all(math.isfinite(value) for value in bbox) or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError("bbox_original must be finite xywh with positive size")
        object.__setattr__(self, "bbox_original", bbox)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["bbox_original"] = list(self.bbox_original)
        return result


@dataclass(frozen=True)
class SearchStateRecord:
    state_id: int
    focus_keys: tuple[str, ...]
    context_keys: tuple[str, ...]
    visited_keys: tuple[str, ...]
    remaining_steps: int
    remaining_model_calls: int
    remaining_pixels: int

    def __post_init__(self) -> None:
        _nonnegative_integer(self.state_id, "state_id")
        for name in ("focus_keys", "context_keys", "visited_keys"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not all(isinstance(item, str) and item for item in values):
                raise TypeError(f"{name} must be a tuple of nonempty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        for name in ("remaining_steps", "remaining_model_calls", "remaining_pixels"):
            _nonnegative_integer(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("focus_keys", "context_keys", "visited_keys"):
            result[name] = list(getattr(self, name))
        return result


@dataclass(frozen=True)
class ModelFingerprint:
    name: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("model fingerprint name must be nonempty")
        digest = self.artifact_sha256
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")


def validate_independent_checkpoints(
    generator: ModelFingerprint, verifier: ModelFingerprint,
) -> None:
    if not isinstance(generator, ModelFingerprint) or not isinstance(verifier, ModelFingerprint):
        raise TypeError("generator and verifier must be model fingerprints")
    if generator.artifact_sha256 == verifier.artifact_sha256:
        raise ValueError("independent verifier must use a different checkpoint")
