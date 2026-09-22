"""Strict configuration for the independent query-aware controller."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from typing import Any

from qavs.evidence_gap.pdf_types import (
    BudgetConfig,
    ControllerConfig,
    PDFSearchConfig,
    RankingConfig,
)


def _mapping(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != keys:
        raise ValueError(f"{name} must contain exact keys {sorted(keys)}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _integer(value: Any, name: str, *, positive: bool) -> int:
    qualifier = "positive" if positive else "non-negative"
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a {qualifier} integer")
    if value < (1 if positive else 0):
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


@dataclass(frozen=True)
class IndependentModules:
    query_ranking: bool
    adaptive_observation: bool
    evidence_validation: bool

    @classmethod
    def from_mapping(cls, value: Any) -> "IndependentModules":
        names = {"query_ranking", "adaptive_observation", "evidence_validation"}
        data = _mapping(value, names, "modules")
        return cls(**{name: _boolean(data[name], name) for name in names})

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class GlobalGateConfig:
    min_support: float
    min_margin: float
    min_consistency: float

    @classmethod
    def from_mapping(cls, value: Any) -> "GlobalGateConfig":
        names = {"min_support", "min_margin", "min_consistency"}
        data = _mapping(value, names, "global_gate")
        return cls(**{name: _probability(data[name], name) for name in names})

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ProposalConfig:
    dedup_iou: float
    min_count: int
    min_spatial_coverage: float
    active_top_k: int

    @classmethod
    def from_mapping(cls, value: Any) -> "ProposalConfig":
        data = _mapping(value, {
            "dedup_iou", "min_count", "min_spatial_coverage", "active_top_k",
        }, "proposals")
        return cls(
            dedup_iou=_probability(data["dedup_iou"], "dedup_iou"),
            min_count=_integer(data["min_count"], "min_count", positive=False),
            min_spatial_coverage=_probability(
                data["min_spatial_coverage"], "min_spatial_coverage"
            ),
            active_top_k=_integer(data["active_top_k"], "active_top_k", positive=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BranchAcceptanceConfig:
    """Paper-aligned acceptance thresholds for branch-scoped evidence."""

    min_absolute_support: float
    min_normalized_margin: float
    min_view_support: float
    aggregation: str
    global_verifier_min_support: float | None = None
    global_verifier_min_margin: float | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "BranchAcceptanceConfig":
        base_keys = {
            "min_absolute_support", "min_normalized_margin",
            "min_view_support", "aggregation",
        }
        verifier_keys = {
            "global_verifier_min_support", "global_verifier_min_margin",
        }
        if not isinstance(value, Mapping):
            raise TypeError("acceptance must be a mapping")
        hybrid = str(value.get("aggregation", "")).startswith(
            "global_verifier_then_"
        )
        expected_keys = base_keys | verifier_keys if hybrid else base_keys
        data = _mapping(value, expected_keys, "acceptance")
        if data["aggregation"] not in {
            "branch_equal_mean", "grounded_answer_consensus",
            "global_verifier_then_branch_equal_mean",
            "global_verifier_then_grounded_answer_consensus",
        }:
            raise ValueError(
                "aggregation must be branch_equal_mean, "
                "grounded_answer_consensus, or a global verifier routing hybrid"
            )
        return cls(
            min_absolute_support=_probability(
                data["min_absolute_support"], "min_absolute_support"
            ),
            min_normalized_margin=_probability(
                data["min_normalized_margin"], "min_normalized_margin"
            ),
            min_view_support=_probability(
                data["min_view_support"], "min_view_support"
            ),
            aggregation=data["aggregation"],
            global_verifier_min_support=(
                _probability(
                    data["global_verifier_min_support"],
                    "global_verifier_min_support",
                )
                if hybrid else None
            ),
            global_verifier_min_margin=(
                _probability(
                    data["global_verifier_min_margin"],
                    "global_verifier_min_margin",
                )
                if hybrid else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "min_absolute_support": self.min_absolute_support,
            "min_normalized_margin": self.min_normalized_margin,
            "min_view_support": self.min_view_support,
            "aggregation": self.aggregation,
        }
        if self.global_verifier_min_support is not None:
            result.update({
                "global_verifier_min_support": (
                    self.global_verifier_min_support
                ),
                "global_verifier_min_margin": self.global_verifier_min_margin,
            })
        return result


@dataclass(frozen=True)
class VerificationConfig:
    labels: tuple[str, str, str]
    grounding_labels: tuple[str, str, str]
    grounding_threshold: float
    target_instance_iou: float

    @classmethod
    def from_mapping(cls, value: Any) -> "VerificationConfig":
        data = _mapping(value, {
            "labels", "grounding_labels", "grounding_threshold",
            "target_instance_iou",
        }, "verification")
        labels = tuple(data["labels"]) if isinstance(data["labels"], list) else None
        grounding_labels = (
            tuple(data["grounding_labels"])
            if isinstance(data["grounding_labels"], list) else None
        )
        if labels != ("Support", "Refute", "Insufficient"):
            raise ValueError("labels must be Support, Refute, Insufficient in order")
        if grounding_labels != (
            "Grounded", "NotGrounded", "Insufficient",
        ):
            raise ValueError(
                "grounding_labels must be Grounded, NotGrounded, "
                "Insufficient in order"
            )
        return cls(
            labels=labels,
            grounding_labels=grounding_labels,
            grounding_threshold=_probability(
                data["grounding_threshold"], "grounding_threshold"
            ),
            target_instance_iou=_probability(
                data["target_instance_iou"], "target_instance_iou"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "grounding_labels": list(self.grounding_labels),
            "grounding_threshold": self.grounding_threshold,
            "target_instance_iou": self.target_instance_iou,
        }


@dataclass(frozen=True)
class ObservationGeometryConfig:
    min_zoom_factor: float
    context_max_normalized_gap: float

    @classmethod
    def from_mapping(cls, value: Any) -> "ObservationGeometryConfig":
        data = _mapping(value, {
            "min_zoom_factor", "context_max_normalized_gap",
        }, "observation_geometry")
        min_zoom_factor = _probability(
            data["min_zoom_factor"], "min_zoom_factor"
        )
        if min_zoom_factor == 0.0:
            raise ValueError("min_zoom_factor must be greater than zero")
        return cls(
            min_zoom_factor=min_zoom_factor,
            context_max_normalized_gap=_probability(
                data["context_max_normalized_gap"],
                "context_max_normalized_gap",
            ),
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class IndependentSearchConfig:
    schema_version: int
    method: str
    profile: str
    direct_threshold: float
    ranking: RankingConfig
    modules: IndependentModules
    budget: BudgetConfig
    controller: ControllerConfig
    global_gate: GlobalGateConfig
    proposals: ProposalConfig
    acceptance: BranchAcceptanceConfig
    verification: VerificationConfig
    observation_geometry: ObservationGeometryConfig

    @classmethod
    def from_mapping(cls, value: Any) -> "IndependentSearchConfig":
        if not isinstance(value, Mapping):
            raise TypeError("independent search config must be a mapping")
        schema_version = value.get("schema_version")
        base_keys = {
            "schema_version", "method", "profile", "direct_threshold",
            "ranking", "modules", "budget", "controller",
            "global_gate", "proposals", "acceptance",
        }
        if schema_version != 3:
            raise ValueError("schema_version must be 3")
        data = _mapping(
            value, base_keys | {"verification", "observation_geometry"},
            "independent search config",
        )
        expected_method = "independent_query_aware_v3"
        if data["method"] != expected_method:
            raise ValueError(f"method must be {expected_method}")
        if data["profile"] != "full":
            raise ValueError("profile must be full")
        threshold = data["direct_threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("direct_threshold must be a finite number in [-1, 1]")
        threshold = float(threshold)
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("direct_threshold must be a finite number in [-1, 1]")
        ranking = RankingConfig.from_mapping(data["ranking"])
        modules = IndependentModules.from_mapping(data["modules"])
        if data["profile"] == "full":
            for name, enabled in modules.to_dict().items():
                if not enabled:
                    raise ValueError(f"full profile requires module {name}")
            if ranking.top_k_augmented != 3:
                raise ValueError("full profile requires top_k_augmented=3")
            for name in ("main_query", "augmented_query", "complexity", "edge_density"):
                if not getattr(ranking, name):
                    raise ValueError(f"full profile requires ranking component {name}")
        return cls(
            schema_version=schema_version,
            method=expected_method,
            profile=data["profile"],
            direct_threshold=threshold,
            ranking=ranking,
            modules=modules,
            budget=BudgetConfig.from_mapping(data["budget"]),
            controller=ControllerConfig.from_mapping(data["controller"]),
            global_gate=GlobalGateConfig.from_mapping(data["global_gate"]),
            proposals=ProposalConfig.from_mapping(data["proposals"]),
            acceptance=BranchAcceptanceConfig.from_mapping(data["acceptance"]),
            verification=VerificationConfig.from_mapping(data["verification"]),
            observation_geometry=ObservationGeometryConfig.from_mapping(
                data["observation_geometry"]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "method": self.method,
            "profile": self.profile,
            "direct_threshold": self.direct_threshold,
            "ranking": self.ranking.to_dict(),
            "modules": self.modules.to_dict(),
            "budget": self.budget.to_dict(),
            "controller": self.controller.to_dict(),
            "global_gate": self.global_gate.to_dict(),
            "proposals": self.proposals.to_dict(),
            "acceptance": self.acceptance.to_dict(),
        }
        result.update({
            "verification": self.verification.to_dict(),
            "observation_geometry": self.observation_geometry.to_dict(),
        })
        return result

    def to_pdf_config(self) -> PDFSearchConfig:
        ranking = self.ranking.to_dict()
        if not self.modules.query_ranking:
            ranking.update({
                "alpha": 0.0,
                "main_query": False,
                "augmented_query": False,
            })
        adaptive = self.modules.adaptive_observation
        validation = self.modules.evidence_validation
        return PDFSearchConfig.from_mapping({
            "schema_version": 1,
            "method": "pdf_faithful_v1",
            "profile": "core",
            "ranking": ranking,
            "modules": {
                "structured_planner": True,
                "zoom": adaptive,
                "split": adaptive,
                "expand": adaptive,
                "next": True,
                "reanswer": True,
                "independent_verifier": validation,
                "backtracking": validation,
                "certified_stop": False,
            },
            "budget": self.budget.to_dict(),
            "controller": self.controller.to_dict(),
        })
