"""Shared candidate-free verifier veto and exact-P0 cascade decisions."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .freeze_uncertainty_support import _RiskExample, _RiskTopic
from .replay_uncertainty_support import CandidateSnapshot, RiskCalibrator


VERIFIER_CONFIDENCE = 0.6
PROPOSAL_AGREEMENT = 0.4


@dataclass(frozen=True)
class CandidateFreeVerifierEvidence:
    """One label-blind source answer plus scored development proposal."""

    verifier_feasible: bool
    verifier_canonical: Any
    verifier_confidence: float
    proposal_feasible: bool
    proposal_canonical: Any
    proposal_agreement: float
    proposal_corrections: int
    proposal_corruptions: int
    observations: int

    def __post_init__(self) -> None:
        if type(self.verifier_feasible) is not bool:
            raise TypeError("verifier feasibility must be an exact boolean")
        if type(self.proposal_feasible) is not bool:
            raise TypeError("proposal feasibility must be an exact boolean")
        for name in ("verifier_confidence", "proposal_agreement"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be finite in [0, 1]")
            object.__setattr__(self, name, float(value))
        for name in (
            "proposal_corrections", "proposal_corruptions", "observations",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative exact integer")
        if self.verifier_feasible and self.verifier_canonical is None:
            raise ValueError("feasible verifier answer must be canonical")
        if self.proposal_feasible and self.proposal_canonical is None:
            raise ValueError("feasible proposal must be canonical")


@dataclass(frozen=True)
class CandidateFreeCascadeOutcome:
    corrections: int
    corruptions: int
    observations: int
    selected_source: str
    vetoed: bool

    @property
    def net_gain(self) -> int:
        return self.corrections - self.corruptions


@dataclass(frozen=True)
class CandidateFreeRuntimeEvidence:
    """Label-blind proposal and verifier material required at deployment."""

    verifier_feasible: bool
    verifier_canonical: Any
    verifier_confidence: float
    proposal_feasible: bool
    proposal_output: Any
    proposal_canonical: Any
    proposal_agreement: float
    observations: int

    def __post_init__(self) -> None:
        if type(self.verifier_feasible) is not bool:
            raise TypeError("runtime verifier feasibility must be an exact boolean")
        if type(self.proposal_feasible) is not bool:
            raise TypeError("runtime proposal feasibility must be an exact boolean")
        for name in ("verifier_confidence", "proposal_agreement"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"runtime {name} must be finite in [0, 1]")
            object.__setattr__(self, name, float(value))
        if type(self.observations) is not int or self.observations < 0:
            raise ValueError("runtime observations must be a nonnegative exact integer")
        if self.verifier_feasible and self.verifier_canonical is None:
            raise ValueError("feasible runtime verifier answer must be canonical")
        if self.proposal_feasible and self.proposal_canonical is None:
            raise ValueError("feasible runtime proposal must be canonical")


@dataclass(frozen=True)
class CandidateFreeRuntimeOutcome:
    selected_output: Any
    selected_canonical: Any
    observations: int
    selected_source: str
    vetoed: bool


def _runtime_aggregate_candidate(
    snapshots: Sequence[CandidateSnapshot], calibrator: RiskCalibrator,
) -> CandidateSnapshot | None:
    if not isinstance(calibrator, RiskCalibrator):
        raise TypeError("runtime calibrator must be a frozen risk calibrator")
    checkpoints: list[list[CandidateSnapshot]] = []
    for snapshot in snapshots:
        if not isinstance(snapshot, CandidateSnapshot):
            raise TypeError("runtime snapshots must be frozen candidate snapshots")
        checkpoint = (snapshot.branch_index, snapshot.revealed_roles)
        previous = None if not checkpoints else (
            checkpoints[-1][0].branch_index,
            checkpoints[-1][0].revealed_roles,
        )
        if checkpoint != previous:
            checkpoints.append([])
        checkpoints[-1].append(snapshot)
    for candidates in checkpoints:
        if candidates[0].observations > calibrator.maximum_observations:
            break
        selected = min(
            candidates,
            key=lambda snapshot: (
                -calibrator.predict(
                    snapshot.features, snapshot.evidence_features,
                )[2],
                -snapshot.features.agreement,
                -snapshot.features.support,
            ),
        )
        if (
            selected.observations >= calibrator.minimum_observations
            and len(selected.agreeing_hashes)
            >= calibrator.minimum_agreeing_views
            and calibrator.predict(
                selected.features, selected.evidence_features,
            )[2] >= 0.0
        ):
            return selected
    return None


def candidate_free_runtime_outcome(
    snapshots: Sequence[CandidateSnapshot],
    p0_output: Any,
    calibrator: RiskCalibrator,
    evidence: CandidateFreeRuntimeEvidence,
    *,
    veto_confidence: float = VERIFIER_CONFIDENCE,
    proposal_confidence: float = VERIFIER_CONFIDENCE,
    proposal_agreement: float = PROPOSAL_AGREEMENT,
) -> CandidateFreeRuntimeOutcome:
    """Apply the development-equivalent cascade without evaluator labels."""
    if not isinstance(evidence, CandidateFreeRuntimeEvidence):
        raise TypeError("runtime evidence must be frozen candidate-free evidence")
    for name, value in (
        ("veto confidence", veto_confidence),
        ("proposal confidence", proposal_confidence),
        ("proposal agreement", proposal_agreement),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{name} boundary must be in [0, 1]")
    selected = _runtime_aggregate_candidate(snapshots, calibrator)
    vetoed = bool(
        selected is not None
        and evidence.verifier_feasible
        and evidence.verifier_confidence >= veto_confidence
        and evidence.verifier_canonical != selected.canonical_answer
    )
    if selected is not None and not vetoed:
        return CandidateFreeRuntimeOutcome(
            copy.deepcopy(selected.output),
            copy.deepcopy(selected.canonical_answer),
            max(selected.observations, evidence.observations),
            "AGGREGATE",
            False,
        )
    if (
        evidence.verifier_feasible
        and evidence.proposal_feasible
        and evidence.proposal_agreement >= proposal_agreement
        and evidence.verifier_confidence >= proposal_confidence
        and evidence.verifier_canonical == evidence.proposal_canonical
    ):
        return CandidateFreeRuntimeOutcome(
            copy.deepcopy(evidence.proposal_output),
            copy.deepcopy(evidence.proposal_canonical),
            evidence.observations,
            "VERIFIED_PROPOSAL",
            vetoed,
        )
    return CandidateFreeRuntimeOutcome(
        copy.deepcopy(p0_output), None, evidence.observations, "P0", vetoed,
    )


def _aggregate_candidate(
    topic: _RiskTopic, calibrator: RiskCalibrator,
) -> _RiskExample | None:
    checkpoints: list[list[_RiskExample]] = []
    for example in topic.examples:
        if not checkpoints or checkpoints[-1][0].checkpoint != example.checkpoint:
            checkpoints.append([])
        checkpoints[-1].append(example)
    for candidates in checkpoints:
        if candidates[0].observations > calibrator.maximum_observations:
            break
        selected = min(
            candidates,
            key=lambda example: (
                -calibrator.predict(
                    example.features, example.evidence_features,
                )[2],
                -example.features.agreement,
                -example.features.support,
            ),
        )
        if (
            selected.observations >= calibrator.minimum_observations
            and selected.agreeing_views >= calibrator.minimum_agreeing_views
            and calibrator.predict(
                selected.features, selected.evidence_features,
            )[2] >= 0.0
        ):
            return selected
    return None


def candidate_free_cascade_outcome(
    topic: _RiskTopic,
    calibrator: RiskCalibrator,
    evidence: CandidateFreeVerifierEvidence,
    *,
    veto_confidence: float = VERIFIER_CONFIDENCE,
    proposal_confidence: float = VERIFIER_CONFIDENCE,
    proposal_agreement: float = PROPOSAL_AGREEMENT,
) -> CandidateFreeCascadeOutcome:
    """Apply one shared uncertainty veto, proposal fallback, then exact P0."""
    if not isinstance(topic, _RiskTopic):
        raise TypeError("cascade topic must be a frozen risk topic")
    if not isinstance(calibrator, RiskCalibrator):
        raise TypeError("cascade calibrator must be a frozen risk calibrator")
    if not isinstance(evidence, CandidateFreeVerifierEvidence):
        raise TypeError("cascade evidence must be frozen candidate-free evidence")
    for name, value in (
        ("veto confidence", veto_confidence),
        ("proposal confidence", proposal_confidence),
        ("proposal agreement", proposal_agreement),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{name} boundary must be in [0, 1]")
    selected = _aggregate_candidate(topic, calibrator)
    return candidate_free_outcome_for_selected(
        selected, evidence,
        veto_confidence=veto_confidence,
        proposal_confidence=proposal_confidence,
        proposal_agreement=proposal_agreement,
    )


def candidate_free_outcome_for_selected(
    selected: _RiskExample | None,
    evidence: CandidateFreeVerifierEvidence,
    *,
    veto_confidence: float = VERIFIER_CONFIDENCE,
    proposal_confidence: float = VERIFIER_CONFIDENCE,
    proposal_agreement: float = PROPOSAL_AGREEMENT,
) -> CandidateFreeCascadeOutcome:
    """Finish the shared cascade after one label-blind aggregate decision."""
    if selected is not None and not isinstance(selected, _RiskExample):
        raise TypeError("selected aggregate candidate must be a risk example")
    if not isinstance(evidence, CandidateFreeVerifierEvidence):
        raise TypeError("cascade evidence must be frozen candidate-free evidence")
    for name, value in (
        ("veto confidence", veto_confidence),
        ("proposal confidence", proposal_confidence),
        ("proposal agreement", proposal_agreement),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{name} boundary must be in [0, 1]")
    vetoed = bool(
        selected is not None
        and evidence.verifier_feasible
        and evidence.verifier_confidence >= veto_confidence
        and evidence.verifier_canonical != selected.candidate_canonical
    )
    if selected is not None and not vetoed:
        return CandidateFreeCascadeOutcome(
            selected.correction_units,
            selected.corruption_units,
            max(selected.observations, evidence.observations),
            "AGGREGATE",
            False,
        )
    if (
        evidence.verifier_feasible
        and evidence.proposal_feasible
        and evidence.proposal_agreement >= proposal_agreement
        and evidence.verifier_confidence >= proposal_confidence
        and evidence.verifier_canonical == evidence.proposal_canonical
    ):
        return CandidateFreeCascadeOutcome(
            evidence.proposal_corrections,
            evidence.proposal_corruptions,
            evidence.observations,
            "VERIFIED_PROPOSAL",
            vetoed,
        )
    return CandidateFreeCascadeOutcome(
        0, 0, evidence.observations, "P0", vetoed,
    )


__all__ = [
    "CandidateFreeCascadeOutcome",
    "CandidateFreeRuntimeEvidence",
    "CandidateFreeRuntimeOutcome",
    "CandidateFreeVerifierEvidence",
    "PROPOSAL_AGREEMENT",
    "VERIFIER_CONFIDENCE",
    "candidate_free_cascade_outcome",
    "candidate_free_outcome_for_selected",
    "candidate_free_runtime_outcome",
]
