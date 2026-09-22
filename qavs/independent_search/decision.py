"""Controller-owned best-evidence selection with a full-image fallback."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

from qavs.evidence_gap.helpers import (
    confirmation_prompt_material,
)
from qavs.evidence_gap.helpers import proposed_answer_text
from qavs.evidence_gap.answers import canonical_text
from qavs.evidence_gap.input import sanitize_annotation
from qavs.evidence_gap.pdf_runtime import (
    answer_with_uncertainty,
    verify_answer_support,
    wrapper_yes_no_probability,
)
from qavs.evidence_gap.pdf_types import (
    ModelFingerprint,
    validate_independent_checkpoints,
)
from qavs.evidence_gap.types import sanitize_evidence_requirements
from qavs.models.tree import NodeA, NodeState

from .config import BranchAcceptanceConfig
from .semantics import OptionCatalog, verify_option_support
from .binary import NegativeCoverage, StrictNoConfig, evaluate_negative_gate


def _positive_cost(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def source_image_identity(image: Any) -> dict[str, Any]:
    """Return the decoded RGB pixel identity used by online and offline audit."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("source image identity requires a PIL image")
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    return {
        "mode": "RGB",
        "size": [rgb.width, rgb.height],
        "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
    }


def _phase_record_identity(record: Mapping[str, Any]) -> str:
    state = record.get("state")
    if not isinstance(state, Mapping):
        raise TypeError("state evaluation state must be a mapping")
    required = (
        "source_image_identity", "effective_geometry",
        "rendered_scale", "branch_context",
    )
    if any(name not in state for name in required):
        raise ValueError("state evaluation is missing its evidence identity")
    names = list(required)
    if "render_sha256" in state or "action" in state:
        revision3 = ("action", "render_sha256", "path_keys")
        if any(name not in state for name in revision3):
            raise ValueError("Revision-3 evidence identity is incomplete")
        names.extend(revision3)
    return _strict_json(
        {name: state[name] for name in names}, "evidence identity",
    )


def cumulative_phase_records(
    cumulative: Sequence[Mapping[str, Any]],
    phase_records: Sequence[Mapping[str, Any]],
    *,
    root_key: str,
) -> list[dict[str, Any]]:
    """Return the online cumulative selector stream after one controller phase.

    Only a later rendering that is exactly the original root view is omitted.
    Other repeated views are selector events because a later assessment replaces
    the earlier ledger group.
    """
    if isinstance(cumulative, (str, bytes)) or not isinstance(cumulative, Sequence):
        raise TypeError("cumulative records must be a sequence")
    if isinstance(phase_records, (str, bytes)) or not isinstance(phase_records, Sequence):
        raise TypeError("phase records must be a sequence")
    if not isinstance(root_key, str) or not root_key:
        raise ValueError("root_key must be nonempty text")
    result = [json.loads(_strict_json(record, "state evaluation")) for record in cumulative]
    root_identity = None if not result else _phase_record_identity(result[0])
    for raw in phase_records:
        record = json.loads(_strict_json(raw, "state evaluation"))
        identity = _phase_record_identity(record)
        if root_identity is not None and identity == root_identity:
            continue
        if root_identity is None:
            state = record["state"]
            if state.get("branch_context") != [root_key]:
                raise ValueError("the cumulative ledger must begin with the root view")
            root_identity = identity
        result.append(record)
    return result


@dataclass(frozen=True)
class GlobalCost:
    """Exact immutable cost of producing and validating ``P_global``."""

    model_calls: int
    processed_pixels: int

    def __post_init__(self) -> None:
        _positive_cost(self.model_calls, "global model_calls")
        _positive_cost(self.processed_pixels, "global processed_pixels")

    def to_dict(self) -> dict[str, int]:
        return {
            "model_calls": self.model_calls,
            "processed_pixels": self.processed_pixels,
        }


@dataclass(frozen=True)
class GlobalGateDecision:
    sufficient: bool
    accepted: bool

    @property
    def direct_stop(self) -> bool:
        return self.sufficient and self.accepted


def evaluate_global_gate(
    *, sufficient_score: Any, direct_threshold: Any, support: Any,
    margin: Any, consistency: Any, valid: Any, gate: Any,
) -> GlobalGateDecision:
    """Apply the one pure global gate shared by online inference and replay."""
    from .config import GlobalGateConfig

    if not isinstance(gate, GlobalGateConfig):
        raise TypeError("gate must be GlobalGateConfig")
    if type(valid) is not bool:
        raise TypeError("global valid must be boolean")
    score = _confidence(sufficient_score, "sufficient score")
    threshold = _confidence(direct_threshold, "direct threshold")
    support_value = _probability(support, "global support")
    margin_value = _probability(margin, "global margin")
    consistency_value = _probability(consistency, "global consistency")
    return GlobalGateDecision(
        sufficient=score > threshold,
        accepted=(
            valid
            and support_value >= gate.min_support
            and margin_value >= gate.min_margin
            and consistency_value >= gate.min_consistency
        ),
    )


def build_query_plan(policy: Mapping[str, Any], targets: Sequence[str]) -> Any:
    """Avoid importing independent-search orchestration during module loading."""
    from qavs.evidence_gap.helpers import build_query_plan as implementation

    return implementation(policy, targets)


def _strict_json(value: Any, name: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be strict JSON") from error


def _confidence(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite confidence in [-1, 1]")
    result = float(value)
    if not math.isfinite(result) or not -1.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite confidence in [-1, 1]")
    return result


@dataclass(frozen=True, init=False)
class GlobalObservation:
    """Immutable method-owned full-image prediction and independent validation."""

    _output_json: str
    _answer_record_json: str | None
    _option_support_json: str | None
    sufficient_score: float
    sufficient: bool
    support: float
    margin: float
    consistency: float
    valid: bool
    accepted: bool
    verifier_provenance: Any
    cost: GlobalCost

    def __init__(
        self,
        output: Any,
        sufficient_score: float,
        sufficient: bool,
        support: float,
        margin: float,
        consistency: float,
        valid: bool,
        accepted: bool,
        answer_record: Any,
        verifier_provenance: Any,
        cost: GlobalCost,
        option_support: Any | None = None,
    ) -> None:
        object.__setattr__(self, "_output_json", _strict_json(output, "P_global"))
        object.__setattr__(self, "sufficient_score", _confidence(sufficient_score, "sufficient score"))
        for name, value in (("sufficient", sufficient), ("valid", valid), ("accepted", accepted)):
            if type(value) is not bool:
                raise TypeError(f"{name} must be boolean")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "support", _probability(support, "global support"))
        object.__setattr__(self, "margin", _probability(margin, "global margin"))
        object.__setattr__(self, "consistency", _probability(consistency, "global consistency"))
        answer_payload = None if answer_record is None else answer_record.to_dict()
        object.__setattr__(
            self,
            "_answer_record_json",
            None if answer_payload is None else _strict_json(answer_payload, "answer record"),
        )
        object.__setattr__(self, "verifier_provenance", verifier_provenance)
        option_payload = (
            None if option_support is None else option_support.to_dict()
        )
        object.__setattr__(
            self,
            "_option_support_json",
            None if option_payload is None else _strict_json(
                option_payload, "global option support",
            ),
        )
        if not isinstance(cost, GlobalCost):
            raise TypeError("cost must be GlobalCost")
        object.__setattr__(self, "cost", cost)

    @property
    def output(self) -> Any:
        """Return a fresh copy so callers cannot mutate saved ``P_global``."""
        return json.loads(self._output_json)

    @property
    def answer_record(self) -> dict[str, Any] | None:
        """Return a fresh snapshot of the immutable generation provenance."""
        return (
            None if self._answer_record_json is None
            else json.loads(self._answer_record_json)
        )

    @property
    def option_support(self) -> dict[str, Any] | None:
        """Return a fresh immutable snapshot of the complete global vector."""
        return (
            None if self._option_support_json is None
            else json.loads(self._option_support_json)
        )

    @property
    def direct_stop(self) -> bool:
        return self.sufficient and self.accepted

    def validation_dict(self) -> dict[str, Any]:
        return {
            "sufficient_score": self.sufficient_score,
            "sufficient": self.sufficient,
            "support": self.support,
            "margin": self.margin,
            "consistency": self.consistency,
            "valid": self.valid,
            "accepted": self.accepted,
            "direct_stop": self.direct_stop,
        }

    def to_dict(self) -> dict[str, Any]:
        provenance = self.verifier_provenance
        payload = {
            "output": self.output,
            **self.validation_dict(),
            "answer_record": self.answer_record,
            "verifier_provenance": (
                None if provenance is None else provenance.to_dict()
            ),
            "cost": self.cost.to_dict(),
        }
        if self.option_support is not None:
            payload["option_support"] = self.option_support
        return payload


def observe_global(
    *,
    model: Any,
    verifier_model: Any,
    policy: Mapping[str, Any],
    image: Any,
    direct_threshold: float,
    gate: Any,
    generator_checkpoint_sha256: str,
    verifier_checkpoint_sha256: str,
    option_catalog: OptionCatalog | None = None,
    conditional_losses: Any | None = None,
    pair_scoring: Any | None = None,
) -> GlobalObservation:
    """Generate and validate the one full-image prediction before any SAM work."""
    from .config import GlobalGateConfig

    if not isinstance(gate, GlobalGateConfig):
        raise TypeError("gate must be GlobalGateConfig")
    validated_policy = sanitize_annotation(policy)
    validate_independent_checkpoints(
        ModelFingerprint("generator", generator_checkpoint_sha256),
        ModelFingerprint("independent-verifier", verifier_checkpoint_sha256),
    )
    def generate():
        answer = answer_with_uncertainty(model, validated_policy, image, ())
        # Reject malformed predictions before confidence/legacy verification.
        _strict_json(answer.output, "P_global")
        root = NodeA(NodeState(image, [0, 0, image.width, image.height]))
        root.is_root = True
        root.search_source = "global"
        sufficient_score = _confidence(model.get_confidence_value(
            [root], image, confidence_type="answering", input_ele=validated_policy["question"],
        ), "sufficient score")
        return answer, sufficient_score

    plan = build_query_plan(validated_policy, ())
    requirements = sanitize_evidence_requirements(plan.evidence_items)
    option_vector = None
    if option_catalog is not None:
        if not callable(conditional_losses):
            raise TypeError(
                "Revision-3 global observation requires conditional_losses"
            )
        def verify():
            return verify_option_support(
                question=validated_policy["question"],
                catalog=option_catalog,
                image=image,
                requirements=tuple(
                    getattr(item, "text", str(item)) for item in requirements
                ),
                conditional_losses=conditional_losses,
                checkpoint_sha256=verifier_checkpoint_sha256,
                generator_checkpoint_sha256=generator_checkpoint_sha256,
            )
        generated, option_vector = (
            pair_scoring(generate, verify) if pair_scoring is not None else
            (generate(), verify())
        )
    else:
        generated = generate()
    answer, sufficient_score = generated
    output = json.loads(_strict_json(answer.output, "P_global"))
    if option_catalog is not None:
        ranked = sorted(
            option_vector.options,
            key=lambda item: (-item.normalized_support, item.key),
        )
        top = ranked[0]
        runner_support = (
            ranked[1].normalized_support if len(ranked) > 1 else 0.0
        )
        support_value = top.raw_support
        margin_value = top.normalized_support - runner_support
        try:
            agrees = _strict_json(
                option_catalog.project(top.key), "global verifier output",
            ) == _strict_json(output, "P_global")
        except (KeyError, TypeError, ValueError):
            agrees = False
        valid = bool(
            answer.aggregation_available and option_vector.valid and agrees
        )
        verifier_provenance = None
        support_calls = option_vector.model_calls
    else:
        if conditional_losses is not None:
            raise ValueError(
                "conditional_losses requires a Revision-3 option catalog"
            )
        semantic_answer = proposed_answer_text(
            validated_policy["answer_type"], validated_policy["options"], output,
        )
        support = verify_answer_support(
            q0=validated_policy["question"],
            proposed_answer=(semantic_answer or "unavailable semantic answer"),
            requirements=requirements,
            rendered_observation=image,
            probability=lambda view, prompt: wrapper_yes_no_probability(
                verifier_model, view, prompt,
            ),
            checkpoint_sha256=verifier_checkpoint_sha256,
            generator_checkpoint_sha256=generator_checkpoint_sha256,
        )
        support_value = support.support_avg
        margin_value = answer.margin
        valid = (
            bool(answer.aggregation_available)
            and semantic_answer is not None
            and support.independent
            and not support.fallback_used
        )
        verifier_provenance = support
        support_calls = getattr(support, "model_calls", None)
    gate_decision = evaluate_global_gate(
        sufficient_score=sufficient_score,
        direct_threshold=direct_threshold,
        support=support_value,
        margin=margin_value,
        consistency=answer.frequency,
        valid=valid,
        gate=gate,
    )
    prompt_calls = len(confirmation_prompt_material(
        (
            "logits_match"
            if validated_policy["answer_type"] == "yes_no"
            else validated_policy["answer_type"]
        ),
        validated_policy["question"],
        validated_policy["options"],
    )["prompts"])
    if isinstance(support_calls, bool) or not isinstance(support_calls, int) or support_calls < 0:
        raise ValueError("global verifier model_calls must be a non-negative integer")
    model_calls = prompt_calls + 1 + support_calls
    cost = GlobalCost(
        model_calls=model_calls,
        processed_pixels=model_calls * image.width * image.height,
    )
    return GlobalObservation(
        output=output,
        sufficient_score=sufficient_score,
        sufficient=gate_decision.sufficient,
        support=support_value,
        margin=margin_value,
        consistency=answer.frequency,
        valid=valid,
        accepted=gate_decision.accepted,
        answer_record=answer,
        verifier_provenance=verifier_provenance,
        cost=cost,
        option_support=option_vector,
    )


@dataclass(frozen=True)
class FinalDecision:
    answer: Any
    source: str
    accepted_hypothesis: BranchAcceptedHypothesis | None
    ledger_trace: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ledger_trace",
            json.loads(_strict_json(self.ledger_trace, "evidence ledger trace")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": json.loads(_strict_json(self.answer, "final answer")),
            "source": self.source,
            "accepted_hypothesis": (
                None
                if self.accepted_hypothesis is None
                else self.accepted_hypothesis.to_dict()
            ),
            "ledger_trace": json.loads(_strict_json(
                self.ledger_trace, "evidence ledger trace",
            )),
        }


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite probability")
    return result


@dataclass(frozen=True)
class BranchEvidenceGroup:
    group_id: str
    state_id: int
    shared_global: bool
    positive_support_eligible: bool
    path: tuple[str, ...]
    context_anchor_ids: tuple[str, ...]
    action: str
    render_sha256: str
    source_image_identity: dict[str, Any]
    effective_geometry: tuple[float, float, float, float]
    rendered_scale: float
    raw_support: tuple[tuple[str, float], ...]
    normalized_support: tuple[tuple[str, float], ...]
    winning_labels: tuple[tuple[str, str], ...]
    vector_valid: bool
    generated_output_valid: bool
    grounding_valid: bool
    role_to_instance: tuple[tuple[str, str], ...]
    covered_instance_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "state_id": self.state_id,
            "shared_global": self.shared_global,
            "positive_support_eligible": self.positive_support_eligible,
            "path": list(self.path),
            "context_anchor_ids": list(self.context_anchor_ids),
            "action": self.action,
            "render_sha256": self.render_sha256,
            "source_image_identity": self.source_image_identity,
            "effective_geometry": list(self.effective_geometry),
            "rendered_scale": self.rendered_scale,
            "raw_support": dict(self.raw_support),
            "normalized_support": dict(self.normalized_support),
            "winning_labels": dict(self.winning_labels),
            "vector_valid": self.vector_valid,
            "generated_output_valid": self.generated_output_valid,
            "grounding_valid": self.grounding_valid,
            "role_to_instance": [
                {"role": role, "instance_id": instance_id}
                for role, instance_id in self.role_to_instance
            ],
            "covered_instance_ids": list(self.covered_instance_ids),
        }


@dataclass(frozen=True)
class BranchAcceptedHypothesis:
    answer: Any
    canonical_answer: str
    absolute_support: float
    normalized_support: float
    margin: float
    evidence_group_ids: tuple[str, ...]
    confirmation_group_ids: tuple[str, str]
    grounding_signature: tuple[Any, ...]
    raw_support_vector: tuple[tuple[str, float], ...]
    normalized_support_vector: tuple[tuple[str, float], ...]
    accepted_at_state_id: int
    score_source: str = "branch_equal_mean"
    bundle_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": json.loads(_strict_json(self.answer, "accepted answer")),
            "canonical_answer": self.canonical_answer,
            "absolute_support": self.absolute_support,
            "normalized_support": self.normalized_support,
            "margin": self.margin,
            "evidence_group_ids": list(self.evidence_group_ids),
            "confirmation_group_ids": list(self.confirmation_group_ids),
            "grounding_signature": json.loads(_strict_json(
                self.grounding_signature, "grounding signature",
            )),
            "raw_support_vector": dict(self.raw_support_vector),
            "normalized_support_vector": dict(self.normalized_support_vector),
            "accepted_at_state_id": self.accepted_at_state_id,
            "score_source": self.score_source,
            "bundle_id": self.bundle_id,
        }


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _string_tuple(value: Any, name: str, *, nonempty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise TypeError(f"{name} must be a string list")
    if nonempty and not value:
        raise ValueError(f"{name} must be nonempty")
    return tuple(value)


def _branch_record(
    value: Any,
    *,
    root_state_id: int,
    catalog: OptionCatalog,
) -> BranchEvidenceGroup:
    if not isinstance(value, Mapping):
        raise TypeError("branch evidence record must be a mapping")
    state = value.get("state")
    vector = value.get("option_support")
    if not isinstance(state, Mapping) or not isinstance(vector, Mapping):
        raise TypeError("branch evidence requires state and option_support mappings")
    state_id = state.get("state_id")
    if isinstance(state_id, bool) or not isinstance(state_id, int) or state_id < 0:
        raise ValueError("branch state_id must be a non-negative integer")
    path = _string_tuple(state.get("path_keys"), "path_keys", nonempty=True)
    context = _string_tuple(
        state.get("context_keys"), "context_keys", nonempty=False,
    )
    source = state.get("source_image_identity")
    if not isinstance(source, Mapping):
        raise TypeError("branch evidence source identity must be a mapping")
    _sha256(source.get("pixel_sha256"), "source image digest")
    geometry_value = state.get("effective_geometry")
    if (
        not isinstance(geometry_value, list)
        or len(geometry_value) != 4
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in geometry_value
        )
    ):
        raise ValueError("branch evidence geometry must be an xywh number list")
    geometry = tuple(float(item) for item in geometry_value)
    if not all(math.isfinite(item) for item in geometry) or min(geometry[2:]) <= 0:
        raise ValueError("branch evidence geometry must have finite positive area")
    scale = state.get("rendered_scale")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0
    ):
        raise ValueError("branch rendered_scale must be finite and positive")
    action = state.get("action")
    if action not in {
        "GLOBAL", "BASE", "NEXT", "ZOOM", "SPLIT", "EXPAND", "RECOVER",
    }:
        raise ValueError("branch evidence action is invalid")
    render_sha256 = _sha256(state.get("render_sha256"), "render digest")
    if vector.get("catalog_sha256") != catalog.identity_sha256:
        raise ValueError("option vector catalog identity mismatch")
    options = vector.get("options")
    if not isinstance(options, list) or len(options) != len(catalog.entries):
        raise ValueError("option vector must cover the complete catalog")
    raw_support = []
    normalized_support = []
    winning_labels = []
    for expected, option in zip(catalog.entries, options, strict=True):
        if not isinstance(option, Mapping) or option.get("key") != expected.key:
            raise ValueError("option vector keys do not match catalog order")
        expected_text_sha = hashlib.sha256(expected.text.encode("utf-8")).hexdigest()
        if option.get("text_sha256") != expected_text_sha:
            raise ValueError("option vector text digest mismatch")
        distribution = option.get("distribution")
        if (
            not isinstance(distribution, Mapping)
            or distribution.get("labels") != ["Support", "Refute", "Insufficient"]
            or distribution.get("winner") not in {
                "Support", "Refute", "Insufficient",
            }
        ):
            raise ValueError("option vector distribution is invalid")
        raw_support.append((
            expected.key,
            _probability(option.get("raw_support"), "raw option support"),
        ))
        normalized_support.append((
            expected.key,
            _probability(
                option.get("normalized_support"), "normalized option support",
            ),
        ))
        winning_labels.append((expected.key, distribution["winner"]))
    if not math.isclose(
        math.fsum(value for _, value in normalized_support),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("normalized option support must sum to one")
    if type(vector.get("valid")) is not bool:
        raise TypeError("option vector valid flag must be boolean")
    answer = value.get("answer")
    generated_output_valid = (
        isinstance(answer, Mapping)
        and "output" in answer
        and _catalog_output_valid(answer["output"], catalog)
    )

    shared_global = state_id == root_state_id
    positive_support_eligible = True
    if shared_global:
        value = state.get("positive_support_eligible", True)
        if type(value) is not bool:
            raise TypeError("global positive-support eligibility must be boolean")
        positive_support_eligible = value
    grounding_valid = False
    role_to_instance: tuple[tuple[str, str], ...] = ()
    covered_instance_ids: tuple[str, ...] = ()
    if not shared_global:
        grounding = value.get("grounding")
        record = grounding.get("record") if isinstance(grounding, Mapping) else None
        if not isinstance(record, Mapping) or type(record.get("valid")) is not bool:
            raise TypeError("local branch evidence requires a grounding record")
        mapping = record.get("role_to_instance")
        if not isinstance(mapping, list) or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("role"), str)
            or not item.get("role")
            or not isinstance(item.get("instance_id"), str)
            or not item.get("instance_id")
            for item in mapping
        ):
            raise ValueError("grounding role-to-instance mapping is invalid")
        role_to_instance = tuple(sorted(
            (item["role"], item["instance_id"]) for item in mapping
        ))
        covered_instance_ids = tuple(sorted(_string_tuple(
            record.get("covered_instance_ids"),
            "covered_instance_ids",
            nonempty=False,
        )))
        grounding_valid = record["valid"]
    identity = {
        "source_image_sha256": source["pixel_sha256"],
        "effective_geometry": geometry,
        "rendered_scale": float(scale),
        "action": action,
        "render_sha256": render_sha256,
        "path": path,
    }
    group_id = hashlib.sha256(
        _strict_json(identity, "branch evidence identity").encode("utf-8")
    ).hexdigest()
    return BranchEvidenceGroup(
        group_id=group_id,
        state_id=state_id,
        shared_global=shared_global,
        positive_support_eligible=positive_support_eligible,
        path=path,
        context_anchor_ids=context,
        action=action,
        render_sha256=render_sha256,
        source_image_identity=dict(source),
        effective_geometry=geometry,  # type: ignore[arg-type]
        rendered_scale=float(scale),
        raw_support=tuple(raw_support),
        normalized_support=tuple(normalized_support),
        winning_labels=tuple(winning_labels),
        vector_valid=vector["valid"],
        generated_output_valid=generated_output_valid,
        grounding_valid=grounding_valid,
        role_to_instance=role_to_instance,
        covered_instance_ids=covered_instance_ids,
    )


def _prefix(prefix: tuple[str, ...], path: tuple[str, ...]) -> bool:
    return len(prefix) <= len(path) and path[:len(prefix)] == prefix


def _grounding_signature(
    group: BranchEvidenceGroup, question_kind: str,
) -> tuple[Any, ...]:
    signature: tuple[Any, ...] = (group.role_to_instance,)
    if question_kind in {"count", "coverage"}:
        signature += (group.covered_instance_ids,)
    return signature


def _confirmation_pair(
    groups: Sequence[BranchEvidenceGroup],
    *,
    top_key: str,
    min_view_support: float,
    question_kind: str,
) -> tuple[BranchEvidenceGroup, BranchEvidenceGroup] | None:
    eligible = []
    for group in groups:
        if group.shared_global or not group.grounding_valid:
            continue
        raw = dict(group.raw_support)
        labels = dict(group.winning_labels)
        if labels[top_key] != "Support" or raw[top_key] < min_view_support:
            continue
        eligible.append(group)
    for index, first in enumerate(eligible):
        for second in eligible[index + 1:]:
            if _grounding_signature(first, question_kind) != _grounding_signature(
                second, question_kind,
            ):
                continue
            if first.render_sha256 == second.render_sha256:
                continue
            if not any((
                first.effective_geometry != second.effective_geometry,
                first.rendered_scale != second.rendered_scale,
                first.action != second.action,
            )):
                continue
            return first, second
    return None


def _catalog_output_valid(output: Any, catalog: OptionCatalog) -> bool:
    try:
        value = _strict_json(output, "generated option output")
        return any(
            value == _strict_json(catalog.project(entry.key), "catalog output")
            for entry in catalog.entries
        )
    except (KeyError, TypeError, ValueError):
        return False


def _catalog_output_key(output: Any, catalog: OptionCatalog) -> str | None:
    try:
        value = _strict_json(output, "generated option output")
        return next(
            (
                entry.key
                for entry in catalog.entries
                if value == _strict_json(
                    catalog.project(entry.key), "catalog output",
                )
            ),
            None,
        )
    except (KeyError, TypeError, ValueError):
        return None


def select_global_verifier_challenger(
    option_support: Any,
    *,
    catalog: OptionCatalog,
    gate: Any,
) -> Any | None:
    """Select a strongly supported full-image verifier option, if one exists."""
    from .config import GlobalGateConfig

    if not isinstance(catalog, OptionCatalog):
        raise TypeError("global verifier challenger requires an OptionCatalog")
    if not isinstance(gate, GlobalGateConfig):
        raise TypeError("global verifier challenger requires a GlobalGateConfig")
    if not isinstance(option_support, Mapping):
        raise TypeError("global option support must be a mapping")
    if type(option_support.get("valid")) is not bool:
        raise TypeError("global option support valid flag must be boolean")
    options = option_support.get("options")
    if not isinstance(options, list) or len(options) != len(catalog.entries):
        raise ValueError("global option support must cover the complete catalog")
    parsed = {}
    for entry, option in zip(catalog.entries, options, strict=True):
        if not isinstance(option, Mapping) or option.get("key") != entry.key:
            raise ValueError("global option support keys do not match the catalog")
        distribution = option.get("distribution")
        if not isinstance(distribution, Mapping) or distribution.get(
            "winner"
        ) not in {"Support", "Refute", "Insufficient"}:
            raise ValueError("global option support distribution is invalid")
        parsed[entry.key] = {
            "raw": _probability(option.get("raw_support"), "raw option support"),
            "normalized": _probability(
                option.get("normalized_support"),
                "normalized option support",
            ),
            "winner": distribution["winner"],
        }
    if not option_support["valid"]:
        return None
    ranked = sorted(
        catalog.entries,
        key=lambda entry: (-parsed[entry.key]["normalized"], entry.key),
    )
    top, runner = ranked[:2]
    top_score = parsed[top.key]
    margin = top_score["normalized"] - parsed[runner.key]["normalized"]
    if (
        top_score["winner"] != "Support"
        or top_score["raw"] < gate.min_support
        or margin < gate.min_margin
    ):
        return None
    return catalog.project(top.key)


def _independent_group_pair(
    groups: Sequence[BranchEvidenceGroup],
) -> tuple[BranchEvidenceGroup, BranchEvidenceGroup] | None:
    for index, first in enumerate(groups):
        for second in groups[index + 1:]:
            if first.render_sha256 == second.render_sha256:
                continue
            if any((
                first.effective_geometry != second.effective_geometry,
                first.rendered_scale != second.rendered_scale,
                first.action != second.action,
            )):
                return first, second
    return None


def _answer_vector_consensus(
    groups: Sequence[BranchEvidenceGroup],
    *,
    state_id: int,
    generated_keys: Mapping[str, str | None],
    catalog: OptionCatalog,
    acceptance: BranchAcceptanceConfig,
    question_kind: str,
    required_roles: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    eligible: dict[str, list[BranchEvidenceGroup]] = defaultdict(list)
    for group in groups:
        key = generated_keys.get(group.group_id)
        if (
            group.shared_global
            or not group.vector_valid
            or not group.generated_output_valid
            or not group.grounding_valid
            or key is None
        ):
            continue
        normalized = dict(group.normalized_support)
        top_key = min(
            catalog.entries,
            key=lambda entry: (-normalized[entry.key], entry.key),
        ).key
        if (
            top_key != key
            or normalized[key] < acceptance.min_view_support
        ):
            continue
        eligible[key].append(group)

    scoped: dict[str, list[BranchEvidenceGroup]] = {}
    for key, values in eligible.items():
        if question_kind != "attribute":
            scoped[key] = values
            continue
        by_signature: dict[tuple[Any, ...], list[BranchEvidenceGroup]] = (
            defaultdict(list)
        )
        for group in values:
            by_signature[_grounding_signature(group, question_kind)].append(group)
        scoped[key] = min(
            by_signature.values(),
            key=lambda items: (
                -len(items),
                tuple(group.group_id for group in items),
            ),
        )

    total = sum(len(values) for values in scoped.values())
    ranked = sorted(
        scoped,
        key=lambda key: (-len(scoped[key]), key),
    )
    top_key = None if not ranked else ranked[0]
    runner_key = None if len(ranked) < 2 else ranked[1]
    top_count = 0 if top_key is None else len(scoped[top_key])
    runner_count = 0 if runner_key is None else len(scoped[runner_key])
    support = 0.0 if not total else top_count / total
    margin = 0.0 if not total else (top_count - runner_count) / total
    winner_groups = [] if top_key is None else scoped[top_key]
    confirmation = _independent_group_pair(winner_groups)
    required = {canonical_text(role) for role in required_roles}
    grounded_roles = {
        canonical_text(role)
        for group in winner_groups for role, _ in group.role_to_instance
    }
    coverage_valid = True
    if question_kind in {"relation", "comparison"}:
        coverage_valid = required.issubset(grounded_roles)
    elif question_kind in {"count", "coverage"}:
        instances = {
            instance_id
            for group in winner_groups
            for role, instance_id in group.role_to_instance
            if canonical_text(role) in required
        } | {
            instance_id
            for group in winner_groups
            for instance_id in group.covered_instance_ids
        }
        coverage_valid = required.issubset(grounded_roles) and len(instances) >= 2
    accepted = (
        top_key is not None
        and top_count >= 2
        and top_count > runner_count
        and support >= acceptance.min_absolute_support
        and margin >= acceptance.min_normalized_margin
        and confirmation is not None
        and coverage_valid
    )
    vector = tuple(
        (
            entry.key,
            0.0 if not total else len(scoped.get(entry.key, ())) / total,
        )
        for entry in catalog.entries
    )
    evaluation = {
        "state_id": state_id,
        "included_groups": [
            {"group_id": group.group_id, "path": list(group.path)}
            for values in scoped.values() for group in values
        ],
        "excluded_groups": [
            {
                "group_id": group.group_id,
                "path": list(group.path),
                "reason": "not_grounded_generator_vote",
            }
            for group in groups
            if all(group not in values for values in scoped.values())
        ],
        "raw_support_vector": dict(vector),
        "normalized_support_vector": dict(vector),
        "top_key": top_key,
        "runner_up_key": runner_key,
        "margin": margin,
        "diverse_confirm": confirmation is not None,
        "role_coverage": coverage_valid,
        "score_source": acceptance.aggregation,
        "accepted": accepted,
    }
    if not accepted or top_key is None or confirmation is None:
        return evaluation, None
    entry = next(item for item in catalog.entries if item.key == top_key)
    signature = tuple(sorted({
        (role, instance_id)
        for group in winner_groups for role, instance_id in group.role_to_instance
    }))
    return evaluation, {
        "entry": entry,
        "support": support,
        "margin": margin,
        "winner_groups": winner_groups,
        "confirmation": confirmation,
        "signature": signature,
        "vector": vector,
    }


def _bundle_evaluation(
    raw_record: Mapping[str, Any],
    *,
    ledger: Mapping[str, BranchEvidenceGroup],
    catalog: OptionCatalog,
    root_state_id: int,
    question_kind: str,
    required_roles: tuple[str, ...],
    min_view_support: float,
) -> dict[str, Any] | None:
    raw_bundle = raw_record.get("evidence_bundle")
    if raw_bundle is None:
        return None
    if not isinstance(raw_bundle, Mapping):
        raise TypeError("evidence_bundle must be a mapping")
    required_bundle_keys = {
        "plan", "answer", "option_support", "render_sha256", "view_size",
        "model_calls", "processed_pixels", "source",
    }
    if set(raw_bundle) != required_bundle_keys:
        raise ValueError("evidence_bundle has an invalid exact schema")
    if raw_bundle["source"] != "joint_bundle_verification":
        raise ValueError("multi-target support must come from joint bundle verification")
    _sha256(raw_bundle["render_sha256"], "bundle render digest")
    view_size = raw_bundle["view_size"]
    if (
        not isinstance(view_size, list) or len(view_size) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
               for item in view_size)
    ):
        raise ValueError("bundle view size must contain two positive integers")
    for name in ("model_calls", "processed_pixels"):
        value = raw_bundle[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"bundle {name} must be a positive integer")
    answer = raw_bundle["answer"]
    if not isinstance(answer, Mapping) or "output" not in answer:
        raise TypeError("bundle answer must contain an output")

    plan = raw_bundle["plan"]
    if not isinstance(plan, Mapping):
        raise TypeError("bundle plan must be a mapping")
    expected_plan_keys = {
        "question_kind", "required_roles", "constituent_state_ids",
        "role_to_instance", "covered_instance_ids", "coverage_context",
        "valid", "reason",
    }
    if set(plan) != expected_plan_keys:
        raise ValueError("bundle plan has an invalid exact schema")
    if plan["question_kind"] != question_kind:
        raise ValueError("bundle question kind does not match selection")
    roles = _string_tuple(plan["required_roles"], "bundle required roles", nonempty=True)
    if tuple(map(canonical_text, roles)) != tuple(map(canonical_text, required_roles)):
        raise ValueError("bundle required roles do not match the Query plan")
    state_ids = plan["constituent_state_ids"]
    if (
        not isinstance(state_ids, list) or len(state_ids) < 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0
               for item in state_ids)
        or len(state_ids) != len(set(state_ids))
    ):
        raise ValueError("bundle needs at least two unique constituent states")
    if type(plan["coverage_context"]) is not bool or type(plan["valid"]) is not bool:
        raise TypeError("bundle plan validity flags must be boolean")
    if not isinstance(plan["reason"], str) or not plan["reason"]:
        raise ValueError("bundle plan reason must be nonempty")

    by_state = {group.state_id: group for group in ledger.values()}
    if len(by_state) != len(ledger):
        raise ValueError("bundle ledger state identifiers must be unique")
    constituents = []
    missing = []
    for state_id in state_ids:
        group = by_state.get(state_id)
        if group is None or group.shared_global:
            missing.append(state_id)
        else:
            constituents.append(group)

    claimed_mapping = plan["role_to_instance"]
    if not isinstance(claimed_mapping, list) or any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("role"), str) or not item.get("role")
        or not isinstance(item.get("instance_id"), str) or not item.get("instance_id")
        for item in claimed_mapping
    ):
        raise ValueError("bundle role-to-instance mapping is invalid")
    claimed_pairs = {
        (canonical_text(item["role"]), item["instance_id"])
        for item in claimed_mapping
    }
    covered = plan["covered_instance_ids"]
    if not isinstance(covered, list) or any(
        not isinstance(item, str) or not item for item in covered
    ) or len(covered) != len(set(covered)):
        raise ValueError("bundle covered instances are invalid")
    observed_pairs = {
        (canonical_text(role), instance_id)
        for group in constituents for role, instance_id in group.role_to_instance
    }
    claims_grounded = claimed_pairs.issubset(observed_pairs)

    synthetic = dict(raw_record)
    synthetic["option_support"] = raw_bundle["option_support"]
    bundle_vector = _branch_record(
        synthetic, root_state_id=root_state_id, catalog=catalog,
    )
    raw_support = dict(bundle_vector.raw_support)
    normalized_support = dict(bundle_vector.normalized_support)
    ranked = sorted(
        catalog.entries,
        key=lambda entry: (-normalized_support[entry.key], entry.key),
    )
    top, runner = ranked[0], ranked[1]

    eligible = [
        group for group in constituents
        if group.vector_valid
        and group.generated_output_valid
        and group.grounding_valid
        and dict(group.winning_labels)[top.key] == "Support"
        and dict(group.raw_support)[top.key] >= min_view_support
        and bool(group.role_to_instance)
    ]
    required = {canonical_text(role) for role in required_roles}
    grounded_roles = {
        canonical_text(role)
        for group in eligible for role, _ in group.role_to_instance
    }
    render_distinct = len({group.render_sha256 for group in eligible}) >= 2
    view_distinct = any(
        first.effective_geometry != second.effective_geometry
        or first.rendered_scale != second.rendered_scale
        or first.action != second.action
        for index, first in enumerate(eligible)
        for second in eligible[index + 1:]
    )
    if question_kind in {"relation", "comparison"}:
        confirmation_valid = (
            required.issubset(grounded_roles)
            and {role for role, _ in claimed_pairs}.issuperset(required)
            and claims_grounded
            and len(eligible) >= 2 and render_distinct and view_distinct
        )
    else:
        instances = {
            instance_id
            for group in eligible
            for role, instance_id in group.role_to_instance
            if canonical_text(role) in required
        }
        confirmation_valid = (
            len(instances) >= 2
            and len(set(covered)) >= 2
            and set(covered).issubset({
                instance_id
                for group in constituents
                for instance_id in group.covered_instance_ids
                + tuple(value for _, value in group.role_to_instance)
            })
            and claims_grounded
            and plan["coverage_context"]
            and len(eligible) >= 2 and render_distinct and view_distinct
        )
    recomputed_valid = (
        not missing
        and plan["valid"]
        and plan["reason"] == "complete"
        and bundle_vector.vector_valid
        and _catalog_output_valid(answer["output"], catalog)
        and confirmation_valid
    )
    bundle_id = hashlib.sha256(_strict_json({
        "plan": plan,
        "render_sha256": raw_bundle["render_sha256"],
        "catalog_sha256": catalog.identity_sha256,
    }, "bundle identity").encode("utf-8")).hexdigest()
    return {
        "valid": recomputed_valid,
        "bundle_id": bundle_id,
        "constituents": constituents,
        "confirmation_groups": eligible,
        "raw_vector": tuple((entry.key, raw_support[entry.key]) for entry in catalog.entries),
        "normalized_vector": tuple(
            (entry.key, normalized_support[entry.key]) for entry in catalog.entries
        ),
        "top": top,
        "runner": runner,
        "diverse_confirm": confirmation_valid,
        "grounding_signature": tuple(
            (role, instance_id)
            for group in eligible for role, instance_id in group.role_to_instance
        ),
        "missing_constituent_state_ids": tuple(missing),
    }


def _select_accepted_hypothesis_v3(
    records: Sequence[Mapping[str, Any]],
    *,
    root_state_id: int,
    acceptance: BranchAcceptanceConfig,
    validation_enabled: bool,
    root_fallback_answer: Any,
    option_catalog: OptionCatalog,
    question_kind: str,
    required_roles: tuple[str, ...],
    negative_coverage: NegativeCoverage,
    negative_coverage_for_records: Callable[
        [Sequence[Mapping[str, Any]]], NegativeCoverage
    ] | None,
    strict_no: StrictNoConfig,
) -> FinalDecision:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise ValueError("evidence records must be a nonempty sequence")
    if isinstance(root_state_id, bool) or not isinstance(root_state_id, int) or root_state_id < 0:
        raise ValueError("root_state_id must be a non-negative integer")
    if type(validation_enabled) is not bool:
        raise TypeError("validation_enabled must be boolean")
    if not isinstance(option_catalog, OptionCatalog):
        raise TypeError("Revision-3 selection requires an OptionCatalog")
    if question_kind not in {
        "attribute", "relation", "comparison", "count", "coverage",
    }:
        raise ValueError("question_kind is unsupported")
    if question_kind == "attribute":
        if required_roles and any(not isinstance(item, str) for item in required_roles):
            raise TypeError("required_roles must contain text")
    else:
        required_roles = _string_tuple(
            list(required_roles), "required_roles", nonempty=True,
        )
    fallback = json.loads(_strict_json(root_fallback_answer, "root fallback answer"))
    parsed = sorted(
        (
            (
                _branch_record(
                    value, root_state_id=root_state_id, catalog=option_catalog,
                ),
                value,
            )
            for value in records
        ),
        key=lambda item: item[0].state_id,
    )
    normalized = [item[0] for item in parsed]
    if sum(item.shared_global for item in normalized) != 1:
        raise ValueError("Revision-3 records must contain exactly one shared global view")
    if len({item.state_id for item in normalized}) != len(normalized):
        raise ValueError("Revision-3 record state identifiers must be unique")

    aggregation = {
        "global_verifier_then_branch_equal_mean": "branch_equal_mean",
        "global_verifier_then_grounded_answer_consensus": (
            "grounded_answer_consensus"
        ),
    }.get(acceptance.aggregation, acceptance.aggregation)

    ledger: dict[str, BranchEvidenceGroup] = {}
    generated_keys: dict[str, str | None] = {}
    events = []
    evaluations = []
    accepted: BranchAcceptedHypothesis | None = None
    observed_records = []
    negative_gate_trace = None

    def passes_negative_gate(answer: Any, evaluation: dict[str, Any]) -> bool:
        nonlocal negative_gate_trace
        if option_catalog.answer_type != "yes_no" or answer != "no":
            return True
        if negative_coverage_for_records is not None:
            coverage = negative_coverage_for_records(observed_records)
        elif len(observed_records) == len(parsed):
            coverage = negative_coverage
        else:
            # An aggregate supplied by a caller describes only the final state;
            # later observations cannot establish an earlier No decision.
            coverage = NegativeCoverage(False, 0.0)
        gate = evaluate_negative_gate(
            observed_records, coverage=coverage, config=strict_no,
        )
        negative_gate_trace = gate.to_dict()
        evaluation["negative_gate"] = negative_gate_trace
        evaluation["accepted"] = gate.accepted
        return gate.accepted

    for current, raw_current in parsed:
        observed_records.append(raw_current)
        replaced = ledger.get(current.group_id)
        ledger[current.group_id] = current
        answer = raw_current.get("answer")
        generated_keys[current.group_id] = _catalog_output_key(
            answer.get("output") if isinstance(answer, Mapping) else None,
            option_catalog,
        )
        events.append({
            "state_id": current.state_id,
            "decision": "retained" if replaced is None else "replaced_duplicate",
            "group_id": current.group_id,
            "replaced_state_id": None if replaced is None else replaced.state_id,
        })
        included = []
        excluded = []
        for group in sorted(ledger.values(), key=lambda item: item.state_id):
            if not group.vector_valid:
                excluded.append((group, "invalid_option_vector"))
                continue
            if group.shared_global and not group.positive_support_eligible:
                excluded.append((group, "global_not_accepted_as_positive_support"))
                continue
            if group.shared_global:
                included.append(group)
                continue
            if not group.generated_output_valid:
                excluded.append((group, "generated_output_invalid"))
                continue
            if not group.grounding_valid:
                excluded.append((group, "query_grounding_invalid"))
                continue
            if not _prefix(group.path, current.path):
                excluded.append((group, "sibling_or_descendant_branch"))
                continue
            if not set(group.context_anchor_ids).issubset(current.context_anchor_ids):
                excluded.append((group, "context_anchor_not_active"))
                continue
            included.append(group)
        score_groups = list(included)
        if aggregation == "grounded_answer_consensus":
            consensus_groups = (
                score_groups
                if question_kind == "attribute"
                else sorted(ledger.values(), key=lambda item: item.state_id)
            )
            evaluation, candidate = _answer_vector_consensus(
                consensus_groups,
                state_id=current.state_id,
                generated_keys=generated_keys,
                catalog=option_catalog,
                acceptance=acceptance,
                question_kind=question_kind,
                required_roles=required_roles,
            )
            evaluations.append(evaluation)
            if candidate is None:
                continue
            first, second = candidate["confirmation"]
            entry = candidate["entry"]
            if not passes_negative_gate(option_catalog.project(entry.key), evaluation):
                continue
            accepted = BranchAcceptedHypothesis(
                answer=option_catalog.project(entry.key),
                canonical_answer=entry.text,
                absolute_support=candidate["support"],
                normalized_support=candidate["support"],
                margin=candidate["margin"],
                evidence_group_ids=tuple(
                    group.group_id for group in candidate["winner_groups"]
                ),
                confirmation_group_ids=(first.group_id, second.group_id),
                grounding_signature=candidate["signature"],
                raw_support_vector=candidate["vector"],
                normalized_support_vector=candidate["vector"],
                accepted_at_state_id=current.state_id,
                score_source=acceptance.aggregation,
            )
            break
        if question_kind != "attribute":
            bundle = _bundle_evaluation(
                raw_current,
                ledger=ledger,
                catalog=option_catalog,
                root_state_id=root_state_id,
                question_kind=question_kind,
                required_roles=required_roles,
                min_view_support=acceptance.min_view_support,
            )
            if bundle is None:
                evaluations.append({
                    "state_id": current.state_id,
                    "included_groups": [],
                    "excluded_groups": [
                        {
                            "group_id": group.group_id,
                            "path": list(group.path),
                            "reason": "not_in_complete_query_bundle",
                        }
                        for group in sorted(
                            ledger.values(), key=lambda item: item.state_id,
                        )
                    ],
                    "raw_support_vector": {},
                    "normalized_support_vector": {},
                    "top_key": None,
                    "runner_up_key": None,
                    "margin": 0.0,
                    "diverse_confirm": False,
                    "bundle_valid": False,
                    "bundle_id": None,
                    "score_source": "no_complete_bundle",
                    "accepted": False,
                })
                continue
            raw_vector = bundle["raw_vector"]
            normalized_vector = bundle["normalized_vector"]
            top = bundle["top"]
            runner = bundle["runner"]
            top_absolute = dict(raw_vector)[top.key]
            top_normalized = dict(normalized_vector)[top.key]
            margin = top_normalized - dict(normalized_vector)[runner.key]
            gate = (
                bundle["valid"]
                and top_absolute >= acceptance.min_absolute_support
                and margin >= acceptance.min_normalized_margin
            )
            constituents = bundle["constituents"]
            confirmation_groups = bundle["confirmation_groups"]
            evaluations.append({
                "state_id": current.state_id,
                "included_groups": [
                    {"group_id": group.group_id, "path": list(group.path)}
                    for group in constituents
                ],
                "excluded_groups": [
                    {
                        "group_id": group.group_id,
                        "path": list(group.path),
                        "reason": "not_in_query_grounded_bundle",
                    }
                    for group in sorted(ledger.values(), key=lambda item: item.state_id)
                    if group not in constituents
                ],
                "raw_support_vector": dict(raw_vector),
                "normalized_support_vector": dict(normalized_vector),
                "top_key": top.key,
                "runner_up_key": runner.key,
                "margin": margin,
                "diverse_confirm": bundle["diverse_confirm"],
                "bundle_valid": bundle["valid"],
                "bundle_id": bundle["bundle_id"],
                "score_source": "joint_bundle_verification",
                "accepted": gate,
            })
            if not gate:
                continue
            if not passes_negative_gate(option_catalog.project(top.key), evaluations[-1]):
                continue
            accepted = BranchAcceptedHypothesis(
                answer=option_catalog.project(top.key),
                canonical_answer=top.text,
                absolute_support=top_absolute,
                normalized_support=top_normalized,
                margin=margin,
                evidence_group_ids=tuple(
                    group.group_id for group in constituents
                ) + (bundle["bundle_id"],),
                confirmation_group_ids=(
                    confirmation_groups[0].group_id,
                    confirmation_groups[1].group_id,
                ),
                grounding_signature=bundle["grounding_signature"],
                raw_support_vector=raw_vector,
                normalized_support_vector=normalized_vector,
                accepted_at_state_id=current.state_id,
                score_source="joint_bundle_verification",
                bundle_id=bundle["bundle_id"],
            )
            break
        if not score_groups:
            evaluations.append({
                "state_id": current.state_id,
                "included_groups": [],
                "excluded_groups": [
                    {
                        "group_id": group.group_id,
                        "path": list(group.path),
                        "reason": reason,
                    }
                    for group, reason in excluded
                ],
                "raw_support_vector": {},
                "normalized_support_vector": {},
                "top_key": None,
                "runner_up_key": None,
                "margin": 0.0,
                "diverse_confirm": False,
                "accepted": False,
            })
            continue
        raw_vector = tuple(
            (
                entry.key,
                math.fsum(dict(group.raw_support)[entry.key] for group in score_groups)
                / len(score_groups),
            )
            for entry in option_catalog.entries
        )
        normalized_vector = tuple(
            (
                entry.key,
                math.fsum(
                    dict(group.normalized_support)[entry.key]
                    for group in score_groups
                ) / len(score_groups),
            )
            for entry in option_catalog.entries
        )
        ranked = sorted(
            option_catalog.entries,
            key=lambda entry: (-dict(normalized_vector)[entry.key], entry.key),
        )
        top, runner = ranked[0], ranked[1]
        top_absolute = dict(raw_vector)[top.key]
        top_normalized = dict(normalized_vector)[top.key]
        margin = top_normalized - dict(normalized_vector)[runner.key]
        confirmation = _confirmation_pair(
            score_groups,
            top_key=top.key,
            min_view_support=acceptance.min_view_support,
            question_kind=question_kind,
        )
        gate = (
            current.vector_valid
            and current.generated_output_valid
            and top_absolute >= acceptance.min_absolute_support
            and margin >= acceptance.min_normalized_margin
            and confirmation is not None
        )
        evaluation = {
            "state_id": current.state_id,
            "included_groups": [
                {"group_id": group.group_id, "path": list(group.path)}
                for group in included
            ],
            "excluded_groups": [
                {
                    "group_id": group.group_id,
                    "path": list(group.path),
                    "reason": reason,
                }
                for group, reason in excluded
            ],
            "raw_support_vector": dict(raw_vector),
            "normalized_support_vector": dict(normalized_vector),
            "top_key": top.key,
            "runner_up_key": runner.key,
            "margin": margin,
            "diverse_confirm": confirmation is not None,
            "accepted": gate,
        }
        evaluations.append(evaluation)
        if not gate:
            continue
        if not passes_negative_gate(option_catalog.project(top.key), evaluation):
            continue
        first, second = confirmation
        signature = _grounding_signature(first, question_kind)
        accepted = BranchAcceptedHypothesis(
            answer=option_catalog.project(top.key),
            canonical_answer=top.text,
            absolute_support=top_absolute,
            normalized_support=top_normalized,
            margin=margin,
            evidence_group_ids=tuple(group.group_id for group in included),
            confirmation_group_ids=(first.group_id, second.group_id),
            grounding_signature=signature,
            raw_support_vector=raw_vector,
            normalized_support_vector=normalized_vector,
            accepted_at_state_id=current.state_id,
        )
        break

    ledger_trace = {
        "schema_version": 3,
        "aggregation": acceptance.aggregation,
        "root_state_id": root_state_id,
        "events": events,
        "groups": [
            group.to_dict()
            for group in sorted(ledger.values(), key=lambda item: item.state_id)
        ],
        "evaluations": evaluations,
    }
    if negative_gate_trace is not None:
        ledger_trace["negative_gate"] = negative_gate_trace
    if accepted is None:
        return FinalDecision(fallback, "full_image_fallback", None, ledger_trace)
    return FinalDecision(
        accepted.answer,
        "accepted_local_hypothesis",
        accepted,
        ledger_trace,
    )


def select_accepted_hypothesis(
    records: Sequence[Mapping[str, Any]],
    *,
    root_state_id: int,
    acceptance: BranchAcceptanceConfig,
    validation_enabled: bool,
    root_fallback_answer: Any,
    option_catalog: OptionCatalog | None = None,
    question_kind: str = "attribute",
    required_roles: tuple[str, ...] = (),
    negative_coverage: NegativeCoverage = NegativeCoverage(False, 0.0),
    negative_coverage_for_records: Callable[
        [Sequence[Mapping[str, Any]]], NegativeCoverage
    ] | None = None,
    strict_no: StrictNoConfig | None = None,
) -> FinalDecision:
    """Select an answer using each history prefix's evidence and coverage.

    The optional callback recomputes coverage at each candidate decision.
    Without it, aggregate coverage applies only to the final recorded state.
    """
    if isinstance(acceptance, BranchAcceptanceConfig):
        if option_catalog is None:
            raise TypeError("Revision-3 selection requires option_catalog")
        return _select_accepted_hypothesis_v3(
            records,
            root_state_id=root_state_id,
            acceptance=acceptance,
            validation_enabled=validation_enabled,
            root_fallback_answer=root_fallback_answer,
            option_catalog=option_catalog,
            question_kind=question_kind,
            required_roles=required_roles,
            negative_coverage=negative_coverage,
            negative_coverage_for_records=negative_coverage_for_records,
            strict_no=strict_no or StrictNoConfig(),
        )
    raise TypeError("acceptance must be BranchAcceptanceConfig")
