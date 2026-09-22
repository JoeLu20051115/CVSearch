import copy
import hashlib
from types import SimpleNamespace

import pytest
from PIL import Image


def _config():
    return {
        "schema_version": 3,
        "method": "independent_query_aware_v3",
        "profile": "full",
        "direct_threshold": 0.8,
        "ranking": {
            "alpha": 0.6,
            "beta": 0.5,
            "visual_lambda": 0.5,
            "top_k_augmented": 3,
            "main_query": True,
            "augmented_query": True,
            "complexity": True,
            "edge_density": True,
        },
        "modules": {
            "query_ranking": True,
            "adaptive_observation": True,
            "evidence_validation": True,
        },
        "budget": {
            "max_steps": 8,
            "max_model_calls": 96,
            "max_processed_pixels": 600_000_000,
        },
        "controller": {
            "stall_patience": 2,
            "min_progress": 0.02,
            "max_gap_for_stop": 0.2,
            "min_answer_confidence": 0.65,
            "min_support_avg": 0.7,
            "min_support_min": 0.5,
        },
        "global_gate": {
            "min_support": 0.80,
            "min_margin": 0.20,
            "min_consistency": 0.75,
        },
        "proposals": {
            "dedup_iou": 0.90,
            "min_count": 1,
            "min_spatial_coverage": 0.005,
            "active_top_k": 4,
        },
        "acceptance": {
            "min_absolute_support": 0.65,
            "min_normalized_margin": 0.15,
            "min_view_support": 0.60,
            "aggregation": "branch_equal_mean",
        },
        "verification": {
            "labels": ["Support", "Refute", "Insufficient"],
            "grounding_labels": ["Grounded", "NotGrounded", "Insufficient"],
            "grounding_threshold": 0.65,
            "target_instance_iou": 0.5,
        },
        "observation_geometry": {
            "min_zoom_factor": 0.4,
            "context_max_normalized_gap": 0.25,
        },
    }


def _acceptance_v3(**changes):
    from qavs.independent_search.config import BranchAcceptanceConfig

    values = {
        "min_absolute_support": 0.60,
        "min_normalized_margin": 0.20,
        "min_view_support": 0.70,
        "aggregation": "branch_equal_mean",
    }
    values.update(changes)
    return BranchAcceptanceConfig(**values)


def _v3_catalog():
    from qavs.independent_search.semantics import build_option_catalog

    return build_option_catalog({
        "answer_type": "logits_match", "options": ["red", "blue"],
    })


def _v3_vector(catalog, red, blue, *, red_winner="Support", blue_winner="Refute"):
    values = (("index:0", red, red_winner), ("index:1", blue, blue_winner))
    return {
        "catalog_sha256": catalog.identity_sha256,
        "options": [
            {
                "key": key,
                "text_sha256": hashlib.sha256(
                    catalog.entries[index].text.encode("utf-8")
                ).hexdigest(),
                "distribution": {
                    "labels": ["Support", "Refute", "Insufficient"],
                    "losses": [0.0, 1.0, 2.0],
                    "probabilities": (
                        [support, 1.0 - support, 0.0]
                        if winner == "Support" else
                        [support, 1.0 - support, 0.0]
                    ),
                    "winner": winner,
                },
                "raw_support": support,
                "normalized_support": support / (red + blue),
            }
            for index, (key, support, winner) in enumerate(values)
        ],
        "valid": True,
        "model_calls": 8,
        "processed_pixels": 800,
        "checkpoint_sha256": "b" * 64,
        "prompt_template_sha256": "e" * 64,
    }


def _v3_record(
    state_id, path, catalog, red, blue, *, instance="hat-1",
    geometry=(0, 0, 40, 40), scale=1.0, action="NEXT", render=None,
    red_winner="Support", blue_winner="Refute", root=False,
):
    record = {
        "state": {
            "state_id": state_id,
            "path_keys": list(path),
            "focus_keys": [path[-1]],
            "context_keys": [],
            "observation_keys": [f"{path[-1]}@base"],
            "source_image_identity": {
                "mode": "RGB", "size": [100, 100],
                "pixel_sha256": "a" * 64,
            },
            "effective_geometry": list(geometry),
            "rendered_scale": scale,
            "branch_context": list(path),
            "action": "GLOBAL" if root else action,
            "render_sha256": render or f"{state_id:064x}",
        },
        "answer": {
            "output": 0 if red >= blue else 1,
            "canonical_answer": 0 if red >= blue else 1,
            "frequency": 1.0,
            "aggregation_available": True,
        },
        "option_support": _v3_vector(
            catalog, red, blue,
            red_winner=red_winner, blue_winner=blue_winner,
        ),
        "assessment": {"uncertainty": 0.1},
    }
    if not root:
        record["grounding"] = {
            "verification": {"valid": True},
            "record": {
                "role_distributions": [],
                "role_to_instance": [{
                    "role": "hat", "instance_id": instance,
                }],
                "covered_instance_ids": [instance],
                "valid": True,
            },
        }
    return record


def _attach_v3_bundle(
    record, catalog, *, question_kind, required_roles,
    constituent_state_ids, mappings, covered, coverage_context=True,
    red=0.95, blue=0.05,
):
    record["evidence_bundle"] = {
        "plan": {
            "question_kind": question_kind,
            "required_roles": list(required_roles),
            "constituent_state_ids": list(constituent_state_ids),
            "role_to_instance": [
                {"role": role, "instance_id": instance}
                for role, instance in mappings
            ],
            "covered_instance_ids": list(covered),
            "coverage_context": coverage_context,
            "valid": True,
            "reason": "complete",
        },
        "answer": {"output": 0, "frequency": 1.0},
        "option_support": _v3_vector(catalog, red, blue),
        "render_sha256": "d" * 64,
        "view_size": [128, 128],
        "model_calls": 12,
        "processed_pixels": 196608,
        "source": "joint_bundle_verification",
    }
    return record


def test_v3_config_accepts_grounded_answer_consensus_aggregation():
    from qavs.independent_search.config import BranchAcceptanceConfig

    parsed = BranchAcceptanceConfig.from_mapping({
        "min_absolute_support": 0.75,
        "min_normalized_margin": 0.5,
        "min_view_support": 0.25,
        "aggregation": "grounded_answer_consensus",
    })

    assert parsed.aggregation == "grounded_answer_consensus"
    hybrid = BranchAcceptanceConfig.from_mapping({
        **parsed.to_dict(),
        "aggregation": "global_verifier_then_branch_equal_mean",
        "global_verifier_min_support": 0.69,
        "global_verifier_min_margin": 0.03,
    })
    assert hybrid.aggregation == "global_verifier_then_branch_equal_mean"
    assert hybrid.global_verifier_min_support == 0.69
    grounded_hybrid = BranchAcceptanceConfig.from_mapping({
        **parsed.to_dict(),
        "aggregation": "global_verifier_then_grounded_answer_consensus",
        "global_verifier_min_support": 0.70,
        "global_verifier_min_margin": 0.0,
    })
    assert (
        grounded_hybrid.aggregation
        == "global_verifier_then_grounded_answer_consensus"
    )


def test_v3_global_verifier_challenger_requires_support_and_margin():
    from qavs.independent_search.config import GlobalGateConfig
    from qavs.independent_search.decision import (
        select_global_verifier_challenger,
    )

    catalog = _v3_catalog()
    vector = _v3_vector(catalog, 0.75, 0.25)
    gate = GlobalGateConfig(
        min_support=0.69, min_margin=0.03, min_consistency=0.0,
    )

    assert select_global_verifier_challenger(
        vector, catalog=catalog, gate=gate,
    ) == 0
    vector["options"][0]["distribution"]["winner"] = "Refute"
    assert select_global_verifier_challenger(
        vector, catalog=catalog, gate=gate,
    ) is None
    vector["options"][0]["distribution"]["winner"] = "Support"
    strict_gate = GlobalGateConfig(
        min_support=0.80, min_margin=0.03, min_consistency=0.0,
    )
    assert select_global_verifier_challenger(
        vector, catalog=catalog, gate=strict_gate,
    ) is None


def test_v3_grounded_answer_consensus_recovers_cross_role_agreement():
    from qavs.independent_search.decision import select_accepted_hypothesis

    catalog = _v3_catalog()
    root = _v3_record(0, ("root",), catalog, 0.5, 0.5, root=True)
    man = _v3_record(
        1, ("root", "left"), catalog, 0.6, 0.4,
        instance="man-1", geometry=(0, 0, 40, 80), action="NEXT",
        red_winner="Refute", blue_winner="Refute",
    )
    man["grounding"]["record"]["role_to_instance"] = [
        {"role": "man", "instance_id": "man-1"},
    ]
    bicycle = _v3_record(
        2, ("root", "right"), catalog, 0.6, 0.4,
        instance="bike-1", geometry=(60, 0, 40, 80), action="NEXT",
        red_winner="Refute", blue_winner="Refute",
    )
    bicycle["grounding"]["record"]["role_to_instance"] = [
        {"role": "bicycle", "instance_id": "bike-1"},
    ]

    result = select_accepted_hypothesis(
        [root, man, bicycle], root_state_id=0,
        acceptance=_acceptance_v3(
            aggregation="grounded_answer_consensus",
            min_absolute_support=0.75,
            min_normalized_margin=0.5,
            min_view_support=0.25,
        ),
        validation_enabled=True, root_fallback_answer=1,
        option_catalog=catalog, question_kind="relation",
        required_roles=("man", "bicycle"),
    )

    assert result.answer == 0
    assert result.accepted_hypothesis.score_source == (
        "grounded_answer_consensus"
    )
    assert result.ledger_trace["evaluations"][-1]["accepted"] is True


def test_v3_grounded_answer_consensus_requires_complete_role_coverage():
    from qavs.independent_search.decision import select_accepted_hypothesis

    catalog = _v3_catalog()
    records = [
        _v3_record(0, ("root",), catalog, 0.5, 0.5, root=True),
        _v3_record(
            1, ("root", "left"), catalog, 0.6, 0.4,
            instance="man-1", geometry=(0, 0, 40, 80), action="NEXT",
            red_winner="Refute", blue_winner="Refute",
        ),
        _v3_record(
            2, ("root", "right"), catalog, 0.6, 0.4,
            instance="man-2", geometry=(60, 0, 40, 80), action="NEXT",
            red_winner="Refute", blue_winner="Refute",
        ),
    ]
    for record in records[1:]:
        record["grounding"]["record"]["role_to_instance"] = [{
            "role": "man", "instance_id": record["grounding"]["record"][
                "covered_instance_ids"
            ][0],
        }]

    result = select_accepted_hypothesis(
        records, root_state_id=0,
        acceptance=_acceptance_v3(
            aggregation="grounded_answer_consensus",
            min_absolute_support=0.75,
            min_normalized_margin=0.5,
            min_view_support=0.25,
        ),
        validation_enabled=True, root_fallback_answer=1,
        option_catalog=catalog, question_kind="relation",
        required_roles=("man", "bicycle"),
    )

    assert result.answer == 1
    assert result.accepted_hypothesis is None
    assert result.ledger_trace["evaluations"][-1]["accepted"] is False


def test_v3_grounded_answer_consensus_empty_local_set_falls_back():
    from qavs.independent_search.decision import select_accepted_hypothesis

    catalog = _v3_catalog()
    root = _v3_record(0, ("root",), catalog, 0.5, 0.5, root=True)
    root["state"]["positive_support_eligible"] = False

    result = select_accepted_hypothesis(
        [root], root_state_id=0,
        acceptance=_acceptance_v3(
            aggregation="grounded_answer_consensus",
        ),
        validation_enabled=True, root_fallback_answer=1,
        option_catalog=catalog, question_kind="attribute",
    )

    assert result.answer == 1
    assert result.accepted_hypothesis is None
    assert result.ledger_trace["evaluations"][-1]["accepted"] is False


def test_full_config_requires_all_three_owned_modules():
    from qavs.independent_search.config import IndependentSearchConfig

    parsed = IndependentSearchConfig.from_mapping(_config())
    assert parsed.method == "independent_query_aware_v3"
    assert parsed.direct_threshold == 0.8
    assert parsed.to_dict() == _config()
    for name in _config()["modules"]:
        value = _config()
        value["modules"][name] = False
        with pytest.raises(ValueError, match=name):
            IndependentSearchConfig.from_mapping(value)


def test_public_config_rejects_ablation_profile():
    from qavs.independent_search.config import IndependentSearchConfig
    import json
    from pathlib import Path
    value = json.loads(Path("qavs/defaults.json").read_text())
    value["profile"] = "ablation"
    with pytest.raises(ValueError, match="profile must be full"):
        IndependentSearchConfig.from_mapping(value)


def _global_observation(*, sufficient, support, margin, consistency):
    from qavs.independent_search.decision import GlobalCost, GlobalObservation

    return GlobalObservation(
        output="A",
        sufficient_score=0.9 if sufficient else 0.7,
        sufficient=sufficient,
        support=support,
        margin=margin,
        consistency=consistency,
        valid=True,
        accepted=(support >= 0.8 and margin >= 0.2 and consistency >= 0.75),
        answer_record=None,
        verifier_provenance=None,
        cost=GlobalCost(model_calls=6, processed_pixels=72),
    )


def test_sufficient_but_unaccepted_global_enters_search():
    observation = _global_observation(
        sufficient=True, support=0.60, margin=0.40, consistency=1.0,
    )
    assert observation.sufficient is True
    assert observation.accepted is False
    assert observation.direct_stop is False


def test_direct_stop_requires_both_global_decisions():
    accepted = _global_observation(
        sufficient=True, support=0.90, margin=0.30, consistency=1.0,
    )
    insufficient = _global_observation(
        sufficient=False, support=0.90, margin=0.30, consistency=1.0,
    )
    assert accepted.direct_stop is True
    assert insufficient.accepted is True
    assert insufficient.direct_stop is False


def test_shared_global_gate_recomputes_sufficiency_and_acceptance():
    from qavs.independent_search.config import GlobalGateConfig
    from qavs.independent_search.decision import evaluate_global_gate

    gate = evaluate_global_gate(
        sufficient_score=0.85,
        direct_threshold=0.8,
        support=0.79,
        margin=0.4,
        consistency=1.0,
        valid=True,
        gate=GlobalGateConfig(0.8, 0.2, 0.75),
    )

    assert gate.sufficient is True
    assert gate.accepted is False
    assert gate.direct_stop is False


def test_observe_global_saves_one_immutable_prediction_and_validates_it(monkeypatch):
    from qavs.independent_search.config import GlobalGateConfig
    from qavs.independent_search import decision

    output = {"answer": ["A"]}
    answer = SimpleNamespace(
        output=output, margin=0.30, frequency=1.0, aggregation_available=True,
    )
    answer.to_dict = lambda: {"output": answer.output}
    support = SimpleNamespace(
        support_avg=0.90, independent=True, fallback_used=False, model_calls=2,
    )
    support.to_dict = lambda: {"support_avg": support.support_avg}
    answer_calls = []
    verifier_calls = []

    class Model:
        def get_confidence_value(self, nodes, image, confidence_type, input_ele):
            assert len(nodes) == 1 and nodes[0].is_root
            assert confidence_type == "answering"
            assert input_ele == "Which option?"
            return 0.9

    monkeypatch.setattr(
        decision, "answer_with_uncertainty",
        lambda *args: answer_calls.append(args) or answer,
    )
    monkeypatch.setattr(
        decision, "build_query_plan",
        lambda policy, targets: SimpleNamespace(
            evidence_items=({"kind": "question_evidence", "requirement": "visual_detail"},),
        ),
    )
    monkeypatch.setattr(decision, "sanitize_evidence_requirements", lambda items: ("requirement",))
    monkeypatch.setattr(decision, "proposed_answer_text", lambda *args: "A")
    monkeypatch.setattr(
        decision, "verify_answer_support",
        lambda **kwargs: verifier_calls.append(kwargs) or support,
    )
    monkeypatch.setattr(
        decision, "wrapper_yes_no_probability", lambda *args: 0.9,
    )
    observation = decision.observe_global(
        model=Model(), verifier_model=object(),
        policy={
            "question": "Which option?", "options": ["A", "B"],
            "answer_type": "logits_match", "input_image": "sample.png",
        },
        image=Image.new("RGB", (4, 3)), direct_threshold=0.8,
        gate=GlobalGateConfig(0.8, 0.2, 0.75),
        generator_checkpoint_sha256="a" * 64,
        verifier_checkpoint_sha256="b" * 64,
    )
    output["answer"].append("B")

    assert len(answer_calls) == 1
    assert observation.output == {"answer": ["A"]}
    assert observation.to_dict()["answer_record"]["output"] == {"answer": ["A"]}
    assert observation.sufficient is True
    assert observation.accepted is True
    assert observation.direct_stop is True
    assert observation.to_dict()["cost"] == {
        "model_calls": 6,
        "processed_pixels": 72,
    }
    assert verifier_calls[0]["proposed_answer"] == "A"


@pytest.mark.parametrize("paired", (False, True))
def test_v3_global_gate_verifies_all_options_and_requires_argmax_agreement(
    monkeypatch, paired,
):
    from qavs.independent_search.config import GlobalGateConfig
    from qavs.independent_search import decision
    from qavs.independent_search.semantics import build_option_catalog

    answer = SimpleNamespace(
        output=0, margin=0.99, frequency=1.0, aggregation_available=True,
    )
    answer.to_dict = lambda: {"output": answer.output}

    class Model:
        def get_confidence_value(self, *args, **kwargs):
            return 0.9

    monkeypatch.setattr(decision, "answer_with_uncertainty", lambda *args: answer)
    monkeypatch.setattr(
        decision, "build_query_plan",
        lambda policy, targets: SimpleNamespace(evidence_items=({
            "kind": "question_evidence", "requirement": "visual_detail",
        },)),
    )
    monkeypatch.setattr(
        decision, "sanitize_evidence_requirements", lambda items: ("visible color",),
    )
    catalog = build_option_catalog({
        "answer_type": "logits_match", "options": ["red", "blue"],
    })
    def verifier_first(generator, verifier):
        verified = verifier()
        return generator(), verified

    rows = iter(((0, [0.0, 2.0, 4.0], 4), (1, [2.0, 0.0, 4.0], 4)))

    observation = decision.observe_global(
        model=Model(), verifier_model=object(),
        policy={
            "question": "What color?", "options": ["red", "blue"],
            "answer_type": "logits_match", "input_image": "sample.png",
        },
        image=Image.new("RGB", (4, 3)), direct_threshold=0.8,
        gate=GlobalGateConfig(0.8, 0.2, 0.75),
        generator_checkpoint_sha256="a" * 64,
        verifier_checkpoint_sha256="b" * 64,
        option_catalog=catalog,
        conditional_losses=lambda *_: next(rows),
        pair_scoring=verifier_first if paired else None,
    )

    assert observation.accepted is True
    assert observation.option_support["catalog_sha256"] == catalog.identity_sha256
    assert len(observation.option_support["options"]) == 2
    assert observation.support > 0.8
    assert observation.margin > 0.7

    answer.output = 1
    rows = iter(((0, [0.0, 2.0, 4.0], 4), (1, [2.0, 0.0, 4.0], 4)))
    disagreement = decision.observe_global(
        model=Model(), verifier_model=object(),
        policy={
            "question": "What color?", "options": ["red", "blue"],
            "answer_type": "logits_match", "input_image": "sample.png",
        },
        image=Image.new("RGB", (4, 3)), direct_threshold=0.8,
        gate=GlobalGateConfig(0.8, 0.2, 0.75),
        generator_checkpoint_sha256="a" * 64,
        verifier_checkpoint_sha256="b" * 64,
        option_catalog=catalog,
        conditional_losses=lambda *_: next(rows),
        pair_scoring=verifier_first if paired else None,
    )
    assert disagreement.accepted is False


def test_observe_global_rejects_malformed_prediction(monkeypatch):
    from qavs.independent_search.config import GlobalGateConfig
    from qavs.independent_search import decision

    answer = SimpleNamespace(
        output={"answer": float("nan")}, margin=0.30, frequency=1.0,
        aggregation_available=True,
    )
    monkeypatch.setattr(decision, "answer_with_uncertainty", lambda *args: answer)

    with pytest.raises(ValueError, match="strict JSON"):
        decision.observe_global(
            model=object(), verifier_model=object(),
            policy={
                "question": "Which option?", "options": ["A", "B"],
                "answer_type": "logits_match", "input_image": "sample.png",
            },
            image=Image.new("RGB", (4, 3)), direct_threshold=0.8,
            gate=GlobalGateConfig(0.8, 0.2, 0.75),
            generator_checkpoint_sha256="a" * 64,
            verifier_checkpoint_sha256="b" * 64,
        )


def test_observe_global_rejects_nonindependent_checkpoints_before_inference(monkeypatch):
    from qavs.independent_search.config import GlobalGateConfig
    from qavs.independent_search import decision

    monkeypatch.setattr(
        decision, "answer_with_uncertainty",
        lambda *args: pytest.fail("must reject checkpoints before inference"),
    )

    with pytest.raises(ValueError, match="independent"):
        decision.observe_global(
            model=object(), verifier_model=object(),
            policy={
                "question": "Which option?", "options": ["A", "B"],
                "answer_type": "logits_match", "input_image": "sample.png",
            },
            image=Image.new("RGB", (4, 3)), direct_threshold=0.8,
            gate=GlobalGateConfig(0.8, 0.2, 0.75),
            generator_checkpoint_sha256="a" * 64,
            verifier_checkpoint_sha256="a" * 64,
        )


def test_final_decision_ledger_extension_preserves_existing_constructor():
    from qavs.independent_search.decision import FinalDecision

    result = FinalDecision("global", "full_image_fallback", None)

    assert result.ledger_trace == {}


def test_v3_backtracked_sibling_evidence_is_excluded_from_new_branch():
    from qavs.independent_search.decision import select_accepted_hypothesis

    catalog = _v3_catalog()
    records = [
        _v3_record(0, ("root",), catalog, 0.5, 0.5, root=True),
        _v3_record(
            1, ("root", "left"), catalog, 0.95, 0.05,
            instance="hat-left", action="NEXT",
        ),
        _v3_record(
            2, ("root", "left"), catalog, 0.95, 0.05,
            instance="hat-left-other", geometry=(2, 2, 32, 32), scale=2.0,
            action="ZOOM",
        ),
        _v3_record(
            3, ("root", "right"), catalog, 0.05, 0.95,
            instance="hat-right", red_winner="Refute", blue_winner="Support",
            geometry=(60, 0, 40, 40), action="NEXT",
        ),
        _v3_record(
            4, ("root", "right"), catalog, 0.05, 0.95,
            instance="hat-right", red_winner="Refute", blue_winner="Support",
            geometry=(65, 5, 30, 30), scale=2.0, action="ZOOM",
        ),
    ]

    result = select_accepted_hypothesis(
        records,
        root_state_id=0,
        acceptance=_acceptance_v3(),
        validation_enabled=True,
        root_fallback_answer=0,
        option_catalog=catalog,
        question_kind="attribute",
    )

    assert result.answer == 1
    assert result.accepted_hypothesis.canonical_answer == "blue"
    accepted_evaluation = result.ledger_trace["evaluations"][-1]
    assert accepted_evaluation["accepted"] is True
    assert all(
        group["path"] != ["root", "left"]
        for group in accepted_evaluation["included_groups"]
    )
    assert any(
        item["reason"] == "sibling_or_descendant_branch"
        for item in accepted_evaluation["excluded_groups"]
    )


def test_v3_same_answer_on_unrelated_instances_does_not_confirm():
    from qavs.independent_search.decision import select_accepted_hypothesis

    catalog = _v3_catalog()
    result = select_accepted_hypothesis(
        [
            _v3_record(0, ("root",), catalog, 0.5, 0.5, root=True),
            _v3_record(
                1, ("root", "hat"), catalog, 0.95, 0.05,
                instance="hat-left", action="NEXT",
            ),
            _v3_record(
                2, ("root", "hat"), catalog, 0.95, 0.05,
                instance="hat-right", geometry=(2, 2, 32, 32),
                scale=2.0, action="ZOOM",
            ),
        ],
        root_state_id=0,
        acceptance=_acceptance_v3(),
        validation_enabled=True,
        root_fallback_answer=0,
        option_catalog=catalog,
        question_kind="attribute",
    )

    assert result.accepted_hypothesis is None
    assert result.answer == 0
    assert result.ledger_trace["evaluations"][-1]["diverse_confirm"] is False


def test_v3_duplicate_render_and_positive_trend_never_replace_diverse_confirm():
    from qavs.independent_search.decision import select_accepted_hypothesis

    catalog = _v3_catalog()
    shared_render = "f" * 64
    result = select_accepted_hypothesis(
        [
            _v3_record(0, ("root",), catalog, 0.5, 0.5, root=True),
            _v3_record(
                1, ("root", "hat"), catalog, 0.75, 0.25,
                render=shared_render, instance="hat", action="NEXT",
            ),
            _v3_record(
                2, ("root", "hat"), catalog, 0.85, 0.15,
                render=shared_render, instance="hat", scale=2.0, action="ZOOM",
            ),
            _v3_record(
                3, ("root", "hat"), catalog, 0.95, 0.05,
                render=shared_render, instance="hat", scale=3.0, action="ZOOM",
            ),
        ],
        root_state_id=0,
        acceptance=_acceptance_v3(),
        validation_enabled=True,
        root_fallback_answer=0,
        option_catalog=catalog,
        question_kind="attribute",
    )

    assert result.accepted_hypothesis is None
    assert result.ledger_trace["evaluations"][-1]["diverse_confirm"] is False


def test_v3_invalid_global_vector_is_audited_but_not_scored():
    from qavs.independent_search.decision import select_accepted_hypothesis

    catalog = _v3_catalog()
    root = _v3_record(
        0, ("root",), catalog, 0.5, 0.5, root=True,
        red_winner="Insufficient", blue_winner="Insufficient",
    )
    root["option_support"]["valid"] = False
    first = _v3_record(
        1, ("root", "right"), catalog, 0.95, 0.05,
        geometry=(60, 0, 40, 40), action="NEXT",
    )
    second = _v3_record(
        2, ("root", "right"), catalog, 0.95, 0.05,
        geometry=(65, 5, 25, 25), scale=2.0, action="ZOOM",
    )

    result = select_accepted_hypothesis(
        [root, first, second], root_state_id=0,
        acceptance=_acceptance_v3(), validation_enabled=True,
        root_fallback_answer=1, option_catalog=catalog,
        question_kind="attribute",
    )

    assert result.answer == 0
    first_evaluation = result.ledger_trace["evaluations"][0]
    assert first_evaluation["accepted"] is False
    assert first_evaluation["included_groups"] == []


def test_v3_rejected_global_prediction_is_not_positive_branch_support():
    from qavs.independent_search.decision import select_accepted_hypothesis

    catalog = _v3_catalog()
    root = _v3_record(
        0, ("root",), catalog, 0.05, 0.95, root=True,
        red_winner="Refute", blue_winner="Support",
    )
    root["state"]["positive_support_eligible"] = False
    first = _v3_record(
        1, ("root", "hat"), catalog, 0.95, 0.05,
        instance="hat-1", action="NEXT",
    )
    second = _v3_record(
        2, ("root", "hat"), catalog, 0.95, 0.05,
        instance="hat-1", geometry=(2, 2, 32, 32),
        scale=2.0, action="ZOOM",
    )

    result = select_accepted_hypothesis(
        [root, first, second], root_state_id=0,
        acceptance=_acceptance_v3(), validation_enabled=True,
        root_fallback_answer=1, option_catalog=catalog,
        question_kind="attribute",
    )

    assert result.answer == 0
    evaluation = result.ledger_trace["evaluations"][-1]
    assert any(
        item["reason"] == "global_not_accepted_as_positive_support"
        for item in evaluation["excluded_groups"]
    )


def test_v3_attribute_scoring_excludes_invalid_or_ungrounded_local_views():
    from qavs.independent_search.decision import select_accepted_hypothesis

    catalog = _v3_catalog()
    root = _v3_record(0, ("root",), catalog, 0.5, 0.5, root=True)
    root["state"]["positive_support_eligible"] = False
    invalid_output = _v3_record(
        1, ("root", "hat"), catalog, 0.99, 0.01,
        instance="hat-1", action="NEXT",
    )
    invalid_output["answer"]["output"] = "not-an-option"
    ungrounded = _v3_record(
        2, ("root", "hat"), catalog, 0.99, 0.01,
        instance="hat-1", geometry=(2, 2, 32, 32),
        scale=2.0, action="ZOOM",
    )
    ungrounded["grounding"]["record"]["valid"] = False

    result = select_accepted_hypothesis(
        [root, invalid_output, ungrounded], root_state_id=0,
        acceptance=_acceptance_v3(), validation_enabled=True,
        root_fallback_answer=1, option_catalog=catalog,
        question_kind="attribute",
    )

    evaluation = result.ledger_trace["evaluations"][-1]
    assert evaluation["included_groups"] == []
    assert {item["reason"] for item in evaluation["excluded_groups"]} >= {
        "generated_output_invalid", "query_grounding_invalid",
    }


def test_v3_relation_bundle_reverifies_complete_cross_branch_roles():
    from qavs.independent_search.decision import select_accepted_hypothesis

    catalog = _v3_catalog()
    root = _v3_record(0, ("root",), catalog, 0.5, 0.5, root=True)
    man = _v3_record(
        1, ("root", "left"), catalog, 0.9, 0.1,
        instance="man-1", geometry=(0, 0, 40, 80), action="NEXT",
    )
    man["grounding"]["record"]["role_to_instance"] = [
        {"role": "man", "instance_id": "man-1"},
    ]
    bicycle = _v3_record(
        2, ("root", "right"), catalog, 0.9, 0.1,
        instance="bike-1", geometry=(60, 0, 40, 80), action="NEXT",
    )
    bicycle["grounding"]["record"]["role_to_instance"] = [
        {"role": "bicycle", "instance_id": "bike-1"},
    ]
    _attach_v3_bundle(
        bicycle, catalog, question_kind="relation",
        required_roles=("man", "bicycle"),
        constituent_state_ids=(1, 2),
        mappings=(("man", "man-1"), ("bicycle", "bike-1")),
        covered=("man-1", "bike-1"),
    )

    result = select_accepted_hypothesis(
        [root, man, bicycle], root_state_id=0,
        acceptance=_acceptance_v3(), validation_enabled=True,
        root_fallback_answer=1, option_catalog=catalog,
        question_kind="relation", required_roles=("man", "bicycle"),
    )

    assert result.answer == 0
    assert result.accepted_hypothesis.score_source == "joint_bundle_verification"
    assert result.ledger_trace["evaluations"][-1]["bundle_valid"] is True
    assert result.ledger_trace["evaluations"][-1]["accepted"] is True


def test_v3_count_bundle_rejects_duplicate_instances_or_missing_coverage():
    from qavs.independent_search.decision import select_accepted_hypothesis

    catalog = _v3_catalog()
    root = _v3_record(0, ("root",), catalog, 0.5, 0.5, root=True)
    left = _v3_record(
        1, ("root", "left"), catalog, 0.9, 0.1,
        instance="car-1", geometry=(0, 0, 40, 80), action="NEXT",
    )
    left["grounding"]["record"]["role_to_instance"] = [
        {"role": "car", "instance_id": "car-1"},
    ]
    right = _v3_record(
        2, ("root", "right"), catalog, 0.9, 0.1,
        instance="car-1", geometry=(60, 0, 40, 80), action="NEXT",
    )
    right["grounding"]["record"]["role_to_instance"] = [
        {"role": "car", "instance_id": "car-1"},
    ]
    _attach_v3_bundle(
        right, catalog, question_kind="count", required_roles=("car",),
        constituent_state_ids=(1, 2), mappings=(("car", "car-1"),),
        covered=("car-1",), coverage_context=False,
    )

    result = select_accepted_hypothesis(
        [root, left, right], root_state_id=0,
        acceptance=_acceptance_v3(), validation_enabled=True,
        root_fallback_answer=1, option_catalog=catalog,
        question_kind="count", required_roles=("car",),
    )

    assert result.answer == 1
    assert result.accepted_hypothesis is None
    assert result.ledger_trace["evaluations"][-1]["bundle_valid"] is False


@pytest.mark.parametrize("schema_version", [1, 2, 4])
def test_config_rejects_obsolete_or_unknown_schema(schema_version):
    from qavs.independent_search.config import IndependentSearchConfig

    value = _config()
    value["schema_version"] = schema_version
    with pytest.raises(ValueError, match="schema_version must be 3"):
        IndependentSearchConfig.from_mapping(value)


@pytest.mark.parametrize("aggregation", [
    "cross_view_answer_consensus",
    "cross_view_correction_consensus",
    "global_verifier_positive_then_branch_equal_mean",
])
def test_config_rejects_experimental_acceptance_routes(aggregation):
    from qavs.independent_search.config import BranchAcceptanceConfig

    value = _config()["acceptance"]
    value["aggregation"] = aggregation
    with pytest.raises(ValueError, match="aggregation"):
        BranchAcceptanceConfig.from_mapping(value)
