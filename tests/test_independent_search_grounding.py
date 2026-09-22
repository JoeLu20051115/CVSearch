from PIL import Image

from qavs.evidence_gap.pdf_types import SearchStateRecord


def _state(key, bbox_marker="base", *, state_id=1):
    return SearchStateRecord(
        state_id=state_id,
        focus_keys=(key,),
        path_keys=("root", key),
        context_keys=(),
        visited_keys=("root", key),
        observation_keys=("root@root", f"{key}@{bbox_marker}"),
        remaining_steps=7,
        remaining_model_calls=100,
        remaining_pixels=10_000_000,
    )


def _grounded(role):
    from qavs.independent_search.grounding import GroundingLabelDistribution

    return {
        role: GroundingLabelDistribution(
            labels=("Grounded", "NotGrounded", "Insufficient"),
            losses=(0.0, 2.0, 4.0),
            probabilities=(0.86, 0.12, 0.02),
            winner="Grounded",
        ),
    }


def _not_grounded(*roles):
    from qavs.independent_search.grounding import GroundingLabelDistribution

    return {
        role: GroundingLabelDistribution(
            labels=("Grounded", "NotGrounded", "Insufficient"),
            losses=(2.0, 0.0, 4.0),
            probabilities=(0.12, 0.86, 0.02),
            winner="NotGrounded",
        )
        for role in roles
    }


def _proposal(key, role, bbox):
    return {
        "canonical_key": key,
        "bbox_original": list(bbox),
        "source": "sam_proposal",
        "sam_target_id": 17,
        "sam_target": role,
        "sam_has_mask": True,
    }


def _sgap(key, bbox):
    return {
        "canonical_key": key,
        "bbox_original": list(bbox),
        "source": "sgap_recovered",
        "sam_target_id": None,
        "sam_target": None,
        "sam_has_mask": None,
    }


def test_nested_and_zoomed_views_share_one_sam_instance():
    from qavs.independent_search.grounding import TargetInstanceRegistry

    registry = TargetInstanceRegistry.from_frontend(
        image=Image.new("RGB", (100, 80)),
        targets=("hat",),
        candidates=(_proposal("hat-proposal", "hat", (10, 10, 20, 10)),),
        sam_model=None,
        target_instance_iou=0.5,
    )
    first = registry.ground_state(
        _state("hat-proposal"), _grounded("hat"), grounding_threshold=0.65,
    )
    second = registry.ground_state(
        _state("hat-proposal", "zoom1", state_id=2),
        _grounded("hat"), grounding_threshold=0.65,
    )

    assert first.valid is True
    assert first.role_to_instance == second.role_to_instance
    assert len(first.role_to_instance) == 1


def test_unlocalized_sgap_view_cannot_confirm_and_cross_instance_is_rejected():
    from qavs.independent_search.grounding import (
        TargetInstanceRegistry,
        compatible_confirmation,
    )

    registry = TargetInstanceRegistry.from_frontend(
        image=Image.new("RGB", (100, 80)), targets=("hat",),
        candidates=(_sgap("wide-sgap", (0, 0, 40, 40)),),
        sam_model=None, target_instance_iou=0.5,
    )
    missing = registry.ground_state(
        _state("wide-sgap"), _grounded("hat"), grounding_threshold=0.65,
        lazy_sam=lambda image, roles: {},
    )
    assert missing.valid is False

    left_registry = TargetInstanceRegistry.from_frontend(
        image=Image.new("RGB", (100, 80)), targets=("hat",),
        candidates=(
            _proposal("left", "hat", (0, 0, 20, 20)),
            _proposal("right", "hat", (60, 0, 20, 20)),
        ),
        sam_model=None, target_instance_iou=0.5,
    )
    left = left_registry.ground_state(
        _state("left"), _grounded("hat"), grounding_threshold=0.65,
    )
    right = left_registry.ground_state(
        _state("right", state_id=2), _grounded("hat"),
        grounding_threshold=0.65,
    )
    assert compatible_confirmation(left, right, question_kind="attribute") is False


def test_lazy_local_sam_maps_to_original_coordinates_merges_and_caches():
    from qavs.independent_search.grounding import TargetInstanceRegistry

    calls = []

    def lazy(image, roles):
        calls.append((image.size, roles))
        return {"hat": [[2, 3, 12, 13]]}

    registry = TargetInstanceRegistry.from_frontend(
        image=Image.new("RGB", (120, 100)), targets=("hat",),
        candidates=(_sgap("sgap", (50, 60, 40, 30)),),
        sam_model=None, target_instance_iou=0.5,
    )
    first = registry.ground_state(
        _state("sgap"), _grounded("hat"), grounding_threshold=0.65,
        lazy_sam=lazy,
    )
    second = registry.ground_state(
        _state("sgap", state_id=2), _grounded("hat"),
        grounding_threshold=0.65, lazy_sam=lazy,
    )

    assert calls == [((40, 30), ("hat",))]
    assert first.valid is second.valid is True
    assert first.role_to_instance == second.role_to_instance
    instance = registry.instance(first.role_to_instance[0][1])
    assert instance.bbox_original == (52.0, 63.0, 10.0, 10.0)


def test_grounding_verifier_scores_every_role_and_fails_below_threshold():
    from qavs.independent_search.grounding import verify_target_grounding

    rows = iter((
        (0, [0.0, 2.0, 4.0], 4),
        (1, [2.0, 0.0, 4.0], 4),
    ))
    result = verify_target_grounding(
        question="What is the man wearing?", roles=("man", "hat"),
        image=Image.new("RGB", (8, 6)),
        conditional_losses=lambda *_: next(rows),
        checkpoint_sha256="b" * 64,
        generator_checkpoint_sha256="a" * 64,
        grounding_threshold=0.65,
    )

    assert [item.role for item in result.roles] == ["man", "hat"]
    assert result.roles[0].grounded is True
    assert result.roles[1].grounded is False
    assert result.valid is False
    assert result.model_calls == 8


def test_relation_confirmation_allows_complementary_grounded_roles():
    from qavs.independent_search.grounding import (
        GroundingRecord,
        compatible_confirmation,
    )

    man = GroundingRecord(
        role_distributions=(),
        role_to_instance=(("man", "man-1"),),
        covered_instance_ids=("man-1",), valid=True,
    )
    bicycle = GroundingRecord(
        role_distributions=(),
        role_to_instance=(("bicycle", "bike-1"),),
        covered_instance_ids=("bike-1",), valid=True,
    )

    assert compatible_confirmation(
        man, bicycle, question_kind="relation",
        required_roles=("man", "bicycle"),
    ) is True


def test_query_targeted_sam_proposals_ground_complementary_relation_roles():
    from qavs.independent_search.grounding import (
        TargetInstanceRegistry,
        compatible_confirmation,
    )

    registry = TargetInstanceRegistry.from_frontend(
        image=Image.new("RGB", (100, 80)), targets=("umbrella", "traffic light"),
        candidates=(
            _proposal("umbrella-box", "umbrella", (10, 20, 20, 20)),
            _proposal("light-box", "traffic light", (70, 10, 10, 30)),
        ),
        sam_model=None, target_instance_iou=0.5,
    )
    umbrella = registry.ground_state(
        _state("umbrella-box"), {**_grounded("umbrella"), **_not_grounded("traffic light")}, grounding_threshold=0.65,
    )
    light = registry.ground_state(
        _state("light-box", state_id=2), {**_not_grounded("umbrella"), **_grounded("traffic light")}, grounding_threshold=0.65,
    )

    assert umbrella.valid is light.valid is True
    assert [role for role, _ in umbrella.role_to_instance] == ["umbrella"]
    assert [role for role, _ in light.role_to_instance] == ["traffic light"]
    assert compatible_confirmation(
        umbrella, light, question_kind="relation",
        required_roles=("umbrella", "traffic light"),
    ) is True


def test_overlapping_sam_proposals_preserve_all_instance_aliases():
    from qavs.independent_search.grounding import TargetInstanceRegistry

    registry = TargetInstanceRegistry.from_frontend(
        image=Image.new("RGB", (100, 80)), targets=("mailbox",),
        candidates=(
            _proposal("mailbox-wide", "mailbox", (10, 10, 30, 30)),
            _proposal("mailbox-tight", "mailbox", (12, 12, 28, 28)),
        ),
        sam_model=None, target_instance_iou=0.5,
    )
    record = registry.ground_state(
        _state("mailbox-tight"), _grounded("mailbox"),
        grounding_threshold=0.65,
    )

    assert record.valid is True
    instances = registry.to_dict()["instances"]
    assert len(instances) == 1
    assert instances[0]["proposal_keys"] == ["mailbox-tight", "mailbox-wide"]


def test_sam_proposal_cannot_override_independent_not_grounded_verdict():
    from qavs.independent_search.grounding import TargetInstanceRegistry

    registry = TargetInstanceRegistry.from_frontend(
        image=Image.new("RGB", (100, 80)), targets=("mailbox",),
        candidates=(_proposal("mailbox", "mailbox", (10, 10, 30, 30)),),
        sam_model=None, target_instance_iou=0.5,
    )
    record = registry.ground_state(
        _state("mailbox"), _not_grounded("mailbox"), grounding_threshold=0.65,
    )
    assert not record.valid
    assert record.role_to_instance == ()
