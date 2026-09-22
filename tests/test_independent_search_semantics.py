import copy

import pytest
from PIL import Image


def _four_shuffled_blocks(*choices):
    cat, dog, bird, fish = choices
    return [
        f"A. {cat}\nB. {dog}\nC. {bird}\nD. {fish}",
        f"A. {dog}\nB. {bird}\nC. {fish}\nD. {cat}",
        f"A. {fish}\nB. {dog}\nC. {cat}\nD. {bird}",
        f"A. {fish}\nB. {cat}\nC. {bird}\nD. {dog}",
    ]


def _config_v3():
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
            "max_model_calls": 256,
            "max_processed_pixels": 1_600_000_000,
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
        "verification": {
            "labels": ["Support", "Refute", "Insufficient"],
            "grounding_labels": ["Grounded", "NotGrounded", "Insufficient"],
            "grounding_threshold": 0.65,
            "target_instance_iou": 0.50,
        },
        "observation_geometry": {
            "min_zoom_factor": 0.40,
            "context_max_normalized_gap": 0.25,
        },
        "acceptance": {
            "min_absolute_support": 0.65,
            "min_normalized_margin": 0.15,
            "min_view_support": 0.60,
            "aggregation": "branch_equal_mean",
        },
    }


def _catalog(*options):
    from qavs.independent_search.semantics import build_option_catalog

    return build_option_catalog({
        "answer_type": "logits_match", "options": list(options),
    })


def _image():
    return Image.new("RGB", (8, 6), "white")


def test_v3_catalog_projects_vstar_and_shuffled_hr_outputs():
    from qavs.independent_search.semantics import build_option_catalog

    vstar = build_option_catalog({
        "answer_type": "logits_match", "options": ["red", "blue"],
    })
    assert [entry.key for entry in vstar.entries] == ["index:0", "index:1"]
    assert vstar.project("index:1") == 1

    hr = build_option_catalog({
        "answer_type": "option_list",
        "options": _four_shuffled_blocks("cat", "dog", "bird", "fish"),
    })
    assert len(hr.entries) == 4
    assert hr.project(hr.entries[0].key) == ["A", "D", "C", "B"]


def test_v3_catalog_preserves_case_distinct_hr_options_when_canonical_text_collides():
    from qavs.independent_search.semantics import build_option_catalog

    hr = build_option_catalog({
        "answer_type": "option_list",
        "options": _four_shuffled_blocks("Red", "red", "Brown", "Blue"),
    })

    assert [entry.text for entry in hr.entries] == [
        "Red", "red", "Brown", "Blue",
    ]
    assert len({entry.key for entry in hr.entries}) == 4
    assert hr.project(hr.entries[0].key) == ["A", "D", "C", "B"]
    assert hr.project(hr.entries[1].key) == ["B", "A", "B", "D"]


def test_option_catalog_is_immutable_and_rejects_nonprojectable_inputs():
    from qavs.independent_search.semantics import build_option_catalog

    policy = {"answer_type": "logits_match", "options": ["red", "blue"]}
    catalog = build_option_catalog(policy)
    projected = catalog.project("index:0")
    policy["options"][0] = "poison"
    assert catalog.entries[0].text == "red"
    assert projected == 0
    with pytest.raises(ValueError, match="unique"):
        build_option_catalog({
            "answer_type": "logits_match", "options": ["Red", " red  "],
        })
    with pytest.raises((ValueError, TypeError)):
        build_option_catalog({
            "answer_type": "option_list",
            "options": _four_shuffled_blocks("cat", "cat", "bird", "fish"),
        })


def test_v3_config_freezes_labels_geometry_and_branch_aggregation():
    from qavs.independent_search import IndependentSearchConfig

    value = _config_v3()
    parsed = IndependentSearchConfig.from_mapping(value)
    assert parsed.schema_version == 3
    assert parsed.method == "independent_query_aware_v3"
    assert parsed.verification.labels == (
        "Support", "Refute", "Insufficient",
    )
    assert parsed.verification.grounding_labels == (
        "Grounded", "NotGrounded", "Insufficient",
    )
    assert parsed.acceptance.aggregation == "branch_equal_mean"
    assert not hasattr(parsed.acceptance, "min_gain")
    assert parsed.to_dict() == value


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("labels", ["Support", "Insufficient", "Refute"], "labels"),
        (
            "grounding_labels",
            ["Grounded", "Insufficient", "NotGrounded"],
            "grounding_labels",
        ),
        ("grounding_threshold", 1.1, "grounding_threshold"),
        ("target_instance_iou", -0.1, "target_instance_iou"),
    ],
)
def test_v3_config_rejects_unfrozen_verification_contract(field, value, match):
    from qavs.independent_search import IndependentSearchConfig

    config = copy.deepcopy(_config_v3())
    config["verification"][field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        IndependentSearchConfig.from_mapping(config)


def test_option_verifier_softmaxes_all_three_labels_and_normalizes_across_options():
    from qavs.independent_search.semantics import verify_option_support

    rows = iter(((0, [0.0, 2.0, 4.0], 4), (2, [4.0, 3.0, 0.0], 4)))
    prompts = []

    def conditional(image, prompt, labels):
        prompts.append((image, prompt, labels))
        return next(rows)

    result = verify_option_support(
        question="What color?", catalog=_catalog("red", "blue"), image=_image(),
        requirements=("visible color",), conditional_losses=conditional,
        checkpoint_sha256="b" * 64, generator_checkpoint_sha256="a" * 64,
    )
    assert result.valid is True
    assert result.options[0].distribution.winner == "Support"
    assert result.options[1].distribution.winner == "Insufficient"
    assert sum(item.normalized_support for item in result.options) == pytest.approx(1.0)
    assert result.model_calls == 8
    assert result.processed_pixels == 8 * 8 * 6
    assert all(item[2] == ("Support", "Refute", "Insufficient") for item in prompts)
    assert "red" in prompts[0][1] and "blue" not in prompts[0][1]
    assert "visible color" in prompts[0][1]


def test_option_verifier_rejects_incomplete_nonfinite_or_shared_checkpoint_vectors():
    from qavs.independent_search.semantics import verify_option_support

    with pytest.raises(ValueError, match="three finite losses"):
        verify_option_support(
            question="What color?", catalog=_catalog("red", "blue"),
            image=_image(), requirements=("visible color",),
            conditional_losses=lambda *_: (0, [0.0, float("nan")], 3),
            checkpoint_sha256="b" * 64,
            generator_checkpoint_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="different checkpoint"):
        verify_option_support(
            question="What color?", catalog=_catalog("red", "blue"),
            image=_image(), requirements=("visible color",),
            conditional_losses=lambda *_: (0, [0.0, 1.0, 2.0], 4),
            checkpoint_sha256="a" * 64,
            generator_checkpoint_sha256="a" * 64,
        )


def test_option_verifier_marks_all_insufficient_vector_invalid():
    from qavs.independent_search.semantics import verify_option_support

    result = verify_option_support(
        question="What color?", catalog=_catalog("red", "blue"), image=_image(),
        requirements=("visible color",),
        conditional_losses=lambda *_: (2, [3.0, 2.0, 0.0], 4),
        checkpoint_sha256="b" * 64, generator_checkpoint_sha256="a" * 64,
    )
    assert result.valid is False
    assert result.top_key == "index:0"
