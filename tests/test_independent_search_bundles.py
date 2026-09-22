from PIL import Image
import pytest


def _record(
    state_id, branch, mappings, covered, *, geometry, render=None,
):
    return {
        "state": {
            "state_id": state_id,
            "path_keys": ["root", branch],
            "effective_geometry": list(geometry),
            "render_sha256": render or f"{state_id:064x}",
        },
        "grounding": {
            "record": {
                "role_to_instance": [
                    {"role": role, "instance_id": instance}
                    for role, instance in mappings
                ],
                "covered_instance_ids": list(covered),
                "valid": bool(mappings),
            },
        },
    }


def test_relation_bundle_combines_only_records_covering_all_required_roles():
    from qavs.independent_search.bundles import build_bundle_plan

    records = [
        _record(
            1, "left", (("man", "man-1"),), ("man-1",),
            geometry=(0, 0, 40, 80),
        ),
        _record(
            2, "right", (("bicycle", "bike-1"),), ("bike-1",),
            geometry=(60, 0, 40, 80),
        ),
        _record(
            3, "noise", (("tree", "tree-1"),), ("tree-1",),
            geometry=(40, 0, 20, 80),
        ),
    ]

    plan = build_bundle_plan(
        records,
        question_kind="relation",
        required_roles=("man", "bicycle"),
        coverage_context=True,
    )

    assert plan.valid is True
    assert plan.constituent_state_ids == (1, 2)
    assert dict(plan.role_to_instance) == {"man": "man-1", "bicycle": "bike-1"}
    assert "tree-1" not in plan.covered_instance_ids


def test_relation_bundle_fails_closed_when_one_role_is_missing():
    from qavs.independent_search.bundles import build_bundle_plan

    plan = build_bundle_plan(
        [_record(
            1, "left", (("man", "man-1"),), ("man-1",),
            geometry=(0, 0, 40, 80),
        )],
        question_kind="comparison",
        required_roles=("man", "bicycle"),
        coverage_context=True,
    )

    assert plan.valid is False
    assert plan.reason == "required_roles_incomplete"


def test_relation_bundle_requires_distinct_role_branches():
    from qavs.independent_search.bundles import build_bundle_plan

    combined = _record(
        2, "combined", (("man", "man-1"), ("bicycle", "bike-1")),
        ("man-1", "bike-1"), geometry=(0, 0, 100, 80),
    )
    man_only = _record(
        1, "left", (("man", "man-1"),), ("man-1",),
        geometry=(0, 0, 40, 80),
    )

    plan = build_bundle_plan(
        [man_only, combined], question_kind="relation",
        required_roles=("man", "bicycle"), coverage_context=True,
    )

    assert plan.valid is True
    assert plan.constituent_state_ids == (1, 2)

    invalid = build_bundle_plan(
        [combined], question_kind="relation",
        required_roles=("man", "bicycle"), coverage_context=True,
    )
    assert invalid.valid is False
    assert invalid.reason == "distinct_role_branches_incomplete"


def test_count_bundle_requires_distinct_instances_and_coverage_context():
    from qavs.independent_search.bundles import build_bundle_plan

    duplicate = [
        _record(
            1, "left", (("car", "car-1"),), ("car-1",),
            geometry=(0, 0, 40, 80),
        ),
        _record(
            2, "right", (("car", "car-1"),), ("car-1",),
            geometry=(60, 0, 40, 80),
        ),
    ]
    distinct = duplicate + [_record(
        3, "right-2", (("car", "car-2"),), ("car-2",),
        geometry=(45, 0, 15, 80),
    )]

    assert build_bundle_plan(
        duplicate, question_kind="count", required_roles=("car",),
        coverage_context=True,
    ).valid is False
    assert build_bundle_plan(
        distinct, question_kind="count", required_roles=("car",),
        coverage_context=False,
    ).valid is False
    valid = build_bundle_plan(
        distinct, question_kind="count", required_roles=("car",),
        coverage_context=True,
    )
    assert valid.valid is True
    assert valid.covered_instance_ids == ("car-1", "car-2")
    assert len(valid.constituent_state_ids) == 2


@pytest.mark.parametrize("question_kind", ("count", "coverage"))
def test_bundle_rejects_instances_collapsed_into_one_selected_state(question_kind):
    from qavs.independent_search.bundles import build_bundle_plan

    records = [
        _record(1, "left", (("tv", "tv-1"),), ("tv-1",),
                geometry=(0, 0, 40, 80)),
        _record(2, "combined", (("tv", "tv-1"), ("screen", "screen-1")),
                ("tv-1", "screen-1"), geometry=(0, 0, 100, 80)),
    ]
    plan = build_bundle_plan(records, question_kind=question_kind,
                             required_roles=("tv", "screen"), coverage_context=True)
    assert plan.constituent_state_ids == (2,)
    assert plan.covered_instance_ids == ("screen-1", "tv-1")
    assert plan.valid is False
    assert plan.reason == "distinct_constituent_states_incomplete"


def test_singleton_bundle_stays_an_incomplete_plan_before_rendering(monkeypatch):
    from types import SimpleNamespace

    from qavs.evidence_gap.pdf_runtime import PDFStateEvaluator
    from qavs.independent_search import bundles

    candidate = _record(
        1, "combined", (("electronic device", "device-1"), ("flat screen", "screen-1")),
        ("device-1", "screen-1"), geometry=(0, 70, 160, 355),
    )
    evaluator = PDFStateEvaluator.__new__(PDFStateEvaluator)
    evaluator.option_catalog = object()
    evaluator.question_kind = "coverage"
    evaluator.required_roles = ("electronic device", "flat screen")
    evaluator._prior_bundle_candidates = []
    evaluator._bundle_candidates = []
    evaluator.adapter = SimpleNamespace(image=Image.new("RGB", (640, 425)))

    def reject_render(*args):
        raise AssertionError("a singleton plan must not be rendered or sent to models")

    monkeypatch.setattr(bundles, "render_evidence_bundle", reject_render)
    answer, vector, payload = evaluator._joint_bundle(candidate, answer_calls=3)
    assert answer is None and vector is None
    assert set(payload) == {"plan"}
    assert payload["plan"]["constituent_state_ids"] == [1]
    assert len(payload["plan"]["covered_instance_ids"]) == 2
    assert payload["plan"]["valid"] is False
    assert payload["plan"]["reason"] == "distinct_constituent_states_incomplete"


def test_bundle_dedup_keeps_latest_state_for_a_repeated_render():
    from qavs.independent_search.bundles import build_bundle_plan

    repeated = "d" * 64
    records = [
        _record(
            1, "old", (("man", "man-1"),), ("man-1",),
            geometry=(0, 0, 40, 80), render=repeated,
        ),
        _record(
            3, "new", (("man", "man-1"),), ("man-1",),
            geometry=(0, 0, 40, 80), render=repeated,
        ),
        _record(
            2, "right", (("bicycle", "bike-1"),), ("bike-1",),
            geometry=(60, 0, 40, 80),
        ),
    ]

    plan = build_bundle_plan(
        records, question_kind="relation",
        required_roles=("man", "bicycle"), coverage_context=True,
    )

    assert plan.valid is True
    assert plan.constituent_state_ids == (2, 3)


def test_bundle_render_is_deterministic_and_contains_overview_plus_crops():
    from qavs.independent_search.bundles import (
        build_bundle_plan,
        render_evidence_bundle,
    )

    records = [
        _record(
            1, "left", (("man", "man-1"),), ("man-1",),
            geometry=(0, 0, 40, 80),
        ),
        _record(
            2, "right", (("bicycle", "bike-1"),), ("bike-1",),
            geometry=(60, 0, 40, 80),
        ),
    ]
    plan = build_bundle_plan(
        records, question_kind="relation",
        required_roles=("man", "bicycle"), coverage_context=True,
    )
    image = Image.new("RGB", (100, 80), "white")

    first = render_evidence_bundle(image, records, plan)
    second = render_evidence_bundle(image, list(reversed(records)), plan)

    assert first.tobytes() == second.tobytes()
    assert first.height > image.height
    assert first.width >= 100


def test_large_bundle_render_is_bounded_for_exact_budget_preflight():
    from qavs.independent_search.bundles import (
        build_bundle_plan,
        render_evidence_bundle,
    )

    records = [
        _record(
            state_id, f"branch-{state_id}", (("car", f"car-{state_id}"),),
            (f"car-{state_id}",),
            geometry=((state_id - 1) * 1000, 0, 1000, 1000),
        )
        for state_id in range(1, 8)
    ]
    plan = build_bundle_plan(
        records, question_kind="count", required_roles=("car",),
        coverage_context=True,
    )
    image = Image.new("RGB", (8000, 1000), "white")

    rendered = render_evidence_bundle(image, records, plan)

    assert max(rendered.size) <= 4096
