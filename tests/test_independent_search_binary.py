import hashlib

import pytest


STRICT_NO = {
    "min_coverage": 0.80,
    "min_refute_probability": 0.70,
    "max_yes_support_exclusive": 0.70,
    "min_distinct_refute_views": 2,
}


def _record(
    state_id,
    *,
    action,
    geometry,
    refute_probability=0.80,
    yes_support=0.10,
    grounding_valid=True,
    vector_valid=True,
):
    insufficient = 1.0 - refute_probability - yes_support
    record = {
        "state": {
            "state_id": state_id,
            "action": action,
            "effective_geometry": list(geometry),
            "render_sha256": hashlib.sha256(
                f"{state_id}:{action}:{geometry}".encode()
            ).hexdigest(),
        },
        "option_support": {
            "valid": vector_valid,
            "options": [
                {
                    "key": "yes",
                    "distribution": {
                        "labels": ["Support", "Refute", "Insufficient"],
                        "probabilities": [
                            yes_support,
                            refute_probability,
                            insufficient,
                        ],
                        "winner": "Refute",
                    },
                },
                {
                    "key": "no",
                    "distribution": {
                        "labels": ["Support", "Refute", "Insufficient"],
                        "probabilities": [0.80, 0.10, 0.10],
                        "winner": "Support",
                    },
                },
            ],
        },
    }
    if action != "GLOBAL":
        record["grounding"] = {"record": {"valid": grounding_valid}}
    return record


def _evaluate(records, coverage):
    from qavs.independent_search.binary import (
        NegativeCoverage,
        StrictNoConfig,
        evaluate_negative_gate,
    )
    return evaluate_negative_gate(
        records,
        coverage=NegativeCoverage(True, coverage),
        config=StrictNoConfig(**STRICT_NO),
    )


def _decision_record(catalog, state_id, *, path, action, geometry):
    probabilities = {
        "yes": (0.10, 0.80, 0.10, "Refute"),
        "no": (0.80, 0.10, 0.10, "Support"),
    }
    record = {
        "state": {
            "state_id": state_id,
            "path_keys": list(path),
            "context_keys": [],
            "source_image_identity": {
                "mode": "RGB",
                "size": [100, 100],
                "pixel_sha256": "a" * 64,
            },
            "effective_geometry": list(geometry),
            "rendered_scale": 1.0,
            "action": action,
            "render_sha256": f"{state_id + 1:064x}",
        },
        "answer": {"output": "no"},
        "option_support": {
            "catalog_sha256": catalog.identity_sha256,
            "options": [
                {
                    "key": entry.key,
                    "text_sha256": hashlib.sha256(
                        entry.text.encode("utf-8")
                    ).hexdigest(),
                    "distribution": {
                        "labels": ["Support", "Refute", "Insufficient"],
                        "losses": [0.0, 1.0, 2.0],
                        "probabilities": list(probabilities[entry.key][:3]),
                        "winner": probabilities[entry.key][3],
                    },
                    "raw_support": probabilities[entry.key][0],
                    "normalized_support": probabilities[entry.key][0]
                    / sum(value[0] for value in probabilities.values()),
                }
                for entry in catalog.entries
            ],
            "valid": True,
        },
    }
    if action != "GLOBAL":
        record["grounding"] = {
            "record": {
                "role_to_instance": [
                    {"role": "object", "instance_id": "object-1"},
                ],
                "covered_instance_ids": ["object-1"],
                "valid": True,
            },
        }
    return record


def test_binary_catalog_projects_yes_and_no():
    from qavs.independent_search.semantics import build_option_catalog

    catalog = build_option_catalog({"answer_type": "yes_no", "options": []})

    assert [(entry.key, entry.text) for entry in catalog.entries] == [
        ("yes", "Yes"),
        ("no", "No"),
    ]
    assert catalog.project("yes") == "yes"
    assert catalog.project("no") == "no"


def test_missing_target_alone_never_accepts_no():
    decision = _evaluate([], coverage=1.0)

    assert decision.accepted is False
    assert decision.status == "Insufficient"
    assert "distinct_refute_views" in decision.reasons


def test_two_distinct_refutes_and_coverage_accept_no():
    decision = _evaluate(
        [
            _record(0, action="GLOBAL", geometry=(0, 0, 100, 100)),
            _record(1, action="EXPAND", geometry=(10, 10, 80, 80)),
        ],
        coverage=0.84,
    )

    assert decision.accepted is True
    assert decision.status == "Refute"
    assert decision.eligible_state_ids == (0, 1)


def test_below_coverage_stays_insufficient():
    decision = _evaluate(
        [
            _record(0, action="GLOBAL", geometry=(0, 0, 100, 100)),
            _record(1, action="EXPAND", geometry=(10, 10, 80, 80)),
        ],
        coverage=0.79,
    )

    assert decision.status == "Insufficient"
    assert "spatial_coverage" in decision.reasons


def test_duplicate_geometry_is_not_independent_confirmation():
    geometry = (0, 0, 100, 100)
    decision = _evaluate(
        [
            _record(0, action="GLOBAL", geometry=geometry),
            _record(1, action="EXPAND", geometry=geometry),
        ],
        coverage=0.90,
    )

    assert decision.status == "Insufficient"
    assert "distinct_refute_views" in decision.reasons


def test_yes_support_at_exclusive_limit_blocks_no():
    decision = _evaluate(
        [
            _record(0, action="GLOBAL", geometry=(0, 0, 100, 100)),
            _record(1, action="EXPAND", geometry=(10, 10, 80, 80)),
            _record(
                2,
                action="NEXT",
                geometry=(30, 30, 20, 20),
                refute_probability=0.20,
                yes_support=0.70,
            ),
        ],
        coverage=0.90,
    )

    assert decision.status == "Insufficient"
    assert "yes_support" in decision.reasons


def test_invalid_grounding_cannot_confirm_no():
    decision = _evaluate(
        [
            _record(0, action="GLOBAL", geometry=(0, 0, 100, 100)),
            _record(
                1,
                action="EXPAND",
                geometry=(10, 10, 80, 80),
                grounding_valid=False,
            ),
        ],
        coverage=0.90,
    )

    assert decision.status == "Insufficient"
    assert decision.eligible_state_ids == (0,)


def test_context_confirmation_requires_global_or_expand_view():
    decision = _evaluate(
        [
            _record(1, action="NEXT", geometry=(0, 0, 40, 40)),
            _record(2, action="ZOOM", geometry=(60, 0, 40, 40)),
        ],
        coverage=0.90,
    )

    assert decision.status == "Insufficient"
    assert "context_refute_view" in decision.reasons


def test_general_no_acceptance_is_intercepted_until_strict_gate_passes():
    from qavs.independent_search.config import BranchAcceptanceConfig
    from qavs.independent_search.decision import select_accepted_hypothesis
    from qavs.independent_search.semantics import build_option_catalog

    catalog = build_option_catalog({"answer_type": "yes_no", "options": []})
    records = [
        _decision_record(
            catalog,
            0,
            path=("root",),
            action="GLOBAL",
            geometry=(0, 0, 100, 100),
        ),
        _decision_record(
            catalog,
            1,
            path=("root", "object"),
            action="NEXT",
            geometry=(0, 0, 40, 40),
        ),
        _decision_record(
            catalog,
            2,
            path=("root", "object", "context"),
            action="EXPAND",
            geometry=(0, 0, 80, 80),
        ),
    ]
    acceptance = BranchAcceptanceConfig(
        min_absolute_support=0.60,
        min_normalized_margin=0.20,
        min_view_support=0.70,
        aggregation="branch_equal_mean",
    )

    insufficient = select_accepted_hypothesis(
        records,
        root_state_id=0,
        acceptance=acceptance,
        validation_enabled=True,
        root_fallback_answer="yes",
        option_catalog=catalog,
        negative_coverage=__import__(
            "qavs.independent_search.binary", fromlist=["NegativeCoverage"]
        ).NegativeCoverage(True, 0.79),
    )
    accepted = select_accepted_hypothesis(
        records,
        root_state_id=0,
        acceptance=acceptance,
        validation_enabled=True,
        root_fallback_answer="yes",
        option_catalog=catalog,
        negative_coverage=__import__(
            "qavs.independent_search.binary", fromlist=["NegativeCoverage"]
        ).NegativeCoverage(True, 1.0),
    )

    assert insufficient.answer == "yes"
    assert insufficient.accepted_hypothesis is None
    assert insufficient.ledger_trace["negative_gate"]["status"] == "Insufficient"
    assert accepted.answer == "no"
    assert accepted.accepted_hypothesis is not None
    assert accepted.ledger_trace["negative_gate"]["status"] == "Refute"


def test_candidate_coverage_does_not_cross_gate_before_all_are_visited():
    from qavs.independent_search.method import _negative_candidate_coverage

    records = [{"state": {
        "visited_keys": ["a", "b", "c", "d"],
        "action": "EXPAND",
        "source_image_identity": {"size": [100, 100]},
        "effective_geometry": [0, 0, 90, 90],
    }}]

    partial = _negative_candidate_coverage(records, ["a", "b", "c", "d", "e"])
    complete = _negative_candidate_coverage(records, ["a", "b", "c", "d"])

    assert partial.all_relevant_visited is False
    assert complete.all_relevant_visited is True
    assert complete.region_union_ratio == 0.81


def test_presence_accepts_one_instance_only_after_two_distinct_grounded_views():
    import copy
    from qavs.evidence_gap.pdf_runtime import build_pdf_query_plan, question_kind_from_plan
    from qavs.independent_search.config import BranchAcceptanceConfig
    from qavs.independent_search.decision import select_accepted_hypothesis
    from qavs.independent_search.semantics import build_option_catalog

    policy = {"question": "Is there a backpack in the image?", "answer_type": "yes_no",
              "options": ["Yes", "No"], "input_image": "example.jpg"}
    catalog = build_option_catalog(policy)
    plan = build_pdf_query_plan(policy, ("backpack",), generator=lambda _: "unused")
    root = _decision_record(catalog, 0, path=("root",), action="GLOBAL", geometry=(0, 0, 100, 100))
    root["state"]["positive_support_eligible"] = False
    first = _decision_record(catalog, 1, path=("root", "object"), action="NEXT", geometry=(0, 0, 40, 40))
    second = _decision_record(catalog, 2, path=("root", "object"), action="ZOOM", geometry=(2, 2, 32, 32))
    for record in (first, second):
        record["answer"]["output"] = "yes"
        record["grounding"]["record"]["role_to_instance"][0]["role"] = "backpack"
        for option in record["option_support"]["options"]:
            yes = option["key"] == "yes"
            option["distribution"]["probabilities"] = [0.8, 0.1, 0.1] if yes else [0.1, 0.8, 0.1]
            option["distribution"]["winner"] = "Support" if yes else "Refute"
            option["raw_support"] = 0.8 if yes else 0.1
            option["normalized_support"] = option["raw_support"] / 0.9

    def decide(local):
        return select_accepted_hypothesis(
            [root, *local], root_state_id=0,
            acceptance=BranchAcceptanceConfig(0.65, 0.15, 0.6, "branch_equal_mean"),
            validation_enabled=True, root_fallback_answer="no", option_catalog=catalog,
            question_kind=question_kind_from_plan(plan), required_roles=("backpack",),
        )

    assert decide([first]).accepted_hypothesis is None
    assert decide([first, second]).answer == "yes"
    assert decide([first, second]).accepted_hypothesis is not None
    duplicate = copy.deepcopy(second)
    duplicate["state"]["render_sha256"] = first["state"]["render_sha256"]
    assert decide([first, duplicate]).accepted_hypothesis is None
    ungrounded = copy.deepcopy(second)
    ungrounded["grounding"]["record"]["valid"] = False
    assert decide([first, ungrounded]).accepted_hypothesis is None


def test_candidate_coverage_deduplicates_tiny_overlapping_crops():
    from qavs.independent_search.method import _negative_candidate_coverage

    records = [
        {"state": {
            "visited_keys": [key],
            "action": "NEXT",
            "source_image_identity": {"size": [100, 100]},
            "effective_geometry": geometry,
        }}
        for key, geometry in (("a", [0, 0, 20, 20]), ("b", [10, 0, 20, 20]))
    ]

    coverage = _negative_candidate_coverage(records, ["a", "b"])

    assert coverage.all_relevant_visited is True
    assert coverage.region_union_ratio == 0.06


@pytest.mark.parametrize("aggregation", ["branch_equal_mean", "grounded_answer_consensus"])
@pytest.mark.parametrize("coverage_mode", ["aggregate", "per_prefix"])
def test_rejected_no_does_not_hide_later_accepted_yes(aggregation, coverage_mode):
    from qavs.independent_search.config import BranchAcceptanceConfig
    from qavs.independent_search.decision import select_accepted_hypothesis
    from qavs.independent_search.method import _negative_candidate_coverage
    from qavs.independent_search.semantics import build_option_catalog

    catalog = build_option_catalog({"answer_type": "yes_no", "options": []})
    root = _decision_record(
        catalog, 0, path=("root",), action="GLOBAL", geometry=(0, 0, 100, 100),
    )
    root["state"]["positive_support_eligible"] = False
    no_views = [
        _decision_record(catalog, 1, path=("root", "a"), action="NEXT", geometry=(0, 0, 70, 100)),
        _decision_record(catalog, 2, path=("root", "a"), action="ZOOM", geometry=(0, 0, 65, 95)),
    ]
    yes_views = [
        _decision_record(catalog, 3, path=("root", "b"), action="NEXT", geometry=(70, 0, 20, 100)),
        _decision_record(catalog, 4, path=("root", "b"), action="ZOOM", geometry=(72, 2, 16, 90)),
    ]
    root["state"]["visited_keys"] = ["root"]
    for record in no_views:
        record["state"]["visited_keys"] = ["root", "a"]
    for record in yes_views:
        record["state"]["visited_keys"] = ["root", "a", "b"]
        record["answer"]["output"] = "yes"
        for option in record["option_support"]["options"]:
            yes = option["key"] == "yes"
            option["distribution"]["probabilities"] = [0.8, 0.1, 0.1] if yes else [0.1, 0.8, 0.1]
            option["distribution"]["winner"] = "Support" if yes else "Refute"
            option["raw_support"] = 0.8 if yes else 0.1
            option["normalized_support"] = option["raw_support"] / 0.9

    def decide(views):
        records = [root, *views]
        kwargs = {}
        if coverage_mode == "per_prefix":
            kwargs["negative_coverage_for_records"] = (
                lambda prefix: _negative_candidate_coverage(prefix, ["a", "b"])
            )
        return select_accepted_hypothesis(
            records, root_state_id=0,
            acceptance=BranchAcceptanceConfig(0.65, 0.15, 0.6, aggregation),
            validation_enabled=True, root_fallback_answer="no", option_catalog=catalog,
            negative_coverage=_negative_candidate_coverage(records, ["a", "b"]),
            **kwargs,
        )

    assert _negative_candidate_coverage([root, *no_views], ["a", "b"]).region_union_ratio == 0.70
    assert _negative_candidate_coverage([root, *no_views, *yes_views], ["a", "b"]).region_union_ratio == 0.90
    assert decide(no_views).accepted_hypothesis is None
    assert decide(yes_views).answer == "yes"
    decision = decide([*no_views, *yes_views])
    assert decision.answer == "yes"
    assert decision.accepted_hypothesis.accepted_at_state_id == 4
    rejected = next(item for item in decision.ledger_trace["evaluations"] if item["state_id"] == 2)
    assert rejected["accepted"] is False
    assert "candidate_coverage" in rejected["negative_gate"]["reasons"]
    if coverage_mode == "per_prefix":
        assert rejected["negative_gate"]["coverage"] == 0.70
