import copy
import hashlib
import json
import unittest

from cvsearch.eval.replay_adaptive_search import freeze_selected_calibration
from cvsearch.eval.replay_split_search import (
    _candidate_output,
    select_split_candidate,
    select_split_candidate_cascade,
)
from tests.test_evidence_gap_split_observation import audit


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def calibration():
    return freeze_selected_calibration([
        {"row_id": "a", "source_group": "g1", "raw_support": 0.1,
         "support_sufficient": 0},
        {"row_id": "b", "source_group": "g1", "raw_support": 0.9,
         "support_sufficient": 1},
        {"row_id": "c", "source_group": "g2", "raw_support": 0.2,
         "support_sufficient": 0},
        {"row_id": "d", "source_group": "g2", "raw_support": 0.8,
         "support_sufficient": 1},
    ])


def positive_calibration():
    """Keep calibrated support high even when the raw observation is weak."""
    return freeze_selected_calibration([
        {"row_id": "a", "source_group": "g1", "raw_support": 0.1,
         "support_sufficient": 1},
        {"row_id": "b", "source_group": "g1", "raw_support": 0.9,
         "support_sufficient": 1},
        {"row_id": "c", "source_group": "g2", "raw_support": 0.2,
         "support_sufficient": 1},
        {"row_id": "d", "source_group": "g2", "raw_support": 0.8,
         "support_sufficient": 1},
    ])


def rows():
    ranks = [{"identity": "p0", "score": 0.9}]
    rank_sha256 = digest(ranks)
    phase1 = {
        "_eg_ordinal": 7,
        "answer_type": "option_single",
        "options": "A. red\nB. blue",
        "output": "A",
        "method_trace": {"candidate_ranks": copy.deepcopy(ranks)},
    }
    split_audit = audit().to_dict()
    split_audit["rank_sha256"] = rank_sha256
    second = copy.deepcopy(split_audit["branches"][0])
    second["visit_index"] = 1
    second["backtracked"] = True
    second["observed_path"] = [1]
    second["tight_view"]["patch_path"] = [1]
    second["context_view"]["patch_path"] = [1]
    second["tight_view"]["render_sha256"] = "5" * 64
    second["context_view"]["render_sha256"] = "6" * 64
    split_audit["branches"].append(second)
    split = copy.deepcopy(phase1)
    split["method_trace"] = {
        "candidate_ranks": copy.deepcopy(ranks),
        "query_plan": {"evidence_items": [{
            "kind": "target_detail",
            "target": "small sign",
            "requirements": ["presence", "visual_detail"],
        }]},
        "steps": [{"action": "SPLIT", "split_search_audit": split_audit}],
    }
    return phase1, split


def make_hr(phase1, split):
    option_block = "A. red\nB. blue\nC. green\nD. yellow"
    for row in (phase1, split):
        row["answer_type"] = "option_list"
        row["options"] = [option_block] * 4
        row["output"] = ["A"] * 4


def rescue_rows(*, prefix_answer="A", rescue_answer="B"):
    phase1, split = rows()
    split_audit = split["method_trace"]["steps"][0]["split_search_audit"]
    while len(split_audit["branches"]) < 4:
        index = len(split_audit["branches"])
        candidate = copy.deepcopy(split_audit["branches"][0])
        candidate["visit_index"] = index
        candidate["backtracked"] = True
        candidate["observed_path"] = [index % 2, index]
        for role, digit in (("tight", index + 1), ("context", index + 5)):
            candidate[f"{role}_view"]["patch_path"] = candidate["observed_path"]
            candidate[f"{role}_view"]["answer"] = prefix_answer
            candidate[f"{role}_view"]["render_sha256"] = format(digit, "x") * 64
        split_audit["branches"].append(candidate)
    for candidate in split_audit["branches"]:
        candidate["tight_view"]["answer"] = prefix_answer
        candidate["context_view"]["answer"] = prefix_answer
        if prefix_answer == "A":
            candidate["tight_view"]["raw_support"] = 0.10
            candidate["context_view"]["raw_support"] = 0.20
    for index, root in ((4, 2), (5, 3)):
        candidate = copy.deepcopy(split_audit["branches"][0])
        candidate["visit_index"] = index
        candidate["backtracked"] = True
        candidate["observed_path"] = [root, 0]
        for offset, role in enumerate(("tight", "medium", "context")):
            source = copy.deepcopy(candidate[
                "context_view" if role == "context" else "tight_view"
            ])
            source["role"] = role
            source["patch_path"] = [root, 0]
            source["answer"] = (
                rescue_answer if index == 4 and role != "context" else prefix_answer
            )
            source["raw_support"] = (
                0.90 + 0.02 * offset
                if index == 4 and role != "context"
                else 0.10 + 0.02 * offset
            )
            source["render_sha256"] = hashlib.sha256(
                f"rescue-{index}-{role}".encode()
            ).hexdigest()
            candidate[f"{role}_view"] = source
        split_audit["branches"].append(candidate)
    split_audit["render_policy"] = (
        "native_2x2_overlap_support_screen_three_scale_all_roots_depth2_v3"
    )
    split_audit["max_observed_branches"] = 6
    split_audit["max_screening_probes"] = 16
    return phase1, split


class SplitReplayTest(unittest.TestCase):
    POLICY = {
        "minimum_final_support": 0.60,
        "minimum_support_gain": 0.10,
        "maximum_support_drop": 0.0,
        "minimum_conflict_margin": 1.0,
        "minimum_uncontested_support": 1.0,
        "minimum_consensus_raw_support": 0.8,
        "minimum_p0_uncertainty": 0.2,
        "minimum_local_raw_support": 0.2,
    }

    STATE_POLICY = {
        **POLICY,
        "minimum_state_calibrated_support": 0.25,
        "minimum_state_raw_support": 0.4,
        "minimum_state_vote_margin": 0.0,
    }

    def test_cascade_returns_v3_before_inspecting_malformed_rescue(self):
        phase1, split = rescue_rows(prefix_answer="B")
        split["method_trace"]["steps"][0]["split_search_audit"]["branches"][
            4
        ]["medium_view"]["render_sha256"] = "bad"
        expected_prefix = copy.deepcopy(split)
        expected_prefix["method_trace"]["steps"][0]["split_search_audit"][
            "branches"
        ] = expected_prefix["method_trace"]["steps"][0]["split_search_audit"][
            "branches"
        ][:4]
        expected = select_split_candidate(
            phase1, expected_prefix, calibration(), self.POLICY,
        )
        selected = select_split_candidate_cascade(
            phase1, split, calibration(), calibration(), self.POLICY,
            {**self.POLICY, "minimum_conflict_margin": 0.05},
        )
        self.assertEqual(selected, expected)
        self.assertEqual(selected["selected_source"], "SPLIT")
        self.assertLess(selected["selected_branch"], 4)

    def test_cascade_uses_three_scale_rescue_only_after_v3_abstains(self):
        phase1, split = rescue_rows()
        selected = select_split_candidate_cascade(
            phase1, split, calibration(), calibration(), self.POLICY,
            {**self.POLICY, "minimum_conflict_margin": 0.05},
        )
        self.assertEqual(selected["selected_output"], "B")
        self.assertEqual(selected["selected_source"], "SPLIT")
        self.assertEqual(selected["reason"], "stage3b_confirmed_three_scale_rescue")
        self.assertEqual(selected["selected_branch"], 4)
        self.assertEqual(selected["rescue_votes"], 2)
        self.assertTrue(selected["used_backtrack"])

    def test_cascade_fails_closed_on_invalid_rescue_when_v3_abstains(self):
        phase1, split = rescue_rows()
        split["method_trace"]["steps"][0]["split_search_audit"]["branches"][
            4
        ]["medium_view"]["render_sha256"] = "bad"
        expected_prefix = copy.deepcopy(split)
        expected_prefix["method_trace"]["steps"][0]["split_search_audit"][
            "branches"
        ] = expected_prefix["method_trace"]["steps"][0]["split_search_audit"][
            "branches"
        ][:4]
        expected = select_split_candidate(
            phase1, expected_prefix, calibration(), self.POLICY,
        )
        selected = select_split_candidate_cascade(
            phase1, split, calibration(), calibration(), self.POLICY,
            {**self.POLICY, "minimum_conflict_margin": 0.05},
        )
        self.assertEqual(selected, expected)

    def test_cascade_rejects_rescue_with_equally_strong_p0_evidence(self):
        phase1, split = rescue_rows()
        for candidate in split["method_trace"]["steps"][0][
            "split_search_audit"
        ]["branches"][:4]:
            candidate["tight_view"]["raw_support"] = 0.90
            candidate["context_view"]["raw_support"] = 0.92
        selected = select_split_candidate_cascade(
            phase1, split, calibration(), calibration(), self.POLICY,
            {**self.POLICY, "minimum_conflict_margin": 0.05},
        )
        self.assertEqual(selected["selected_output"], "A")
        self.assertEqual(selected["selected_source"], "P0")

    def test_cascade_does_not_treat_one_disagreeing_scale_as_p0_consensus(self):
        phase1, split = rescue_rows()
        branch = split["method_trace"]["steps"][0]["split_search_audit"][
            "branches"
        ][0]
        branch["tight_view"]["answer"] = "A"
        branch["tight_view"]["raw_support"] = 0.90
        branch["context_view"]["answer"] = "C"
        branch["context_view"]["raw_support"] = 0.10
        selected = select_split_candidate_cascade(
            phase1, split, calibration(), calibration(), self.POLICY,
            {**self.POLICY, "minimum_conflict_margin": 0.05},
        )
        self.assertEqual(selected["selected_output"], "B")
        self.assertEqual(selected["selected_source"], "SPLIT")
        self.assertEqual(selected["selected_branch"], 4)

    def test_cascade_never_uses_split_calibration_for_p0_support(self):
        phase1, split = rescue_rows()
        selected = select_split_candidate_cascade(
            phase1, split, calibration(), positive_calibration(), self.POLICY,
            {
                **self.POLICY, "minimum_conflict_margin": 0.05,
                "minimum_support_gain": 0.15,
            },
        )
        self.assertEqual(selected["selected_output"], "B")
        self.assertEqual(selected["selected_source"], "SPLIT")
        self.assertEqual(selected["selected_branch"], 4)

    def test_cascade_admits_cross_branch_state_competition_after_trajectory_abstains(self):
        phase1, split = rescue_rows()
        phase1["options"] = split["options"] = (
            "A. red\nB. blue\nC. green\nD. yellow"
        )
        branches = split["method_trace"]["steps"][0]["split_search_audit"][
            "branches"
        ]
        answers = (("B", "A"), ("B", "A"), ("C", "D"), ("C", "D"))
        for branch, (tight, context) in zip(branches[:4], answers):
            branch["tight_view"]["answer"] = tight
            branch["context_view"]["answer"] = context
        for role, answer in (("tight", "C"), ("medium", "D"), ("context", "C")):
            branches[5][f"{role}_view"]["answer"] = answer
        selected = select_split_candidate_cascade(
            phase1, split, calibration(), calibration(), self.POLICY,
            {**self.STATE_POLICY, "minimum_support_gain": 1.0},
        )
        self.assertEqual(selected["selected_output"], "B")
        self.assertEqual(selected["selected_source"], "SPLIT")
        self.assertEqual(
            selected["reason"], "stage3b_cross_branch_state_competition",
        )
        self.assertEqual(selected["selected_branch"], 4)
        self.assertEqual(selected["rescue_votes"], 2)
        self.assertGreaterEqual(selected["candidate_global_votes"], 4)
        self.assertGreaterEqual(
            selected["candidate_global_votes"], selected["p0_global_votes"],
        )
        self.assertGreaterEqual(selected["candidate_branch_count"], 2)

    def test_cascade_state_competition_requires_cross_branch_vote_dominance(self):
        phase1, split = rescue_rows()
        selected = select_split_candidate_cascade(
            phase1, split, calibration(), calibration(), self.POLICY,
            {**self.STATE_POLICY, "minimum_support_gain": 1.0},
        )
        self.assertEqual(selected["selected_output"], "A")
        self.assertEqual(selected["selected_source"], "P0")

    def test_cascade_state_competition_never_rewrites_the_same_semantic_answer(self):
        phase1, split = rescue_rows(rescue_answer="A")
        branch = split["method_trace"]["steps"][0]["split_search_audit"][
            "branches"
        ][4]
        for role in ("tight", "medium", "context"):
            branch[f"{role}_view"]["raw_support"] = 0.9
        selected = select_split_candidate_cascade(
            phase1, split, calibration(), calibration(), self.POLICY,
            {**self.STATE_POLICY, "minimum_support_gain": 1.0},
        )
        self.assertEqual(selected["selected_output"], "A")
        self.assertEqual(selected["selected_source"], "P0")

    def test_selects_two_view_confirmed_split_after_frozen_stage2(self):
        phase1, split = rows()
        selected = select_split_candidate(
            phase1, split, calibration(), self.POLICY,
        )
        self.assertEqual(selected["stage2_selected_output"], "A")
        self.assertEqual(selected["selected_output"], "B")
        self.assertEqual(selected["selected_source"], "SPLIT")
        self.assertEqual(selected["reason"], "confirmed_two_view_trajectory")
        self.assertFalse(selected["used_backtrack"])
        self.assertEqual(selected["branches"][0]["trajectory_s"], 3)

    def test_rejects_calibrated_trajectory_with_weak_raw_visual_support(self):
        phase1, split = rows()
        audit_value = split["method_trace"]["steps"][0]["split_search_audit"]
        for branch in audit_value["branches"]:
            branch["tight_view"]["raw_support"] = 0.01
            branch["context_view"]["raw_support"] = 0.02
        audit_value["p0_stability"]["confidence"] = 0.0
        result = select_split_candidate(
            phase1, split, positive_calibration(), {
                **self.POLICY,
                "minimum_support_gain": 0.0,
            },
        )
        self.assertEqual(result["selected_output"], "A")
        self.assertEqual(result["selected_source"], "P0")
        self.assertEqual(result["reason"], "weak_raw_visual_support")

    def test_rank_drift_missing_calibration_and_malformed_audit_retain_stage2(self):
        phase1, split = rows()
        cases = []
        drift = copy.deepcopy(split)
        drift["method_trace"]["candidate_ranks"].append({"identity": "p1"})
        cases.append((drift, calibration(), "phase1_rank_drift"))
        cases.append((split, None, "calibration_unavailable"))
        malformed = copy.deepcopy(split)
        malformed["method_trace"]["steps"][0]["split_search_audit"][
            "branches"
        ][0]["tight_view"]["render_sha256"] = "bad"
        cases.append((malformed, calibration(), "split_audit_invalid"))
        for candidate, calibrator, reason in cases:
            with self.subTest(reason=reason):
                result = select_split_candidate(
                    phase1, candidate, calibrator, self.POLICY,
                )
                self.assertEqual(result["selected_output"], "A")
                self.assertEqual(result["selected_source"], "P0")
                self.assertEqual(result["reason"], reason)

    def test_disagreement_uses_second_branch_and_marks_backtrack(self):
        phase1, split = rows()
        branches = split["method_trace"]["steps"][0]["split_search_audit"][
            "branches"
        ]
        branches[0]["context_view"]["answer"] = "C"
        result = select_split_candidate(
            phase1, split, calibration(), self.POLICY,
        )
        self.assertEqual(result["selected_output"], "B")
        self.assertTrue(result["used_backtrack"])
        self.assertEqual(result["selected_branch"], 1)

    def test_unparseable_branch_is_skipped_without_invalidating_later_evidence(self):
        phase1, split = rows()
        branches = split["method_trace"]["steps"][0]["split_search_audit"][
            "branches"
        ]
        branches[0]["tight_view"]["answer"] = "not an option"
        branches[0]["context_view"]["answer"] = "not an option"
        result = select_split_candidate(
            phase1, split, calibration(), self.POLICY,
        )
        self.assertEqual(result["selected_output"], "B")
        self.assertEqual(result["selected_branch"], 1)
        self.assertEqual(
            result["branches"][0]["confirmation_reason"], "unparseable_view",
        )

    def test_support_selected_branch_needs_positive_gain_not_counterevidence(self):
        phase1, split = rows()
        audit_value = split["method_trace"]["steps"][0]["split_search_audit"]
        branches = audit_value["branches"]
        for branch_value in branches:
            branch_value["tight_view"]["answer"] = "A"
            branch_value["context_view"]["answer"] = "A"
            branch_value["tight_view"]["raw_support"] = 0.1
            branch_value["context_view"]["raw_support"] = 0.1
        audit_value["p0_stability"]["confidence"] = 0.9
        audit_value["p0_stability"]["uncertainty"] = 0.1
        third = copy.deepcopy(branches[0])
        third["visit_index"] = 2
        third["backtracked"] = True
        third["observed_path"] = [2]
        for role, digest in (("tight_view", "7"), ("context_view", "8")):
            third[role]["patch_path"] = [2]
            third[role]["answer"] = "B"
            third[role]["raw_support"] = 0.85
            third[role]["render_sha256"] = digest * 64
        branches.append(third)
        result = select_split_candidate(
            phase1, split, calibration(), {
                **self.POLICY,
                "maximum_support_drop": 0.1,
                "minimum_conflict_margin": 0.1,
            },
        )
        self.assertEqual(result["selected_output"], "A")
        self.assertEqual(
            result["reason"],
            "support_selected_counterevidence_requires_positive_gain",
        )

    def test_local_trajectory_can_beat_a_weaker_p0_conflict(self):
        phase1, split = rows()
        make_hr(phase1, split)
        audit_value = split["method_trace"]["steps"][0]["split_search_audit"]
        branches = audit_value["branches"]
        for branch_value in branches:
            branch_value["tight_view"]["answer"] = ["A"] * 4
            branch_value["context_view"]["answer"] = ["A"] * 4
            branch_value["tight_view"]["raw_support"] = 0.1
            branch_value["context_view"]["raw_support"] = 0.1
        audit_value["p0_stability"]["confidence"] = 0.75
        audit_value["p0_stability"]["uncertainty"] = 0.25
        third = copy.deepcopy(branches[0])
        third["visit_index"] = 2
        third["backtracked"] = True
        third["observed_path"] = [2]
        for role, raw_support, digest_value in (
            ("tight_view", 0.3, "7"),
            ("context_view", 0.4, "8"),
        ):
            third[role]["patch_path"] = [2]
            third[role]["answer"] = ["B"] * 4
            third[role]["raw_support"] = raw_support
            third[role]["render_sha256"] = digest_value * 64
        branches.append(third)
        result = select_split_candidate(
            phase1, split, calibration(), {
                **self.POLICY,
                "minimum_conflict_margin": 0.1,
            },
        )
        self.assertEqual(result["selected_output"], ["B"] * 4)
        self.assertEqual(
            result["reason"],
            "confirmed_uncertain_p0_local_trajectory",
        )
        self.assertEqual(result["selected_branch"], 2)
        third["context_view"]["raw_support"] = 0.15
        weak = select_split_candidate(
            phase1, split, calibration(), {
                **self.POLICY,
                "minimum_conflict_margin": 0.1,
            },
        )
        self.assertEqual(weak["selected_output"], ["A"] * 4)
        self.assertNotEqual(
            weak["reason"], "confirmed_uncertain_p0_local_trajectory",
        )

    def test_structured_vote_blocks_counterevidence_on_a_stable_p0(self):
        phase1, split = rows()
        make_hr(phase1, split)
        audit_value = split["method_trace"]["steps"][0]["split_search_audit"]
        audit_value["p0_stability"]["confidence"] = 1.0
        audit_value["p0_stability"]["uncertainty"] = 0.0
        branches = audit_value["branches"]
        for role in ("tight_view", "context_view"):
            branches[0][role]["answer"] = ["B"] * 4
            branches[0][role]["raw_support"] = 0.9
            branches[1][role]["answer"] = ["A"] * 4
            branches[1][role]["raw_support"] = 0.1
        result = select_split_candidate(
            phase1, split, calibration(), {
                **self.POLICY,
                "maximum_support_drop": 0.1,
                "minimum_conflict_margin": 0.1,
                "minimum_uncontested_support": 0.8,
            },
        )
        self.assertEqual(result["selected_output"], ["A"] * 4)
        self.assertEqual(
            result["reason"],
            "structured_vote_counterevidence_requires_uncertain_p0_trajectory",
        )

    def test_cross_branch_plurality_can_confirm_three_independent_views(self):
        phase1, split = rows()
        audit_value = split["method_trace"]["steps"][0]["split_search_audit"]
        audit_value["p0_stability"]["confidence"] = 0.9
        audit_value["p0_stability"]["uncertainty"] = 0.1
        template = copy.deepcopy(audit_value["branches"][0])
        pairs = (("A", "A"), ("B", "C"), ("B", "C"), ("B", "D"))
        branches = []
        for index, pair in enumerate(pairs):
            branch_value = copy.deepcopy(template)
            branch_value["visit_index"] = index
            branch_value["backtracked"] = index > 0
            branch_value["observed_path"] = [index]
            for offset, (role, answer) in enumerate(zip(
                ("tight_view", "context_view"), pair,
            )):
                branch_value[role]["patch_path"] = [index]
                branch_value[role]["answer"] = answer
                branch_value[role]["raw_support"] = 0.85
                branch_value[role]["render_sha256"] = format(
                    2 * index + offset + 1, "x",
                ) * 64
            branches.append(branch_value)
        audit_value["branches"] = branches
        result = select_split_candidate(
            phase1, split, calibration(), {
                **self.POLICY,
                "maximum_support_drop": 0.1,
            },
        )
        self.assertEqual(result["selected_output"], "B")
        self.assertEqual(result["reason"], "confirmed_cross_branch_consensus")
        self.assertEqual(result["consensus"]["votes"], 3)
        self.assertEqual(result["consensus"]["distinct_paths"], 3)

    def test_cross_branch_consensus_rejects_weak_raw_support(self):
        phase1, split = rows()
        result = select_split_candidate(
            phase1, split, calibration(), {
                **self.POLICY,
                "minimum_consensus_raw_support": 0.9,
            },
        )
        self.assertNotEqual(result["reason"], "confirmed_cross_branch_consensus")

    def test_structured_vote_does_not_use_cross_branch_consensus(self):
        phase1, split = rows()
        make_hr(phase1, split)
        audit_value = split["method_trace"]["steps"][0]["split_search_audit"]
        audit_value["p0_stability"]["confidence"] = 0.9
        audit_value["p0_stability"]["uncertainty"] = 0.1
        template = copy.deepcopy(audit_value["branches"][0])
        branches = []
        for index in range(4):
            branch_value = copy.deepcopy(template)
            branch_value["visit_index"] = index
            branch_value["backtracked"] = index > 0
            branch_value["observed_path"] = [index]
            for offset, role in enumerate(("tight_view", "context_view")):
                branch_value[role]["patch_path"] = [index]
                branch_value[role]["answer"] = ["B"] * 4
                branch_value[role]["raw_support"] = 0.85
                branch_value[role]["render_sha256"] = format(
                    2 * index + offset + 1, "x",
                ) * 64
            branches.append(branch_value)
        audit_value["branches"] = branches
        result = select_split_candidate(phase1, split, calibration(), {
            **self.POLICY,
            "maximum_support_drop": 0.1,
        })
        self.assertEqual(result["selected_output"], ["A"] * 4)
        self.assertNotEqual(result["reason"], "confirmed_cross_branch_consensus")

    def test_equally_strong_p0_conflict_blocks_answer_change(self):
        phase1, split = rows()
        branches = split["method_trace"]["steps"][0]["split_search_audit"][
            "branches"
        ]
        branches[1]["tight_view"]["answer"] = "A"
        branches[1]["context_view"]["answer"] = "A"
        branches[1]["tight_view"]["raw_support"] = 0.70
        branches[1]["context_view"]["raw_support"] = 0.85
        result = select_split_candidate(
            phase1, split, calibration(), self.POLICY,
        )
        self.assertEqual(result["selected_output"], "A")
        self.assertEqual(result["reason"], "equally_strong_p0_conflict")

    def test_evaluator_labels_do_not_change_the_label_blind_decision(self):
        phase1, split = rows()
        first = select_split_candidate(
            phase1, split, calibration(), self.POLICY,
        )
        phase1["answer"] = "A"
        phase1["correct"] = True
        split["answer"] = "B"
        split["bbox"] = [[1, 2, 3, 4]]
        second = select_split_candidate(
            phase1, split, calibration(), self.POLICY,
        )
        self.assertEqual(first, second)

    def test_policy_rejects_benchmark_or_backbone_routes(self):
        phase1, split = rows()
        for key in ("benchmark", "backbone", "answer", "category"):
            policy = dict(self.POLICY, **{key: "forbidden"})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "exact"):
                select_split_candidate(phase1, split, calibration(), policy)

    def test_hr_projection_replaces_one_aligned_semantic_vote(self):
        output = _candidate_output(
            {"answer_type": "option_list"},
            ["A", "B", "C", "D"],
            "red mailbox",
            {"tight": {"output": ["B", "C", "D", "A"]}},
            "blue mailbox",
        )
        self.assertEqual(output, ["B", "C", "D", "A"])


if __name__ == "__main__":
    unittest.main()
