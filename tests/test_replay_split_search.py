import copy
import hashlib
import json
import unittest

from cvsearch.eval.replay_adaptive_search import freeze_selected_calibration
from cvsearch.eval.replay_split_search import select_split_candidate
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


class SplitReplayTest(unittest.TestCase):
    POLICY = {"minimum_final_support": 0.60, "minimum_support_gain": 0.10}

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


if __name__ == "__main__":
    unittest.main()
