import json
import unittest

from cvsearch.eval.replay_adaptive_search import (
    freeze_isotonic_calibration,
    replay_adaptive_search,
)


HR_OPTIONS = [
    "A. cat\nB. dog\nC. bird\nD. fish",
    "A. dog\nB. cat\nC. fish\nD. bird",
    "A. bird\nB. dog\nC. cat\nD. fish",
    "A. fish\nB. bird\nC. dog\nD. cat",
]


def calibration():
    return freeze_isotonic_calibration((
        {"row_id": "c0", "raw_support": 0.1, "support_sufficient": 0},
        {"row_id": "c1", "raw_support": 0.3, "support_sufficient": 0},
        {"row_id": "c2", "raw_support": 0.7, "support_sufficient": 1},
        {"row_id": "c3", "raw_support": 0.9, "support_sufficient": 1},
    ))


def action_step(action, *, current, candidate, output, frequency=1.0, losses=None):
    candidate_answer = output if losses is None else {
        "losses": losses,
        "winner": min(range(len(losses)), key=losses.__getitem__),
    }
    audit = {
        "batch_result": {
            "status": "success",
            "candidate_answer": candidate_answer,
        },
        "current_gap_support": {"p_yes": current, "p_no": 1.0 - current},
        "candidate_gap_support": {"p_yes": candidate, "p_no": 1.0 - candidate},
        "p0_stability": {"frequency": 1.0, "aggregation_available": True},
        "candidate_stability": {
            "output": output,
            "frequency": frequency,
            "aggregation_available": True,
        },
        "normalized_actual_cost": 0.1,
    }
    return {
        "action": action,
        "zoom_audit" if action == "ZOOM" else "expand_audit": audit,
    }


def rows(*steps, answer_type="logits_match", output=0, options=None, evidence_items=None):
    ranks = [{"candidate_key": "patch-a", "final_rank": 1}]
    phase1 = {
        "output": output,
        "answer_type": answer_type,
        "options": ["red", "blue"] if options is None else options,
        "method_trace": {"candidate_ranks": ranks},
    }
    observed = {
        "output": output,
        "answer_type": answer_type,
        "options": phase1["options"],
        "method_trace": {
            "candidate_ranks": ranks,
            "query_plan": {"evidence_items": evidence_items or [{
                "kind": "target_detail",
                "target": "sign",
                "requirements": ["presence", "visual_detail"],
            }]},
            "steps": list(steps),
        },
    }
    return phase1, observed


class AdaptiveReplayTest(unittest.TestCase):
    def test_missing_calibration_or_candidates_retains_exact_phase1(self):
        phase1, observed = rows(action_step(
            "ZOOM", current=0.1, candidate=0.9, output=1,
            losses=[0.9, 0.1],
        ))
        uncalibrated = replay_adaptive_search(phase1, observed, None)
        no_candidates = replay_adaptive_search(
            phase1, rows()[1], calibration(),
        )

        self.assertEqual(uncalibrated["selected_output"], phase1["output"])
        self.assertEqual(uncalibrated["selected_source"], "P0")
        self.assertEqual(uncalibrated["reason"], "calibration_unavailable")
        self.assertEqual(no_candidates["selected_output"], phase1["output"])
        self.assertEqual(no_candidates["reason"], "no_admitted_candidate")

    def test_strong_observed_gain_overrides_detail_soft_prior(self):
        zoom = action_step(
            "ZOOM", current=0.2, candidate=0.3, output=0,
            losses=[0.1, 0.9],
        )
        expand = action_step(
            "EXPAND", current=0.2, candidate=0.9, output=1,
            losses=[0.9, 0.1],
        )
        phase1, observed = rows(zoom, expand)
        replay = replay_adaptive_search(phase1, observed, calibration())

        self.assertEqual(replay["demand"]["detail"], 1.0)
        self.assertGreater(replay["demand"]["detail"], replay["demand"]["context"])
        self.assertEqual(replay["selected_source"], "EXPAND")
        self.assertEqual(replay["selected_output"], 1)
        self.assertEqual(replay["reason"], "calibrated_support_gain")

    def test_hr_outputs_are_compared_and_selected_by_semantic_answer(self):
        candidate = ["A", "B", "C", "D"]
        p0 = ["B", "A", "B", "C"]
        phase1, observed = rows(
            action_step(
                "EXPAND", current=0.2, candidate=0.9, output=candidate,
            ),
            answer_type="option_list", output=p0, options=HR_OPTIONS,
            evidence_items=[{
                "kind": "relation_context", "targets": ["cat", "table"],
            }],
        )
        replay = replay_adaptive_search(phase1, observed, calibration())

        self.assertEqual(replay["selected_output"], candidate)
        selected = next(
            item for item in replay["candidates"] if item["action"] == "EXPAND"
        )
        self.assertEqual(selected["canonical_answer"], "cat")
        self.assertEqual(selected["answer_consistency"], 1.0)

    def test_rank_drift_and_evaluator_metadata_cannot_change_decision(self):
        step = action_step(
            "ZOOM", current=0.1, candidate=0.9, output=1,
            losses=[0.9, 0.1],
        )
        phase1, observed = rows(step)
        first = replay_adaptive_search(phase1, dict(observed, answer=0), calibration())
        second = replay_adaptive_search(
            phase1, dict(observed, answer=1, benchmark="poison"), calibration(),
        )
        self.assertEqual(first, second)
        self.assertNotIn("poison", json.dumps(first))

        drifted = json.loads(json.dumps(observed))
        drifted["method_trace"]["candidate_ranks"][0]["final_rank"] = 2
        fallback = replay_adaptive_search(phase1, drifted, calibration())
        self.assertEqual(fallback["selected_source"], "P0")
        self.assertEqual(fallback["reason"], "phase1_rank_drift")


class FrozenCalibrationTest(unittest.TestCase):
    def test_manifest_is_deterministic_and_rejects_answer_correctness(self):
        first = calibration()
        second = calibration()
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(len(first.manifest_sha256), 64)
        with self.assertRaises(ValueError):
            freeze_isotonic_calibration(({
                "row_id": "bad", "raw_support": 0.5,
                "support_sufficient": 1, "answer_correct": True,
            },))


if __name__ == "__main__":
    unittest.main()
