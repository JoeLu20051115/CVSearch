import json
import unittest
from pathlib import Path

from cvsearch.eval.replay_adaptive_search import (
    FrozenCalibration,
    FrozenSelectedCalibration,
    freeze_isotonic_calibration,
    freeze_selected_calibration,
    replay_adaptive_search,
)
from cvsearch.evidence_gap.adaptive_controller import IsotonicCalibrator


ROOT = Path(__file__).resolve().parents[1]


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
    def test_vstar_unavailable_p0_repeatability_is_neutral_not_zero(self):
        step = action_step(
            "ZOOM", current=0.8, candidate=0.2, output=1,
            losses=[0.9, 0.1],
        )
        step["zoom_audit"]["p0_stability"].update({
            "frequency": 0.0,
            "aggregation_available": None,
        })
        phase1, observed = rows(step)
        replay = replay_adaptive_search(phase1, observed, calibration())

        candidate = replay["candidates"][0]
        self.assertEqual(candidate["raw_current_support"], 0.8)
        self.assertEqual(replay["selected_source"], "P0")

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

    def test_raw_progress_can_break_a_calibrated_plateau(self):
        phase1, observed = rows(action_step(
            "EXPAND", current=0.7, candidate=0.8, output=1,
            losses=[0.9, 0.1],
        ))
        replay = replay_adaptive_search(phase1, observed, calibration())

        self.assertEqual(replay["selected_source"], "EXPAND")
        self.assertEqual(replay["selected_output"], 1)
        self.assertEqual(replay["reason"], "calibrated_plateau_raw_progress")

    def test_stable_alternative_action_supporting_p0_vetoes_change(self):
        zoom = action_step(
            "ZOOM", current=0.7, candidate=0.65, output=0,
            losses=[0.1, 0.9],
        )
        expand = action_step(
            "EXPAND", current=0.7, candidate=0.8, output=1,
            losses=[0.9, 0.1],
        )
        phase1, observed = rows(zoom, expand)
        replay = replay_adaptive_search(phase1, observed, calibration())

        self.assertEqual(replay["selected_source"], "P0")
        self.assertEqual(replay["selected_output"], 0)
        self.assertEqual(replay["reason"], "stable_cross_action_conflict")

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

    def test_three_of_four_hr_votes_is_not_stable_enough_to_replace_p0(self):
        phase1, observed = rows(
            action_step(
                "EXPAND", current=0.2, candidate=0.9,
                output=["A", "B", "C", "C"], frequency=0.75,
            ),
            answer_type="option_list", output=["B", "A", "B", "C"],
            options=HR_OPTIONS,
        )
        replay = replay_adaptive_search(phase1, observed, calibration())

        self.assertEqual(replay["selected_source"], "P0")
        self.assertEqual(replay["reason"], "insufficient_calibrated_gain")

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
    def test_checked_in_development_manifest_is_self_authenticating(self):
        path = (
            ROOT / "reproduction" / "evidence_gap" / "adaptive_search_v2"
            / "calibration-manifest.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        frozen = FrozenCalibration(
            IsotonicCalibrator(
                tuple(payload["calibrator"]["upper_bounds"]),
                tuple(payload["calibrator"]["probabilities"]),
            ),
            payload["sample_count"],
            payload["manifest_sha256"],
        )
        self.assertEqual(frozen.to_dict(), payload)

    def test_frozen_prediction_keeps_raw_order_inside_isotonic_plateau(self):
        frozen = freeze_isotonic_calibration((
            {"row_id": "a", "raw_support": 0.2, "support_sufficient": 1},
            {"row_id": "b", "raw_support": 0.8, "support_sufficient": 1},
        ))
        self.assertLess(frozen.predict(0.3), frozen.predict(0.7))
        self.assertEqual(frozen.raw_tiebreak_weight, 1 / 3)
        self.assertEqual(
            frozen.to_dict()["prediction_rule"],
            "isotonic_plus_one_sample_raw_tiebreak_v1",
        )

    def test_manifest_is_deterministic_and_rejects_answer_correctness(self):
        first = calibration()
        second = calibration()
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(
            first.manifest_sha256,
            "285c4c54deefc9dc4cd6e6512b5359727e08ce0502921a5ed1edb69bf8c66b7c",
        )
        self.assertEqual(len(first.manifest_sha256), 64)
        with self.assertRaises(ValueError):
            freeze_isotonic_calibration(({
                "row_id": "bad", "raw_support": 0.5,
                "support_sufficient": 1, "answer_correct": True,
            },))


class SelectedCalibrationTest(unittest.TestCase):
    @staticmethod
    def _rows(samples):
        return tuple({
            "row_id": f"r{index}",
            "source_group": group,
            "raw_support": support,
            "support_sufficient": label,
        } for index, (group, support, label) in enumerate(samples))

    def test_grouped_selection_can_keep_exact_identity_calibration(self):
        frozen = freeze_selected_calibration(self._rows((
            ("g0", 0.05, 0), ("g0", 0.35, 0), ("g0", 0.95, 1),
            ("g1", 0.05, 0), ("g1", 0.05, 0), ("g1", 0.45, 1),
            ("g2", 0.25, 0), ("g2", 0.95, 1), ("g2", 0.85, 0),
        )))

        self.assertIsInstance(frozen, FrozenSelectedCalibration)
        self.assertEqual(frozen.selected_weight, 0.0)
        self.assertEqual(frozen.predict(0.314159), 0.314159)
        self.assertEqual(frozen.source_group_count, 3)
        self.assertEqual(frozen.to_dict()["schema_version"], 2)

    def test_grouped_selection_can_choose_positive_isotonic_weight(self):
        frozen = freeze_selected_calibration(self._rows((
            ("g0", 0.55, 0), ("g0", 0.65, 0), ("g0", 0.15, 0),
            ("g1", 0.55, 0), ("g1", 0.85, 0), ("g1", 0.05, 0),
            ("g2", 0.65, 1), ("g2", 0.15, 0), ("g2", 0.15, 1),
        )))

        self.assertEqual(frozen.selected_weight, 0.5)
        metrics = frozen.to_dict()["candidate_metrics"]
        selected = next(
            row for row in metrics if row["weight"] == frozen.selected_weight
        )
        identity = next(row for row in metrics if row["weight"] == 0.0)
        self.assertLess(selected["brier"], identity["brier"])

    def test_selected_manifest_is_deterministic_and_schema_is_exact(self):
        rows = self._rows((
            ("g0", 0.1, 0), ("g0", 0.9, 1),
            ("g1", 0.2, 0), ("g1", 0.8, 1),
        ))
        first = freeze_selected_calibration(rows)
        second = freeze_selected_calibration(rows)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(len(first.manifest_sha256), 64)

        for bad in (
            dict(rows[0], answer="A"),
            {key: value for key, value in rows[0].items() if key != "source_group"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                freeze_selected_calibration((bad, rows[1], rows[2], rows[3]))

    def test_replay_accepts_selected_calibration_without_changing_contract(self):
        frozen = freeze_selected_calibration(self._rows((
            ("g0", 0.05, 0), ("g0", 0.35, 0), ("g0", 0.95, 1),
            ("g1", 0.05, 0), ("g1", 0.05, 0), ("g1", 0.45, 1),
            ("g2", 0.25, 0), ("g2", 0.95, 1), ("g2", 0.85, 0),
        )))
        phase1, observed = rows(action_step(
            "ZOOM", current=0.1, candidate=0.9, output=1,
            losses=[0.9, 0.1],
        ))

        replay = replay_adaptive_search(phase1, observed, frozen)

        self.assertEqual(replay["selected_source"], "ZOOM")
        self.assertEqual(replay["selected_output"], 1)


if __name__ == "__main__":
    unittest.main()
