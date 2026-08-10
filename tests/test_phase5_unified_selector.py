import copy
import math
import unittest

from cvsearch.eval.phase5_unified_selector import select_unified_state


class UnifiedSelectorTests(unittest.TestCase):
    @staticmethod
    def p0(confidence=0.5, output=None):
        return {
            "action": "P0",
            "output": ["A", "B", "C", "D"] if output is None else output,
            "p0_stability": {"confidence": confidence},
        }

    @staticmethod
    def candidate(action, confidence, *, feasible=True, output=None):
        return {
            "action": action,
            "feasible": feasible,
            "output": [action] if output is None else output,
            "candidate_stability": {"confidence": confidence},
        }

    def test_exact_threshold_selects_candidate_without_changing_its_output(self):
        p0 = {
            "action": "P0",
            "output": ["A", "B", "C", "D"],
            "p0_stability": {"confidence": 0.5},
        }
        zoom = {
            "action": "ZOOM",
            "feasible": True,
            "output": ["B", "A", "D", "C"],
            "candidate_stability": {"confidence": 0.75},
        }

        decision = select_unified_state(p0, [zoom])

        self.assertEqual(decision.action, "ZOOM")
        self.assertEqual(decision.status, "selected_candidate")
        self.assertEqual(decision.output, ["B", "A", "D", "C"])
        self.assertEqual(decision.stability_gain, 0.25)

    def test_next_representable_gain_below_threshold_retains_exact_p0(self):
        p0 = self.p0(0.5, output={"raw_outputs": ["A", "B", "C", "D"]})
        candidate = self.candidate(
            "ZOOM", math.nextafter(0.75, -math.inf), output=["candidate"],
        )

        decision = select_unified_state(p0, [candidate])

        self.assertEqual(decision.action, "P0")
        self.assertEqual(decision.status, "retained_p0")
        self.assertEqual(decision.output, {"raw_outputs": ["A", "B", "C", "D"]})
        self.assertIsNone(decision.stability_gain)

    def test_largest_gain_wins_regardless_of_candidate_order(self):
        p0 = self.p0(0.2)
        zoom = self.candidate("ZOOM", 0.7)
        expand = self.candidate("EXPAND", 0.8)

        forward = select_unified_state(p0, [zoom, expand])
        reverse = select_unified_state(p0, [expand, zoom])

        self.assertEqual((forward.action, forward.output), ("EXPAND", ["EXPAND"]))
        self.assertEqual(reverse, forward)
        self.assertAlmostEqual(forward.stability_gain, 0.6)

    def test_exact_gain_tie_uses_frozen_action_name_order(self):
        p0 = self.p0(0.1)
        zoom = self.candidate("ZOOM", 0.6)
        expand = self.candidate("EXPAND", 0.6)

        forward = select_unified_state(p0, [zoom, expand])
        reverse = select_unified_state(p0, [expand, zoom])

        self.assertEqual(forward.action, "EXPAND")
        self.assertEqual(reverse, forward)

    def test_infeasible_candidate_is_never_admitted(self):
        p0 = self.p0(0.0)
        infeasible = {
            "action": "ZOOM",
            "feasible": False,
            "output": None,
            "candidate_stability": None,
        }

        decision = select_unified_state(p0, [infeasible])

        self.assertEqual(decision.action, "P0")
        self.assertEqual(decision.output, p0["output"])

    def test_selector_rejects_forbidden_metadata_before_selection(self):
        forbidden_fields = (
            "benchmark", "resolution", "category", "ordinal", "label",
            "target_box", "question_type", "answer", "ground_truth",
            "evaluator_labels",
        )
        for field in forbidden_fields:
            with self.subTest(field=field):
                candidate = self.candidate("ZOOM", 0.9)
                candidate[field] = "forbidden"
                with self.assertRaises(ValueError):
                    select_unified_state(self.p0(), [candidate])

    def test_selector_rejects_forbidden_metadata_nested_in_output(self):
        for field in ("benchmark", "resolution", "category", "ordinal", "label", "target"):
            with self.subTest(field=field):
                candidate = self.candidate(
                    "ZOOM", 0.9, output={"value": "A", field: "forbidden"},
                )
                with self.assertRaises(ValueError):
                    select_unified_state(self.p0(), [candidate])
                p0 = self.p0(output={"value": "A", field: "forbidden"})
                with self.assertRaises(ValueError):
                    select_unified_state(p0, [])

    def test_selector_rejects_nonfinite_bool_and_malformed_confidence(self):
        malformed = (True, False, None, "0.8", math.nan, math.inf, -math.inf, -0.1, 1.1)
        for value in malformed:
            with self.subTest(location="p0", value=value):
                with self.assertRaises((TypeError, ValueError)):
                    select_unified_state(self.p0(value), [])
            with self.subTest(location="candidate", value=value):
                with self.assertRaises((TypeError, ValueError)):
                    select_unified_state(self.p0(), [self.candidate("ZOOM", value)])

    def test_selector_rejects_noncanonical_dtos_and_duplicate_actions(self):
        cases = []
        wrong_p0 = self.p0()
        wrong_p0["action"] = "ZOOM"
        cases.append((wrong_p0, []))
        wrong_feasible = self.candidate("ZOOM", 0.9)
        wrong_feasible["feasible"] = 1
        cases.append((self.p0(), [wrong_feasible]))
        cases.append((self.p0(), [self.candidate("SPLIT", 0.9)]))
        cases.append((self.p0(), [
            self.candidate("ZOOM", 0.8), self.candidate("ZOOM", 0.9),
        ]))
        for p0, candidates in cases:
            with self.subTest(p0=p0, candidates=candidates):
                with self.assertRaises((TypeError, ValueError)):
                    select_unified_state(p0, candidates)

    def test_selector_never_mutates_or_aliases_input_records(self):
        p0 = self.p0(0.2, output={"raw_outputs": ["A", "B", "C", "D"]})
        candidates = [
            self.candidate("ZOOM", 0.6, output={"raw_outputs": ["B", "A", "D", "C"]}),
            self.candidate("EXPAND", 0.3, output=["unchanged"]),
        ]
        original_p0 = copy.deepcopy(p0)
        original_candidates = copy.deepcopy(candidates)

        decision = select_unified_state(p0, candidates)
        decision.output["raw_outputs"][0] = "mutated decision"
        fallback = select_unified_state(p0, [])
        fallback.output["raw_outputs"][0] = "mutated fallback"

        self.assertEqual(p0, original_p0)
        self.assertEqual(candidates, original_candidates)


if __name__ == "__main__":
    unittest.main()
