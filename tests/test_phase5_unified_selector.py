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

    def test_expand_exact_quarter_gain_is_admitted_without_output_change(self):
        p0 = {
            "action": "P0",
            "output": ["A", "B", "C", "D"],
            "p0_stability": {"confidence": 0.5},
        }
        expand = {
            "action": "EXPAND",
            "feasible": True,
            "output": ["B", "A", "D", "C"],
            "candidate_stability": {"confidence": 0.75},
        }

        decision = select_unified_state(p0, [expand])

        self.assertEqual(decision.action, "EXPAND")
        self.assertEqual(decision.status, "selected_candidate")
        self.assertEqual(decision.output, ["B", "A", "D", "C"])
        self.assertEqual(decision.stability_gain, 0.25)

    def test_zoom_exact_half_gain_is_rejected(self):
        decision = select_unified_state(
            self.p0(0.25), [self.candidate("ZOOM", 0.75)],
        )

        self.assertEqual(decision.action, "P0")
        self.assertEqual(decision.output, self.p0(0.25)["output"])
        self.assertIsNone(decision.stability_gain)

    def test_zoom_gain_above_half_is_admitted(self):
        decision = select_unified_state(
            self.p0(0.25),
            [self.candidate("ZOOM", math.nextafter(0.75, math.inf))],
        )

        self.assertEqual(decision.action, "ZOOM")
        self.assertGreater(decision.stability_gain, 0.5)

    def test_next_representable_gain_below_threshold_retains_exact_p0(self):
        p0 = self.p0(0.5, output={"raw_outputs": ["A", "B", "C", "D"]})
        candidate = self.candidate(
            "EXPAND", math.nextafter(0.75, -math.inf), output=["candidate"],
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
        zoom = self.candidate("ZOOM", 0.8)
        expand = self.candidate("EXPAND", 0.8)

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

    def test_selector_rejects_joined_and_camelcase_forbidden_keys(self):
        joined_fields = (
            "questionType", "resolution4K", "benchmarkName",
            "evaluatorLabel", "correctAnswer",
        )
        for field in joined_fields:
            with self.subTest(field=field):
                candidate = self.candidate(
                    "ZOOM", 0.9, output={"value": "A", field: "forbidden"},
                )
                with self.assertRaises(ValueError):
                    select_unified_state(self.p0(), [candidate])

    def test_selector_rejects_container_subclasses_before_their_hooks_run(self):
        class SwitchingDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.items_calls = 0
                self.getitem_calls = 0

            def items(self):
                self.items_calls += 1
                return super().items()

            def __getitem__(self, key):
                self.getitem_calls += 1
                if key == "candidate_stability":
                    return {"confidence": 1.0}
                return super().__getitem__(key)

        class SwitchingList(list):
            def __iter__(self):
                raise AssertionError("untrusted list iterator was executed")

        class TupleSubclass(tuple):
            pass

        candidate = SwitchingDict(self.candidate("ZOOM", 0.5))
        with self.assertRaises(TypeError):
            select_unified_state(self.p0(), [candidate])
        self.assertEqual((candidate.items_calls, candidate.getitem_calls), (0, 0))

        p0 = SwitchingDict(self.p0())
        with self.assertRaises(TypeError):
            select_unified_state(p0, [])
        self.assertEqual((p0.items_calls, p0.getitem_calls), (0, 0))

        candidates = SwitchingList([self.candidate("ZOOM", 0.9)])
        with self.assertRaises(TypeError):
            select_unified_state(self.p0(), candidates)
        with self.assertRaises(TypeError):
            select_unified_state(
                self.p0(), TupleSubclass((self.candidate("ZOOM", 0.9),)),
            )

    def test_selector_snapshots_only_exact_finite_json_builtins(self):
        class DictSubclass(dict):
            pass

        class ListSubclass(list):
            pass

        class StringSubclass(str):
            pass

        class IntSubclass(int):
            pass

        malformed_outputs = (
            DictSubclass({"value": "A"}),
            ListSubclass(["A"]),
            StringSubclass("A"),
            IntSubclass(1),
            math.nan,
            math.inf,
        )
        for output in malformed_outputs:
            with self.subTest(output=output):
                with self.assertRaises((TypeError, ValueError)):
                    select_unified_state(
                        self.p0(), [self.candidate("ZOOM", 0.9, output=output)],
                    )

        nested_stability = DictSubclass({"confidence": 0.9})
        candidate = self.candidate("ZOOM", 0.9)
        candidate["candidate_stability"] = nested_stability
        with self.assertRaises(TypeError):
            select_unified_state(self.p0(), [candidate])

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
