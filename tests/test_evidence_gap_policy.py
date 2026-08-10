import math
import unittest

import numpy as np

from cvsearch.evidence_gap.types import (
    BACKTRACK,
    EXPAND,
    FORCED_RETURN,
    NEXT,
    SPLIT,
    ZOOM,
    AnswerRecord,
    HistoryRecord,
)
from cvsearch.evidence_gap.policy import (
    CERTIFICATION_GAP_ACTIONS,
    HistoryBuffer,
    PolicyState,
    StopThresholds,
    feasible_actions,
    select_root_or_search,
    should_certify,
)


class EvidenceGapPolicyTest(unittest.TestCase):
    def test_feasible_actions_defaults_to_next_only_when_queue_has_work(self):
        self.assertEqual(feasible_actions(PolicyState(has_unvisited_next=True)), (NEXT,))

    def test_feasible_actions_forces_return_after_budget_exhaustion(self):
        state = PolicyState(budget_exhausted=True, has_unvisited_next=True)
        self.assertEqual(feasible_actions(state), (FORCED_RETURN,))

    def test_feasible_actions_suppresses_failed_noop_actions(self):
        state = PolicyState(has_unvisited_next=True, failed_actions=(NEXT,))
        self.assertEqual(feasible_actions(state), (FORCED_RETURN,))

    def test_feasible_actions_enables_advanced_actions_only_when_enabled_and_available(self):
        state = PolicyState(
            zoom_available=True,
            zoom_enabled=True,
            split_available=True,
            split_enabled=True,
            expand_available=True,
            expand_enabled=True,
        )
        self.assertEqual(feasible_actions(state), (ZOOM, SPLIT, EXPAND))
        self.assertEqual(feasible_actions(PolicyState(zoom_available=True)), (FORCED_RETURN,))

    def test_feasible_actions_backtracks_only_after_two_strictly_small_deltas(self):
        common = dict(
            has_unvisited_history_branch=True,
            progress_deltas=(0.02, 0.029),
            progress_threshold=0.03,
        )
        self.assertEqual(feasible_actions(PolicyState(**common)), (BACKTRACK,))
        self.assertEqual(
            feasible_actions(PolicyState(**dict(common, progress_deltas=(0.03, 0.01)))),
            (FORCED_RETURN,),
        )
        self.assertEqual(
            feasible_actions(PolicyState(**dict(common, progress_deltas=(0.02,)))),
            (FORCED_RETURN,),
        )

    def test_feasible_actions_honors_explicit_backtrack_request(self):
        state = PolicyState(has_unvisited_history_branch=True, backtrack_requested=True)
        self.assertEqual(feasible_actions(state), (BACKTRACK,))

    def test_feasible_actions_prioritizes_recoverable_backtrack_over_other_work(self):
        common = dict(
            has_unvisited_next=True,
            has_unvisited_history_branch=True,
            zoom_available=True,
            zoom_enabled=True,
            split_available=True,
            split_enabled=True,
            expand_available=True,
            expand_enabled=True,
        )
        expected = (BACKTRACK, ZOOM, SPLIT, EXPAND, NEXT)
        self.assertEqual(
            feasible_actions(PolicyState(**common, backtrack_requested=True)),
            expected,
        )
        self.assertEqual(
            feasible_actions(PolicyState(**common, progress_deltas=(0.02, 0.02))),
            expected,
        )
        self.assertEqual(
            feasible_actions(PolicyState(**common, backtrack_requested=True,
                                         failed_actions=(BACKTRACK,))),
            (ZOOM, SPLIT, EXPAND, NEXT),
        )
        self.assertEqual(
            feasible_actions(PolicyState(**dict(common, has_unvisited_history_branch=False),
                                         backtrack_requested=True)),
            (ZOOM, SPLIT, EXPAND, NEXT),
        )

    def test_policy_state_rejects_unknown_actions_and_invalid_progress(self):
        for kwargs in (
            {"failed_actions": ("UNKNOWN",)},
            {"failed_actions": (NEXT, NEXT)},
            {"progress_deltas": (-0.01,)},
            {"progress_deltas": (math.nan,)},
            {"progress_threshold": math.inf},
            {"progress_threshold": -0.01},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    PolicyState(**kwargs)

    def test_should_certify_requires_all_four_strict_or_inclusive_gates(self):
        answer = AnswerRecord(uncertainty=0.19)
        thresholds = StopThresholds(gap=0.2, uncertainty=0.2, support_avg=0.7, support_min=0.6)
        gaps = {action: 0.19 for action in CERTIFICATION_GAP_ACTIONS}
        self.assertEqual(CERTIFICATION_GAP_ACTIONS, (ZOOM, SPLIT, EXPAND, NEXT))
        self.assertTrue(should_certify(gaps, answer, 0.7, 0.6, thresholds))
        self.assertFalse(should_certify({**gaps, ZOOM: 0.2}, answer, 0.7, 0.6, thresholds))
        self.assertFalse(should_certify(gaps, AnswerRecord(uncertainty=0.2), 0.7, 0.6, thresholds))
        self.assertFalse(should_certify(gaps, answer, 0.69, 0.6, thresholds))
        self.assertFalse(should_certify(gaps, answer, 0.7, 0.59, thresholds))
        for missing in CERTIFICATION_GAP_ACTIONS:
            incomplete = dict(gaps)
            incomplete.pop(missing)
            with self.subTest(missing=missing):
                with self.assertRaises(ValueError):
                    should_certify(incomplete, answer, 0.7, 0.6, thresholds)
        with self.assertRaises(ValueError):
            should_certify({**gaps, "unknown": 0.1}, answer, 0.7, 0.6, thresholds)

    def test_should_certify_rejects_empty_or_malformed_scores(self):
        thresholds = StopThresholds()
        answer = AnswerRecord(uncertainty=0.1)
        with self.assertRaises(ValueError):
            should_certify({}, answer, 1.0, 1.0, thresholds)
        invalid_cases = (
            ({1: 0.1}, answer, 1.0, 1.0),
            ({ZOOM: True}, answer, 1.0, 1.0),
            ({ZOOM: math.nan}, answer, 1.0, 1.0),
            ({ZOOM: 0.1}, AnswerRecord(uncertainty=math.inf), 1.0, 1.0),
            ({ZOOM: 0.1}, answer, -0.1, 1.0),
        )
        for gaps, invalid_answer, avg, minimum in invalid_cases:
            with self.subTest(gaps=gaps, avg=avg, minimum=minimum):
                with self.assertRaises((TypeError, ValueError)):
                    should_certify(gaps, invalid_answer, avg, minimum, thresholds)
        for invalid in (-0.1, 1.1, math.nan):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    StopThresholds(gap=invalid)

    def test_history_best_uses_stable_forced_return_order(self):
        history = HistoryBuffer()
        records = (
            HistoryRecord(step=4, answer=AnswerRecord(confidence=0.8), support_min=0.7,
                          support_avg=0.9, cost=5),
            HistoryRecord(step=3, answer=AnswerRecord(confidence=0.8), support_min=0.8,
                          support_avg=0.1, cost=99),
            HistoryRecord(step=2, answer=AnswerRecord(confidence=0.8), support_min=0.8,
                          support_avg=0.9, cost=99),
            HistoryRecord(step=1, answer=AnswerRecord(confidence=0.8), support_min=0.8,
                          support_avg=0.9, cost=1),
        )
        for record in records:
            history.add(record)
        self.assertEqual(history.best().step, 1)

    def test_history_tie_uses_earlier_step_and_preserves_first_exact_duplicate(self):
        history = HistoryBuffer()
        history.add(HistoryRecord(step=2, answer=AnswerRecord(confidence=0.5), cost=1))
        history.add(HistoryRecord(step=1, answer=AnswerRecord(confidence=0.5), cost=1))
        self.assertEqual(history.best().step, 1)
        with self.assertRaises(ValueError):
            history.add(HistoryRecord(step=1, answer=AnswerRecord(confidence=0.4), cost=2))

    def test_history_step_is_a_canonical_nonnegative_integer_identity(self):
        history = HistoryBuffer()
        history.add(HistoryRecord(step=np.int64(2)))
        self.assertIs(type(history.best().step), int)
        with self.assertRaises(ValueError):
            history.add(HistoryRecord(step=2))
        for step in (True, -1, 1.5, np.float64(2.0), math.nan, math.inf):
            with self.subTest(step=step):
                with self.assertRaises((TypeError, ValueError)):
                    HistoryBuffer().add(HistoryRecord(step=step))

    def test_history_filters_unvisited_branches_and_returns_none_when_empty(self):
        history = HistoryBuffer()
        self.assertIsNone(history.best())
        self.assertIsNone(history.best_with_unvisited_branch())
        history.add(HistoryRecord(step=1, has_unvisited_branch=False))
        history.add(HistoryRecord(step=2, has_unvisited_branch=True, answer=AnswerRecord(confidence=0.1)))
        self.assertEqual(history.best_with_unvisited_branch().step, 2)

    def test_history_snapshots_inputs_and_results(self):
        record = HistoryRecord(
            step=1,
            answer=AnswerRecord(confidence=0.9),
            state={"visited": ["a"]},
        )
        history = HistoryBuffer()
        history.add(record)
        record.answer.confidence = 0.1
        record.state["visited"].append("b")
        selected = history.best()
        self.assertEqual(selected.answer.confidence, 0.9)
        self.assertEqual(selected.state, {"visited": ["a"]})
        selected.answer.confidence = 0.2
        self.assertEqual(history.best().answer.confidence, 0.9)

    def test_history_rejects_nonrecords_invalid_scores_and_nonjson_snapshot(self):
        history = HistoryBuffer()
        invalid_records = (
            object(),
            HistoryRecord(step=-1),
            HistoryRecord(answer=AnswerRecord(confidence=1.1)),
            HistoryRecord(support_avg=math.nan),
            HistoryRecord(support_min=-0.1),
            HistoryRecord(cost=-1),
            HistoryRecord(state={"bad": object()}),
        )
        for record in invalid_records:
            with self.subTest(record=record):
                with self.assertRaises((TypeError, ValueError)):
                    history.add(record)

    def test_select_root_or_search_uses_strict_tolerance_and_copies_result(self):
        root = AnswerRecord(confidence=0.8, selected_from="unchanged")
        search = AnswerRecord(confidence=0.75, selected_from="unchanged")
        selected = select_root_or_search(root, search, tolerance=0.05)
        self.assertEqual(selected.selected_from, "search")
        self.assertEqual((root.selected_from, search.selected_from), ("unchanged", "unchanged"))
        self.assertIsNot(selected, search)
        self.assertEqual(select_root_or_search(root, search, tolerance=0.04).selected_from, "root")
        self.assertEqual(select_root_or_search(root, AnswerRecord(confidence=0.8), tolerance=0).selected_from, "search")

    def test_select_root_or_search_rejects_invalid_confidence_or_tolerance(self):
        for root, search, tolerance in (
            (object(), AnswerRecord(), 0),
            (AnswerRecord(confidence=math.nan), AnswerRecord(), 0),
            (AnswerRecord(), AnswerRecord(confidence=1.1), 0),
            (AnswerRecord(), AnswerRecord(), -0.01),
            (AnswerRecord(), AnswerRecord(), math.inf),
        ):
            with self.subTest(tolerance=tolerance):
                with self.assertRaises((TypeError, ValueError)):
                    select_root_or_search(root, search, tolerance)

    def test_select_root_or_search_isolates_nested_answer_data_and_bounds_tolerance(self):
        root = AnswerRecord(confidence=0.9, output={"items": ["root"]},
                            groups={"root": {"votes": ["A"]}})
        search = AnswerRecord(confidence=0.7, output={"items": ["search"]},
                              groups={"search": {"votes": ["B"]}})
        selected = select_root_or_search(root, search, 0.1)
        selected.output["items"].append("changed")
        selected.groups["root"]["votes"].append("changed")
        self.assertEqual(root.output, {"items": ["root"]})
        self.assertEqual(root.groups, {"root": {"votes": ["A"]}})
        with self.assertRaises(ValueError):
            select_root_or_search(root, search, 1.0001)


if __name__ == "__main__":
    unittest.main()
