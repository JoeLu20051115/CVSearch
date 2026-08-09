import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cvsearch.evidence_gap.types import (
    ACTION_VALUES,
    BACKTRACK,
    CERTIFIED_STOP,
    EXPAND,
    FORCED_RETURN,
    NEXT,
    SPLIT,
    ZOOM,
    AnswerRecord,
    BudgetExceeded,
    BudgetLedger,
    CandidateScore,
    HistoryRecord,
    MethodTrace,
    QueryPlan,
    SearchCandidate,
    StepTrace,
    canonical_key,
)


class EvidenceGapTypesTest(unittest.TestCase):
    def test_canonical_key_is_stable_for_numeric_bbox_types(self):
        self.assertEqual(
            canonical_key([1, 2, 30, 40], 2, 1),
            canonical_key([1.0, 2.0, 30.0, 40.0], 2, 1),
        )

    def test_canonical_key_rejects_noncanonical_bbox_values(self):
        invalid_boxes = (
            [1, 2, 3],
            [1, 2, 3, 4, 5],
            [1, 2, math.inf, 4],
            [1, 2, math.nan, 4],
            [1, 2, "3", 4],
            [1, 2, True, 4],
        )
        for bbox in invalid_boxes:
            with self.subTest(bbox=bbox):
                with self.assertRaises((TypeError, ValueError)):
                    canonical_key(bbox, 2, 1)

    def test_budget_rejects_before_mutating(self):
        budget = BudgetLedger(max_mllm_calls=2, max_processed_pixels=100)
        budget.consume("mllm_calls", 2)
        with self.assertRaises(BudgetExceeded):
            budget.consume("mllm_calls", 1)
        self.assertEqual(budget.mllm_calls, 2)

    def test_budget_validates_kind_amount_and_count_semantics_before_mutating(self):
        budget = BudgetLedger(max_mllm_calls=2, max_processed_pixels=100)
        for kind, amount in (
            ("unknown", 1),
            ("mllm_calls", -1),
            ("mllm_calls", 0.5),
            ("mllm_calls", math.inf),
            ("processed_pixels", math.nan),
        ):
            with self.subTest(kind=kind, amount=amount):
                with self.assertRaises((TypeError, ValueError)):
                    budget.consume(kind, amount)
                self.assertEqual((budget.mllm_calls, budget.processed_pixels), (0, 0))

        budget.consume("mllm_calls", 1.0)
        budget.consume("processed_pixels", 12.5)
        self.assertEqual((budget.mllm_calls, budget.processed_pixels), (1, 12.5))
        self.assertFalse(hasattr(budget, "unknown"))

    def test_action_values_are_centralized_and_complete(self):
        self.assertEqual(
            ACTION_VALUES,
            (ZOOM, SPLIT, EXPAND, NEXT, BACKTRACK, CERTIFIED_STOP, FORCED_RETURN),
        )
        self.assertEqual(set(ACTION_VALUES), {
            "ZOOM", "SPLIT", "EXPAND", "NEXT", "BACKTRACK", "CERTIFIED_STOP", "FORCED_RETURN"
        })

    def test_search_candidate_serializes_only_contract_fields(self):
        score = CandidateScore(main=0.1, augmented=0.2, complexity=0.3, edge_density=0.4,
                               relevance=0.5, visual=0.6, rank=0.7)
        candidate = SearchCandidate(
            key="1:2:3:4:d0:r0",
            bbox=(1, 2, 3, 4),
            parent_key="root",
            child_keys=("child",),
            depth=0,
            source="expert",
            render_level=0,
            score=score,
            node=object(),
        )

        self.assertEqual(candidate.to_dict(), {
            "key": "1:2:3:4:d0:r0",
            "bbox": [1, 2, 3, 4],
            "parent_key": "root",
            "child_keys": ["child"],
            "depth": 0,
            "source": "expert",
            "render_level": 0,
            "score": score.to_dict(),
        })

    def test_every_state_record_round_trips_as_json_and_json_tool_accepts_fixture(self):
        plan = QueryPlan(
            main_query="Which sign is blue?",
            targets=("sign",),
            augmented_queries=("blue sign",),
            evidence_items=({"target": "sign", "coverage": ("left", "right")},),
            global_scope_required=False,
        )
        answer = AnswerRecord(
            output=["B"], canonical_answer="blue", raw_outputs=("B.",),
            groups={"blue": ("B",)}, frequency=1.0, margin=0.8,
            confidence=0.9, uncertainty=0.1, losses=(0.1, 0.9), selected_from="search",
        )
        trace = MethodTrace(
            query_plan=plan,
            candidate_ranks=[{"key": "candidate", "score": CandidateScore(rank=0.9)}],
            steps=[StepTrace(step=1, action=NEXT, focus_key="candidate", feasible_actions=(NEXT,),
                             gaps={ZOOM: 0.4}, answer=answer, support_avg=0.8, support_min=0.7,
                             budget=BudgetLedger(3, 100, mllm_calls=1, processed_pixels=20))],
            history=[HistoryRecord(step=1, answer=answer, support_avg=0.8, support_min=0.7,
                                   cost=20, has_unvisited_branch=True, state={"seen": ("candidate",)})],
            final_answer=answer,
            budget=BudgetLedger(3, 100, mllm_calls=1, processed_pixels=20),
            termination=CERTIFIED_STOP,
            final_boxes=((1, 2, 3, 4),),
        )

        payload = trace.to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "method-trace.json"
            fixture.write_text(json.dumps(payload), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "json.tool", str(fixture)],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
