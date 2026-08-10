from dataclasses import FrozenInstanceError
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

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

    def test_canonical_key_normalizes_integral_depth_and_render_level(self):
        self.assertEqual(
            canonical_key([1, 2, 3, 4], np.int64(2), np.float32(1.0)),
            "1:2:3:4:d2:r1",
        )
        for value in (math.nan, math.inf, True, 1.5, "2", []):
            for argument in ("depth", "render_level"):
                with self.subTest(value=value, argument=argument):
                    args = [2, 1]
                    args[0 if argument == "depth" else 1] = value
                    with self.assertRaises((TypeError, ValueError)):
                        canonical_key([1, 2, 3, 4], *args)

    def test_canonical_key_rejects_negative_levels(self):
        for depth, render_level in ((-1, 1), (1, np.int64(-1))):
            with self.subTest(depth=depth, render_level=render_level):
                with self.assertRaises(ValueError):
                    canonical_key([1, 2, 3, 4], depth, render_level)

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

    def test_candidate_score_is_immutable_after_ranking(self):
        score = CandidateScore(rank=0.7)
        with self.assertRaises(FrozenInstanceError):
            score.rank = 0.8

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

    def test_trace_normalizes_numpy_scalars_for_strict_json(self):
        candidate = SearchCandidate(
            key="1:2:3:4:d0:r0",
            bbox=(np.int64(1), np.int64(2), np.int64(3), np.int64(4)),
            score=CandidateScore(main=np.float32(0.5)),
        )
        trace = MethodTrace(
            candidate_ranks=[candidate.to_dict()],
            budget=BudgetLedger(3, np.float32(100), processed_pixels=np.float32(20)),
        )

        payload = trace.to_dict()
        self.assertEqual(payload["candidate_ranks"][0]["bbox"], [1, 2, 3, 4])
        self.assertIsInstance(payload["candidate_ranks"][0]["score"]["main"], float)
        self.assertIsInstance(payload["budget"]["processed_pixels"], float)
        encoded = json.dumps(payload, allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "numpy-trace.json"
            fixture.write_text(encoded, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "json.tool", str(fixture)],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_public_dataclass_numeric_paths_reject_nonfinite_values(self):
        invalid_records = (
            QueryPlan(evidence_items=(math.nan,)),
            CandidateScore(main=math.nan),
            SearchCandidate(bbox=(math.nan, 0, 0, 0)),
            AnswerRecord(frequency=math.nan),
            HistoryRecord(step=math.nan),
            StepTrace(step=math.nan),
            MethodTrace(final_boxes=((math.nan, 0, 0, 0),)),
        )
        for record in invalid_records:
            with self.subTest(record=type(record).__name__):
                with self.assertRaises(ValueError):
                    record.to_dict()
        with self.assertRaises(ValueError):
            BudgetLedger(max_mllm_calls=1, max_processed_pixels=math.nan)
        self.assertIs(
            type(QueryPlan(evidence_items=(np.bool_(True),)).to_dict()["evidence_items"][0]),
            bool,
        )

    def test_trace_records_gap_fallback_and_elapsed_time(self):
        trace = MethodTrace(
            steps=[StepTrace(
                step=1,
                action=NEXT,
                gap_fallback_used=True,
                elapsed_seconds=np.float32(0.25),
            )],
            elapsed_seconds=np.float64(1.5),
        )

        payload = trace.to_dict()
        self.assertEqual(payload["steps"][0]["gap_fallback_used"], True)
        self.assertEqual(payload["steps"][0]["elapsed_seconds"], 0.25)
        self.assertEqual(payload["elapsed_seconds"], 1.5)
        self.assertEqual(json.loads(json.dumps(payload, allow_nan=False)), payload)

    def test_trace_serializes_explicit_runtime_config_and_accounting_metadata(self):
        trace = MethodTrace(
            method_mode="rerank_only",
            config_id="frozen_v1",
            effective_config={"alpha": np.float32(0.65)},
            cvsearch_search_mode=2,
            root_ans_conf=np.float32(0.25),
            num_pop=[6, [1, 2]],
            num_zoom_in=[1],
            num_zoom_out=[0],
            budget_interrupted=np.bool_(True),
            effective_ranking_query="main_query_plus_current_visual_cue",
            pixel_accounting="source_image_area_per_logical_forward_approximation",
        )
        payload = trace.to_dict()
        self.assertEqual(payload["method_mode"], "rerank_only")
        self.assertEqual(payload["config_id"], "frozen_v1")
        self.assertAlmostEqual(payload["effective_config"]["alpha"], 0.65)
        self.assertEqual(payload["cvsearch_search_mode"], 2)
        self.assertEqual(payload["num_pop"], [6, [1, 2]])
        self.assertIs(payload["budget_interrupted"], True)
        self.assertEqual(json.loads(json.dumps(payload, allow_nan=False)), payload)

    def test_answer_record_serializes_typed_aggregation_availability(self):
        available = AnswerRecord(aggregation_available=True, aggregation_reason=None)
        unavailable = AnswerRecord(
            aggregation_available=False,
            aggregation_reason="ambiguous_winner_projection",
        )
        plain = AnswerRecord()
        self.assertEqual(
            (available.to_dict()["aggregation_available"], available.to_dict()["aggregation_reason"]),
            (True, None),
        )
        self.assertEqual(
            (unavailable.to_dict()["aggregation_available"], unavailable.to_dict()["aggregation_reason"]),
            (False, "ambiguous_winner_projection"),
        )
        self.assertEqual(
            (plain.to_dict()["aggregation_available"], plain.to_dict()["aggregation_reason"]),
            (None, None),
        )
        json.dumps(unavailable.to_dict(), allow_nan=False)

    def test_trace_runtime_metadata_rejects_nonfinite_values(self):
        for trace in (
            MethodTrace(root_ans_conf=math.nan),
            MethodTrace(effective_config={"alpha": math.inf}),
            MethodTrace(num_pop=[math.nan]),
        ):
            with self.subTest(trace=trace):
                with self.assertRaises(ValueError):
                    trace.to_dict()

    def test_trace_elapsed_time_rejects_nonfinite_values(self):
        for record in (
            StepTrace(elapsed_seconds=math.nan),
            MethodTrace(elapsed_seconds=math.inf),
        ):
            with self.subTest(record=type(record).__name__):
                with self.assertRaises(ValueError):
                    record.to_dict()

    def test_public_boolean_fields_normalize_numpy_booleans_for_strict_json(self):
        trace = MethodTrace(
            query_plan=QueryPlan(
                global_scope_required=np.bool_(True),
                fallback_used=np.bool_(False),
            ),
            history=[HistoryRecord(has_unvisited_branch=np.bool_(True))],
            steps=[StepTrace(
                gap_fallback_used=np.bool_(False),
                certified=np.bool_(True),
            )],
        )

        payload = trace.to_dict()
        values = (
            payload["query_plan"]["global_scope_required"],
            payload["query_plan"]["fallback_used"],
            payload["history"][0]["has_unvisited_branch"],
            payload["steps"][0]["gap_fallback_used"],
            payload["steps"][0]["certified"],
        )
        self.assertTrue(all(type(value) is bool for value in values))
        self.assertEqual(json.loads(json.dumps(payload, allow_nan=False)), payload)

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
