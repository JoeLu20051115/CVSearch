import json
import math
import unittest
from dataclasses import replace

from cvsearch.evidence_gap.types import (
    SPLIT,
    AnswerRecord,
    BudgetLedger,
    P0Anchor,
    SplitBranchObservation,
    SplitSearchAudit,
    SplitViewObservation,
    StepTrace,
)


def p0_answer(output="A"):
    return AnswerRecord(
        output=output,
        canonical_answer=output,
        raw_outputs=(output,),
        frequency=1.0,
        margin=1.0,
        confidence=0.30,
        uncertainty=0.70,
        selected_from="stage2",
        aggregation_available=True,
    )


def anchor(output="A"):
    return P0Anchor(
        emitted_answer=output,
        cvsearch_raw=output,
        producing_phase="response",
        node_keys=(),
        support_view=None,
    )


def view(role, digest, *, answer="B", support=None, path=(0,), target=None, crop=None):
    target = (0, 0, 56, 56) if target is None else target
    if crop is None:
        crop = target if role == "tight" else (0, 0, 64, 64)
    return SplitViewObservation(
        role=role,
        patch_path=path,
        target_box_xyxy=target,
        crop_xyxy=crop,
        source_size=(100, 100),
        render_sha256=digest * 64,
        raw_support=(0.65 if role == "tight" else 0.80)
        if support is None else support,
        answer=answer,
        mllm_calls=2,
        processed_pixels=(56 * 56 if role == "tight" else 64 * 64),
    )


def branch(*, visit_index=0, tight=None, context=None, backtracked=False):
    tight = view("tight", "1") if tight is None else tight
    context = view("context", "2") if context is None else context
    return SplitBranchObservation(
        visit_index=visit_index,
        ranked_sibling_paths=((0,), (1,), (2,), (3,)),
        ranked_sibling_boxes=(
            (0, 0, 56, 56),
            (43, 0, 100, 56),
            (0, 43, 56, 100),
            (43, 43, 100, 100),
        ),
        ranked_sibling_scores=(0.90, 0.80, 0.20, 0.10),
        tight_view=tight,
        context_view=context,
        backtracked=backtracked,
    )


def audit(*, branches=None, p0=None, ledger_before=None, ledger_after=None, **changes):
    branches = (branch(),) if branches is None else branches
    p0 = p0_answer() if p0 is None else p0
    ledger_before = BudgetLedger(10, 100_000) if ledger_before is None else ledger_before
    ledger_after = BudgetLedger(
        10, 100_000, mllm_calls=4, processed_pixels=56 * 56 + 64 * 64,
    ) if ledger_after is None else ledger_after
    arguments = {
        "p0_anchor": anchor(),
        "p0_stability": p0,
        "branches": branches,
        "rank_sha256": "3" * 64,
        "query_sha256": "4" * 64,
        "render_policy": "native_2x2_overlap_two_scale_depth2_v1",
        "ledger_before": ledger_before,
        "ledger_after": ledger_after,
        "no_op_reason": None,
    }
    arguments.update(changes)
    return SplitSearchAudit(**arguments)


class SplitViewObservationTest(unittest.TestCase):
    def test_view_serializes_exact_native_geometry_and_answer_snapshot(self):
        observation = view("context", "2")
        self.assertEqual(observation.to_dict(), {
            "role": "context",
            "patch_path": [0],
            "target_box_xyxy": [0, 0, 56, 56],
            "crop_xyxy": [0, 0, 64, 64],
            "source_size": [100, 100],
            "render_sha256": "2" * 64,
            "raw_support": 0.80,
            "answer": "B",
            "mllm_calls": 2,
            "processed_pixels": 64 * 64,
        })
        self.assertEqual(json.loads(json.dumps(observation.to_dict())), observation.to_dict())

    def test_view_rejects_invalid_role_geometry_hash_support_and_cost(self):
        valid = view("tight", "1")
        cases = (
            {"role": "zoomed"},
            {"patch_path": (0, 1, 2)},
            {"crop_xyxy": (1, 1, 55, 55)},
            {"source_size": (50, 50)},
            {"render_sha256": "bad"},
            {"raw_support": math.nan},
            {"answer": {1, 2}},
            {"mllm_calls": 0},
            {"processed_pixels": -1},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(
                (TypeError, ValueError)
            ):
                replace(valid, **changes)


class SplitBranchObservationTest(unittest.TestCase):
    def test_branch_binds_two_distinct_views_to_one_ranked_child(self):
        observation = branch()
        payload = observation.to_dict()
        self.assertEqual(payload["visit_index"], 0)
        self.assertEqual(payload["observed_path"], [0])
        self.assertEqual(len(payload["ranked_siblings"]), 4)
        self.assertEqual(payload["tight_view"]["render_sha256"], "1" * 64)
        self.assertEqual(payload["context_view"]["render_sha256"], "2" * 64)

    def test_branch_rejects_duplicate_views_bad_pool_and_backtrack_order(self):
        valid = branch()
        cases = (
            {"context_view": view("context", "1")},
            {"context_view": view("context", "2", path=(1,))},
            {"ranked_sibling_paths": ((0,), (1,), (2,))},
            {"ranked_sibling_scores": (0.9, 0.8, math.inf, 0.1)},
            {"backtracked": True},
            {"visit_index": 1},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(
                (TypeError, ValueError)
            ):
                replace(valid, **changes)


class SplitSearchAuditTest(unittest.TestCase):
    def test_audit_binds_p0_hashes_branch_budget_and_step_trace(self):
        record = audit()
        payload = record.to_dict()
        self.assertEqual(payload["p0_anchor"]["emitted_answer"], "A")
        self.assertEqual(payload["rank_sha256"], "3" * 64)
        self.assertEqual(payload["query_sha256"], "4" * 64)
        self.assertEqual(payload["ledger_before"]["mllm_calls"], 0)
        self.assertEqual(payload["ledger_after"]["mllm_calls"], 4)

        step = StepTrace(
            action=SPLIT,
            focus_key="p0",
            feasible_actions=(SPLIT,),
            answer=p0_answer(),
            budget=BudgetLedger(
                10, 100_000, mllm_calls=4,
                processed_pixels=56 * 56 + 64 * 64,
            ),
            split_search_audit=record,
        )
        self.assertIn("split_search_audit", step.to_dict())
        self.assertNotIn("split_search_audit", StepTrace().to_dict())

    def test_audit_rejects_identity_branch_and_budget_mismatches(self):
        first = branch()
        second_tight = view(
            "tight", "5", path=(1,), target=(43, 0, 100, 56),
            crop=(43, 0, 100, 56),
        )
        second_context = view(
            "context", "6", path=(1,), target=(43, 0, 100, 56),
            crop=(36, 0, 100, 64),
        )
        second = branch(
            visit_index=1,
            tight=second_tight,
            context=second_context,
            backtracked=True,
        )
        valid_branches = (first, second)
        calls = sum(
            item.tight_view.mllm_calls + item.context_view.mllm_calls
            for item in valid_branches
        )
        pixels = sum(
            item.tight_view.processed_pixels + item.context_view.processed_pixels
            for item in valid_branches
        )
        valid = {
            "branches": valid_branches,
            "ledger_after": BudgetLedger(
                10, 100_000, mllm_calls=calls, processed_pixels=pixels,
            ),
        }
        self.assertEqual(len(audit(**valid).branches), 2)
        cases = (
            {"rank_sha256": "bad"},
            {"query_sha256": "bad"},
            {"p0_stability": p0_answer("B")},
            {"branches": (first, first, first)},
            {"branches": valid_branches[::-1], "ledger_after": valid["ledger_after"]},
            {"ledger_after": BudgetLedger(10, 100_000, mllm_calls=3)},
        )
        for changes in cases:
            arguments = dict(valid)
            arguments.update(changes)
            with self.subTest(changes=changes), self.assertRaises(
                (TypeError, ValueError)
            ):
                audit(**arguments)

    def test_mutated_nested_material_is_detected_before_serialization(self):
        p0 = p0_answer()
        after = BudgetLedger(
            10, 100_000, mllm_calls=4, processed_pixels=56 * 56 + 64 * 64,
        )
        record = audit(p0=p0, ledger_after=after)
        p0.output = "B"
        with self.assertRaisesRegex(ValueError, "mutated"):
            record.to_dict()

        p0 = p0_answer()
        after = BudgetLedger(
            10, 100_000, mllm_calls=4, processed_pixels=56 * 56 + 64 * 64,
        )
        record = audit(p0=p0, ledger_after=after)
        after.mllm_calls += 1
        with self.assertRaisesRegex(ValueError, "mutated"):
            record.to_dict()


if __name__ == "__main__":
    unittest.main()
