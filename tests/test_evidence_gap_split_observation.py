import json
import math
import unittest
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image

from cvsearch.evidence_gap.method import (
    _BudgetedZoomModel,
    _observe_split_search,
    load_method_config,
)
from cvsearch.evidence_gap.types import (
    SPLIT,
    AnswerRecord,
    BudgetLedger,
    P0Anchor,
    QueryPlan,
    SplitBranchObservation,
    SplitProbeObservation,
    SplitSearchAudit,
    SplitViewObservation,
    StepTrace,
)
from tests.test_evidence_gap_support import evidence_items, support_adapter


@dataclass(frozen=True)
class FrozenSupportDescriptor:
    canonical_key: str
    bbox_original: tuple[int, int, int, int]

    def to_dict(self):
        return {
            "canonical_key": self.canonical_key,
            "bbox_original": list(self.bbox_original),
        }


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
        selected_sibling_rank=visit_index,
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


def probe(path=(0, 0), digest="7", support=0.7, score=0.8):
    return SplitProbeObservation(
        patch_path=path,
        target_box_xyxy=(0, 0, 56, 56),
        source_size=(100, 100),
        render_sha256=digest * 64,
        raw_support=support,
        ranking_score=score,
        mllm_calls=1,
        processed_pixels=100 * 100,
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
        "root_ranked_paths": ((0,), (1,), (2,), (3,)) if branches else (),
        "root_ranked_boxes": (
            (0, 0, 56, 56),
            (43, 0, 100, 56),
            (0, 43, 56, 100),
            (43, 43, 100, 100),
        ) if branches else (),
        "root_ranked_scores": (0.90, 0.80, 0.20, 0.10) if branches else (),
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


class SplitProbeObservationTest(unittest.TestCase):
    def test_probe_serializes_answer_free_screening_evidence(self):
        observation = probe()
        self.assertEqual(observation.to_dict(), {
            "patch_path": [0, 0],
            "target_box_xyxy": [0, 0, 56, 56],
            "source_size": [100, 100],
            "render_sha256": "7" * 64,
            "raw_support": 0.7,
            "ranking_score": 0.8,
            "mllm_calls": 1,
            "processed_pixels": 100 * 100,
        })


class SplitBranchObservationTest(unittest.TestCase):
    def test_branch_binds_two_distinct_views_to_one_ranked_child(self):
        observation = branch()
        payload = observation.to_dict()
        self.assertEqual(payload["visit_index"], 0)
        self.assertEqual(payload["selected_sibling_rank"], 0)
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
            {"selected_sibling_rank": 1},
            {"selected_sibling_rank": 4},
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


class SplitObservationConfigTest(unittest.TestCase):
    CONFIG_KEYS = (
        "p5a_split_enabled",
        "p5a_split_replacement_enabled",
        "p5a_split_render_policy",
        "p5a_split_max_observed_branches",
    )

    @staticmethod
    def config(**changes):
        path = (
            Path(__file__).parents[1] / "reproduction" / "evidence_gap" / "configs"
            / "dev_adaptive_ranking_observe_v7.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        value.update({
            "config_id": "adaptive-ranking-observe-split-v1",
            "p5a_split_enabled": True,
            "p5a_split_replacement_enabled": False,
            "p5a_split_render_policy": "native_2x2_overlap_support_screen_two_scale_depth2_v2",
            "p5a_split_max_observed_branches": 4,
        })
        value.update(changes)
        return value

    def test_split_group_is_all_or_none_and_requires_frozen_stage2_profile(self):
        loaded = load_method_config(self.config())
        self.assertEqual(
            {key: loaded[key] for key in self.CONFIG_KEYS},
            {key: self.config()[key] for key in self.CONFIG_KEYS},
        )
        for missing in self.CONFIG_KEYS:
            with self.subTest(missing=missing), self.assertRaisesRegex(
                ValueError, "all-or-none"
            ):
                value = self.config()
                value.pop(missing)
                load_method_config(value)
        invalid = (
            {"p5a_split_replacement_enabled": True},
            {"p5a_split_render_policy": "super_resolution"},
            {"p5a_split_max_observed_branches": 5},
            {"enable_split": True},
            {"alpha": 0.66},
            {"p2c_zoom_enabled": False},
            {"p4a_expand_enabled": False},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(
                (TypeError, ValueError)
            ):
                load_method_config(self.config(**changes))

    def test_checked_in_split_config_is_exact(self):
        path = (
            Path(__file__).parents[1] / "reproduction" / "evidence_gap" / "configs"
            / "dev_adaptive_ranking_observe_split_v1.json"
        )
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), self.config())
        self.assertEqual(load_method_config(path), self.config())


class FakeSplitScorer:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def score(self, images, texts):
        if self.fail:
            raise RuntimeError("ranking failed")
        self.calls.append((tuple(image.size for image in images), tuple(texts)))
        return [
            [
                (image.getpixel((image.width // 2, image.height // 2))[0] + index)
                / 256.0
                for index, _ in enumerate(texts)
            ]
            for image in images
        ]


def gradient(size=(100, 100)):
    image = Image.new("RGB", size)
    for y in range(image.height):
        for x in range(image.width):
            image.putpixel((x, y), ((x * 5 + y) % 256, (y * 7) % 256, x % 256))
    return image


class SplitCandidateRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.source = gradient()
        self.raw = support_adapter(yes_logit=1.0, no_logit=0.0)
        self.raw.answer_inputs = []

        def answer(image, question, options, nodes):
            self.raw.answer_inputs.append((image.size, question, tuple(options), tuple(nodes)))
            return 1, [2.0, 1.0]

        self.raw.multiple_choices_with_losses = answer
        self.ledger = BudgetLedger(512, 10_000_000_000)
        self.model = _BudgetedZoomModel(self.raw, self.ledger)
        self.plan = QueryPlan(
            main_query="What color is the small sign?",
            augmented_queries=("small sign",),
            targets=("small sign",),
            evidence_items=evidence_items(),
        )
        self.p0 = p0_answer(0)
        self.anchor = P0Anchor(
            emitted_answer=0,
            cvsearch_raw=0,
            producing_phase="response",
            node_keys=(),
            support_view=None,
        )

    def observe(self, scorer=None):
        return _observe_split_search(
            budgeted_model=self.model,
            source_image=self.source,
            scorer=FakeSplitScorer() if scorer is None else scorer,
            query_plan=self.plan,
            answer_type="logits_match",
            options=("red", "blue"),
            p0_anchor=self.anchor,
            p0_stability=deepcopy(self.p0),
            stage1_rank_sha256="7" * 64,
            render_policy="native_2x2_overlap_support_screen_two_scale_depth2_v2",
            max_observed_branches=4,
        )

    def test_observes_two_depth_two_branches_at_tight_and_context_scales(self):
        record = self.observe()
        self.assertEqual(len(record.screening_probes), 8)
        self.assertEqual(len(record.branches), 4)
        self.assertTrue(all(len(item.observed_path) == 2 for item in record.branches))
        self.assertFalse(record.branches[0].backtracked)
        self.assertTrue(all(item.backtracked for item in record.branches[1:]))
        self.assertEqual(len(self.raw.answer_inputs), 8)
        self.assertEqual(self.raw.model.calls, 12)
        self.assertEqual(record.ledger_after.mllm_calls, 36)
        self.assertEqual(record.ledger_after.processed_pixels, 36 * 100 * 100)
        self.assertTrue(all(
            item.tight_view.answer["winner"] == 1
            and item.context_view.answer["winner"] == 1
            for item in record.branches
        ))
        self.assertEqual(record.p0_stability.output, 0)

    def test_split_recovers_from_a_local_p0_by_starting_at_the_full_source(self):
        descriptor = FrozenSupportDescriptor("local-p0", (10, 10, 20, 20))
        self.anchor = P0Anchor(
            emitted_answer=0,
            cvsearch_raw=0,
            producing_phase="search",
            node_keys=(descriptor.canonical_key,),
            support_view=(descriptor,),
        )
        record = self.observe()
        self.assertEqual(min(box[0] for box in record.root_ranked_boxes), 0)
        self.assertEqual(min(box[1] for box in record.root_ranked_boxes), 0)
        self.assertEqual(max(box[2] for box in record.root_ranked_boxes), 100)
        self.assertEqual(max(box[3] for box in record.root_ranked_boxes), 100)

    def test_ranking_failure_is_a_zero_cost_fail_closed_audit(self):
        record = self.observe(FakeSplitScorer(fail=True))
        self.assertEqual(record.branches, ())
        self.assertEqual(record.no_op_reason, "split_ranking_failed")
        self.assertEqual(record.ledger_before.to_dict(), record.ledger_after.to_dict())
        self.assertEqual(self.raw.answer_inputs, [])
        self.assertEqual(self.raw.model.calls, 0)


if __name__ == "__main__":
    unittest.main()
