import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from cvsearch.evidence_gap.method import MINIMAL_V1, get_evidence_gap_response, load_method_config
from cvsearch.evidence_gap.types import FORCED_RETURN, ZOOM


def zoom_config(**overrides):
    config = deepcopy(MINIMAL_V1)
    config.update({
        "mode": "root_search_fallback",
        "rerank_enabled": False,
        "enable_zoom": True,
    })
    config.update(overrides)
    return config


class ZoomNode:
    def __init__(self, bbox=(10, 10, 20, 20), *, source="fast", is_root=False):
        self.state = SimpleNamespace(bbox=bbox)
        self.search_source = source
        self.is_root = is_root


class LossModel:
    def __init__(self, rows, *, fail_at=None):
        self.view_size = 336
        self.rows = iter(rows)
        self.fail_at = fail_at
        self.calls = []

    @staticmethod
    def get_patch(bbox, image_width, image_height, patch_size, patch_scale=None):
        object_width = int(bbox[2])
        object_height = int(bbox[3])
        center_x = int(bbox[0] + bbox[2] / 2)
        center_y = int(bbox[1] + bbox[3] / 2)
        patch_width = max(object_width, patch_size)
        patch_height = max(object_height, patch_size)
        left = max(0, center_x - patch_width // 2)
        top = max(0, center_y - patch_height // 2)
        return [left, top, min(left + patch_width, image_width), min(top + patch_height, image_height)]

    def multiple_choices_with_losses(self, image_pil, question, options, searched_nodes=None):
        self.calls.append((self.view_size, tuple(searched_nodes or ())))
        if self.fail_at == len(self.calls):
            raise RuntimeError("synthetic zoom failure")
        losses = next(self.rows)
        return min(range(len(losses)), key=losses.__getitem__), losses


class EvidenceGapZoomTest(unittest.TestCase):
    def setUp(self):
        self.policy = {
            "question": "Which sign is visible?",
            "options": ["red", "blue"],
            "answer_type": "logits_match",
            "input_image": "image.jpg",
        }

    def _run_vstar(self, model, nodes, *, config=None, image_size=(100, 100), raw=0,
                   targets=("sign",), question=None, runtime_targets=None):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", image_size, "white").save(image_path)
            policy = dict(self.policy, input_image=str(image_path))
            if question is not None:
                policy["question"] = question

            def fake_cvsearch(**kwargs):
                if runtime_targets is not None:
                    kwargs["annotation"]["targets"] = list(runtime_targets)
                kwargs["answer_observer"]("search", list(nodes), raw)
                return raw

            return get_evidence_gap_response(
                sam_model=object(), zoom_model=model, nlp_model=object(),
                policy_annotation=policy, original_annotation={"answer": 1},
                ic_examples=[], decomposed_question_template="{}",
                config=zoom_config() if config is None else config,
                cvsearch_fn=fake_cvsearch,
                targets=targets,
            )

    def test_config_allows_zoom_only_for_root_fallback_without_reranking(self):
        loaded = load_method_config(zoom_config())
        self.assertTrue(loaded["enable_zoom"])

        invalid = (
            zoom_config(mode="rerank_only"),
            zoom_config(rerank_enabled=True),
            zoom_config(enable_split=True),
            zoom_config(enable_expand=True),
            zoom_config(enable_certified_stop=True),
        )
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    load_method_config(config)

    def test_eligible_agreeing_render_levels_select_zoom_and_trace_exact_cost(self):
        node = ZoomNode()
        model = LossModel((
            [0.1, 0.9],
            [0.4, 0.6],
            [0.8, 0.2],
            [0.7, 0.3],
        ))

        response, trace = self._run_vstar(model, [node])

        self.assertEqual(response, 1)
        self.assertEqual(trace.final_answer.selected_from, "zoom")
        self.assertEqual(trace.final_boxes, ((10, 10, 20, 20),))
        self.assertEqual([size for size, _ in model.calls], [336, 336, 112, 84])
        self.assertEqual(model.view_size, 336)
        self.assertEqual((trace.budget.mllm_calls, trace.budget.processed_pixels), (12, 120_000))
        self.assertEqual([item.answer.selected_from for item in trace.history], ["root", "search", "zoom"])
        self.assertEqual(trace.history[-1].state, {
            "action": ZOOM,
            "render_levels": [112, 84],
            "render_identities": [[2, 2, 39, 39], [6, 6, 34, 34]],
            "feasible": True,
            "no_op_reason": None,
        })
        zoom_step = trace.steps[-2]
        self.assertEqual(zoom_step.action, ZOOM)
        self.assertEqual(zoom_step.focus_key, "render_levels=112,84")
        self.assertEqual(zoom_step.feasible_actions, (ZOOM,))
        self.assertIsNone(zoom_step.no_op_reason)
        self.assertEqual(zoom_step.answer.selected_from, "zoom")
        self.assertEqual(zoom_step.budget.mllm_calls, 12)
        self.assertEqual(
            [(step.step, step.action) for step in trace.steps],
            [(0, ZOOM), (1, FORCED_RETURN)],
        )
        json.dumps(trace.to_dict(), allow_nan=False)

    def test_disagreeing_render_level_winners_are_a_traced_no_op(self):
        model = LossModel((
            [0.1, 0.9],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.7, 0.3],
        ))

        response, trace = self._run_vstar(model, [ZoomNode()])

        self.assertEqual(response, 0)
        self.assertEqual(trace.final_answer.selected_from, "root")
        self.assertEqual([size for size, _ in model.calls], [336, 336, 112, 84])
        self.assertEqual(model.view_size, 336)
        zoom_step = trace.steps[-2]
        self.assertEqual(zoom_step.feasible_actions, (ZOOM,))
        self.assertEqual(zoom_step.no_op_reason, "render_level_winners_disagree")
        self.assertEqual(zoom_step.answer.selected_from, "root")
        self.assertEqual(trace.history[-1].state["no_op_reason"], "render_level_winners_disagree")

    def test_duplicate_crop_identities_are_rejected_before_zoom_inference(self):
        for size in (62, 98):
            with self.subTest(size=size):
                model = LossModel(([0.1, 0.9], [0.4, 0.6]))

                response, trace = self._run_vstar(
                    model, [ZoomNode((10, 10, size, size))], image_size=(200, 200),
                )

                self.assertEqual(response, 0)
                self.assertEqual([view_size for view_size, _ in model.calls], [336, 336])
                self.assertEqual(model.view_size, 336)
                self.assertEqual(trace.steps[-2].action, ZOOM)
                self.assertEqual(trace.steps[-2].feasible_actions, ())
                self.assertEqual(trace.steps[-2].no_op_reason, "render_levels_not_distinct")
                self.assertEqual(trace.history[-1].state["render_levels"], [112, 84])
                self.assertEqual(len(trace.history[-1].state["render_identities"]), 2)

    def test_nonlocal_query_plans_are_ineligible_before_zoom_inference(self):
        cases = (
            ("no_target_detail", (), "Which sign is visible?"),
            ("relation_context", ("sign",), "Which sign is left of the bus?"),
            ("coverage", ("sign",), "How many signs are visible?"),
            ("multiple_targets", ("sign", "bus"), "Which object is visible?"),
        )
        for name, targets, question in cases:
            with self.subTest(name=name):
                model = LossModel((
                    [0.1, 0.9], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7],
                ))

                response, trace = self._run_vstar(
                    model, [ZoomNode()], targets=targets, question=question,
                )

                self.assertEqual(response, 0)
                self.assertEqual([view_size for view_size, _ in model.calls], [336, 336])
                self.assertEqual(trace.steps[-2].feasible_actions, ())
                self.assertEqual(trace.steps[-2].no_op_reason, "query_plan_is_not_single_target_detail")

    def test_single_runtime_target_detail_remains_eligible_with_audit_context(self):
        model = LossModel((
            [0.1, 0.9], [0.4, 0.6], [0.8, 0.2], [0.7, 0.3],
        ))

        response, trace = self._run_vstar(
            model, [ZoomNode()], targets=(), runtime_targets=("sign",),
        )

        self.assertEqual(response, 1)
        self.assertEqual(trace.final_answer.selected_from, "zoom")
        self.assertEqual(
            [item["kind"] for item in trace.query_plan.evidence_items],
            ["target_detail", "runtime_ranking_context"],
        )

    def test_root_fine_multiple_and_large_search_observations_are_ineligible(self):
        cases = (
            ("root", [ZoomNode(is_root=True)]),
            ("fine", [ZoomNode(source="fine")]),
            ("multiple", [ZoomNode(), ZoomNode((40, 40, 10, 10))]),
            ("large", [ZoomNode((0, 0, 336, 20))]),
        )
        for name, nodes in cases:
            with self.subTest(name=name):
                model = LossModel(([0.1, 0.9], [0.4, 0.6]))
                response, trace = self._run_vstar(model, nodes, image_size=(400, 400))
                self.assertEqual(response, 0)
                self.assertEqual([size for size, _ in model.calls], [336, 336])
                self.assertEqual(trace.steps[-2].action, ZOOM)
                self.assertEqual(trace.steps[-2].feasible_actions, ())
                self.assertIsNotNone(trace.steps[-2].no_op_reason)
                self.assertEqual(trace.history[-1].state["feasible"], False)

    def test_hr_never_executes_zoom(self):
        blocks = ["A. cat\nB. dog", "A. dog\nB. cat"] * 2

        class HrModel:
            def __init__(self):
                self.outputs = iter(("A", "B", "A", "B"))
                self.calls = 0

            def free_form_using_nodes(self, image_pil, question, searched_nodes):
                self.calls += 1
                return next(self.outputs)

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (100, 100), "white").save(image_path)
            policy = {
                "question": "Which animal?", "options": blocks,
                "answer_type": "option_list", "input_image": str(image_path),
            }
            model = HrModel()

            def fake_cvsearch(**kwargs):
                raw = ["A", "B", "A", "B"]
                kwargs["answer_observer"]("search", [ZoomNode()], raw)
                return raw

            response, trace = get_evidence_gap_response(
                sam_model=object(), zoom_model=model, nlp_model=object(),
                policy_annotation=policy, original_annotation={}, ic_examples=[],
                decomposed_question_template="{}", config=zoom_config(), cvsearch_fn=fake_cvsearch,
            )

        self.assertEqual(response, ["A", "B", "A", "B"])
        self.assertEqual(model.calls, 4)
        self.assertEqual(trace.steps[-2].action, ZOOM)
        self.assertEqual(trace.steps[-2].no_op_reason, "answer_type_is_not_logits_match")

    def test_zoom_budget_rejection_is_atomic_for_calls_and_pixels(self):
        cases = (
            ("calls", {"max_mllm_calls": 11, "max_processed_pixels": 1_000_000}),
            ("pixels", {"max_mllm_calls": 12, "max_processed_pixels": 119_999}),
        )
        for name, limits in cases:
            with self.subTest(name=name):
                model = LossModel(([0.1, 0.9], [0.4, 0.6]))
                response, trace = self._run_vstar(
                    model, [ZoomNode()], config=zoom_config(**limits),
                )
                self.assertEqual(response, 0)
                self.assertEqual([size for size, _ in model.calls], [336, 336])
                self.assertEqual((trace.budget.mllm_calls, trace.budget.processed_pixels), (6, 60_000))
                self.assertEqual(trace.steps[-2].feasible_actions, ())
                self.assertEqual(trace.steps[-2].no_op_reason, f"{name}_budget_insufficient")

    def test_underlying_view_size_is_restored_when_second_zoom_render_raises(self):
        model = LossModel((
            [0.1, 0.9],
            [0.4, 0.6],
            [0.2, 0.8],
        ), fail_at=4)

        with self.assertRaisesRegex(RuntimeError, "synthetic zoom failure"):
            self._run_vstar(model, [ZoomNode()])

        self.assertEqual([size for size, _ in model.calls], [336, 336, 112, 84])
        self.assertEqual(model.view_size, 336)

    def test_disabled_zoom_preserves_the_old_path_and_call_count(self):
        model = LossModel(([0.1, 0.9], [0.4, 0.6]))
        config = zoom_config(enable_zoom=False)

        response, trace = self._run_vstar(model, [ZoomNode()], config=config)

        self.assertEqual(response, 0)
        self.assertEqual([size for size, _ in model.calls], [336, 336])
        self.assertEqual([step.action for step in trace.steps], [FORCED_RETURN])
        self.assertEqual([item.answer.selected_from for item in trace.history], ["root", "search"])


if __name__ == "__main__":
    unittest.main()
