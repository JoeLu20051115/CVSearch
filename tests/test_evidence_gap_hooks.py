import ast
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cvsearch"))

from cvsearch import CVSearch


class FakeState:
    def __init__(self, bbox):
        self.bbox = bbox


class FakeNode:
    def __init__(self, identifier, depth, prior_prob, parent=None):
        self.id = identifier
        self.depth = depth
        self.prior_prob = prior_prob
        self.parent = parent
        self.children = []
        self.state = FakeState((0, 0, 2, 2))
        if parent is not None:
            parent.children.append(self)


class FakeTree:
    def __init__(self, root, max_depth):
        self.root = root
        self.max_depth = max_depth


class FakeZoom:
    def __init__(self, existence=None, answering=None, root_answering=1.0):
        self.existence = existence or {}
        self.answering = answering or {}
        self.root_answering = root_answering
        self.calls = []
        self.answer_calls = 0
        self.searched_node_lists = []

    def get_confidence_value(self, nodes, image_pil, confidence_type, input_ele):
        node = nodes[0]
        self.calls.append((confidence_type, getattr(node, "id", None), input_ele))
        if confidence_type == "existence":
            return self.existence[node.id]
        return self.answering.get(getattr(node, "id", None), self.root_answering)

    def generate_visual_cues_using_ic(self, ic_examples, question):
        return ["object"]

    def multiple_choices_inference(self, image_pil, question, options, searched_nodes):
        self.answer_calls += 1
        self.searched_node_lists.append(searched_nodes)
        return 2

    def free_form_using_nodes(self, image_pil, question, searched_nodes):
        self.answer_calls += 1
        self.searched_node_lists.append(searched_nodes)
        return f"raw-{self.answer_calls}"


class FakeArray:
    def __init__(self, value):
        self.value = np.asarray(value)

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeSam:
    def batch_inference(self, image_pil, text_target):
        result = {0: {"boxes": FakeArray([[1, 1, 4, 4]]), "scores": FakeArray([1.0])}}
        return {}, result, [0]


class FailingFakeSam:
    def batch_inference(self, image_pil, text_target):
        result = {
            index: {"boxes": FakeArray(np.empty((0, 4))), "scores": FakeArray([])}
            for index in range(len(text_target))
        }
        features = np.zeros((1, 1, 2, 2), dtype=np.float32)
        return {"vision_features": features}, result, list(result)


class FakeBuilder:
    def __init__(self, *args, **kwargs):
        pass

    def build_tree(self, **kwargs):
        return {}


class StrictTrace:
    __slots__ = ("query_plan", "candidate_ranks")

    def __init__(self, augmented_queries=None):
        self.query_plan = None if augmented_queries is None else SimpleNamespace(
            augmented_queries=tuple(augmented_queries)
        )
        self.candidate_ranks = []


def make_depth_two_tree():
    root = FakeNode("root", 0, 1.0)
    parent = FakeNode("parent", 1, 0.9, root)
    low = FakeNode("low", 2, 0.8, parent)
    middle = FakeNode("middle", 2, 0.8, parent)
    high = FakeNode("high", 2, 0.8, parent)
    invalid = FakeNode("invalid", 2, 0.1, parent)
    return FakeTree(root, 2), [low, middle, high, invalid]


def run_semantic(tree, zoom, **kwargs):
    return CVSearch.semantic_guide_search_dynamic_depth(
        zoom_model=zoom,
        pop_limit=10,
        num_intervel=2,
        threshold_descrease=[0.1],
        depth_limit=tree.max_depth,
        question="Which sign is visible?",
        visual_cue="sign",
        answering_confidence_threshold_lower=0.0,
        answering_confidence_threshold_upper=0.9,
        image_pil=Image.new("RGB", (8, 8), "white"),
        image_tree=tree,
        enable_parent_verification=False,
        **kwargs,
    )


class HookSignatureTest(unittest.TestCase):
    def test_hooks_are_trailing_defaults_and_old_calls_still_bind(self):
        get_parameters = inspect.signature(CVSearch.get_cvsearch_response).parameters
        self.assertEqual(
            list(get_parameters)[-3:],
            ["node_ranker", "answer_observer", "method_trace"],
        )
        self.assertTrue(all(get_parameters[name].default is None for name in list(get_parameters)[-3:]))
        inspect.signature(CVSearch.get_cvsearch_response).bind(
            None, None, None, {}, [], "{}", 0.9, 0.0, 0.6, 10, [0.1]
        )

        semantic_parameters = inspect.signature(CVSearch.semantic_guide_search_dynamic_depth).parameters
        self.assertEqual(list(semantic_parameters)[-2:], ["node_ranker", "rank_context"])
        self.assertTrue(all(semantic_parameters[name].default is None for name in list(semantic_parameters)[-2:]))
        inspect.signature(CVSearch.semantic_guide_search_dynamic_depth).bind(
            None, 10, 2, [0.1], 2, "question", "cue", 0.0, 0.9
        )

    def test_all_four_get_response_search_calls_propagate_rank_hooks(self):
        tree = ast.parse(Path(CVSearch.__file__).read_text())
        get_response = next(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_cvsearch_response"
        )
        calls = [
            node for node in ast.walk(get_response)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "semantic_guide_search_dynamic_depth"
        ]
        self.assertEqual(len(calls), 4)
        scopes = []
        origins = []
        for call in calls:
            with self.subTest(line=call.lineno):
                keywords = {keyword.arg: keyword.value for keyword in call.keywords}
                self.assertIn("node_ranker", keywords)
                self.assertIn("rank_context", keywords)
                context_call = next(
                    node for node in ast.walk(keywords["rank_context"])
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_make_rank_context"
                )
                self.assertEqual(ast.unparse(context_call.args[1]), "question")
                scopes.append(ast.literal_eval(context_call.args[3]))
                origins.append(ast.unparse(context_call.args[4]))
        self.assertCountEqual(scopes, ["main", "main", "cropped", "cropped"])
        self.assertCountEqual(origins, ["(0, 0)", "(0, 0)", "(left, top)", "(left, top)"])


class SemanticRankHookTest(unittest.TestCase):
    def test_no_hook_keeps_posterior_order_and_reverse_hook_changes_first_visit(self):
        tree, _ = make_depth_two_tree()
        existence = {"low": -0.8, "middle": 0.0, "high": 0.8}
        baseline_zoom = FakeZoom(existence=existence, answering={"high": 1.0})
        baseline_result = run_semantic(tree, baseline_zoom)
        self.assertEqual([node.id for node in baseline_result[0]], ["high"])
        self.assertEqual(
            [node_id for kind, node_id, _ in baseline_zoom.calls if kind == "answering"],
            ["high"],
        )

        tree, _ = make_depth_two_tree()
        hook_zoom = FakeZoom(existence=existence, answering={"low": 1.0})
        rank_calls = []

        def reverse_ranker(nodes, image_pil, main_query, augmented_queries):
            rank_calls.append((list(nodes), image_pil, main_query, list(augmented_queries)))
            ranked = list(reversed(nodes))
            return ranked, [{"node_id": node.id} for node in ranked]

        trace = StrictTrace(["planned sign", "street marker"])
        hook_result = run_semantic(
            tree,
            hook_zoom,
            node_ranker=reverse_ranker,
            rank_context=CVSearch._make_rank_context(trace, "Which sign is visible?", "sign", "main", (0, 0)),
        )
        self.assertEqual([node.id for node in hook_result[0]], ["low"])
        self.assertEqual([node.id for node in rank_calls[0][0]], ["high", "middle", "low"])
        self.assertEqual(rank_calls[0][2:], ("Which sign is visible?", ["planned sign", "street marker"]))
        self.assertEqual([detail["node_id"] for detail in trace.candidate_ranks], ["low", "middle", "high"])
        self.assertTrue(all(detail["target"] == "sign" for detail in trace.candidate_ranks))
        self.assertTrue(all(detail["stage"] == "Depth 2" for detail in trace.candidate_ranks))
        self.assertTrue(all(detail["tree_scope"] == "main" for detail in trace.candidate_ranks))
        self.assertTrue(all(detail["crop_origin"] == [0, 0] for detail in trace.candidate_ranks))
        self.assertTrue(all(detail["bbox_original"] == [0, 0, 2, 2] for detail in trace.candidate_ranks))
        self.assertNotIn("invalid", [node.id for node in rank_calls[0][0]])

    def test_ranker_runs_once_for_each_reached_depth_and_records_all_details(self):
        tree, _ = make_depth_two_tree()
        zoom = FakeZoom(
            existence={"low": -0.8, "middle": 0.0, "high": 0.8, "parent": 0.5},
            answering={"low": -0.5, "middle": -0.5, "high": -0.5, "parent": -0.5},
        )
        calls = []
        trace = StrictTrace()

        def identity_ranker(nodes, image_pil, main_query, augmented_queries):
            calls.append([node.id for node in nodes])
            return list(nodes), [{"node_id": node.id} for node in nodes]

        result, _, success = run_semantic(
            tree,
            zoom,
            node_ranker=identity_ranker,
            rank_context=CVSearch._make_rank_context(trace, "Which sign is visible?", "sign", "main", (0, 0)),
        )
        self.assertFalse(success)
        self.assertEqual(calls, [["high", "middle", "low"], ["parent"]])
        self.assertEqual([detail["node_id"] for detail in trace.candidate_ranks], ["high", "middle", "low", "parent"])
        self.assertEqual([node.id for node in result], ["parent"])

    def test_depth_one_hook_only_sees_valid_candidates_and_preserves_failure_fallback(self):
        root = FakeNode("root", 0, 1.0)
        low = FakeNode("low", 1, 0.8, root)
        high = FakeNode("high", 1, 0.8, root)
        invalid = FakeNode("invalid", 1, 0.1, root)
        tree = FakeTree(root, 1)
        zoom = FakeZoom(
            existence={"low": -0.8, "high": 0.8},
            answering={"low": -0.5, "high": -0.5},
        )
        seen = []

        def reverse_ranker(nodes, image_pil, main_query, augmented_queries):
            seen.append(list(nodes))
            ranked = list(reversed(nodes))
            return ranked, [{} for _ in ranked]

        result, _, success = run_semantic(tree, zoom, node_ranker=reverse_ranker, rank_context=None)
        self.assertFalse(success)
        self.assertEqual([node.id for node in seen[0]], ["high", "low"])
        self.assertEqual(
            [node_id for kind, node_id, _ in zoom.calls if kind == "answering"],
            ["low"],
        )
        self.assertEqual([node.id for node in result], ["high", "low", "invalid"])

    def test_all_invalid_depths_skip_ranker_and_keep_baseline_depth_one_fallback(self):
        root = FakeNode("root", 0, 1.0)
        depth_one = FakeNode("depth-one-invalid", 1, 0.1, root)
        FakeNode("depth-two-invalid", 2, 0.1, depth_one)
        tree = FakeTree(root, 2)
        zoom = FakeZoom()
        rank_calls = []

        def reject_empty_ranker(nodes, *args):
            rank_calls.append(list(nodes))
            raise AssertionError("empty candidate stages must bypass node_ranker")

        result, total_pop, success = run_semantic(tree, zoom, node_ranker=reject_empty_ranker)

        self.assertFalse(success)
        self.assertEqual(total_pop, 0)
        self.assertEqual([node.id for node in result], ["depth-one-invalid"])
        self.assertEqual(rank_calls, [])
        self.assertEqual(zoom.calls, [])

    def test_invalid_or_duplicate_rank_results_fail_hard(self):
        rankers = {
            "missing": lambda nodes, *_: (list(nodes[:-1]), [{} for _ in nodes[:-1]]),
            "duplicate": lambda nodes, *_: ([nodes[0]] * len(nodes), [{} for _ in nodes]),
            "details": lambda nodes, *_: (list(nodes), []),
            "detail-type": lambda nodes, *_: (list(nodes), ["not-a-mapping"] * len(nodes)),
            "detail-mismatch": lambda nodes, *_: (
                list(nodes),
                [{"node_id": nodes[0].id}] * len(nodes),
            ),
        }
        for name, ranker in rankers.items():
            with self.subTest(name=name):
                tree, _ = make_depth_two_tree()
                zoom = FakeZoom(existence={"low": -0.8, "middle": 0.0, "high": 0.8})
                with self.assertRaisesRegex(ValueError, "node_ranker"):
                    run_semantic(tree, zoom, node_ranker=ranker)

    def test_rank_details_are_enriched_for_main_and_cropped_trees_without_mutation(self):
        trace = StrictTrace()
        raw_details = []

        def ranker(nodes, image_pil, main_query, augmented_queries):
            detail = {}
            raw_details.append(detail)
            return list(nodes), [detail]

        for scope, origin in (("main", (0, 0)), ("cropped", (10, 20))):
            root = FakeNode("root", 0, 1.0)
            FakeNode("same-local-id", 1, 0.8, root)
            tree = FakeTree(root, 1)
            zoom = FakeZoom(existence={"same-local-id": 0.8}, answering={"same-local-id": 1.0})
            result, _, success = run_semantic(
                tree,
                zoom,
                node_ranker=ranker,
                rank_context=CVSearch._make_rank_context(
                    trace,
                    "Which sign is visible?",
                    "sign",
                    scope,
                    origin,
                ),
            )
            self.assertTrue(success)
            self.assertEqual(result[0].id, "same-local-id")

        self.assertEqual(raw_details, [{}, {}])
        self.assertEqual(
            trace.candidate_ranks,
            [
                {
                    "node_id": "same-local-id",
                    "target": "sign",
                    "stage": "Depth 1",
                    "tree_scope": "main",
                    "crop_origin": [0, 0],
                    "bbox_original": [0, 0, 2, 2],
                },
                {
                    "node_id": "same-local-id",
                    "target": "sign",
                    "stage": "Depth 1",
                    "tree_scope": "cropped",
                    "crop_origin": [10, 20],
                    "bbox_original": [10, 20, 2, 2],
                },
            ],
        )
        json.dumps(trace.candidate_ranks, allow_nan=False)


class AnswerObserverTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.image_path = Path(self.tempdir.name) / "image.png"
        Image.new("RGB", (8, 8), "white").save(self.image_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def call_response(self, zoom, answer_type, options, observer, sam=None):
        annotation = {
            "input_image": str(self.image_path),
            "question": "What is shown?",
            "answer_type": answer_type,
            "options": options,
        }
        result = CVSearch.get_cvsearch_response(
            sam_model=sam,
            zoom_model=zoom,
            nlp_model=object(),
            annotation=annotation,
            ic_examples=[],
            decomposed_question_template="What is the appearance of the {}?",
            answering_confidence_threshold_upper=0.9,
            answering_confidence_threshold_lower=0.0,
            fast_threshold=0.6,
            pop_limit=10,
            threshold_descrease=[0.1],
            answer_observer=observer,
        )
        return result, annotation

    def test_observer_receives_each_complete_raw_answer_once(self):
        cases = (
            ("logits_match", ["a", "b", "c"], 2),
            ("free_form", None, "raw-1"),
            ("option_list", ["A. red", "B. blue", "C. green", "D. black"], ["raw-1", "raw-2", "raw-3", "raw-4"]),
            ("Multiple Choice", ["A. red", "B. blue"], "raw-1"),
            ("option_single", "A. red\nB. blue", "raw-1"),
        )
        for answer_type, options, expected in cases:
            with self.subTest(answer_type=answer_type):
                observed = []
                zoom = FakeZoom()
                result, annotation = self.call_response(
                    zoom, answer_type, options,
                    lambda name, nodes, answer: observed.append((name, list(nodes), answer)),
                )
                self.assertEqual(result, expected)
                self.assertEqual(len(observed), 1)
                self.assertEqual(observed[0][0], "quick")
                self.assertEqual(observed[0][2], expected)
                self.assertEqual(len(observed[0][1]), 1)
                self.assertEqual(annotation["search_mode"], 0)

    def test_hr_observer_cannot_mutate_returned_answer_or_live_searched_nodes(self):
        zoom = FakeZoom()

        def nested_answer(image_pil, question, searched_nodes):
            zoom.answer_calls += 1
            zoom.searched_node_lists.append(searched_nodes)
            return {"tokens": [f"raw-{zoom.answer_calls}"]}

        zoom.free_form_using_nodes = nested_answer

        def malicious_observer(name, nodes, answer):
            nodes.clear()
            answer[0]["tokens"].clear()
            answer.clear()

        result, annotation = self.call_response(
            zoom,
            "option_list",
            ["A. red", "B. blue", "C. green", "D. black"],
            malicious_observer,
        )

        self.assertEqual(
            result,
            [
                {"tokens": ["raw-1"]},
                {"tokens": ["raw-2"]},
                {"tokens": ["raw-3"]},
                {"tokens": ["raw-4"]},
            ],
        )
        self.assertTrue(all(len(nodes) == 1 for nodes in zoom.searched_node_lists))
        self.assertEqual(len(annotation["searched_bbox"]), 1)

    def test_cross_target_main_and_cropped_calls_keep_outer_question_and_scope(self):
        question = "What words are on the blue sign beside the bus?"
        annotation = {
            "input_image": str(self.image_path),
            "question": question,
            "answer_type": "free_form",
            "options": None,
        }
        zoom = FakeZoom(root_answering=-1.0)
        zoom.generate_visual_cues_using_ic = lambda *_: ["blue sign", "bus"]
        zoom.existence = {"local": 0.5}
        zoom.answering = {"local": -0.5}
        trace = StrictTrace()
        rank_calls = []

        def make_tree(*args):
            root = FakeNode("root", 0, 1.0)
            FakeNode("local", 1, 0.8, root)
            return FakeTree(root, 1)

        def ranker(nodes, image_pil, main_query, augmented_queries):
            rank_calls.append((main_query, list(augmented_queries), image_pil.size))
            return list(nodes), [{} for _ in nodes]

        with patch.object(CVSearch, "include_pronouns", return_value=False), patch.object(
            CVSearch, "normalize_target_text", side_effect=lambda target: (target, False)
        ), patch.object(CVSearch, "ConstrainedTreeBuilder", FakeBuilder), patch.object(
            CVSearch, "AdaptiveImageTree", side_effect=make_tree
        ):
            response = CVSearch.get_cvsearch_response(
                sam_model=FailingFakeSam(),
                zoom_model=zoom,
                nlp_model=object(),
                annotation=annotation,
                ic_examples=[],
                decomposed_question_template="What is the appearance of the {}?",
                answering_confidence_threshold_upper=0.9,
                answering_confidence_threshold_lower=0.0,
                fast_threshold=0.6,
                pop_limit=10,
                threshold_descrease=[0.1],
                node_ranker=ranker,
                method_trace=trace,
            )

        self.assertEqual(response, "raw-1")
        self.assertEqual(len(rank_calls), 4)
        self.assertEqual([call[0] for call in rank_calls], [question] * 4)
        self.assertEqual(
            [call[1] for call in rank_calls],
            [["blue sign"], ["blue sign"], ["bus"], ["bus"]],
        )
        self.assertEqual([detail["tree_scope"] for detail in trace.candidate_ranks], ["main", "cropped", "main", "cropped"])

    def test_rank_context_prefers_plan_main_and_falls_back_for_empty_qaug(self):
        trace = StrictTrace([])
        trace.query_plan.main_query = "planned q0"
        context = CVSearch._make_rank_context(trace, "outer q0", "sign", "cropped", (3, 4))
        self.assertEqual(context["main_query"], "planned q0")
        self.assertEqual(context["augmented_queries"], ["sign"])
        self.assertEqual(context["tree_scope"], "cropped")
        self.assertEqual(context["crop_origin"], (3, 4))

        for main_query in (None, "", "   "):
            trace.query_plan.main_query = main_query
            with self.subTest(main_query=main_query):
                self.assertEqual(
                    CVSearch._make_rank_context(trace, "outer q0", "sign", "main", (0, 0))["main_query"],
                    "outer q0",
                )
        for outer_question in (None, "", "   "):
            trace.query_plan.main_query = ""
            with self.subTest(outer_question=outer_question):
                with self.assertRaises(ValueError):
                    CVSearch._make_rank_context(trace, outer_question, "sign", "main", (0, 0))

    def test_search_observation_is_distinct_and_observer_errors_propagate(self):
        observed = []
        search_zoom = FakeZoom(root_answering=-1.0)
        with patch.object(CVSearch, "include_pronouns", return_value=False), patch.object(
            CVSearch, "normalize_target_text", side_effect=lambda target: (target, False)
        ):
            result, annotation = self.call_response(
                search_zoom,
                "free_form",
                None,
                lambda name, nodes, answer: observed.append((name, answer)),
                sam=FakeSam(),
            )
        self.assertEqual(result, "raw-1")
        self.assertEqual(annotation["search_mode"], 1)
        self.assertEqual(observed, [("search", "raw-1")])

        no_observer_zoom = FakeZoom()
        result, annotation = self.call_response(no_observer_zoom, "logits_match", ["a", "b"], None)
        self.assertEqual((result, annotation["search_mode"], no_observer_zoom.answer_calls), (2, 0, 1))

        quick_zoom = FakeZoom()
        with self.assertRaisesRegex(RuntimeError, "observer failed"):
            self.call_response(
                quick_zoom,
                "logits_match",
                ["a", "b"],
                lambda *_: (_ for _ in ()).throw(RuntimeError("observer failed")),
            )
        self.assertEqual(quick_zoom.answer_calls, 1)


if __name__ == "__main__":
    unittest.main()
