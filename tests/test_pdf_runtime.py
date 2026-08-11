import hashlib
import json
import re
import unittest

from PIL import Image

from cvsearch.evidence_gap.pdf_runtime import (
    PDFQueryPlan,
    TreeActionAdapter,
    TreeCatalog,
    answer_with_uncertainty,
    build_pdf_query_plan,
)
from cvsearch.evidence_gap.pdf_types import ActionName, SearchStateRecord
from cvsearch.evidence_gap.search_state import SearchStateCollector
from cvsearch.evidence_gap.types import sanitize_evidence_requirements
from tests.test_evidence_gap_search_state import (
    candidate_key,
    event_for,
    source_identity,
)


def tree_candidate(bbox, *, depth, parent, children, rank, complexity):
    return {
        "canonical_key": candidate_key(bbox, depth),
        "bbox_original": list(bbox),
        "parent_key": parent,
        "child_keys": list(children),
        "depth": depth,
        "render_level": 0,
        "source": "global" if depth == 0 else "fine",
        "stage_rank": rank,
        "prior_prob": 0.5,
        "complexity": complexity,
        "fast_confidence": None,
        "posterior_score": 0.5,
        "is_evaluated": False,
        "answering_confidence": None,
    }


def make_full_catalog():
    image = Image.new("RGB", (8, 8), "white")
    root_key = candidate_key((0, 0, 8, 8), 0)
    left_key = candidate_key((0, 0, 4, 8), 1)
    right_key = candidate_key((4, 0, 4, 8), 1)
    candidates = [
        tree_candidate((0, 0, 8, 8), depth=0, parent=None,
                       children=(left_key, right_key), rank=0, complexity=0.2),
        tree_candidate((0, 0, 4, 8), depth=1, parent=root_key,
                       children=(), rank=1, complexity=0.8),
        tree_candidate((4, 0, 4, 8), depth=1, parent=root_key,
                       children=(), rank=2, complexity=0.4),
    ]
    refs, snapshot = event_for(image, candidates, event="tree_ready", remaining=())
    snapshot.update({"stage": "Full Tree", "depth": 0})
    collector = SearchStateCollector(image)
    collector(refs, snapshot)
    return image, TreeCatalog.from_collector(collector, image), (root_key, left_key, right_key)


class PDFQueryPlannerTest(unittest.TestCase):
    def setUp(self):
        self.policy = {
            "question": "What color is the sign above the door?",
            "options": ["red", "blue", "green", "yellow"],
            "answer_type": "logits_match",
            "input_image": "image.jpg",
        }

    def test_structured_plan_keeps_q0_and_answer_free_main_top3_material(self):
        raw = json.dumps({
            "augmented_queries": [
                "sign above the door", "small sign lettering",
                "door and overhead sign context", "building entrance sign",
            ],
            "evidence_items": [
                {"kind": "target_detail", "target": "sign above the door",
                 "requirements": ["presence", "visual_detail"]},
                {"kind": "relation_context", "targets": ["sign", "door"]},
            ],
            "global_scope_required": False,
        })
        prompts = []
        plan = build_pdf_query_plan(
            self.policy, ("sign above the door",),
            generator=lambda prompt: prompts.append(prompt) or raw,
        )
        self.assertIsInstance(plan, PDFQueryPlan)
        self.assertEqual(plan.main_query, self.policy["question"])
        self.assertEqual(len(plan.augmented_queries), 4)
        self.assertFalse(plan.fallback_used)
        self.assertIsNone(re.search(r"\bred\b", prompts[0].casefold()))
        self.assertNotIn("options", prompts[0].casefold())
        self.assertEqual(len(sanitize_evidence_requirements(plan.evidence_items)), 2)
        self.assertEqual(json.loads(json.dumps(plan.to_dict())), plan.to_dict())

    def test_structured_planner_canonicalizes_answer_free_detail_vocabularies(self):
        for requirements in (["visual_detail", "presence"], ["color"]):
            with self.subTest(requirements=requirements):
                raw = json.dumps({
                    "augmented_queries": ["locate sign", "sign detail", "sign context"],
                    "evidence_items": [{
                        "kind": "target_detail", "target": "sign",
                        "requirements": requirements,
                    }],
                    "global_scope_required": False,
                })
                plan = build_pdf_query_plan(
                    self.policy, ("sign",), generator=lambda prompt: raw,
                )
                self.assertFalse(plan.fallback_used)
                self.assertEqual(
                    plan.evidence_items[0]["requirements"],
                    ["presence", "visual_detail"],
                )

    def test_structured_planner_canonicalizes_global_scope_from_q0(self):
        relation_raw = json.dumps({
            "augmented_queries": ["locate sign", "sign detail", "door context"],
            "evidence_items": [
                {"kind": "target_detail", "target": "sign",
                 "requirements": ["presence", "visual_detail"]},
                {"kind": "coverage", "requirement": "global_scope"},
            ],
            "global_scope_required": True,
        })
        relation = build_pdf_query_plan(
            self.policy, ("sign",), generator=lambda prompt: relation_raw,
        )
        self.assertFalse(relation.global_scope_required)
        self.assertFalse(any(
            item["kind"] == "coverage" for item in relation.evidence_items
        ))

        count_policy = dict(self.policy, question="How many signs are above the door?")
        count_raw = json.dumps({
            "augmented_queries": ["locate signs", "sign count", "door signs"],
            "evidence_items": [{
                "kind": "target_detail", "target": "sign",
                "requirements": ["presence", "visual_detail"],
            }],
            "global_scope_required": False,
        })
        count = build_pdf_query_plan(
            count_policy, ("sign",), generator=lambda prompt: count_raw,
        )
        self.assertTrue(count.global_scope_required)
        self.assertTrue(any(item["kind"] == "coverage" for item in count.evidence_items))

    def test_malformed_or_option_leaking_plan_gets_nontrivial_logged_fallback(self):
        for raw in (
            "not json",
            json.dumps({
                "augmented_queries": ["red"],
                "evidence_items": [],
                "global_scope_required": False,
            }),
        ):
            with self.subTest(raw=raw):
                plan = build_pdf_query_plan(
                    self.policy, ("sign above the door",), generator=lambda prompt: raw,
                )
                self.assertTrue(plan.fallback_used)
                self.assertGreaterEqual(len(plan.augmented_queries), 3)
                self.assertTrue(plan.evidence_items)
                self.assertTrue(plan.fallback_reason)
                self.assertEqual(
                    plan.raw_response_sha256, hashlib.sha256(raw.encode()).hexdigest()
                )


class TreeCatalogTest(unittest.TestCase):
    def test_cropped_tree_root_is_grafted_to_matching_main_tree_region(self):
        image = Image.new("RGB", (8, 8), "white")
        root_key = candidate_key((0, 0, 8, 8), 0)
        anchor_key = candidate_key((2, 2, 4, 4), 1)
        cropped_root_key = candidate_key((2, 2, 4, 4), 0)
        cropped_leaf_key = candidate_key((3, 3, 2, 2), 1)
        main = [
            tree_candidate(
                (0, 0, 8, 8), depth=0, parent=None,
                children=(anchor_key,), rank=0, complexity=0.2,
            ),
            tree_candidate(
                (2, 2, 4, 4), depth=1, parent=root_key,
                children=(), rank=1, complexity=0.8,
            ),
        ]
        cropped = [
            tree_candidate(
                (2, 2, 4, 4), depth=0, parent=None,
                children=(cropped_leaf_key,), rank=0, complexity=0.7,
            ),
            tree_candidate(
                (3, 3, 2, 2), depth=1, parent=cropped_root_key,
                children=(), rank=1, complexity=0.9,
            ),
        ]
        collector = SearchStateCollector(image)
        refs, snapshot = event_for(
            image, main, event="tree_ready", scope="main", ordinal=1,
            remaining=(),
        )
        collector(refs, snapshot)
        refs, snapshot = event_for(
            image, cropped, event="tree_ready", scope="cropped", origin=(2, 2),
            ordinal=2, remaining=(),
        )
        collector(refs, snapshot)

        catalog = TreeCatalog.from_collector(collector, image)

        self.assertEqual(
            catalog.path_to(cropped_leaf_key),
            (root_key, anchor_key, cropped_root_key, cropped_leaf_key),
        )
        self.assertIn(cropped_root_key, catalog.children(anchor_key))

    def test_declared_children_below_emitted_max_depth_are_explicitly_truncated(self):
        image = Image.new("RGB", (8, 8), "white")
        root_key = candidate_key((0, 0, 8, 8), 0)
        leaf_key = candidate_key((0, 0, 4, 8), 1)
        omitted_key = candidate_key((0, 0, 2, 8), 2)
        candidates = [
            tree_candidate((0, 0, 8, 8), depth=0, parent=None,
                           children=(leaf_key,), rank=0, complexity=0.2),
            tree_candidate((0, 0, 4, 8), depth=1, parent=root_key,
                           children=(omitted_key,), rank=1, complexity=0.8),
        ]
        refs, snapshot = event_for(image, candidates, event="tree_ready", remaining=())
        snapshot.update({"stage": "Full Tree", "depth": 0})
        collector = SearchStateCollector(image)
        collector(refs, snapshot)
        catalog = TreeCatalog.from_collector(collector, image)
        self.assertEqual(catalog.children(leaf_key), ())
        self.assertEqual(catalog.truncated_child_edges, 1)

    def test_full_tree_snapshot_preserves_root_parent_children_and_rendering(self):
        image = Image.new("RGB", (8, 8), "white")
        root_key = candidate_key((0, 0, 8, 8), 0)
        left_key = candidate_key((0, 0, 4, 8), 1)
        right_key = candidate_key((4, 0, 4, 8), 1)
        candidates = [
            tree_candidate((0, 0, 8, 8), depth=0, parent=None,
                           children=(left_key, right_key), rank=0, complexity=0.2),
            tree_candidate((0, 0, 4, 8), depth=1, parent=root_key,
                           children=(), rank=1, complexity=0.8),
            tree_candidate((4, 0, 4, 8), depth=1, parent=root_key,
                           children=(), rank=2, complexity=0.4),
        ]
        refs, snapshot = event_for(
            image, candidates, event="tree_ready", remaining=(),
        )
        snapshot.update({"stage": "Full Tree", "depth": 0})
        collector = SearchStateCollector(image)
        collector(refs, snapshot)

        catalog = TreeCatalog.from_collector(collector, image)
        self.assertEqual(catalog.root_key, root_key)
        self.assertEqual(catalog.children(root_key), (left_key, right_key))
        self.assertEqual(catalog.path_to(right_key), (root_key, right_key))
        rendered = catalog.render_nodes((left_key, right_key))
        self.assertEqual([node.complexity for node in rendered], [0.8, 0.4])
        self.assertEqual([node.state.bbox for node in rendered], [[0, 0, 4, 8], [4, 0, 4, 8]])

    def test_fast_candidates_get_a_whole_image_root_fallback(self):
        image = Image.new("RGB", (8, 8), "white")
        fast = tree_candidate(
            (2, 2, 2, 2), depth=1, parent=None, children=(), rank=0, complexity=0.3,
        )
        fast["source"] = "fast"
        refs, snapshot = event_for(
            image, [fast], event="p0_selected", selected=(fast["canonical_key"],), remaining=(),
        )
        collector = SearchStateCollector(image)
        collector(refs, snapshot)
        catalog = TreeCatalog.from_collector(collector, image)
        self.assertEqual(catalog.children(catalog.root_key), (fast["canonical_key"],))
        self.assertEqual(catalog.path_to(fast["canonical_key"]), (catalog.root_key, fast["canonical_key"]))
        self.assertTrue(catalog.render_nodes((catalog.root_key,))[0].is_root)

    def test_actions_follow_joint_sibling_order_and_keep_explicit_tree_path(self):
        image, catalog, (root_key, left_key, right_key) = make_full_catalog()
        plan = PDFQueryPlan(
            main_query="question", targets=("object",),
            augmented_queries=("locate object", "object detail", "object context"),
            evidence_items=({"kind": "target_detail", "target": "object",
                             "requirements": ["presence", "visual_detail"]},),
            global_scope_required=False, fallback_used=False,
            fallback_reason=None, raw_response_sha256="a" * 64,
        )

        def reverse_rank(nodes, image_pil, main_query, augmented_queries):
            ranked = list(reversed(nodes))
            details = []
            for index, node in enumerate(ranked):
                details.append({"node_id": getattr(node, "id", None), "score": {"rank": 0.9 - 0.8 * index}})
            return ranked, details

        adapter = TreeActionAdapter(catalog, image, plan, reverse_rank)
        root = SearchStateRecord(
            state_id=0, focus_keys=(root_key,), path_keys=(root_key,),
            context_keys=(), visited_keys=(root_key,),
            observation_keys=(f"{root_key}@root",), remaining_steps=8,
            remaining_model_calls=40, remaining_pixels=100000,
        )
        self.assertEqual(adapter.queue.native_first_choice_changes, 1)
        self.assertTrue(adapter.feasible(root)[ActionName.SPLIT])
        self.assertFalse(adapter.feasible(root)[ActionName.NEXT])
        self.assertFalse(adapter.feasible(root)[ActionName.EXPAND])
        split = adapter.execute(ActionName.SPLIT, root)
        self.assertEqual(split.path_keys, (root_key, right_key))
        self.assertEqual(split.focus_keys, (right_key,))

        child = SearchStateRecord(
            state_id=1, focus_keys=split.focus_keys, path_keys=split.path_keys,
            context_keys=split.context_keys, visited_keys=split.visited_keys,
            observation_keys=split.observation_keys, remaining_steps=7,
            remaining_model_calls=40, remaining_pixels=100000,
        )
        self.assertTrue(adapter.branch_available(child))
        expanded = adapter.execute(ActionName.EXPAND, child)
        self.assertEqual(expanded.context_keys, (left_key,))
        zoomed = adapter.execute(ActionName.ZOOM, child)
        zoom_state = SearchStateRecord(
            state_id=2, focus_keys=zoomed.focus_keys, path_keys=zoomed.path_keys,
            context_keys=zoomed.context_keys, visited_keys=zoomed.visited_keys,
            observation_keys=zoomed.observation_keys, remaining_steps=6,
            remaining_model_calls=40, remaining_pixels=100000,
        )
        rendered, nodes = adapter.render_state(zoom_state)
        self.assertEqual(rendered.size, (8, 16))
        self.assertEqual(nodes, [])

        next_outcome = adapter.execute(ActionName.NEXT, child)
        self.assertEqual(next_outcome.path_keys, (root_key, left_key))


class FakeAnswerModel:
    def __init__(self, losses=(), outputs=()):
        self.losses = iter(losses)
        self.outputs = iter(outputs)
        self.questions = []

    def multiple_choices_with_losses(self, image, question, options, nodes):
        self.questions.append(question)
        row = next(self.losses)
        return min(range(len(row)), key=row.__getitem__), row

    def free_form_using_nodes(self, image, question, nodes):
        self.questions.append(question)
        return next(self.outputs)


class AnswerUncertaintyTest(unittest.TestCase):
    def test_vstar_uses_three_paraphrases_and_combines_margin_with_agreement(self):
        model = FakeAnswerModel(losses=(
            [0.1, 0.9], [0.2, 0.8], [0.9, 0.1],
        ))
        policy = {
            "question": "Which side?", "options": ["left", "right"],
            "answer_type": "logits_match", "input_image": "x.jpg",
        }
        record = answer_with_uncertainty(
            model, policy, Image.new("RGB", (4, 4)), [],
        )
        self.assertEqual(len(model.questions), 3)
        self.assertEqual(record.groups["prompt_winners"], [0, 0, 1])
        self.assertAlmostEqual(record.frequency, 2 / 3)
        self.assertAlmostEqual(record.uncertainty, 1.0 - record.confidence)
        self.assertNotEqual(model.questions[0], model.questions[1])

    def test_hr_uses_all_four_shuffles_and_semantic_uncertainty(self):
        block = "A. red\nB. blue\nC. green\nD. yellow"
        model = FakeAnswerModel(outputs=("A", "A", "A", "A"))
        policy = {
            "question": "What color?", "options": [block] * 4,
            "answer_type": "option_list", "input_image": "x.jpg",
        }
        record = answer_with_uncertainty(
            model, policy, Image.new("RGB", (4, 4)), [],
        )
        self.assertEqual(len(model.questions), 4)
        self.assertEqual(record.output, ["A"] * 4)
        self.assertEqual(record.frequency, 1.0)
        self.assertEqual(record.uncertainty, 0.0)


if __name__ == "__main__":
    unittest.main()
