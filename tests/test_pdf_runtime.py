import hashlib
import json
import re
import unittest

from PIL import Image

from qavs.evidence_gap.pdf_runtime import (
    PDFQueryPlan,
    TreeActionAdapter,
    TreeCatalog,
    absolute_location_geometry,
    answer_with_uncertainty,
    build_pdf_query_plan,
    question_kind_from_plan,
    query_required_roles,
)
from qavs.evidence_gap.pdf_types import ActionName, SearchStateRecord
from qavs.evidence_gap.search_state import SearchStateCollector
from qavs.evidence_gap.types import sanitize_evidence_requirements
from tests.helpers import (
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

    def test_presence_plan_uses_question_target_without_text_generation(self):
        policy = dict(self.policy, question="Is there a backpack in the image?\nAnswer yes or no only.",
                      options=["Yes", "No"], answer_type="yes_no")
        plan = build_pdf_query_plan(
            policy, ("image",),
            generator=lambda prompt: self.fail("plain presence needs no generated plan"),
        )
        self.assertEqual(plan.targets, ("backpack",))
        self.assertEqual(query_required_roles(plan), ("backpack",))
        # The existing single-target gate requires two views of ONE instance.
        self.assertEqual(question_kind_from_plan(plan), "attribute")
        self.assertTrue(plan.global_scope_required)  # Keep coverage for No.
        self.assertFalse(plan.fallback_used)
        self.assertEqual(len(plan.augmented_queries), 3)

    def test_presence_single_target_gate_does_not_reclassify_count_or_relation(self):
        for question, expected in (
            ("How many backpacks are in the image?", "count"),
            ("Is the car to the left of the person?", "relation"),
        ):
            with self.subTest(question=question):
                plan = PDFQueryPlan(
                    main_query=question, targets=("car", "person"),
                    augmented_queries=("car", "person", "car and person"),
                    evidence_items=({"kind": "relation_context", "targets": ["car", "person"]},),
                    global_scope_required=True, fallback_used=False,
                    fallback_reason=None, raw_response_sha256="a" * 64,
                )
                self.assertEqual(question_kind_from_plan(plan), expected)

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
            "detail_demand": 1.0,
            "context_demand": 0.75,
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
        self.assertEqual(plan.detail_demand, 1.0)
        self.assertEqual(plan.context_demand, 0.75)
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
                    "detail_demand": 1.0,
                    "context_demand": 0.0,
                })
                plan = build_pdf_query_plan(
                    self.policy, ("sign",), generator=lambda prompt: raw,
                )
                self.assertFalse(plan.fallback_used)
                self.assertEqual(
                    plan.evidence_items[0]["requirements"],
                    ["presence", "visual_detail"],
                )

    def test_question_text_recovers_relation_domain_when_planner_emits_only_details(self):
        plan = PDFQueryPlan(
            main_query=(
                "Is the blue mask on the left or right side of the black mask?"
            ),
            targets=("blue mask", "black mask"),
            augmented_queries=("blue mask", "black mask", "left or right"),
            evidence_items=(
                {
                    "kind": "target_detail", "target": "blue mask",
                    "requirements": ["presence", "visual_detail"],
                },
                {
                    "kind": "target_detail", "target": "black mask",
                    "requirements": ["presence", "visual_detail"],
                },
            ),
            global_scope_required=False,
            fallback_used=False,
            fallback_reason=None,
            raw_response_sha256="a" * 64,
        )

        self.assertEqual(question_kind_from_plan(plan), "relation")
        self.assertEqual(
            query_required_roles(plan), ("blue mask", "black mask"),
        )

    def test_unary_spatial_locator_is_not_a_multi_target_relation(self):
        plan = PDFQueryPlan(
            main_query="What type of structure is visible to the right of the image?",
            targets=("structure",),
            augmented_queries=(
                "structure on image right", "right-side structure", "structure detail",
            ),
            evidence_items=(
                {
                    "kind": "relation_context",
                    "targets": ["structure", "right side"],
                },
            ),
            global_scope_required=False,
            fallback_used=False,
            fallback_reason=None,
            raw_response_sha256="a" * 64,
        )

        self.assertEqual(question_kind_from_plan(plan), "attribute")
        self.assertEqual(query_required_roles(plan), ("structure",))

    def test_relation_choice_recovers_second_role_from_planner_evidence(self):
        plan = PDFQueryPlan(
            main_query=(
                "Is the volleyball on the left or right side of the man with white cap?"
            ),
            targets=("volleyball",),
            augmented_queries=(
                "volleyball", "man with white cap", "volleyball relative position",
            ),
            evidence_items=(
                {
                    "kind": "relation_context",
                    "targets": ["volleyball", "man with white cap"],
                },
            ),
            global_scope_required=False,
            fallback_used=False,
            fallback_reason=None,
            raw_response_sha256="a" * 64,
        )

        self.assertEqual(question_kind_from_plan(plan), "relation")
        self.assertEqual(
            query_required_roles(plan), ("volleyball", "man with white cap"),
        )

    def test_closer_question_is_a_comparison(self):
        plan = PDFQueryPlan(
            main_query=(
                "Which one is closer to the camera, the water bottle or the vehicle?"
            ),
            targets=("camera", "water bottle", "vehicle"),
            augmented_queries=(
                "water bottle", "vehicle", "distance from camera",
            ),
            evidence_items=(
                {
                    "kind": "target_detail",
                    "target": "water bottle",
                    "requirements": ["presence", "visual_detail"],
                },
                {
                    "kind": "target_detail",
                    "target": "vehicle",
                    "requirements": ["presence", "visual_detail"],
                },
            ),
            global_scope_required=False,
            fallback_used=False,
            fallback_reason=None,
            raw_response_sha256="a" * 64,
        )

        self.assertEqual(question_kind_from_plan(plan), "comparison")
        self.assertEqual(
            query_required_roles(plan), ("camera", "water bottle", "vehicle"),
        )

    def test_requested_property_remains_attribute_with_relational_locator(self):
        plan = PDFQueryPlan(
            main_query="What is the color of the pink-haired woman's handbag?",
            targets=("pink-haired woman", "handbag"),
            augmented_queries=(
                "pink-haired woman", "woman's handbag", "handbag color detail",
            ),
            evidence_items=(
                {
                    "kind": "target_detail",
                    "target": "pink-haired woman's handbag",
                    "requirements": ["presence", "visual_detail"],
                },
                {
                    "kind": "relation_context",
                    "targets": ["pink-haired woman", "handbag"],
                },
            ),
            global_scope_required=False,
            fallback_used=False,
            fallback_reason=None,
            raw_response_sha256="a" * 64,
        )

        self.assertEqual(question_kind_from_plan(plan), "attribute")

    def test_identifier_number_is_not_a_count_question(self):
        policy = {
            **self.policy,
            "question": "What is the license plate number of the vehicle?",
        }
        raw = json.dumps({
            "augmented_queries": [
                "vehicle license plate", "license plate detail", "vehicle detail",
            ],
            "evidence_items": [{
                "kind": "target_detail", "target": "vehicle",
                "requirements": ["presence", "visual_detail"],
            }],
            "global_scope_required": False,
            "detail_demand": 1.0,
            "context_demand": 0.0,
        })
        plan = build_pdf_query_plan(
            policy, ("vehicle",), generator=lambda prompt: raw,
        )

        self.assertFalse(plan.global_scope_required)
        self.assertEqual(question_kind_from_plan(plan), "attribute")
        self.assertEqual(query_required_roles(plan), ("vehicle",))

    def test_number_of_objects_remains_a_count_question(self):
        policy = {
            **self.policy,
            "question": "What is the number of people visible in the image?",
        }
        raw = json.dumps({
            "augmented_queries": [
                "visible people", "people instances", "full image people coverage",
            ],
            "evidence_items": [{
                "kind": "target_detail", "target": "people",
                "requirements": ["presence", "visual_detail"],
            }],
            "global_scope_required": False,
            "detail_demand": 0.5,
            "context_demand": 1.0,
        })
        plan = build_pdf_query_plan(
            policy, ("people",), generator=lambda prompt: raw,
        )

        self.assertTrue(plan.global_scope_required)
        self.assertEqual(question_kind_from_plan(plan), "count")
        self.assertEqual(query_required_roles(plan), ("people",))

    def test_structured_planner_canonicalizes_global_scope_from_q0(self):
        relation_raw = json.dumps({
            "augmented_queries": ["locate sign", "sign detail", "door context"],
            "evidence_items": [
                {"kind": "target_detail", "target": "sign",
                 "requirements": ["presence", "visual_detail"]},
                {"kind": "coverage", "requirement": "global_scope"},
            ],
            "global_scope_required": True,
            "detail_demand": 0.5,
            "context_demand": 1.0,
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
            "detail_demand": 0.5,
            "context_demand": 1.0,
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
    def test_zoom_changes_original_geometry_and_expand_rejects_distant_context(self):
        image = Image.new("RGB", (100, 80), "white")
        root_key = candidate_key((0, 0, 100, 80), 0)
        focus_key = candidate_key((10, 10, 40, 40), 1)
        distant_key = candidate_key((90, 0, 10, 10), 1)
        candidates = [
            tree_candidate(
                (0, 0, 100, 80), depth=0, parent=None,
                children=(focus_key, distant_key), rank=0, complexity=0.2,
            ),
            tree_candidate(
                (10, 10, 40, 40), depth=1, parent=root_key,
                children=(), rank=1, complexity=0.8,
            ),
            tree_candidate(
                (90, 0, 10, 10), depth=1, parent=root_key,
                children=(), rank=2, complexity=0.4,
            ),
        ]
        refs, snapshot = event_for(
            image, candidates, event="tree_ready", remaining=(),
        )
        collector = SearchStateCollector(image)
        collector(refs, snapshot)
        catalog = TreeCatalog.from_collector(collector, image)
        plan = PDFQueryPlan(
            main_query="What color is the object?", targets=("object",),
            augmented_queries=("locate object", "object color", "object detail"),
            evidence_items=({
                "kind": "target_detail", "target": "object",
                "requirements": ["presence", "visual_detail"],
            },),
            global_scope_required=False, fallback_used=False,
            fallback_reason=None, raw_response_sha256="a" * 64,
            detail_demand=1.0, context_demand=0.5,
        )
        adapter = TreeActionAdapter(
            catalog,
            image,
            plan,
            lambda nodes, image_pil, main_query, augmented_queries: (
                list(nodes), [
                    {"score": {"rank": 1.0 - index * 0.1}}
                    for index, _ in enumerate(nodes)
                ],
            ),
            min_zoom_factor=0.4,
            context_max_normalized_gap=0.1,
        )
        state = SearchStateRecord(
            state_id=1, focus_keys=(focus_key,), path_keys=(root_key, focus_key),
            context_keys=(), visited_keys=(root_key, focus_key),
            observation_keys=(f"{root_key}@root", f"{focus_key}@base"),
            remaining_steps=7, remaining_model_calls=40,
            remaining_pixels=100000,
        )

        zoomed = adapter.execute(ActionName.ZOOM, state)
        zoom_state = SearchStateRecord(
            state_id=2,
            focus_keys=zoomed.focus_keys,
            path_keys=zoomed.path_keys,
            context_keys=zoomed.context_keys,
            visited_keys=zoomed.visited_keys,
            observation_keys=zoomed.observation_keys,
            remaining_steps=6,
            remaining_model_calls=40,
            remaining_pixels=100000,
        )
        assert adapter.effective_geometry(state) == (10.0, 10.0, 40.0, 40.0)
        assert adapter.effective_geometry(zoom_state) == (15.0, 15.0, 30.0, 30.0)
        assert adapter.execute(ActionName.EXPAND, state).reason == "no_adjacent_context"

    def test_large_shallow_detail_crop_needs_zoom_or_deeper_path(self):
        image, catalog, (root_key, left_key, _) = make_full_catalog()
        plan = PDFQueryPlan(
            main_query="What color is the object?", targets=("object",),
            augmented_queries=("locate object", "object color", "object detail"),
            evidence_items=({
                "kind": "target_detail", "target": "object",
                "requirements": ["presence", "visual_detail"],
            },),
            global_scope_required=False, fallback_used=False,
            fallback_reason=None, raw_response_sha256="a" * 64,
        )

        def ranker(nodes, image_pil, main_query, augmented_queries):
            return list(nodes), [{"score": {"rank": 1.0}} for _ in nodes]

        adapter = TreeActionAdapter(catalog, image, plan, ranker)
        state = SearchStateRecord(
            state_id=1, focus_keys=(left_key,), path_keys=(root_key, left_key),
            context_keys=(), visited_keys=(root_key, left_key),
            observation_keys=(f"{root_key}@root", f"{left_key}@base"),
            remaining_steps=7, remaining_model_calls=40, remaining_pixels=100000,
        )
        requirements = sanitize_evidence_requirements(plan.evidence_items)

        shallow = absolute_location_geometry(
            adapter, state, plan.main_query, requirements,
        )
        self.assertEqual(shallow["focus_area_fraction"], 0.5)
        self.assertEqual(shallow["focus_path_depth"], 1)
        self.assertFalse(shallow["detail_resolution_met"])
        self.assertFalse(shallow["detail_localized"])

        zoomed = absolute_location_geometry(
            adapter,
            SearchStateRecord(
                **{
                    **state.__dict__,
                    "observation_keys": state.observation_keys + (f"{left_key}@zoom1",),
                }
            ),
            plan.main_query,
            requirements,
        )
        self.assertTrue(zoomed["detail_resolution_met"])
        self.assertTrue(zoomed["detail_localized"])

    def test_nonroot_verifier_view_keeps_overview_and_local_detail(self):
        image, catalog, (root_key, left_key, _) = make_full_catalog()
        image.paste((255, 0, 0), (0, 0, 4, 8))
        image.paste((0, 0, 255), (4, 0, 8, 8))
        plan = PDFQueryPlan(
            main_query="Which object is on the left?", targets=("object",),
            augmented_queries=("left object", "object detail", "image context"),
            evidence_items=({"kind": "relation_context", "targets": ["object"]},),
            global_scope_required=False, fallback_used=False,
            fallback_reason=None, raw_response_sha256="a" * 64,
        )

        def ranker(nodes, image_pil, main_query, augmented_queries):
            return list(nodes), [
                {"score": {"rank": 1.0 - index * 0.1}}
                for index, _ in enumerate(nodes)
            ]

        adapter = TreeActionAdapter(catalog, image, plan, ranker)
        root = SearchStateRecord(
            state_id=0, focus_keys=(root_key,), path_keys=(root_key,),
            context_keys=(), visited_keys=(root_key,),
            observation_keys=(f"{root_key}@root",), remaining_steps=8,
            remaining_model_calls=40, remaining_pixels=100000,
        )
        child = SearchStateRecord(
            state_id=1, focus_keys=(left_key,), path_keys=(root_key, left_key),
            context_keys=(), visited_keys=(root_key, left_key),
            observation_keys=(f"{root_key}@root", f"{left_key}@base"),
            remaining_steps=7, remaining_model_calls=40,
            remaining_pixels=100000,
        )

        self.assertEqual(adapter.render_verifier_view(root).size, image.size)
        verifier_view = adapter.render_verifier_view(child)
        self.assertGreater(verifier_view.height, image.height)
        self.assertEqual(verifier_view.getpixel((0, 0)), (255, 215, 0))
        self.assertEqual(verifier_view.getpixel((7, 4)), (0, 0, 255))
        self.assertEqual(
            verifier_view.getpixel((verifier_view.width // 2, verifier_view.height - 1)),
            (255, 0, 0),
        )

        local_view = adapter.render_local_verifier_view(child)
        self.assertEqual(local_view.size, (4, 8))
        self.assertEqual(set(local_view.getdata()), {(255, 0, 0)})

    def test_root_relation_verifier_view_obeys_2048_edge_cap(self):
        _, catalog, (root_key, _, _) = make_full_catalog()
        image = Image.new("RGB", (4096, 64), "white")
        plan = PDFQueryPlan(
            main_query="What is behind the statue?", targets=("building", "statue"),
            augmented_queries=("locate building", "locate statue", "relation context"),
            evidence_items=({
                "kind": "relation_context", "targets": ["building", "statue"],
            },),
            global_scope_required=False, fallback_used=False,
            fallback_reason=None, raw_response_sha256="a" * 64,
        )
        adapter = TreeActionAdapter(
            catalog, image, plan,
            lambda nodes, image_pil, main_query, augmented_queries: (
                list(nodes), [
                    {"score": {"rank": 1.0 - index * 0.1}}
                    for index, _ in enumerate(nodes)
                ]
            ),
        )
        root = SearchStateRecord(
            state_id=0, focus_keys=(root_key,), path_keys=(root_key,),
            context_keys=(), visited_keys=(root_key,),
            observation_keys=(f"{root_key}@root",), remaining_steps=8,
            remaining_model_calls=40, remaining_pixels=600_000_000,
        )

        self.assertEqual(adapter.render_verifier_view(root).size, (2048, 32))

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

    def test_catalog_drops_child_edges_owned_by_another_parent(self):
        image = Image.new("RGB", (8, 8), "white")
        root_key = candidate_key((0, 0, 8, 8), 0)
        left_key = candidate_key((0, 0, 4, 8), 1)
        right_key = candidate_key((4, 0, 4, 8), 1)
        child_key = candidate_key((0, 0, 2, 4), 2)
        candidates = [
            tree_candidate(
                (0, 0, 8, 8), depth=0, parent=None,
                children=(left_key, right_key), rank=0, complexity=0.2,
            ),
            tree_candidate(
                (0, 0, 4, 8), depth=1, parent=root_key,
                children=(child_key,), rank=1, complexity=0.8,
            ),
            tree_candidate(
                (4, 0, 4, 8), depth=1, parent=root_key,
                children=(child_key,), rank=2, complexity=0.4,
            ),
            tree_candidate(
                (0, 0, 2, 4), depth=2, parent=left_key,
                children=(), rank=3, complexity=0.9,
            ),
        ]
        refs, snapshot = event_for(
            image, candidates, event="tree_ready", remaining=(),
        )
        collector = SearchStateCollector(image)
        collector(refs, snapshot)

        catalog = TreeCatalog.from_collector(collector, image)

        self.assertEqual(catalog.children(left_key), (child_key,))
        self.assertEqual(catalog.children(right_key), ())
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
        self.assertEqual(rendered.size, (8, 12))
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


class CachedOptionLabelLossesTest(unittest.TestCase):
    def test_exact_inputs_reuse_scores_without_changing_logical_cost(self):
        from qavs.evidence_gap.pdf_runtime import CachedOptionLabelLosses
        from types import SimpleNamespace

        calls = []
        def score(image, question, codes, nodes):
            calls.append((image.size, image.tobytes(), question, tuple(codes)))
            return 1, [0.875, 0.125, 1.75]
        model = SimpleNamespace(label_token_losses=score)
        cached = CachedOptionLabelLosses(model)
        view = Image.new("RGB", (3, 2), "red")
        labels = ("Support", "Refute", "Insufficient")
        original = cached(view, "Candidate answer: Yes", labels)
        repeat = cached(view.copy(), "Candidate answer: Yes", labels)
        self.assertEqual(original, (1, [0.875, 0.125, 1.75], 1))
        self.assertEqual(repeat, original)
        self.assertEqual(len(calls), 1)
        repeat[1][0] = 99
        self.assertEqual(cached(view, "Candidate answer: Yes", labels), original)
        cached(view, "Candidate answer: No", labels)
        cached(view, "Candidate answer: Yes. Confirm independently.", labels)
        cached(Image.new("RGB", (3, 2), "blue"), "Candidate answer: Yes", labels)
        cached(Image.new("RGB", (2, 3), "red"), "Candidate answer: Yes", labels)
        cached(view, "Candidate answer: Yes", ("Grounded", "NotGrounded", "Insufficient"))
        self.assertEqual(len(calls), 6)
        CachedOptionLabelLosses(model)(view, "Candidate answer: Yes", labels)
        self.assertEqual(len(calls), 7)  # A fresh sample never shares a cache.

    def test_non_rgb_inputs_do_not_alias_alpha_or_palette(self):
        from qavs.evidence_gap.pdf_runtime import CachedOptionLabelLosses
        from types import SimpleNamespace
        calls = []
        def score(image, *args):
            calls.append(1)
            return 0, [float(image.getpixel((0, 0))[3]), 256.0, 257.0]
        cached = CachedOptionLabelLosses(SimpleNamespace(label_token_losses=score))
        labels = ("Support", "Refute", "Insufficient")
        transparent = Image.new("RGBA", (2, 2), (255, 0, 0, 0))
        opaque = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
        self.assertNotEqual(cached(transparent, "Candidate answer: Yes", labels),
                            cached(opaque, "Candidate answer: Yes", labels))
        self.assertEqual(len(calls), 2)

    def test_training_mode_never_reuses_scores(self):
        from qavs.evidence_gap.pdf_runtime import CachedOptionLabelLosses
        from types import SimpleNamespace
        calls = []
        def score(*args):
            calls.append(1)
            return 0, [0.1, 0.4, 0.5]
        model = SimpleNamespace(model=SimpleNamespace(training=True), label_token_losses=score)
        cached = CachedOptionLabelLosses(model)
        args = (Image.new("RGB", (2, 2)), "Candidate answer: Yes", ("Support", "Refute", "Insufficient"))
        cached(*args)
        cached(*args)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
