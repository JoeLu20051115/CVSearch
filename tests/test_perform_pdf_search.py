import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from PIL import Image

from cvsearch import perform_PDFSearch as pdf_runner
from cvsearch.eval.pdf_trace_audit import audit_pdf_trace
from cvsearch.evidence_gap.provenance import canonical_sha256
from cvsearch.perform_PDFSearch import (
    RUNNER_VERSION,
    _history_spatial_support_count,
    build_parser,
    family_candidate_kwargs,
    run_pdf_sample,
)
from tests.test_evidence_gap_search_state import candidate_key, source_key
from tests.test_pdf_types import full_config


class FakeGenerator:
    def generate_text_only(self, prompt):
        return json.dumps({
            "augmented_queries": ["locate sign", "sign detail", "sign context", "small sign"],
            "evidence_items": [{
                "kind": "target_detail", "target": "sign",
                "requirements": ["presence", "visual_detail"],
            }],
            "global_scope_required": False,
        })

    def multiple_choices_with_losses(self, image, question, options, nodes):
        return 0, [0.1, 0.9]

    def free_form_using_nodes(self, image, question, nodes):
        if nodes and getattr(nodes[0], "depth", 0) == 0:
            return '{"zoom":0.1,"split":0.9,"expand":0.2,"next":0.3}'
        return '{"zoom":0.1,"split":0.1,"expand":0.1,"next":0.1}'


class FakeVerifier:
    def get_confidence_value(self, nodes, image, confidence_type, input_ele):
        return 0.8


class FakeClip:
    def score(self, images, texts):
        rows = []
        for index, _ in enumerate(images):
            rows.append([0.1 + index * 0.5] * len(texts))
        return rows


def fake_cvsearch(**kwargs):
    annotation = kwargs["annotation"]
    image = Image.open(Path(kwargs["image_folder"]) / annotation["input_image"]).convert("RGB")
    sink = kwargs["search_state_sink"]
    root_key = candidate_key((0, 0, 8, 8), 0)
    left_key = candidate_key((0, 0, 2, 8), 1)
    right_key = candidate_key((6, 0, 2, 8), 1)
    candidates = [
        (root_key, (0, 0, 8, 8), 0, None, [left_key, right_key], 0.2),
        (left_key, (0, 0, 2, 8), 1, root_key, [], 0.3),
        (right_key, (6, 0, 2, 8), 1, root_key, [], 0.9),
    ]
    snapshots = [{
        "canonical_key": key, "bbox_original": list(box), "parent_key": parent,
        "child_keys": children, "depth": depth, "render_level": 0,
        "source": "global" if depth == 0 else "fine", "stage_rank": index,
        "prior_prob": 0.5, "complexity": complexity, "fast_confidence": None,
        "posterior_score": 0.5, "is_evaluated": False,
        "answering_confidence": None,
    } for index, (key, box, depth, parent, children, complexity) in enumerate(candidates)]
    source_image_key = source_key(image)
    refs = tuple(SimpleNamespace(
        canonical_key=key, bbox_original=box, depth=depth, render_level=0,
        tree_scope="main", crop_origin=(0, 0), source_image_key=source_image_key,
    ) for key, box, depth, _, _, _ in candidates)
    sink({"ordered_nodes": refs}, {
        "schema_version": 1, "event": "tree_ready", "tree_scope": "main",
        "crop_origin": [0, 0], "source_image_identity": {
            "mode": image.mode, "size": [image.width, image.height],
            "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        },
        "search_call_ordinal": 1, "visual_cue": "sign", "stage": "Full Tree",
        "depth": 0, "candidate_count": 3, "candidates": snapshots,
        "ordered_keys": [item[0] for item in candidates], "popped_keys": [],
        "selected_keys": [], "remaining_keys": [item[0] for item in candidates],
    })
    annotation["targets"] = ["sign"]
    annotation["root_ans_conf"] = 0.0
    annotation["search_mode"] = 2
    annotation["num_pop"] = [1]
    return 0


class PerformPDFSearchTest(unittest.TestCase):
    def test_runner_version_identifies_conservative_cross_backbone_route(self):
        self.assertEqual(
            RUNNER_VERSION, "pdf-faithful-v16-independent-full-image-route",
        )

    def test_family_ranking_route_freezes_qwen_and_protects_target_backbones(self):
        route = getattr(pdf_runner, "family_ranking_route", None)
        self.assertIsNotNone(route)
        ranking = __import__(
            "cvsearch.evidence_gap.pdf_types",
            fromlist=["PDFSearchConfig"],
        ).PDFSearchConfig.from_mapping(full_config()).ranking

        self.assertEqual(route("qwen", ranking), {
            "alpha": 0.6,
            "beta": 0.5,
            "visual_lambda": 0.5,
            "protected_head": None,
        })
        for family in ("internvl", "llava"):
            with self.subTest(family=family):
                self.assertEqual(route(family, ranking), {
                    "alpha": 0.1,
                    "beta": 0.7,
                    "visual_lambda": 0.7,
                    "protected_head": 3,
                })

    def test_family_verification_route_freezes_qwen_and_requires_full_image_agreement(self):
        route = getattr(pdf_runner, "family_verification_route", None)
        self.assertIsNotNone(route)
        self.assertEqual(route("qwen"), {
            "mode": "paired_local_v1",
            "min_confidence": None,
            "min_proposal_frequency": None,
        })
        for family in ("internvl", "llava"):
            with self.subTest(family=family):
                self.assertEqual(route(family), {
                    "mode": "candidate_independent_full_image_v1",
                    "min_confidence": 0.9,
                    "min_proposal_frequency": 0.4,
                })

    def test_full_image_route_requires_exact_output_and_frozen_confidence(self):
        agrees = getattr(pdf_runner, "independent_full_image_agrees", None)
        self.assertIsNotNone(agrees)
        self.assertTrue(agrees(
            proposal_output=["A", "B", "C", "D"],
            proposal_frequency=2.0 / 3.0,
            independent_output=["A", "B", "C", "D"],
            independent_confidence=0.9,
            aggregation_available=True,
        ))
        self.assertFalse(agrees(
            proposal_output=["A", "B", "C", "D"],
            proposal_frequency=1.0,
            independent_output=["A", "B", "D", "D"],
            independent_confidence=1.0,
            aggregation_available=True,
        ))
        self.assertFalse(agrees(
            proposal_output=1,
            proposal_frequency=1.0,
            independent_output=1,
            independent_confidence=0.899,
            aggregation_available=True,
        ))
        self.assertFalse(agrees(
            proposal_output=1,
            proposal_frequency=0.399,
            independent_output=1,
            independent_confidence=1.0,
            aggregation_available=True,
        ))

    def test_relation_change_requires_zoom_or_explicit_context(self):
        class RelationGenerator(FakeGenerator):
            def generate_text_only(self, prompt):
                return json.dumps({
                    "augmented_queries": [
                        "sign position", "door position", "sign door relation",
                        "sign relative to door",
                    ],
                    "evidence_items": [{
                        "kind": "relation_context", "targets": ["sign", "door"],
                    }],
                    "global_scope_required": False,
                })

            def multiple_choices_with_losses(self, image, question, options, nodes):
                if nodes and not getattr(nodes[0], "is_root", False):
                    return 1, [0.9, 0.1]
                return 0, [0.1, 0.9]

        class PairVerifier(FakeVerifier):
            def multiple_choices_with_losses(self, image, question, options, nodes):
                return 0, [0.1, 1.1]

        def baseline_zero(**kwargs):
            fake_cvsearch(**kwargs)
            return 0

        mapping = full_config()
        mapping["budget"]["max_steps"] = 1
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new("RGB", (8, 8), "white").save(folder / "image.png")
            response, trace = run_pdf_sample(
                original_annotation={
                    "question": "Is the sign left or right of the door?",
                    "options": ["right", "left"],
                    "answer_type": "logits_match", "input_image": "image.png",
                },
                image_folder=folder, ic_examples={},
                config=__import__("cvsearch.evidence_gap.pdf_types", fromlist=["PDFSearchConfig"]).PDFSearchConfig.from_mapping(mapping),
                sam_model=object(), generator_model=RelationGenerator(),
                verifier_model=PairVerifier(), nlp_model=object(),
                clip_scorer=FakeClip(), cvsearch_fn=baseline_zero,
                generator_checkpoint_sha256="a" * 64,
                verifier_checkpoint_sha256="b" * 64,
            )

        self.assertEqual(response, 0)
        paired = trace["final_decision"]["paired_reference"]
        self.assertTrue(paired["geometry"]["relation_context_required"])
        self.assertFalse(paired["geometry"]["relation_enriched"])
        self.assertFalse(paired["attempted"])
        self.assertEqual(paired["reason"], "proposal_lacks_relation_enrichment")
        audit_pdf_trace(trace, require_operational=True)

    def test_absolute_image_location_rejects_a_confident_wrong_region(self):
        class RegionGenerator(FakeGenerator):
            def multiple_choices_with_losses(self, image, question, options, nodes):
                if nodes and not getattr(nodes[0], "is_root", False):
                    return 1, [0.9, 0.1]
                return 0, [0.1, 0.9]

        class WrongRegionVerifier(FakeVerifier):
            def get_confidence_value(self, nodes, image, confidence_type, input_ele):
                return 0.8 if "Proposed answer: Sony" in input_ele else -0.8

        mapping = full_config()
        mapping["budget"]["max_steps"] = 1
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new("RGB", (8, 8), "white").save(folder / "image.png")
            response, trace = run_pdf_sample(
                original_annotation={
                    "question": "Which brand is on the device at the left of the image?",
                    "options": ["Insignia", "Sony"],
                    "answer_type": "logits_match", "input_image": "image.png",
                },
                image_folder=folder, ic_examples={},
                config=__import__("cvsearch.evidence_gap.pdf_types", fromlist=["PDFSearchConfig"]).PDFSearchConfig.from_mapping(mapping),
                sam_model=object(), generator_model=RegionGenerator(),
                verifier_model=WrongRegionVerifier(), nlp_model=object(),
                clip_scorer=FakeClip(), cvsearch_fn=fake_cvsearch,
                generator_checkpoint_sha256="a" * 64,
                verifier_checkpoint_sha256="b" * 64,
            )

        self.assertEqual(response, 0)
        paired = trace["final_decision"]["paired_reference"]
        self.assertFalse(paired["geometry"]["eligible"])
        self.assertEqual(paired["geometry"]["constraints"], ["left"])
        self.assertFalse(paired["attempted"])
        self.assertEqual(paired["reason"], "proposal_outside_question_region")
        self.assertEqual(trace["final_decision"]["source"], "cvsearch_safety_fallback")
        audit_pdf_trace(trace, require_operational=True)
        forged = copy.deepcopy(trace)
        forged["final_decision"]["paired_reference"]["geometry"]["eligible"] = True
        with self.assertRaisesRegex(ValueError, "geometry eligibility"):
            audit_pdf_trace(forged, require_operational=True)

    def test_forced_return_rejects_single_region_history_answer(self):
        class HistoryGenerator(FakeGenerator):
            def multiple_choices_with_losses(self, image, question, options, nodes):
                if nodes and not getattr(nodes[0], "is_root", False):
                    return 0, [0.1, 0.9]
                return 1, [0.9, 0.1]

        class ViewRelativeVerifier(FakeVerifier):
            def get_confidence_value(self, nodes, image, confidence_type, input_ele):
                is_child_sheet = image.height > image.width
                if "Proposed answer: left" in input_ele:
                    return 0.2 if is_child_sheet else -0.8
                if "Proposed answer: right" in input_ele:
                    return -0.8 if is_child_sheet else 0.8
                return 0.2

        def baseline_right(**kwargs):
            fake_cvsearch(**kwargs)
            return 1

        mapping = full_config()
        mapping["budget"]["max_steps"] = 1
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new("RGB", (8, 8), "white").save(folder / "image.png")
            response, trace = run_pdf_sample(
                original_annotation={
                    "question": "Which sign?", "options": ["left", "right"],
                    "answer_type": "logits_match", "input_image": "image.png",
                },
                image_folder=folder, ic_examples={},
                config=__import__("cvsearch.evidence_gap.pdf_types", fromlist=["PDFSearchConfig"]).PDFSearchConfig.from_mapping(mapping),
                sam_model=object(), generator_model=HistoryGenerator(),
                verifier_model=ViewRelativeVerifier(), nlp_model=object(),
                clip_scorer=FakeClip(), cvsearch_fn=baseline_right,
                generator_checkpoint_sha256="a" * 64,
                verifier_checkpoint_sha256="b" * 64,
            )

        self.assertEqual(response, 1)
        self.assertEqual(trace["controller"]["selected_history_state_id"], 0)
        self.assertEqual(trace["final_decision"]["source"], "cvsearch_safety_fallback")
        paired = trace["final_decision"]["paired_reference"]
        self.assertEqual(paired["state_id"], 1)
        self.assertTrue(paired["state_support_floor_met"])
        self.assertTrue(paired["history_spatial_consensus_required"])
        self.assertEqual(paired["history_spatial_support_count"], 1)
        self.assertFalse(paired["history_spatial_consensus_met"])
        self.assertFalse(paired["attempted"])
        self.assertFalse(paired["selected"])
        self.assertEqual(paired["reason"], "proposal_lacks_spatial_consensus")
        audit_pdf_trace(trace, require_operational=True)

    def test_history_spatial_support_counts_distinct_focus_nodes_only(self):
        records = [
            {
                "state": {"path_keys": ["root", "left"]},
                "answer": {"output": 0, "aggregation_available": True, "frequency": 1.0},
                "support": {"independent": True, "fallback_used": False, "support_min": 0.8},
            },
            {
                "state": {"path_keys": ["root", "left"]},
                "answer": {"output": 0, "aggregation_available": True, "frequency": 1.0},
                "support": {"independent": True, "fallback_used": False, "support_min": 0.9},
            },
            {
                "state": {"path_keys": ["root", "right"]},
                "answer": {"output": 0, "aggregation_available": True, "frequency": 2.0 / 3.0},
                "support": {"independent": True, "fallback_used": False, "support_min": 0.7},
            },
            {
                "state": {"path_keys": ["root", "other"]},
                "answer": {"output": 1, "aggregation_available": True, "frequency": 1.0},
                "support": {"independent": True, "fallback_used": False, "support_min": 0.9},
            },
        ]
        self.assertEqual(_history_spatial_support_count(records, 0, min_support=0.5), 2)

    def test_history_spatial_support_rejects_nodes_below_verifier_floor(self):
        records = [
            {
                "state": {"path_keys": ["root", "left"]},
                "answer": {"output": 0, "aggregation_available": True, "frequency": 1.0},
                "support": {"independent": True, "fallback_used": False, "support_min": 0.8},
            },
            {
                "state": {"path_keys": ["root", "right"]},
                "answer": {"output": 0, "aggregation_available": True, "frequency": 1.0},
                "support": {"independent": True, "fallback_used": False, "support_min": 0.49},
            },
        ]
        self.assertEqual(_history_spatial_support_count(records, 0, min_support=0.5), 1)

    def test_hr_semantic_equivalence_keeps_the_consistent_projection(self):
        class HRGenerator(FakeGenerator):
            def free_form_using_nodes(self, image, question, nodes):
                if "four keys zoom, split, expand, next" in question:
                    return super().free_form_using_nodes(image, question, nodes)
                return "A"

        class LowVerifier(FakeVerifier):
            def get_confidence_value(self, nodes, image, confidence_type, input_ele):
                return -1.0

        baseline = ["A", "A", "A", "B"]

        def inconsistent_same_semantics(**kwargs):
            fake_cvsearch(**kwargs)
            return baseline

        block = "A. left\nB. right\nC. up\nD. down\n"
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new("RGB", (8, 8), "white").save(folder / "image.png")
            response, trace = run_pdf_sample(
                original_annotation={
                    "question": "Where is the sign?", "options": [block] * 4,
                    "answer_type": "option_list", "input_image": "image.png",
                },
                image_folder=folder, ic_examples={},
                config=__import__("cvsearch.evidence_gap.pdf_types", fromlist=["PDFSearchConfig"]).PDFSearchConfig.from_mapping(full_config()),
                sam_model=object(), generator_model=HRGenerator(),
                verifier_model=LowVerifier(), nlp_model=object(),
                clip_scorer=FakeClip(), cvsearch_fn=inconsistent_same_semantics,
                generator_checkpoint_sha256="a" * 64,
                verifier_checkpoint_sha256="b" * 64,
            )

        self.assertEqual(response, ["A"] * 4)
        paired = trace["final_decision"]["paired_reference"]
        self.assertFalse(paired["required"])
        self.assertFalse(paired["attempted"])
        self.assertEqual(paired["reason"], "semantic_answers_match")
        self.assertEqual(trace["final_decision"]["source"], "controller")
        self.assertEqual(trace["final_decision"]["reason"], "semantic_answers_match")
        audit_pdf_trace(trace, require_operational=True)

    def test_hr_answer_change_is_paired_by_semantic_answer(self):
        class HRGenerator(FakeGenerator):
            def free_form_using_nodes(self, image, question, nodes):
                if "four keys zoom, split, expand, next" in question:
                    return super().free_form_using_nodes(image, question, nodes)
                return "A"

        def baseline_right(**kwargs):
            fake_cvsearch(**kwargs)
            return ["B"] * 4

        block = "A. left\nB. right\nC. up\nD. down\n"
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new("RGB", (8, 8), "white").save(folder / "image.png")
            response, trace = run_pdf_sample(
                original_annotation={
                    "question": "Where is the sign?", "options": [block] * 4,
                    "answer_type": "option_list", "input_image": "image.png",
                },
                image_folder=folder, ic_examples={},
                config=__import__("cvsearch.evidence_gap.pdf_types", fromlist=["PDFSearchConfig"]).PDFSearchConfig.from_mapping(full_config()),
                sam_model=object(), generator_model=HRGenerator(),
                verifier_model=FakeVerifier(), nlp_model=object(),
                clip_scorer=FakeClip(), cvsearch_fn=baseline_right,
                generator_checkpoint_sha256="a" * 64,
                verifier_checkpoint_sha256="b" * 64,
            )

        self.assertEqual(response, ["B"] * 4)
        paired = trace["final_decision"]["paired_reference"]
        self.assertTrue(paired["attempted"])
        self.assertEqual(paired["proposed"]["support_avg"], paired["reference"]["support_avg"])
        self.assertEqual(trace["final_decision"]["source"], "cvsearch_safety_fallback")
        audit_pdf_trace(trace, require_operational=True)

    def test_paired_veto_can_fallback_after_controller_certification(self):
        def baseline_right(**kwargs):
            fake_cvsearch(**kwargs)
            return 1

        mapping = full_config()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new("RGB", (8, 8), "white").save(folder / "image.png")
            response, trace = run_pdf_sample(
                original_annotation={
                    "question": "Which sign?", "options": ["left", "right"],
                    "answer_type": "logits_match", "input_image": "image.png",
                },
                image_folder=folder, ic_examples={},
                config=__import__("cvsearch.evidence_gap.pdf_types", fromlist=["PDFSearchConfig"]).PDFSearchConfig.from_mapping(mapping),
                sam_model=object(), generator_model=FakeGenerator(),
                verifier_model=FakeVerifier(), nlp_model=object(),
                clip_scorer=FakeClip(), cvsearch_fn=baseline_right,
                generator_checkpoint_sha256="a" * 64,
                verifier_checkpoint_sha256="b" * 64,
            )

        self.assertEqual(response, 1)
        self.assertEqual(trace["controller"]["termination"], "CERTIFIED_STOP")
        self.assertEqual(trace["final_decision"]["source"], "cvsearch_safety_fallback")
        self.assertEqual(
            trace["final_decision"]["paired_reference"]["reason"],
            "paired_support_rejected",
        )
        audit_pdf_trace(trace, require_operational=True)

    def test_cross_backbone_change_requires_candidate_independent_full_image_agreement(self):
        class FullImageVerifier(FakeVerifier):
            def multiple_choices_with_losses(self, image, question, options, nodes):
                return 0, [0.0, 10.0]

        def anchor_one(**kwargs):
            fake_cvsearch(**kwargs)
            return 1

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new("RGB", (8, 8), "white").save(folder / "image.png")
            response, trace = run_pdf_sample(
                original_annotation={
                    "question": "Which sign?", "options": ["left", "right"],
                    "answer_type": "logits_match", "input_image": "image.png",
                },
                image_folder=folder, ic_examples={},
                config=__import__(
                    "cvsearch.evidence_gap.pdf_types",
                    fromlist=["PDFSearchConfig"],
                ).PDFSearchConfig.from_mapping(full_config()),
                sam_model=object(), generator_model=FakeGenerator(),
                verifier_model=FullImageVerifier(), nlp_model=object(),
                clip_scorer=FakeClip(), cvsearch_fn=anchor_one,
                generator_checkpoint_sha256="a" * 64,
                verifier_checkpoint_sha256="b" * 64,
                generator_family="internvl",
            )

        self.assertEqual(response, 0)
        self.assertEqual(trace["final_decision"]["source"], "independent_full_image")
        paired = trace["final_decision"]["paired_reference"]
        self.assertFalse(paired["attempted"])
        self.assertEqual(
            paired["reason"], "routed_to_candidate_independent_full_image",
        )
        check = trace["final_decision"]["independent_full_image"]
        self.assertTrue(check["attempted"])
        self.assertTrue(check["selected"])
        self.assertEqual(check["independent_output_sha256"], canonical_sha256(0))
        self.assertEqual(
            trace["module_activity"]["verifier"][
                "candidate_independent_full_image_calls"
            ],
            3,
        )
        audit_pdf_trace(trace, require_operational=True)

    def test_answer_change_requires_paired_reference_support_on_same_view(self):
        class ContrastiveVerifier(FakeVerifier):
            def multiple_choices_with_losses(self, image, question, options, nodes):
                self.pair_view_size = image.size
                return 0, [0.1, 1.1]

            def get_confidence_value(self, nodes, image, confidence_type, input_ele):
                return 0.0

        def baseline_right(**kwargs):
            fake_cvsearch(**kwargs)
            return 1

        mapping = full_config()
        mapping["budget"]["max_steps"] = 1
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new("RGB", (8, 8), "white").save(folder / "image.png")
            response, trace = run_pdf_sample(
                original_annotation={
                    "question": "Which sign?", "options": ["left", "right"],
                    "answer_type": "logits_match", "input_image": "image.png",
                },
                image_folder=folder, ic_examples={},
                config=__import__("cvsearch.evidence_gap.pdf_types", fromlist=["PDFSearchConfig"]).PDFSearchConfig.from_mapping(mapping),
                sam_model=object(), generator_model=FakeGenerator(),
                verifier_model=ContrastiveVerifier(), nlp_model=object(),
                clip_scorer=FakeClip(), cvsearch_fn=baseline_right,
                generator_checkpoint_sha256="a" * 64,
                verifier_checkpoint_sha256="b" * 64,
            )

        self.assertEqual(response, 0)
        paired = trace["final_decision"]["paired_reference"]
        self.assertTrue(paired["required"])
        self.assertTrue(paired["selected"])
        self.assertGreater(paired["avg_delta"], 0.05)
        self.assertEqual(
            paired["comparison_mode"],
            "worst_case_3x_conditional_option_loss",
        )
        self.assertEqual(paired["extra_model_calls"], 9)
        self.assertNotEqual(paired["proposed"], trace["state_evaluations"][0]["support"])
        self.assertEqual(trace["final_decision"]["source"], "controller_paired_reference")
        audit_pdf_trace(trace, require_operational=True)
        forged = copy.deepcopy(trace)
        forged["final_decision"]["paired_reference"]["avg_delta"] += 0.1
        with self.assertRaisesRegex(ValueError, "paired average delta"):
            audit_pdf_trace(forged, require_operational=True)

    def test_answer_change_rejects_weak_contrastive_margin(self):
        class WeakContrastiveVerifier(FakeVerifier):
            def multiple_choices_with_losses(self, image, question, options, nodes):
                return 0, [0.0, 0.16]

        def baseline_right(**kwargs):
            fake_cvsearch(**kwargs)
            return 1

        mapping = full_config()
        mapping["budget"]["max_steps"] = 1
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new("RGB", (8, 8), "white").save(folder / "image.png")
            response, trace = run_pdf_sample(
                original_annotation={
                    "question": "Which sign?", "options": ["left", "right"],
                    "answer_type": "logits_match", "input_image": "image.png",
                },
                image_folder=folder, ic_examples={},
                config=__import__("cvsearch.evidence_gap.pdf_types", fromlist=["PDFSearchConfig"]).PDFSearchConfig.from_mapping(mapping),
                sam_model=object(), generator_model=FakeGenerator(),
                verifier_model=WeakContrastiveVerifier(), nlp_model=object(),
                clip_scorer=FakeClip(), cvsearch_fn=baseline_right,
                generator_checkpoint_sha256="a" * 64,
                verifier_checkpoint_sha256="b" * 64,
            )

        self.assertEqual(response, 1)
        paired = trace["final_decision"]["paired_reference"]
        self.assertTrue(paired["attempted"])
        self.assertEqual(paired["min_avg_delta"], 0.1)
        self.assertGreater(paired["avg_delta"], 0.05)
        self.assertLess(paired["avg_delta"], 0.1)
        self.assertFalse(paired["selected"])
        self.assertEqual(paired["reason"], "paired_support_rejected")
        audit_pdf_trace(trace, require_operational=True)

    def test_target_detail_change_requires_non_root_localization(self):
        class RootZoomGenerator(FakeGenerator):
            def free_form_using_nodes(self, image, question, nodes):
                return '{"zoom":0.9,"split":0.1,"expand":0.1,"next":0.1}'

        class ContrastiveVerifier(FakeVerifier):
            def multiple_choices_with_losses(self, image, question, options, nodes):
                return 0, [0.1, 1.1]

        def baseline_right(**kwargs):
            fake_cvsearch(**kwargs)
            return 1

        mapping = full_config()
        mapping["budget"]["max_steps"] = 1
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new("RGB", (8, 8), "white").save(folder / "image.png")
            response, trace = run_pdf_sample(
                original_annotation={
                    "question": "What color is the sign?",
                    "options": ["blue", "red"],
                    "answer_type": "logits_match", "input_image": "image.png",
                },
                image_folder=folder, ic_examples={},
                config=__import__("cvsearch.evidence_gap.pdf_types", fromlist=["PDFSearchConfig"]).PDFSearchConfig.from_mapping(mapping),
                sam_model=object(), generator_model=RootZoomGenerator(),
                verifier_model=ContrastiveVerifier(), nlp_model=object(),
                clip_scorer=FakeClip(), cvsearch_fn=baseline_right,
                generator_checkpoint_sha256="a" * 64,
                verifier_checkpoint_sha256="b" * 64,
            )

        self.assertEqual(response, 1)
        paired = trace["final_decision"]["paired_reference"]
        self.assertFalse(paired["geometry"]["detail_localized"])
        self.assertFalse(paired["attempted"])
        self.assertEqual(paired["reason"], "proposal_lacks_detail_localization")
        audit_pdf_trace(trace, require_operational=True)

    def test_relation_overview_remains_eligible_with_auxiliary_target_details(self):
        class MixedRelationGenerator(FakeGenerator):
            def generate_text_only(self, prompt):
                return json.dumps({
                    "augmented_queries": [
                        "sign position", "door position", "sign door relation",
                        "sign relative to door",
                    ],
                    "evidence_items": [
                        {
                            "kind": "target_detail", "target": "sign",
                            "requirements": ["presence", "visual_detail"],
                        },
                        {"kind": "relation_context", "targets": ["sign", "door"]},
                    ],
                    "global_scope_required": False,
                })

            def free_form_using_nodes(self, image, question, nodes):
                return '{"zoom":0.9,"split":0.1,"expand":0.1,"next":0.1}'

        class ContrastiveVerifier(FakeVerifier):
            def multiple_choices_with_losses(self, image, question, options, nodes):
                return 0, [0.1, 1.1]

        def baseline_right(**kwargs):
            fake_cvsearch(**kwargs)
            return 1

        mapping = full_config()
        mapping["budget"]["max_steps"] = 1
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new("RGB", (8, 8), "white").save(folder / "image.png")
            response, trace = run_pdf_sample(
                original_annotation={
                    "question": "Is the sign left or right of the door?",
                    "options": ["left", "right"],
                    "answer_type": "logits_match", "input_image": "image.png",
                },
                image_folder=folder, ic_examples={},
                config=__import__("cvsearch.evidence_gap.pdf_types", fromlist=["PDFSearchConfig"]).PDFSearchConfig.from_mapping(mapping),
                sam_model=object(), generator_model=MixedRelationGenerator(),
                verifier_model=ContrastiveVerifier(), nlp_model=object(),
                clip_scorer=FakeClip(), cvsearch_fn=baseline_right,
                generator_checkpoint_sha256="a" * 64,
                verifier_checkpoint_sha256="b" * 64,
            )

        self.assertEqual(response, 0)
        geometry = trace["final_decision"]["paired_reference"]["geometry"]
        self.assertTrue(geometry["relation_context_required"])
        self.assertFalse(geometry["detail_localization_required"])
        self.assertTrue(geometry["eligible"])
        audit_pdf_trace(trace, require_operational=True)

    def test_answer_change_rejects_contrastive_win_without_state_evidence_floor(self):
        class LowEvidenceContrastiveVerifier(FakeVerifier):
            def multiple_choices_with_losses(self, image, question, options, nodes):
                return 0, [0.1, 1.1]

            def get_confidence_value(self, nodes, image, confidence_type, input_ele):
                return -0.8

        def baseline_right(**kwargs):
            fake_cvsearch(**kwargs)
            return 1

        mapping = full_config()
        mapping["budget"]["max_steps"] = 1
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new("RGB", (8, 8), "white").save(folder / "image.png")
            response, trace = run_pdf_sample(
                original_annotation={
                    "question": "Which sign?", "options": ["left", "right"],
                    "answer_type": "logits_match", "input_image": "image.png",
                },
                image_folder=folder, ic_examples={},
                config=__import__("cvsearch.evidence_gap.pdf_types", fromlist=["PDFSearchConfig"]).PDFSearchConfig.from_mapping(mapping),
                sam_model=object(), generator_model=FakeGenerator(),
                verifier_model=LowEvidenceContrastiveVerifier(), nlp_model=object(),
                clip_scorer=FakeClip(), cvsearch_fn=baseline_right,
                generator_checkpoint_sha256="a" * 64,
                verifier_checkpoint_sha256="b" * 64,
            )

        self.assertEqual(response, 1)
        paired = trace["final_decision"]["paired_reference"]
        self.assertTrue(paired["required"])
        self.assertFalse(paired["attempted"])
        self.assertFalse(paired["state_support_floor_met"])
        self.assertEqual(paired["state_support_floor"], 0.5)
        self.assertEqual(paired["reason"], "proposal_below_state_support_floor")
        self.assertEqual(trace["final_decision"]["source"], "cvsearch_safety_fallback")
        audit_pdf_trace(trace, require_operational=True)
        forged = copy.deepcopy(trace)
        forged["final_decision"]["paired_reference"]["state_support_floor_met"] = True
        with self.assertRaisesRegex(ValueError, "support floor decision"):
            audit_pdf_trace(forged, require_operational=True)

    def test_unverified_forced_return_uses_explicit_cvsearch_safety_fallback(self):
        class LowVerifier(FakeVerifier):
            def get_confidence_value(self, nodes, image, confidence_type, input_ele):
                return -1.0

        def anchor_one(**kwargs):
            fake_cvsearch(**kwargs)
            return 1

        mapping = full_config()
        mapping["budget"]["max_steps"] = 1
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new("RGB", (8, 8), "white").save(folder / "image.png")
            response, trace = run_pdf_sample(
                original_annotation={
                    "question": "Which sign?", "options": ["left", "right"],
                    "answer_type": "logits_match", "input_image": "image.png",
                },
                image_folder=folder, ic_examples={},
                config=__import__("cvsearch.evidence_gap.pdf_types", fromlist=["PDFSearchConfig"]).PDFSearchConfig.from_mapping(mapping),
                sam_model=object(), generator_model=FakeGenerator(),
                verifier_model=LowVerifier(), nlp_model=object(),
                clip_scorer=FakeClip(), cvsearch_fn=anchor_one,
                generator_checkpoint_sha256="a" * 64,
                verifier_checkpoint_sha256="b" * 64,
            )
        self.assertEqual(response, 1)
        self.assertEqual(trace["controller"]["termination"], "FORCED_RETURN")
        self.assertEqual(trace["final_decision"]["source"], "cvsearch_safety_fallback")

    def test_initial_assessment_over_budget_returns_exact_cvsearch_with_audited_trace(self):
        def anchor_one(**kwargs):
            fake_cvsearch(**kwargs)
            return 1

        mapping = full_config()
        mapping["budget"]["max_processed_pixels"] = 1
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new("RGB", (8, 8), "white").save(folder / "image.png")
            response, trace = run_pdf_sample(
                original_annotation={
                    "question": "Which sign?", "options": ["left", "right"],
                    "answer_type": "logits_match", "input_image": "image.png",
                },
                image_folder=folder, ic_examples={},
                config=__import__(
                    "cvsearch.evidence_gap.pdf_types",
                    fromlist=["PDFSearchConfig"],
                ).PDFSearchConfig.from_mapping(mapping),
                sam_model=object(), generator_model=FakeGenerator(),
                verifier_model=FakeVerifier(), nlp_model=object(),
                clip_scorer=FakeClip(), cvsearch_fn=anchor_one,
                generator_checkpoint_sha256="a" * 64,
                verifier_checkpoint_sha256="b" * 64,
            )

        self.assertEqual(response, 1)
        self.assertEqual(trace["controller"]["termination"], "BUDGET_FALLBACK")
        self.assertEqual(trace["final_decision"]["source"], "cvsearch_safety_fallback")
        self.assertEqual(
            trace["final_decision"]["reason"],
            "initial_assessment_exceeds_budget",
        )
        self.assertGreater(
            trace["controller"]["budget_fallback"]["estimated_processed_pixels"],
            mapping["budget"]["max_processed_pixels"],
        )
        report = audit_pdf_trace(trace)
        self.assertTrue(report["safety_fallback"])
        self.assertEqual(report["termination"], "BUDGET_FALLBACK")
        with self.assertRaisesRegex(ValueError, "uncertainty re-answering"):
            audit_pdf_trace(trace, require_operational=True)

    def test_internvl_quick_p0_survives_forced_tree_materialization(self):
        calls = []

        def quick_then_tree(**kwargs):
            calls.append(kwargs["fast_threshold"])
            if kwargs["fast_threshold"] == 2.0:
                fake_cvsearch(**kwargs)
                return 0
            annotation = kwargs["annotation"]
            annotation["targets"] = None
            annotation["root_ans_conf"] = 0.9
            annotation["search_mode"] = 0
            annotation["num_pop"] = []
            return 1

        mapping = full_config()
        mapping["budget"]["max_processed_pixels"] = 1
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new("RGB", (8, 8), "white").save(folder / "image.png")
            response, trace = run_pdf_sample(
                original_annotation={
                    "question": "Which sign?", "options": ["left", "right"],
                    "answer_type": "logits_match", "input_image": "image.png",
                },
                image_folder=folder, ic_examples={},
                config=__import__(
                    "cvsearch.evidence_gap.pdf_types",
                    fromlist=["PDFSearchConfig"],
                ).PDFSearchConfig.from_mapping(mapping),
                sam_model=object(), generator_model=FakeGenerator(),
                verifier_model=FakeVerifier(), nlp_model=object(),
                clip_scorer=FakeClip(), cvsearch_fn=quick_then_tree,
                generator_checkpoint_sha256="a" * 64,
                verifier_checkpoint_sha256="b" * 64,
                generator_family="internvl",
            )

        self.assertEqual(calls, [0.6, 2.0])
        self.assertEqual(response, 1)
        self.assertEqual(
            trace["candidate_factory"]["native_output_sha256"],
            canonical_sha256(1),
        )
        self.assertEqual(trace["candidate_factory"]["fast_threshold"], 2.0)
        self.assertTrue(trace["candidate_factory"]["second_call_used"])
        self.assertIn("ranking_route", trace)
        self.assertEqual(trace["ranking_route"], {
            "alpha": 0.1,
            "beta": 0.7,
            "visual_lambda": 0.7,
            "protected_head": 3,
        })
        audit_pdf_trace(trace)
        forged = copy.deepcopy(trace)
        forged["ranking_route"]["alpha"] = 0.2
        with self.assertRaisesRegex(ValueError, "ranking route"):
            audit_pdf_trace(forged)

    def test_parser_requires_independent_verifier_and_strict_config(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args([
            "--root-path", "/", "--generator-model-path", "/generator",
            "--verifier-model-path", "/verifier", "--annotation-path", "/data",
            "--sam-model-path", "/sam.pt", "--nlp-model-path", "/spacy",
            "--clip-model-path", "/clip", "--benchmark", "vstar",
            "--answers-file", "/tmp/out.jsonl", "--config", "/tmp/config.json",
        ])
        self.assertEqual(args.verifier_model_path, "/verifier")

    def test_candidate_factory_forces_tree_collection_but_keeps_family_thresholds(self):
        llava = family_candidate_kwargs("llava")
        intern = family_candidate_kwargs("internvl")
        qwen = family_candidate_kwargs("qwen")
        try:
            native_llava = family_candidate_kwargs("llava", force_tree=False)
            native_intern = family_candidate_kwargs("internvl", force_tree=False)
            native_qwen = family_candidate_kwargs("qwen", force_tree=False)
        except TypeError as error:
            self.fail(str(error))
        self.assertEqual(llava["fast_threshold"], 2.0)
        self.assertEqual(native_llava["fast_threshold"], 0.8)
        self.assertEqual(native_intern["fast_threshold"], 0.6)
        self.assertEqual(native_qwen["fast_threshold"], 0.8)
        self.assertEqual(intern["answering_confidence_threshold_lower"], -0.2)
        self.assertEqual(qwen["answering_confidence_threshold_upper"], 0.9)
        self.assertEqual(llava["pop_limit"](3), 9)

    def test_one_sample_runs_root_to_ranked_leaf_without_label_access(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            Image.new("RGB", (8, 8), "white").save(folder / "image.png")
            original = {
                "question": "Which sign is visible?", "options": ["left", "right"],
                "answer_type": "logits_match", "input_image": "image.png",
                "target_object": "secret", "bbox": [1, 2, 3, 4],
            }
            before = copy.deepcopy(original)
            response, trace = run_pdf_sample(
                original_annotation=original,
                image_folder=folder,
                ic_examples={},
                config=__import__("cvsearch.evidence_gap.pdf_types", fromlist=["PDFSearchConfig"]).PDFSearchConfig.from_mapping(full_config()),
                sam_model=object(), generator_model=FakeGenerator(),
                verifier_model=FakeVerifier(), nlp_model=object(),
                clip_scorer=FakeClip(), cvsearch_fn=fake_cvsearch,
                generator_checkpoint_sha256="a" * 64,
                verifier_checkpoint_sha256="b" * 64,
            )
        self.assertEqual(original, before)
        self.assertEqual(response, 0)
        self.assertEqual(trace["controller"]["termination"], "CERTIFIED_STOP")
        self.assertEqual(trace["controller"]["steps"][0]["action"], "SPLIT")
        self.assertEqual(trace["module_activity"]["joint_ranking"]["candidate_count"], 2)
        self.assertNotIn("target_object", json.dumps(trace))
        self.assertNotIn("secret", json.dumps(trace))
        self.assertEqual(len(trace["state_evaluations"]), 2)


if __name__ == "__main__":
    unittest.main()
