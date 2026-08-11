import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from PIL import Image

from cvsearch.eval.pdf_trace_audit import audit_pdf_trace
from cvsearch.perform_PDFSearch import (
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
    left_key = candidate_key((0, 0, 4, 8), 1)
    right_key = candidate_key((4, 0, 4, 8), 1)
    candidates = [
        (root_key, (0, 0, 8, 8), 0, None, [left_key, right_key], 0.2),
        (left_key, (0, 0, 4, 8), 1, root_key, [], 0.3),
        (right_key, (4, 0, 4, 8), 1, root_key, [], 0.9),
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

    def test_forced_return_can_rescue_a_low_absolute_support_history_answer(self):
        class HistoryGenerator(FakeGenerator):
            def multiple_choices_with_losses(self, image, question, options, nodes):
                if nodes and not getattr(nodes[0], "is_root", False):
                    return 0, [0.1, 0.9]
                return 1, [0.9, 0.1]

        class ViewRelativeVerifier(FakeVerifier):
            def get_confidence_value(self, nodes, image, confidence_type, input_ele):
                is_child_sheet = image.height > image.width
                if "Proposed answer: left" in input_ele:
                    return -0.2 if is_child_sheet else -0.8
                if "Proposed answer: right" in input_ele:
                    return -0.8 if is_child_sheet else 0.8
                return -1.0

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

        self.assertEqual(response, 0)
        self.assertEqual(trace["controller"]["selected_history_state_id"], 0)
        self.assertEqual(trace["final_decision"]["source"], "history_paired_reference")
        paired = trace["final_decision"]["paired_reference"]
        self.assertEqual(paired["state_id"], 1)
        self.assertTrue(paired["selected"])
        self.assertGreater(paired["avg_delta"], 0.05)
        audit_pdf_trace(trace, require_operational=True)

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
        self.assertEqual(paired["comparison_mode"], "conditional_option_loss")
        self.assertEqual(paired["extra_model_calls"], 3)
        self.assertNotEqual(paired["proposed"], trace["state_evaluations"][0]["support"])
        self.assertEqual(trace["final_decision"]["source"], "controller_paired_reference")
        audit_pdf_trace(trace, require_operational=True)
        forged = copy.deepcopy(trace)
        forged["final_decision"]["paired_reference"]["avg_delta"] += 0.1
        with self.assertRaisesRegex(ValueError, "paired average delta"):
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
        self.assertEqual(llava["fast_threshold"], 2.0)
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
