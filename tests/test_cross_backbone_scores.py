import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cvsearch.eval.cross_backbone_scores import (
    build_scores,
    build_paper_vstar_reproduction,
    extract_same_run_cvsearch_control_rows,
    parse_hr_choice,
    score_rows,
    score_vstar_letter_rows,
)


class CrossBackboneScoreTests(unittest.TestCase):
    def test_vstar_overall_is_weighted_over_attribute_and_spatial_rows(self):
        annotations = [
            {"input_image": "a", "question": "q1", "options": ["x", "y"],
             "answer_type": "logits_match", "test_type": "direct_attributes"},
            {"input_image": "b", "question": "q2", "options": ["x", "y"],
             "answer_type": "logits_match", "test_type": "direct_attributes"},
            {"input_image": "c", "question": "q3", "options": ["x", "y"],
             "answer_type": "logits_match", "test_type": "relative_position"},
        ]
        predictions = [dict(row, output=output) for row, output in zip(
            annotations, (0, 1, 0),
        )]
        score = score_rows("vstar", annotations, predictions)
        self.assertEqual(score["attribute"]["correct"], 1)
        self.assertEqual(score["attribute"]["accuracy"], 50.0)
        self.assertEqual(score["spatial"]["accuracy"], 100.0)
        self.assertAlmostEqual(score["overall"]["accuracy"], 200 / 3)

    def test_hr_scores_all_four_cycles_and_matches_official_parser(self):
        annotations = [
            {"input_image": "a", "question": "q", "options": ["o"] * 4,
             "answer_type": "option_list", "answer": ["A", "B", "C", "D"],
             "category": "single"},
            {"input_image": "b", "question": "q", "options": ["o"] * 4,
             "answer_type": "option_list", "answer": ["A", "B", "C", "D"],
             "category": "cross"},
        ]
        predictions = [
            dict(annotations[0], output=["A", "answer B", "xC", "wrong"]),
            dict(annotations[1], output=["A", "B", "wrong", "wrong"]),
        ]
        score = score_rows("hr-bench_4k", annotations, predictions)
        self.assertEqual(parse_hr_choice("answer B"), "B")
        self.assertEqual(score["fsp"]["correct"], 3)
        self.assertEqual(score["fcp"]["correct"], 2)
        self.assertEqual(score["overall"]["accuracy"], 62.5)

    def test_identity_or_cardinality_mismatch_fails_closed(self):
        annotation = {
            "input_image": "a", "question": "q", "options": ["x"],
            "answer_type": "logits_match", "test_type": "direct_attributes",
        }
        with self.assertRaises(ValueError):
            score_rows("vstar", [annotation], [])
        with self.assertRaises(ValueError):
            score_rows("vstar", [annotation], [dict(annotation, question="changed", output=0)])

    def test_llava_paper_letter_protocol_scores_updated_labels(self):
        annotations = [
            {"input_image": "a", "question": "q1", "options": ["x", "y"],
             "answer_type": "free_form", "test_type": "direct_attributes",
             "label": "C", "text": "q1 choices"},
            {"input_image": "b", "question": "q2", "options": ["x", "y"],
             "answer_type": "free_form", "test_type": "relative_position",
             "label": "A", "text": "q2 choices"},
        ]
        predictions = [
            dict(annotations[0], output="The answer is C."),
            dict(annotations[1], output="B"),
        ]
        score = score_vstar_letter_rows(annotations, predictions)
        self.assertEqual(score["attribute"]["accuracy"], 100.0)
        self.assertEqual(score["spatial"]["accuracy"], 0.0)
        self.assertEqual(score["overall"]["accuracy"], 50.0)

    def test_paper_protocol_block_scores_both_local_vectors(self):
        annotations = [
            {"input_image": "a", "question": "q1", "options": ["x", "y"],
             "answer_type": "free_form", "test_type": "direct_attributes",
             "label": "C", "text": "q1 choices"},
            {"input_image": "b", "question": "q2", "options": ["x", "y"],
             "answer_type": "free_form", "test_type": "relative_position",
             "label": "A", "text": "q2 choices"},
        ]
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            annotation_path = root / "annotation_vstar_updated.json"
            annotation_path.write_text(json.dumps(annotations))
            for filename, outputs in (
                ("direct_answer_paper.jsonl", ("C", "B")),
                ("cvsearch_paper.jsonl", ("C", "A")),
            ):
                rows = [dict(row, output=output) for row, output in zip(
                    annotations, outputs,
                )]
                (root / filename).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )

            result = build_paper_vstar_reproduction(root, annotation_path)

        self.assertEqual(result["protocol"], "paper_letter")
        self.assertEqual(result["methods"]["direct"]["metrics"]["overall"]["accuracy"], 50.0)
        self.assertEqual(result["methods"]["cvsearch"]["metrics"]["overall"]["accuracy"], 100.0)
        self.assertEqual(result["local_delta"]["overall"], 50.0)

    def test_cross_backbone_verdict_uses_same_run_disabled_control(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            annotation_root = root / "annotations"
            result_root = root / "results"
            for benchmark in ("vstar", "hr-bench_4k", "hr-bench_8k"):
                directory = annotation_root / benchmark
                directory.mkdir(parents=True)
                if benchmark == "vstar":
                    annotations = [{
                        "input_image": image, "question": "q", "options": ["x", "y"],
                        "answer_type": "logits_match", "test_type": test_type,
                    } for image, test_type in (
                        ("a", "direct_attributes"), ("b", "relative_position"),
                    )]
                else:
                    annotations = [{
                        "input_image": image, "question": "q", "options": [["x"]] * 4,
                        "answer_type": "option_list", "answer": ["A"] * 4,
                        "category": category,
                    } for image, category in (("a", "single"), ("b", "cross"))]
                (directory / f"annotation_{benchmark}.json").write_text(
                    json.dumps(annotations)
                )
                if benchmark == "vstar":
                    updated = [
                        dict(annotation, answer_type="free_form", label="A", text="q")
                        for annotation in annotations
                    ]
                    (directory / "annotation_vstar_updated.json").write_text(
                        json.dumps(updated)
                    )
                for model in ("llava", "internvl"):
                    output_directory = result_root / model / benchmark
                    (output_directory / "logiv").mkdir(parents=True)
                    correct = 0 if benchmark == "vstar" else ["A"] * 4
                    wrong = 1 if benchmark == "vstar" else ["B"] * 4
                    for relative, output in (
                        ("direct_answer.jsonl", wrong),
                        ("cvsearch.jsonl", correct),
                        ("logiv/disabled.jsonl", wrong),
                        ("logiv_v2.jsonl", correct),
                    ):
                        records = []
                        for annotation in annotations:
                            record = dict(annotation, output=output)
                            if relative == "logiv/disabled.jsonl":
                                if benchmark == "vstar":
                                    record["method_trace"] = {
                                        "steps": [{
                                            "action": "FORCED_RETURN",
                                            "feasible_actions": [],
                                        }],
                                        "history": [{"answer": {
                                            "output": correct,
                                            "selected_from": "search",
                                        }}],
                                    }
                                else:
                                    record["method_trace"] = {
                                        "steps": [{
                                            "action": "FORCED_RETURN",
                                            "feasible_actions": [],
                                        }],
                                        "anchor_answer": {
                                            "output": correct,
                                            "selected_from": "cvsearch_anchor",
                                        },
                                    }
                            records.append(record)
                        (output_directory / relative).write_text(
                            "".join(
                                json.dumps(record) + "\n" for record in records
                            )
                        )
                    if model == "llava" and benchmark == "vstar":
                        for filename in ("direct_answer_paper.jsonl", "cvsearch_paper.jsonl"):
                            (output_directory / filename).write_text(
                                "".join(
                                    json.dumps(dict(annotation, output="A")) + "\n"
                                    for annotation in updated
                                )
                            )

            scores = build_scores(result_root, annotation_root)

        vstar = scores["models"]["llava"]["vstar"]
        self.assertEqual(
            vstar["methods"]["logiv_disabled_control"]["metrics"]["overall"]["accuracy"],
            0.0,
        )
        self.assertEqual(
            vstar["methods"]["same_run_cvsearch_control"]["metrics"]["overall"]["accuracy"],
            100.0,
        )
        self.assertEqual(
            vstar["local_paired_deltas"]["logiv_v2_minus_disabled_control"]["overall"],
            100.0,
        )
        self.assertEqual(vstar["paper_reference_comparison"]["protocol"], "paper_letter")
        self.assertEqual(
            scores["models"]["internvl"]["vstar"]["paper_reference_comparison"]["protocol"],
            "common_logits",
        )
        self.assertEqual(
            scores["verdict"]["classification"],
            "no_positive_effect_vs_original_cvsearch",
        )
        self.assertEqual(
            scores["disabled_control_audit"]["classification"],
            "positive_on_all_six_disabled_control_pairs",
        )

    def test_extracts_same_run_cvsearch_before_root_fallback(self):
        vstar = {
            "output": 1,
            "method_trace": {"steps": [{
                "action": "FORCED_RETURN", "feasible_actions": [],
            }], "history": [
                {"answer": {"output": 1, "selected_from": "root"}},
                {"answer": {"output": 0, "selected_from": "search"}},
            ]},
        }
        hr = {
            "output": ["A"] * 4,
            "method_trace": {
                "steps": [{
                    "action": "FORCED_RETURN", "feasible_actions": [],
                }],
                "anchor_answer": {
                    "output": ["B"] * 4,
                    "selected_from": "cvsearch_anchor",
                },
            },
        }
        self.assertEqual(
            extract_same_run_cvsearch_control_rows("vstar", [vstar])[0]["output"],
            0,
        )
        self.assertEqual(
            extract_same_run_cvsearch_control_rows("hr-bench_4k", [hr])[0]["output"],
            ["B"] * 4,
        )


if __name__ == "__main__":
    unittest.main()
