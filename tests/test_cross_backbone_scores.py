import unittest

from cvsearch.eval.cross_backbone_scores import (
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


if __name__ == "__main__":
    unittest.main()
