import unittest

from cvsearch.evidence_gap.input import sanitize_annotation, split_bucket


class AnnotationSanitizationTest(unittest.TestCase):
    def test_vstar_truth_is_not_policy_visible(self):
        raw = {"question": "q", "options": ["a"], "answer_type": "logits_match",
               "input_image": "i.jpg", "bbox": [[1, 2, 3, 4]], "target_object": ["comb"]}
        self.assertEqual(sanitize_annotation(raw), {
            "question": "q", "options": ["a"], "answer_type": "logits_match", "input_image": "i.jpg"})

    def test_hr_truth_and_metadata_are_not_policy_visible(self):
        raw = {"question": "q", "options": ["A. x"], "answer_type": "option_list",
               "input_image": "0.jpg", "answer": ["A"], "category": "single", "index": 0}
        self.assertEqual(set(sanitize_annotation(raw)), {"question", "options", "answer_type", "input_image"})

    def test_missing_policy_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "input_image"):
            sanitize_annotation({"question": "q", "options": [], "answer_type": "option_list"})


class SourceGroupedSplitTest(unittest.TestCase):
    def test_known_sources_have_stable_buckets(self):
        self.assertEqual(split_bucket("vstar", "direct_attributes/sa_7429.jpg"), "holdout")
        self.assertEqual(split_bucket("hr-bench_4k", "image/2.jpg"), "holdout")

    def test_explicit_seed_changes_the_known_source_bucket(self):
        self.assertEqual(split_bucket("hr-bench_4k", "image/2.jpg", 7), "dev")

    def test_hr_cycles_for_one_source_share_a_bucket(self):
        cycles = [
            {"question": f"cycle {index}", "options": ["A. x"], "input_image": "image/4.jpg"}
            for index in range(4)
        ]
        buckets = {
            split_bucket("hr-bench_4k", annotation["input_image"])
            for annotation in cycles
        }
        self.assertEqual(len(buckets), 1)


if __name__ == "__main__":
    unittest.main()
