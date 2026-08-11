import unittest

from cvsearch.eval.pdf_scores import compare_pdf_to_cvsearch


class PDFScoresTest(unittest.TestCase):
    def test_vstar_reports_net_corrections_and_regressions_by_ordinal(self):
        baseline = [
            {"input_image": "a", "options": ["x", "y"], "output": 1},
            {"input_image": "b", "options": ["x", "y"], "output": 0},
            {"input_image": "c", "options": ["x", "y"], "output": 1},
        ]
        proposed = [
            {"_eg_ordinal": 0, "input_image": "a", "options": ["x", "y"], "output": 0},
            {"_eg_ordinal": 1, "input_image": "b", "options": ["x", "y"], "output": 1},
        ]
        report = compare_pdf_to_cvsearch("vstar", proposed, baseline)
        self.assertEqual(report["units"], 2)
        self.assertEqual(report["corrections"], 1)
        self.assertEqual(report["regressions"], 1)
        self.assertEqual(report["net_corrections"], 0)
        self.assertEqual(report["cvsearch_accuracy"], 0.5)
        self.assertEqual(report["pdf_accuracy"], 0.5)

    def test_hr_scores_four_cycles_and_requires_aligned_identity(self):
        options = ["A. red\nB. blue"] * 4
        baseline = [{
            "input_image": "a", "options": options,
            "answer": ["A", "A", "A", "A"],
            "output": ["A", "B", "A", "B"],
        }]
        proposed = [{
            "_eg_ordinal": 0, "input_image": "a", "options": options,
            "answer": ["A", "A", "A", "A"],
            "output": ["A", "A", "A", "B"],
        }]
        report = compare_pdf_to_cvsearch("hr-bench_4k", proposed, baseline)
        self.assertEqual(report["units"], 4)
        self.assertEqual(report["corrections"], 1)
        self.assertEqual(report["regressions"], 0)
        self.assertEqual(report["cvsearch_accuracy"], 0.5)
        self.assertEqual(report["pdf_accuracy"], 0.75)
        proposed[0]["input_image"] = "wrong"
        with self.assertRaises(ValueError):
            compare_pdf_to_cvsearch("hr-bench_4k", proposed, baseline)


if __name__ == "__main__":
    unittest.main()
