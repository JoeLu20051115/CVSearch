import unittest

from cvsearch.evidence_gap.answers import aggregate_hr_answers
from cvsearch.evidence_gap.fusion import soft_fuse_hr
from cvsearch.evidence_gap.types import AnswerRecord


BLOCKS = [
    "A. cat\nB. dog\nC. bird\nD. fish\n",
    "A. dog\nB. cat\nC. fish\nD. bird\n",
    "A. bird\nB. fish\nC. cat\nD. dog\n",
    "A. fish\nB. bird\nC. dog\nD. cat\n",
]
RAW = ["A", "A", "A", "A"]


class SoftFusionTest(unittest.TestCase):
    def test_zero_gamma_is_exact_raw_and_ties_keep_raw(self):
        evidence = aggregate_hr_answers(BLOCKS, ["A", "B", "C", "D"])
        self.assertEqual(soft_fuse_hr(BLOCKS, RAW, evidence, 0.0).output, RAW)

    def test_one_global_weight_can_correct_only_supported_shuffle_slots(self):
        evidence = aggregate_hr_answers(BLOCKS, ["A", "B", "C", "D"])
        result = soft_fuse_hr(BLOCKS, RAW, evidence, 2.1)
        self.assertEqual(result.canonical_answer, "cat")
        self.assertEqual(result.output, ["A", "B", "C", "D"])
        self.assertEqual(result.selected_from, "unified_soft_fusion")

    def test_nonraw_leader_tie_preserves_exact_raw_output(self):
        blocks = ["A. bird\nB. cat\nC. dog\n"] * 4
        raw = ["A", "A", "A", "A"]
        evidence = aggregate_hr_answers(blocks, ["B", "C", "B", "C"])
        self.assertEqual(soft_fuse_hr(blocks, raw, evidence, 3.0).output, raw)

    def test_mixed_case_semantics_receive_shared_global_support(self):
        blocks = [
            "A. bird\nB. Cat\nC. dog\n",
            "A. dog\nB. bird\nC. cAt\n",
            "A. CAT\nB. dog\nC. bird\n",
            "A. bird\nB. cat\nC. dog\n",
        ]
        evidence = aggregate_hr_answers(blocks, ["B", "C", "A", "B"])
        result = soft_fuse_hr(blocks, ["A", "A", "A", "A"], evidence, 2.1)
        self.assertEqual(result.canonical_answer, "cat")
        self.assertEqual(result.output, ["B", "C", "A", "B"])

    def test_unavailable_aggregation_returns_exact_raw(self):
        evidence = AnswerRecord(aggregation_available=False)
        self.assertEqual(soft_fuse_hr(BLOCKS, RAW, evidence, 4.1).output, RAW)

    def test_rejects_negative_nonfinite_bool_or_mismatched_inputs(self):
        evidence = aggregate_hr_answers(BLOCKS, ["A", "B", "C", "D"])
        for gamma in (-0.1, float("nan"), float("inf"), True):
            with self.subTest(gamma=gamma), self.assertRaises((TypeError, ValueError)):
                soft_fuse_hr(BLOCKS, RAW, evidence, gamma)


if __name__ == "__main__":
    unittest.main()
