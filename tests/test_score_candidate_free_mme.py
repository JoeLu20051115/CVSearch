import unittest

from cvsearch.eval.score_candidate_free_mme import score_external_gate


class CandidateFreeMMEScoreTest(unittest.TestCase):
    def test_external_gate_requires_each_backbone_nonnegative_and_pooled_positive(self):
        report = score_external_gate({
            "qwen": {
                "delta": 0, "corrections": 0, "corruptions": 0,
            },
            "internvl": {
                "delta": 1, "corrections": 1, "corruptions": 0,
            },
        }, provenance_ok=True)

        self.assertEqual(report["pooled_delta"], 1)
        self.assertEqual(report["failures"], [])
        self.assertTrue(report["passed"])

    def test_external_gate_rejects_regression_or_failed_provenance(self):
        report = score_external_gate({
            "qwen": {
                "delta": -1, "corrections": 0, "corruptions": 1,
            },
            "internvl": {
                "delta": 1, "corrections": 1, "corruptions": 0,
            },
        }, provenance_ok=False)

        self.assertIn("qwen backbone regressed", report["failures"])
        self.assertIn("corrections do not exceed corruptions", report["failures"])
        self.assertIn("provenance audit failed", report["failures"])
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
