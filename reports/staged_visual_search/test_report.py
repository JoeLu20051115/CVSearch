from pathlib import Path
import unittest

from reports.staged_visual_search.build_report import collect_evidence


REPO_ROOT = Path(__file__).resolve().parents[2]


class EvidenceTest(unittest.TestCase):
    def test_frozen_qwen_results(self):
        evidence = collect_evidence(REPO_ROOT)
        self.assertEqual(
            evidence["stage1_dev"],
            {
                "baseline_r3": 0.8571428571428571,
                "selected_r3": 0.9523809523809523,
                "baseline_acc": 0.8,
                "selected_acc": 0.8333333333333334,
            },
        )
        self.assertEqual(evidence["stage1_v6_qwen"]["r3_delta"], 0.0)
        self.assertEqual(evidence["stage2_qwen"]["vstar"], (17, 18))
        self.assertEqual(evidence["stage2_qwen"]["treebench"], (5, 6))
        self.assertEqual(evidence["accepted_development_qwen_delta"], 7)
        self.assertEqual(evidence["mme_qwen"]["delta"], 27)


if __name__ == "__main__":
    unittest.main()
