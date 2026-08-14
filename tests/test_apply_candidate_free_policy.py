import copy
import unittest

from cvsearch.eval.apply_candidate_free_policy import runtime_evidence_from_record


class ApplyCandidateFreePolicyTest(unittest.TestCase):
    def test_runtime_evidence_uses_only_proposal_projection_and_cost(self):
        record = {
            "proposal": {
                "feasible": True,
                "candidate_output": "E raw",
                "candidate_canonical": "E",
                "agreement": 0.4,
            },
            "projection": {
                "feasible": True,
                "canonical_answer": "E",
                "confidence": 0.95,
            },
            "cost": {"observations": 10},
            "answer": "A",
            "category": "hidden",
        }
        changed = copy.deepcopy(record)
        changed["answer"] = "E"
        changed["category"] = "different"

        evidence = runtime_evidence_from_record(record)

        self.assertEqual(evidence, runtime_evidence_from_record(changed))
        self.assertEqual(evidence.proposal_output, "E raw")
        self.assertEqual(evidence.verifier_canonical, "E")

    def test_malformed_runtime_evidence_fails_closed_before_selection(self):
        with self.assertRaises(ValueError):
            runtime_evidence_from_record({
                "proposal": {"feasible": True},
                "projection": None,
                "cost": {"observations": 8},
            })


if __name__ == "__main__":
    unittest.main()
