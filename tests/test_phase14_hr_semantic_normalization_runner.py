from types import SimpleNamespace
import unittest

from cvsearch.eval.phase14_hr_semantic_normalization_runner import _produce_record
from tests.test_phase14_hr_semantic_normalization import BLOCKS, p0


def candidate(action, output, confidence):
    return {
        "action": action,
        "feasible": True,
        "output": output,
        "candidate_stability": {"confidence": confidence},
    }


class HRSemanticNormalizationRunnerTests(unittest.TestCase):
    def test_applies_normalization_only_after_v2_retains_p0(self):
        pair = SimpleNamespace(
            ordinal=7,
            p0=p0(["A", "A", "C", "D"]),
            candidates=(
                candidate("ZOOM", ["B", "D", "D", "C"], 0.5),
                candidate("EXPAND", ["C", "A", "A", "A"], 0.6),
            ),
            input_identity={"ordinal": 7},
            extracted_digest="1" * 64,
        )
        record = _produce_record(pair, {"options": BLOCKS})
        self.assertEqual(record["v2_decision"]["action"], "P0")
        self.assertEqual(record["decision"]["action"], "HR_NORMALIZE")
        self.assertEqual(record["decision"]["output"], ["A", "B", "C", "D"])
        self.assertTrue(record["projection"]["feasible"])

    def test_existing_v2_action_is_not_replaced_or_normalized(self):
        pair = SimpleNamespace(
            ordinal=9,
            p0={
                "action": "P0", "output": ["A", "A", "C", "D"],
                "p0_stability": {"confidence": 0.2},
            },
            candidates=(
                candidate("ZOOM", ["B", "D", "D", "C"], 0.3),
                candidate("EXPAND", ["C", "A", "A", "A"], 0.5),
            ),
            input_identity={"ordinal": 9},
            extracted_digest="2" * 64,
        )
        record = _produce_record(pair, {"options": BLOCKS})
        self.assertEqual(record["v2_decision"]["action"], "EXPAND")
        self.assertEqual(record["decision"], record["v2_decision"])
        self.assertIsNone(record["projection"])


if __name__ == "__main__":
    unittest.main()
