from types import SimpleNamespace
import unittest

from cvsearch.eval.phase7_confirmation_runner import eligible_confirmation_actions


class Phase7ConfirmationRunnerTest(unittest.TestCase):
    def test_eligible_actions_are_changed_feasible_and_not_already_v2_admitted(self):
        pair = SimpleNamespace(
            p0={
                "action": "P0",
                "output": "A",
                "p0_stability": {"confidence": 0.50},
            },
            candidates=(
                {
                    "action": "EXPAND",
                    "feasible": True,
                    "output": "B",
                    "candidate_stability": {"confidence": 0.50},
                },
                {
                    "action": "EXPAND",
                    "feasible": True,
                    "output": "C",
                    "candidate_stability": {"confidence": 0.75},
                },
                {
                    "action": "ZOOM",
                    "feasible": True,
                    "output": "A",
                    "candidate_stability": {"confidence": 1.00},
                },
                {
                    "action": "ZOOM",
                    "feasible": False,
                    "output": None,
                    "candidate_stability": None,
                },
            ),
        )

        self.assertEqual(
            eligible_confirmation_actions(pair),
            (pair.candidates[0],),
        )


if __name__ == "__main__":
    unittest.main()
