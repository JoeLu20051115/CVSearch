import unittest

from cvsearch.eval.cross_backbone_phase15 import select_phase15_output
from cvsearch.eval.phase15_vstar_confirmed_backtrack import B8_FROZEN_B5_RULE
from cvsearch.eval.phase12_generated_query_lazy_runner import (
    generate_localization_queries,
)


BLOCKS = [
    "A. Blue\nB. Red\nC. Green\nD. Black\n",
    "A. Green\nB. Blue\nC. Black\nD. Red\n",
    "A. Green\nB. Black\nC. Blue\nD. Red\n",
    "A. Green\nB. Black\nC. Red\nD. Blue\n",
]


def p0(output, confidence=0.2):
    return {
        "action": "P0",
        "output": output,
        "p0_stability": {"confidence": confidence},
    }


def candidate(action, output=None, confidence=None):
    return {
        "action": action,
        "feasible": output is not None,
        "output": output,
        "candidate_stability": (
            {"confidence": confidence} if output is not None else None
        ),
    }


def observation(winner):
    losses = [3.0, 3.0, 3.0]
    losses[winner] = 1.0
    return {"winner": winner, "losses": losses}


class _TextOnlyModel:
    def __init__(self):
        self.prompts = []

    def generate_text_only(self, prompt):
        self.prompts.append(prompt)
        return "LOC: red object\nLOC: green area\nLOC: left edge\nLOC: lower corner"


class CrossBackbonePhase15Tests(unittest.TestCase):
    def test_phase6_candidate_is_retained_without_phase15_fallback(self):
        decision = select_phase15_output(
            "vstar", p0(0),
            [candidate("EXPAND", 2, 0.6), candidate("ZOOM")],
            ["red", "green", "blue"],
        )
        self.assertEqual(decision.action, "EXPAND")
        self.assertEqual(decision.output, 2)
        self.assertEqual(decision.phase6_action, "EXPAND")

    def test_hr_p0_uses_frozen_semantic_normalization(self):
        decision = select_phase15_output(
            "hr-bench_4k", p0(["A", "A", "C", "D"], 0.75),
            [candidate("EXPAND"), candidate("ZOOM")], BLOCKS,
        )
        self.assertEqual(decision.action, "HR_NORMALIZE")
        self.assertEqual(decision.output, ["A", "B", "C", "D"])
        self.assertEqual(decision.phase6_action, "P0")

    def test_vstar_p0_uses_frozen_confirmed_backtrack(self):
        lazy_record = {
            "decisions": {
                B8_FROZEN_B5_RULE: {
                    "action": "LAZY",
                    "status": "selected_generated_query_lazy_search",
                    "output": 2,
                    "confidence": 0.5000000000000001,
                    "confidence_gain": 0.3000000000000001,
                    "path_consensus": 2 / 3,
                },
            },
            "observations": [observation(1), observation(2), observation(2)],
        }
        decision = select_phase15_output(
            "vstar", p0(1),
            [candidate("EXPAND"), candidate("ZOOM")],
            ["red", "green", "blue"], lazy_record,
        )
        self.assertEqual(decision.action, "VSTAR_BACKTRACK")
        self.assertEqual(decision.output, 2)
        self.assertEqual(decision.phase6_action, "P0")

    def test_text_only_generation_uses_model_neutral_adapter(self):
        model = _TextOnlyModel()
        result = generate_localization_queries(
            model, "Where are the red object, green area, left edge, and lower corner?",
        )
        self.assertEqual(
            result["queries"],
            ["red object", "green area", "left edge", "lower corner"],
        )
        self.assertEqual(len(model.prompts), 1)


if __name__ == "__main__":
    unittest.main()
