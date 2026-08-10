import unittest

from cvsearch.eval.phase13_hr_semantic_option_loss import (
    UnprojectableSemanticOptionSet,
    all_hr_loss_rule_keys,
    build_semantic_option_set,
    project_hr_loss_candidate,
    select_hr_loss_candidate,
)


BLOCKS = [
    "A. Blue\nB. Red\nC. Green\nD. Black\n",
    "A. Green\nB. Blue\nC. Black\nD. Red\n",
    "A. Green\nB. Black\nC. Blue\nD. Red\n",
    "A. Green\nB. Black\nC. Red\nD. Blue\n",
]

DUPLICATE_BLOCKS = [
    "A. Brown\nB. Red\nC. red\nD. Blue\n",
    "A. Blue\nB. Brown\nC. red\nD. Red\n",
    "A. Blue\nB. red\nC. Brown\nD. Red\n",
    "A. Red\nB. red\nC. Blue\nD. Brown\n",
]


def p0(output=None):
    return {
        "action": "P0", "output": output or ["A", "B", "C", "D"],
        "p0_stability": {"confidence": 0.5},
    }


def candidate(confidence=0.3, path=1.0):
    return {
        "action": "HR_LOSS", "feasible": True,
        "output": ["B", "D", "D", "C"],
        "stability": {"confidence": confidence},
        "path_consensus": path,
        "view_sha256": ["1" * 64, "2" * 64],
        "b5_record_sha256": "3" * 64,
        "option_set_sha256": "4" * 64,
    }


class HRSemanticOptionLossTests(unittest.TestCase):
    def test_reconstructs_one_shared_semantic_set_and_reverse_letters(self):
        material = build_semantic_option_set(BLOCKS)
        self.assertEqual(material["choices"], ["Blue", "Red", "Green", "Black"])
        self.assertEqual(material["letters_by_choice"][1], ["B", "D", "D", "C"])
        contaminated = list(BLOCKS)
        contaminated[3] = "A. Green\nB. White\nC. Red\nD. Blue\n"
        with self.assertRaisesRegex(ValueError, "same semantic choices"):
            build_semantic_option_set(contaminated)

    def test_casefold_duplicate_is_explicitly_unprojectable(self):
        with self.assertRaisesRegex(
            UnprojectableSemanticOptionSet, "duplicate semantic choices",
        ):
            build_semantic_option_set(DUPLICATE_BLOCKS)

    def test_two_view_agreement_and_backtrack_majority_map_to_letters(self):
        material = build_semantic_option_set(BLOCKS)
        agree = [
            {"winner": 1, "losses": [2.0, 1.0, 3.0, 4.0]},
            {"winner": 1, "losses": [3.0, 1.0, 2.0, 4.0]},
        ]
        projection = project_hr_loss_candidate(material, agree)
        self.assertTrue(projection["feasible"])
        self.assertEqual(projection["output"], ["B", "D", "D", "C"])
        self.assertEqual(projection["path_consensus"], 1.0)

        majority = [
            {"winner": 0, "losses": [1.0, 3.0, 4.0, 5.0]},
            {"winner": 1, "losses": [3.0, 1.0, 4.0, 5.0]},
            {"winner": 0, "losses": [1.0, 2.0, 4.0, 5.0]},
        ]
        projection = project_hr_loss_candidate(material, majority)
        self.assertEqual(projection["output"], ["A", "B", "C", "D"])
        self.assertEqual(projection["path_consensus"], 2 / 3)

    def test_eight_rule_grid_requires_path_and_absolute_loss_margin(self):
        self.assertEqual(len(all_hr_loss_rule_keys()), 8)
        selected = select_hr_loss_candidate(
            p0(), candidate(), min_path_consensus=1.0,
            min_confidence=0.25,
        )
        self.assertEqual(selected.action, "HR_LOSS")
        retained = select_hr_loss_candidate(
            p0(), candidate(path=2 / 3), min_path_consensus=1.0,
            min_confidence=0.25,
        )
        self.assertEqual(retained.action, "P0")


if __name__ == "__main__":
    unittest.main()
