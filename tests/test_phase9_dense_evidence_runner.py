from types import SimpleNamespace
import unittest


class Phase9DenseEvidenceRunnerTest(unittest.TestCase):
    def _module(self):
        from cvsearch.eval import phase9_dense_evidence_runner

        return phase9_dense_evidence_runner

    @staticmethod
    def _pair(expand_confidence=0.4):
        return SimpleNamespace(
            p0={
                "action": "P0", "output": 0,
                "p0_stability": {"confidence": 0.5},
            },
            candidates=(
                {"action": "ZOOM", "feasible": False, "output": None,
                 "candidate_stability": None},
                {"action": "EXPAND", "feasible": True, "output": 1,
                 "candidate_stability": {"confidence": expand_confidence}},
            ),
        )

    def test_eligibility_only_supplements_v2_p0_supported_answer_contracts(self):
        module = self._module()
        self.assertTrue(module.is_eligible_dense_pair(
            self._pair(), {"answer_type": "logits_match"},
        ))
        self.assertTrue(module.is_eligible_dense_pair(
            self._pair(), {"answer_type": "option_list"},
        ))
        self.assertFalse(module.is_eligible_dense_pair(
            self._pair(expand_confidence=0.75), {"answer_type": "logits_match"},
        ))
        self.assertFalse(module.is_eligible_dense_pair(
            self._pair(), {"answer_type": "free_form"},
        ))

    def test_candidate_dto_and_rule_keys_are_exact_and_fail_closed(self):
        module = self._module()
        projection = {
            "feasible": True,
            "output": 2,
            "confidence": 2 / 3,
            "canonical_answer": 2,
            "tile_majority_fraction": 2 / 3,
            "tile_outputs": [2, 2, 1],
        }
        candidate = module.candidate_dto(
            projection,
            sheet_sha256=["1" * 64, "2" * 64, "3" * 64],
            rank_sha256="4" * 64,
        )
        self.assertEqual(candidate["action"], "DENSE")
        self.assertEqual(candidate["output"], 2)
        self.assertEqual(candidate["stability"], {"confidence": 2 / 3})

        projection["feasible"] = False
        failed = module.candidate_dto(
            projection,
            sheet_sha256=["1" * 64, "2" * 64, "3" * 64],
            rank_sha256="4" * 64,
        )
        self.assertIsNone(failed["output"])
        self.assertIsNone(failed["stability"])

        self.assertEqual(module.rule_key(2 / 3, 0.1), "c0.6666666666666666-g0.1")
        with self.assertRaisesRegex(ValueError, "frozen"):
            module.rule_key(0.6, 0.1)


if __name__ == "__main__":
    unittest.main()
