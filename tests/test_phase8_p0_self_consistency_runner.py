from types import SimpleNamespace
import unittest


class Phase8P0SelfConsistencyRunnerTest(unittest.TestCase):
    def _module(self):
        from cvsearch.eval import phase8_p0_self_consistency_runner

        return phase8_p0_self_consistency_runner

    @staticmethod
    def _pair(expand_confidence=0.4):
        return SimpleNamespace(
            p0={
                "action": "P0",
                "output": 0,
                "p0_stability": {"confidence": 0.5},
            },
            candidates=(
                {
                    "action": "ZOOM",
                    "feasible": False,
                    "output": None,
                    "candidate_stability": None,
                },
                {
                    "action": "EXPAND",
                    "feasible": True,
                    "output": 1,
                    "candidate_stability": {"confidence": expand_confidence},
                },
            ),
        )

    @staticmethod
    def _row(answer_type="logits_match", boxes=None):
        return {
            "answer_type": answer_type,
            "method_trace": {
                "final_boxes": [(1, 2, 3, 4)] if boxes is None else boxes,
            },
        }

    def test_eligibility_requires_v2_p0_logits_contract_and_final_boxes(self):
        module = self._module()
        self.assertTrue(module.is_eligible_p0_pair(self._pair(), self._row()))
        self.assertFalse(
            module.is_eligible_p0_pair(
                self._pair(expand_confidence=0.75), self._row(),
            )
        )
        self.assertFalse(
            module.is_eligible_p0_pair(
                self._pair(), self._row(answer_type="option_list"),
            )
        )
        self.assertFalse(
            module.is_eligible_p0_pair(self._pair(), self._row(boxes=[]))
        )

    def test_view_dto_binds_projection_hashes_and_fails_closed(self):
        module = self._module()
        projection = {
            "feasible": True,
            "output": 2,
            "confidence": 0.6,
            "majority_fraction": 2 / 3,
            "aggregate_output": 2,
            "majority_output": 2,
        }
        dto = module.view_dto(
            projection, view_sha256="1" * 64,
            prompt_sha256=["3" * 64, "4" * 64, "5" * 64],
        )
        self.assertEqual(dto["output"], 2)
        self.assertEqual(dto["stability"], {"confidence": 0.6})
        self.assertEqual(dto["majority_fraction"], 2 / 3)

        projection["feasible"] = False
        failed = module.view_dto(
            projection, view_sha256="2" * 64,
            prompt_sha256=["3" * 64, "4" * 64, "5" * 64],
        )
        self.assertEqual(
            {key: failed[key] for key in ("output", "stability", "majority_fraction")},
            {"output": None, "stability": None, "majority_fraction": None},
        )


if __name__ == "__main__":
    unittest.main()
