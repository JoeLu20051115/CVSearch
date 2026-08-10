import math
import unittest


class Phase7UncertaintyConfirmationTest(unittest.TestCase):
    def _module(self):
        from cvsearch.eval import phase7_uncertainty_confirmation

        return phase7_uncertainty_confirmation

    @staticmethod
    def _p0(output=0, confidence=0.2):
        return {
            "action": "P0",
            "output": output,
            "p0_stability": {"confidence": confidence},
        }

    @staticmethod
    def _candidate(action="ZOOM", output=1, confidence=0.1):
        return {
            "action": action,
            "feasible": True,
            "output": output,
            "stability": {"confidence": confidence},
            "view_sha256": "1" * 64,
        }

    @staticmethod
    def _confirmation(action="ZOOM", output=1, confidence=0.5):
        return {
            "action": action,
            "feasible": True,
            "output": output,
            "stability": {"confidence": confidence},
            "aggregate_stability": {"confidence": confidence},
            "view_sha256": "2" * 64,
            "prompt_sha256": ["3" * 64, "4" * 64, "5" * 64],
        }

    def test_search_boundary_is_inclusive_and_below_boundary_retains_p0(self):
        module = self._module()
        accepted = module.confirm_search_candidate(
            self._p0(confidence=0.1),
            self._candidate(action="SEARCH", confidence=0.85),
        )
        self.assertEqual(accepted.action, "SEARCH")
        self.assertEqual(accepted.status, "selected_deferred_search")
        self.assertAlmostEqual(accepted.stability_gain, 0.75)

        retained = module.confirm_search_candidate(
            self._p0(confidence=0.1),
            self._candidate(action="SEARCH", confidence=math.nextafter(0.85, 0.0)),
        )
        self.assertEqual(retained.action, "P0")
        self.assertEqual(retained.status, "retained_p0")

    def test_action_confirmation_requires_distinct_agreeing_views(self):
        module = self._module()
        selected = module.confirm_action_candidate(
            self._p0(), self._candidate(), self._confirmation(), threshold=0.5,
        )
        self.assertEqual(selected.action, "ZOOM")
        self.assertEqual(selected.output, 1)
        self.assertEqual(selected.status, "selected_confirmed_action")
        self.assertAlmostEqual(selected.stability_gain, 0.3)

        disagreeing = self._confirmation(output=2)
        retained = module.confirm_action_candidate(
            self._p0(), self._candidate(), disagreeing, threshold=0.5,
        )
        self.assertEqual(retained.action, "P0")
        self.assertEqual(retained.status, "retained_p0")

        same_view = self._confirmation()
        same_view["view_sha256"] = self._candidate()["view_sha256"]
        retained = module.confirm_action_candidate(
            self._p0(), self._candidate(), same_view, threshold=0.5,
        )
        self.assertEqual(retained.action, "P0")
        self.assertEqual(retained.status, "retained_p0")

    def test_action_confirmation_fails_closed_on_threshold_or_nonpositive_gain(self):
        module = self._module()
        below = self._confirmation(confidence=math.nextafter(0.5, 0.0))
        retained = module.confirm_action_candidate(
            self._p0(), self._candidate(), below, threshold=0.5,
        )
        self.assertEqual(retained.action, "P0")

        retained = module.confirm_action_candidate(
            self._p0(confidence=0.5), self._candidate(),
            self._confirmation(confidence=0.5), threshold=0.5,
        )
        self.assertEqual(retained.action, "P0")

    def test_exact_dtos_reject_extra_forbidden_and_nonfinite_material(self):
        module = self._module()
        candidate = self._candidate()
        candidate["ordinal"] = 17
        with self.assertRaisesRegex(ValueError, "exact schema"):
            module.confirm_action_candidate(
                self._p0(), candidate, self._confirmation(), threshold=0.5,
            )

        candidate = self._candidate(output={"label": "A"})
        confirmation = self._confirmation(output={"label": "A"})
        with self.assertRaisesRegex(ValueError, "forbidden metadata"):
            module.confirm_action_candidate(
                self._p0(), candidate, confirmation, threshold=0.5,
            )

        with self.assertRaisesRegex(ValueError, "finite"):
            module.confirm_action_candidate(
                self._p0(), self._candidate(confidence=float("nan")),
                self._confirmation(), threshold=0.5,
            )

    def test_confirmation_threshold_must_be_one_frozen_coarse_value(self):
        module = self._module()
        for threshold in (0.25, 0.5, 0.75):
            decision = module.confirm_action_candidate(
                self._p0(confidence=0.0), self._candidate(),
                self._confirmation(confidence=0.8), threshold=threshold,
            )
            self.assertEqual(decision.action, "ZOOM")
        with self.assertRaisesRegex(ValueError, "frozen coarse"):
            module.confirm_action_candidate(
                self._p0(), self._candidate(), self._confirmation(), threshold=0.6,
            )

    def test_confirmation_snapshot_is_immutable_from_caller_mutation(self):
        module = self._module()
        p0 = self._p0()
        candidate = self._candidate(output=["A", "B"])
        confirmation = self._confirmation(output=["A", "B"])
        decision = module.confirm_action_candidate(
            p0, candidate, confirmation, threshold=0.5,
        )
        candidate["output"][0] = "D"
        confirmation["output"][0] = "C"
        self.assertEqual(decision.output, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
