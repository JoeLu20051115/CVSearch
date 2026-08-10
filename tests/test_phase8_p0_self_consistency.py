import math
import unittest

from PIL import Image


class Phase8P0SelfConsistencyTest(unittest.TestCase):
    def _module(self):
        from cvsearch.eval import phase8_p0_self_consistency

        return phase8_p0_self_consistency

    @staticmethod
    def _p0(output=0, confidence=0.8):
        return {
            "action": "P0",
            "output": output,
            "p0_stability": {"confidence": confidence},
        }

    @staticmethod
    def _view(output=1, confidence=0.5, majority=2 / 3, digit="1"):
        return {
            "feasible": True,
            "output": output,
            "stability": {"confidence": confidence},
            "majority_fraction": majority,
            "view_sha256": digit * 64,
            "prompt_sha256": ["3" * 64, "4" * 64, "5" * 64],
        }

    def test_three_prompt_projection_requires_majority_to_match_mean_loss_winner(self):
        module = self._module()
        accepted = module.aggregate_p0_view(
            ["A", "B"],
            [
                {"winner": 1, "losses": [0.5, 0.1]},
                {"winner": 1, "losses": [0.6, 0.2]},
                {"winner": 0, "losses": [0.1, 0.2]},
            ],
        )
        self.assertTrue(accepted["feasible"])
        self.assertEqual(accepted["output"], 1)
        self.assertAlmostEqual(accepted["majority_fraction"], 2 / 3)
        self.assertLessEqual(accepted["confidence"], 2 / 3)

        rejected = module.aggregate_p0_view(
            ["A", "B"],
            [
                {"winner": 1, "losses": [0.21, 0.20]},
                {"winner": 1, "losses": [0.21, 0.20]},
                {"winner": 0, "losses": [0.00, 1.00]},
            ],
        )
        self.assertFalse(rejected["feasible"])
        self.assertIsNone(rejected["output"])

    def test_selector_requires_distinct_agreeing_views_and_inclusive_threshold(self):
        module = self._module()
        decision = module.select_p0_self_consistency(
            self._p0(), self._view(), self._view(digit="2"), threshold=0.5,
        )
        self.assertEqual(decision.action, "CONSISTENCY")
        self.assertEqual(decision.status, "selected_p0_self_consistency")
        self.assertEqual(decision.output, 1)
        self.assertEqual(decision.confidence, 0.5)

        below = self._view(confidence=math.nextafter(0.5, 0.0))
        retained = module.select_p0_self_consistency(
            self._p0(), below, self._view(digit="2"), threshold=0.5,
        )
        self.assertEqual(retained.action, "P0")

        retained = module.select_p0_self_consistency(
            self._p0(), self._view(), self._view(output=2, digit="2"),
            threshold=0.5,
        )
        self.assertEqual(retained.action, "P0")

        retained = module.select_p0_self_consistency(
            self._p0(), self._view(), self._view(), threshold=0.5,
        )
        self.assertEqual(retained.action, "P0")

    def test_selector_rejects_unfrozen_threshold_and_noncanonical_dto(self):
        module = self._module()
        with self.assertRaisesRegex(ValueError, "frozen coarse"):
            module.select_p0_self_consistency(
                self._p0(), self._view(), self._view(digit="2"), threshold=0.6,
            )

        poisoned = self._view()
        poisoned["ordinal"] = 3
        with self.assertRaisesRegex(ValueError, "exact schema"):
            module.select_p0_self_consistency(
                self._p0(), poisoned, self._view(digit="2"), threshold=0.5,
            )

    def test_infeasible_or_nonmajority_view_fails_closed(self):
        module = self._module()
        infeasible = {
            "feasible": False,
            "output": None,
            "stability": None,
            "majority_fraction": None,
            "view_sha256": "1" * 64,
            "prompt_sha256": ["3" * 64, "4" * 64, "5" * 64],
        }
        retained = module.select_p0_self_consistency(
            self._p0(), infeasible, self._view(digit="2"), threshold=0.25,
        )
        self.assertEqual(retained.action, "P0")

        with self.assertRaisesRegex(ValueError, "strict majority"):
            module.select_p0_self_consistency(
                self._p0(), self._view(majority=0.5), self._view(digit="2"),
                threshold=0.25,
            )

    def test_native_replay_uses_nonroot_fine_nodes_and_returns_exact_views(self):
        module = self._module()
        source = Image.new("RGB", (20, 12), (1, 2, 3))

        class FakeModel:
            def process_nodes_to_image_list(self, nodes, image, root_anyres=True):
                self.nodes = nodes
                self.source = image
                self.root_anyres = root_anyres
                return [
                    Image.new("RGB", (8, 8), (10, 0, 0)),
                    Image.new("RGB", (6, 6), (0, 10, 0)),
                    Image.new("RGB", (10, 10), (0, 0, 10)),
                ]

        model = FakeModel()
        before = source.tobytes()
        focus, context, audit = module.render_p0_native_views(
            source, [(2, 3, 4, 5)], model,
        )

        self.assertEqual(focus.getpixel((0, 0)), (0, 0, 10))
        self.assertEqual(context.getpixel((0, 0)), (10, 0, 0))
        self.assertEqual(source.tobytes(), before)
        self.assertIsNot(model.source, source)
        self.assertTrue(model.root_anyres)
        self.assertEqual(len(model.nodes), 1)
        self.assertFalse(model.nodes[0].is_root)
        self.assertEqual(model.nodes[0].search_source, "fine")
        self.assertEqual(model.nodes[0].state.bbox, (2.0, 3.0, 4.0, 5.0))
        self.assertNotEqual(
            audit["focus_view_sha256"], audit["context_view_sha256"],
        )
        self.assertEqual(audit["boxes"], [[2.0, 3.0, 4.0, 5.0]])

    def test_native_replay_rejects_missing_invalid_or_noop_views(self):
        module = self._module()
        source = Image.new("RGB", (20, 12), (1, 2, 3))

        class NoopModel:
            def process_nodes_to_image_list(self, nodes, image, root_anyres=True):
                view = Image.new("RGB", (8, 8), (4, 5, 6))
                return [view, view.copy(), view.copy()]

        with self.assertRaisesRegex(ValueError, "final boxes"):
            module.render_p0_native_views(source, [], NoopModel())
        for boxes in (
            [(2, 3, 0, 5)],
            [(-1, 3, 4, 5)],
            [(18, 3, 4, 5)],
            [(2, 3, float("nan"), 5)],
        ):
            with self.subTest(boxes=boxes):
                with self.assertRaises((TypeError, ValueError)):
                    module.render_p0_native_views(source, boxes, NoopModel())
        with self.assertRaisesRegex(ValueError, "pixel-distinct"):
            module.render_p0_native_views(
                source, [(2, 3, 4, 5)], NoopModel(),
            )


if __name__ == "__main__":
    unittest.main()
