import math
import unittest

from PIL import Image


class Phase9DenseEvidenceSearchTest(unittest.TestCase):
    def _module(self):
        from cvsearch.eval import phase9_dense_evidence_search

        return phase9_dense_evidence_search

    def test_dense_bank_has_29_overlap_expanded_in_bounds_unique_tiles(self):
        module = self._module()
        tiles = module.generate_dense_tiles(100, 80)
        self.assertEqual(len(tiles), 29)
        self.assertEqual(len({tile.box for tile in tiles}), 29)
        self.assertEqual(tiles[0].identity, "g2-r0-c0")
        self.assertEqual(tiles[0].box, (0, 0, 57, 45))
        self.assertEqual(tiles[1].box, (43, 0, 100, 45))
        self.assertEqual(tiles[-1].identity, "g4-r3-c3")
        self.assertTrue(all(
            0 <= tile.box[0] < tile.box[2] <= 100
            and 0 <= tile.box[1] < tile.box[3] <= 80
            for tile in tiles
        ))
        with self.assertRaisesRegex(ValueError, "positive"):
            module.generate_dense_tiles(0, 80)

    def test_rank_is_layer_percentile_based_finite_and_deterministic(self):
        module = self._module()
        tiles = module.generate_dense_tiles(100, 80)
        count = len(tiles)
        all_equal = [0.5] * count
        ranked = module.rank_dense_tiles(
            tiles, all_equal, all_equal, all_equal,
        )
        self.assertEqual(ranked[0].tile.grid, 4)
        self.assertEqual(ranked[0].tile.row, 0)
        self.assertEqual(ranked[0].tile.col, 0)
        self.assertTrue(all(item.score == ranked[0].score for item in ranked))

        relevance = [float(index) for index in range(count)]
        ranked_again = module.rank_dense_tiles(
            tiles, relevance, all_equal, all_equal,
        )
        self.assertEqual(ranked_again[0].tile, tiles[-1])
        self.assertGreater(
            ranked_again[0].relevance_percentile,
            ranked_again[-1].relevance_percentile,
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            bad = relevance.copy()
            bad[0] = math.nan
            module.rank_dense_tiles(tiles, bad, all_equal, all_equal)

    def test_query_templates_are_answer_free_and_evidence_sheet_is_exact(self):
        module = self._module()
        question = "What color is the sign?"
        self.assertEqual(
            module.dense_query_texts(question),
            (
                question,
                "visual region needed to answer: " + question,
                "visual evidence for: " + question,
            ),
        )
        source = Image.new("RGB", (100, 80), (10, 20, 30))
        before = source.tobytes()
        tile = module.generate_dense_tiles(100, 80)[5]
        sheet, audit = module.render_dense_evidence_sheet(source, tile)
        self.assertEqual(sheet.mode, "RGB")
        self.assertEqual(sheet.size, (448, 904))
        self.assertEqual(source.tobytes(), before)
        self.assertEqual(
            sheet.crop((0, 448, 448, 456)).getcolors(),
            [(448 * 8, module.DENSE_BACKGROUND_RGB)],
        )
        self.assertNotEqual(audit["focus_panel_sha256"], audit["context_panel_sha256"])
        self.assertEqual(audit["tile_identity"], tile.identity)
        self.assertEqual(audit["tile_box"], list(tile.box))

    def test_visual_features_are_normalized_finite_and_source_immutable(self):
        module = self._module()
        source = Image.new("RGB", (100, 80), (10, 10, 10))
        before = source.tobytes()
        tiles = module.generate_dense_tiles(*source.size)
        edges, variances = module.dense_visual_features(source, tiles)
        self.assertEqual(len(edges), 29)
        self.assertEqual(len(variances), 29)
        self.assertEqual(source.tobytes(), before)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in edges + variances))
        self.assertTrue(all(value == 0.0 for value in variances))

    def test_cross_tile_vstar_and_hr_projection_require_strict_majority(self):
        module = self._module()
        vstar = module.project_dense_candidate(
            "logits_match", ["A", "B"],
            [
                {"winner": 1, "losses": [0.5, 0.1]},
                {"winner": 1, "losses": [0.6, 0.2]},
                {"winner": 0, "losses": [0.1, 0.2]},
            ],
        )
        self.assertTrue(vstar["feasible"])
        self.assertEqual(vstar["output"], 1)
        self.assertEqual(vstar["tile_majority_fraction"], 2 / 3)

        options = [
            "A. Blue\nB. Red\nC. Green\nD. Black\n",
            "A. Green\nB. Blue\nC. Black\nD. Red\n",
            "A. Green\nB. Black\nC. Blue\nD. Red\n",
            "A. Green\nB. Black\nC. Red\nD. Blue\n",
        ]
        hr = module.project_dense_candidate(
            "option_list", options,
            [
                ["A", "B", "C", "D"],
                ["A", "B", "C", "D"],
                ["D", "C", "B", "A"],
            ],
        )
        self.assertTrue(hr["feasible"])
        self.assertEqual(hr["output"], ["A", "B", "C", "D"])
        self.assertEqual(hr["canonical_answer"], "blue")
        self.assertEqual(hr["tile_majority_fraction"], 2 / 3)

        no_majority = module.project_dense_candidate(
            "option_list", options,
            [
                ["A", "B", "C", "D"],
                ["B", "D", "D", "C"],
                ["C", "A", "A", "B"],
            ],
        )
        self.assertFalse(no_majority["feasible"])
        self.assertIsNone(no_majority["output"])

    @staticmethod
    def _p0(output=0, confidence=0.4):
        return {
            "action": "P0", "output": output,
            "p0_stability": {"confidence": confidence},
        }

    @staticmethod
    def _candidate(output=1, confidence=0.7):
        return {
            "action": "DENSE",
            "feasible": True,
            "output": output,
            "stability": {"confidence": confidence},
            "tile_majority_fraction": 2 / 3,
            "sheet_sha256": ["1" * 64, "2" * 64, "3" * 64],
            "rank_sha256": "4" * 64,
        }

    def test_selector_uses_one_frozen_confidence_gain_grid_and_exact_dto(self):
        module = self._module()
        accepted = module.select_dense_candidate(
            self._p0(), self._candidate(confidence=0.5),
            min_confidence=0.5, min_gain=0.1,
        )
        self.assertEqual(accepted.action, "DENSE")
        self.assertEqual(accepted.output, 1)

        retained = module.select_dense_candidate(
            self._p0(), self._candidate(confidence=math.nextafter(0.5, 0.0)),
            min_confidence=0.5, min_gain=0.0,
        )
        self.assertEqual(retained.action, "P0")
        retained = module.select_dense_candidate(
            self._p0(confidence=0.5), self._candidate(confidence=0.5),
            min_confidence=0.5, min_gain=0.1,
        )
        self.assertEqual(retained.action, "P0")

        poisoned = self._candidate()
        poisoned["benchmark"] = "vstar"
        with self.assertRaisesRegex(ValueError, "exact schema"):
            module.select_dense_candidate(
                self._p0(), poisoned, min_confidence=0.5, min_gain=0.0,
            )
        with self.assertRaisesRegex(ValueError, "frozen"):
            module.select_dense_candidate(
                self._p0(), self._candidate(), min_confidence=0.6, min_gain=0.0,
            )


if __name__ == "__main__":
    unittest.main()
