import ast
import inspect
import json
import math
from pathlib import Path
import sys
import unittest

from cvsearch.evidence_gap.answers import (
    aggregate_hr_answers,
    aggregate_vstar_losses,
    parse_option_block,
)
from cvsearch.models.modeling_qwenvl import ModelQwenVL


class EvidenceGapAnswersTest(unittest.TestCase):
    def test_parse_option_block_normalizes_real_hr_lines_and_crlf(self):
        self.assertEqual(
            parse_option_block(" A.  Blue   sky \r\nB.\tRed car\r\n"),
            {"A": "Blue sky", "B": "Red car"},
        )

    def test_parse_option_block_rejects_empty_malformed_or_duplicate_labels(self):
        for block in ("", "\n\t", "A red", "E. green", "A. red\nA. blue"):
            with self.subTest(block=block):
                with self.assertRaises(ValueError):
                    parse_option_block(block)

    def test_shuffled_letters_vote_for_one_semantic_answer(self):
        blocks = ["A. red\nB. blue\n", "A. blue\nB. red\n", "A. red\nB. blue\n"]
        record = aggregate_hr_answers(blocks, ["A", "B.", "A"])
        self.assertEqual(record.canonical_answer, "red")
        self.assertEqual(record.output, ["A", "B", "A"])
        self.assertEqual(record.frequency, 1.0)
        self.assertIs(record.aggregation_available, True)
        self.assertIsNone(record.aggregation_reason)

    def test_hr_uses_official_letter_extraction_and_ignores_invalid_votes(self):
        record = aggregate_hr_answers(
            ["A. red\nB. blue", "A. blue\nB. red"],
            ["A.", "unknown"],
        )
        self.assertEqual(record.canonical_answer, "red")
        self.assertEqual(record.output, ["A", "B"])
        self.assertEqual(record.raw_outputs, ("A.", "unknown"))
        self.assertEqual(record.groups["red"]["count"], 1)
        self.assertEqual(record.frequency, 0.5)
        self.assertEqual(record.margin, 0.5)
        self.assertEqual(record.confidence, 0.5)
        self.assertEqual(record.uncertainty, 0.5)

    def test_hr_invalid_votes_remain_in_stability_denominator(self):
        record = aggregate_hr_answers(
            [
                "A. red\nB. blue",
                "A. blue\nB. red",
                "A. red\nB. blue",
                "A. blue\nB. red",
            ],
            ["A", "", "?", "nothing"],
        )
        self.assertEqual(record.canonical_answer, "red")
        self.assertEqual(record.output, ["A", "B", "A", "B"])
        self.assertEqual(record.groups["red"]["count"], 1)
        self.assertEqual(record.frequency, 0.25)
        self.assertEqual(record.margin, 0.25)
        self.assertEqual(record.confidence, 0.25)
        self.assertEqual(record.uncertainty, 0.75)

    def test_hr_tie_with_invalid_vote_uses_all_shuffles_for_stability(self):
        record = aggregate_hr_answers(
            ["A. red\nB. blue", "A. blue\nB. red", "A. red\nB. blue"],
            ["A", "A", "?"],
        )
        self.assertEqual(record.canonical_answer, "red")
        self.assertEqual(record.output, ["A", "B", "A"])
        self.assertAlmostEqual(record.frequency, 1 / 3)
        self.assertEqual(record.margin, 0.0)
        self.assertAlmostEqual(record.confidence, 1 / 3)
        self.assertAlmostEqual(record.uncertainty, 2 / 3)

    def test_hr_tie_breaks_by_first_valid_vote(self):
        record = aggregate_hr_answers(
            ["A. red\nB. blue", "A. blue\nB. red"],
            ["A", "A"],
        )
        self.assertEqual(record.canonical_answer, "red")
        self.assertEqual(record.output, ["A", "B"])
        self.assertEqual(record.frequency, 0.5)
        self.assertEqual(record.margin, 0.0)
        self.assertEqual(record.confidence, 0.5)
        self.assertEqual(record.uncertainty, 0.5)

    def test_hr_rejects_mismatched_inputs_and_nonstring_outputs(self):
        with self.assertRaises(ValueError):
            aggregate_hr_answers(["A. red"], [])
        with self.assertRaises(TypeError):
            aggregate_hr_answers(["A. red"], [1])

    def test_hr_ordinal_80_losing_duplicate_keeps_unique_brown_projection(self):
        blocks = [
            "A. Brown\nB. Red\nC. red\nD. Blue",
            "A. Blue\nB. Brown\nC. red\nD. Red",
            "A. Blue\nB. red\nC. Brown\nD. Red",
            "A. Red\nB. red\nC. Blue\nD. Brown",
        ]
        record = aggregate_hr_answers(blocks, ["A", "B", "C", "D"])
        self.assertEqual(record.output, ["A", "B", "C", "D"])
        self.assertEqual(record.canonical_answer, "brown")
        self.assertIs(record.aggregation_available, True)
        self.assertIsNone(record.aggregation_reason)
        self.assertEqual(record.confidence, 1.0)

    def test_hr_winning_duplicate_fails_closed_to_exact_raw_outputs(self):
        blocks = [
            "A. Red\nB. red\nC. Blue",
            "A. Blue\nB. red\nC. Green",
            "A. Green\nB. Blue\nC. red",
            "A. red\nB. Blue\nC. Green",
        ]
        raw = ["A", "B.", "C answer", "A"]
        record = aggregate_hr_answers(blocks, raw)
        self.assertEqual(record.output, raw)
        self.assertEqual(record.raw_outputs, tuple(raw))
        self.assertIsNone(record.canonical_answer)
        self.assertIs(record.aggregation_available, False)
        self.assertEqual(record.aggregation_reason, "ambiguous_winner_projection")
        self.assertEqual((record.confidence, record.uncertainty), (0.0, 1.0))
        json.dumps(record.to_dict(), allow_nan=False)

    def test_hr_all_invalid_votes_fail_closed_to_exact_raw_outputs(self):
        raw = ["", "unknown"]
        record = aggregate_hr_answers(
            ["A. red\nB. blue", "A. blue\nB. red"], raw
        )
        self.assertIsNone(record.canonical_answer)
        self.assertEqual(record.output, raw)
        self.assertEqual(record.groups, {})
        self.assertEqual(record.frequency, 0.0)
        self.assertEqual(record.confidence, 0.0)
        self.assertEqual(record.uncertainty, 1.0)
        self.assertIs(record.aggregation_available, False)
        self.assertEqual(record.aggregation_reason, "no_valid_votes")

    def test_vstar_averages_losses_and_uses_normalized_top_two_margin(self):
        record = aggregate_vstar_losses([[1.0, 3.0, 2.0], [2.0, 1.0, 4.0]])
        self.assertEqual(record.output, 0)
        self.assertEqual(record.canonical_answer, 0)
        self.assertEqual(record.losses, (1.5, 2.0, 3.0))
        self.assertAlmostEqual(record.margin, 1 / 7)
        self.assertAlmostEqual(record.confidence, 1 / 7)
        self.assertAlmostEqual(record.uncertainty, 6 / 7)
        self.assertEqual(record.raw_outputs, ((1.0, 3.0, 2.0), (2.0, 1.0, 4.0)))
        self.assertEqual(record.groups["prompt_count"], 2)
        self.assertEqual(json.loads(json.dumps(record.to_dict(), allow_nan=False)), record.to_dict())

    def test_vstar_single_option_has_full_confidence_and_ties_choose_lower_index(self):
        single = aggregate_vstar_losses([[0.0], [0.0]])
        self.assertEqual((single.output, single.margin, single.confidence, single.uncertainty), (0, 1.0, 1.0, 0.0))
        tied = aggregate_vstar_losses([[2.0, 2.0], [0.0, 0.0]])
        self.assertEqual(tied.output, 0)
        self.assertEqual((tied.margin, tied.confidence, tied.uncertainty), (0.0, 0.0, 1.0))

    def test_vstar_accepts_negative_losses_and_rejects_ragged_or_nonfinite_rows(self):
        record = aggregate_vstar_losses([[-2.0, -1.0]])
        self.assertEqual(record.output, 0)
        self.assertGreater(record.margin, 0.0)
        for rows in ([], [[]], [[1.0], [1.0, 2.0]], [[math.inf]], [[math.nan]]):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    aggregate_vstar_losses(rows)

    def test_vstar_extreme_finite_losses_stay_finite_and_json_safe(self):
        equal_extremes = aggregate_vstar_losses([[1e308, 1e308], [1e308, 1e308]])
        self.assertEqual(equal_extremes.output, 0)
        self.assertEqual(equal_extremes.losses, (1e308, 1e308))
        self.assertEqual(equal_extremes.margin, 0.0)
        json.dumps(equal_extremes.to_dict(), allow_nan=False)

        opposite_extremes = aggregate_vstar_losses([[-1e308, 1e308]])
        self.assertEqual(opposite_extremes.output, 0)
        self.assertEqual(opposite_extremes.margin, 1.0)
        self.assertEqual(opposite_extremes.confidence, 1.0)
        self.assertEqual(opposite_extremes.uncertainty, 0.0)
        json.dumps(opposite_extremes.to_dict(), allow_nan=False)

    def test_vstar_three_maximum_finite_losses_do_not_overflow_mean(self):
        maximum = sys.float_info.max
        equal_maximums = aggregate_vstar_losses([[maximum, maximum]] * 3)
        self.assertEqual(equal_maximums.losses, (maximum, maximum))
        self.assertEqual(equal_maximums.output, 0)
        json.dumps(equal_maximums.to_dict(), allow_nan=False)

        mixed_maximums = aggregate_vstar_losses([[maximum, -maximum, maximum]] * 3)
        self.assertEqual(mixed_maximums.losses, (maximum, -maximum, maximum))
        self.assertEqual(mixed_maximums.output, 1)
        self.assertEqual(mixed_maximums.margin, 1.0)
        json.dumps(mixed_maximums.to_dict(), allow_nan=False)

    def test_vstar_extreme_cancellation_preserves_tiny_finite_mean(self):
        maximum = sys.float_info.max
        rows = [[maximum, maximum], [-maximum, -maximum], [1e-300, 0.0]]
        expected_losses = (float(1e-300 / 3), 0.0)
        record = aggregate_vstar_losses(rows)
        self.assertEqual(record.losses, expected_losses)
        self.assertEqual(record.output, 1)
        json.dumps(record.to_dict(), allow_nan=False)

        permuted = aggregate_vstar_losses([rows[2], rows[0], rows[1]])
        self.assertEqual(permuted.losses, expected_losses)
        self.assertEqual(permuted.output, 1)
        json.dumps(permuted.to_dict(), allow_nan=False)

    def test_multiple_choice_wrapper_preserves_signature_and_delegates(self):
        signature = inspect.signature(ModelQwenVL.multiple_choices_inference)
        self.assertEqual(list(signature.parameters), ["self", "image_pil", "question", "options", "searched_nodes"])
        self.assertIsNone(signature.parameters["searched_nodes"].default)

        class Delegate:
            def multiple_choices_with_losses(self, *args):
                self.args = args
                return 2, [0.4, 0.2, 0.1]

        delegate = Delegate()
        wrapper = getattr(ModelQwenVL.multiple_choices_inference, "__wrapped__", ModelQwenVL.multiple_choices_inference)
        self.assertEqual(wrapper(delegate, "image", "question", ["a", "b", "c"], "nodes"), 2)
        self.assertEqual(delegate.args, ("image", "question", ["a", "b", "c"], "nodes"))

        module = ast.parse(Path(inspect.getsourcefile(ModelQwenVL)).read_text(encoding="utf-8"))
        method = next(
            item for cls in module.body if isinstance(cls, ast.ClassDef) and cls.name == "ModelQwenVL"
            for item in cls.body if isinstance(item, ast.FunctionDef) and item.name == "multiple_choices_inference"
        )
        self.assertTrue(any(
            getattr(getattr(decorator, "func", decorator), "attr", None) == "inference_mode"
            for decorator in method.decorator_list
        ))

    def test_multiple_choices_with_losses_rejects_empty_options_before_model_access(self):
        method = getattr(ModelQwenVL.multiple_choices_with_losses, "__wrapped__", ModelQwenVL.multiple_choices_with_losses)
        with self.assertRaisesRegex(ValueError, "options"):
            method(object(), None, "question", [])


if __name__ == "__main__":
    unittest.main()
