import unittest
from unittest.mock import patch

from cvsearch.perform_EGSearch import (
    _compose_policy_output,
    _normalize_policy,
    _validate_output,
)


class MMEPolicyCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.original = {
            "question": "Which object is visible?",
            "options": [
                "A. alpha", "B. beta", "C. gamma", "D. delta", "E. epsilon",
            ],
            "answer_type": "Multiple Choice",
            "input_image": "images/example.jpg",
            "answer": "E",
            "category": "hidden metadata",
        }

    def test_mme_policy_uses_only_visible_fields_and_normalizes_to_option_single(self):
        policy = _normalize_policy("mme-realworld-lite", self.original)

        self.assertEqual(set(policy), {
            "question", "options", "answer_type", "input_image",
        })
        self.assertEqual(policy["answer_type"], "option_single")
        self.assertEqual(
            policy["options"],
            "A. alpha\nB. beta\nC. gamma\nD. delta\nE. epsilon",
        )

    def test_mme_policy_normalizes_official_parenthesized_option_labels(self):
        official = dict(
            self.original,
            options=[
                "(A) alpha", "(B) beta", "(C) gamma", "(D) delta",
                "(E) epsilon",
            ],
        )

        policy = _normalize_policy("mme-realworld-lite", official)

        self.assertEqual(
            policy["options"],
            "A. alpha\nB. beta\nC. gamma\nD. delta\nE. epsilon",
        )

    def test_mme_policy_coalesces_adjacent_duplicate_official_labels(self):
        official = dict(
            self.original,
            options=[
                "(A) alpha", "(B) beta", "(C) gamma", "(D) delta",
                "(D) None of the above", "(E) epsilon",
            ],
        )

        policy = _normalize_policy("mme-realworld-lite", official)

        self.assertEqual(
            policy["options"],
            "A. alpha\nB. beta\nC. gamma\n"
            "D. delta / None of the above\nE. epsilon",
        )

    def test_mme_policy_normalizes_multiline_option_text(self):
        official = dict(
            self.original,
            options=[
                "(A) alpha", "(B) beta\ncontinued", "(C) gamma",
                "(D) delta", "(E) epsilon",
            ],
        )

        policy = _normalize_policy("mme-realworld-lite", official)

        self.assertEqual(
            policy["options"],
            "A. alpha\nB. beta continued\nC. gamma\nD. delta\nE. epsilon",
        )

    def test_hidden_mme_metadata_does_not_change_policy(self):
        changed = dict(self.original, answer="A", category="different")
        self.assertEqual(
            _normalize_policy("mme-realworld-lite", self.original),
            _normalize_policy("mme-realworld-lite", changed),
        )

    def test_mme_output_requires_a_parseable_a_to_e_letter(self):
        policy = _normalize_policy("mme-realworld-lite", self.original)
        _validate_output("mme-realworld-lite", policy, "The answer is E.")
        for output in ("F", "unknown", ["E"], 4):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    _validate_output("mme-realworld-lite", policy, output)

    def test_output_record_carries_the_normalized_visible_schema_downstream(self):
        policy = _normalize_policy("mme-realworld-lite", self.original)
        with patch(
            "cvsearch.perform_EGSearch.compose_output_record",
            return_value={**self.original, "output": "E", "method_trace": {}},
        ):
            record = _compose_policy_output(
                self.original, policy, "E", object(),
            )

        self.assertEqual(record["answer_type"], "option_single")
        self.assertEqual(record["options"], policy["options"])
        self.assertEqual(record["output"], "E")

    def test_mme_option_schema_fails_closed(self):
        for options in (
            self.original["options"][:-1],
            ["A. a", "B. b", "C. c", "D. d", "F. f"],
            "A. a\nB. b\nC. c\nD. d\nE. e",
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    _normalize_policy(
                        "mme-realworld-lite", dict(self.original, options=options),
                    )


if __name__ == "__main__":
    unittest.main()
