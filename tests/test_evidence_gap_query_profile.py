import unittest

from cvsearch.evidence_gap.query_profile import adaptive_alpha, infer_query_profile


class QueryProfileTest(unittest.TestCase):
    def test_attribute_question_has_high_detail_and_low_context(self):
        profile = infer_query_profile("What is the color of the comb?", ["comb"])

        self.assertEqual(profile.detail_demand, 1.0)
        self.assertEqual(profile.context_demand, 0.0)

    def test_relation_question_has_high_context(self):
        profile = infer_query_profile(
            "Is the motorcycle on the left or right side of the street?",
            ["motorcycle", "street"],
        )

        self.assertEqual(profile.detail_demand, 0.0)
        self.assertEqual(profile.context_demand, 1.0)

    def test_mixed_question_keeps_both_demands(self):
        profile = infer_query_profile(
            "What color is the cup left of the plate?", ["cup", "plate"]
        )

        self.assertEqual(profile.detail_demand, 1.0)
        self.assertEqual(profile.context_demand, 1.0)

    def test_written_number_is_a_detail_attribute(self):
        profile = infer_query_profile("What number is on the bus?", ["bus"])

        self.assertEqual(profile.detail_demand, 1.0)
        self.assertEqual(profile.context_demand, 0.0)

    def test_neutral_single_target_question_keeps_both_demands_low(self):
        profile = infer_query_profile("Is there a dog?", ["dog"])

        self.assertEqual(profile.to_dict(), {
            "detail_demand": 0.0,
            "context_demand": 0.0,
        })

    def test_adaptive_alpha_uses_both_demands_and_clamps(self):
        detail = infer_query_profile("What color is the comb?", ["comb"])
        context = infer_query_profile("Is the comb left of the cup?", ["comb", "cup"])
        mixed = infer_query_profile("What color is the comb left of the cup?", ["comb", "cup"])

        self.assertAlmostEqual(adaptive_alpha(0.65, detail, 0.3, 0.3), 0.35)
        self.assertAlmostEqual(adaptive_alpha(0.65, context, 0.3, 0.3), 0.95)
        self.assertAlmostEqual(adaptive_alpha(0.65, mixed, 0.3, 0.3), 0.65)
        self.assertEqual(adaptive_alpha(0.1, detail, 0.9, 0.0), 0.0)
        self.assertEqual(adaptive_alpha(0.9, context, 0.0, 0.9), 1.0)

    def test_rejects_invalid_question_and_queries(self):
        for question, queries in (("", ["dog"]), (None, ["dog"]), ("question", "dog")):
            with self.subTest(question=question, queries=queries):
                with self.assertRaises((TypeError, ValueError)):
                    infer_query_profile(question, queries)
        for query in ("", None):
            with self.subTest(query=query):
                with self.assertRaises((TypeError, ValueError)):
                    infer_query_profile("question", [query])


if __name__ == "__main__":
    unittest.main()
