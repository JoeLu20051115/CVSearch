import unittest

from cvsearch.eval.run_split_search import build_parser, split_policy_budget


class SplitRunnerPolicyTest(unittest.TestCase):
    def test_default_preserves_v2_and_stage3b_requires_explicit_v3(self):
        required = [
            "--input-jsonl", "in.jsonl", "--model-path", "model",
            "--clip-model-path", "clip", "--image-root", "images",
            "--output", "out.jsonl",
        ]
        default = build_parser().parse_args(required)
        self.assertEqual(split_policy_budget(default.render_policy), (
            "native_2x2_overlap_support_screen_two_scale_depth2_v2", 4,
        ))
        rescue = build_parser().parse_args(required + [
            "--render-policy",
            "native_2x2_overlap_support_screen_three_scale_all_roots_depth2_v3",
        ])
        self.assertEqual(split_policy_budget(rescue.render_policy)[1], 6)

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            split_policy_budget("unknown")


if __name__ == "__main__":
    unittest.main()
