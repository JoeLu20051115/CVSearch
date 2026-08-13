import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvsearch.eval.run_split_search import (
    _load_model,
    _model_family,
    build_parser,
    split_policy_budget,
)


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

    def test_llava_checkpoint_is_available_only_as_shared_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "config.json").write_text(
                json.dumps({"model_type": "llava"}), encoding="utf-8",
            )
            self.assertEqual(_model_family(model_path), "llava_verifier")

            with patch(
                "cvsearch.models.modeling_llava.ModelGlobalLocal",
            ) as wrapper:
                loaded = _load_model(model_path)

            self.assertIs(loaded, wrapper.return_value)
            wrapper.assert_called_once_with(
                model_path=str(model_path), conv_type="qwen_1_5",
                device="cuda:0", torch_dtype=__import__("torch").bfloat16,
                patch_scale=1.2,
            )


if __name__ == "__main__":
    unittest.main()
