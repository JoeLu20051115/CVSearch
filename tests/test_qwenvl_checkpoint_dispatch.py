import unittest

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from cvsearch.models.modeling_qwenvl import _model_class_for_type


class QwenVLCheckpointDispatchTests(unittest.TestCase):
    def test_dispatches_by_model_type_instead_of_path_name(self):
        self.assertIs(
            _model_class_for_type("qwen2_5_vl"),
            Qwen2_5_VLForConditionalGeneration,
        )
        self.assertIs(
            _model_class_for_type("qwen3_vl"),
            Qwen3VLForConditionalGeneration,
        )

    def test_rejects_unsupported_model_type(self):
        with self.assertRaisesRegex(ValueError, "unsupported Qwen-VL model type"):
            _model_class_for_type("qwen2")


if __name__ == "__main__":
    unittest.main()
