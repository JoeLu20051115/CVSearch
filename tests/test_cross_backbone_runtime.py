import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from cvsearch.models.modeling_dispatch import (
    detect_model_family,
    finalize_option_losses,
)
from cvsearch.models.modeling_internvl import ModelInternvl
from cvsearch.models.modeling_llava import Model, ModelGlobalLocal


class CrossBackboneRuntimeTests(unittest.TestCase):
    def test_detects_supported_families_from_checkpoint_config(self):
        cases = (
            ({"model_type": "llava_qwen", "image_aspect_ratio": "anyres_max_9"}, "llava"),
            ({"architectures": ["InternVLChatModel"]}, "internvl"),
            ({"model_type": "qwen2_5_vl"}, "qwen"),
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (config, expected) in enumerate(cases):
                checkpoint = root / f"checkpoint-{index}"
                checkpoint.mkdir()
                (checkpoint / "config.json").write_text(json.dumps(config))
                with self.subTest(config=config):
                    self.assertEqual(detect_model_family(checkpoint), expected)

    def test_rejects_unsupported_checkpoint_config(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "config.json").write_text('{"model_type":"clip"}')
            with self.assertRaisesRegex(ValueError, "unsupported multimodal backbone"):
                detect_model_family(checkpoint)

    def test_finalizes_finite_losses_and_preserves_first_argmin(self):
        winner, losses = finalize_option_losses([
            torch.tensor(2.0), torch.tensor(1.0), torch.tensor(1.0),
        ])
        self.assertEqual(winner, 1)
        self.assertEqual(losses, [2.0, 1.0, 1.0])
        for invalid in ([], [torch.tensor(math.inf)], [torch.tensor(math.nan)]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    finalize_option_losses(invalid)

    def test_llava_choice_api_delegates_to_loss_api(self):
        model = object.__new__(ModelGlobalLocal)
        calls = []
        model.multiple_choices_with_losses = lambda *args: calls.append(args) or (2, [3.0, 2.0, 1.0])
        result = ModelGlobalLocal.multiple_choices_inference(
            model, "image", "question", ["a", "b", "c"], [],
        )
        self.assertEqual(result, 2)
        self.assertEqual(len(calls), 1)

    def test_internvl_choice_api_delegates_to_loss_api(self):
        model = object.__new__(ModelInternvl)
        calls = []
        model.multiple_choices_with_losses = lambda *args: calls.append(args) or (1, [2.0, 1.0])
        result = ModelInternvl.multiple_choices_inference(
            model, "image", "question", ["a", "b"], [],
        )
        self.assertEqual(result, 1)
        self.assertEqual(len(calls), 1)

    def test_cross_backbone_wrappers_expose_text_only_generation(self):
        self.assertTrue(callable(getattr(ModelGlobalLocal, "generate_text_only", None)))
        self.assertTrue(callable(getattr(ModelInternvl, "generate_text_only", None)))
        for wrapper in (ModelGlobalLocal, ModelInternvl):
            self.assertTrue(callable(getattr(wrapper, "_prepare_evidence_support", None)))
            self.assertTrue(callable(getattr(wrapper, "evidence_support", None)))

    def test_llava_anyres_overrides_free_form_for_search_prompt_schema(self):
        self.assertIsNot(
            ModelGlobalLocal.free_form_using_nodes,
            Model.free_form_using_nodes,
        )


if __name__ == "__main__":
    unittest.main()
