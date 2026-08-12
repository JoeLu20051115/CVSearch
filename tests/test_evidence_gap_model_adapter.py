import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import torch

from cvsearch.evidence_gap.adaptive_controller import (
    IsotonicCalibrator,
    SupportObservation,
)
from cvsearch.eval.replay_adaptive_search import FrozenCalibration
from cvsearch.evidence_gap.types import (
    EvidenceSupportResult,
    _validate_support_token_contract,
    sanitize_evidence_requirements,
)
from cvsearch.models import modeling_internvl
from cvsearch.models.modeling_internvl import (
    ANSWER_FREE_SUPPORT_PROMPT_VERSION,
    ModelInternvl,
)
from cvsearch.perform_EGSearch import _model_family_from_config


ROOT = Path(__file__).resolve().parents[1]


class FakeBatch(dict):
    def to(self, device):
        for key, value in tuple(self.items()):
            if hasattr(value, "to"):
                self[key] = value.to(device)
        return self


class FakeTokenizer:
    name_or_path = "frozen/internvl-checkpoint"
    padding_side = "right"

    def __call__(self, text, **kwargs):
        if text == "Yes":
            return SimpleNamespace(input_ids=[1, 9583])
        if text == "No":
            return SimpleNamespace(input_ids=[1, 2917])
        if isinstance(text, list):
            return FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))
        raise AssertionError(f"unexpected tokenizer input: {text!r}")


class FakeInternVLForward:
    num_image_token = 2

    def __init__(self):
        self.calls = 0
        self.config = SimpleNamespace(
            _name_or_path="frozen/internvl-checkpoint",
            model_type="internvl_chat",
            architectures=["InternVLChatModel"],
            transformers_version="4.37.2",
            force_image_size=448,
            llm_config=SimpleNamespace(vocab_size=92553),
        )
        self.dtype = torch.bfloat16
        self.device = torch.device("cpu")

    def __call__(self, **kwargs):
        self.calls += 1
        self.last_inputs = kwargs
        logits = torch.zeros((1, 3, 9584), dtype=torch.float32)
        logits[0, -1, 9583] = 2.0
        logits[0, -1, 2917] = -1.0
        return SimpleNamespace(logits=logits)


def internvl_adapter():
    adapter = ModelInternvl.__new__(ModelInternvl)
    adapter.device = "cpu"
    adapter.dtype = torch.bfloat16
    adapter.model_checkpoint = "frozen/internvl-checkpoint"
    adapter.use_flash_attn = True
    adapter.tokenizer = FakeTokenizer()
    adapter.model = FakeInternVLForward()
    adapter.index_yes = 9583
    adapter.index_no = 2917
    adapter.input_size = (448, 448)
    adapter.anyres_num = 12
    adapter.get_prompt_from_qs = lambda text: text
    return adapter


class InternVLAdapterConformanceTest(unittest.TestCase):
    def setUp(self):
        self.requirements = sanitize_evidence_requirements(({
            "kind": "target_detail",
            "target": "small red sign",
            "requirements": ["presence", "visual_detail"],
        },))

    def test_exposes_answer_and_normalized_answer_free_support_contract(self):
        adapter = internvl_adapter()
        identity = json.dumps(
            {"canonical_keys": ["focus"], "renderer_identity": "internvl-view"},
            sort_keys=True, separators=(",", ":"),
        )
        with patch.object(
            modeling_internvl, "load_image",
            return_value=torch.ones((1, 3, 2, 2), dtype=torch.float32),
        ):
            result = adapter.evidence_support(
                question="What color is the sign?",
                requirements=self.requirements,
                rendered_observation=Image.new("RGB", (4, 3), "red"),
                observation_identity=identity,
            )

        self.assertIsInstance(result, EvidenceSupportResult)
        self.assertEqual(result.prompt_version, ANSWER_FREE_SUPPORT_PROMPT_VERSION)
        self.assertEqual(result.processor_fingerprint["model_type"], "internvl_chat")
        self.assertEqual(result.yes_tokenization, (9583,))
        self.assertEqual(result.no_tokenization, (2917,))
        self.assertAlmostEqual(result.p_yes + result.p_no, 1.0)
        self.assertAlmostEqual(result.p_yes, math.exp(2) / (math.exp(2) + math.exp(-1)))
        self.assertEqual(adapter.model.calls, 1)
        self.assertTrue(callable(adapter.free_form_using_nodes))
        self.assertTrue(callable(adapter.multiple_choices_with_losses))

        controller_input = SupportObservation(
            p_full=result.p_yes, p_partial=1.0 - result.p_yes, p_none=0.0,
            support_consistency=1.0, answer_consistency=1.0,
            missing_reason="none", normalized_cost=0.1,
        ).to_dict()
        self.assertFalse(any("logit" in key or "token" in key for key in controller_input))

    def test_config_type_dispatch_does_not_trust_checkpoint_directory_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            internvl = root / "misleading-qwen-name"
            internvl.mkdir()
            (internvl / "config.json").write_text(
                json.dumps({"model_type": "internvl_chat"}), encoding="utf-8",
            )
            qwen = root / "misleading-internvl-name"
            qwen.mkdir()
            (qwen / "config.json").write_text(
                json.dumps({"model_type": "qwen2_5_vl"}), encoding="utf-8",
            )

            self.assertEqual(_model_family_from_config(internvl), "internvl")
            self.assertEqual(_model_family_from_config(qwen), "qwen")

    def test_unknown_model_type_fails_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps({"model_type": "unknown_vlm"}), encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported model_type"):
                _model_family_from_config(root)

    def test_batch_contract_admits_pinned_internvl_tokens_without_weakening_qwen(self):
        _validate_support_token_contract({
            "processor_fingerprint": {"model_type": "internvl_chat"},
            "yes_tokenization": [9583], "no_tokenization": [2917],
            "yes_token_id": 9583, "no_token_id": 2917,
        })
        with self.assertRaisesRegex(ValueError, "token"):
            _validate_support_token_contract({
                "processor_fingerprint": {"model_type": "qwen2_5_vl"},
                "yes_tokenization": [9583], "no_tokenization": [2917],
                "yes_token_id": 9583, "no_token_id": 2917,
            })

    def test_transfer_report_calibration_is_authentic_and_does_not_overclaim(self):
        path = (
            ROOT / "reproduction" / "evidence_gap" / "reports"
            / "adaptive-search-v2-transfer.json"
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        manifest = report["calibration"]["manifest"]
        frozen = FrozenCalibration(
            IsotonicCalibrator(
                tuple(manifest["calibrator"]["upper_bounds"]),
                tuple(manifest["calibrator"]["probabilities"]),
            ),
            manifest["sample_count"],
            manifest["manifest_sha256"],
        )

        self.assertEqual(frozen.to_dict(), manifest)
        self.assertFalse(report["transfer_gate"]["passed"])
        self.assertFalse(report["claim_allowed"])
        self.assertFalse(report["extra_dataset"]["executed"])


if __name__ == "__main__":
    unittest.main()
