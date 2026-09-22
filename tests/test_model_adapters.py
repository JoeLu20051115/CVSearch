import ast
import inspect
from pathlib import Path
import unittest
import pytest

pytest.importorskip("transformers")
from importlib.util import find_spec
if find_spec("llava") is None:
    pytest.skip("optional LLaVA runtime is not installed", allow_module_level=True)

from qavs.models.modeling_qwenvl import ModelQwenVL
from qavs.models.modeling_llava import Model as ModelLlava


class ModelAdapterTest(unittest.TestCase):
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

        llava_method = getattr(
            ModelLlava.multiple_choices_with_losses, "__wrapped__",
            ModelLlava.multiple_choices_with_losses,
        )
        with self.assertRaisesRegex(ValueError, "options"):
            llava_method(object(), None, "question", [])



def test_llava_qwen_checkpoint_loading_does_not_depend_on_directory_name(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from qavs.models import modeling_llava as adapter

    checkpoint = tmp_path / "verifier"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        '{"model_type":"llava_qwen","architectures":["LlavaQwenForCausalLM"]}'
    )
    monkeypatch.setattr(adapter, "disable_torch_init", lambda: None)
    monkeypatch.setattr(adapter.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: SimpleNamespace(padding_side="left"))

    class SelectedQwen(Exception):
        pass

    def load_qwen(*args, **kwargs):
        raise SelectedQwen

    monkeypatch.setattr(adapter.LlavaQwenForCausalLM, "from_pretrained", load_qwen)
    with pytest.raises(SelectedQwen):
        adapter.ModelGlobalLocal(model_path=str(checkpoint), attn_implementation="sdpa")
