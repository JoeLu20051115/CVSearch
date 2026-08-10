from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from cvsearch.eval.phase12_generated_query_lazy_runner import (
    _calls_per_record,
    _observe_lazy_path,
    generate_localization_queries,
)


class FakeInputs(dict):
    input_ids = torch.tensor([[10, 11]])

    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return FakeInputs(input_ids=torch.tensor([[10, 11]]))

    def batch_decode(self, ids, **kwargs):
        return ["LOC: red storefront\nLOC: sign above door\nLOC: small text"]


class FakeCore:
    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[10, 11, 20, 21]])


class FakeQueryModel:
    device = "cuda:0"

    def __init__(self):
        self.processor = FakeProcessor()
        self.model = FakeCore()

    def get_prompt_from_qs(self, prompt):
        self.prompt = prompt
        return "CHAT:" + prompt


class Phase12GeneratedQueryRunnerTests(unittest.TestCase):
    def test_localization_generation_is_text_only_and_deterministic(self):
        model = FakeQueryModel()
        result = generate_localization_queries(model, "What is over the door?")
        self.assertEqual(result["queries"], [
            "red storefront", "sign above door", "small text",
        ])
        self.assertIsNone(model.processor.kwargs["images"])
        self.assertNotIn("option", model.prompt.casefold())
        self.assertFalse(model.model.kwargs["do_sample"])

    def test_observation_path_backtracks_only_on_disagreement(self):
        row = {"answer_type": "logits_match", "options": ["x", "y"]}
        agree = [
            {"winner": 1, "losses": [2.0, 1.0]},
            {"winner": 1, "losses": [3.0, 1.0]},
        ]
        with patch(
            "cvsearch.eval.phase12_generated_query_lazy_runner._answer_patch",
            side_effect=agree,
        ) as answer:
            observations, used = _observe_lazy_path(
                object(), row, ["parent", "child", "backtrack"],
            )
        self.assertEqual(len(observations), 2)
        self.assertFalse(used)
        self.assertEqual(answer.call_count, 2)

        disagree = [
            {"winner": 0, "losses": [1.0, 2.0]},
            {"winner": 1, "losses": [2.0, 1.0]},
            {"winner": 0, "losses": [1.0, 3.0]},
        ]
        with patch(
            "cvsearch.eval.phase12_generated_query_lazy_runner._answer_patch",
            side_effect=disagree,
        ) as answer:
            observations, used = _observe_lazy_path(
                object(), row, ["parent", "child", "backtrack"],
            )
        self.assertEqual(len(observations), 3)
        self.assertTrue(used)
        self.assertEqual(answer.call_count, 3)

    def test_worst_case_call_budget_includes_query_and_backtrack(self):
        self.assertEqual(_calls_per_record("logits_match"), 4)
        self.assertEqual(_calls_per_record("option_list"), 13)


if __name__ == "__main__":
    unittest.main()
