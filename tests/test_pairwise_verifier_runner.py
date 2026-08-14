import copy
import unittest
from types import SimpleNamespace

from PIL import Image

from cvsearch.eval.pairwise_verifier_runner import (
    _set_verifier_max_pixels,
    build_parser,
    produce_pairwise_record,
)
from tests.test_pairwise_verifier import split_audit
from tests.test_replay_split_search import calibration, rescue_rows


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def multiple_choices_with_losses(self, image, prompt, choices, nodes):
        self.calls.append((image.copy(), prompt, list(choices), list(nodes)))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def feasible_rows():
    stage2, split = rescue_rows()
    stage2["question"] = "Which answer is visible?"
    stage2["options"] = ["A", "B"]
    for branch_index, branch in enumerate(split_audit(split)["branches"]):
        for role in ("tight_view", "medium_view", "context_view"):
            if role in branch:
                branch[role]["answer"] = "A"
        if branch_index < 2:
            for role in ("tight_view", "context_view"):
                branch[role]["answer"] = "B"
                branch[role]["raw_support"] = 0.9
    return stage2, split


class PairwiseRunnerTests(unittest.TestCase):
    def test_cli_accepts_one_shared_external_verifier_checkpoint(self):
        args = build_parser().parse_args([
            "--development-root", "/development",
            "--stage2-root", "/stage2",
            "--split-root", "/split",
            "--support-calibration", "/calibration.json",
            "--workspace-root", "/workspace",
            "--backbone", "internvl",
            "--verifier-model-path", "/shared/cosmos",
            "--output", "/output.jsonl",
        ])

        self.assertEqual(str(args.verifier_model_path), "/shared/cosmos")

    def test_cli_binds_and_applies_external_verifier_pixel_budget(self):
        args = build_parser().parse_args([
            "--development-root", "/development",
            "--stage2-root", "/stage2",
            "--split-root", "/split",
            "--support-calibration", "/calibration.json",
            "--workspace-root", "/workspace",
            "--backbone", "qwen",
            "--verifier-max-pixels", "4194304",
            "--output", "/output.jsonl",
        ])
        model = SimpleNamespace(processor=SimpleNamespace(
            image_processor=SimpleNamespace(max_pixels=12_845_056),
        ))

        self.assertEqual(args.verifier_max_pixels, 4_194_304)
        self.assertEqual(
            _set_verifier_max_pixels(model, args.verifier_max_pixels),
            4_194_304,
        )
        self.assertEqual(model.processor.image_processor.max_pixels, 4_194_304)

        with self.assertRaises(ValueError):
            _set_verifier_max_pixels(model, 0)

    def test_record_is_label_blind_order_symmetric_and_frozen_for_grid(self):
        stage2, split = feasible_rows()
        stage2["answer"] = "GOLD_DO_NOT_USE"
        stage2["category"] = "POISON_CATEGORY"
        split["answer"] = "OTHER_GOLD"
        model = FakeModel(((1, [1.0, 0.0]), (0, [0.0, 1.0])))

        record = produce_pairwise_record(
            stage2, split, calibration(), Image.new("RGB", (100, 100)), model,
        )

        self.assertTrue(record["proposal"]["feasible"])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0][2], model.calls[1][2])
        self.assertIn('Answer 1: "A"', model.calls[0][1])
        self.assertIn('Answer 1: "B"', model.calls[1][1])
        self.assertNotIn("GOLD_DO_NOT_USE", repr(model.calls))
        self.assertNotIn("POISON_CATEGORY", repr(model.calls))
        self.assertEqual(record["cost"]["planned_verifier_calls"], 2)
        self.assertEqual(record["cost"]["charged_verifier_calls"], 2)
        self.assertEqual(record["cost"]["observations"], 10)
        self.assertEqual(len(record["decisions"]), 15)
        self.assertEqual(
            record["decisions"]["agreement=0.4,confidence=0.5"]
            ["selected_source"],
            "PAIRWISE",
        )

    def test_exception_is_hashed_and_all_grid_points_fall_back(self):
        stage2, split = feasible_rows()
        model = FakeModel((RuntimeError("secret failure payload"),))

        record = produce_pairwise_record(
            stage2, split, calibration(), Image.new("RGB", (100, 100)), model,
        )

        self.assertEqual(record["failure"]["type"], "RuntimeError")
        self.assertNotIn("secret failure payload", repr(record))
        self.assertEqual(record["cost"]["charged_verifier_calls"], 1)
        self.assertTrue(all(
            decision["selected_source"] == "P0"
            for decision in record["decisions"].values()
        ))

    def test_independent_source_mode_uses_whole_image_and_short_choices(self):
        stage2, split = feasible_rows()
        source = Image.new("RGB", (100, 100), (7, 8, 9))
        model = FakeModel(((1, [1.0, 0.0]), (0, [0.0, 1.0])))

        record = produce_pairwise_record(
            stage2, split, calibration(), source, model,
            evidence_mode="independent_source",
        )

        self.assertEqual(record["evidence_mode"], "independent_source")
        self.assertEqual(model.calls[0][0].size, source.size)
        self.assertEqual(model.calls[0][0].tobytes(), source.tobytes())
        self.assertEqual(model.calls[0][2], ["1", "2"])
        self.assertEqual(record["render_audit"]["view_size"], [100, 100])

    def test_independent_answer_mode_hides_candidates_and_uses_original_task(self):
        stage2, split = feasible_rows()
        stage2["answer_type"] = "option_single"
        split["answer_type"] = "option_single"
        stage2["options"] = "A. first\nB. second\nC. third\nD. fourth"
        split["options"] = stage2["options"]
        stage2["answer"] = "GOLD_DO_NOT_USE"
        model = FakeModel(((1, [2.0, 0.0, 3.0, 4.0]),))

        record = produce_pairwise_record(
            stage2, split, calibration(), Image.new("RGB", (100, 100)), model,
            evidence_mode="independent_answer",
        )

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][2], ["A", "B", "C", "D"])
        self.assertNotIn("Proposal", model.calls[0][1])
        self.assertNotIn("GOLD_DO_NOT_USE", repr(model.calls))
        self.assertEqual(record["cost"]["planned_verifier_calls"], 1)
        self.assertEqual(record["cost"]["charged_verifier_calls"], 1)
        self.assertEqual(
            record["decisions"]["agreement=0.4,confidence=0.5"]
            ["selected_source"],
            "INDEPENDENT_ANSWER",
        )

    def test_independent_crop_answers_require_two_candidate_free_views(self):
        stage2, split = feasible_rows()
        stage2["answer_type"] = "option_single"
        split["answer_type"] = "option_single"
        stage2["options"] = "A. first\nB. second\nC. third\nD. fourth"
        split["options"] = stage2["options"]
        model = FakeModel((
            (1, [2.0, 0.0, 3.0, 4.0]),
            (1, [3.0, 0.0, 2.0, 4.0]),
        ))

        record = produce_pairwise_record(
            stage2, split, calibration(), Image.new("RGB", (100, 100)), model,
            evidence_mode="independent_crop_answers",
        )

        self.assertEqual(len(model.calls), 2)
        self.assertNotIn("Proposal", repr(model.calls))
        self.assertEqual(record["cost"]["planned_verifier_calls"], 2)
        self.assertEqual(record["projection"]["canonical_answer"], "B")
        self.assertEqual(
            record["decisions"]["agreement=0.4,confidence=0.5"]
            ["selected_source"],
            "INDEPENDENT_ANSWER",
        )

    def test_infeasible_proposal_skips_model_and_keeps_exact_p0(self):
        stage2, split = rescue_rows()
        model = FakeModel(())

        record = produce_pairwise_record(
            stage2, split, calibration(), Image.new("RGB", (100, 100)), model,
        )

        self.assertFalse(record["proposal"]["feasible"])
        self.assertEqual(model.calls, [])
        self.assertEqual(record["cost"]["planned_verifier_calls"], 0)
        self.assertTrue(all(
            decision["selected_output"] == record["proposal"]["stage2_output"]
            for decision in record["decisions"].values()
        ))


if __name__ == "__main__":
    unittest.main()
