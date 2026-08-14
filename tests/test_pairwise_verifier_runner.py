import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from cvsearch.eval.pairwise_verifier_runner import (
    _partition_specs,
    _set_verifier_max_pixels,
    _split_manifest,
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


class FakeGenerationModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def free_form_using_nodes(self, image, prompt, nodes):
        self.calls.append((image.copy(), prompt, list(nodes)))
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
    def test_legacy_launch_manifest_reconstructs_bound_source_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "qwen" / "mme-realworld-lite.jsonl"
            output.parent.mkdir()
            output.write_text(
                '{"_eg_ordinal":3,"input_image":"image/3.jpg"}\n',
                encoding="utf-8",
            )
            source = root / "dataset" / "image" / "3.jpg"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"image")
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                '{"model_type":"qwen2_5_vl"}\n', encoding="utf-8",
            )
            launch = {
                "schema_version": 1,
                "benchmark": "mme-realworld-lite",
                "selected_partition": {"rows": 1, "ordinals": [3]},
                "artifacts": {
                    "processor": {"path": str(model)},
                    "source_images": {
                        "kind": "selected_source_images",
                        "files": [{
                            "path": str(source),
                            "size": source.stat().st_size,
                            "sha256": hashlib.sha256(
                                source.read_bytes(),
                            ).hexdigest(),
                        }],
                    },
                },
            }
            Path(f"{output}.launch-manifest.json").write_text(
                json.dumps(launch), encoding="utf-8",
            )

            manifest = _split_manifest(output, "qwen")

            self.assertEqual(manifest["model_path"], str(model))
            self.assertEqual(manifest["image_root"], str(root / "dataset"))
            self.assertEqual(
                manifest["binding_path"],
                f"{output}.launch-manifest.json",
            )
            self.assertEqual(manifest["provenance_mode"], "legacy_launch")

    def test_cli_accepts_explicit_disjoint_partition_roots(self):
        args = build_parser().parse_args([
            "--partition", "validation_v1", "/v1", "/v1",
            "--partition", "validation_v2", "/v2-stage2", "/v2-split",
            "--support-calibration", "/calibration.json",
            "--workspace-root", "/workspace",
            "--backbone", "qwen",
            "--output", "/output.jsonl",
        ])

        self.assertEqual(
            tuple(
                (name, str(stage2), str(split))
                for name, stage2, split in _partition_specs(args)
            ),
            (
                ("validation_v1", "/v1", "/v1"),
                ("validation_v2", "/v2-stage2", "/v2-split"),
            ),
        )

    def test_cli_can_bind_only_the_external_mme_benchmark(self):
        args = build_parser().parse_args([
            "--partition", "mme_blind", "/stage2", "/split",
            "--benchmark", "mme-realworld-lite",
            "--support-calibration", "/calibration.json",
            "--workspace-root", "/workspace",
            "--backbone", "qwen",
            "--output", "/output.jsonl",
        ])

        self.assertEqual(args.benchmark, "mme-realworld-lite")

    def test_explicit_partitions_reject_legacy_root_mixing(self):
        args = build_parser().parse_args([
            "--partition", "validation_v1", "/v1", "/v1",
            "--development-root", "/development",
            "--support-calibration", "/calibration.json",
            "--workspace-root", "/workspace",
            "--backbone", "qwen",
            "--output", "/output.jsonl",
        ])

        with self.assertRaises(ValueError):
            _partition_specs(args)

    def test_cli_accepts_one_shared_external_verifier_checkpoint(self):
        args = build_parser().parse_args([
            "--development-root", "/development",
            "--stage2-root", "/stage2",
            "--split-root", "/split",
            "--support-calibration", "/calibration.json",
            "--workspace-root", "/workspace",
            "--backbone", "internvl",
            "--verifier-model-path", "/shared/cosmos",
            "--reuse-independent-answers", "/source.jsonl",
            "--output", "/output.jsonl",
        ])

        self.assertEqual(str(args.verifier_model_path), "/shared/cosmos")
        self.assertEqual(
            str(args.reuse_independent_answers), "/source.jsonl",
        )
        self.assertEqual(
            tuple(
                (name, str(stage2), str(split))
                for name, stage2, split in _partition_specs(args)
            ),
            (
                ("development", "/development", "/development"),
                ("validation_v3", "/stage2", "/split"),
            ),
        )

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

    def test_generation_consensus_uses_two_free_form_prompts_and_exact_fallback(self):
        stage2, split = feasible_rows()
        stage2["answer_type"] = split["answer_type"] = "option_single"
        stage2["options"] = split["options"] = (
            "A. first\nB. second\nC. third\nD. fourth"
        )
        model = FakeGenerationModel(("Answer: B.", "B"))

        record = produce_pairwise_record(
            stage2, split, calibration(), Image.new("RGB", (100, 100)), model,
            evidence_mode="generation_consensus",
        )

        self.assertEqual(len(model.calls), 2)
        self.assertNotIn("Proposal", repr(model.calls))
        self.assertEqual(record["projection"]["canonical_answer"], "B")
        self.assertEqual(record["cost"]["planned_verifier_calls"], 2)
        self.assertEqual(record["cost"]["charged_verifier_calls"], 2)
        self.assertEqual(
            record["decisions"]["agreement=0.4,confidence=0.9"]
            ["selected_source"],
            "INDEPENDENT_ANSWER",
        )

        rejected = produce_pairwise_record(
            stage2, split, calibration(), Image.new("RGB", (100, 100)),
            FakeGenerationModel(("B", "C")),
            evidence_mode="generation_consensus",
        )
        self.assertEqual(rejected["projection"]["feasible"], False)
        self.assertTrue(all(
            decision["selected_output"] == rejected["proposal"]["stage2_output"]
            for decision in rejected["decisions"].values()
        ))

    def test_independent_answer_reuses_exact_bound_source_observations(self):
        stage2, split = feasible_rows()
        stage2["answer_type"] = split["answer_type"] = "option_single"
        stage2["options"] = split["options"] = (
            "A. first\nB. second\nC. third\nD. fourth"
        )
        source_model = FakeModel(((1, [2.0, 0.0, 3.0, 4.0]),))
        source_record = produce_pairwise_record(
            stage2, split, calibration(), Image.new("RGB", (100, 100)),
            source_model, evidence_mode="independent_answer",
        )
        target_model = FakeModel(())

        target_record = produce_pairwise_record(
            stage2, split, calibration(), Image.new("RGB", (100, 100)),
            target_model, evidence_mode="independent_answer",
            reused_independent_answer=source_record,
        )

        self.assertEqual(target_model.calls, [])
        self.assertEqual(target_record["projection"], source_record["projection"])
        self.assertEqual(target_record["observations"], source_record["observations"])
        self.assertEqual(target_record["cost"]["charged_verifier_calls"], 0)
        self.assertIsNotNone(target_record["observation_reuse_sha256"])

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
