import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cvsearch.eval.phase4_expand_oracle as phase4
import cvsearch.eval.phase6_final_scorer as final
import cvsearch.eval.phase6_combined_selection as combined
from tests.test_phase4_expand_oracle import FROZEN_REVISION, PROMPT_SHA
from tests.test_phase6_combined_selection import CombinedPairFactory


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


class Phase6FinalScorerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.factory = CombinedPairFactory(self.root)

    def write_pair(self, benchmark="vstar", ordinal=0):
        disabled, enabled, left, right = self.factory.pair(benchmark, ordinal)
        disabled_path = self.root / f"{benchmark}-disabled.jsonl"
        enabled_path = self.root / f"{benchmark}-combined.jsonl"
        disabled_path.write_text(_canonical(disabled) + "\n", encoding="utf-8")
        enabled_path.write_text(_canonical(enabled) + "\n", encoding="utf-8")
        Path(f"{disabled_path}.launch-manifest.json").write_text(
            _canonical(left) + "\n", encoding="utf-8",
        )
        Path(f"{enabled_path}.launch-manifest.json").write_text(
            _canonical(right) + "\n", encoding="utf-8",
        )
        return disabled_path, enabled_path, left, right

    @staticmethod
    def expectation(benchmark="vstar", ordinal=0):
        cycles = 1 if benchmark == "vstar" else 4
        return final.Phase6Expectation(
            benchmark=benchmark, profile="dev", split="dev", topics=1,
            cycles=cycles,
            ordinal_sha256=hashlib.sha256(str(ordinal).encode()).hexdigest(),
            locked_cvsearch_correct=0,
        )

    def prepare(self, benchmark="vstar"):
        disabled_path, enabled_path, left, right = self.write_pair(benchmark)
        expectation = self.expectation(benchmark)
        with (
            patch.dict(
                final.PROFILE_EXPECTATIONS,
                {("dev", benchmark): expectation},
            ),
            patch.object(
                combined, "_FROZEN_COMBINED_INFERENCE_REVISION", FROZEN_REVISION,
            ),
            patch.object(phase4, "_support_prompt_sha256", return_value=PROMPT_SHA),
        ):
            prepared = final.prepare_selected_partition_from_paths(
                benchmark, disabled_path, enabled_path, profile="dev",
            )
        return prepared, disabled_path, enabled_path, left, right

    def test_final_scorer_module_exists(self):
        self.assertIsNotNone(
            importlib.util.find_spec("cvsearch.eval.phase6_final_scorer")
        )

    def test_public_api_is_explicit_and_does_not_import_selector(self):
        expected = (
            "Phase6Expectation",
            "PreparedSelectedPartition",
            "ScoredSelectedPartition",
            "prepare_selected_partition_from_paths",
            "bind_and_score_trusted_annotation",
            "write_selected_bundle_atomic",
            "score_phase6_suite",
        )
        self.assertEqual(
            [name for name in expected if not hasattr(final, name)],
            [],
        )
        self.assertNotIn("select_unified_state", vars(final))

    def test_prepare_freezes_label_blind_partition_before_annotation_read(self):
        disabled_path, enabled_path, _, right = self.write_pair("hr-bench_4k")
        annotation_path = Path(right["artifacts"]["annotation_file"]["path"])
        expectation = self.expectation("hr-bench_4k")
        real_read_bytes = Path.read_bytes

        def reject_annotation_read(path):
            if path == annotation_path:
                raise AssertionError("annotation opened before complete freeze")
            return real_read_bytes(path)

        with (
            patch.dict(
                final.PROFILE_EXPECTATIONS,
                {("dev", "hr-bench_4k"): expectation},
            ),
            patch.object(
                combined, "_FROZEN_COMBINED_INFERENCE_REVISION", FROZEN_REVISION,
            ),
            patch.object(phase4, "_support_prompt_sha256", return_value=PROMPT_SHA),
            patch.object(Path, "read_bytes", reject_annotation_read),
            patch.object(
                final, "freeze_combined_decisions",
                wraps=combined.freeze_combined_decisions,
            ) as freeze,
        ):
            prepared = final.prepare_selected_partition_from_paths(
                "hr-bench_4k", disabled_path, enabled_path, profile="dev",
            )

        self.assertEqual(freeze.call_count, 1)
        frozen_disabled, frozen_enabled = freeze.call_args.args[1:3]
        self.assertNotIn("answer", frozen_disabled[0])
        self.assertNotIn("answer", frozen_enabled[0])
        self.assertEqual(prepared.frozen_batch[0].decision.action, "P0")
        expected_row = {
            "schema_version": 1,
            "_eg_ordinal": 0,
            "output": ["A", "B", "C", "D"],
            "selection": {
                "selector_id": "phase5-stability-gain-tau025-v1",
                "action": "P0",
                "status": "retained_p0",
                "stability_gain": None,
            },
        }
        self.assertEqual(prepared.selected_rows, (expected_row,))
        self.assertEqual(
            prepared.selected_jsonl_bytes,
            (_canonical(expected_row) + "\n").encode("utf-8"),
        )
        self.assertNotIn(b"answer", prepared.selected_jsonl_bytes)
        self.assertNotIn(b"method_trace", prepared.selected_jsonl_bytes)

    def test_prepare_parses_only_bytes_bound_by_its_file_snapshots(self):
        disabled_path, enabled_path, _, _ = self.write_pair("vstar")
        expectation = self.expectation("vstar")
        source_paths = {
            disabled_path, enabled_path,
            Path(f"{disabled_path}.launch-manifest.json"),
            Path(f"{enabled_path}.launch-manifest.json"),
        }
        real_read_bytes = Path.read_bytes
        real_read_text = Path.read_text

        def reject_second_path_read(path):
            if path in source_paths:
                raise AssertionError("source reopened outside its exact snapshot")
            return real_read_bytes(path)

        def reject_second_text_read(path, *args, **kwargs):
            if path in source_paths:
                raise AssertionError("manifest reopened outside snapshot")
            return real_read_text(path, *args, **kwargs)

        with (
            patch.dict(
                final.PROFILE_EXPECTATIONS,
                {("dev", "vstar"): expectation},
            ),
            patch.object(
                combined, "_FROZEN_COMBINED_INFERENCE_REVISION", FROZEN_REVISION,
            ),
            patch.object(phase4, "_support_prompt_sha256", return_value=PROMPT_SHA),
            patch.object(Path, "read_bytes", reject_second_path_read),
            patch.object(Path, "read_text", reject_second_text_read),
        ):
            final.prepare_selected_partition_from_paths(
                "vstar", disabled_path, enabled_path, profile="dev",
            )

    def test_bind_scores_frozen_vstar_and_topic_oracle_without_selector(self):
        prepared, *_ = self.prepare("vstar")
        with patch.object(
            combined, "select_unified_state",
            side_effect=AssertionError("selector reopened after frozen boundary"),
        ):
            scored = final._bind_prepared(prepared)

        self.assertEqual(scored.report["sample"], {
            "topics": 1, "cycles": 1,
            "topic_unit": "input row; HR four shuffles remain bound",
        })
        self.assertEqual(scored.report["p0"]["correct"], 0)
        self.assertEqual(scored.report["selected"]["correct"], 0)
        self.assertEqual(scored.report["oracle"]["correct"], 1)
        self.assertTrue(
            scored.report["oracle"]["candidate_generation_headroom_only"]
        )
        self.assertTrue(scored.report["oracle"]["not_used_for_selection"])
        self.assertFalse(scored.report["gate"]["formal_full_success"])
        self.assertEqual(
            scored.derived_manifest["decision_freeze"]["canonical_digest"],
            prepared.frozen_batch.canonical_digest,
        )
        self.assertEqual(
            scored.derived_manifest["selected_jsonl"]["sha256"],
            prepared.selected_jsonl_sha256,
        )
        self.assertTrue(
            scored.derived_manifest["trusted_annotation"]
            ["loaded_after_complete_partition_freeze"]
        )
        self.assertIn(
            "runtime_environment_sha256",
            scored.derived_manifest["execution_identity"],
        )
        validator_paths = {
            item["path"] for item in
            scored.derived_manifest["decision_freeze"]
            ["validator_source_manifest"]["files"]
        }
        self.assertTrue({
            "cvsearch/eval/phase3_zoom_oracle.py",
            "cvsearch/eval/phase4_expand_oracle.py",
            "cvsearch/eval/phase5_unified_selector.py",
            "cvsearch/eval/phase6_combined_selection.py",
        }.issubset(validator_paths))

    def test_canonical_hr_parser_scores_natural_language_not_string_equality(self):
        labels = ["C", "B", "A", "D"]
        outputs = ["C. green", "The option is B", "A", "D"]
        self.assertEqual(sum(left == right for left, right in zip(labels, outputs)), 2)
        correct, parsed = final._score_output(
            "hr-bench_4k", labels, outputs, ["unused"] * 4,
        )
        self.assertEqual(correct, 4)
        self.assertEqual(parsed, ("C", "B", "A", "D"))

    def test_annotation_or_prepared_tampering_rejects_without_reselection(self):
        prepared, *_ = self.prepare("vstar")
        annotation_path = Path(prepared.annotation_claim["path"])
        original = annotation_path.read_bytes()
        annotation_path.write_bytes(original + b" ")
        with self.assertRaises(RuntimeError):
            final._bind_prepared(prepared)
        annotation_path.write_bytes(original)

        forged_rows = tuple(copy.deepcopy(prepared.selected_rows))
        forged_rows[0]["output"] = 0
        forged = replace(prepared, selected_rows=forged_rows)
        with self.assertRaises(ValueError):
            final._bind_prepared(forged)

    def test_trusted_annotation_bytes_are_opened_once_after_freeze(self):
        prepared, *_ = self.prepare("vstar")
        annotation = Path(prepared.annotation_claim["path"])
        real_snapshot = final._snapshot_file
        reads = []

        def observe(path):
            if path == annotation:
                reads.append(path)
            return real_snapshot(path)

        with patch.object(final, "_snapshot_file", observe):
            final._bind_prepared(prepared)
        self.assertEqual(reads, [annotation])

    def test_strict_annotation_reader_rejects_duplicate_nan_and_symlink(self):
        for payload in (
            b'[{"question":"a","question":"b"}]',
            b'[{"value":NaN}]',
        ):
            with self.subTest(payload=payload):
                path = self.root / f"bad-{hashlib.sha256(payload).hexdigest()}.json"
                path.write_bytes(payload)
                claim = {
                    "path": str(path), "size": len(payload),
                    "file_sha256": hashlib.sha256(payload).hexdigest(),
                }
                with self.assertRaises(ValueError):
                    final._read_trusted_annotation(claim)
        target = self.root / "target.json"
        target.write_bytes(b"[]")
        link = self.root / "annotation-link.json"
        link.symlink_to(target)
        with self.assertRaises(ValueError):
            final._read_trusted_annotation({
                "path": str(link), "size": 2,
                "file_sha256": hashlib.sha256(b"[]").hexdigest(),
            })

    def test_file_identity_rejects_source_mutation_during_snapshot(self):
        source = self.root / "changing.jsonl"
        source.write_bytes(b'{"row":1}\n')
        real_read = final.os.read
        mutated = False

        def mutate_after_read(fd, size):
            nonlocal mutated
            payload = real_read(fd, size)
            if payload and not mutated:
                mutated = True
                with source.open("ab") as stream:
                    stream.write(b" ")
            return payload

        with (
            patch.object(final.os, "read", mutate_after_read),
            self.assertRaises(RuntimeError),
        ):
            final._file_identity(source)

    def test_atomic_bundle_publishes_exact_pair_and_never_overwrites(self):
        scored = final._bind_prepared(self.prepare("vstar")[0])
        bundle = self.root / "selected-bundle"

        final._write_selected_bundle_atomic(bundle, scored)

        self.assertEqual(
            (bundle / "selected.jsonl").read_bytes(),
            scored.prepared.selected_jsonl_bytes,
        )
        manifest_bytes = (bundle / "selection-manifest.json").read_bytes()
        self.assertTrue(manifest_bytes.endswith(b"\n"))
        self.assertEqual(json.loads(manifest_bytes), scored.derived_manifest)
        with self.assertRaises(FileExistsError):
            final._write_selected_bundle_atomic(bundle, scored)

    def test_atomic_bundle_rename_failure_leaves_no_visible_or_temporary_bundle(self):
        scored = final._bind_prepared(self.prepare("vstar")[0])
        bundle = self.root / "failed-bundle"
        with (
            patch.object(final.os, "rename", side_effect=OSError("rename failed")),
            self.assertRaises(OSError),
        ):
            final._write_selected_bundle_atomic(bundle, scored)
        self.assertFalse(bundle.exists())
        self.assertEqual(list(self.root.glob(".failed-bundle.tmp-*")), [])

    def test_atomic_bundle_fsync_failure_and_symlink_target_are_safe(self):
        scored = final._bind_prepared(self.prepare("vstar")[0])
        bundle = self.root / "fsync-failed-bundle"
        with (
            patch.object(final.os, "fsync", side_effect=OSError("fsync failed")),
            self.assertRaises(OSError),
        ):
            final._write_selected_bundle_atomic(bundle, scored)
        self.assertFalse(bundle.exists())
        self.assertEqual(list(self.root.glob(".fsync-failed-bundle.tmp-*")), [])

        target = self.root / "real-bundle-target"
        target.mkdir()
        symlink = self.root / "bundle-link"
        symlink.symlink_to(target, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            final._write_selected_bundle_atomic(symlink, scored)

    def test_full_fairness_is_strict_against_both_paired_and_locked_scores(self):
        cases = (
            (613, 613, 613, False),
            (610, 613, 613, False),
            (614, 614, 613, False),
            (613, 614, 613, True),
        )
        for p0, selected, locked, expected in cases:
            with self.subTest(p0=p0, selected=selected, locked=locked):
                self.assertEqual(
                    final._fairness_gate("full", p0, selected, locked), expected,
                )

    def test_full_hr_requires_200_topic_rows_not_800_rows(self):
        expectation = final.PROFILE_EXPECTATIONS[("full", "hr-bench_4k")]
        manifest = {
            "selected_partition": {
                "ordinals": list(range(800)), "rows": 800, "split": "all",
                "split_seed": 260809, "num_chunks": 1, "chunk_idx": 0,
            }
        }
        with self.assertRaises(ValueError):
            final._validate_partition(expectation, manifest, list(range(800)))

    def test_suite_requires_all_datasets_and_shared_frozen_identity(self):
        scored = {
            benchmark: final._bind_prepared(
                self.prepare(benchmark)[0]
            )
            for benchmark in ("vstar", "hr-bench_4k", "hr-bench_8k")
        }
        report = final._build_suite_report(scored, "dev")
        self.assertEqual(set(report["benchmarks"]), {
            "vstar", "hr-bench_4k", "hr-bench_8k",
        })
        self.assertFalse(report["gate"]["all_three_full_success"])

        with self.assertRaises(ValueError):
            final._build_suite_report({"vstar": scored["vstar"]}, "dev")
        forged_manifest = copy.deepcopy(scored["hr-bench_8k"].derived_manifest)
        forged_manifest["execution_identity"]["model_runtime_artifact_sha256"][
            "qwen"
        ] = "0" * 64
        forged = replace(
            scored["hr-bench_8k"], derived_manifest=forged_manifest,
        )
        with self.assertRaises(ValueError):
            final._build_suite_report({**scored, "hr-bench_8k": forged}, "dev")

        different_visible_devices = {}
        for index, (benchmark, value) in enumerate(scored.items()):
            manifest = copy.deepcopy(value.derived_manifest)
            manifest["execution_identity"]["environment_sha256"] = (
                f"{index + 1:x}" * 64
            )
            different_visible_devices[benchmark] = replace(
                value, derived_manifest=manifest,
            )
        final._build_suite_report(different_visible_devices, "dev")

        package_drift_manifest = copy.deepcopy(
            different_visible_devices["hr-bench_8k"].derived_manifest
        )
        package_drift_manifest["execution_identity"][
            "runtime_environment_sha256"
        ] = "f" * 64
        with self.assertRaises(ValueError):
            final._build_suite_report({
                **different_visible_devices,
                "hr-bench_8k": replace(
                    different_visible_devices["hr-bench_8k"],
                    derived_manifest=package_drift_manifest,
                ),
            }, "dev")

    def test_caller_resigned_prepared_cannot_cross_authoritative_label_boundary(self):
        prepared, *_ = self.prepare("vstar")
        forged_record = replace(
            prepared.frozen_batch[0],
            _decision_json=_canonical({
                "action": "EXPAND", "status": "selected_candidate",
                "output": 0, "stability_gain": 0.9,
            }),
        )
        unsigned_batch = replace(
            prepared.frozen_batch,
            records=(forged_record,), canonical_digest="0" * 64,
        )
        forged_batch = replace(
            unsigned_batch, canonical_digest=unsigned_batch.recompute_digest(),
        )
        rows, encoded, selected_sha, output_digest, input_digest = (
            final._selected_material(prepared.expectation, forged_batch)
        )
        unsigned_prepared = replace(
            prepared,
            frozen_batch=forged_batch,
            selected_rows=rows,
            selected_jsonl_bytes=encoded,
            selected_jsonl_sha256=selected_sha,
            selected_output_digest=output_digest,
            label_blind_input_sha256=input_digest,
            canonical_digest="0" * 64,
        )
        forged_prepared = replace(
            unsigned_prepared,
            canonical_digest=final.canonical_sha256(
                final._prepared_material(unsigned_prepared)
            ),
        )

        with self.assertRaises(TypeError):
            final.bind_and_score_trusted_annotation(forged_prepared)


if __name__ == "__main__":
    unittest.main()
