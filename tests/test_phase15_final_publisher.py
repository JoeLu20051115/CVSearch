import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from cvsearch.eval.phase15_final_publisher import (
    PreparedPhase15Suite,
    FrozenFileSnapshot,
    _validate_decision,
    prepare_phase15_suite,
    publish_phase15_suite,
)
from cvsearch.evidence_gap.provenance import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / (
    "reproduction/evidence_gap/reports/"
    "phase15-b8-b7-complete-full-freeze.json"
)
ARTIFACTS = {
    "vstar": ROOT / (
        "reproduction/evidence_gap/phase15_full/b8_df8b6b7/"
        "vstar/backtrack.jsonl"
    ),
    "hr-bench_4k": ROOT / (
        "reproduction/evidence_gap/phase14_full/b7_01ff5f1/"
        "hr-bench_4k/normalized.jsonl"
    ),
    "hr-bench_8k": ROOT / (
        "reproduction/evidence_gap/phase14_full/b7_01ff5f1/"
        "hr-bench_8k/normalized.jsonl"
    ),
}
ANNOTATIONS = {
    "vstar": ROOT / "datasets/hr_data/vstar/annotation_vstar.json",
    "hr-bench_4k": ROOT / (
        "datasets/hr_data/hr-bench_4k/annotation_hr-bench_4k.json"
    ),
    "hr-bench_8k": ROOT / (
        "datasets/hr_data/hr-bench_8k/annotation_hr-bench_8k.json"
    ),
}


class Phase15FinalPublisherTests(unittest.TestCase):
    def test_prepares_exact_scores_and_atomically_publishes_complete_suite(self):
        prepared = prepare_phase15_suite(
            FREEZE, artifacts=ARTIFACTS, annotations=ANNOTATIONS,
        )
        self.assertEqual(
            {name: bundle.score["correct"] for name, bundle in prepared.bundles.items()},
            {"vstar": 173, "hr-bench_4k": 620, "hr-bench_8k": 628},
        )
        self.assertTrue(prepared.manifest["gate"]["all_three_strictly_improved"])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "selected"
            publish_phase15_suite(target, prepared)
            for benchmark, bundle in prepared.bundles.items():
                selected = target / benchmark / "selected.jsonl"
                manifest = target / benchmark / "selection-manifest.json"
                self.assertEqual(
                    hashlib.sha256(selected.read_bytes()).hexdigest(),
                    bundle.manifest["selected_jsonl"]["sha256"],
                )
                self.assertTrue(manifest.is_file())
            self.assertTrue((target / "suite-manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                publish_phase15_suite(target, prepared)

    def test_prepared_manifests_are_immutable_and_gate_is_enforced(self):
        prepared = prepare_phase15_suite(
            FREEZE, artifacts=ARTIFACTS, annotations=ANNOTATIONS,
        )
        exposed = prepared.manifest
        exposed["gate"]["all_three_strictly_improved"] = False
        self.assertTrue(prepared.manifest["gate"]["all_three_strictly_improved"])
        changed = prepared.manifest
        changed["gate"]["all_three_strictly_improved"] = False
        changed_bytes = json.dumps(
            changed, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
        invalid = PreparedPhase15Suite(
            prepared.bundle_values, changed_bytes, prepared.freeze_snapshot,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "gate"):
                publish_phase15_suite(Path(directory) / "selected", invalid)

    def test_decision_schema_rejects_unknown_nested_metadata(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            _validate_decision("vstar", {
                "action": "VSTAR_BACKTRACK",
                "status": "selected_confirmed_backtrack",
                "output": 0,
                "correct_answer": 0,
            })

    def test_publish_reauthenticates_caller_forged_snapshot_bytes(self):
        prepared = prepare_phase15_suite(
            FREEZE, artifacts=ARTIFACTS, annotations=ANNOTATIONS,
        )
        bundle = prepared.bundle_values[0]
        lines = bundle.source_snapshot.data.decode().splitlines()
        first = json.loads(lines[0])
        first["correct_answer"] = 0
        lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
        forged_data = ("\n".join(lines) + "\n").encode()
        forged_snapshot = FrozenFileSnapshot(
            bundle.source_snapshot.path, forged_data,
            hashlib.sha256(forged_data).hexdigest(),
            bundle.source_snapshot.stat_seal,
        )
        forged_bundle = replace(bundle, source_snapshot=forged_snapshot)
        forged = replace(
            prepared,
            bundle_values=(forged_bundle, *prepared.bundle_values[1:]),
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "trusted"):
                publish_phase15_suite(Path(directory) / "selected", forged)

    def test_publish_rebuilds_bundle_manifest_instead_of_trusting_metadata(self):
        prepared = prepare_phase15_suite(
            FREEZE, artifacts=ARTIFACTS, annotations=ANNOTATIONS,
        )
        bundle = prepared.bundle_values[0]
        forged_manifest = bundle.manifest
        forged_manifest["selector_id"] = "caller-forged-selector"
        forged_bundle = replace(
            bundle,
            manifest_bytes=json.dumps(
                forged_manifest, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode(),
        )
        forged_suite_manifest = prepared.manifest
        forged_suite_manifest["bundle_manifest_sha256"][bundle.benchmark] = (
            canonical_sha256(forged_manifest)
        )
        forged = replace(
            prepared,
            bundle_values=(forged_bundle, *prepared.bundle_values[1:]),
            manifest_bytes=json.dumps(
                forged_suite_manifest, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode(),
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "authenticated inputs"):
                publish_phase15_suite(Path(directory) / "selected", forged)

    def test_publish_rebuilds_suite_manifest_instead_of_trusting_metadata(self):
        prepared = prepare_phase15_suite(
            FREEZE, artifacts=ARTIFACTS, annotations=ANNOTATIONS,
        )
        forged_manifest = prepared.manifest
        forged_manifest["selector_id"] = "caller-forged-selector"
        forged = replace(
            prepared,
            manifest_bytes=json.dumps(
                forged_manifest, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode(),
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "gate"):
                publish_phase15_suite(Path(directory) / "selected", forged)

    def test_rejects_modified_freeze_before_prediction_use(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "freeze.json"
            changed.write_bytes(FREEZE.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "trusted"):
                prepare_phase15_suite(
                    changed, artifacts=ARTIFACTS, annotations=ANNOTATIONS,
                )

    def test_rejects_symlinked_trust_root(self):
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "freeze.json"
            link.symlink_to(FREEZE)
            with self.assertRaisesRegex(ValueError, "regular file"):
                prepare_phase15_suite(
                    link, artifacts=ARTIFACTS, annotations=ANNOTATIONS,
                )


if __name__ == "__main__":
    unittest.main()
