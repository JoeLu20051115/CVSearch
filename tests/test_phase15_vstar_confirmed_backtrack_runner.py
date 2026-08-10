from types import SimpleNamespace
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

from cvsearch.eval.phase15_vstar_confirmed_backtrack import B8_FROZEN_B5_RULE
from cvsearch.eval.phase15_vstar_confirmed_backtrack_runner import (
    _load_trusted_b5_freeze,
    main,
    _produce_record,
    _validate_b5_manifest_schema,
    _validate_partition_layout,
)
from cvsearch.eval.phase2_oracle import load_jsonl
from cvsearch.evidence_gap.provenance import canonical_sha256


def p0(output):
    return {
        "action": "P0", "output": output,
        "p0_stability": {"confidence": 0.2},
    }


def b5_record(ordinal, winners):
    observations = []
    for winner in winners:
        losses = [3.0, 3.0, 3.0]
        losses[winner] = 1.0
        observations.append({"winner": winner, "losses": losses})
    return {
        "source_ordinal": ordinal,
        "observations": observations,
        "decisions": {
            B8_FROZEN_B5_RULE: {
                "action": "LAZY",
                "status": "selected_generated_query_lazy_search",
                "output": winners[-1],
                "confidence": 0.5000000000000001,
                "confidence_gain": 0.3000000000000001,
                "path_consensus": 2 / 3,
            },
        },
    }


class VstarConfirmedBacktrackRunnerTests(unittest.TestCase):
    def test_produce_record_applies_b8_only_to_v2_p0_with_b5_evidence(self):
        pair = SimpleNamespace(
            ordinal=7, p0=p0(1), candidates=(),
            input_identity={"ordinal": 7}, extracted_digest="1" * 64,
        )
        record = _produce_record(
            pair, {"options": ["red", "green", "blue"]},
            b5_record(7, [1, 2, 2]),
        )
        self.assertEqual(record["v2_decision"]["action"], "P0")
        self.assertEqual(record["decision"]["action"], "VSTAR_BACKTRACK")
        self.assertEqual(record["decision"]["output"], 2)
        self.assertEqual(len(record["b5_record_sha256"]), 64)

    def test_partition_layout_requires_exact_disjoint_strided_union(self):
        eligible = [1, 3, 7, 8, 12]
        artifacts = [
            ({"partition": {"num_chunks": 2, "chunk_idx": 0,
                              "ordinals": [1, 7, 12],
                              "ordinals_sha256": canonical_sha256([1, 7, 12])}},
             [{"source_ordinal": 1}, {"source_ordinal": 7},
              {"source_ordinal": 12}]),
            ({"partition": {"num_chunks": 2, "chunk_idx": 1,
                              "ordinals": [3, 8],
                              "ordinals_sha256": canonical_sha256([3, 8])}},
             [{"source_ordinal": 3}, {"source_ordinal": 8}]),
        ]
        indexed = _validate_partition_layout(eligible, artifacts)
        self.assertEqual(set(indexed), set(eligible))
        with self.assertRaisesRegex(ValueError, "partition"):
            _validate_partition_layout(eligible, [artifacts[0], artifacts[0]])
        missing = [artifacts[0]]
        with self.assertRaisesRegex(ValueError, "partition"):
            _validate_partition_layout(eligible, missing)

    def test_partition_layout_accepts_only_exact_legacy_singleton(self):
        records = [{"source_ordinal": value} for value in (1, 3, 7)]
        indexed = _validate_partition_layout([1, 3, 7], [({}, records)])
        self.assertEqual(list(indexed), [1, 3, 7])
        with self.assertRaisesRegex(ValueError, "partition"):
            _validate_partition_layout([1, 3, 8], [({}, records)])

    def test_partition_layout_rejects_corrupt_or_incomplete_modern_metadata(self):
        records = [{"source_ordinal": value} for value in (1, 3, 7)]
        with self.assertRaisesRegex(ValueError, "partition"):
            _validate_partition_layout(
                [1, 3, 7], [({"partition": "corrupt"}, records)],
            )
        with self.assertRaisesRegex(ValueError, "partition"):
            _validate_partition_layout(
                [1, 3, 7], [({"partition": {
                    "num_chunks": 1, "chunk_idx": 0,
                    "ordinals": [1, 3, 7],
                }}, records)],
            )

    def test_manifest_schema_is_exact_for_legacy_and_modern_b5(self):
        legacy = load_jsonl(Path(
            "reproduction/evidence_gap/phase12_dev/b5_3f9d817/vstar/"
            "lazy.jsonl.lazy-manifest.json"
        ))[0]
        modern = load_jsonl(Path(
            "reproduction/evidence_gap/phase12_full/b5_3d2ee9e/vstar/chunk0/"
            "lazy.jsonl.lazy-manifest.json"
        ))[0]
        self.assertEqual(_validate_b5_manifest_schema(legacy), "legacy")
        self.assertEqual(_validate_b5_manifest_schema(modern), "modern")
        leaked = dict(modern, correct_answer=0)
        with self.assertRaisesRegex(ValueError, "schema"):
            _validate_b5_manifest_schema(leaked)
        corrupt = dict(modern, partition="corrupt")
        with self.assertRaisesRegex(ValueError, "partition"):
            _validate_b5_manifest_schema(corrupt)

    def test_only_precommitted_b5_freeze_reports_are_trusted(self):
        path = Path(
            "reproduction/evidence_gap/reports/"
            "phase14-b7-full-freeze.json"
        )
        trusted = _load_trusted_b5_freeze(path)
        self.assertEqual(trusted["stage"], "phase14-b7-complete-full-freeze")
        self.assertEqual(len(trusted["expected_artifacts"]), 2)
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "freeze.json"
            changed.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "trusted"):
                _load_trusted_b5_freeze(changed)

    def test_main_authenticates_dev_artifact_and_emits_complete_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "backtrack.jsonl"
            with redirect_stdout(io.StringIO()):
                status = main([
                    "--base-jsonl",
                    "reproduction/evidence_gap/phase7_dev/a1_2f09c6a/"
                    "vstar/base.jsonl",
                    "--combined-jsonl",
                    "reproduction/evidence_gap/phase7_dev/a2_dcd38a0/"
                    "vstar/combined.jsonl",
                    "--b5-jsonl",
                    "reproduction/evidence_gap/phase12_dev/b5_3f9d817/"
                    "vstar/lazy.jsonl",
                    "--b5-freeze-report",
                    "reproduction/evidence_gap/reports/"
                    "phase12-b5-generated-query-lazy-dev-freeze.json",
                    "--output", str(output),
                ])
            self.assertEqual(status, 0)
            records = load_jsonl(output)
            manifest = load_jsonl(Path(
                f"{output}.backtrack-manifest.json"
            ))[0]
            self.assertEqual(len(records), 37)
            self.assertEqual(
                [record["source_ordinal"] for record in records],
                [3, 4, 19, 24, 33, 39, 41, 48, 50, 62, 66, 78, 84,
                 85, 87, 94, 98, 101, 103, 104, 105, 116, 124, 132,
                 136, 138, 142, 145, 149, 153, 155, 158, 169, 174,
                 182, 186, 188],
            )
            self.assertEqual(manifest["selected_confirmed_backtracks"], 1)
            self.assertEqual(
                manifest["selected_output_sha256"],
                "e94f9a0726b34e7fa709fc41b218a7d498572a3933283ea37577d884affd3336",
            )
            self.assertEqual(
                manifest["output_sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
