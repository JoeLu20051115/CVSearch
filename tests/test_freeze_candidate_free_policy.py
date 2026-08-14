import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from cvsearch.eval.freeze_candidate_free_policy import verified_pairwise_manifest


class CandidateFreePolicyFreezeTest(unittest.TestCase):
    def test_pairwise_observation_is_bound_to_its_manifest_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = root / "observations.jsonl"
            observations.write_text('{"ordinal":0}\n', encoding="utf-8")
            digest = hashlib.sha256(observations.read_bytes()).hexdigest()
            manifest_path = Path(f"{observations}.pairwise-manifest.json")
            manifest_path.write_text(json.dumps({
                "schema_version": 1,
                "artifact_kind": "pairwise-uncertainty-verifier-observations",
                "output_sha256": digest,
                "records": 1,
                "input_bindings": [],
            }), encoding="utf-8")

            manifest = verified_pairwise_manifest(observations)

            self.assertEqual(manifest["output_sha256"], digest)
            observations.write_text('{"ordinal":1}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                verified_pairwise_manifest(observations)

    def test_pairwise_manifest_requires_the_exact_artifact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            observations = Path(directory) / "observations.jsonl"
            observations.write_text("", encoding="utf-8")
            manifest_path = Path(f"{observations}.pairwise-manifest.json")
            manifest_path.write_text(json.dumps({
                "schema_version": 1,
                "artifact_kind": "wrong",
                "output_sha256": hashlib.sha256(b"").hexdigest(),
                "records": 0,
                "input_bindings": [],
            }), encoding="utf-8")

            with self.assertRaises(ValueError):
                verified_pairwise_manifest(observations)


if __name__ == "__main__":
    unittest.main()
