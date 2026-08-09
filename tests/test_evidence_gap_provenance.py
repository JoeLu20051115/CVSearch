import copy
import importlib.metadata
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvsearch.evidence_gap.provenance import (
    build_launch_manifest,
    canonical_sha256,
    content_manifest,
    runtime_environment,
    write_or_validate_manifest,
)


class ProvenanceTest(unittest.TestCase):
    def test_runtime_environment_binds_sentence_transformers(self):
        real_version = importlib.metadata.version

        def version(distribution):
            if distribution == "sentence-transformers":
                return "bound-sentence-transformers"
            return real_version(distribution)

        with patch(
            "cvsearch.evidence_gap.provenance.importlib.metadata.version",
            side_effect=version,
        ):
            environment = runtime_environment()
        self.assertEqual(
            environment["packages"]["sentence-transformers"],
            "bound-sentence-transformers",
        )

    def test_content_manifest_is_sorted_recursive_and_changes_with_any_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "z.bin").write_bytes(b"z")
            (root / "nested" / "a.bin").write_bytes(b"a")
            first = content_manifest(root)
            self.assertEqual(
                [entry["path"] for entry in first["files"]],
                ["nested/a.bin", "z.bin"],
            )
            (root / "nested" / "a.bin").write_bytes(b"changed")
            second = content_manifest(root)
            self.assertNotEqual(first["sha256"], second["sha256"])

    def test_content_manifest_rejects_symlinks_and_special_or_empty_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaises(ValueError):
                content_manifest(empty)
            target = root / "target"
            target.write_bytes(b"x")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                content_manifest(link)

    def test_huggingface_snapshot_symlinks_are_hashed_only_inside_explicit_store(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            store = Path(directory) / "models--Qwen--fixture"
            snapshot = store / "snapshots" / ("a" * 40)
            blobs = store / "blobs"
            snapshot.mkdir(parents=True)
            blobs.mkdir()
            blob = blobs / ("b" * 64)
            blob.write_bytes(b"weights")
            (snapshot / "model.safetensors").symlink_to(blob)

            manifest = content_manifest(snapshot, allowed_symlink_root=store)
            entry = manifest["files"][0]
            self.assertEqual(entry["path"], "model.safetensors")
            self.assertEqual(entry["symlink_target"], str(blob))
            self.assertEqual(entry["resolved_path"], str(blob.resolve()))
            original = manifest["sha256"]
            blob.write_bytes(b"changed weights")
            self.assertNotEqual(
                content_manifest(snapshot, allowed_symlink_root=store)["sha256"],
                original,
            )

            escaped = snapshot / "escaped.bin"
            escaped.symlink_to(Path(outside) / "outside.bin")
            (Path(outside) / "outside.bin").write_bytes(b"outside")
            with self.assertRaisesRegex(ValueError, "outside"):
                content_manifest(snapshot, allowed_symlink_root=store)
            escaped.unlink()
            (snapshot / "broken.bin").symlink_to(blobs / "missing")
            with self.assertRaisesRegex(ValueError, "broken"):
                content_manifest(snapshot, allowed_symlink_root=store)

    def test_launch_manifest_binds_selected_rows_images_artifacts_environment_and_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {}
            for name in ("qwen", "spacy"):
                path = root / name
                path.mkdir()
                (path / "data.bin").write_bytes(name.encode())
                artifacts[name] = path
            for name in ("sam", "config", "annotation", "ic"):
                path = root / f"{name}.bin"
                path.write_bytes(name.encode())
                artifacts[name] = path
            image = root / "image.jpg"
            image.write_bytes(b"image")
            kwargs = {
                "benchmark": "vstar", "code_revision": "a" * 64,
                "code_manifest": {"files": [{"path": "runner.py", "sha256": "b" * 64}]},
                "loaded_config": {"config_id": "x"}, "config_source": artifacts["config"],
                "annotation_file": artifacts["annotation"],
                "selected_rows": [(3, {"input_image": "image.jpg", "question": "q"})],
                "source_images": [image], "ic_examples": artifacts["ic"],
                "model_path": artifacts["qwen"], "sam_path": artifacts["sam"],
                "spacy_path": artifacts["spacy"], "clip_path": None,
                "split": "dev", "split_seed": 260809, "num_chunks": 1, "chunk_idx": 0,
                "environment": {"python": "3.11", "torch": "2.7"},
                "gpu_uuids": ["GPU-test"],
            }
            baseline = build_launch_manifest(**kwargs)
            self.assertEqual(baseline["selected_partition"]["ordinals"], [3])
            self.assertEqual(baseline["hardware"]["gpu_uuids"], ["GPU-test"])
            for key, path in (
                ("source image", image), ("model", artifacts["qwen"] / "data.bin"),
                ("SAM", artifacts["sam"]), ("spaCy", artifacts["spacy"] / "data.bin"),
                ("annotation", artifacts["annotation"]), ("config", artifacts["config"]),
            ):
                with self.subTest(key=key):
                    original = path.read_bytes()
                    path.write_bytes(original + b"!")
                    changed = build_launch_manifest(**kwargs)
                    self.assertNotEqual(
                        canonical_sha256(baseline), canonical_sha256(changed),
                    )
                    path.write_bytes(original)
            changed_environment = dict(kwargs)
            changed_environment["environment"] = {"python": "3.12", "torch": "2.7"}
            self.assertNotEqual(
                canonical_sha256(baseline),
                canonical_sha256(build_launch_manifest(**changed_environment)),
            )
            changed_gpu = dict(kwargs)
            changed_gpu["gpu_uuids"] = ["GPU-other"]
            self.assertNotEqual(
                canonical_sha256(baseline),
                canonical_sha256(build_launch_manifest(**changed_gpu)),
            )

            image_link = root / "image-link.jpg"
            image_link.symlink_to(image)
            linked = dict(kwargs, source_images=[image_link])
            with self.assertRaisesRegex(ValueError, "symlink"):
                build_launch_manifest(**linked)

    def test_manifest_resume_is_exact_and_force_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.manifest.json"
            first = {"schema_version": 1, "value": "a"}
            second = {"schema_version": 1, "value": "b"}
            write_or_validate_manifest(path, first, resume=False, force=False)
            with self.assertRaises(FileExistsError):
                write_or_validate_manifest(path, first, resume=False, force=False)
            write_or_validate_manifest(path, first, resume=True, force=False)
            with self.assertRaisesRegex(ValueError, "manifest"):
                write_or_validate_manifest(path, second, resume=True, force=False)
            write_or_validate_manifest(path, second, resume=False, force=True)
            self.assertEqual(path.read_text(encoding="utf-8").count("\n"), 1)


if __name__ == "__main__":
    unittest.main()
