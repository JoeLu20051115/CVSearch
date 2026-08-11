"""Content-addressed, fail-closed launch provenance for evidence-gap runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_PROCESSOR_NAMES = frozenset({
    "added_tokens.json", "chat_template.json", "chat_template.jinja",
    "config.json", "generation_config.json", "merges.txt",
    "preprocessor_config.json", "processor_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "vocab.txt",
})


def _strict_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_strict_json(value)).hexdigest()


def _file_sha256(path: Path) -> tuple[int, str]:
    before = path.stat()
    if not path.is_file():
        raise ValueError(f"provenance artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    after = path.stat()
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
    )
    if identity(before) != identity(after) or size != after.st_size:
        raise RuntimeError(f"provenance artifact changed while hashing: {path}")
    return size, digest.hexdigest()


def content_manifest(
    path: str | os.PathLike[str], *,
    allowed_symlink_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Hash one regular file or every regular file in a directory tree."""
    root = Path(path).expanduser()
    if root.is_symlink():
        raise ValueError(f"provenance paths may not be symlinks: {root}")
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        size, digest = _file_sha256(root)
        files = [{"path": root.name, "size": size, "sha256": digest}]
        kind = "file"
    elif root.is_dir():
        entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        special = [
            entry for entry in entries
            if not entry.is_symlink() and not entry.is_dir() and not entry.is_file()
        ]
        if special:
            raise ValueError(f"provenance trees may contain only regular files: {special[0]}")
        allowed = None if allowed_symlink_root is None else Path(allowed_symlink_root).resolve()
        files = []
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                continue
            extra = {}
            source = entry
            if entry.is_symlink():
                if allowed is None:
                    raise ValueError(f"provenance trees may not contain symlinks: {entry}")
                link_target = os.readlink(entry)
                link_before = entry.lstat()
                try:
                    source = entry.resolve(strict=True)
                except FileNotFoundError as error:
                    raise ValueError(f"broken provenance symlink: {entry}") from error
                if not source.is_relative_to(allowed):
                    raise ValueError(f"provenance symlink resolves outside allowed root: {entry}")
                if not source.is_file():
                    raise ValueError(f"provenance symlink must resolve to a regular file: {entry}")
                extra = {
                    "symlink_target": link_target,
                    "resolved_path": str(source),
                }
            if source.is_file():
                size, digest = _file_sha256(source)
                if entry.is_symlink():
                    link_after = entry.lstat()
                    if (
                        os.readlink(entry) != link_target
                        or (link_before.st_dev, link_before.st_ino, link_before.st_mtime_ns)
                        != (link_after.st_dev, link_after.st_ino, link_after.st_mtime_ns)
                    ):
                        raise RuntimeError(f"provenance symlink changed while hashing: {entry}")
                files.append({
                    "path": entry.relative_to(root).as_posix(),
                    "size": size,
                    "sha256": digest,
                    **extra,
                })
        if not files:
            raise ValueError(f"provenance directory is empty: {root}")
        kind = "directory"
    else:
        raise ValueError(f"provenance path is not a regular file or directory: {root}")
    material = {"kind": kind, "files": files}
    return {
        "path": str(root.resolve()), **material,
        "sha256": canonical_sha256(material),
    }


def _processor_manifest(model_path: Path, model_manifest: Mapping[str, Any]) -> dict[str, Any]:
    files = [
        dict(entry) for entry in model_manifest["files"]
        if Path(entry["path"]).name in _PROCESSOR_NAMES
        or Path(entry["path"]).name.startswith("tokenizer")
    ]
    material = {"kind": "processor_files", "files": files}
    return {
        "path": str(model_path.resolve()), **material,
        "sha256": canonical_sha256(material),
    }


def runtime_environment() -> dict[str, Any]:
    packages = {}
    required = (
        "einops", "hydra-core", "iopath", "matplotlib", "networkx", "numpy",
        "pillow", "scikit-image", "scikit-learn", "scipy", "sentencepiece",
        "spacy", "torch",
        "torchvision", "transformers", "sentence-transformers",
    )
    for distribution in required:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required distribution is not installed: {distribution}") from error
    for distribution in ("accelerate", "flash-attn", "safetensors"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    import torch

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": str(Path(sys.executable).resolve()),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cudnn": None if not torch.backends.cudnn.is_available() else torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def visible_gpu_uuids() -> list[str]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("at least one visible CUDA GPU is required")
    result = []
    for index in range(torch.cuda.device_count()):
        uuid = getattr(torch.cuda.get_device_properties(index), "uuid", None)
        if uuid is None or not str(uuid).strip():
            raise RuntimeError(f"CUDA device {index} does not expose a UUID")
        result.append(str(uuid))
    if len(result) != len(set(result)):
        raise RuntimeError("visible CUDA GPU UUIDs must be unique")
    return result


def _source_image_manifest(paths: Sequence[Path]) -> dict[str, Any]:
    lexical = [Path(path).expanduser() for path in paths]
    if any(path.is_symlink() for path in lexical):
        raise ValueError("source image paths may not be symlinks")
    resolved = [path.resolve() for path in lexical]
    if not resolved or len(resolved) != len(set(resolved)):
        raise ValueError("source image paths must be a nonempty unique sequence")
    files = []
    by_resolved = {path.resolve(): path for path in lexical}
    for path in sorted(resolved, key=lambda item: str(item)):
        manifest = content_manifest(by_resolved[path])
        if manifest["kind"] != "file":
            raise ValueError(f"source image must be a file: {path}")
        entry = dict(manifest["files"][0])
        entry["path"] = str(path)
        files.append(entry)
    material = {"kind": "selected_source_images", "files": files}
    return {**material, "sha256": canonical_sha256(material)}


def build_launch_manifest(
    *, benchmark: str, code_revision: str, code_manifest: Mapping[str, Any],
    loaded_config: Mapping[str, Any], config_source: Path,
    annotation_file: Path, selected_rows: Sequence[tuple[int, Mapping[str, Any]]],
    source_images: Sequence[Path], ic_examples: Path, model_path: Path,
    sam_path: Path, spacy_path: Path, clip_path: Path | None,
    split: str, split_seed: int, num_chunks: int, chunk_idx: int,
    environment: Mapping[str, Any], gpu_uuids: Sequence[str],
) -> dict[str, Any]:
    if not benchmark or not isinstance(benchmark, str):
        raise ValueError("benchmark must be a nonempty string")
    if len(code_revision) != 64 or any(char not in "0123456789abcdef" for char in code_revision):
        raise ValueError("code revision must be a lowercase SHA256")
    if not selected_rows:
        raise ValueError("selected annotation partition must not be empty")
    ordinals = [ordinal for ordinal, _ in selected_rows]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ordinals):
        raise ValueError("selected ordinals must be non-negative integers")
    if ordinals != sorted(set(ordinals)):
        raise ValueError("selected ordinals must be unique and sorted")
    if not gpu_uuids or any(not isinstance(value, str) or not value for value in gpu_uuids):
        raise ValueError("GPU UUIDs must be a nonempty string sequence")

    model_path = Path(model_path)
    model_store_root = (
        model_path.parent.parent
        if model_path.parent.name == "snapshots" else None
    )
    qwen = content_manifest(
        model_path, allowed_symlink_root=model_store_root,
    )
    partition_material = [
        {"ordinal": ordinal, "annotation": dict(row)} for ordinal, row in selected_rows
    ]
    partition = {
        "ordinals": ordinals,
        "rows_sha256": canonical_sha256(partition_material),
        "rows": len(selected_rows),
        "split": split,
        "split_seed": split_seed,
        "num_chunks": num_chunks,
        "chunk_idx": chunk_idx,
    }
    artifacts = {
        "annotation_file": content_manifest(annotation_file),
        "source_images": _source_image_manifest(source_images),
        "ic_examples": content_manifest(ic_examples),
        "qwen": qwen,
        "processor": _processor_manifest(Path(model_path), qwen),
        "sam": content_manifest(sam_path),
        "spacy": content_manifest(spacy_path),
    }
    if clip_path is not None:
        artifacts["clip"] = content_manifest(clip_path)
    config = {
        "loaded": dict(loaded_config),
        "loaded_sha256": canonical_sha256(dict(loaded_config)),
        "source": content_manifest(config_source),
    }
    return {
        "schema_version": 1,
        "benchmark": benchmark,
        "code": {
            "revision": code_revision,
            "manifest": dict(code_manifest),
            "manifest_sha256": canonical_sha256(dict(code_manifest)),
        },
        "config": config,
        "selected_partition": partition,
        "artifacts": artifacts,
        "environment": dict(environment),
        "hardware": {"gpu_uuids": list(gpu_uuids)},
    }


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def write_or_validate_manifest(
    path: str | os.PathLike[str], manifest: Mapping[str, Any], *,
    resume: bool, force: bool,
) -> None:
    target = Path(path)
    if resume and force:
        raise ValueError("resume and force are mutually exclusive")
    if target.is_symlink():
        raise ValueError("launch manifest path may not be a symlink")
    encoded = _strict_json(dict(manifest)) + b"\n"
    if resume:
        if not target.is_file():
            raise FileNotFoundError(target)
        try:
            existing = json.loads(
                target.read_text(encoding="utf-8"), parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("existing launch manifest is invalid") from error
        if _strict_json(existing) != encoded.rstrip(b"\n"):
            raise ValueError("launch manifest does not match current content")
        return
    if target.exists() and not force:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
