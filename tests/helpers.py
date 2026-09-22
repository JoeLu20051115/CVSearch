from dataclasses import dataclass
import hashlib
import json
from PIL import Image


@dataclass(frozen=True)
class FrozenRef:
    canonical_key: str
    bbox_original: tuple[int, int, int, int]
    depth: int
    render_level: int
    tree_scope: str
    crop_origin: tuple[int, int]
    source_image_key: str


def source_identity(image: Image.Image) -> dict[str, object]:
    return {
        "mode": image.mode,
        "size": [image.width, image.height],
        "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }


def source_key(image: Image.Image) -> str:
    return json.dumps(source_identity(image), sort_keys=True, separators=(",", ":"))


def candidate_key(bbox, depth=1, render_level=0):
    payload = {"bbox": list(bbox), "depth": depth, "render_level": render_level}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def event_for(
    image,
    candidates,
    *,
    event="stage_ready",
    scope="main",
    origin=(0, 0),
    ordinal=1,
    popped=(),
    selected=(),
    remaining=None,
):
    keys = [candidate["canonical_key"] for candidate in candidates]
    remaining = keys if remaining is None else list(remaining)
    snapshot = {
        "schema_version": 1,
        "event": event,
        "tree_scope": scope,
        "crop_origin": list(origin),
        "source_image_identity": source_identity(image),
        "search_call_ordinal": ordinal,
        "visual_cue": "sign",
        "stage": "Depth 1",
        "depth": 1,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "ordered_keys": keys,
        "popped_keys": list(popped),
        "selected_keys": list(selected),
        "remaining_keys": remaining,
    }
    refs = tuple(
        FrozenRef(
            canonical_key=candidate["canonical_key"],
            bbox_original=tuple(candidate["bbox_original"]),
            depth=candidate["depth"],
            render_level=candidate["render_level"],
            tree_scope=scope,
            crop_origin=origin,
            source_image_key=source_key(image),
        )
        for candidate in candidates
    )
    role = "selected_nodes" if event == "p0_selected" else "ordered_nodes"
    return {role: refs}, snapshot
