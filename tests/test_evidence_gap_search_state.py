from dataclasses import FrozenInstanceError, dataclass
import hashlib
import json
import unittest

from PIL import Image

from cvsearch.evidence_gap.search_state import (
    NextNoOpReason,
    SearchStateCollector,
)
from cvsearch.evidence_gap.types import P0Anchor


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


def candidate_snapshot(bbox, *, posterior=0.5, depth=1, render_level=0):
    key = candidate_key(bbox, depth, render_level)
    return {
        "canonical_key": key,
        "bbox_original": list(bbox),
        "parent_key": None,
        "child_keys": [],
        "depth": depth,
        "render_level": render_level,
        "source": None,
        "stage_rank": 0,
        "prior_prob": 0.5,
        "fast_confidence": 0.5,
        "posterior_score": posterior,
        "is_evaluated": True,
        "answering_confidence": -1.0,
    }


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


class SearchStateCollectorTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (8, 6))
        for y in range(self.image.height):
            for x in range(self.image.width):
                self.image.putpixel((x, y), (x * 20, y * 30, x + y))

    def observe(self, collector, candidates, **kwargs):
        refs, snapshot = event_for(self.image, candidates, **kwargs)
        collector(refs, snapshot)

    def test_first_seen_main_descriptor_wins_cropped_collision_and_native_ties(self):
        first = candidate_snapshot((2, 1, 2, 2), posterior=0.8)
        tied = candidate_snapshot((4, 1, 2, 2), posterior=0.8)
        missing = candidate_snapshot((0, 4, 2, 2), posterior=None)
        collector = SearchStateCollector(self.image)
        self.observe(collector, [first, tied, missing])

        duplicate = candidate_snapshot((2, 1, 2, 2), posterior=0.99)
        self.observe(
            collector,
            [duplicate],
            scope="cropped",
            origin=(1, 1),
            ordinal=2,
        )

        decisions = [collector.next_candidate() for _ in range(3)]
        self.assertEqual(
            [decision.candidate.canonical_key for decision in decisions],
            [first["canonical_key"], tied["canonical_key"], missing["canonical_key"]],
        )
        self.assertEqual(decisions[0].candidate.tree_scope, "main")
        self.assertEqual(decisions[0].candidate.posterior_score, 0.8)
        self.assertEqual(decisions[0].candidate.first_seen_ordinal, 0)

    def test_finished_popped_selected_and_next_current_are_all_visited(self):
        popped = candidate_snapshot((0, 0, 2, 2), posterior=0.9)
        selected = candidate_snapshot((2, 0, 2, 2), posterior=0.8)
        remaining = candidate_snapshot((4, 0, 2, 2), posterior=0.7)
        collector = SearchStateCollector(self.image)
        self.observe(collector, [popped, selected, remaining])
        self.observe(
            collector,
            [popped, selected, remaining],
            event="stage_finished",
            popped=(popped["canonical_key"],),
            selected=(selected["canonical_key"],),
            remaining=(remaining["canonical_key"],),
        )

        decision = collector.next_candidate()
        self.assertEqual(decision.candidate.canonical_key, remaining["canonical_key"])
        exhausted = collector.next_candidate()
        self.assertIsNone(exhausted.candidate)
        self.assertEqual(exhausted.no_op_reason, NextNoOpReason.ALL_CANDIDATES_VISITED)
        self.assertEqual(
            set(collector.to_dict()["visited_keys"]),
            {popped["canonical_key"], selected["canonical_key"], remaining["canonical_key"]},
        )

    def test_invalid_geometry_empty_queue_and_visited_only_have_distinct_no_op_reasons(self):
        empty = SearchStateCollector(self.image).next_candidate()
        self.assertEqual(empty.no_op_reason, NextNoOpReason.EMPTY_QUEUE)

        invalid_collector = SearchStateCollector(self.image)
        invalid = candidate_snapshot((1, 1, 0, 2), posterior=0.9)
        self.observe(invalid_collector, [invalid])
        invalid_result = invalid_collector.next_candidate()
        self.assertEqual(invalid_result.no_op_reason, NextNoOpReason.NO_VALID_CANDIDATES)
        self.assertEqual(invalid_collector.to_dict()["rejected_candidates"][0]["reason"],
                         "non_positive_bbox")

        visited_collector = SearchStateCollector(self.image)
        valid = candidate_snapshot((1, 1, 2, 2), posterior=0.9)
        self.observe(
            visited_collector,
            [valid],
            event="stage_finished",
            popped=(valid["canonical_key"],),
            remaining=(),
        )
        visited_result = visited_collector.next_candidate()
        self.assertEqual(visited_result.no_op_reason, NextNoOpReason.ALL_CANDIDATES_VISITED)

    def test_cropped_candidate_renders_original_geometry_and_pixels_on_fresh_nodes(self):
        candidate = candidate_snapshot((3, 2, 2, 2), posterior=0.7)
        collector = SearchStateCollector(self.image)
        self.observe(
            collector,
            [candidate],
            scope="cropped",
            origin=(2, 1),
            ordinal=3,
        )

        selected = collector.next_candidate().candidate
        first_node = selected.render_node
        second_node = selected.render_node
        self.assertIsNot(first_node, second_node)
        first_node.state.original_image_pil.putpixel((3, 2), (255, 0, 0))
        self.assertEqual(second_node.state.original_image_pil.getpixel((3, 2)),
                         self.image.getpixel((3, 2)))
        self.assertEqual(first_node.state.bbox, [3, 2, 2, 2])
        self.assertEqual(
            list(second_node.state.original_image_pil.crop((3, 2, 5, 4)).getdata()),
            list(self.image.crop((3, 2, 5, 4)).getdata()),
        )
        self.assertEqual(first_node.depth, 0)
        self.assertEqual(first_node.children, [])
        self.assertFalse(hasattr(first_node, "posterior_score"))

    def test_trace_is_strict_json_and_never_contains_render_objects(self):
        candidate = candidate_snapshot((1, 1, 2, 2), posterior=0.7)
        collector = SearchStateCollector(self.image)
        self.observe(collector, [candidate])
        selected = collector.next_candidate().candidate
        _ = selected.render_node

        payload = collector.to_dict()
        encoded = json.dumps(payload, allow_nan=False)
        self.assertEqual(json.loads(encoded), payload)
        self.assertNotIn("original_image_pil", encoded)
        self.assertNotIn("render_node", encoded)
        payload["source_image_identity"]["size"][0] = -1
        self.assertEqual(collector.to_dict()["source_image_identity"]["size"], [8, 6])

    def test_source_identity_mismatch_is_rejected_before_queue_mutation(self):
        candidate = candidate_snapshot((1, 1, 2, 2), posterior=0.7)
        refs, snapshot = event_for(self.image, [candidate])
        snapshot["source_image_identity"]["pixel_sha256"] = "f" * 64
        collector = SearchStateCollector(self.image)

        with self.assertRaisesRegex(ValueError, "source image identity"):
            collector(refs, snapshot)
        self.assertEqual(collector.next_candidate().no_op_reason, NextNoOpReason.EMPTY_QUEUE)

    def test_exact_support_view_returns_none_instead_of_substituting_another_candidate(self):
        candidate = candidate_snapshot((1, 1, 2, 2), posterior=0.7)
        collector = SearchStateCollector(self.image)
        self.observe(collector, [candidate])

        view = collector.support_view((candidate["canonical_key"],))
        self.assertEqual(tuple(item.canonical_key for item in view), (candidate["canonical_key"],))
        self.assertIsNone(collector.support_view(("missing-canonical-key",)))
        self.assertIsNone(collector.support_view(()))


class P0AnchorTest(unittest.TestCase):
    def test_anchor_freezes_outputs_and_support_view_and_serializes_descriptors_only(self):
        image = Image.new("RGB", (4, 4), "blue")
        candidate = candidate_snapshot((1, 1, 2, 2), posterior=0.7)
        refs, snapshot = event_for(image, [candidate])
        collector = SearchStateCollector(image)
        collector(refs, snapshot)
        view = collector.support_view((candidate["canonical_key"],))
        emitted = ["B"]
        raw = ["C"]

        anchor = P0Anchor(
            emitted_answer=emitted,
            cvsearch_raw=raw,
            producing_phase="cvsearch_raw",
            node_keys=(candidate["canonical_key"],),
            support_view=view,
        )
        emitted[0] = "mutated"
        raw[0] = "mutated"
        returned = anchor.emitted_answer
        returned[0] = "also-mutated"

        self.assertEqual(anchor.emitted_answer, ["B"])
        self.assertEqual(anchor.cvsearch_raw, ["C"])
        self.assertEqual(anchor.support_view, view)
        first_support = anchor.support_view
        first_support[0].render_node.state.original_image_pil.putpixel((1, 1), (255, 0, 0))
        second_support = anchor.support_view
        self.assertIsNot(first_support, second_support)
        self.assertEqual(
            second_support[0].render_node.state.original_image_pil.getpixel((1, 1)),
            image.getpixel((1, 1)),
        )
        with self.assertRaises(FrozenInstanceError):
            anchor.producing_phase = "search"
        payload = anchor.to_dict()
        self.assertEqual(payload["support_view"][0]["canonical_key"], candidate["canonical_key"])
        encoded = json.dumps(payload, allow_nan=False)
        self.assertNotIn("original_image_pil", encoded)
        self.assertNotIn("render_node", encoded)

    def test_missing_support_bundle_remains_explicitly_unavailable(self):
        for phase in ("quick", "fast"):
            with self.subTest(phase=phase):
                anchor = P0Anchor(
                    emitted_answer="A",
                    cvsearch_raw="B",
                    producing_phase=phase,
                    node_keys=(),
                    support_view=None,
                )
                self.assertEqual(anchor.producing_phase, phase)
                self.assertIsNone(anchor.support_view)
                self.assertIsNone(anchor.to_dict()["support_view"])

    def test_anchor_rejects_mismatched_support_keys_and_unknown_phase(self):
        image = Image.new("RGB", (4, 4), "blue")
        candidate = candidate_snapshot((1, 1, 2, 2), posterior=0.7)
        refs, snapshot = event_for(image, [candidate])
        collector = SearchStateCollector(image)
        collector(refs, snapshot)
        view = collector.support_view((candidate["canonical_key"],))

        with self.assertRaisesRegex(ValueError, "support view keys"):
            P0Anchor("A", "A", "search", ("different",), view)
        with self.assertRaisesRegex(ValueError, "producing_phase"):
            P0Anchor("A", "A", "", (), None)

        @dataclass
        class MutableSupportDescriptor:
            canonical_key: str

            def to_dict(self):
                return {"canonical_key": self.canonical_key}

        with self.assertRaisesRegex(TypeError, "immutable render descriptors"):
            P0Anchor(
                "A", "A", "search", (candidate["canonical_key"],),
                (MutableSupportDescriptor(candidate["canonical_key"]),),
            )


if __name__ == "__main__":
    unittest.main()
