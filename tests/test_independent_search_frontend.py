import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from qavs.independent_search.config import ProposalConfig


class FakeZoom:
    def __init__(self, targets=("sign",)):
        self.targets = list(targets)
        self.calls = 0

    def generate_visual_cues_using_ic(self, ic_examples, question):
        self.calls += 1
        return list(self.targets)


class FakeSam:
    def __init__(self, boxes, scores=None, *, target_ids=(17,)):
        self.calls = 0
        self.boxes = boxes
        self.scores = scores
        self.target_ids = list(target_ids)

    def batch_inference(self, image, targets):
        self.calls += 1
        result = {"boxes": np.asarray(self.boxes, dtype=np.float32)}
        if self.scores is not None:
            result["scores"] = np.asarray(self.scores, dtype=np.float32)
        return (
            {"vision_features": np.ones((1, 4, 4, 4), dtype=np.float32)},
            {self.target_ids[0]: result},
            self.target_ids,
        )


def _proposal_config(**updates):
    values = {
        "dedup_iou": 0.9,
        "min_count": 1,
        "min_spatial_coverage": 0.005,
        "active_top_k": 4,
    }
    values.update(updates)
    return ProposalConfig.from_mapping(values)


def _fake_tree(image):
    from qavs.models.tree import NodeA, NodeState

    root = NodeA(NodeState(image, [0, 0, image.width, image.height]))
    root.is_root = True
    root.search_source = "global"
    left = NodeA(NodeState(image, [0, 0, image.width // 2, image.height]), root)
    left.depth = 1
    left.search_source = "fine"
    left.complexity = 0.8
    root.add_child(left)
    return SimpleNamespace(root=root, max_depth=1)


@pytest.mark.parametrize(("question", "target"), [
    ("Is there a backpack in the image?\nAnswer yes or no only.", "backpack"),
    ("Is there an orange in the picture?", "orange"),
    ("Is there an umbrella in the imange?\nAnswer yes or no only.", "umbrella"),
    ("Is there any rice in this photo?", "rice"),
    ("Are there any red cups in the image?", "red cups"),
    ("Are there some black-and-white cats in this picture?", "black-and-white cats"),
    ("  IS THERE A Red   backpack IN THE IMAGE?  ", "Red backpack"),
    ("Is there a parking meter in the image?", "parking meter"),
    ("Is there a dining table in the image?", "dining table"),
    ("Is there a hot dog in the image?", "hot dog"),
    ("Is there a tie in the image?", "tie"),
    ("Is there a train in the image?", "train"),
])
def test_presence_targets_preserve_objects_without_cue_generation(monkeypatch, question, target):
    from qavs.independent_search import frontend

    monkeypatch.setattr(frontend, "include_pronouns", lambda nlp, value: False)
    zoom = FakeZoom(("image",))

    assert frontend.presence_question_target(question) == target
    assert frontend._targets(zoom, object(), {}, question) == (target,)
    assert zoom.calls == 0


@pytest.mark.parametrize("question", [
    "What color is the backpack in the image?",
    "Is the backpack red in the image?",
    "How many backpacks are there in the image?",
    "Is there a backpack and a suitcase in the image?",
    "Is there a backpack or a suitcase in the image?",
    "Is there a backpack next to the suitcase in the image?",
    "Is there a man holding an umbrella in the image?",
    "Is there a cat chasing mice in the image?",
    "Is there a man reading in the image?",
    "Is there a dog running in the image?",
    "Is there a cat painted red in the image?",
    "Is there a man pushing the cart in the image?",
    "Is there a man who is wearing a hat in the image?",
    "Is there a picture of a dog in the image?",
    "Is there not a backpack in the image?",
    "Are there at least two backpacks in the image?",
    "Is there a backpack in the image? Is it red?",
])
def test_presence_shortcut_defers_other_questions_to_existing_cues(monkeypatch, question):
    from qavs.independent_search import frontend

    monkeypatch.setattr(frontend, "include_pronouns", lambda nlp, value: False)
    zoom = FakeZoom(("red backpack", "suitcase"))

    assert frontend._targets(zoom, object(), {}, question) == ("red backpack", "suitcase")
    assert zoom.calls == 1
    assert frontend.presence_question_target(question) is None


@pytest.mark.parametrize("generic", ["image", "picture", "photo", " The IMAGE. "])
def test_generic_generated_target_uses_question_object_fallback(monkeypatch, generic):
    from qavs.independent_search import frontend

    monkeypatch.setattr(frontend, "include_pronouns", lambda nlp, value: False)
    monkeypatch.setattr(frontend, "extract_visual_objects", lambda nlp, question: ["red backpack"])

    assert frontend._targets(
        FakeZoom((generic,)), object(), {}, "What color is the backpack?",
    ) == ("red backpack",)


def test_generic_targets_are_filtered_without_dropping_object_modifiers(monkeypatch):
    from qavs.independent_search import frontend

    monkeypatch.setattr(frontend, "include_pronouns", lambda nlp, value: False)
    assert frontend._targets(
        FakeZoom(("image", "picture frame", "red backpack", "red backpack")),
        object(), {}, "Describe the framed picture and the backpack.",
    ) == ("picture frame", "red backpack")


def test_generic_fallback_targets_use_last_resort_evidence(monkeypatch):
    from qavs.independent_search import frontend

    monkeypatch.setattr(frontend, "extract_visual_objects", lambda nlp, question: ["image", "the photo"])

    assert frontend._targets(
        FakeZoom(()), object(), {}, "Describe the image.",
    ) == ("visible question evidence",)


def test_frontend_emits_root_and_sam_proposals_before_sgap(monkeypatch):
    from qavs.independent_search import frontend

    builds = []
    monkeypatch.setattr(frontend, "_build_tree", lambda features, image: builds.append(1))
    monkeypatch.setattr(frontend, "include_pronouns", lambda nlp, target: False)
    result = frontend.materialize_frontend(
        image=Image.new("RGB", (32, 24)),
        question="What color is the sign?",
        ic_examples={},
        sam_model=FakeSam(boxes=[[1, 2, 9, 10]]),
        zoom_model=FakeZoom(),
        nlp_model=object(),
        proposal_config=_proposal_config(),
    )

    snapshot = result.collector.to_dict()["snapshots"][0]
    assert builds == []
    assert [item["source"] for item in snapshot["candidates"]] == [
        "global", "sam_proposal",
    ]
    assert snapshot["candidates"][1]["bbox_original"] == [1, 2, 8, 8]
    assert result.coverage.adequate is True
    assert len(result.proposal_keys) == 1


def test_tree_catalog_preserves_sam_proposal_metadata(monkeypatch):
    from qavs.evidence_gap.pdf_runtime import TreeCatalog
    from qavs.independent_search import frontend

    monkeypatch.setattr(frontend, "include_pronouns", lambda nlp, target: False)
    image = Image.new("RGB", (32, 24))
    result = frontend.materialize_frontend(
        image=image,
        question="What color is the sign?",
        ic_examples={},
        sam_model=FakeSam(boxes=[[1, 2, 9, 10]]),
        zoom_model=FakeZoom(),
        nlp_model=object(),
        proposal_config=_proposal_config(),
    )
    node = TreeCatalog.from_collector(result.collector, image).node(
        result.proposal_keys[0]
    )

    assert node.sam_target_id == 17
    assert node.sam_target == "sign"
    assert node.sam_has_mask is False
    assert node.proposal_key == result.proposal_keys[0]


def test_empty_proposals_recover_once_and_preserve_existing_candidates(monkeypatch):
    from qavs.independent_search import frontend

    builds = []
    monkeypatch.setattr(frontend, "_build_tree", lambda features, image: (
        builds.append(1), _fake_tree(image)
    )[1])
    monkeypatch.setattr(frontend, "include_pronouns", lambda nlp, target: False)
    result = frontend.materialize_frontend(
        image=Image.new("RGB", (32, 24)),
        question="Find the object",
        ic_examples={},
        sam_model=FakeSam(boxes=[]),
        zoom_model=FakeZoom(),
        nlp_model=object(),
        proposal_config=_proposal_config(),
    )

    first = result.recovery.materialize(result.collector, trigger="empty_proposals")
    second = result.recovery.materialize(result.collector, trigger="empty_proposals")
    payload = result.collector.to_dict()
    assert builds == [1]
    assert first == second
    assert first.added_keys
    assert len(payload["snapshots"]) == 2
    assert payload["snapshots"][0]["candidates"][0]["source"] == "global"
    assert payload["snapshots"][1]["search_call_ordinal"] == 2
    assert payload["snapshots"][1]["candidates"][1]["source"] == "sgap_recovered"


def test_proposals_are_clipped_sorted_deduplicated_and_coverage_uses_union(monkeypatch):
    from qavs.independent_search import frontend

    monkeypatch.setattr(frontend, "include_pronouns", lambda nlp, target: False)
    result = frontend.materialize_frontend(
        image=Image.new("RGB", (20, 10)), question="Find signs", ic_examples={},
        sam_model=FakeSam(
            boxes=[
                [10, 0, 20, 10], [0, 0, 10, 10], [-2, -2, 9, 9],
                [0, 0, 9, 9], [3, 3, 3, 8], [0, 0, float("nan"), 4],
            ],
            scores=[0.2, 0.9, 0.8, 0.8, 1.0, 1.0],
        ),
        zoom_model=FakeZoom(), nlp_model=object(),
        proposal_config=_proposal_config(dedup_iou=0.9, min_count=3),
    )

    snapshot = result.collector.to_dict()["snapshots"][0]
    proposals = snapshot["candidates"][1:]
    assert [item["bbox_original"] for item in proposals] == [
        [0, 0, 10, 10], [0, 0, 9, 9], [10, 0, 10, 10],
    ]
    assert np.allclose([item["sam_score"] for item in proposals], [0.9, 0.8, 0.2])
    assert result.coverage.valid_count == 3
    assert result.coverage.spatial_coverage == 1.0
    assert result.coverage.adequate is True


def test_target_ids_control_flattening_before_stable_sort(monkeypatch):
    from qavs.independent_search import frontend

    class MultiTargetSam:
        def batch_inference(self, image, targets):
            return (
                {"vision_features": np.ones((1, 4, 4, 4), dtype=np.float32)},
                {
                    9: {"boxes": [[1, 1, 4, 4]], "scores": [0.5]},
                    4: {"boxes": [[1, 1, 4, 4]], "scores": [0.5]},
                },
                [4, 9],
            )

    monkeypatch.setattr(frontend, "include_pronouns", lambda nlp, target: False)
    result = frontend.materialize_frontend(
        image=Image.new("RGB", (20, 10)), question="Find signs", ic_examples={},
        sam_model=MultiTargetSam(), zoom_model=FakeZoom(("left", "right")),
        nlp_model=object(), proposal_config=_proposal_config(),
    )

    proposals = result.collector.to_dict()["snapshots"][0]["candidates"][1:]
    assert [item["sam_target_id"] for item in proposals] == [4]
    assert [item["bbox_original"] for item in proposals] == [[1, 1, 3, 3]]


def test_recovery_freezes_sam_features_until_materialization(monkeypatch):
    from qavs.independent_search import frontend

    features = np.ones((1, 2, 2, 2), dtype=np.float32)

    class MutableFeatureSam:
        def batch_inference(self, image, targets):
            return {"vision_features": features}, {1: {"boxes": []}}, [1]

    observed = []
    monkeypatch.setattr(frontend, "_build_tree", lambda value, image: (
        observed.append(float(np.asarray(value).sum())), _fake_tree(image)
    )[1])
    monkeypatch.setattr(frontend, "include_pronouns", lambda nlp, target: False)
    result = frontend.materialize_frontend(
        image=Image.new("RGB", (16, 12)), question="Find the sign", ic_examples={},
        sam_model=MutableFeatureSam(), zoom_model=FakeZoom(), nlp_model=object(),
        proposal_config=_proposal_config(),
    )
    features.fill(7)

    result.recovery.materialize(result.collector, trigger="empty_proposals")
    assert observed == [8.0]


def test_recovery_does_not_reemit_after_post_commit_failure(monkeypatch):
    from qavs.independent_search import frontend

    builds = []
    real_emit = frontend._emit
    emissions = []

    def fail_once(*args, **kwargs):
        emissions.append(1)
        if len(emissions) == 1:
            real_emit(*args, **kwargs)
            raise ValueError("synthetic emission failure")
        return real_emit(*args, **kwargs)

    monkeypatch.setattr(frontend, "_build_tree", lambda features, image: (
        builds.append(1), _fake_tree(image)
    )[1])
    monkeypatch.setattr(frontend, "include_pronouns", lambda nlp, target: False)
    result = frontend.materialize_frontend(
        image=Image.new("RGB", (16, 12)), question="Find the sign", ic_examples={},
        sam_model=FakeSam(boxes=[]), zoom_model=FakeZoom(), nlp_model=object(),
        proposal_config=_proposal_config(),
    )
    monkeypatch.setattr(frontend, "_emit", fail_once)

    try:
        result.recovery.materialize(result.collector, trigger="empty_proposals")
    except ValueError as error:
        assert "synthetic emission failure" in str(error)
    else:
        raise AssertionError("recovery emission unexpectedly succeeded")
    assert [
        snapshot["search_call_ordinal"]
        for snapshot in result.collector.to_dict()["snapshots"]
    ] == [1, 2]
    recovered = result.recovery.materialize(result.collector, trigger="empty_proposals")
    assert builds == [1]
    assert emissions == [1]
    assert recovered.added_keys
    assert [
        snapshot["search_call_ordinal"]
        for snapshot in result.collector.to_dict()["snapshots"]
    ] == [1, 2]


def test_recovery_deduplicates_against_proposals_and_catalog_remains_renderable(
    monkeypatch,
):
    from qavs.evidence_gap.pdf_runtime import TreeCatalog
    from qavs.independent_search import frontend
    from qavs.models.tree import NodeA, NodeState

    def recovered_tree(image):
        root = NodeA(NodeState(image, [0, 0, image.width, image.height]))
        root.is_root = True
        duplicate = NodeA(NodeState(image, [0, 0, 8, 8]), root)
        duplicate.depth = 1
        child = NodeA(NodeState(image, [1, 1, 3, 3]), duplicate)
        child.depth = 2
        duplicate.add_child(child)
        root.add_child(duplicate)
        return SimpleNamespace(root=root, max_depth=2)

    monkeypatch.setattr(frontend, "_build_tree", lambda features, image: recovered_tree(image))
    monkeypatch.setattr(frontend, "include_pronouns", lambda nlp, target: False)
    image = Image.new("RGB", (16, 12))
    result = frontend.materialize_frontend(
        image=image, question="Find the sign", ic_examples={},
        sam_model=FakeSam(boxes=[[0, 0, 8, 8]]), zoom_model=FakeZoom(),
        nlp_model=object(), proposal_config=_proposal_config(),
    )

    recovered = result.recovery.materialize(result.collector, trigger="pool_exhausted")
    catalog = TreeCatalog.from_collector(result.collector, image)
    rendered = catalog.render_nodes(recovered.added_keys)
    assert len(recovered.added_keys) == 1
    assert len(recovered.duplicate_keys) == 2
    assert len(rendered) == 1
    assert catalog.path_to(recovered.added_keys[0])[0] == catalog.root_key


def test_empty_recovery_result_is_memoized_and_hashed(monkeypatch):
    from qavs.independent_search import frontend

    builds = []
    monkeypatch.setattr(frontend, "_build_tree", lambda features, image: (
        builds.append(1), _fake_tree(Image.new("RGB", image.size))
    )[1])
    monkeypatch.setattr(frontend, "include_pronouns", lambda nlp, target: False)
    image = Image.new("RGB", (32, 24), "white")
    result = frontend.materialize_frontend(
        image=image, question="Find the sign", ic_examples={},
        sam_model=FakeSam(boxes=[[0, 0, 16, 24]]), zoom_model=FakeZoom(),
        nlp_model=object(), proposal_config=_proposal_config(dedup_iou=0.9),
    )

    first = result.recovery.materialize(result.collector, trigger="coverage_failed")
    snapshot_count = len(result.collector.to_dict()["snapshots"])
    second = result.recovery.materialize(result.collector, trigger="different_trigger")
    assert builds == [1]
    assert first == second
    assert first.added_keys == ()
    assert len(result.collector.to_dict()["snapshots"]) == snapshot_count == 2
    assert first.collector_sha256 == hashlib.sha256(
        frontend._canonical_json(result.collector.to_dict()).encode("utf-8")
    ).hexdigest()


def test_tree_snapshot_is_bound_to_source_pixels(monkeypatch):
    from qavs.independent_search import frontend

    monkeypatch.setattr(frontend, "include_pronouns", lambda nlp, target: False)
    image = Image.new("RGB", (12, 10), (1, 2, 3))
    result = frontend.materialize_frontend(
        image=image, question="Read the sign", ic_examples={},
        sam_model=FakeSam(boxes=[]), zoom_model=FakeZoom(), nlp_model=object(),
        proposal_config=_proposal_config(),
    )

    identity = result.collector.to_dict()["snapshots"][0]["source_image_identity"]
    assert identity == {
        "mode": "RGB",
        "size": [12, 10],
        "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }
