"""Exercise the complete search controller with deterministic model doubles."""

import json
from pathlib import Path

from PIL import Image
import pytest

from qavs.evidence_gap.search_state import SearchStateCollector
from qavs.independent_search import IndependentSearchConfig, method
from qavs.independent_search.frontend import (
    ProposalCoverage,
    ProposalFrontendResult,
    RecoveryResult,
)
from tests.helpers import candidate_key, event_for


class Generator:
    def __init__(self, *, global_answer=0, confidence=0.2):
        self.global_answer = global_answer
        self.confidence = confidence
        self.calls = 0

    def get_confidence_value(self, *args, **kwargs):
        self.calls += 1
        return self.confidence

    def generate_text_only(self, prompt):
        return json.dumps({
            "augmented_queries": ["locate sign", "sign detail", "sign context"],
            "evidence_items": [{
                "kind": "target_detail", "target": "sign",
                "requirements": ["presence", "visual_detail"],
            }],
            "global_scope_required": False,
            "detail_demand": 1.0,
            "context_demand": 0.0,
        })

    def multiple_choices_with_losses(self, image, question, options, nodes):
        self.calls += 1
        local = bool(nodes) and not getattr(nodes[0], "is_root", False)
        winner = 1 if local else self.global_answer
        return winner, [0.1, 0.9] if winner == 0 else [0.9, 0.1]

    def free_form_using_nodes(self, image, question, nodes):
        return '{"zoom":0.9,"split":0.1,"expand":0.1,"next":0.1}'


class Verifier:
    def __init__(self, *, global_answer=None, local_answer="blue"):
        self.global_answer = global_answer
        self.local_answer = local_answer

    def label_token_losses(self, image, prompt, options, nodes):
        assert options == ["A", "B", "C"]
        if "Target role:" in prompt:
            return 0, [0.0, 2.0, 4.0]
        answer = self.global_answer if image.size == (16, 16) else self.local_answer
        if answer is None:
            return 2, [4.0, 2.0, 0.0]
        if f"Candidate answer: {answer}" in prompt:
            return 0, [0.0, 2.0, 4.0]
        return 1, [2.0, 0.0, 4.0]


class Sam:
    def batch_inference(self, image, roles):
        return None, {
            index: {"boxes": [[0, 0, image.width, image.height]]}
            for index, _ in enumerate(roles)
        }, list(range(len(roles)))


class Clip:
    def score(self, images, texts):
        return [[0.1 + index * 0.6 for _ in texts] for index, _ in enumerate(images)]


def candidate(box, depth, parent=None):
    return {
        "canonical_key": candidate_key(box, depth), "bbox_original": list(box),
        "parent_key": parent, "child_keys": [], "depth": depth,
        "render_level": 0, "source": "global" if parent is None else "fine",
        "stage_rank": depth, "prior_prob": 0.5, "complexity": 0.5,
        "fast_confidence": None, "posterior_score": None,
        "is_evaluated": False, "answering_confidence": None,
    }


def frontend(image, calls, *, adequate, boxes=None):
    root = candidate((0, 0, 16, 16), 0)
    boxes = boxes or [(0, 0, 8, 16), (8, 0, 8, 16)]
    children = [candidate(box, 1, root["canonical_key"]) for box in boxes]
    root["child_keys"] = [item["canonical_key"] for item in children]
    candidates = [root, *children]
    collector = SearchStateCollector(image)

    def emit(ordinal):
        refs, snapshot = event_for(image, candidates, event="tree_ready", ordinal=ordinal)
        collector(refs, snapshot)

    class Recovery:
        def materialize(self, collector, *, trigger):
            calls.append(trigger)
            recovered = candidate((0, 0, 4, 8), 2, children[0]["canonical_key"])
            children[0]["child_keys"] = [recovered["canonical_key"]]
            candidates.append(recovered)
            emit(2)
            return RecoveryResult(trigger, (recovered["canonical_key"],), (), "c" * 64)

    emit(1)
    return ProposalFrontendResult(
        image=image, targets=("sign",), collector=collector,
        proposal_keys=tuple(root["child_keys"]),
        coverage=ProposalCoverage(len(children), 1.0, 1, 0.005, adequate),
        recovery=Recovery(), diagnostics={},
    )


@pytest.fixture
def run_search(monkeypatch, tmp_path):
    image = Image.new("RGB", (16, 16))
    image.putdata([(x * 16, y * 16, (x + y) * 8) for y in range(16) for x in range(16)])
    image.save(tmp_path / "image.png")
    calls = []

    def run(*, generator=None, verifier=None, binary=False, adequate=True,
            max_steps=4, same_checkpoint=False, expect_frontend=True, boxes=None):
        def materialize(**kwargs):
            assert expect_frontend, "accepted global answer must stop before proposals"
            calls.append("proposals")
            return frontend(kwargs["image"], calls, adequate=adequate, boxes=boxes)

        monkeypatch.setattr(method, "materialize_frontend", materialize)
        raw = json.loads((Path(method.__file__).parents[1] / "defaults.json").read_text())
        raw["budget"]["max_steps"] = max_steps
        output, trace = method.run_independent_sample(
            original_annotation={
                "question": "Is there a sign in the image?" if binary else "What color is the sign?",
                "options": ["Yes", "No"] if binary else ["red", "blue"],
                "answer_type": "yes_no" if binary else "logits_match",
                "input_image": "image.png",
            },
            image_folder=tmp_path, ic_examples=[],
            config=IndependentSearchConfig.from_mapping(raw),
            sam_model=Sam(), generator_model=generator or Generator(),
            verifier_model=verifier or Verifier(), nlp_model=object(), clip_scorer=Clip(),
            generator_checkpoint_sha256="a" * 64,
            verifier_checkpoint_sha256=("a" if same_checkpoint else "b") * 64,
            generator_family="qwen",
        )
        return output, trace, calls

    return run


def test_global_verifier_can_stop_with_answer_different_from_generator(run_search):
    output, trace, calls = run_search(
        verifier=Verifier(global_answer="blue"), expect_frontend=False,
    )
    assert output == 1
    assert trace["global_observation"]["output"] == 0
    assert trace["mode"] == "direct"
    assert trace["final_decision"]["source"] == "accepted_global_verifier"
    assert calls == []


def test_global_no_must_enter_search_even_when_both_models_agree(run_search):
    output, trace, calls = run_search(
        binary=True, generator=Generator(global_answer=1, confidence=0.9),
        verifier=Verifier(global_answer="No", local_answer=None), max_steps=1,
    )
    assert output == "no"
    assert trace["mode"] == "search"
    assert calls[0] == "proposals"
    assert trace["final_decision"]["source"] == "full_image_fallback"
    assert trace["accepted_hypothesis"] is None


def test_search_accepts_after_two_grounded_distinct_local_views(run_search):
    output, trace, _ = run_search()
    phase = trace["controller"]["phases"][0]
    assert output == 1
    assert phase["termination"] == "ACCEPTED_STOP"
    assert phase["assessment_count"] == 2
    assert trace["final_decision"]["source"] == "accepted_local_hypothesis"
    assert len(trace["accepted_hypothesis"]["confirmation_group_ids"]) == 2


def test_budget_exhaustion_returns_saved_global_answer(run_search):
    output, trace, _ = run_search(verifier=Verifier(local_answer=None), max_steps=1)
    assert output == 0
    assert trace["mode"] == "search"
    assert trace["accepted_hypothesis"] is None
    assert trace["final_decision"]["source"] == "full_image_fallback"
    assert trace["controller"]["phases"][0]["budget_after"]["remaining_steps"] == 0


def test_failed_proposal_coverage_scans_before_local_search(run_search):
    output, trace, calls = run_search(
        adequate=False, verifier=Verifier(local_answer=None), max_steps=1,
    )
    assert calls == ["proposals", "coverage_failed"]
    assert trace["controller"]["phases"][0]["phase"] == "recovery"
    assert output == 0
    assert trace["final_decision"]["source"] == "full_image_fallback"


def test_identical_checkpoints_fail_before_model_inference(run_search):
    generator = Generator()
    with pytest.raises(ValueError, match="different checkpoint"):
        run_search(generator=generator, same_checkpoint=True, expect_frontend=False)
    assert generator.calls == 0


def test_negative_coverage_includes_candidates_beyond_active_pool(monkeypatch, run_search):
    checked = []
    original = method._negative_candidate_coverage

    def record_coverage(records, keys):
        checked.append(tuple(keys))
        return original(records, keys)

    monkeypatch.setattr(method, "_negative_candidate_coverage", record_coverage)
    _, trace, _ = run_search(
        binary=True, verifier=Verifier(local_answer=None), max_steps=1,
        boxes=[(x, 0, 3, 16) for x in (0, 3, 6, 9, 12)],
    )
    retained = set(trace["sam_proposals"]["proposal_keys"])
    assert len(retained) == 5
    assert checked and all(set(keys) == retained for keys in checked)


def test_final_negative_coverage_includes_recovered_candidates(monkeypatch, run_search):
    checked = []
    original = method._negative_candidate_coverage

    def record_coverage(records, keys):
        checked.append(tuple(keys))
        return original(records, keys)

    class NextGenerator(Generator):
        def free_form_using_nodes(self, image, question, nodes):
            return '{"zoom":0.0,"split":0.0,"expand":0.0,"next":1.0}'

    monkeypatch.setattr(method, "_negative_candidate_coverage", record_coverage)
    _, trace, calls = run_search(
        binary=True, generator=NextGenerator(),
        verifier=Verifier(local_answer=None), max_steps=20,
    )
    assert calls == ["proposals", "pool_exhausted"]
    assert len(trace["controller"]["phases"]) == 2
    recovered = set(trace["scan_recover"][0]["added_keys"])
    assert recovered
    assert set(checked[-1]) == set(trace["sam_proposals"]["proposal_keys"]) | recovered
    first, second = trace["controller"]["phases"]
    assert second["budget_before"] == first["budget_after"]
