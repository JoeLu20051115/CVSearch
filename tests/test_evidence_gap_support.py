import hashlib
import json
import math
from dataclasses import replace
from types import MethodType, SimpleNamespace
import unittest

from PIL import Image
import torch

from cvsearch.evidence_gap.method import _BudgetedZoomModel
from cvsearch.evidence_gap.search_state import (
    NextCandidate,
    _RenderSource,
    _canonical_key,
    _renderer_identity,
    _renderer_kind,
    _source_identity,
    _source_key,
)
from cvsearch.evidence_gap.types import (
    BudgetLedger,
    EvidenceRequirement,
    EVIDENCE_SUPPORT_NORMALIZATION_TOLERANCE,
    EvidenceSupportResult,
    ObservationBatchResult,
    QueryPlan,
    sanitize_evidence_requirements,
)
from cvsearch.models.modeling_qwenvl import (
    ANSWER_FREE_SUPPORT_PROMPT_VERSION,
    EVIDENCE_SUPPORT_TRANSFORM,
    ModelQwenVL,
)


def evidence_items():
    return ({
        "kind": "target_detail",
        "target": "red sign",
        "requirements": ["presence", "visual_detail"],
    },)


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return FakeInputs(input_ids=torch.tensor([[1, 2]]))


class FakeForward:
    def __init__(self, yes_logit=2.0, no_logit=-1.0, dtype=torch.float32):
        self.yes_logit = yes_logit
        self.no_logit = no_logit
        self.logits_dtype = dtype
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        logits = torch.zeros((1, 1, 9455), dtype=self.logits_dtype)
        logits[0, -1, 9454] = self.yes_logit
        logits[0, -1, 2753] = self.no_logit
        return SimpleNamespace(logits=logits)


class FakeTokenizer:
    name_or_path = "frozen/qwen-checkpoint"
    padding_side = "right"

    def __init__(self, yes_tokens=(9454,), no_tokens=(2753,)):
        self.yes_tokens = list(yes_tokens)
        self.no_tokens = list(no_tokens)

    def __call__(self, text):
        return SimpleNamespace(input_ids={"Yes": self.yes_tokens, "No": self.no_tokens}[text])


def support_adapter(*, yes_logit=2.0, no_logit=-1.0,
                    yes_tokens=(9454,), no_tokens=(2753,),
                    logits_dtype=torch.float32):
    adapter = ModelQwenVL.__new__(ModelQwenVL)
    adapter.device = "cpu"
    adapter.dtype = torch.bfloat16
    adapter.index_yes = 9454
    adapter.index_no = 2753
    adapter.tokenizer = FakeTokenizer(yes_tokens, no_tokens)
    adapter.processor = FakeProcessor()
    adapter.model = FakeForward(yes_logit, no_logit, logits_dtype)
    adapter.model.config = SimpleNamespace(
        _name_or_path="frozen/qwen-checkpoint",
        model_type="qwen2_5_vl",
        architectures=["Qwen2_5_VLForConditionalGeneration"],
        _attn_implementation="flash_attention_2",
        transformers_version="4.41.2",
    )
    adapter.model.dtype = torch.bfloat16
    adapter.model.device = torch.device("cpu")
    adapter.use_flash_attn = True
    adapter.model_checkpoint = "frozen/qwen-checkpoint"
    adapter.processor.image_processor = SimpleNamespace(
        is_fast=True, min_pixels=3136, max_pixels=12845056,
        patch_size=14, temporal_patch_size=2, merge_size=2,
    )
    adapter.processor.tokenizer = adapter.tokenizer
    adapter.get_prompt_from_qs = lambda question: question
    adapter.process_nodes_to_image_list = lambda nodes, image, root_anyres=True: [image.copy()]
    return adapter


class EvidenceRequirementSanitizationTest(unittest.TestCase):
    def test_requirement_dto_rejects_nonstring_or_mismatched_identity(self):
        with self.assertRaises(TypeError):
            EvidenceRequirement(1, "coverage", "global scope coverage")
        with self.assertRaisesRegex(ValueError, "identity"):
            EvidenceRequirement("req-not-the-content-hash", "coverage", "global scope coverage")

    def test_exact_answer_free_schemas_have_deterministic_ids_and_text(self):
        items = (
            {
                "kind": "target_detail",
                "target": "  Red   sign ",
                "requirements": ["presence", "visual_detail"],
            },
            {"kind": "relation_context", "targets": ["red sign", "door"]},
            {"kind": "coverage", "requirement": "global_scope"},
            {
                "kind": "runtime_ranking_context",
                "query_source": "main_query_plus_current_visual_cue",
                "planned_augmented_queries_used": False,
            },
        )

        first = sanitize_evidence_requirements(items)
        second = sanitize_evidence_requirements(json.loads(json.dumps(items)))

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(
            tuple(requirement.text for requirement in first),
            (
                "presence and visual detail of Red sign",
                "relation context among red sign and door",
                "global scope coverage",
            ),
        )
        self.assertEqual(len({requirement.requirement_id for requirement in first}), 3)
        json.dumps([requirement.to_dict() for requirement in first], allow_nan=False)

    def test_rejects_contaminated_unknown_or_non_answer_free_schemas(self):
        contaminated_fields = ("answer", "options", "labels", "policy", "benchmark", "verifier")
        for field in contaminated_fields:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "contaminated"):
                sanitize_evidence_requirements(({
                    "kind": "target_detail",
                    "target": "sign",
                    "requirements": ["presence"],
                    field: "forbidden",
                },))
        with self.assertRaisesRegex(ValueError, "must be exactly"):
            sanitize_evidence_requirements(({
                "kind": "target_detail", "target": "sign", "requirements": ["answer is A"],
            },))
        with self.assertRaisesRegex(ValueError, "unknown"):
            sanitize_evidence_requirements(({"kind": "question_family", "label": "ocr"},))

    def test_excluded_runtime_context_accepts_only_the_frozen_audit_values(self):
        base = {
            "kind": "runtime_ranking_context",
            "query_source": "main_query_plus_current_visual_cue",
            "planned_augmented_queries_used": False,
        }
        for field, value in (
            ("query_source", "main_query"),
            ("planned_augmented_queries_used", True),
        ):
            contaminated = dict(base, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                sanitize_evidence_requirements((contaminated,))

    def test_duplicate_excluded_metadata_and_unicode_category_c_fail_closed(self):
        runtime = {
            "kind": "runtime_ranking_context",
            "query_source": "main_query_plus_current_visual_cue",
            "planned_augmented_queries_used": False,
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            sanitize_evidence_requirements((runtime, dict(runtime)))
        for character in ("\x7f", "\x85", "\u200b", "\ue000", "\u0378"):
            with self.subTest(codepoint=ord(character)), self.assertRaisesRegex(
                ValueError, "Unicode category C"
            ):
                sanitize_evidence_requirements(({
                    "kind": "target_detail",
                    "target": f"red{character}sign",
                    "requirements": ["presence", "visual_detail"],
                },))


class QwenAnswerFreeSupportTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (4, 3), (1, 2, 3))
        self.requirements = sanitize_evidence_requirements(evidence_items())
        self.identity = {
            "canonical_keys": ["root"],
            "renderer_identity": "root-view",
        }

    def test_one_aggregate_yes_no_forward_records_strict_provenance(self):
        model = support_adapter()
        identity = json.dumps(self.identity, sort_keys=True, separators=(",", ":"))
        result = model.evidence_support(
            question="What is visible?", requirements=self.requirements,
            rendered_observation=self.image, observation_identity=identity,
        )

        expected = math.exp(2.0) / (math.exp(2.0) + math.exp(-1.0))
        self.assertIsInstance(result, EvidenceSupportResult)
        self.assertEqual(model.model.calls, 1)
        self.assertEqual(len(model.processor.calls), 1)
        self.assertEqual(result.prompt_version, ANSWER_FREE_SUPPORT_PROMPT_VERSION)
        self.assertEqual((result.yes_token_id, result.no_token_id), (9454, 2753))
        self.assertEqual((result.yes_logit, result.no_logit), (2.0, -1.0))
        self.assertAlmostEqual(result.p_yes, expected)
        self.assertEqual((result.support_avg, result.support_min), (result.p_yes, result.p_yes))
        self.assertEqual(result.logical_calls, 1)
        self.assertIsNone(result.accounted_pixels)
        self.assertEqual(result.checkpoint, "frozen/qwen-checkpoint")
        self.assertEqual(result.observation_identity, identity)
        self.assertEqual(tuple(item.requirement_id for item in result.requirements),
                         tuple(item.requirement_id for item in self.requirements))
        self.assertEqual(result.requirement_set_id,
                         EvidenceSupportResult.requirement_set_id_for(self.requirements))
        self.assertEqual(result.view_sha256,
                         hashlib.sha256(b"RGB\x00" + b"4x3\x00" + self.image.tobytes()).hexdigest())
        payload = result.to_dict()
        self.assertNotIn("per_requirement_logits", payload)
        self.assertEqual(
            payload["p_yes_transform"], EVIDENCE_SUPPORT_TRANSFORM,
        )
        self.assertIn("final_position", EVIDENCE_SUPPORT_TRANSFORM)
        self.assertIn("preserve_model_dtype", EVIDENCE_SUPPORT_TRANSFORM)
        self.assertIn("no_float32_cast", EVIDENCE_SUPPORT_TRANSFORM)
        self.assertIn("no_legacy_2p_minus_1", EVIDENCE_SUPPORT_TRANSFORM)
        json.dumps(payload, allow_nan=False)
        prompt = model.processor.calls[0]["text"][0]
        self.assertIn("What is visible?", prompt)
        self.assertIn(self.requirements[0].requirement_id, prompt)
        self.assertIn(self.requirements[0].text, prompt)
        self.assertNotIn("option-a", prompt)

    def test_zero_requirements_and_contaminated_identity_make_no_forward(self):
        model = support_adapter()
        self.assertIsNone(model.evidence_support(
            question="What is visible?", requirements=(),
            rendered_observation=self.image,
            observation_identity=json.dumps(self.identity),
        ))
        self.assertEqual(model.model.calls, 0)
        with self.assertRaisesRegex(ValueError, "observation_identity"):
            model.evidence_support(
                question="What is visible?", requirements=self.requirements,
                rendered_observation=self.image,
                observation_identity='{"canonical_key":"root","posterior":NaN}',
            )
        self.assertEqual(model.model.calls, 0)

    def test_nonfinite_logits_are_rejected(self):
        model = support_adapter(yes_logit=float("nan"))
        with self.assertRaisesRegex(ValueError, "support logits"):
            model.evidence_support(
                question="What is visible?", requirements=self.requirements,
                rendered_observation=self.image,
                observation_identity=json.dumps(
                    self.identity, sort_keys=True, separators=(",", ":")
                ),
            )
        self.assertEqual(model.model.calls, 1)

    def test_bfloat16_softmax_pair_preserves_raw_probability_and_is_accepted(self):
        model = support_adapter(
            yes_logit=3.0, no_logit=-2.0, logits_dtype=torch.bfloat16,
        )
        expected_pair = torch.softmax(
            torch.tensor([3.0, -2.0], dtype=torch.bfloat16), dim=-1,
        )

        try:
            result = model.evidence_support(
                question="What is visible?", requirements=self.requirements,
                rendered_observation=self.image,
                observation_identity=json.dumps(
                    self.identity, sort_keys=True, separators=(",", ":"),
                ),
            )
        except ValueError as error:
            self.fail(f"valid bfloat16 softmax pair was rejected: {error}")

        self.assertGreater(abs(sum(float(value) for value in expected_pair) - 1.0), 1e-5)
        self.assertEqual(
            EVIDENCE_SUPPORT_NORMALIZATION_TOLERANCE,
            float(torch.finfo(torch.bfloat16).eps),
        )
        self.assertEqual(result.p_yes, float(expected_pair[0]))
        self.assertEqual(result.p_no, float(expected_pair[1]))
        self.assertEqual((result.support_avg, result.support_min),
                         (result.p_yes, result.p_yes))

    def test_support_result_still_rejects_invalid_probability_pairs(self):
        result = support_adapter().evidence_support(
            question="What is visible?", requirements=self.requirements,
            rendered_observation=self.image,
            observation_identity=json.dumps(
                self.identity, sort_keys=True, separators=(",", ":"),
            ),
        )
        for changes, message in (
            ({"p_yes": 0.90, "p_no": 0.05,
              "support_avg": 0.90, "support_min": 0.90}, "normalized"),
            ({"p_yes": float("nan"), "support_avg": float("nan"),
              "support_min": float("nan")}, "p_yes"),
            ({"p_yes": 1.01, "p_no": -0.01,
              "support_avg": 1.01, "support_min": 1.01}, "p_yes"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                replace(result, **changes)

    def test_yes_and_no_must_each_be_one_distinct_frozen_token(self):
        identity = json.dumps(self.identity, sort_keys=True, separators=(",", ":"))
        for yes_tokens, no_tokens in (
            ((99, 9454), (2753,)),
            ((9454,), (88, 2753)),
            ((9454,), (9454,)),
        ):
            with self.subTest(yes=yes_tokens, no=no_tokens):
                model = support_adapter(yes_tokens=yes_tokens, no_tokens=no_tokens)
                with self.assertRaisesRegex(ValueError, "exactly one distinct frozen token"):
                    model.evidence_support(
                        question="What is visible?", requirements=self.requirements,
                        rendered_observation=self.image, observation_identity=identity,
                    )
                self.assertEqual(model.model.calls, 0)
                self.assertEqual(model.processor.calls, [])


class RawObservationModel:
    def __init__(self, *, fail_phase=None):
        self.support = support_adapter()
        self.calls = []
        self.fail_phase = fail_phase
        self.answer_index = 0
        self.render_calls = 0

    def evidence_support(self, *, question, requirements, rendered_observation,
                         observation_identity):
        phase = "current_support" if not any(call[0] == "support" for call in self.calls) else "candidate_support"
        self.calls.append(("support", phase, (question, observation_identity)))
        if self.fail_phase == phase:
            raise RuntimeError(f"failed {phase}")
        return self.support.evidence_support(
            question=question, requirements=requirements,
            rendered_observation=rendered_observation,
            observation_identity=observation_identity,
        )

    def _prepare_evidence_support(self, question, requirements):
        return self.support._prepare_evidence_support(question, requirements)

    def process_nodes_to_image_list(self, nodes, image, root_anyres=True):
        self.render_calls += 1
        return self.support.process_nodes_to_image_list(nodes, image, root_anyres=root_anyres)

    def free_form_using_nodes(self, image, question, nodes):
        phase = f"hr_answer_{self.answer_index}"
        self.calls.append(("hr_answer", question, tuple(nodes)))
        self.answer_index += 1
        if self.fail_phase == phase:
            raise RuntimeError(f"failed {phase}")
        return ("A", "B", "C", "D")[self.answer_index - 1]

    def multiple_choices_with_losses(self, image, question, options, nodes):
        self.calls.append(("vstar_answer", question, tuple(options), tuple(nodes)))
        if self.fail_phase == "vstar_answer":
            raise RuntimeError("failed vstar_answer")
        losses = [0.4, 0.1] + [0.8 + index for index in range(max(0, len(options) - 2))]
        return min(1, len(options) - 1), losses[:len(options)]


def next_candidate_for(image, *, bbox=(1, 1, 2, 1), depth=1, render_level=0,
                       source="fine", scope="main", crop_origin=(0, 0)):
    source_key = _source_key(_source_identity(image))
    kind = _renderer_kind(source)
    key = _canonical_key(bbox, depth, render_level)
    return NextCandidate(
        canonical_key=key,
        bbox_original=tuple(bbox),
        depth=depth,
        render_level=render_level,
        posterior_score=0.8,
        first_seen_ordinal=0,
        tree_scope=scope,
        crop_origin=tuple(crop_origin),
        source_image_key=source_key,
        source=source,
        renderer_kind=kind,
        renderer_identity=_renderer_identity(
            source_key, tuple(bbox), render_level, kind,
        ),
        _render_source=_RenderSource.capture(image),
    )


def run_batch(raw, *, answer_type, options, max_calls, max_pixels, evidence=evidence_items()):
    image = Image.new("RGB", (4, 3), (1, 2, 3))
    ledger = BudgetLedger(max_calls, max_pixels)
    budgeted = _BudgetedZoomModel(raw, ledger, answer_reserve_calls=0,
                                 answer_image_loader=lambda: image.copy(),
                                 answer_type=answer_type)
    if answer_type == "option_list":
        budgeted._free_form_started = True
        budgeted._free_form_remaining = 0
    result = budgeted.post_anchor_observation_batch(
        source_image=image,
        q0="What is visible?",
        query_plan=QueryPlan(main_query="What is visible?", evidence_items=evidence),
        current_support_view=(),
        candidate_support_view=(next_candidate_for(image),),
        answer_type=answer_type,
        options=options,
    )
    return result, ledger, budgeted


class AtomicObservationBatchTest(unittest.TestCase):
    HR_OPTIONS = tuple(f"A. choice {index}\nB. other" for index in range(4))

    def test_result_rejects_negative_elapsed_and_inconsistent_charge_state(self):
        plan = {"schema_version": 1, "total_calls": 0, "total_pixels": 0}
        ledger = BudgetLedger(1, 1).to_dict()
        with self.assertRaisesRegex(ValueError, "elapsed"):
            ObservationBatchResult(
                status="no_requirements", batch_plan=plan, admitted=False, charged=False,
                ledger_before=ledger, ledger_after=ledger,
                failure_phase="preflight", failure_reason="no_requirements",
                elapsed_seconds=-1.0,
            )
        with self.assertRaisesRegex(ValueError, "admitted and charged"):
            ObservationBatchResult(
                status="model_failed", batch_plan=plan, admitted=False, charged=True,
                ledger_before=ledger, ledger_after=ledger,
                failure_phase="current_support", failure_reason="boom",
            )

    def test_result_rejects_fabricated_success_support_hash_and_ledger_mismatches(self):
        zero_plan = {"schema_version": 1, "total_calls": 0, "total_pixels": 0}
        zero_ledger = BudgetLedger(1, 1).to_dict()
        with self.assertRaisesRegex(ValueError, "plan"):
            ObservationBatchResult(
                status="success", batch_plan=zero_plan, admitted=True, charged=True,
                ledger_before=zero_ledger, ledger_after=zero_ledger,
                candidate_answer={"winner": 0, "losses": [0.0]},
            )

        original, _, _ = run_batch(
            RawObservationModel(), answer_type="logits_match", options=("a", "b"),
            max_calls=5, max_pixels=60,
        )
        payload = original.to_dict()
        wrong_hash = replace(original.current_support, batch_plan_hash="0" * 64)
        with self.assertRaisesRegex(ValueError, "support.*plan hash"):
            ObservationBatchResult(
                status="success", batch_plan=original.batch_plan,
                admitted=True, charged=True,
                ledger_before=payload["ledger_before"], ledger_after=payload["ledger_after"],
                current_support=wrong_hash, candidate_support=original.candidate_support,
                candidate_answer=original.candidate_answer,
                executed_stages=original.executed_stages,
            )
        with self.assertRaisesRegex(ValueError, "ledger delta"):
            ObservationBatchResult(
                status="success", batch_plan=original.batch_plan,
                admitted=True, charged=True,
                ledger_before=payload["ledger_before"], ledger_after=payload["ledger_before"],
                current_support=original.current_support,
                candidate_support=original.candidate_support,
                candidate_answer=original.candidate_answer,
                executed_stages=original.executed_stages,
            )
        self.assertEqual(
            original.to_dict()["current_support"]["batch_plan_hash"],
            original.current_support.batch_plan_hash,
        )

    def test_hr_exact_boundary_charges_six_and_returns_four_raw_outputs(self):
        raw = RawObservationModel()
        result, ledger, budgeted = run_batch(
            raw, answer_type="option_list", options=self.HR_OPTIONS,
            max_calls=6, max_pixels=72,
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.candidate_answer, ["A", "B", "C", "D"])
        self.assertEqual((ledger.mllm_calls, ledger.processed_pixels), (6, 72))
        self.assertEqual([call[0] for call in raw.calls],
                         ["support", "support", "hr_answer", "hr_answer", "hr_answer", "hr_answer"])
        self.assertTrue(all("Answer the option letter directly." in call[1]
                            for call in raw.calls[2:]))
        self.assertEqual(result.verifier_status, "disabled_same_checkpoint_unpromoted")
        self.assertIsNone(result.verifier_avg)
        self.assertIsNone(result.verifier_min)
        self.assertFalse(budgeted._answer_started)
        self.assertTrue(budgeted._free_form_started)
        self.assertEqual(budgeted._free_form_remaining, 0)
        payload = result.to_dict()
        self.assertEqual(payload["batch_plan_hash"], result.batch_plan_hash)
        self.assertEqual(payload["current_support"]["batch_plan_hash"], result.batch_plan_hash)
        self.assertEqual(payload["candidate_support"]["batch_plan_hash"], result.batch_plan_hash)
        self.assertEqual(result.batch_plan["yes_tokenization"], [9454])
        self.assertEqual(result.batch_plan["no_tokenization"], [2753])
        self.assertEqual((result.batch_plan["yes_token_id"], result.batch_plan["no_token_id"]),
                         (9454, 2753))
        self.assertEqual(result.batch_plan["p_yes_transform"], EVIDENCE_SUPPORT_TRANSFORM)
        json.dumps(payload, allow_nan=False)

    def test_root_and_real_next_candidate_select_frozen_qwen_views_and_hashes(self):
        image = Image.new("RGB", (8, 6))
        for y in range(image.height):
            for x in range(image.width):
                image.putpixel((x, y), (x * 20, y * 30, x + y))
        candidate = next_candidate_for(
            image, bbox=(2, 1, 3, 2), depth=2, scope="cropped", crop_origin=(2, 1),
        )

        class ViewRaw(RawObservationModel):
            def process_nodes_to_image_list(self, nodes, source, root_anyres=True):
                if not nodes:
                    return [source.copy()]
                x, y, width, height = nodes[0].state.bbox
                crop = source.crop((x, y, x + width, y + height))
                zoom = crop.resize((crop.width * 2, crop.height * 2))
                return [source.copy(), crop, zoom]

        raw = ViewRaw()
        ledger = BudgetLedger(5, 5 * image.width * image.height)
        budgeted = _BudgetedZoomModel(raw, ledger, answer_type="logits_match")
        result = budgeted.post_anchor_observation_batch(
            source_image=image, q0="What is visible?",
            query_plan=QueryPlan(main_query="What is visible?", evidence_items=evidence_items()),
            current_support_view=(), candidate_support_view=(candidate,),
            answer_type="logits_match", options=("a", "b"),
        )

        expected_crop = image.crop((2, 1, 5, 3)).resize((6, 4))
        self.assertEqual(result.current_support.observation_size, image.size)
        self.assertEqual(result.current_support.view_sha256,
                         _BudgetedZoomModel._observation_sha256(image))
        self.assertEqual(result.candidate_support.observation_size, expected_crop.size)
        self.assertEqual(result.candidate_support.view_sha256,
                         _BudgetedZoomModel._observation_sha256(expected_crop))
        plan = result.batch_plan
        self.assertEqual(plan["current_observation"]["canonical_keys"], [])
        self.assertEqual(plan["candidate_observation"]["canonical_keys"],
                         [candidate.canonical_key])
        self.assertEqual(plan["candidate_observation"]["renderer_identities"],
                         [candidate.renderer_identity])

    def test_foreign_fake_or_malformed_task2_descriptors_fail_before_render_or_charge(self):
        source = Image.new("RGB", (4, 3), (1, 2, 3))
        foreign = next_candidate_for(Image.new("RGB", (4, 3), (9, 8, 7)))
        safe = next_candidate_for(source)
        malformed = replace(safe, renderer_identity="forged-renderer")
        fake = SimpleNamespace(
            canonical_key=safe.canonical_key,
            renderer_identity=safe.renderer_identity,
            source_image_key=safe.source_image_key,
            render_node=safe.render_node,
            to_dict=safe.to_dict,
        )
        for descriptor in (foreign, malformed, fake):
            with self.subTest(descriptor=type(descriptor).__name__):
                raw = RawObservationModel()
                ledger = BudgetLedger(5, 60)
                budgeted = _BudgetedZoomModel(raw, ledger, answer_type="logits_match")
                with self.assertRaisesRegex((TypeError, ValueError), "descriptor"):
                    budgeted.post_anchor_observation_batch(
                        source_image=source, q0="What is visible?",
                        query_plan=QueryPlan(
                            main_query="What is visible?", evidence_items=evidence_items(),
                        ),
                        current_support_view=(), candidate_support_view=(descriptor,),
                        answer_type="logits_match", options=("a", "b"),
                    )
                self.assertEqual((ledger.mllm_calls, ledger.processed_pixels), (0, 0))
                self.assertEqual(raw.render_calls, 0)
                self.assertEqual(raw.calls, [])
                self.assertEqual(raw.support.processor.calls, [])
                self.assertEqual(raw.support.model.calls, 0)

    def test_control_or_duplicate_metadata_rejects_before_renderer_processor_or_model(self):
        runtime = {
            "kind": "runtime_ranking_context",
            "query_source": "main_query_plus_current_visual_cue",
            "planned_augmented_queries_used": False,
        }
        contaminated = (
            ({
                "kind": "target_detail", "target": "red\u200bsign",
                "requirements": ["presence", "visual_detail"],
            },),
            (runtime, dict(runtime)),
        )
        for items in contaminated:
            raw = RawObservationModel()
            with self.subTest(items=items), self.assertRaises(ValueError):
                run_batch(
                    raw, answer_type="logits_match", options=("a", "b"),
                    max_calls=5, max_pixels=60, evidence=items,
                )
            self.assertEqual(raw.render_calls, 0)
            self.assertEqual(raw.calls, [])
            self.assertEqual(raw.support.processor.calls, [])
            self.assertEqual(raw.support.model.calls, 0)

    def test_call_and_pixel_one_less_reject_atomically_without_raw_calls(self):
        for max_calls, max_pixels, reason in ((5, 72, "mllm_calls"), (6, 71, "processed_pixels")):
            with self.subTest(reason=reason):
                raw = RawObservationModel()
                result, ledger, budgeted = run_batch(
                    raw, answer_type="option_list", options=self.HR_OPTIONS,
                    max_calls=max_calls, max_pixels=max_pixels,
                )
                self.assertEqual(result.status, "budget_rejected")
                self.assertIn(reason, result.failure_reason)
                self.assertEqual((ledger.mllm_calls, ledger.processed_pixels), (0, 0))
                self.assertEqual(raw.calls, [])
                self.assertFalse(budgeted._answer_started)

    def test_vstar_cost_is_n_plus_three_for_every_option_count(self):
        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                options = tuple(f"option-{index}" for index in range(count))
                calls = count + 3
                raw = RawObservationModel()
                result, ledger, _ = run_batch(
                    raw, answer_type="logits_match", options=options,
                    max_calls=calls, max_pixels=calls * 12,
                )
                self.assertEqual(result.status, "success")
                self.assertEqual(result.candidate_answer["winner"], min(1, count - 1))
                self.assertEqual(len(result.candidate_answer["losses"]), count)
                self.assertEqual((ledger.mllm_calls, ledger.processed_pixels),
                                 (calls, calls * 12))
                self.assertEqual([call[0] for call in raw.calls],
                                 ["support", "support", "vstar_answer"])

    def test_vstar_n4_short_boundaries_and_raw_exception_are_atomic(self):
        options = tuple(f"option-{index}" for index in range(4))
        for max_calls, max_pixels, reason in (
            (6, 84, "mllm_calls"),
            (7, 83, "processed_pixels"),
        ):
            with self.subTest(reason=reason):
                raw = RawObservationModel()
                result, ledger, _ = run_batch(
                    raw, answer_type="logits_match", options=options,
                    max_calls=max_calls, max_pixels=max_pixels,
                )
                self.assertEqual(result.status, "budget_rejected")
                self.assertIn(reason, result.failure_reason)
                self.assertEqual((ledger.mllm_calls, ledger.processed_pixels), (0, 0))
                self.assertEqual(raw.calls, [])

        failing = RawObservationModel(fail_phase="vstar_answer")
        result, ledger, _ = run_batch(
            failing, answer_type="logits_match", options=options,
            max_calls=7, max_pixels=84,
        )
        self.assertEqual(result.status, "model_failed")
        self.assertEqual(result.failure_phase, "vstar_answer")
        self.assertIsNone(result.candidate_answer)
        self.assertEqual((ledger.mllm_calls, ledger.processed_pixels), (7, 84))

    def test_zero_requirements_do_not_charge_or_call_raw_model(self):
        raw = RawObservationModel()
        result, ledger, _ = run_batch(
            raw, answer_type="option_list", options=self.HR_OPTIONS,
            max_calls=6, max_pixels=72, evidence=(),
        )
        self.assertEqual(result.status, "no_requirements")
        self.assertEqual(result.failure_reason, "no_requirements")
        self.assertEqual((ledger.mllm_calls, ledger.processed_pixels), (0, 0))
        self.assertEqual(raw.calls, [])

    def test_each_raw_exception_stays_fully_charged_and_cannot_return_partial_answer(self):
        for phase in ("current_support", "candidate_support", "hr_answer_0", "hr_answer_2"):
            with self.subTest(phase=phase):
                raw = RawObservationModel(fail_phase=phase)
                result, ledger, _ = run_batch(
                    raw, answer_type="option_list", options=self.HR_OPTIONS,
                    max_calls=6, max_pixels=72,
                )
                self.assertEqual(result.status, "model_failed")
                self.assertEqual(result.failure_phase, phase)
                self.assertIsNone(result.candidate_answer)
                self.assertFalse(result.promotable)
                self.assertEqual((ledger.mllm_calls, ledger.processed_pixels), (6, 72))
                json.dumps(result.to_dict(), allow_nan=False)

    def test_invalid_hr_output_and_vstar_winner_loss_mismatch_fail_fully_charged(self):
        hr = RawObservationModel()
        outputs = iter(("A", {"not": "a string"}, "C", "D"))
        hr.free_form_using_nodes = MethodType(
            lambda model, image, question, nodes: next(outputs), hr,
        )
        hr_result, hr_ledger, _ = run_batch(
            hr, answer_type="option_list", options=self.HR_OPTIONS,
            max_calls=6, max_pixels=72,
        )
        self.assertEqual(hr_result.status, "model_failed")
        self.assertEqual(hr_result.failure_phase, "hr_answer_1")
        self.assertIsNone(hr_result.candidate_answer)
        self.assertEqual((hr_ledger.mllm_calls, hr_ledger.processed_pixels), (6, 72))

        vstar = RawObservationModel()
        vstar.multiple_choices_with_losses = MethodType(
            lambda model, image, question, options, nodes: (0, [0.4, 0.1]), vstar,
        )
        vstar_result, vstar_ledger, _ = run_batch(
            vstar, answer_type="logits_match", options=("a", "b"),
            max_calls=5, max_pixels=60,
        )
        self.assertEqual(vstar_result.status, "model_failed")
        self.assertEqual(vstar_result.failure_phase, "vstar_answer")
        self.assertIsNone(vstar_result.candidate_answer)
        self.assertEqual((vstar_ledger.mllm_calls, vstar_ledger.processed_pixels), (5, 60))

    def test_positive_unconsumed_public_answer_reserve_rejects_without_charge(self):
        image = Image.new("RGB", (4, 3), (1, 2, 3))
        raw = RawObservationModel()
        ledger = BudgetLedger(20, 240)
        budgeted = _BudgetedZoomModel(
            raw, ledger, answer_reserve_calls=4,
            answer_image_loader=lambda: image.copy(), answer_type="option_list",
        )
        result = budgeted.post_anchor_observation_batch(
            source_image=image, q0="What is visible?",
            query_plan=QueryPlan(main_query="What is visible?", evidence_items=evidence_items()),
            current_support_view=(),
            candidate_support_view=(next_candidate_for(image),),
            answer_type="option_list", options=self.HR_OPTIONS,
        )
        self.assertEqual(result.status, "budget_rejected")
        self.assertEqual(result.failure_reason, "public_answer_reserve_unconsumed")
        self.assertEqual((ledger.mllm_calls, ledger.processed_pixels), (0, 0))
        self.assertEqual(raw.calls, [])
        self.assertFalse(budgeted._answer_started)


if __name__ == "__main__":
    unittest.main()
