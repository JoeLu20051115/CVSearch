import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from cvsearch.evidence_gap.method import get_evidence_gap_response, load_method_config
from cvsearch.evidence_gap.types import EXPAND, FORCED_RETURN, QueryPlan, ZOOM

from tests.test_evidence_gap_expand_observation import (
    RuntimeExpandRaw,
    expand_config,
)
from tests.test_evidence_gap_search_state import candidate_snapshot, event_for
from tests.test_evidence_gap_zoom_observation import gradient_image


CONFIG_ROOT = Path(__file__).parents[1] / "reproduction/evidence_gap/configs"
DISABLED_CONFIG = CONFIG_ROOT / (
    "dev_unified_zoom_expand_disabled_gamma000_budget512.json"
)
ENABLED_CONFIG = CONFIG_ROOT / (
    "dev_unified_zoom_expand_observation_gamma000_budget512.json"
)
ZOOM_KEYS = (
    "p2c_zoom_enabled",
    "p2c_zoom_admission_mode",
    "p2c_zoom_replacement_enabled",
    "p2c_zoom_render_policy",
)
EXPAND_KEYS = (
    "p4a_expand_enabled",
    "p4a_expand_admission_mode",
    "p4a_expand_replacement_enabled",
    "p4a_expand_selection_policy",
)


def combined_config(*, enabled=True, **overrides):
    config = expand_config(enabled=enabled)
    config.update({
        "config_id": (
            "unified-zoom-expand-observation-gamma000-v1"
            if enabled else "unified-zoom-expand-disabled-gamma000-v1"
        ),
        "p2c_zoom_enabled": enabled,
        "p2c_zoom_admission_mode": "all_feasible" if enabled else "disabled",
    })
    config.update(overrides)
    return config


def without_elapsed(value):
    if isinstance(value, dict):
        return {
            key: without_elapsed(item)
            for key, item in value.items()
            if key != "elapsed_seconds"
        }
    if isinstance(value, list):
        return [without_elapsed(item) for item in value]
    return value


class CombinedRuntimeRaw(RuntimeExpandRaw):
    """Existing complete runtime double with repeatable answers for both actions."""

    def free_form_using_nodes(self, image, question, nodes):
        phase = f"hr_answer_{self.zoom_hr_index % 4}"
        self.answer_inputs.append(
            (self.view_size, image.mode, image.size, image.tobytes(), tuple(nodes))
        )
        if self.fail_phase == phase:
            raise RuntimeError(f"failed {phase}")
        output = ("zoom-A", "zoom-B", "zoom-C", "zoom-D")[
            self.zoom_hr_index % 4
        ]
        self.zoom_hr_index += 1
        return output


class ZoomCandidateSupportOnceFailureRaw(CombinedRuntimeRaw):
    def evidence_support(self, **kwargs):
        try:
            return super().evidence_support(**kwargs)
        except RuntimeError:
            self.fail_phase = None
            raise


class CombinedProducerTest(unittest.TestCase):
    HR_OPTIONS = [
        "A. cat\nB. dog\nC. bird\nD. fish",
        "A. dog\nB. cat\nC. fish\nD. bird",
        "A. cat\nB. dog\nC. fish\nD. bird",
        "A. dog\nB. cat\nC. bird\nD. fish",
    ]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image_path = Path(self.directory.name) / "source.png"
        gradient_image().save(self.image_path)

    def run_combined(self, *, raw=None, enabled=True, original_annotation=None):
        raw = CombinedRuntimeRaw() if raw is None else raw
        raw_response = ["A", "B", "A", "B"]
        policy = {
            "question": "What evidence is visible?",
            "options": deepcopy(self.HR_OPTIONS),
            "answer_type": "option_list",
            "input_image": str(self.image_path),
        }
        calls = []
        captured = {}

        def fake_cvsearch(**kwargs):
            calls.append(kwargs)
            if not enabled:
                self.assertIsNone(kwargs["search_state_sink"])
                return deepcopy(raw_response)
            sink = kwargs["search_state_sink"]
            self.assertIsNotNone(sink)
            image = gradient_image()
            focus = candidate_snapshot(
                (5, 4, 4, 3), posterior=0.9, source="fine",
            )
            context = candidate_snapshot(
                (18, 10, 4, 4), posterior=0.8, depth=2, source="fine",
            )
            refs, snapshot = event_for(
                image, (focus, context), event="p0_selected",
                selected=(focus["canonical_key"],),
                remaining=(context["canonical_key"],),
            )
            sink(refs, snapshot)
            captured["collector"] = sink
            captured["collector_before"] = deepcopy(sink.to_dict())
            node = type("LiveNode", (), {})()
            node.depth = focus["depth"]
            node.state = type("State", (), {
                "bbox": focus["bbox_original"],
                "original_image_pil": image.copy(),
            })()
            kwargs["annotation"]["search_mode"] = 1
            kwargs["answer_observer"]("search", [node], deepcopy(raw_response))
            return deepcopy(raw_response)

        output, trace = get_evidence_gap_response(
            sam_model=object(), zoom_model=raw, nlp_model=object(),
            policy_annotation=policy,
            original_annotation=(
                {"benchmark": "poison", "resolution": "8K", "category": "poison"}
                if original_annotation is None else original_annotation
            ),
            ic_examples=[], decomposed_question_template="{}",
            config=combined_config(enabled=enabled), cvsearch_fn=fake_cvsearch,
            planner=lambda policy_snapshot, targets: QueryPlan(
                main_query=policy_snapshot["question"], targets=("object",),
                evidence_items=({
                    "kind": "target_detail",
                    "target": "object",
                    "requirements": ["presence", "visual_detail"],
                },),
            ),
        )
        if enabled:
            captured["collector_after"] = deepcopy(captured["collector"].to_dict())
        return output, trace, raw, raw_response, calls, captured

    def test_checked_in_siblings_are_exact_and_combined_config_is_strict(self):
        enabled = load_method_config(combined_config(enabled=True))
        disabled = load_method_config(combined_config(enabled=False))
        self.assertEqual(
            {key for key in enabled if enabled[key] != disabled[key]},
            {
                "config_id",
                "p2c_zoom_enabled",
                "p2c_zoom_admission_mode",
                "p4a_expand_enabled",
                "p4a_expand_admission_mode",
            },
        )
        self.assertFalse(enabled["p2c_zoom_replacement_enabled"])
        self.assertFalse(enabled["p4a_expand_replacement_enabled"])
        self.assertEqual(
            json.loads(ENABLED_CONFIG.read_text(encoding="utf-8")),
            combined_config(enabled=True),
        )
        self.assertEqual(
            json.loads(DISABLED_CONFIG.read_text(encoding="utf-8")),
            combined_config(enabled=False),
        )

        for missing in ZOOM_KEYS + EXPAND_KEYS:
            partial = combined_config()
            partial.pop(missing)
            with self.subTest(missing=missing), self.assertRaisesRegex(
                ValueError, "all-or-none",
            ):
                load_method_config(partial)

        invalid = (
            {"p2c_zoom_admission_mode": "disabled"},
            {"p4a_expand_admission_mode": "disabled"},
            {"p2c_zoom_render_policy": "question_router"},
            {"p4a_expand_selection_policy": "resolution_router"},
            {"p2c_zoom_replacement_enabled": True},
            {"p4a_expand_replacement_enabled": True},
            {"p4a_expand_enabled": False, "p4a_expand_admission_mode": "disabled"},
            {"next_enabled": True, "next_admission_mode": "all_feasible"},
            {"quick_gate": 0.8},
            {"max_mllm_calls": 511},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                load_method_config(combined_config(**changes))

    def test_one_pass_orders_actions_binds_exact_p0_and_chains_cumulative_ledger(self):
        output, trace, raw, p0, calls, captured = self.run_combined()

        self.assertEqual(len(calls), 1)
        self.assertEqual([step.action for step in trace.steps], [
            ZOOM, EXPAND, FORCED_RETURN,
        ])
        self.assertEqual(output, p0)
        self.assertEqual(trace.final_answer.output, p0)
        self.assertEqual(trace.anchor_answer.output, p0)
        zoom_step, expand_step, final_step = trace.steps
        zoom = zoom_step.zoom_audit.to_dict()
        expand = expand_step.expand_audit.to_dict()
        for audit in (zoom, expand):
            self.assertEqual(audit["p0_anchor"]["emitted_answer"], p0)
            self.assertEqual(audit["p0_anchor"]["cvsearch_raw"], p0)
            self.assertEqual(audit["p0_stability"], zoom["p0_stability"])
        self.assertEqual(zoom_step.answer.output, p0)
        self.assertEqual(expand_step.answer.output, p0)
        self.assertEqual(final_step.answer.output, p0)
        self.assertEqual(
            zoom["batch_result"]["ledger_after"],
            expand["batch_result"]["ledger_before"],
        )
        self.assertEqual(zoom_step.budget.to_dict(), zoom["batch_result"]["ledger_after"])
        self.assertEqual(expand_step.budget.to_dict(), expand["batch_result"]["ledger_after"])
        self.assertEqual(final_step.budget.to_dict(), expand["batch_result"]["ledger_after"])
        self.assertEqual(trace.budget.to_dict(), expand["batch_result"]["ledger_after"])
        self.assertEqual(captured["collector_after"], captured["collector_before"])
        self.assertEqual(expand["batch_result"]["batch_plan"]["base_view_size"], 12)
        self.assertEqual(raw.view_size, 12)

    def test_zoom_success_noop_or_model_failure_never_suppresses_expand(self):
        cases = (
            ("success", CombinedRuntimeRaw(), "success"),
            ("no_op", CombinedRuntimeRaw(identical_render=True), "render_noop"),
            (
                "model_failed",
                ZoomCandidateSupportOnceFailureRaw(fail_phase="candidate_support"),
                "model_failed",
            ),
        )
        for name, raw, zoom_status in cases:
            with self.subTest(name=name):
                output, trace, raw, p0, calls, _ = self.run_combined(raw=raw)
                self.assertEqual(output, p0)
                self.assertEqual(len(calls), 1)
                self.assertEqual([step.action for step in trace.steps], [
                    ZOOM, EXPAND, FORCED_RETURN,
                ])
                zoom_step, expand_step, _ = trace.steps
                self.assertEqual(zoom_step.zoom_audit.batch_result.status, zoom_status)
                self.assertIsNotNone(expand_step.expand_audit.batch_result)
                self.assertEqual(
                    expand_step.expand_audit.batch_result.to_dict()["ledger_before"],
                    zoom_step.zoom_audit.batch_result.to_dict()["ledger_after"],
                )
                self.assertEqual(raw.view_size, 12)

    def test_disabled_mode_and_evaluator_poison_cannot_route_combined_actions(self):
        output, trace, _, p0, calls, _ = self.run_combined(enabled=False)
        self.assertEqual(output, p0)
        self.assertEqual(len(calls), 1)
        self.assertEqual([step.action for step in trace.steps], [FORCED_RETURN])

        poisons = (
            {
                "benchmark": "POISON_BENCHMARK_ALPHA",
                "resolution": "POISON_RESOLUTION_ALPHA",
                "category": "POISON_CATEGORY_ALPHA",
                "question": "POISON_QUESTION_ALPHA",
            },
            {
                "benchmark": "POISON_BENCHMARK_OMEGA",
                "resolution": "POISON_RESOLUTION_OMEGA",
                "category": "POISON_CATEGORY_OMEGA",
                "question": "POISON_QUESTION_OMEGA",
            },
        )
        observed = []
        for poison in poisons:
            output, trace, _, p0, calls, _ = self.run_combined(
                original_annotation=poison,
            )
            self.assertEqual(output, p0)
            self.assertEqual(len(calls), 1)
            self.assertEqual([step.action for step in trace.steps], [
                ZOOM, EXPAND, FORCED_RETURN,
            ])
            audits = [
                trace.steps[0].zoom_audit.to_dict(),
                trace.steps[1].expand_audit.to_dict(),
            ]
            encoded = json.dumps(audits, sort_keys=True)
            for value in poison.values():
                self.assertNotIn(value, encoded)
            observed.append(without_elapsed(audits))
        self.assertEqual(observed[0], observed[1])


if __name__ == "__main__":
    unittest.main()
