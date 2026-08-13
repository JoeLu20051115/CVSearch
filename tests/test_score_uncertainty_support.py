import copy
import json
import tempfile
import unittest
from pathlib import Path

from cvsearch.eval.score_uncertainty_support import (
    evaluate_locked_gates,
    generate_decisions,
    main,
    score_decisions,
)
from cvsearch.eval.replay_uncertainty_support import (
    UnifiedPolicy,
    UtilityIsotonicCalibrator,
)
from tests.test_replay_split_search import calibration, make_hr, rescue_rows
from tests.test_freeze_uncertainty_support import record


def frozen_policy():
    return UnifiedPolicy(
        profile="balanced",
        threshold=0.0,
        raw_support_floor=0.2,
        utility_calibrator=UtilityIsotonicCalibrator((1.0,), (1.0,)),
        payload_sha256="a" * 64,
    )


def hr_cell():
    phase1, split = rescue_rows()
    make_hr(phase1, split)
    for row in (phase1, split):
        row["_eg_ordinal"] = 7
        row["answer"] = ["B"] * 4
    branches = split["method_trace"]["steps"][0]["split_search_audit"][
        "branches"
    ]
    for branch in branches:
        for role in ("tight_view", "medium_view", "context_view"):
            if role in branch:
                branch[role]["answer"] = ["A"] * 4
                branch[role]["raw_support"] = 0.9
    for role in ("tight_view", "context_view"):
        branches[0][role]["answer"] = ["B"] * 4
    return phase1, split


class DecisionGenerationTests(unittest.TestCase):
    def test_decisions_do_not_change_when_evaluator_labels_change(self):
        phase1, split = hr_cell()
        cells = {"qwen/hr_bench_4k": [phase1]}
        splits = {"qwen/hr_bench_4k": [split]}
        first = generate_decisions(
            cells, splits, {"qwen": calibration()}, frozen_policy(),
        )
        phase1["answer"] = ["A"] * 4
        phase1["category"] = "poison"
        split["answer"] = ["C"] * 4
        split["category"] = "different"

        second = generate_decisions(
            cells, splits, {"qwen": calibration()}, frozen_policy(),
        )

        self.assertEqual(first, second)
        serialized = repr(first).lower()
        self.assertNotIn("category", serialized)
        self.assertNotIn("candidate_correct", serialized)
        self.assertNotIn("stage3_correct", serialized)

    def test_hr_official_units_are_scored_atomically(self):
        phase1, split = hr_cell()
        decisions = generate_decisions(
            {"qwen/hr_bench_4k": [phase1]},
            {"qwen/hr_bench_4k": [split]},
            {"qwen": calibration()},
            frozen_policy(),
        )

        report = score_decisions(
            {"qwen/hr_bench_4k": [phase1]}, decisions,
            expected_topics=1, expected_units=4,
        )

        cell = report["cells"]["qwen/hr_bench_4k"]
        self.assertEqual(cell["official_units"], 4)
        self.assertEqual(cell["baseline_correct"], 0)
        self.assertEqual(cell["selected_correct"], 4)
        self.assertEqual(cell["corrections"], 4)
        self.assertEqual(report["aggregate"]["official_units"], 4)

    def test_input_alignment_and_policy_hash_are_strict(self):
        phase1, split = hr_cell()
        bad = copy.deepcopy(split)
        bad["_eg_ordinal"] = 8
        with self.assertRaises(ValueError):
            generate_decisions(
                {"qwen/hr_bench_4k": [phase1]},
                {"qwen/hr_bench_4k": [bad]},
                {"qwen": calibration()},
                frozen_policy(),
            )

    def test_fixed_observation_contract_binds_sixteen_probes_or_explicit_noop(self):
        phase1, split = hr_cell()
        audit = split["method_trace"]["steps"][0]["split_search_audit"]
        audit["screening_probes"] = [{"ordinal": index} for index in range(16)]
        valid = generate_decisions(
            {"qwen/hr_bench_4k": [phase1]},
            {"qwen/hr_bench_4k": [split]},
            {"qwen": calibration()},
            frozen_policy(),
        )
        audit["max_screening_probes"] = 15
        drifted = generate_decisions(
            {"qwen/hr_bench_4k": [phase1]},
            {"qwen/hr_bench_4k": [split]},
            {"qwen": calibration()},
            frozen_policy(),
        )

        self.assertTrue(valid["inputs"]["qwen/hr_bench_4k"]["fixed_observations"])
        self.assertFalse(drifted["inputs"]["qwen/hr_bench_4k"]["fixed_observations"])

    def test_explicit_split_noop_serializes_exact_fallback_detail(self):
        phase1, split = hr_cell()
        audit = split["method_trace"]["steps"][0]["split_search_audit"]
        audit["no_op_reason"] = "split_invalid_evidence_requirements"
        audit["screening_probes"] = []
        audit["root_ranked_siblings"] = []
        audit["branches"] = []

        artifact = generate_decisions(
            {"qwen/hr_bench_4k": [phase1]},
            {"qwen/hr_bench_4k": [split]},
            {"qwen": calibration()},
            frozen_policy(),
        )

        self.assertEqual(
            artifact["decisions"][0]["failure_detail"],
            "frozen SPLIT no-op: split_invalid_evidence_requirements",
        )


class SelectorCliTests(unittest.TestCase):
    def test_development_freeze_cli_is_deterministic_and_cpu_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_root = root / "development"
            examples = (
                record("g1", "qwen", helpful=True),
                record("g2", "internvl", helpful=True, ordinal=1),
            )
            for example in examples:
                path = split_root / example.backbone / "treebench.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(example.split_row, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            policy_path = root / "policy.json"
            report_path = root / "report.json"
            support_path = Path(
                "reproduction/evidence_gap/adaptive_search_v7/"
                "split-calibration-manifest-v2.json"
            )
            arguments = [
                "freeze-development",
                "--split-root", str(split_root),
                "--support-calibration", str(support_path),
                "--policy-out", str(policy_path),
                "--report-out", str(report_path),
            ]

            self.assertEqual(main(arguments), 0)
            first_policy = policy_path.read_bytes()
            first_report = report_path.read_bytes()
            self.assertEqual(main(arguments), 0)

            self.assertEqual(policy_path.read_bytes(), first_policy)
            self.assertEqual(report_path.read_bytes(), first_report)
            payload = json.loads(first_policy)
            self.assertEqual(payload["data_scope"], "opened_development_only")
            self.assertNotIn("validation_v3", first_policy.decode())


def passing_locked_report():
    baseline = {
        "qwen/hr_bench_4k": (48, 33, 33),
        "qwen/hr_bench_8k": (48, 43, 43),
        "qwen/treebench": (12, 7, 7),
        "qwen/vstar": (20, 20, 20),
        "internvl/hr_bench_4k": (48, 27, 28),
        "internvl/hr_bench_8k": (48, 44, 44),
        "internvl/treebench": (12, 3, 3),
        "internvl/vstar": (20, 18, 18),
    }
    return {
        "cells": {
            key: {
                "official_units": units,
                "baseline_correct": before,
                "selected_correct": after,
                "delta": after - before,
            }
            for key, (units, before, after) in baseline.items()
        },
        "aggregate": {
            "topics": 112,
            "official_units": 256,
            "baseline_correct": 195,
            "selected_correct": 196,
            "corrections": 1,
            "corruptions": 0,
        },
        "audits": {
            "accounting": True,
            "input_hashes": True,
            "policy_hash": True,
            "decision_hash": True,
            "fixed_observations": True,
        },
    }


class LockedGateTests(unittest.TestCase):
    def test_declared_locked_gate_passes_only_exact_no_harm_scope(self):
        gates = evaluate_locked_gates(passing_locked_report())

        self.assertTrue(gates["passed"])
        self.assertEqual(gates["failures"], [])

    def test_gate_reports_qwen_hr4_and_cell_regression(self):
        report = passing_locked_report()
        report["cells"]["qwen/hr_bench_4k"]["selected_correct"] = 32
        report["cells"]["qwen/hr_bench_4k"]["delta"] = -1
        report["aggregate"]["selected_correct"] = 195

        gates = evaluate_locked_gates(report)

        self.assertFalse(gates["passed"])
        self.assertIn("qwen/hr_bench_4k below 33/48", gates["failures"])
        self.assertIn(
            "qwen/hr_bench_4k regressed below CVSearch",
            gates["failures"],
        )


if __name__ == "__main__":
    unittest.main()
