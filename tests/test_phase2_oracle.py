import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cvsearch.eval.phase2_oracle import (
    PairExpectation,
    bootstrap_draw_index,
    canonical_output_digest,
    score_paired_rows,
    validate_frozen_dev_identity,
    validate_launch_pair,
)
from cvsearch.evidence_gap.provenance import canonical_sha256
from cvsearch.evidence_gap.method import load_method_config


ROOT = Path(__file__).resolve().parents[1]
ENABLED_CONFIG_PATH = (
    ROOT / "reproduction/evidence_gap/configs/"
    "dev_unified_next_oracle_gamma000_budget512.json"
)
DISABLED_CONFIG_PATH = (
    ROOT / "reproduction/evidence_gap/configs/"
    "dev_unified_next_disabled_gamma000_budget512.json"
)


def _configs():
    enabled = load_method_config(str(ENABLED_CONFIG_PATH))
    disabled = load_method_config(str(DISABLED_CONFIG_PATH))
    return disabled, enabled


def _answer(output):
    return {"output": copy.deepcopy(output)}


def _row(benchmark, ordinal, *, p0, candidate, truth, feasible=True):
    disabled_config, enabled_config = _configs()
    revision = "a" * 64
    image = f"image/{ordinal}.jpg"
    answer_type = "logits_match" if benchmark == "vstar" else "option_list"
    options = ["red", "blue"] if benchmark == "vstar" else [
        "A. cat\nB. dog\nC. bird\nD. fish",
        "A. dog\nB. cat\nC. fish\nD. bird",
        "A. bird\nB. fish\nC. cat\nD. dog",
        "A. fish\nB. bird\nC. dog\nD. cat",
    ]
    budget_before = {
        "max_mllm_calls": 512, "max_processed_pixels": 10_000_000_000,
        "mllm_calls": 10, "processed_pixels": 100,
    }
    budget_after = dict(
        budget_before, mllm_calls=16, processed_pixels=200,
    )
    base = {
        "question": f"q{ordinal}", "options": options,
        "answer_type": answer_type, "input_image": image,
        "answer": copy.deepcopy(truth), "output": copy.deepcopy(p0),
        "_eg_ordinal": ordinal, "_eg_code_revision": revision,
        "_eg_run_fingerprint": "b" * 64,
    }
    disabled = copy.deepcopy(base)
    disabled["method_trace"] = {
        "steps": [{"action": "FORCED_RETURN"}],
        "final_answer": _answer(p0), "anchor_answer": _answer(p0),
        "budget": budget_before, "elapsed_seconds": 1.0,
        "termination": "FORCED_RETURN", "config_id": disabled_config["config_id"],
        "effective_config": disabled_config, "budget_interrupted": False,
        "pixel_accounting": disabled_config["pixel_accounting"],
        "replacement_margin": 0.0, "support_status": "not_observed",
    }
    enabled = copy.deepcopy(base)
    enabled["_eg_run_fingerprint"] = "c" * 64
    audit = {
        "p0_anchor": {
            "emitted_answer": copy.deepcopy(p0),
            "cvsearch_raw": copy.deepcopy(p0),
            "producing_phase": "search" if benchmark == "vstar" else "cvsearch_raw",
            "node_keys": ["p0-key"],
            "support_view": [{"canonical_key": "p0-key"}],
        },
        "feasible": feasible, "support_contract_status": "matched" if feasible else "not_observed",
        "coverage_status": "not_observed",
        "verifier_status": "disabled_same_checkpoint_unpromoted",
        "verifier_avg": None, "verifier_min": None, "score_margin": None,
        "score_status": "unavailable_missing_verifier_coverage",
        "g_next": 0.4 if feasible else None,
        "support_delta": 0.2 if feasible else None,
        "normalized_actual_cost": 6 / 512 if feasible else None,
        "replacement_reason": "replacement_disabled_p2a" if feasible else None,
        # This deliberately disagrees for HR: the oracle must score candidate_answer.
        "candidate_stability": {"output": copy.deepcopy(p0)} if feasible else None,
        "batch_result": None,
    }
    if feasible:
        plan_without_hash = {
            "answer_type": answer_type,
            "total_calls": 6,
            "total_pixels": 100,
        }
        plan_hash = hashlib.sha256(json.dumps(
            plan_without_hash, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        batch_plan = dict(plan_without_hash, plan_hash=plan_hash)
        audit["batch_result"] = {
            "status": "success", "admitted": True, "charged": True,
            "candidate_answer": copy.deepcopy(candidate),
            "batch_plan": batch_plan, "batch_plan_hash": plan_hash,
            "ledger_before": budget_before, "ledger_after": budget_after,
            "elapsed_seconds": 0.25,
            "current_support": {"p_yes": 0.6},
            "candidate_support": {"p_yes": 0.8},
            "verifier_status": "disabled_same_checkpoint_unpromoted",
            "verifier_avg": None, "verifier_min": None, "promotable": False,
        }
    enabled["method_trace"] = {
        "steps": [
            {"action": "NEXT", "feasible_actions": ["NEXT"] if feasible else [],
             "no_op_reason": None if feasible else "next_queue_empty", "next_audit": audit},
            {"action": "FORCED_RETURN"},
        ],
        "final_answer": _answer(p0), "anchor_answer": _answer(p0),
        "budget": budget_after if feasible else budget_before, "elapsed_seconds": 1.5,
        "termination": "FORCED_RETURN", "config_id": enabled_config["config_id"],
        "effective_config": enabled_config, "budget_interrupted": False,
        "pixel_accounting": enabled_config["pixel_accounting"],
        "anchor_state_score": None, "selected_state_score": None,
        "replacement_margin": None,
        "support_status": "observed_answer_free" if feasible else "next_queue_empty",
    }
    return disabled, enabled


class Phase2OracleTest(unittest.TestCase):
    def test_frozen_dev_identity_binds_benchmark_split_seed_chunk_and_ordinals(self):
        ordinals = [
            3, 4, 19, 24, 33, 39, 41, 48, 50, 62, 66, 78, 84, 85, 87,
            94, 98, 101, 103, 104, 105, 116, 124, 132, 136, 138, 142,
            145, 149, 153, 155, 158, 169, 174, 182, 186, 188,
        ]
        manifest = {
            "benchmark": "vstar",
            "selected_partition": {
                "ordinals": ordinals, "rows": 37, "split": "dev",
                "split_seed": 260809, "num_chunks": 1, "chunk_idx": 0,
            },
        }
        rows = [{"_eg_ordinal": ordinal} for ordinal in ordinals]
        validate_frozen_dev_identity("vstar", manifest, rows)
        mutations = (
            ("benchmark", lambda value: value.__setitem__("benchmark", "hr-bench_4k")),
            ("split", lambda value: value["selected_partition"].__setitem__("split", "holdout")),
            ("seed", lambda value: value["selected_partition"].__setitem__("split_seed", 260810)),
            ("chunks", lambda value: value["selected_partition"].__setitem__("num_chunks", 2)),
            ("chunk", lambda value: value["selected_partition"].__setitem__("chunk_idx", 1)),
            ("ordinals", lambda value: value["selected_partition"]["ordinals"].__setitem__(0, 2)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                corrupted = copy.deepcopy(manifest)
                mutate(corrupted)
                with self.assertRaises(ValueError):
                    validate_frozen_dev_identity("vstar", corrupted, rows)

    def test_bootstrap_draw_uses_first_eight_sha256_digest_bytes(self):
        self.assertEqual(bootstrap_draw_index("vstar", 0, 0, 37), 36)

    def test_disabled_config_differs_only_in_the_three_frozen_control_fields(self):
        disabled, enabled = _configs()
        changed = {
            key for key in enabled if enabled.get(key) != disabled.get(key)
        }
        self.assertEqual(
            changed, {"config_id", "next_enabled", "next_admission_mode"},
        )
        self.assertEqual(disabled["next_enabled"], False)
        self.assertEqual(disabled["next_admission_mode"], "disabled")
        self.assertEqual(disabled["next_replacement_enabled"], False)

    def test_vstar_oracle_uses_feasible_winner_and_reports_deterministic_ci(self):
        pairs = (
            _row("vstar", 0, p0=0, candidate={"winner": 1, "losses": [1.0, 0.1]}, truth=0),
            _row("vstar", 1, p0=1, candidate={"winner": 0, "losses": [0.1, 1.0]}, truth=0),
        )
        disabled_rows = [pair[0] for pair in pairs]
        enabled_rows = [pair[1] for pair in pairs]
        expected_digest = canonical_output_digest(disabled_rows)
        report = score_paired_rows(
            "vstar", disabled_rows, enabled_rows,
            PairExpectation(topics=2, cycles=2, p0_correct=1, output_digest=expected_digest),
            bootstrap_replicates=10_000,
        )
        self.assertEqual(report["p0"]["correct"], 1)
        self.assertEqual(report["candidate_alone"]["correct"], 1)
        self.assertEqual(report["oracle"]["correct"], 2)
        self.assertEqual(report["oracle"]["corrected"], 1)
        self.assertEqual(report["oracle"]["corrupted"], 0)
        self.assertEqual(report["candidate"]["feasible_topics"], 2)
        self.assertEqual(report["bootstrap"]["replicates"], 10_000)
        self.assertEqual(report["bootstrap"]["index_rule"], [249, 9749])
        self.assertEqual(
            report["bootstrap"],
            score_paired_rows(
                "vstar", disabled_rows, enabled_rows,
                PairExpectation(2, 2, 1, expected_digest),
                bootstrap_replicates=10_000,
            )["bootstrap"],
        )

    def test_hr_oracle_selects_one_whole_state_per_topic_and_never_stability_output(self):
        p0 = ["A", "A", "A", "A"]
        candidate = ["B answer", "B answer", "C answer", "D answer"]
        truth = ["A", "B", "C", "D"]
        disabled, enabled = _row(
            "hr-bench_4k", 7, p0=p0, candidate=candidate, truth=truth,
        )
        digest = canonical_output_digest([disabled])
        report = score_paired_rows(
            "hr-bench_4k", [disabled], [enabled],
            PairExpectation(1, 4, 1, digest), bootstrap_replicates=10_000,
        )
        self.assertEqual(report["p0"]["correct"], 1)
        self.assertEqual(report["candidate_alone"]["correct"], 3)
        self.assertEqual(report["oracle"]["correct"], 3)
        self.assertEqual(report["oracle"]["selected_candidate_topics"], 1)
        self.assertNotEqual(report["oracle"]["correct"], 4)

    def test_tied_hr_topic_retains_p0_even_when_candidate_strings_change(self):
        disabled, enabled = _row(
            "hr-bench_8k", 2,
            p0=["A", "B", "wrong", "wrong"],
            candidate=["wrong", "wrong", "C", "D"],
            truth=["A", "B", "C", "D"],
        )
        report = score_paired_rows(
            "hr-bench_8k", [disabled], [enabled],
            PairExpectation(1, 4, 2, canonical_output_digest([disabled])),
            bootstrap_replicates=10_000,
        )
        self.assertEqual(report["oracle"]["correct"], 2)
        self.assertEqual(report["oracle"]["selected_candidate_topics"], 0)

    def test_pair_and_trace_validation_fail_closed_before_scoring_labels(self):
        disabled, enabled = _row(
            "vstar", 0, p0=0,
            candidate={"winner": 1, "losses": [1.0, 0.1]}, truth=0,
        )
        expectation = PairExpectation(1, 1, 1, canonical_output_digest([disabled]))
        mutations = (
            ("ordinal", lambda row: row.__setitem__("_eg_ordinal", 9)),
            ("revision", lambda row: row.__setitem__("_eg_code_revision", "d" * 64)),
            ("input", lambda row: row.__setitem__("input_image", "other.jpg")),
            ("answer", lambda row: row.__setitem__("answer", 99)),
            ("emitted", lambda row: row.__setitem__("output", 1)),
            ("anchor", lambda row: row["method_trace"]["steps"][0]["next_audit"]["p0_anchor"].__setitem__("emitted_answer", 1)),
            ("raw", lambda row: row["method_trace"]["steps"][0]["next_audit"]["p0_anchor"].__setitem__("cvsearch_raw", 1)),
            ("phase", lambda row: row["method_trace"]["steps"][0]["next_audit"]["p0_anchor"].__setitem__("producing_phase", "response")),
            ("replacement", lambda row: row["method_trace"]["steps"][0]["next_audit"].__setitem__("replacement_reason", "promoted")),
            ("budget", lambda row: row["method_trace"]["budget"].__setitem__("max_mllm_calls", 511)),
            ("forced", lambda row: row["method_trace"].__setitem__("termination", "CERTIFIED_STOP")),
            ("config", lambda row: row["method_trace"]["effective_config"].__setitem__("quick_gate", 0.8)),
            ("status", lambda row: row["method_trace"]["steps"][0]["next_audit"]["batch_result"].__setitem__("status", "model_failed")),
            ("promotable", lambda row: row["method_trace"]["steps"][0]["next_audit"]["batch_result"].__setitem__("promotable", True)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                corrupted = copy.deepcopy(enabled)
                mutate(corrupted)
                with self.assertRaises((TypeError, ValueError)):
                    score_paired_rows(
                        "vstar", [disabled], [corrupted], expectation,
                        bootstrap_replicates=10_000,
                    )

    def test_vstar_root_anchor_binds_empty_root_view_without_forcing_search_raw_equal(self):
        disabled, enabled = _row(
            "vstar", 0, p0=0,
            candidate={"winner": 1, "losses": [1.0, 0.1]}, truth=0,
        )
        anchor = enabled["method_trace"]["steps"][0]["next_audit"]["p0_anchor"]
        anchor.update(
            producing_phase="root", cvsearch_raw=1,
            node_keys=[], support_view=[],
        )
        expectation = PairExpectation(1, 1, 1, canonical_output_digest([disabled]))
        report = score_paired_rows(
            "vstar", [disabled], [enabled], expectation,
            bootstrap_replicates=10_000,
        )
        self.assertEqual(report["p0"]["correct"], 1)
        anchor["node_keys"] = ["search-key"]
        with self.assertRaisesRegex(ValueError, "support view|root"):
            score_paired_rows(
                "vstar", [disabled], [enabled], expectation,
                bootstrap_replicates=10_000,
            )

    def test_launch_sidecars_bind_pair_identity_and_row_fingerprints(self):
        disabled, enabled = _row(
            "vstar", 0, p0=0,
            candidate={"winner": 1, "losses": [1.0, 0.1]}, truth=0,
        )
        disabled_config, enabled_config = _configs()
        common = {
            "schema_version": 1, "benchmark": "vstar",
            "code": {"revision": "a" * 64, "manifest": {"files": []}},
            "selected_partition": {"ordinals": [0], "rows_sha256": "d" * 64},
            "artifacts": {"qwen": {"sha256": "e" * 64}},
            "environment": {"python": "3.11"},
            "hardware": {"gpu_uuids": ["GPU-test"]},
        }
        disabled_manifest = copy.deepcopy(common)
        disabled_manifest["config"] = {"loaded": disabled_config}
        enabled_manifest = copy.deepcopy(common)
        enabled_manifest["config"] = {"loaded": enabled_config}
        disabled["_eg_run_fingerprint"] = canonical_sha256(disabled_manifest)
        enabled["_eg_run_fingerprint"] = canonical_sha256(enabled_manifest)
        validate_launch_pair(
            [disabled], [enabled], disabled_manifest, enabled_manifest,
        )

        corrupted = copy.deepcopy(enabled_manifest)
        corrupted["artifacts"]["qwen"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_launch_pair([disabled], [enabled], disabled_manifest, corrupted)
        corrupted = copy.deepcopy(enabled_manifest)
        corrupted["hardware"]["gpu_uuids"] = ["GPU-other"]
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_launch_pair([disabled], [enabled], disabled_manifest, corrupted)
        enabled["_eg_run_fingerprint"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_launch_pair(
                [disabled], [enabled], disabled_manifest, enabled_manifest,
            )

    def test_duplicate_or_reordered_pairs_and_wrong_frozen_reference_are_rejected(self):
        first = _row(
            "vstar", 0, p0=0,
            candidate={"winner": 1, "losses": [1.0, 0.1]}, truth=0,
        )
        second = _row(
            "vstar", 1, p0=1,
            candidate={"winner": 0, "losses": [0.1, 1.0]}, truth=0,
        )
        disabled = [first[0], second[0]]
        enabled = [second[1], first[1]]
        with self.assertRaisesRegex(ValueError, "order"):
            score_paired_rows(
                "vstar", disabled, enabled,
                PairExpectation(2, 2, 1, canonical_output_digest(disabled)),
                bootstrap_replicates=10_000,
            )
        with self.assertRaisesRegex(ValueError, "reference"):
            score_paired_rows(
                "vstar", disabled, [first[1], second[1]],
                PairExpectation(2, 2, 2, canonical_output_digest(disabled)),
                bootstrap_replicates=10_000,
            )

    def test_scorer_never_writes_or_mutates_jsonl_inputs(self):
        disabled, enabled = _row(
            "vstar", 0, p0=0,
            candidate={"winner": 1, "losses": [1.0, 0.1]}, truth=0,
        )
        before = json.dumps([disabled, enabled], sort_keys=True)
        score_paired_rows(
            "vstar", [disabled], [enabled],
            PairExpectation(1, 1, 1, canonical_output_digest([disabled])),
            bootstrap_replicates=10_000,
        )
        self.assertEqual(json.dumps([disabled, enabled], sort_keys=True), before)


if __name__ == "__main__":
    unittest.main()
