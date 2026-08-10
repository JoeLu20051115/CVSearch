import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cvsearch.eval.phase4_expand_oracle as phase4
from cvsearch.eval.phase7_uncertainty_confirmation import (
    BASE_SEARCH_CONFIG,
    DEFERRED_SEARCH_CONFIG,
    freeze_search_decisions,
    validate_and_extract_search_pairs,
)
from cvsearch.evidence_gap.method import load_method_config
from cvsearch.evidence_gap.method import build_query_plan
from cvsearch.evidence_gap.provenance import canonical_sha256
from cvsearch.evidence_gap.answers import aggregate_vstar_losses
from tests.test_phase4_expand_oracle import FROZEN_REVISION, PairFactory


def _config_identity(config, path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    files = [{"path": path.name, "sha256": digest, "size": path.stat().st_size}]
    material = {"kind": "file", "files": files}
    return {
        "loaded": copy.deepcopy(config),
        "loaded_sha256": canonical_sha256(config),
        "source": {
            "kind": "file", "path": str(path), "files": files,
            "sha256": canonical_sha256(material),
        },
    }


def _history(step, answer, cost):
    return {
        "step": step, "answer": copy.deepcopy(answer),
        "support_avg": 0.0, "support_min": 0.0, "cost": cost,
        "has_unvisited_branch": False, "state": {},
    }


def _record(output, losses, source):
    record = aggregate_vstar_losses([losses])
    if record.output != output:
        raise AssertionError("test loss winner differs from requested output")
    record.selected_from = source
    return record.to_dict()


class SearchPairFactory:
    def __init__(self, directory):
        self.factory = PairFactory(directory)
        self.base_config = load_method_config(BASE_SEARCH_CONFIG)
        self.search_config = load_method_config(DEFERRED_SEARCH_CONFIG)

    def pair(self, *, candidate_output=1):
        base, _, _ = self.factory.pair("vstar", 0)
        base = copy.deepcopy(base)
        root = _record(0, [1.0, 1.0], "root")
        base["output"] = 0
        policy = {
            field: copy.deepcopy(base[field])
            for field in ("question", "options", "answer_type", "input_image")
        }
        base["method_trace"].update(
            final_answer=copy.deepcopy(root), anchor_answer=copy.deepcopy(root),
            history=[_history(0, root, 3)], final_boxes=[],
            query_plan=build_query_plan(policy, ()).to_dict(),
            config_id=self.base_config["config_id"],
            effective_config=copy.deepcopy(self.base_config),
        )
        base["method_trace"]["steps"][0]["answer"] = copy.deepcopy(root)

        search = copy.deepcopy(base)
        search["method_trace"]["config_id"] = self.search_config["config_id"]
        search["method_trace"]["effective_config"] = copy.deepcopy(self.search_config)
        if candidate_output == 1:
            candidate = _record(1, [1.0, 0.0], "search")
        else:
            candidate = _record(0, [0.0, 1.0], "search")
        search["output"] = candidate_output
        search["method_trace"].update(
            final_answer=copy.deepcopy(candidate),
            anchor_answer=copy.deepcopy(candidate),
            history=[_history(0, root, 3), _history(1, candidate, 9)],
            final_boxes=[[10, 20, 80, 90]],
        )
        search["method_trace"]["steps"][0]["answer"] = copy.deepcopy(candidate)

        left = self.factory.manifest(
            config=None, benchmark="vstar", ordinal=0,
        )
        right = self.factory.manifest(
            config=None, benchmark="vstar", ordinal=0,
        )
        left["config"] = _config_identity(self.base_config, BASE_SEARCH_CONFIG)
        right["config"] = _config_identity(
            self.search_config, DEFERRED_SEARCH_CONFIG,
        )
        base["_eg_run_fingerprint"] = canonical_sha256(left)
        search["_eg_run_fingerprint"] = canonical_sha256(right)
        return base, search, left, right


class Phase7DeferredSearchPairTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.factory = SearchPairFactory(self.directory.name)

    def extract(self, base, search, left, right):
        return validate_and_extract_search_pairs(
            "vstar", [base], [search],
            base_launch_manifest=left, search_launch_manifest=right,
        )

    def test_exact_pair_extracts_label_blind_dtos_and_freezes_search(self):
        base, search, left, right = self.factory.pair()
        before = json.dumps(
            [base, search, left, right], sort_keys=True, separators=(",", ":"),
        )

        pairs = self.extract(base, search, left, right)
        decisions = freeze_search_decisions(
            "vstar", [base], [search],
            base_launch_manifest=left, search_launch_manifest=right,
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].p0, {
            "action": "P0", "output": 0,
            "p0_stability": {"confidence": 0.0},
        })
        self.assertEqual(set(pairs[0].candidate), {
            "action", "feasible", "output", "stability", "view_sha256",
        })
        self.assertEqual(pairs[0].candidate["action"], "SEARCH")
        self.assertEqual(pairs[0].candidate["output"], 1)
        self.assertNotIn("ordinal", pairs[0].candidate)
        self.assertNotIn("benchmark", pairs[0].candidate)
        self.assertEqual(decisions[0].decision.action, "SEARCH")
        self.assertEqual(decisions[0].decision.output, 1)
        self.assertEqual(
            before,
            json.dumps(
                [base, search, left, right],
                sort_keys=True, separators=(",", ":"),
            ),
        )

    def test_same_answer_is_observed_but_retains_p0(self):
        base, search, left, right = self.factory.pair(candidate_output=0)

        pairs = self.extract(base, search, left, right)
        decisions = freeze_search_decisions(
            "vstar", [base], [search],
            base_launch_manifest=left, search_launch_manifest=right,
        )

        self.assertTrue(pairs[0].candidate["feasible"])
        self.assertEqual(decisions[0].decision.action, "P0")

    def test_config_contract_allows_only_id_and_quick_gate(self):
        base, search, left, right = self.factory.pair()
        changed = {
            key for key in self.factory.search_config
            if self.factory.search_config[key] != self.factory.base_config[key]
        }
        self.assertEqual(changed, {"config_id", "quick_gate"})
        self.assertEqual(self.factory.base_config["quick_gate"], 0.6)
        self.assertEqual(self.factory.search_config["quick_gate"], 0.8)

        forged = copy.deepcopy(right)
        forged["config"]["loaded"]["beta"] = 0.7
        forged["config"]["loaded_sha256"] = canonical_sha256(
            forged["config"]["loaded"],
        )
        search["_eg_run_fingerprint"] = canonical_sha256(forged)
        with self.assertRaisesRegex(ValueError, "reviewed siblings"):
            self.extract(base, search, left, forged)

    def test_pair_rejects_input_model_partition_and_root_observation_drift(self):
        mutations = []
        base, search, left, right = self.factory.pair()
        changed_input = copy.deepcopy(search)
        changed_input["question"] = "different"
        mutations.append((base, changed_input, left, right))

        base, search, left, right = self.factory.pair()
        changed_model = copy.deepcopy(right)
        changed_model["artifacts"]["qwen"]["path"] = "another/qwen"
        search["_eg_run_fingerprint"] = canonical_sha256(changed_model)
        mutations.append((base, search, left, changed_model))

        base, search, left, right = self.factory.pair()
        changed_partition = copy.deepcopy(right)
        changed_partition["selected_partition"]["split_seed"] += 1
        search["_eg_run_fingerprint"] = canonical_sha256(changed_partition)
        mutations.append((base, search, left, changed_partition))

        base, search, left, right = self.factory.pair()
        changed_root = copy.deepcopy(search)
        changed_root["method_trace"]["history"][0]["cost"] += 1
        mutations.append((base, changed_root, left, right))

        for material in mutations:
            with self.subTest(kind=mutations.index(material)):
                with self.assertRaises((TypeError, ValueError)):
                    self.extract(*material)

    def test_existing_base_search_cannot_be_reobserved_as_a_new_candidate(self):
        base, search, left, right = self.factory.pair()
        base = copy.deepcopy(search)
        base["method_trace"]["config_id"] = self.factory.base_config["config_id"]
        base["method_trace"]["effective_config"] = copy.deepcopy(
            self.factory.base_config,
        )
        base["_eg_run_fingerprint"] = canonical_sha256(left)

        pairs = self.extract(base, search, left, right)
        self.assertFalse(pairs[0].candidate["feasible"])
        self.assertIsNone(pairs[0].candidate["output"])

    def test_canonical_post_search_runtime_plan_is_allowed_but_forgery_rejects(self):
        base, search, left, right = self.factory.pair()
        policy = {
            field: copy.deepcopy(search[field])
            for field in ("question", "options", "answer_type", "input_image")
        }
        plan = build_query_plan(policy, ["object"]).to_dict()
        plan["evidence_items"].append({
            "kind": "runtime_ranking_context",
            "query_source": "main_query_plus_current_visual_cue",
            "planned_augmented_queries_used": False,
        })
        search["method_trace"]["query_plan"] = plan

        pairs = self.extract(base, search, left, right)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0].candidate["feasible"])

        forged = copy.deepcopy(search)
        forged["method_trace"]["query_plan"]["targets"] = ["label-derived"]
        with self.assertRaisesRegex(ValueError, "query plan"):
            self.extract(base, forged, left, right)


if __name__ == "__main__":
    unittest.main()
