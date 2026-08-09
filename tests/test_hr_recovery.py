import copy
import hashlib
import importlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from cvsearch.evidence_gap import hr_recovery


def recovery_fixture(split_name="recovery-a"):
    direct = []
    search = []
    for ordinal in range(200):
        identity = {
            "question": f"question {ordinal}",
            "index": ordinal,
            "category": "single" if ordinal % 2 else "cross",
            "options": ["A. red", "B. blue", "C. green", "D. black"],
            "answer": ["A", "A", "A", "A"],
        }
        direct.append(dict(identity, output=["A", "A", "A", "A"]))
        search.append(dict(
            identity, root_ans_conf=0.6, output=["A", "A", "B", "B"]
        ))

    method = {}
    for resolution, output in (
        ("hr-bench_4k", ["A", "A", "A", "B"]),
        ("hr-bench_8k", ["A", "A", "A", "A"]),
    ):
        method[resolution] = [
            {
                **{key: direct[ordinal][key] for key in (
                    "question", "index", "category", "options", "answer"
                )},
                "output": list(output),
                "method_trace": {
                    "config_id": "frozen-v1",
                    "budget_interrupted": False,
                },
                "_eg_ordinal": ordinal,
                "_eg_code_revision": "revision-1",
            }
            for ordinal in hr_recovery.locked_ordinals(split_name)
        ]
    return method, direct, search


class LockedRecoverySplitTest(unittest.TestCase):
    def test_embedded_splits_match_preregistered_counts_and_hashes(self):
        try:
            recovery = importlib.import_module("cvsearch.evidence_gap.hr_recovery")
        except ModuleNotFoundError:
            self.fail("HR recovery scorer module is missing")

        expected = {
            "recovery-a": (71, "d731fe78614b9165cb22deaf1956f2924d41e8c2be38edd5ed1fa477e9e9ae13"),
            "vault-b": (71, "ae6305b064ac93b861109297f906914870e20fe2a85289be6069855f32069c75"),
        }
        recovery.validate_locked_splits()
        for name, (count, digest) in expected.items():
            ordinals = recovery.locked_ordinals(name)
            self.assertEqual(len(ordinals), count)
            self.assertEqual(len(set(ordinals)), count)
            encoded = ",".join(map(str, ordinals)).encode("ascii")
            self.assertEqual(hashlib.sha256(encoded).hexdigest(), digest)


class RecoveryScoringTest(unittest.TestCase):
    @staticmethod
    def score(method, direct, search, split_name="recovery-a"):
        return hr_recovery.score_recovery(
            split_name,
            method_4k=method["hr-bench_4k"],
            method_8k=method["hr-bench_8k"],
            direct_4k=direct,
            cvsearch_4k=search,
            direct_8k=direct,
            cvsearch_8k=search,
        )

    def test_scores_four_shuffles_per_topic_and_bootstraps_resolutions_together(self):
        self.assertTrue(
            hasattr(hr_recovery, "score_recovery"),
            "score_recovery pure function is missing",
        )
        method, direct, search = recovery_fixture()

        report = hr_recovery.score_recovery(
            "recovery-a",
            method_4k=method["hr-bench_4k"],
            method_8k=method["hr-bench_8k"],
            direct_4k=direct,
            cvsearch_4k=search,
            direct_8k=direct,
            cvsearch_8k=search,
        )

        self.assertEqual(report, {
            "n_topics": 71,
            "hr-bench_4k": {
                "method_accuracy": 0.75,
                "baseline_accuracy": 0.5,
                "delta": 0.25,
                "one_sided_95_lower": 0.25,
            },
            "hr-bench_8k": {
                "method_accuracy": 1.0,
                "baseline_accuracy": 0.5,
                "delta": 0.5,
                "one_sided_95_lower": 0.5,
            },
            "interaction_8k_minus_4k": {
                "delta": 0.25,
                "two_sided_95_ci": [0.25, 0.25],
            },
            "point_gate_pass": True,
        })

    def test_vault_b_is_supported_only_when_explicitly_selected(self):
        method, direct, search = recovery_fixture("vault-b")
        report = self.score(method, direct, search, "vault-b")
        self.assertEqual(report["n_topics"], 71)
        self.assertTrue(report["point_gate_pass"])

    def test_protocol_identity_and_budget_violations_fail_closed(self):
        method, direct, search = recovery_fixture()
        cases = []

        missing = copy.deepcopy(method)
        missing["hr-bench_4k"].pop()
        cases.append(("locked split", missing))

        bad_ordinal = copy.deepcopy(method)
        bad_ordinal["hr-bench_4k"][0]["_eg_ordinal"] += 1
        cases.append(("ordinals", bad_ordinal))

        mixed_revision = copy.deepcopy(method)
        mixed_revision["hr-bench_4k"][0]["_eg_code_revision"] = "revision-2"
        cases.append(("one code revision", mixed_revision))

        cross_revision = copy.deepcopy(method)
        for row in cross_revision["hr-bench_8k"]:
            row["_eg_code_revision"] = "revision-2"
        cases.append(("same code revision", cross_revision))

        mixed_config = copy.deepcopy(method)
        mixed_config["hr-bench_8k"][0]["method_trace"]["config_id"] = "other"
        cases.append(("one config_id", mixed_config))

        interrupted = copy.deepcopy(method)
        interrupted["hr-bench_4k"][0]["method_trace"]["budget_interrupted"] = True
        cases.append(("budget interruption", interrupted))

        malformed_output = copy.deepcopy(method)
        malformed_output["hr-bench_8k"][0]["output"] = ["A"]
        cases.append(("four string outputs", malformed_output))

        unpaired = copy.deepcopy(method)
        unpaired["hr-bench_8k"][0]["question"] = "different topic"
        cases.append(("topic identity mismatch", unpaired))

        for expected_message, invalid_method in cases:
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(ValueError, expected_message):
                    self.score(invalid_method, direct, search)

    def test_retained_baselines_must_be_complete_paired_and_finite(self):
        method, direct, search = recovery_fixture()
        with self.assertRaisesRegex(ValueError, "200 rows"):
            self.score(method, direct[:-1], search)

        bad_pair = copy.deepcopy(search)
        bad_pair[0]["question"] = "wrong pairing"
        with self.assertRaisesRegex(ValueError, "baseline topic mismatch"):
            self.score(method, direct, bad_pair)

        bad_confidence = copy.deepcopy(search)
        bad_confidence[0]["root_ans_conf"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.score(method, direct, bad_confidence)

    def test_output_schema_requires_four_option_blocks(self):
        method, direct, search = recovery_fixture()
        ordinal = hr_recovery.locked_ordinals("recovery-a")[0]
        for row in (
            direct[ordinal], search[ordinal],
            method["hr-bench_4k"][0], method["hr-bench_8k"][0],
        ):
            row["options"] = ["A. only"]

        with self.assertRaisesRegex(ValueError, "four option"):
            hr_recovery.score_recovery(
                "recovery-a",
                method_4k=method["hr-bench_4k"],
                method_8k=method["hr-bench_8k"],
                direct_4k=direct,
                cvsearch_4k=search,
                direct_8k=direct,
                cvsearch_8k=search,
            )


class RecoveryCliTest(unittest.TestCase):
    def test_cli_emits_one_aggregate_json_object_for_explicit_split(self):
        try:
            cli = importlib.import_module("cvsearch.eval.score_hr_recovery")
        except ModuleNotFoundError:
            self.fail("HR recovery scoring CLI is missing")
        method, direct, search = recovery_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name, rows):
                path = root / name
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                return path

            paths = {
                "method_4k": write("method-4k.jsonl", method["hr-bench_4k"]),
                "method_8k": write("method-8k.jsonl", method["hr-bench_8k"]),
                "direct_4k": write("direct-4k.jsonl", direct),
                "cvsearch_4k": write("cvsearch-4k.jsonl", search),
                "direct_8k": write("direct-8k.jsonl", direct),
                "cvsearch_8k": write("cvsearch-8k.jsonl", search),
            }
            args = ["--split", "recovery-a"]
            for name, path in paths.items():
                args.extend((f"--{name.replace('_', '-')}", str(path)))
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(cli.main(args), 0)

        self.assertEqual(output.getvalue().count("\n"), 1)
        report = json.loads(output.getvalue())
        self.assertEqual(set(report), {
            "n_topics", "hr-bench_4k", "hr-bench_8k",
            "interaction_8k_minus_4k", "point_gate_pass",
        })
        self.assertTrue(report["point_gate_pass"])

    def test_strict_jsonl_loader_rejects_ambiguous_or_corrupt_inputs(self):
        invalid_documents = (
            "",
            "\n",
            '{"a":1,"a":2}\n',
            '{"a":NaN}\n',
            "[]\n",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            for document in invalid_documents:
                with self.subTest(document=document):
                    path.write_text(document, encoding="utf-8")
                    with self.assertRaises((TypeError, ValueError)):
                        hr_recovery.load_jsonl(path)


if __name__ == "__main__":
    unittest.main()
