import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cvsearch.eval.replay_unified_fusion import replay_paths


BLOCKS = [
    "A. cat\nB. dog\nC. bird\nD. fish\n",
    "A. dog\nB. cat\nC. fish\nD. bird\n",
    "A. bird\nB. fish\nC. cat\nD. dog\n",
    "A. fish\nB. bird\nC. dog\nD. cat\n",
]
DEV_PATHS = {
    "hr-bench_4k": Path(
        "/mnt/data3/data_xingrui/lueq/CVSearch/reproduction/evidence_gap/"
        "hr_gate060_dev/7b34a4efc368/hr-bench_4k/dev_gate060.jsonl"
    ),
    "hr-bench_8k": Path(
        "/mnt/data3/data_xingrui/lueq/CVSearch/reproduction/evidence_gap/"
        "hr_gate060_dev/7b34a4efc368/hr-bench_8k/dev_gate060.jsonl"
    ),
}


def answer(output, raw_outputs, selected_from):
    return {
        "output": output,
        "canonical_answer": "cat",
        "raw_outputs": raw_outputs,
        "groups": {"cat": {"count": 4, "first_vote": 0, "vote_indices": [0, 1, 2, 3]}},
        "frequency": 1.0,
        "margin": 1.0,
        "confidence": 1.0,
        "uncertainty": 0.0,
        "losses": [],
        "selected_from": selected_from,
        "aggregation_available": True,
        "aggregation_reason": None,
    }


def row(ordinal, answers, raw_outputs, projected_output):
    root = answer(["A", "B", "C", "D"], raw_outputs, "root")
    search = answer(["A", "B", "C", "D"], raw_outputs, "search")
    return {
        "_eg_ordinal": ordinal,
        "question": f"question {ordinal}",
        "options": BLOCKS,
        "answer": answers,
        # This deliberately differs from the exact CVSearch raw anchor.
        "output": projected_output,
        "method_trace": {
            "effective_config": {"config_id": "old-gate", "quick_gate": 0.6},
            "history": [{"step": 0, "answer": root}, {"step": 1, "answer": search}],
            "final_answer": answer(projected_output, raw_outputs, "search"),
        },
    }


def write_rows(tmp_path, rows, name="valid.jsonl"):
    path = Path(tmp_path) / name
    path.write_text("".join(json.dumps(item, separators=(",", ":")) + "\n" for item in rows), encoding="utf-8")
    return path


def write_two_topic_fixture(tmp_path):
    return write_rows(tmp_path, [
        row(4, ["A", "B", "C", "D"], ["A", "A", "A", "A"], ["A", "B", "C", "D"]),
        row(8, ["A", "B", "C", "D"], ["A", "A", "A", "A"], ["A", "B", "C", "D"]),
    ])


def write_invalid_fixture(tmp_path):
    valid = row(4, ["A", "A", "A", "A"], ["A", "A", "A", "A"], ["A", "B", "C", "D"])
    invalid = row(4, ["A", "A", "A", "A"], ["A", "A", "A", "A"], ["A", "B", "C", "D"])
    invalid.pop("method_trace")
    return write_rows(tmp_path, [valid, invalid], "invalid.jsonl")


class UnifiedFusionReplayTest(unittest.TestCase):
    def test_replay_scores_topics_not_shuffle_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_two_topic_fixture(directory)
            report = replay_paths({"hr-bench_4k": path}, gammas=(0.0, 2.1))
        self.assertEqual(report["hr-bench_4k"]["n_topics"], 2)
        self.assertEqual(report["hr-bench_4k"]["n_cycles"], 8)

    def test_replay_gamma_zero_matches_latest_history_raw_anchor_not_row_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_two_topic_fixture(directory)
            report = replay_paths({"hr-bench_4k": path}, gammas=(0.0,))
        candidate = report["hr-bench_4k"]["candidates"]["0.0"]
        self.assertEqual(candidate["delta"], 0.0)
        self.assertEqual(candidate["accuracy"], candidate["anchor_accuracy"])
        self.assertEqual(candidate["anchor_accuracy"], 0.25)

    def test_replay_is_read_only_and_reports_exploratory_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_two_topic_fixture(directory)
            before = (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
            report = replay_paths({"hr-bench_4k": path}, gammas=(0.0, 2.1))
            after = (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(before, after)
        self.assertEqual(report["calibration_status"], "exploratory_development")
        self.assertEqual(report["selected_gamma"], 2.1)
        self.assertIn("config_hash", report["hr-bench_4k"])

    def test_replay_reports_correct_cycle_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            report = replay_paths({"hr-bench_4k": write_two_topic_fixture(directory)}, gammas=(0.0, 2.1))
        candidates = report["hr-bench_4k"]["candidates"]
        self.assertEqual(candidates["0.0"]["correct_cycles"], 2)
        self.assertEqual(candidates["2.1"]["correct_cycles"], 8)

    @unittest.skipUnless(all(path.is_file() for path in DEV_PATHS.values()), "exposed development artifacts unavailable")
    def test_exposed_development_candidate_table_uses_history_raw_anchors(self):
        report = replay_paths(DEV_PATHS)
        expected = {
            "0.0": (78, 90, 0.0),
            "1.1": (79, 90, 0.0),
            "1.5": (79, 90, 0.0),
            "2.1": (81, 93, 0.025),
            "4.1": (80, 92, 1 / 60),
        }
        for gamma, (correct_4k, correct_8k, joint) in expected.items():
            self.assertEqual(report["hr-bench_4k"]["candidates"][gamma]["correct_cycles"], correct_4k)
            self.assertEqual(report["hr-bench_8k"]["candidates"][gamma]["correct_cycles"], correct_8k)
            self.assertAlmostEqual(report["candidates"][gamma]["min_delta"], joint)
        self.assertEqual(report["selected_gamma"], 2.1)

    def test_replay_rejects_duplicate_ordinals_and_missing_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                replay_paths({"hr-bench_4k": write_invalid_fixture(directory)}, gammas=(2.1,))


if __name__ == "__main__":
    unittest.main()
