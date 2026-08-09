import json
import os
import unittest
from pathlib import Path

from cvsearch.evidence_gap.baselines import baseline_envelope, reconstruct_quick_gate, score_rows


ROOT = Path(__file__).resolve().parents[1]
RETAINED_ARTIFACTS = tuple(
    ROOT / "reproduction" / "answers" / "qwen2.5-vl-7b" / benchmark / filename
    for benchmark in ("vstar", "hr-bench_4k", "hr-bench_8k")
    for filename in ("direct_answer.jsonl", "cvsearch.jsonl")
)
MISSING_RETAINED_ARTIFACTS = tuple(path for path in RETAINED_ARTIFACTS if not path.is_file())

if os.environ.get("CVSEARCH_REQUIRE_RETAINED_ARTIFACTS") == "1" and MISSING_RETAINED_ARTIFACTS:
    missing = ", ".join(str(path.relative_to(ROOT)) for path in MISSING_RETAINED_ARTIFACTS)
    raise FileNotFoundError(f"required retained artifacts are unavailable: {missing}")


class QuickGateReconstructionTest(unittest.TestCase):
    def test_strict_threshold_selects_direct_rows_and_copies_records(self):
        direct = [{"output": ["A"], "nested": {"source": "direct"}}, {"output": ["B"]}]
        search = [
            {"root_ans_conf": 0.6001, "output": ["search-A"]},
            {"root_ans_conf": 0.6, "output": ["search-B"]},
        ]

        reconstructed = reconstruct_quick_gate(direct, search, 0.6)

        self.assertEqual(reconstructed, [direct[0], search[1]])
        self.assertIsNot(reconstructed[0], direct[0])
        self.assertIsNot(reconstructed[1], search[1])
        reconstructed[0]["nested"]["source"] = "changed"
        self.assertEqual(direct[0]["nested"]["source"], "direct")

    def test_mismatched_pairs_are_rejected(self):
        with self.assertRaises(ValueError):
            reconstruct_quick_gate([{"output": 0}], [], 0.6)


class OfficialScoringTest(unittest.TestCase):
    def test_vstar_scores_zero_output_as_correct(self):
        self.assertEqual(score_rows("vstar", [{"output": 0}, {"output": 1}, {"output": 0}]), 100 * 2 / 3)

    def test_hr_scoring_matches_official_letter_parser(self):
        rows = [{"answer": ["A", "B"], "output": [["A"], ["ignored", "B"]]}]
        self.assertEqual(score_rows("hr-bench_4k", rows), 100.0)

    def test_hr_rejects_multi_character_list_token(self):
        rows = [{"answer": ["AB"], "output": [["noise", "AB"]]}]
        self.assertEqual(score_rows("hr-bench_4k", rows), 0.0)


class SyntheticBaselineEnvelopeTest(unittest.TestCase):
    def test_default_envelope_scores_both_gates_without_retained_artifacts(self):
        direct = {
            "vstar": [{"output": 0}],
            "hr-bench_4k": [{"answer": ["A"], "output": [["A"]]}],
        }
        search = {
            "vstar": [{"root_ans_conf": 0.7, "output": 1}],
            "hr-bench_4k": [{"root_ans_conf": 0.7, "answer": ["A"], "output": [["B"]]}],
        }

        self.assertEqual(
            baseline_envelope(direct, search),
            {"vstar": 100.0, "hr-bench_4k": 100.0},
        )


@unittest.skipIf(
    MISSING_RETAINED_ARTIFACTS,
    "retained artifacts unavailable: " + ", ".join(str(path.relative_to(ROOT)) for path in MISSING_RETAINED_ARTIFACTS),
)
class RetainedBaselineEnvelopeTest(unittest.TestCase):
    @staticmethod
    def _rows(benchmark, filename):
        path = ROOT / "reproduction" / "answers" / "qwen2.5-vl-7b" / benchmark / filename
        with path.open() as handle:
            return [json.loads(line) for line in handle]

    def test_retained_gate_scores_and_envelope(self):
        direct = {
            benchmark: self._rows(benchmark, "direct_answer.jsonl")
            for benchmark in ("vstar", "hr-bench_4k", "hr-bench_8k")
        }
        search = {
            benchmark: self._rows(benchmark, "cvsearch.jsonl")
            for benchmark in direct
        }

        envelope = baseline_envelope(direct, search)

        self.assertEqual(
            {benchmark: round(score_rows(benchmark, reconstruct_quick_gate(direct[benchmark], search[benchmark], 0.6)), 6)
             for benchmark in direct},
            {"vstar": 86.910995, "hr-bench_4k": 76.625, "hr-bench_8k": 76.75},
        )
        self.assertEqual(envelope, {"vstar": 87.43455497382199, "hr-bench_4k": 76.625, "hr-bench_8k": 76.75})


if __name__ == "__main__":
    unittest.main()
