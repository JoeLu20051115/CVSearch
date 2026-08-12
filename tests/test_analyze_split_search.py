import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cvsearch.eval.analyze_split_search import (
    candidate_outputs,
    load_selected_calibrations,
    official_correctness,
    score_cell,
    select_policy,
)
from cvsearch.eval.replay_adaptive_search import freeze_selected_calibration


class AnalyzeSplitSearchTest(unittest.TestCase):
    def test_loads_each_frozen_backbone_calibration_from_suite_manifest(self):
        frozen = freeze_selected_calibration([
            {"row_id": "a", "source_group": "g1", "raw_support": 0.1,
             "support_sufficient": 0},
            {"row_id": "b", "source_group": "g1", "raw_support": 0.9,
             "support_sufficient": 1},
            {"row_id": "c", "source_group": "g2", "raw_support": 0.2,
             "support_sufficient": 0},
            {"row_id": "d", "source_group": "g2", "raw_support": 0.8,
             "support_sufficient": 1},
        ])
        record = frozen.to_dict()
        record["raw_tiebreak_weight"] = frozen.raw_tiebreak_weight
        record["prediction_rule"] = frozen.prediction_rule
        payload = {"backbones": {"qwen2_5_vl_7b": record}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_selected_calibrations(path)
        self.assertEqual(loaded["qwen2_5_vl_7b"], frozen)

    def test_extracts_all_four_aggregated_view_outputs(self):
        row = {
            "answer_type": "option_single",
            "options": "A. red\nB. blue",
            "method_trace": {"steps": [{
                "action": "SPLIT",
                "split_search_audit": {"branches": [
                    {"tight_view": {"answer": "A"},
                     "context_view": {"answer": "B"}},
                    {"tight_view": {"answer": "B"},
                     "context_view": {"answer": "A"}},
                ]},
            }]},
        }
        self.assertEqual(candidate_outputs(row), ("A", "B", "B", "A"))

    def test_official_correctness_uses_each_benchmark_unit(self):
        self.assertEqual(official_correctness("vstar", {}, 0), (True,))
        self.assertEqual(official_correctness("vstar", {}, 1), (False,))
        hr = {"answer": ["A", "B", "C", "D"]}
        self.assertEqual(
            official_correctness("hr_bench_4k", hr, ["A", "xB", "D", "D"]),
            (True, True, False, True),
        )
        self.assertEqual(
            official_correctness("treebench", {"answer": "C"}, "C"), (True,),
        )

    def test_policy_selection_is_global_and_prefers_no_harm_then_gain(self):
        reports = {
            (0.6, 0.0): {
                "aggregate_delta": 2, "corruptions": 1,
                "cell_deltas": {"q/v": 2, "i/v": -1}, "split_selections": 4,
            },
            (0.7, 0.1): {
                "aggregate_delta": 1, "corruptions": 0,
                "cell_deltas": {"q/v": 1, "i/v": 0}, "split_selections": 1,
            },
            (0.8, 0.2): {
                "aggregate_delta": 0, "corruptions": 0,
                "cell_deltas": {"q/v": 0, "i/v": 0}, "split_selections": 0,
            },
        }
        selected = select_policy(reports)
        self.assertEqual(selected, (0.7, 0.1))

    def test_scores_stage2_stage3_and_candidate_oracle_in_official_units(self):
        calibration = freeze_selected_calibration([
            {"row_id": "a", "source_group": "g1", "raw_support": 0.1,
             "support_sufficient": 0},
            {"row_id": "b", "source_group": "g1", "raw_support": 0.9,
             "support_sufficient": 1},
            {"row_id": "c", "source_group": "g2", "raw_support": 0.2,
             "support_sufficient": 0},
            {"row_id": "d", "source_group": "g2", "raw_support": 0.8,
             "support_sufficient": 1},
        ])
        stage2 = {
            "_eg_ordinal": 3,
            "answer_type": "option_single",
            "options": "A. red\nB. blue",
            "answer": "B",
            "output": "A",
            "method_trace": {"candidate_ranks": [{"identity": "p0"}]},
        }
        split = copy.deepcopy(stage2)
        from tests.test_replay_split_search import rows
        _, synthetic = rows()
        split["method_trace"] = synthetic["method_trace"]
        split["method_trace"]["candidate_ranks"] = copy.deepcopy(
            stage2["method_trace"]["candidate_ranks"]
        )
        split["method_trace"]["steps"][0]["split_search_audit"][
            "rank_sha256"
        ] = hashlib.sha256(json.dumps(
            stage2["method_trace"]["candidate_ranks"], sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode()).hexdigest()
        report = score_cell(
            "treebench", [stage2], [split], calibration,
            {
                "minimum_final_support": 0.6,
                "minimum_support_gain": 0.1,
                "maximum_support_drop": 0.0,
                "minimum_conflict_margin": 1.0,
                "minimum_uncontested_support": 1.0,
            },
        )
        self.assertEqual(report["official_units"], 1)
        self.assertEqual(report["stage2_correct"], 0)
        self.assertEqual(report["stage3_correct"], 1)
        self.assertEqual(report["aggregate_delta"], 1)
        self.assertEqual(report["corrections"], 1)
        self.assertEqual(report["corruptions"], 0)
        self.assertEqual(report["oracle_fixes"], 1)
        self.assertEqual(report["split_selections"], 1)


if __name__ == "__main__":
    unittest.main()
