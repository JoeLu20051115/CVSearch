import copy
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cvsearch.eval.eval_split_ranking_ablation import (
    decompose_frozen_failures,
    evaluate_fixed_pool,
    exact_random_metrics,
    fixed_pool_scores,
    main,
)


def _box(root, child):
    root_x = 20 * (root % 2)
    root_y = 20 * (root // 2)
    child_x = root_x + 10 * (child % 2)
    child_y = root_y + 10 * (child // 2)
    return [child_x, child_y, child_x + 10, child_y + 10]


def _ranking_score(rank):
    clip = (3 - rank) / 3
    return 0.7 * clip + 0.3 * 0.5


def observed_row(*, ordinal=2):
    ranked_paths = (3, 2, 1, 0)
    roots = [
        {
            "path": [root],
            "box": [20 * (root % 2), 20 * (root // 2),
                    20 * (root % 2) + 20, 20 * (root // 2) + 20],
            "score": _ranking_score(rank),
        }
        for rank, root in enumerate(ranked_paths)
    ]
    probes = []
    for root in ranked_paths:
        for rank, child in enumerate(ranked_paths):
            probes.append({
                "patch_path": [root, child],
                "target_box_xyxy": _box(root, child),
                "ranking_score": _ranking_score(rank),
                "raw_support": 0.5,
            })
    branch_paths = ((3, 3), (2, 3), (3, 2), (3, 1), (1, 3), (0, 3))
    branches = []
    for index, path in enumerate(branch_paths):
        view = {
            "role": "tight",
            "patch_path": list(path),
            "crop_xyxy": _box(*path),
            "answer": 1,
        }
        branches.append({
            "visit_index": index,
            "observed_path": list(path),
            "tight_view": view,
            "context_view": {**view, "role": "context"},
        })
    return {
        "_eg_ordinal": ordinal,
        "input_image": "image.jpg",
        "question": "What color is the target?",
        "bbox": [[35, 35, 2, 2]],
        "test_type": "direct_attributes",
        "output": 0,
        "method_trace": {"steps": [{
            "action": "SPLIT",
            "split_search_audit": {
                "render_policy": (
                    "native_2x2_overlap_support_screen_three_scale_"
                    "all_roots_depth2_v3"
                ),
                "root_ranked_siblings": roots,
                "screening_probes": probes,
                "branches": branches,
            },
        }]},
    }


def _treebench_row(row):
    result = copy.deepcopy(row)
    result.pop("bbox", None)
    result.pop("test_type", None)
    result.update({
        "category": "Perception/OCR",
        "target_instances": "[[35,35,37,37]]",
    })
    return result


def _ranking_inputs(qwen, internvl):
    return {
        "qwen": {
            "vstar": [qwen],
            "treebench": [_treebench_row(qwen)],
        },
        "internvl": {
            "vstar": [internvl],
            "treebench": [_treebench_row(internvl)],
        },
    }


def _vstar_failure_row(ordinal, bbox, winners):
    row = observed_row(ordinal=ordinal)
    row.update({
        "bbox": [list(bbox)],
        "answer_type": "logits_match",
        "output": 1,
    })
    audit = row["method_trace"]["steps"][0]["split_search_audit"]
    for branch, winner in zip(audit["branches"], winners):
        answer = {"winner": winner, "losses": [float(winner), float(1 - winner)]}
        branch["tight_view"]["answer"] = copy.deepcopy(answer)
        branch["context_view"]["answer"] = copy.deepcopy(answer)
    return row


def _decision(
    ordinal, before, after, *, selected_source="P0", selected_branch=None,
):
    return {
        "ordinal": ordinal,
        "stage2_output": 0 if before else 1,
        "stage2_correct": [before],
        "stage3b_output": 0 if after else 1,
        "stage3b_correct": [after],
        "selected_source": selected_source,
        "selected_branch": (
            0 if selected_source == "SPLIT" and selected_branch is None
            else selected_branch
        ),
        "reason": "test",
    }


def _filler_validation_row(benchmark, ordinal):
    row = observed_row(ordinal=ordinal)
    audit = row["method_trace"]["steps"][0]["split_search_audit"]
    if benchmark == "vstar":
        row.update({"answer_type": "logits_match", "output": 0})
        answer = {"winner": 0, "losses": [0.0, 1.0]}
        output = 0
    elif benchmark == "treebench":
        row.update({"answer_type": "option_single", "answer": "A", "output": "A"})
        answer = "A"
        output = "A"
    else:
        row.update({
            "answer_type": "option_list",
            "options": ["A. yes\nB. no"],
            "answer": ["A"],
            "output": ["A"],
        })
        answer = ["A"]
        output = ["A"]
    for branch in audit["branches"]:
        branch["tight_view"]["answer"] = copy.deepcopy(answer)
        branch["context_view"]["answer"] = copy.deepcopy(answer)
    decision = {
        "ordinal": ordinal,
        "stage2_output": copy.deepcopy(output),
        "stage2_correct": [True],
        "stage3b_output": copy.deepcopy(output),
        "stage3b_correct": [True],
        "selected_source": "P0",
        "selected_branch": None,
        "reason": "test_filler",
    }
    return decision, row


def _complete_validation(primary_rows, primary_observations):
    cells = {}
    observations = {}
    ordinal = 100
    for backbone in ("internvl", "qwen"):
        for benchmark in ("hr_bench_4k", "hr_bench_8k", "treebench", "vstar"):
            cell = f"{backbone}/{benchmark}"
            if cell == "qwen/vstar":
                decisions = primary_rows
                raw_rows = primary_observations
            else:
                decision, raw = _filler_validation_row(benchmark, ordinal)
                ordinal += 1
                decisions = [decision]
                raw_rows = [raw]
            cells[cell] = {"rows": decisions}
            observations[cell] = raw_rows
    decisions = [row for cell in cells.values() for row in cell["rows"]]
    aggregate = {
        "topics": len(decisions),
        "official_units": sum(len(row["stage2_correct"]) for row in decisions),
        "stage2_correct": sum(sum(row["stage2_correct"]) for row in decisions),
        "stage3b_correct": sum(sum(row["stage3b_correct"]) for row in decisions),
        "corrections": sum(
            not old and new for row in decisions
            for old, new in zip(row["stage2_correct"], row["stage3b_correct"])
        ),
        "corruptions": sum(
            old and not new for row in decisions
            for old, new in zip(row["stage2_correct"], row["stage3b_correct"])
        ),
    }
    return {"aggregate": aggregate, "cells": cells}, observations


class FixedPoolReconstructionTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (40, 40), "gray")

    def test_recovers_clip_percentiles_from_frozen_combined_scores(self):
        scores = fixed_pool_scores(observed_row(), self.image)
        self.assertEqual(len(scores), 16)
        self.assertAlmostEqual(scores[(3, 3)]["clip_only"], 1.0)
        self.assertAlmostEqual(scores[(0, 0)]["clip_only"], 0.0)
        self.assertAlmostEqual(scores[(3, 3)]["visual_only"], 0.5)
        self.assertAlmostEqual(scores[(3, 3)]["combined"], 0.85)

    def test_rejects_any_candidate_pool_other_than_exact_sixteen(self):
        row = observed_row()
        audit = row["method_trace"]["steps"][0]["split_search_audit"]
        audit["screening_probes"].pop()
        with self.assertRaisesRegex(ValueError, "sixteen"):
            fixed_pool_scores(row, self.image)

    def test_rejects_duplicate_candidate_identity(self):
        row = observed_row()
        probes = row["method_trace"]["steps"][0]["split_search_audit"][
            "screening_probes"
        ]
        probes[-1] = copy.deepcopy(probes[0])
        with self.assertRaisesRegex(ValueError, "unique"):
            fixed_pool_scores(row, self.image)


class FixedPoolMetricTest(unittest.TestCase):
    def test_exact_random_baseline_uses_closed_form_not_sampling(self):
        metrics = exact_random_metrics([True] + [False] * 15, (1, 3, 6))
        self.assertEqual(metrics["recall_at"], {
            "1": 1 / 16,
            "3": 3 / 16,
            "6": 6 / 16,
        })
        self.assertAlmostEqual(
            metrics["mrr"], sum(1 / rank for rank in range(1, 17)) / 16,
        )
        self.assertEqual(metrics["pool_recall"], 1.0)

    def test_scores_each_backbone_over_the_same_candidate_set(self):
        row = observed_row()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (40, 40), "gray").save(root / "image.jpg")
            report = evaluate_fixed_pool(
                _ranking_inputs(row, copy.deepcopy(row)),
                {"vstar": root, "treebench": root},
            )
        self.assertEqual(report["source_topics"], 2)
        self.assertEqual(report["topic_backbone_evaluations"], 4)
        self.assertEqual(report["candidate_pool_size"], 16)
        self.assertEqual(report["backbone_copies_verified"], 2)
        self.assertEqual(report["policies"]["combined"]["recall_at"]["1"], 1.0)
        self.assertEqual(report["policies"]["grid"]["recall_at"]["6"], 0.0)
        self.assertEqual(
            report["policies"]["random_expected"]["recall_at"]["3"],
            3 / 16,
        )

    def test_allows_backbone_specific_scores(self):
        first = observed_row()
        second = copy.deepcopy(first)
        second["method_trace"]["steps"][0]["split_search_audit"][
            "screening_probes"
        ][0]["ranking_score"] -= 0.01
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (40, 40), "gray").save(root / "image.jpg")
            report = evaluate_fixed_pool(
                _ranking_inputs(first, second),
                {"vstar": root, "treebench": root},
            )
        self.assertEqual(report["topic_backbone_evaluations"], 4)

    def test_rejects_cross_backbone_candidate_geometry_drift(self):
        first = observed_row()
        second = copy.deepcopy(first)
        second["method_trace"]["steps"][0]["split_search_audit"][
            "screening_probes"
        ][0]["target_box_xyxy"][0] += 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (40, 40), "gray").save(root / "image.jpg")
            with self.assertRaisesRegex(ValueError, "candidate geometry"):
                evaluate_fixed_pool(
                    _ranking_inputs(first, second),
                    {"vstar": root, "treebench": root},
                )

    def test_rejects_incomplete_or_extra_ranking_scope(self):
        row = observed_row()
        with self.assertRaisesRegex(ValueError, "exactly qwen and internvl"):
            evaluate_fixed_pool(
                {"qwen": {"vstar": [row], "treebench": [_treebench_row(row)]}},
                {"vstar": Path("."), "treebench": Path(".")},
            )
        inputs = _ranking_inputs(row, copy.deepcopy(row))
        inputs["other"] = copy.deepcopy(inputs["qwen"])
        with self.assertRaisesRegex(ValueError, "exactly qwen and internvl"):
            evaluate_fixed_pool(
                inputs, {"vstar": Path("."), "treebench": Path(".")},
            )
        inputs = _ranking_inputs(row, copy.deepcopy(row))
        del inputs["qwen"]["treebench"]
        with self.assertRaisesRegex(ValueError, r"exactly V\* and TreeBench"):
            evaluate_fixed_pool(
                inputs, {"vstar": Path("."), "treebench": Path(".")},
            )
        inputs = _ranking_inputs(row, copy.deepcopy(row))
        inputs["qwen"]["other"] = [copy.deepcopy(row)]
        with self.assertRaisesRegex(ValueError, r"exactly V\* and TreeBench"):
            evaluate_fixed_pool(
                inputs, {"vstar": Path("."), "treebench": Path(".")},
            )


class FrozenFailureDecompositionTest(unittest.TestCase):
    def test_partitions_errors_and_refines_geometry_failures(self):
        decisions = [
            _decision(1, False, True, selected_source="SPLIT"),
            _decision(2, False, False, selected_source="ZOOM"),
            _decision(
                3, False, False, selected_source="SPLIT", selected_branch=1,
            ),
            _decision(4, False, False),
            _decision(5, False, False),
            _decision(6, False, False),
            _decision(7, True, False, selected_source="SPLIT"),
        ]
        wrong = [1] * 6
        correct_available = [0, 1, 1, 1, 1, 1]
        observations = [
            _vstar_failure_row(1, [35, 35, 2, 2], correct_available),
            _vstar_failure_row(2, [35, 35, 2, 2], correct_available),
            _vstar_failure_row(3, [35, 35, 2, 2], correct_available),
            _vstar_failure_row(4, [100, 100, 2, 2], wrong),
            _vstar_failure_row(5, [5, 5, 2, 2], wrong),
            _vstar_failure_row(6, [35, 35, 2, 2], wrong),
            _vstar_failure_row(7, [35, 35, 2, 2], wrong),
        ]
        score, raw = _complete_validation(decisions, observations)
        report = decompose_frozen_failures(score, raw)
        self.assertEqual(report["by_backbone"]["qwen"]["stage2_errors"], 6)
        self.assertEqual(report["aggregate"]["converted"], 1)
        self.assertEqual(report["aggregate"]["selector_abstained"], 1)
        self.assertEqual(report["aggregate"]["selector_wrong_choice"], 1)
        self.assertEqual(report["aggregate"]["no_correct_observed_answer"], 3)
        self.assertEqual(report["aggregate"]["geometry_pool_miss"], 1)
        self.assertEqual(report["aggregate"]["six_branch_budget_miss"], 1)
        self.assertEqual(report["aggregate"]["vlm_answer_miss"], 1)
        self.assertEqual(report["aggregate"]["corruption"], 1)
        self.assertTrue(all(report["gates"].values()))

    def test_rejects_frozen_report_accounting_drift(self):
        decision = _decision(1, False, True, selected_source="SPLIT")
        raw = _vstar_failure_row(1, [35, 35, 2, 2], [0] * 6)
        score, observations = _complete_validation([decision], [raw])
        score["aggregate"]["corrections"] = 0
        with self.assertRaisesRegex(ValueError, "corrections"):
            decompose_frozen_failures(score, observations)

    def test_rejects_incomplete_or_extra_validation_scope(self):
        decision = _decision(1, False, True, selected_source="SPLIT")
        raw = _vstar_failure_row(1, [35, 35, 2, 2], [0] * 6)
        score, observations = _complete_validation([decision], [raw])
        del score["cells"]["internvl/treebench"]
        del observations["internvl/treebench"]
        with self.assertRaisesRegex(ValueError, "exact eight-cell"):
            decompose_frozen_failures(score, observations)
        score, observations = _complete_validation([decision], [raw])
        score["cells"]["other/vstar"] = copy.deepcopy(score["cells"]["qwen/vstar"])
        observations["other/vstar"] = copy.deepcopy(observations["qwen/vstar"])
        with self.assertRaisesRegex(ValueError, "exact eight-cell"):
            decompose_frozen_failures(score, observations)

    def test_rejects_split_output_absent_from_selected_raw_branch(self):
        decision = _decision(1, False, True, selected_source="SPLIT")
        raw = _vstar_failure_row(1, [35, 35, 2, 2], [1] * 6)
        score, observations = _complete_validation([decision], [raw])
        with self.assertRaisesRegex(ValueError, "selected SPLIT output"):
            decompose_frozen_failures(score, observations)


class RankingAblationCliTest(unittest.TestCase):
    def test_writes_deterministic_hash_bound_label_safe_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development = root / "development"
            validation = root / "validation"
            images = {name: root / name for name in ("vstar", "treebench")}
            for image_root in images.values():
                image_root.mkdir(parents=True)
                Image.new("RGB", (40, 40), "gray").save(image_root / "image.jpg")
            for backbone in ("qwen", "internvl"):
                (development / backbone).mkdir(parents=True)
                vstar = observed_row()
                treebench = observed_row()
                treebench.pop("bbox")
                treebench.pop("test_type")
                treebench.update({
                    "category": "Perception/OCR",
                    "target_instances": "[[35,35,37,37]]",
                })
                for dataset, row in (("vstar", vstar), ("treebench", treebench)):
                    (development / backbone / f"{dataset}.jsonl").write_text(
                        json.dumps(row, separators=(",", ":")) + "\n",
                        encoding="utf-8",
                    )
            raw = _vstar_failure_row(1, [35, 35, 2, 2], [0] * 6)
            decision = _decision(1, False, True, selected_source="SPLIT")
            score, validation_rows = _complete_validation([decision], [raw])
            for cell, rows in validation_rows.items():
                backbone, benchmark = cell.split("/", 1)
                (validation / backbone).mkdir(parents=True, exist_ok=True)
                (validation / backbone / f"{benchmark}.jsonl").write_text(
                    "".join(
                        json.dumps(row, separators=(",", ":")) + "\n"
                        for row in rows
                    ),
                    encoding="utf-8",
                )
            score_path = root / "validation.json"
            score_path.write_text(
                json.dumps(score, separators=(",", ":")),
                encoding="utf-8",
            )
            output = root / "report.json"
            arguments = [
                "--development-root", str(development),
                "--vstar-image-root", str(images["vstar"]),
                "--treebench-image-root", str(images["treebench"]),
                "--validation-report", str(score_path),
                "--validation-split-root", str(validation),
                "--output", str(output),
            ]
            self.assertEqual(main(arguments), 0)
            first = output.read_bytes()
            self.assertEqual(main(arguments), 0)
            self.assertEqual(output.read_bytes(), first)
            report = json.loads(first)
            incomplete = copy.deepcopy(score)
            del incomplete["cells"]["internvl/treebench"]
            score_path.write_text(json.dumps(incomplete), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact eight-cell"):
                main(arguments)
            score_path.write_text(json.dumps(score), encoding="utf-8")
            missing_dataset = development / "qwen" / "treebench.jsonl"
            saved_dataset = missing_dataset.read_text(encoding="utf-8")
            missing_dataset.unlink()
            with self.assertRaisesRegex(ValueError, "exactly qwen and internvl"):
                main(arguments)
            missing_dataset.write_text(saved_dataset, encoding="utf-8")
            extra = development / "other"
            extra.mkdir()
            for dataset in ("vstar", "treebench"):
                (extra / f"{dataset}.jsonl").write_text(
                    (development / "qwen" / f"{dataset}.jsonl").read_text(
                        encoding="utf-8",
                    ),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(ValueError, "exactly qwen and internvl"):
                main(arguments)
        self.assertTrue(report["success"])
        self.assertEqual(report["artifact_kind"], "fixed-pool-split-ranking-ablation")
        self.assertFalse(report["data_scope"]["validation_used_for_ranking_or_tuning"])
        self.assertTrue(all(report["gates"].values()))
        self.assertEqual(len(report["bindings"]["validation_report_sha256"]), 64)
        forbidden = {"answer", "bbox", "target_instances", "raw_outputs"}

        def keys(value):
            if isinstance(value, dict):
                return set(value).union(*(keys(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value))
            return set()

        self.assertTrue(forbidden.isdisjoint(keys(report)))


if __name__ == "__main__":
    unittest.main()
