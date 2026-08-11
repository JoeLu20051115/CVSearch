"""Score complete cross-backbone vectors against trusted local annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


IDENTITY_FIELDS = ("input_image", "question", "options", "answer_type")


def parse_hr_choice(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("HR output must be text")
    if len(value) == 1:
        return value
    return next((character for character in value if character in "ABCD"), "")


def _metric(correct: int, total: int) -> dict[str, Any]:
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("score counts are invalid")
    return {
        "correct": correct, "total": total,
        "accuracy": 100.0 * correct / total,
    }


def _validate_alignment(
    annotations: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> None:
    if not annotations or len(annotations) != len(predictions):
        raise ValueError("prediction cardinality differs from trusted annotations")
    identities = []
    for annotation, prediction in zip(annotations, predictions):
        identity = {field: annotation.get(field) for field in IDENTITY_FIELDS}
        if any(prediction.get(field) != value for field, value in identity.items()):
            raise ValueError("prediction identity or order differs from annotations")
        identities.append(json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ))
    if len(identities) != len(set(identities)):
        raise ValueError("trusted annotation identities are not unique")


def score_rows(
    benchmark: str,
    annotations: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    _validate_alignment(annotations, predictions)
    if benchmark == "vstar":
        counts = {
            "attribute": [0, 0], "spatial": [0, 0], "overall": [0, 0],
        }
        category_names = {
            "direct_attributes": "attribute", "relative_position": "spatial",
        }
        for annotation, prediction in zip(annotations, predictions):
            category = category_names.get(annotation.get("test_type"))
            output = prediction.get("output")
            if category is None:
                raise ValueError("V* annotation category is unsupported")
            if (
                type(output) is not int
                or not 0 <= output < len(annotation.get("options", ()))
            ):
                raise ValueError("V* output is not one valid option index")
            correct = int(output == 0)
            counts[category][0] += correct
            counts[category][1] += 1
            counts["overall"][0] += correct
            counts["overall"][1] += 1
    elif benchmark in {"hr-bench_4k", "hr-bench_8k"}:
        counts = {"fsp": [0, 0], "fcp": [0, 0], "overall": [0, 0]}
        category_names = {"single": "fsp", "cross": "fcp"}
        for annotation, prediction in zip(annotations, predictions):
            category = category_names.get(annotation.get("category"))
            truth = annotation.get("answer")
            output = prediction.get("output")
            if (
                category is None or type(truth) is not list or len(truth) != 4
                or type(output) is not list or len(output) != 4
            ):
                raise ValueError("HR row does not contain one four-cycle category")
            for expected, raw in zip(truth, output):
                if expected not in "ABCD":
                    raise ValueError("HR trusted answer is invalid")
                correct = int(parse_hr_choice(raw) == expected)
                counts[category][0] += correct
                counts[category][1] += 1
                counts["overall"][0] += correct
                counts["overall"][1] += 1
    else:
        raise ValueError(f"unsupported benchmark: {benchmark!r}")
    return {key: _metric(*value) for key, value in counts.items()}


def score_vstar_letter_rows(
    annotations: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    _validate_alignment(annotations, predictions)
    counts = {"attribute": [0, 0], "spatial": [0, 0], "overall": [0, 0]}
    category_names = {
        "direct_attributes": "attribute", "relative_position": "spatial",
    }
    for annotation, prediction in zip(annotations, predictions):
        category = category_names.get(annotation.get("test_type"))
        label = annotation.get("label")
        if category is None or label not in "ABCD":
            raise ValueError("updated V* category or label is invalid")
        correct = int(parse_hr_choice(prediction.get("output")) == label)
        counts[category][0] += correct
        counts[category][1] += 1
        counts["overall"][0] += correct
        counts["overall"][1] += 1
    return {key: _metric(*value) for key, value in counts.items()}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not all(type(row) is dict for row in rows):
        raise ValueError("prediction JSONL must contain objects")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


PAPER_REFERENCE = {
    "llava": {
        "vstar": {
            "direct": [75.7, 75.0, 75.4], "cvsearch": [95.7, 85.5, 91.6],
        },
        "hr-bench_4k": {
            "direct": [72.0, 54.0, 63.0], "cvsearch": [89.5, 61.8, 75.6],
        },
        "hr-bench_8k": {
            "direct": [67.3, 52.3, 59.8], "cvsearch": [89.0, 60.5, 74.8],
        },
    },
    "internvl": {
        "vstar": {
            "direct": [67.8, 71.1, 69.1], "cvsearch": [86.1, 93.4, 89.0],
        },
        "hr-bench_4k": {
            "direct": [75.8, 56.3, 66.0], "cvsearch": [93.0, 61.0, 77.0],
        },
        "hr-bench_8k": {
            "direct": [61.5, 53.3, 57.4], "cvsearch": [92.3, 63.0, 77.6],
        },
    },
}
METHOD_FILES = {
    "direct": "direct_answer.jsonl",
    "cvsearch": "cvsearch.jsonl",
    "logiv_disabled_control": "logiv/disabled.jsonl",
    "logiv_v2": "logiv_v2.jsonl",
}
PAPER_VSTAR_METHOD_FILES = {
    "direct": "direct_answer_paper.jsonl",
    "cvsearch": "cvsearch_paper.jsonl",
}


def _annotation_path(root: Path, benchmark: str) -> Path:
    return root / benchmark / f"annotation_{benchmark}.json"


def build_paper_vstar_reproduction(
    result_directory: Path,
    annotation_path: Path,
) -> dict[str, Any]:
    """Score the randomized-letter LLaVA V* protocol used by the paper."""
    annotations = load_json(annotation_path)
    if type(annotations) is not list:
        raise ValueError("trusted annotations must be one JSON list")
    methods = {}
    for method, filename in PAPER_VSTAR_METHOD_FILES.items():
        path = result_directory / filename
        rows = load_jsonl(path)
        methods[method] = {
            "path": str(path), "sha256": sha256_file(path),
            "records": len(rows),
            "metrics": score_vstar_letter_rows(annotations, rows),
        }
    delta = {
        metric: (
            methods["cvsearch"]["metrics"][metric]["accuracy"]
            - methods["direct"]["metrics"][metric]["accuracy"]
        )
        for metric in ("attribute", "spatial", "overall")
    }
    return {
        "protocol": "paper_letter",
        "annotation": {
            "path": str(annotation_path),
            "sha256": sha256_file(annotation_path),
            "records": len(annotations),
        },
        "methods": methods,
        "local_delta": delta,
    }


def _paper_reference_comparison(
    reference: Mapping[str, Sequence[float]],
    methods: Mapping[str, Mapping[str, Any]],
    metric_names: Sequence[str],
    *,
    protocol: str,
) -> dict[str, Any]:
    compared = {}
    for method in ("direct", "cvsearch"):
        claimed = dict(zip(metric_names, reference[method], strict=True))
        compared[method] = {
            metric: {
                "paper": claimed[metric],
                "local": methods[method]["metrics"][metric]["accuracy"],
                "local_minus_paper": (
                    methods[method]["metrics"][metric]["accuracy"]
                    - claimed[metric]
                ),
            }
            for metric in metric_names
        }
    return {
        "protocol": protocol,
        "methods": compared,
        "paper_gain": {
            metric: compared["cvsearch"][metric]["paper"]
            - compared["direct"][metric]["paper"]
            for metric in metric_names
        },
        "local_gain": {
            metric: compared["cvsearch"][metric]["local"]
            - compared["direct"][metric]["local"]
            for metric in metric_names
        },
    }


def build_scores(result_root: Path, annotation_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "paper_reference_only": PAPER_REFERENCE,
        "models": {},
    }
    all_disabled_control_deltas = []
    all_cvsearch_logiv_deltas = []
    for model in ("llava", "internvl"):
        model_result = {}
        for benchmark in ("vstar", "hr-bench_4k", "hr-bench_8k"):
            annotation_path = _annotation_path(annotation_root, benchmark)
            annotations = load_json(annotation_path)
            if type(annotations) is not list:
                raise ValueError("trusted annotations must be one JSON list")
            method_result = {}
            for method, filename in METHOD_FILES.items():
                path = result_root / model / benchmark / filename
                rows = load_jsonl(path)
                method_result[method] = {
                    "path": str(path), "sha256": sha256_file(path),
                    "records": len(rows),
                    "metrics": score_rows(benchmark, annotations, rows),
                }
            metric_names = (
                ("attribute", "spatial", "overall")
                if benchmark == "vstar" else ("fsp", "fcp", "overall")
            )
            deltas = {}
            for label, left, right in (
                ("cvsearch_minus_direct", "cvsearch", "direct"),
                (
                    "logiv_disabled_control_minus_cvsearch",
                    "logiv_disabled_control", "cvsearch",
                ),
                (
                    "logiv_v2_minus_disabled_control",
                    "logiv_v2", "logiv_disabled_control",
                ),
                ("logiv_v2_minus_cvsearch", "logiv_v2", "cvsearch"),
                ("logiv_v2_minus_direct", "logiv_v2", "direct"),
            ):
                deltas[label] = {
                    metric: (
                        method_result[left]["metrics"][metric]["accuracy"]
                        - method_result[right]["metrics"][metric]["accuracy"]
                    )
                    for metric in metric_names
                }
            all_disabled_control_deltas.append(
                deltas["logiv_v2_minus_disabled_control"]["overall"]
            )
            all_cvsearch_logiv_deltas.append(
                deltas["logiv_v2_minus_cvsearch"]["overall"]
            )
            model_result[benchmark] = {
                "annotation": {
                    "path": str(annotation_path),
                    "sha256": sha256_file(annotation_path),
                    "records": len(annotations),
                },
                "methods": method_result,
                "local_paired_deltas": deltas,
            }
            if model == "llava" and benchmark == "vstar":
                paper_protocol = build_paper_vstar_reproduction(
                    result_root / model / benchmark,
                    annotation_root / benchmark / "annotation_vstar_updated.json",
                )
                model_result[benchmark]["paper_protocol_reproduction"] = paper_protocol
                comparison_methods = paper_protocol["methods"]
                protocol = "paper_letter"
            else:
                comparison_methods = method_result
                protocol = "common_logits" if benchmark == "vstar" else "paper_release"
            model_result[benchmark]["paper_reference_comparison"] = (
                _paper_reference_comparison(
                    PAPER_REFERENCE[model][benchmark], comparison_methods,
                    metric_names, protocol=protocol,
                )
            )
        result["models"][model] = model_result
    if all(delta > 0 for delta in all_cvsearch_logiv_deltas):
        verdict = "higher_than_original_cvsearch_on_all_six_pairs"
    elif all(delta >= 0 for delta in all_cvsearch_logiv_deltas) and any(
        delta > 0 for delta in all_cvsearch_logiv_deltas
    ):
        verdict = "not_lower_than_original_cvsearch_with_some_positive"
    elif any(delta > 0 for delta in all_cvsearch_logiv_deltas):
        verdict = "mixed_vs_original_cvsearch"
    else:
        verdict = "no_positive_effect_vs_original_cvsearch"
    if all(delta > 0 for delta in all_disabled_control_deltas):
        audit_verdict = "positive_on_all_six_disabled_control_pairs"
    elif all(delta >= 0 for delta in all_disabled_control_deltas) and any(
        delta > 0 for delta in all_disabled_control_deltas
    ):
        audit_verdict = "nonnegative_on_all_disabled_control_pairs"
    elif any(delta > 0 for delta in all_disabled_control_deltas):
        audit_verdict = "mixed_vs_disabled_control"
    else:
        audit_verdict = "no_positive_effect_vs_disabled_control"
    result["verdict"] = {
        "classification": verdict,
        "basis": "logiv_v2_minus_original_cvsearch",
        "logiv_v2_minus_cvsearch_overall": all_cvsearch_logiv_deltas,
    }
    result["disabled_control_audit"] = {
        "classification": audit_verdict,
        "basis": "logiv_v2_minus_same-run-disabled-control",
        "logiv_v2_minus_disabled_control_overall": all_disabled_control_deltas,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scores = build_scores(args.result_root, args.annotation_root)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(scores, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(args.output)
    print(json.dumps(scores["verdict"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
