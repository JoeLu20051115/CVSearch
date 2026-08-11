#!/usr/bin/env python3
"""Independent matched-ordinal scoring for PDF search versus original CVSearch."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from cvsearch.evidence_gap.answers import official_letter


_BENCHMARKS = frozenset({"vstar", "hr-bench_4k", "hr-bench_8k"})


def _units(benchmark: str, row: Mapping[str, Any]) -> tuple[bool, ...]:
    if benchmark == "vstar":
        output = row.get("output")
        if isinstance(output, bool) or not isinstance(output, int):
            raise ValueError("V* output must be a plain option index")
        return (output == 0,)
    answers = row.get("answer")
    outputs = row.get("output")
    if (
        not isinstance(answers, list) or not isinstance(outputs, list)
        or len(answers) != 4 or len(outputs) != 4
        or any(not isinstance(item, str) for item in answers + outputs)
    ):
        raise ValueError("HR-Bench rows need four answer and output strings")
    return tuple(answer.strip().upper() == official_letter(output) for answer, output in zip(answers, outputs))


def compare_pdf_to_cvsearch(
    benchmark: str,
    pdf_rows: Sequence[Mapping[str, Any]],
    cvsearch_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if benchmark not in _BENCHMARKS:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    if not pdf_rows or not cvsearch_rows:
        raise ValueError("PDF and CVSearch rows must be nonempty")
    ordinals = []
    for row in pdf_rows:
        if not isinstance(row, Mapping):
            raise TypeError("PDF rows must be mappings")
        ordinal = row.get("_eg_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("PDF rows require a non-negative _eg_ordinal")
        ordinals.append(ordinal)
    if len(ordinals) != len(set(ordinals)) or ordinals != sorted(ordinals):
        raise ValueError("PDF ordinals must be unique and sorted")
    if ordinals[-1] >= len(cvsearch_rows):
        raise ValueError("PDF ordinal is outside the CVSearch baseline")

    base_correct = []
    pdf_correct = []
    changed_rows = 0
    for ordinal, proposed in zip(ordinals, pdf_rows):
        baseline = cvsearch_rows[ordinal]
        if not isinstance(baseline, Mapping):
            raise TypeError("CVSearch rows must be mappings")
        for key in ("input_image", "options"):
            if proposed.get(key) != baseline.get(key):
                raise ValueError(f"row {ordinal} does not align on {key}")
        if benchmark != "vstar" and proposed.get("answer") != baseline.get("answer"):
            raise ValueError(f"row {ordinal} does not align on answer")
        baseline_units = _units(benchmark, baseline)
        proposed_units = _units(benchmark, proposed)
        if len(baseline_units) != len(proposed_units):
            raise AssertionError("baseline and PDF scoring units do not align")
        base_correct.extend(baseline_units)
        pdf_correct.extend(proposed_units)
        changed_rows += proposed.get("output") != baseline.get("output")

    corrections = sum(not base and proposed for base, proposed in zip(base_correct, pdf_correct))
    regressions = sum(base and not proposed for base, proposed in zip(base_correct, pdf_correct))
    units = len(base_correct)
    baseline_hits = sum(base_correct)
    proposed_hits = sum(pdf_correct)
    return {
        "benchmark": benchmark,
        "rows": len(pdf_rows),
        "units": units,
        "ordinals": ordinals,
        "cvsearch_correct": baseline_hits,
        "pdf_correct": proposed_hits,
        "cvsearch_accuracy": baseline_hits / units,
        "pdf_accuracy": proposed_hits / units,
        "delta": (proposed_hits - baseline_hits) / units,
        "corrections": corrections,
        "regressions": regressions,
        "net_corrections": corrections - regressions,
        "changed_rows": changed_rows,
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("JSONL rows must be objects")
                rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(_BENCHMARKS), required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--cvsearch", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(
        compare_pdf_to_cvsearch(
            args.benchmark, _jsonl(args.pdf), _jsonl(args.cvsearch),
        ),
        sort_keys=True, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
