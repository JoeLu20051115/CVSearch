from copy import deepcopy
from typing import Any, Mapping, Sequence


def reconstruct_quick_gate(
    direct_rows: Sequence[Mapping[str, Any]],
    search_rows: Sequence[Mapping[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    if len(direct_rows) != len(search_rows):
        raise ValueError("direct and search rows must be paired")
    return [
        deepcopy(direct_row if search_row["root_ans_conf"] > threshold else search_row)
        for direct_row, search_row in zip(direct_rows, search_rows)
    ]


def score_rows(benchmark: str, rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise ValueError("cannot score empty rows")
    if benchmark == "vstar":
        return 100 * sum(row["output"] == 0 for row in rows) / len(rows)

    correct = 0
    total = 0
    for row in rows:
        for answer, choice in zip(row["answer"], row["output"]):
            predicted = choice[0] if len(choice) == 1 else next(
                (letter for letter in choice if letter in ("A", "B", "C", "D")), ""
            )
            correct += answer == predicted
            total += 1
    if not total:
        raise ValueError("cannot score rows without answers")
    return 100 * correct / total


def baseline_envelope(
    direct_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    search_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    thresholds: Sequence[float] = (0.6, 0.8),
) -> dict[str, float]:
    if direct_rows.keys() != search_rows.keys():
        raise ValueError("direct and search benchmarks must match")
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return {
        benchmark: max(
            score_rows(benchmark, reconstruct_quick_gate(direct_rows[benchmark], search_rows[benchmark], threshold))
            for threshold in thresholds
        )
        for benchmark in direct_rows
    }
