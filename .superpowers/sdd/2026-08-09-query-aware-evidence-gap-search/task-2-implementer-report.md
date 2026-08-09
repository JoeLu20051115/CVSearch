# Task 2 implementer report

## Scope and field choices

- Added `cvsearch/evidence_gap/types.py` and
  `tests/test_evidence_gap_types.py`; no Task 1 source or test was modified.
- `QueryPlan` carries the original query, target/localization phrases, evidence
  items, global-scope flag, and parsing fallback marker needed by Tasks 3 and
  7.
- `CandidateScore` keeps the four frozen ranking inputs plus relevance,
  visual, and final rank for Task 4's observability. `SearchCandidate` keeps
  canonical and tree identifiers, original-image bbox, source/depth/render
  metadata, score, and an in-memory `node`; its serialized contract contains
  only those required fields and never the runtime node.
- `AnswerRecord` contains evaluator output, semantic grouping/stability values,
  losses, and source selection for Tasks 3 and 6. `HistoryRecord`,
  `StepTrace`, and `MethodTrace` retain only the history tie-break, action/gap,
  budget, rank, and termination data described by the design and consumed by
  Tasks 5--7.
- All public dataclasses implement explicit recursive JSON-safe `to_dict()`.
  Tuples/lists/dicts and nested dataclasses are converted; non-finite floats
  are rejected. The implementation uses only the standard library.
- Actions are the one centralized `ACTION_VALUES` tuple: `ZOOM`, `SPLIT`,
  `EXPAND`, `NEXT`, `BACKTRACK`, `CERTIFIED_STOP`, and `FORCED_RETURN`.
- `BudgetLedger.consume` permits only the two declared counters, validates
  finite non-negative amounts before mutation, preserves integer semantics for
  MLLM calls, and rejects over-budget operations before changing a counter.
  `canonical_key` requires exactly four finite numeric (non-boolean)
  coordinates and therefore yields stable keys for equivalent Python integer
  and float boxes.

## RED evidence

The test file was created before `types.py`. The required command failed for
the intended missing-module reason:

```text
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_types -v
ModuleNotFoundError: No module named 'cvsearch.evidence_gap.types'
Ran 1 test in 0.000s
FAILED (errors=1)
```

## GREEN and verification

```text
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_types -v
Ran 7 tests in 0.031s
OK

CVSEARCH_REQUIRE_RETAINED_ARTIFACTS=1 PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_paper_parameters tests.test_evidence_gap_input tests.test_evidence_gap_baselines -v
Ran 16 tests in 0.039s
OK

git diff --check
exit 0
```

The round-trip test creates its JSON fixture in `tempfile.TemporaryDirectory`,
checks `json.loads(json.dumps(trace.to_dict()))`, and runs
`sys.executable -m json.tool <temporary-fixture>` with exit status zero. No
fixture is retained in the repository. Generated `__pycache__` files were
removed before committing.

## Files and commit

- `cvsearch/evidence_gap/types.py`
- `tests/test_evidence_gap_types.py`

Implementation commit:
`ed451c4927ba538de639b26abe19838170b5033d`
(`feat: add evidence-search state contracts`)
