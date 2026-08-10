# Action-Calibrated Unified Selector Design

## Decision Status

Approved by the user on 2026-08-10. This document freezes the design before
production-code changes. The implementation remains pending a review of this
written specification.

## Objective

Replace the phase-5 selector's single inclusive stability-gain threshold with
one label-blind admission rule per action:

| Action | Admission rule |
| --- | --- |
| `EXPAND` | `stability_gain >= 0.25` |
| `ZOOM` | `stability_gain > 0.50` |

The selector remains identical across V*, HR-Bench 4K, and HR-Bench 8K. It
must not receive benchmark, resolution, question type, category, ordinal,
answer label, or correctness metadata. If multiple actions are admitted, the
existing maximum-gain choice and `EXPAND`, then `ZOOM` tie order remain fixed.

## Evidence and Terminology

The immutable current result is recorded in
`reproduction/evidence_gap/reports/phase6-full-result-snapshot.json`. Four
references must remain distinct in reports:

1. Paper Table 1: `90.1 / 76.6 / 75.6` percent.
2. Paper-aligned gate-0.8 reproduction: `167/191`, `610/800`, `601/800`.
3. Conservative cross-gate envelope: `167/191`, `613/800`, `614/800`.
4. Phase-6 execution-local paired P0: `170/191`, `613/800`, `614/800`.

In particular, the HR-4K reproduction baseline is `610/800 = 76.25%`.
`613/800 = 76.625%` is the conservative envelope and paired-P0 gate, not the
headline reproduction baseline.

Under the current inclusive `>= 0.25` rule, the selected scores are
`171/191`, `613/800`, and `618/800`. Exact replay of the approved rules on the
same raw observations gives `171/191`, `616/800`, and `618/800`. The only
changed decisions are HR-4K ZOOM ordinals 96 and 175: both have gain exactly
`0.5`, and together they corrupt three cycles. The V* ZOOM decision has gain
`0.5988136691230058` and remains admitted. Existing EXPAND decisions remain
unchanged.

This replay is aggregate-exposed engineering evidence. The rule is
label-blind at inference time, but it was chosen after full-set diagnosis; it
must not be described as pristine confirmatory evidence.

## Considered Designs

### Disable ZOOM

This removes the HR-4K corruptions but also removes the only accepted V* ZOOM
state, returning V* from `171/191` to `170/191`. It violates the preservation
constraint and is rejected.

### Add a support-delta veto

The existing same-Qwen support signal is not reliable enough for replacement.
A non-negative support veto would reject HR-8K ordinal 83, which is a genuine
one-cycle correction. It is rejected.

### Action-specific admission boundaries (selected)

EXPAND retains the inclusive development boundary. ZOOM uses a stricter,
exclusive boundary that rejects the two observed boundary cases while
retaining the higher-margin V* correction. This changes one selector degree
of freedom, not candidate generation, model inference, or benchmark routing.

## Selector Contract

The phase-5 module will replace `STABILITY_GAIN_THRESHOLD` with an immutable,
ordered action-rule value. Each rule contains exactly `action`, `operator`,
and `threshold`. The canonical order is `EXPAND`, then `ZOOM`; allowed
operators are explicit rather than inferred from threshold values.

Selection remains:

1. Snapshot and validate exact label-blind JSON inputs.
2. Compute candidate stability gain relative to P0.
3. Apply the candidate action's exact admission operator and threshold.
4. Among admitted candidates, select maximum gain; resolve an exact gain tie
   by the existing action order.
5. Retain the exact P0 output when no candidate is admitted.

No epsilon, rounding, benchmark exception, or resolution-dependent tolerance
is allowed. Python's exact floating-point comparison on the serialized values
is the contract.

## Provenance and Publication

The selector ID will advance to a v2 identifier that encodes both boundaries.
The selector source SHA-256 must be recalculated from the reviewed source.
`FrozenCombinedDecisionBatch`, its canonical digest material, and each output
manifest will replace the scalar `threshold` field with the exact ordered
`admission_rules` structure.

The combined-selection validator and final scorer must reject missing,
additional, reordered, or mutated rules, a wrong operator, a wrong threshold,
a stale selector ID, and a stale selector source hash. Suite-level checks must
require byte-for-byte-equivalent rule material across all three benchmarks.

The existing v1 bundles under `full_b080f86_fda2ccd` are immutable historical
artifacts and will not be overwritten. They remain auditable at revision
`fda2ccd1f31f3bb0e201b7ca7f3e69eae7134c48`. The v2 selector will replay the
existing phase-6 raw observations and publish to a new commit-bound output
directory. No Qwen or SAM GPU inference is required because candidate states,
outputs, and stability records do not change.

## TDD and Verification

Implementation begins with failing tests for these boundary cases:

- EXPAND gain exactly `0.25` is admitted;
- EXPAND gain below `0.25` is rejected;
- ZOOM gain exactly `0.50` is rejected;
- ZOOM gain above `0.50` is admitted;
- equal admitted gains retain the `EXPAND`, then `ZOOM` tie order;
- forbidden selector metadata remains rejected.

Combined-selection tests will cover canonical serialization and digest
binding. Final-scorer tests will cover every provenance mutation listed above
and verify that no trusted annotation is read before the frozen decisions are
complete. The focused tests, complete CPU suite, exact three-benchmark replay,
bundle hash audit, and process audit must all pass before publication.

Expected replay acceptance is:

| Benchmark | Expected | Versus reproduction baseline | Versus paired P0 |
| --- | ---: | ---: | ---: |
| V* | `171/191 = 89.53%` | `+4`, `+2.09` points | `+1`, `+0.52` points |
| HR-4K | `616/800 = 77.00%` | `+6`, `+0.75` points | `+3`, `+0.375` points |
| HR-8K | `618/800 = 77.25%` | `+17`, `+2.125` points | `+4`, `+0.50` points |

## Integrity Boundary

The v2 full replay is suitable for an internal engineering pass and for
preserving a reproducible implementation artifact. It is not a new untouched
test. Any further method choice must use only frozen development data until a
new unified Recovery manifest is committed. Recovery-A is opened once for the
frozen candidate; Vault-B is opened only if both paired HR resolution effects
on Recovery-A are strictly positive without intervening changes. External or
secret-label evaluation remains necessary for a strong paper-level claim.
