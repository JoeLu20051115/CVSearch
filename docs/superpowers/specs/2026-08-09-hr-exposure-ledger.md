# HR-Bench exposure ledger and recovery split

This ledger was frozen before evaluating the post-audit
`local-perceptual-only + selected-record stability` policy.  HR-Bench 4K and
8K share the same 200 question identities, so every ordinal below applies to
both resolutions.

## Prior outcome exposure

- Full-benchmark aggregate accuracy has been observed for earlier HR4 and HR8
  policies.  All later in-repository validation is therefore
  `aggregate-exposed internal validation`, not a pristine test.
- During diagnosis, per-question correctness changes were printed for HR4
  ordinals `22,30,153,158,176`.  These are `E_manual` (L3 exposure) and are
  excluded from both recovery splits at both resolutions.
- No Task11 partial result may be resumed, scored, or promoted to a complete
  evaluation.

## Development/exposed question cluster

The union of the previously frozen HR4/HR8 development questions and
`E_manual` contains 58 ordinals:

```text
1,9,12,15,16,21,22,23,24,25,28,30,38,39,41,42,44,48,51,54,58,59,68,70,71,76,80,83,84,86,101,105,106,109,110,125,127,130,134,140,147,152,153,158,159,160,166,167,168,173,176,177,179,181,182,186,188,194
```

These questions are development/exploratory only at both resolutions.

## Locked recovery splits

The remaining 142 question clusters were split before reading the new
policy's outcomes.  Within the public evaluator-only `single`/`cross` strata,
ordinals were sorted by
`SHA256("egap-recovery-20260809-v1:hr-paired:<ordinal>")`; Hamilton allocation
assigns 34 single and 37 cross questions to each 71-question split.  Category
metadata is used only here by the evaluator-side split and is never exposed to
the inference policy.

Recovery-A (71), SHA-256 of its comma-separated list
`d731fe78614b9165cb22deaf1956f2924d41e8c2be38edd5ed1fa477e9e9ae13`:

```text
2,4,5,7,8,10,13,14,19,29,32,34,35,37,40,43,46,62,65,67,69,81,82,85,87,90,91,97,100,102,104,108,111,114,115,116,118,120,122,123,124,126,128,129,131,132,137,138,141,143,144,146,148,150,151,154,155,157,164,169,174,175,183,185,187,190,191,192,196,198,199
```

Vault-B (71), SHA-256 of its comma-separated list
`ae6305b064ac93b861109297f906914870e20fe2a85289be6069855f32069c75`:

```text
0,3,6,11,17,18,20,26,27,31,33,36,45,47,49,50,52,53,55,56,57,60,61,63,64,66,72,73,74,75,77,78,79,88,89,92,93,94,95,96,98,99,103,107,112,113,117,119,121,133,135,136,139,142,145,149,156,161,162,163,165,170,171,172,178,180,184,189,193,195,197
```

## Opening rule

Recovery-A is scored first for the predeclared method-minus-CVSearch
topic-level effects at 4K and 8K.  Vault-B is opened only if both Recovery-A
point estimates are strictly positive without any code, config, threshold, or
query-family change.  Four shuffled options are repeated measures, not four
independent samples.  Both resolutions remain paired within the same question
cluster.  Regardless of the result, these splits do not erase prior aggregate
exposure; strict confirmation still requires external or secret-label data.

## Unified phase-1 supersession (no outcomes opened)

- The `local-perceptual-v1` recovery manifest is retired for the unified
  method: it was bound to the superseded query-family policy and must not be
  used to score, select, or certify the unified controller. Recovery-A,
  Vault-B, and their outcomes remain unopened.
- No unified manifest v2 has been created in this task. The frozen P1/P1R
  items below are development candidates only; they have **not** passed a
  Recovery or locked evaluation and cannot be promoted from this ledger.
- The fixed, exposed development inputs are the old `7b34a4efc368` artifacts:
  HR-4K `dev_gate060.jsonl` (163103 bytes,
  `7dac22b51114f31e1cb4ab5c7e28b39ec384d42e6bad32fad96d98595492c1b9`)
  and HR-8K `dev_gate060.jsonl` (152959 bytes,
  `f1859b925a7e7681bf5342ace35510bbdca64b1afb464ba61061787db5e073dd`).
  The replay snapshots them before and after reading; their hash and size were
  unchanged in the phase-1 replay.
- P1 freezes the same global soft-fusion rule at `gamma=2.1` and P1R is the
  single conservative-ranking-plus-fusion candidate (`rho=0.25`, maximum
  displacement `1`, same gamma). They use no question family, benchmark
  category, or resolution-specific parameter. The frozen code revision is
  `aa53ce8e0cc151268ecae13ddca12d40c2a27fda`; the P1 config SHA-256 is
  `e15991f0309e8d16935381865a3812ab1ae3192b0e6d589c1a70e3828568fe88`
  and the P1R config SHA-256 is
  `a1ca7e9c5d4aa1766123e97d5f7057d37739400bcf6d2c3fea8990b530a46928`.
  Per-file phase-1 code hashes and the complete candidate table are recorded in
  `reproduction/evidence_gap/reports/phase1-dev-replay.json`.
- Full-set aggregates had already been exposed and remain exploratory only.
  In the historical exposed-full proxy, `gamma=2.1` changed HR-4K
  `613/800 -> 610/800` and HR-8K `614/800 -> 611/800`; consequently no
  phase-1 fusion item is evidence of improvement. The old query-RRF clue had
  no HR change and regressed V*, so P1R is also only one predeclared
  development candidate, not a promotion claim.
