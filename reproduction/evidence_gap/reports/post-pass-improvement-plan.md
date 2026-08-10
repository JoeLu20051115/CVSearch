# Post-Pass Evidence-Gap Improvement Plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-10
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## Experiment Overview

- **Title**: Post-pass unified evidence-gap improvement
- **Objective**: Increase the weakest paired gain beyond the action-calibrated
  selector while preserving V* at or above `171/191` and using one policy for
  all benchmarks.
- **Hypothesis**: Most remaining headroom comes from distinguishing useful and
  harmful candidate states and from preserving global context during ZOOM,
  rather than from adding more action families.
- **Type**: analysis followed by frozen inference ablations

## Current Headroom

After the approved selector replay, the selected/oracle-candidate scores are:

| Benchmark | Selected | Candidate oracle | Uncaptured headroom |
| --- | ---: | ---: | ---: |
| V* | 171/191 | 173/191 | 2 cycles |
| HR-4K | 616/800 | 625/800 | 9 cycles |
| HR-8K | 618/800 | 624/800 | 6 cycles |

The oracle is diagnostic only and uses trusted correctness. It demonstrates
candidate-state headroom; it cannot be used by the inference selector.

## Ranked Experiments

### E1: Independent cross-view candidate confirmation

Before accepting a candidate, obtain an answer record from a second,
predeclared view of the same evidence and require canonical answer agreement
plus positive stability gain. The confirming view must differ visually, not
merely repeat the same prompt over the same pixels. This targets the 9/6-cycle
HR candidate-oracle gaps without benchmark routing.

- Expected value: highest, because useful states already exist in the pool.
- Main risk: extra Qwen cost and a conservative veto that loses true
  corrections.
- Development success: V* non-regression; non-negative delta at both HR
  resolutions; positive minimum HR delta; fewer corrupted topics per accepted
  action.

### E2: Context-preserving ZOOM composition

Render the high-resolution focus crop together with a fixed low-resolution
global/context thumbnail. Keep original coordinates, action budget, and answer
prompt fixed. This directly addresses the observed ZOOM failure mode in which
local detail improves while spatial or chart context disappears.

- Expected value: medium; it can make ZOOM useful instead of merely rejecting
  the HR boundary cases.
- Main risk: higher pixel/token cost and distraction from the global panel.
- Development success: accepted ZOOM has non-negative paired effect on both HR
  resolutions and retains the V* correction under a fixed composition.

### E3: Answer-independent evidence verifier

Replace the rejected same-Qwen support scalar with an independently observed
signal that checks required visible evidence rather than answer confidence.
Candidate replacement may use it only after one frozen calibration on
development data. Average and minimum support remain separate.

- Expected value: medium but scientifically stronger than same-model
  self-confirmation.
- Main risk: verifier miscalibration or accidental answer leakage.
- Development success: positive joint HR effect after a single predeclared
  threshold, with an explicit leakage audit and V* non-regression.

### E4: Candidate-generation refinement

Only after E1--E3, test one change at a time to EXPAND neighbourhood coverage
or ZOOM crop context. Do not reopen NEXT or SPLIT merely because they are
available: their earlier stages did not establish replacement value.

- Expected value: uncertain and costliest.
- Main risk: larger search space increases false-positive selection and GPU
  cost faster than useful evidence.
- Development success: candidate oracle rises on both HR resolutions without
  reducing V*, followed by a selector that captures a positive fraction of the
  new headroom.

## Setup

- **Language/Framework**: repository Python environment, frozen Qwen2.5-VL-7B
  and SAM 3 checkpoints
- **Entry Command**: to be frozen separately for each approved experiment; no
  experiment command is authorized by this plan alone
- **Working Directory**:
  `/mnt/data3/data_xingrui/lueq/.worktrees/cvsearch-evidence-gap`
- **Dependencies**: existing repository environment and verified model
  artifacts
- **Environment**: CPU for selector/scorer replay; one fixed GPU identity per
  inference partition when new observations are required

## Inputs

| Input | Path | Description |
| --- | --- | --- |
| Current result snapshot | `reproduction/evidence_gap/reports/phase6-full-result-snapshot.json` | Immutable scores and hashes |
| Raw phase-6 observations | `reproduction/evidence_gap/phase6_full/b080f86` | Existing P0/EXPAND/ZOOM observations |
| Exposure ledger | `docs/superpowers/specs/2026-08-09-hr-exposure-ledger.md` | Development and recovery boundaries |
| Approved selector design | `docs/superpowers/specs/2026-08-10-action-calibrated-selector-design.md` | v2 selector contract |

## Expected Outputs

| Output | Path | Format | Success criterion |
| --- | --- | --- | --- |
| Per-experiment design | `docs/superpowers/specs/` | Markdown | One factor, fixed hypothesis and stop rule |
| Development report | `reproduction/evidence_gap/reports/` | JSON | Exact hashes, paired deltas, cost, corrections/corruptions |
| Unified manifest v2 | future commit-bound path | JSON | Frozen before any Recovery outcome is opened |

## Monitoring Configuration

- **Timeout**: declared per GPU ablation; selector-only replay is CPU-only
- **Monitor files**: launch manifest, JSONL output, process status, GPU UUID
- **Experiment type override**: generic paired evaluation
- **Metric file**: experiment-specific report JSON
- **Metric key**: minimum paired delta across HR-4K and HR-8K

## Analysis Plan

- **Primary metric**: minimum topic-paired delta across HR-4K and HR-8K
- **Hard constraint**: V* `>= 171/191`
- **Secondary metrics**: corrected/corrupted topics and cycles by action,
  candidate-oracle capture rate, Qwen calls, processed pixels, latency
- **Comparison**: action-calibrated selector replay `171/616/618`
- **Statistical unit**: HR semantic topic; four shuffles are repeated measures
- **Resampling**: paired topic bootstrap with shared 4K/8K resamples after a
  candidate is frozen
- **Selection rule**: one change at a time; both HR point estimates
  non-negative and their minimum strictly positive; no full-set threshold
  retuning
- **Opening rule**: freeze code/config/artifact hashes in unified Recovery
  manifest v2, score Recovery-A once, and open Vault-B only after a strict
  positive point estimate at both resolutions with no intervening change

## Recommended Order

Implement the approved selector first because it requires no new inference and
closes the current engineering gate. Then run E1 on development data. If E1 is
too conservative, test E2 as an isolated state-transition change. E3 follows
only with a genuinely independent signal. E4 is deferred until selection of
the existing candidate pool is demonstrably better.
