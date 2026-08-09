# Unified Query-Aware Evidence-Gap Controller

## Objective

Build a single, training-free controller that uses the exact reproduced
`Qwen2.5-VL-7B-Instruct` checkpoint and the existing CVSearch/SAM candidate
generator on V* Bench, HR-Bench 4K, and HR-Bench 8K.  The engineering target is
to exceed the locked CVSearch envelope `87.43 / 76.625 / 76.75` while retaining
the already verified V* score `89.53`.  Once all three targets are exceeded,
further tuning remains development-only and seeks a larger paired gain without
departing from the evidence-gap method.

The final method must not route manually recognized question families to
different answer systems.  Every sample passes through the same query plan,
continuous evidence-gap scorer, action controller, state scorer, and fallback
rule.  A low evidence-gain state may retain the exact CVSearch answer, but that
decision uses the same continuous rule for every question and resolution.

## Superseded Exploratory Policy

The `local-perceptual-v1` HR query-family router is retained only as an
exploratory diagnostic.  It is not the main method and its Recovery-A run must
not be started.  The router, its type allowlist, and its exact raw fallback are
excluded from the unified controller.  Its tests continue to protect the old
artifact until it can be removed without invalidating historical provenance.

## Considered Approaches

### 1. Fusion-only patch

Apply one global semantic-vote weight to the retained CVSearch answers.  This
is cheap and provides an important diagnostic, but it does not implement the
PDF's evidence collection loop and has already shown substantial development
to full-set extrapolation risk.

### 2. Staged unified evidence-gap loop (selected)

Implement one controller for all questions, then enable one action or evidence
module at a time.  The controller is PDF-faithful, while the staged activation
makes regressions attributable.  A module survives only when its paired
development delta is non-negative on both HR resolutions and positive on the
joint objective, or when it preserves accuracy and measurably lowers cost.

### 3. Literal full controller in one run

Enable multi-query answering, four evidence gaps, all actions, independent
verification, history, and certified stopping at once.  This is closest to a
literal transcription but makes a failed score impossible to diagnose and is
too expensive for iterative calibration.

## Invariants

- The policy input contains only `question`, `options`, `answer_type`, and
  `input_image`; labels, category, index, target boxes, and target objects are
  evaluator-only.
- The Qwen checkpoint, SAM checkpoint, preprocessing, annotations, and official
  evaluators remain fixed.
- No question-type, resolution, category, ordinal, or answer-label feature is
  available to routing, fusion, or stopping.
- Candidate generation remains CVSearch/SAM.  Ranking may reorder candidates
  but may not prune them solely through an auxiliary similarity score.
- Every model observation is charged to the call and pixel ledger before it is
  executed.
- Every accepted action changes the canonical observation; no-ops are logged
  and suppressed.
- The exact CVSearch raw response remains the fallback anchor.  It is replaced
  only by the same predeclared evidence-gain rule for all samples.

## Unified Architecture

### Query plan and evidence requirements

The unchanged question is `q0`.  The existing Qwen checkpoint produces a
structured, answer-free plan containing localization targets and evidence
requirements.  Requirements describe visible target detail, relation context,
and coverage; they are not question-family labels and never select an answer
system.  Malformed plans use the deterministic answer-free fallback and record
that fact.

### Candidate state

CVSearch and SAM create the canonical region tree.  A state contains the focus
region, context regions, visited canonical keys, unvisited candidates, child
availability, spatial neighbours, answer distribution, evidence support,
history, and remaining budget.  Root, search, zoomed, split, and expanded views
are states in the same state space rather than separate methods.

### Continuous evidence gaps

At each state, the controller estimates four independent gaps in `[0, 1]`:

- `g_zoom`: the present focus lacks readable local detail;
- `g_split`: the region contains unresolved substructure;
- `g_expand`: required context or coverage is absent;
- `g_next`: the current branch is unsupported or has stopped progressing.

The primary scorer uses `q0`, answer-free evidence requirements, the current
observation summary, and compact history.  Candidate answers are excluded from
the gap prompt.  An analytic scorer based on crop scale, child availability,
context coverage, support, and progress provides a separately logged fallback.
Feasibility masking occurs only after gaps are scored.

### Actions

- `ZOOM` rerenders the same original coordinates at a higher useful token
  resolution.
- `SPLIT` reveals and ranks children of the current broad region.
- `EXPAND` adds the highest-ranked missing spatial/context neighbour while
  preserving the focus.
- `NEXT` moves to the next unvisited global candidate.
- `BACKTRACK` restores the best historical state with an unvisited branch;
  spent budget is never restored.

Every question uses this same feasible action set.  Evidence requirements and
continuous gaps affect action scores, but no hand-written question class gates
an action or answer path.

### HR semantic evidence and soft fusion

For the first diagnostic stage, the selected root/search record is fixed by
the existing tolerance `0.05`.  For topic `i`, shuffle `j`, candidate letter
`a`, and its canonical semantic option `pi_ij(a)`, define

```
S_ij(a; gamma) = 1[a == y_raw_ij] + gamma * q_i(pi_ij(a))
```

where `y_raw` is the exact gate-0.6 CVSearch answer and `q_i` is the four-shuffle
semantic vote distribution.  The highest score wins; a tie retains the exact
raw letter.  Unavailable aggregation also retains raw.  This is one global
formula for both HR resolutions and all questions.  `gamma=0` exactly recovers
CVSearch.

Because four votes produce decision boundaries only near
`{1, 4/3, 2, 4}`, development evaluates interval representatives
`{0, 1.1, 1.5, 2.1, 4.1}` rather than a dense sweep.  Grouped topic-level
selection uses

```
J(gamma) = min(delta_4K(gamma), delta_8K(gamma))
```

and the one-standard-error rule prefers the smallest qualifying weight.  This
single parameter is a diagnostic and initialization, not the final controller.
Existing development replay favours `gamma=2.1`, but a historical full proxy
for an equivalent hard decision missed the 4K baseline.  Consequently the
fusion-only stage cannot by itself justify a frozen claim.

### General historical state selection

After the fusion diagnostic, every observed state exposes four separate terms:

- answer uncertainty from canonical option evidence;
- minimum and average independent visual support;
- evidence requirement coverage;
- normalized inference cost and progress.

The controller ranks history with one predeclared composite shared across all
benchmarks.  A candidate state replaces the CVSearch anchor only if its score
exceeds the anchor by global margin `tau`.  Otherwise the exact raw response is
retained.  Fusion and state selection must not inspect question-family flags.

### Independent support and stopping

CLIP is disabled as a candidate reranker because existing HR runs made no
prediction changes and the V* development run regressed.  It is evaluated
separately in the PDF-intended role of answer-conditioned evidence verifier.
Support is calibrated within a state against neutral and contradictory text
and records average and minimum requirement support separately.

`CERTIFIED_STOP` requires all feasible gaps below threshold, low answer
uncertainty, adequate average support, and adequate minimum support.  Failure
of any gate continues, backtracks, or reaches a budgeted `FORCED_RETURN`.
Forced return never claims certification.

## Staged Ablation and Tuning

The executable stages share inputs, candidate pool, baseline anchor, budget,
and evaluator:

1. `P0`: exact gate-0.6 CVSearch anchor.
2. `P1`: one-parameter soft fusion, no new action.
3. `P2`: history plus `NEXT`.
4. `P3`: add general state-based `ZOOM`.
5. `P4`: add `SPLIT`.
6. `P5`: add `EXPAND` and `BACKTRACK`.
7. `P6`: add independent support and `CERTIFIED_STOP`.
8. `P7`: reconsider query-aware candidate reranking only after the action loop
   is positive; compare CVSearch order, verifier-free query score, and CLIP.

The quick gate remains `0.6`, root tolerance remains `0.05`, and budget `512`
is an engineering ceiling rather than a performance parameter.  Beta, alpha,
temperature, root weight, resolution-specific parameters, and question-family
thresholds are not searched together.  Each stage introduces at most one new
calibration degree of freedom.

For HR, the statistical unit is a semantic topic.  Four shuffles and paired
4K/8K resolutions are correlated observations, not extra independent samples.
A module is retained when both resolution deltas are non-negative and their
minimum is positive under grouped development resampling.  V* uses its frozen
development/holdout partition and must not regress from the verified `89.53`
configuration.  After feasibility, configuration selection maximizes the
minimum normalized gain across the three benchmarks, with V* `>= 89.53` as a
hard constraint; this prevents a large gain on one benchmark from hiding a
regression on another.

## Failure Diagnosis

- If `P1` fails, inspect corrections versus corruptions and revise soft
  evidence construction, not question types.
- If an action has no score effect, inspect feasibility, canonical observation
  identity, and whether its evidence reaches answer generation.
- If an action hurts, compare support/progress before and after it and change
  that action's state transition or admission gap only.
- If support hurts, audit verifier calibration and minimum-support behaviour;
  do not tune answer labels.
- If stopping hurts, separate premature certified stops from forced returns and
  adjust one stop threshold at a time.
- If one resolution improves and the other regresses, inspect shared topic
  pairs and pixel/token scale; do not add a resolution-specific rule.

Each diagnosis changes one module, adds a regression test, reruns focused and
full CPU tests, and obtains independent code review before the next GPU stage.

## Evaluation Integrity

The old full HR aggregate has been exposed, so full-set runs are exploratory
engineering evidence.  Existing Recovery manifest v1 is explicitly bound to
`local-perceptual-v1` and cannot silently certify the new method.

Before locked evaluation, create manifest v2 that supersedes v1 without opening
any Recovery outcomes.  It freezes the unified formula, selected configuration,
code revision, config and artifact hashes, budget, commands, split hashes, and
aggregate-only scorer.  Recovery-A is opened first.  Recovery-B may be opened
only if both 4K and 8K deltas on A are positive, with no intervening code or
configuration change.  Paired topic bootstrap uses 10,000 shared resamples.

Even a two-stage pass is reported as locked internal validation because the
benchmark and earlier full aggregate are public.  A strict paper-level claim
still requires new secret-label or external evaluation data.

## Acceptance Criteria

Engineering success requires all of the following:

- V* remains at or above `89.53`, HR-4K exceeds `76.625`, and HR-8K exceeds
  `76.75`;
- the same Qwen checkpoint and official evaluators are used;
- no question-type routing exists in the main method;
- every retained module has an isolated paired ablation;
- accuracy is reported with calls, processed pixels, latency, and termination;
- all CPU and integration tests pass and independent review reports no
  Critical or Important issue;
- full and locked results are clearly distinguished.

After the first three-dataset pass, further improvement searches remain on
development partitions, use single-factor changes, and maximize the weakest
of the three paired gains while preserving V*.  The locked vault is not
repeatedly queried to chase a higher public number.
