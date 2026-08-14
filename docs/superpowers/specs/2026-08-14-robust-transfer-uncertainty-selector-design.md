# Robust-Transfer Uncertainty Selector Design

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-14
- Verification Status: UNVERIFIED
- Version Label: robust_transfer_selector_design_v1

## Status and scope

This design implements the user-approved next-stage objective without further
intermediate approval. It treats every previously opened outcome as development
data. It preserves Stage 1, Stage 2, the fixed Stage 3 search observations, and
exact P0 fallback. It changes only offline policy selection and the shared
uncertainty-driven STOP, CONTINUE, BACKTRACK, and REPLACE decisions.

The work runs on one GPU when new inference is required. CPU-only replay and
policy fitting may run without a GPU. Existing successful observation artifacts
are immutable inputs and are never regenerated merely to tune the selector.

## Objective

Build one frozen, uncertainty-aware Stage 3 selector that transfers across
Qwen and InternVL instead of maximizing pooled development accuracy. The
primary optimization target is the worst backbone/cell outcome under strict
corruption and observation-cost constraints.

Development acceptance requires all of the following:

- nested validation with outer partition isolation and inner source-group
  isolation;
- all eight backbone/dataset cell deltas nonnegative;
- Qwen and InternVL deltas both strictly positive;
- at least ten net converted official units out of forty available oracle fixes,
  equivalently at least `+10/256` on the opened 112-topic pool;
- corrections greater than corruptions, with zero corruption preferred;
- no dataset-specific threshold, backbone-specific action rule, evaluator label,
  correctness, ground-truth box, or category at inference time;
- mean observed Stage 3 views no greater than `12.8`, with `11.52` as the
  preferred ten-percent-reduction target.

Scientific completion additionally requires one frozen evaluation on
MME-RealWorld-Lite in which Qwen and InternVL are each nonnegative and the
pooled delta is strictly positive.

## Alternatives considered

### A. Robust selection over the existing fixed observation pool — selected

Repair offline/runtime reachability, meter actual revealed views, and select a
shared state-machine policy with nested partition/source-group validation. This
preserves successful search stages, uses existing evidence, and directly
targets the observed failure: a development policy that transferred to
InternVL but harmed Qwen and TreeBench.

### B. Add new Stage 3 actions or regenerate search candidates — rejected

More actions could raise the candidate oracle, but the current pool already
contains forty oracle fixes and the post-fit selector converts twenty-three.
Generation is not the active bottleneck. This option costs GPU time and risks
breaking previously successful stages before selector transfer is solved.

### C. Fit dataset- or backbone-routed action policies — rejected

Routing can reproduce the opened `+23`, but it encodes the benchmark/backbone
failure pattern and weakens transfer. Backbone identity remains permitted only
for frozen numeric calibration strata; the action set, features, transition
logic, grids, and promotion gates remain shared.

## Immutable pipeline and fallback

The immutable prefix is:

1. Stage 1 ranking and evidence generation.
2. Stage 2 selected output and source reconstruction.
3. The six fixed Stage 3 branches, their render hashes, support probes, visit
   order, query/rank provenance, and budget ledgers.

Stage 2 is the P0 anchor. A Stage 3 replacement is allowed only through the
existing uncertainty/support risk model. Missing calibration, malformed
records, duplicate renders, query/rank drift, geometry drift, budget drift, or
unparseable provenance returns the exact reconstructed Stage 2 result via
`FALLBACK_P0`. Exhausted search returns `STOP_P0`.

## Reachability and cost contract

Offline candidate enumeration and real replay must traverse identical view
prefixes. In particular, an unparseable view ends the current branch; no later
view in that branch may become a fitting candidate. Every candidate snapshot
records the exact cumulative number of views revealed when it became
available. A no-replacement outcome records the exact views visited before
`STOP_P0`, including early branch termination.

Policy fitting ranks candidates using this runtime-equivalent observation
count. Generated decision reports remain the final authority for the `12.8`
hard cap and `11.52` preferred target.

## Development data and nested validation

The development pool has two named, already opened partitions:

- `development`: the original Stage 3 development observations;
- `validation_v3`: the previously locked regression observations whose labels
  are now opened.

Source identity is the existing `source_group` key, which already joins paired
HR-4K/HR-8K questions and the two backbones. No source group may occur on both
sides of a fold.

Outer validation leaves one named partition out. Within each outer training
partition, hyperparameters are selected by deterministic four-fold grouped
replay (or one fold per group when fewer than four groups exist). Every source
group occurs in exactly one inner validation fold and never in its calibrator.
The selected configuration is then fitted only on that outer training
partition and scored on the untouched outer partition. The process is repeated
in both directions. A configuration is development-eligible only if the
combined out-of-fold report meets every cell/backbone safety constraint.

After selection, the same configuration family is refitted on the complete
opened development pool for the external run. The final refit report is
explicitly labeled `opened_development_refit`, never unseen evidence.

## Shared robust objective

Candidate configurations retain the existing uncertainty, agreement,
calibrated minimum support, support gain, and P0 conflict-margin features. A
diagnosed v2 family may additionally use label-blind aggregates computed from
the same already-frozen views: mean/maximum/support dispersion, agreeing and
parseable view fractions, agreeing branch and role fractions, recent agreement,
and observation progress. These values expose evidence persistence that the
five minima/ratios erase; they never consume backbone, dataset, ordinal,
category, correctness, evaluator answer, or GT geometry.

The v2 head is one shared standardized logistic benefit/harm model. The feature
family, regularization, class weighting, risk penalty, decision boundary,
minimum observation count, maximum observation count, and minimum agreeing-view
count come from finite authenticated grids. They are selected for the whole
pool, never independently by dataset, backbone, or answer type. Legacy numeric
heads remain loadable only for reproducibility and exact fallback comparison.

The shared structural default requires two distinct render hashes before
replacement. A single high-support observation may keep search alive but cannot
replace P0. A global maximum-observation value stops at P0 when its budget is
reached. This restores the independent-confirmation safety contract and makes
the cost cap an actual runtime constraint rather than only a ranking term.

Feasible candidates are ranked lexicographically by:

1. zero corruption, then corrections greater than corruptions;
2. nonnegative delta in every available cell;
3. strictly positive delta for both backbones;
4. largest minimum backbone delta;
5. largest minimum cell delta;
6. at least ten net converted units, then larger net gain;
7. fewer actual observations, preferring `<=11.52` and requiring `<=12.8`;
8. stable declared grid order.

If no candidate meets the complete target, selection emits a structured
failure report and preserves P0 rather than weakening a gate. Search may
continue through later development iterations, but no failed policy is
promoted as transferable.

## Calibration versus action routing

Backbone/answer-type calibration heads may map different raw score scales onto
the same benefit/harm semantics. They are not action rules: all heads consume
the same five features and feed the same shared transition threshold and state
machine. Dataset names never enter calibration lookup or replay decisions.

Any candidate requiring a dataset-specific threshold, action whitelist,
hard-coded ordinal, evaluator category, or correctness-dependent transition is
invalid even if its score is higher.

The robust-transfer v2 family is stricter: it fits a single global head and a
single global action configuration across both backbones and all answer types.
This removes the remaining ambiguity between numeric calibration and
model-specific routing. Per-stratum legacy heads are retained in the loader,
not searched by v2.

## Diagnosed v2 alternatives

Three remedies were compared after the first exact nested replay failed at
`-2` with Qwen `-3` and mean observations `13.62`:

1. Add shared aggregation/topology features and a two-view confirmation gate —
   selected. A throwaway outer-partition diagnostic reached combined `+12`,
   Qwen `+5`, InternVL `+7`, and mean observations `7.83`; its sole corruption
   was a one-view decision, directly motivating the structural gate.
2. Add a hand-authored safe feature box — rejected. Although one box found two
   Qwen corrections without opened corruption, it was discovered post hoc and
   behaved like a sparse special case rather than a fitted transferable model.
3. Add more generic classifiers over the original five features — rejected.
   Global/per-stratum trees, nearest-neighbor prototypes, and linear heads all
   failed cross-partition safety, showing that model capacity was not the root
   cause.

The aggregate logistic and safe-leaf follow-ups were also rejected after exact
runtime replay. Deterministic four-fold grouped selection produced combined
`+6` but negative InternVL/HR-8K and Qwen/TreeBench cells; safe leaves produced
zero net. These failures establish that the scalar observation summaries do
not contain enough transferable information to decide replacements safely.

## Pairwise verifier extension

The next method adds one bounded action after candidate search, without changing
Stage 1, Stage 2, the six Stage 3 branches, or P0 fallback. At observation eight,
the shared proposer considers only answers supported by at least two distinct
render hashes. It takes the candidates from the latest reachable checkpoint
and ranks them by agreement, mean calibrated support, P0 conflict margin, and a
canonical deterministic tie break. Development inference records a superset at
agreement `>=0.4`; the final agreement boundary is selected from the finite
shared grid `{0.4, 0.5, 0.6}` by nested validation.

For a feasible proposal, a deterministic evidence sheet contains the whole
image and two highest-support agreeing crops reconstructed from frozen crop
coordinates. The same backbone answers two pairwise questions. One lists P0 as
Answer 1 and the candidate as Answer 2; the other reverses the order. The
question, options, and two inference-time answers are permitted inputs. Dataset,
backbone identity, ordinal, evaluator answer, correctness, category, and GT
geometry are not prompt inputs.

Each comparison uses the model's two-choice losses. Losses are converted to a
candidate probability, making Qwen and InternVL outputs share a normalized
scale. Replacement requires both reversed-order comparisons to choose the
candidate and their minimum candidate probability to clear one threshold from
`{0.5, 0.6, 0.7, 0.8, 0.9}`. Otherwise the method returns exact P0. The two
verifier calls count as observations, so proposal at eight yields a hard per-topic
maximum of ten, below both observation targets.

This differs from the rejected Phase-7 cross-view confirmation: that stage
independently re-answered an image without exposing the competing P0/candidate
answers and selected none. The new action is an order-symmetrized contrastive
test whose only job is to reject an unsupported replacement.

## Candidate-free verifier refinement

Exact development replay rejected the candidate-conditioned extension and its
shared-external variants: same-backbone pairwise evidence concentrated gains,
Cosmos and InternVL cross-backbone pairwise evidence introduced corruptions,
and LLaVA/Cosmos candidate-free passes did not transfer safely. These remain
negative controls; none is eligible for policy freezing.

The retained refinement uses Qwen2.5-VL-32B as one shared candidate-free source
answerer at a manifest-bound 4,194,304-pixel processor budget. Its prompt is the
original question and choices only. It receives neither P0 nor a Stage-3
candidate, and the same checkpoint, prompt construction, confidence boundary,
and action rule apply to both evaluated backbones. Exact matching render,
prompt, choice, checkpoint, and pixel-budget hashes permit one source answer to
be reused across backbones; a mismatch triggers real inference or fail-closed
P0, never approximate reuse.

The fixed cascade first evaluates the nested aggregate-risk candidate. A
feasible candidate-free answer with confidence at least 0.6 vetoes that
candidate only when their canonical answers disagree. If aggregate search
falls back, the existing observation-eight proposal may be accepted only when
proposal agreement is at least 0.4, verifier confidence is at least 0.6, and
the two canonical answers agree exactly. All other paths preserve exact P0.

On the two current opened partitions this cascade has reached `+16/512`, with
16 corrections, zero corruptions, Qwen `+7`, InternVL `+9`, all eight cells
nonnegative, and mean observations `9.04`. This is not yet promotable: the
scaled current-pool gate is `+20/512`. A label-exposed zero-corruption
reachability scan found a shared-rule ceiling of `+19/512`; it remains
diagnostic-only. Historical verifier collection is used to run valid nested
selection and to diagnose whether bounded nonzero-corruption or broader
candidate-free coverage is required.

## MME-RealWorld-Lite gate

MME-RealWorld-Lite remains untouched while development continues. Before
inference, freeze and commit:

- code, config, model, dataset, policy, and calibration hashes;
- the exact sample manifest and evaluator contract;
- one-GPU launch commands and output paths;
- the rule that each backbone must have `delta >= 0`, pooled delta must be
  `> 0`, corrections must exceed corruptions, and all audits must pass.

Pure format compatibility, including five-option `A-E` parsing, must be tested
without reading outcomes. Smoke tests are label-blind. After the manifest is
frozen, each backbone is evaluated once. No threshold or feature is retuned
from those results.

## Testing and verification

Implementation follows red-green-refactor. Required tests cover:

- unreachable post-unparseable snapshots are excluded;
- snapshot and stop costs equal actual replay observations;
- outer partitions and source groups never enter their held-out calibrators;
- robust ranking prioritizes minimum backbone/cell safety before pooled gain;
- cost caps and strict-positive backbone gates fail closed;
- dataset/evaluator fields cannot affect decisions;
- all legacy v1/v2 policies and exact fallback behavior still load and replay;
- MME five-option compatibility is label-blind and deterministic.

Focused tests, the complete unit suite, deterministic artifact regeneration,
hash audits, and the frozen external report are required before completion.

## Stop and iteration rules

No user approval is required between development iterations. Each failed
hypothesis is recorded, then the next change must address a diagnosed failure
with one factor at a time. Previously passing stages remain available as exact
fallbacks. Full external inference does not start until CPU tests and the
frozen manifest pass. A genuine inability to obtain the external dataset/model
artifacts or a required new authority is reported as a blocker rather than
bypassing the final gate.
