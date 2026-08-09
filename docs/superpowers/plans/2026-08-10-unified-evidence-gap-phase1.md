# Unified Evidence-Gap Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace HR question-family routing and hard semantic projection with one global soft-evidence rule, preserve V* at or above 89.53, and establish the conservative ranking/history slice needed before enabling active evidence-gap actions.

**Architecture:** Exact CVSearch output is the anchor for every sample.  A pure fusion module converts shuffled HR predictions into semantic vote evidence and applies one global weight; a conservative ranker perturbs, but never replaces or prunes, CVSearch ordering.  Runtime integration records the anchor, evidence, replacement margin, cost, and exact fallback so existing traces can be replayed before new GPU work.

**Tech Stack:** Python 3.11, `unittest`, Qwen2.5-VL-7B-Instruct, SAM 3, existing CVSearch hooks, JSONL traces, official V*/HR evaluators.

## Global Constraints

- Use Qwen snapshot `cc594898137f460bfe9f0759e9844b3ce807cfb5` and the existing SAM/spaCy artifacts.
- Keep policy inputs restricted to `question`, `options`, `answer_type`, and `input_image`.
- Do not use question family, benchmark category, resolution, ordinal, labels, boxes, or target objects in fusion or fallback.
- Use exact gate `0.6`, root tolerance `0.05`, and maximum budget `512`; do not tune them in this phase.
- `gamma=0` must reproduce exact CVSearch output for every HR sample.
- V* must retain the verified root-fallback result `171/191 = 89.53%`.
- Candidate ranking must preserve the identity multiset and may not move a candidate by more than the configured displacement.
- Tune only `gamma` over `{0.0, 1.1, 1.5, 2.1, 4.1}` using grouped semantic topics.
- Full HR aggregates are exploratory; locked evaluation requires a new manifest before any Recovery outcome is read.

---

### Task 1: Pure global HR soft-evidence fusion

**Files:**
- Create: `cvsearch/evidence_gap/fusion.py`
- Modify: `cvsearch/evidence_gap/answers.py`
- Create: `tests/test_evidence_gap_fusion.py`

**Interfaces:**
- Consumes: `parse_option_block(block: str) -> dict[str, str]`, public
  `official_letter(raw_output: str) -> str | None`, and `AnswerRecord`.
- Produces: `soft_fuse_hr(option_blocks: Sequence[str], raw_outputs: Sequence[str], evidence: AnswerRecord, gamma: float) -> AnswerRecord`.

- [ ] **Step 1: Write failing validation and decision tests**

```python
BLOCKS = [
    "A. cat\nB. dog\nC. bird\nD. fish\n",
    "A. dog\nB. cat\nC. fish\nD. bird\n",
    "A. bird\nB. fish\nC. cat\nD. dog\n",
    "A. fish\nB. bird\nC. dog\nD. cat\n",
]
RAW = ["A", "A", "A", "A"]

class SoftFusionTest(unittest.TestCase):
    def test_zero_gamma_is_exact_raw_and_ties_keep_raw(self):
        evidence = aggregate_hr_answers(BLOCKS, ["A", "B", "C", "D"])
        self.assertEqual(soft_fuse_hr(BLOCKS, RAW, evidence, 0.0).output, RAW)

    def test_one_global_weight_can_correct_only_supported_shuffle_slots(self):
        evidence = aggregate_hr_answers(BLOCKS, ["A", "B", "C", "D"])
        result = soft_fuse_hr(BLOCKS, RAW, evidence, 2.1)
        self.assertEqual(result.canonical_answer, "cat")
        self.assertEqual(result.output, ["A", "B", "C", "D"])
        self.assertEqual(result.selected_from, "unified_soft_fusion")

    def test_unavailable_aggregation_returns_exact_raw(self):
        evidence = AnswerRecord(aggregation_available=False)
        self.assertEqual(soft_fuse_hr(BLOCKS, RAW, evidence, 4.1).output, RAW)

    def test_rejects_negative_nonfinite_bool_or_mismatched_inputs(self):
        evidence = aggregate_hr_answers(BLOCKS, ["A", "B", "C", "D"])
        for gamma in (-0.1, float("nan"), float("inf"), True):
            with self.subTest(gamma=gamma), self.assertRaises((TypeError, ValueError)):
                soft_fuse_hr(BLOCKS, RAW, evidence, gamma)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_fusion -v
```

Expected: `ImportError` for `cvsearch.evidence_gap.fusion`.

- [ ] **Step 3: Implement the pure fusion rule**

```python
def _nonnegative_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number

def soft_fuse_hr(option_blocks, raw_outputs, evidence, gamma):
    weight = _nonnegative_finite(gamma, "gamma")
    blocks = tuple(option_blocks)
    raw = tuple(raw_outputs)
    if len(blocks) != len(raw) or not blocks:
        raise ValueError("option blocks and raw outputs must be paired and nonempty")
    semantic_maps = tuple(parse_option_block(block) for block in blocks)
    if evidence.aggregation_available is not True or not evidence.groups:
        return AnswerRecord(output=list(raw), raw_outputs=raw,
                            selected_from="cvsearch_anchor",
                            aggregation_available=False,
                            aggregation_reason=evidence.aggregation_reason)
    valid_votes = sum(int(group["count"]) for group in evidence.groups.values())
    if valid_votes <= 0:
        raise ValueError("semantic evidence must contain a positive vote count")
    vote_probability = {
        semantic: int(group["count"]) / len(blocks)
        for semantic, group in evidence.groups.items()
    }
    fused = []
    for block, raw_output, semantic_map in zip(blocks, raw, semantic_maps):
        raw_letter = official_letter(raw_output)
        if raw_letter not in semantic_map:
            return AnswerRecord(output=list(raw), raw_outputs=raw,
                                selected_from="cvsearch_anchor",
                                aggregation_available=False,
                                aggregation_reason="invalid_raw_anchor")
        scores = {
            letter: float(letter == raw_letter) + weight * vote_probability.get(semantic, 0.0)
            for letter, semantic in semantic_map.items()
        }
        best_score = max(scores.values())
        winners = [letter for letter, score in scores.items() if score == best_score]
        selected = raw_letter if raw_letter in winners else min(winners)
        fused.append(raw_output if selected == raw_letter else selected)
    record = aggregate_hr_answers(list(blocks), fused)
    record.output = list(fused)
    record.raw_outputs = raw
    record.selected_from = "cvsearch_anchor" if fused == list(raw) else "unified_soft_fusion"
    return record
```

Rename `_official_letter` in `answers.py` to `official_letter`, update its local
call sites, and import that public helper in `fusion.py`; do not accept arbitrary
answer text.

- [ ] **Step 4: Run focused and answer regression tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_fusion tests.test_evidence_gap_answers -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add cvsearch/evidence_gap/fusion.py tests/test_evidence_gap_fusion.py cvsearch/evidence_gap/answers.py
git commit -m "feat: add global HR soft evidence fusion"
```

---

### Task 2: Add a unified runtime mode without question-family routing

**Files:**
- Modify: `cvsearch/evidence_gap/method.py`
- Modify: `tests/test_evidence_gap_method.py`
- Create: `reproduction/evidence_gap/configs/dev_unified_gate060_gamma210_budget512.json`

**Interfaces:**
- Consumes: `soft_fuse_hr(option_blocks, raw_outputs, evidence, gamma)` from Task 1.
- Produces: config keys `hr_fusion_mode: "global_soft"` and `hr_fusion_gamma: float`; the main HR output path never calls `_hr_semantic_projection_allowed`.

- [ ] **Step 1: Add failing config and end-to-end tests**

```python
def test_global_soft_config_accepts_one_gamma_and_rejects_question_router(self):
    config = load_method_config(base_config(
        config_id="unified-gamma210",
        hr_fusion_mode="global_soft",
        hr_fusion_gamma=2.1,
    ))
    self.assertEqual(config["hr_fusion_gamma"], 2.1)
    with self.assertRaises(ValueError):
        load_method_config(base_config(hr_fusion_mode="query_family"))

def test_all_hr_questions_use_same_fusion_callback(self):
    outputs = []
    for question in ("What color is the sign?", "How many signs are visible?",
                     "Where is the bus relative to the car?"):
        output, trace = run_fake_hr(question, gamma=2.1)
        outputs.append(output)
        self.assertEqual(trace.effective_config["hr_fusion_mode"], "global_soft")
        self.assertNotIn("query_family", json.dumps(trace.to_dict()).casefold())
    self.assertEqual(len(outputs), 3)

def test_global_soft_mode_does_not_change_vstar_root_fallback(self):
    before = run_fake_vstar(base_config(hr_fusion_mode="off"))
    after = run_fake_vstar(base_config(hr_fusion_mode="global_soft", hr_fusion_gamma=2.1))
    self.assertEqual(before[0], after[0])
```

- [ ] **Step 2: Run the three tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest \
  tests.test_evidence_gap_method.UnifiedFusionRuntimeTest -v
```

Expected: config keys are rejected before implementation.

- [ ] **Step 3: Extend strict config validation**

Add defaults:

```python
"hr_fusion_mode": "off",
"hr_fusion_gamma": 0.0,
```

Validate with:

```python
if result["hr_fusion_mode"] not in {"off", "global_soft"}:
    raise ValueError("hr_fusion_mode must be off or global_soft")
gamma = _runtime_number(result["hr_fusion_gamma"], "hr_fusion_gamma")
if gamma < 0.0:
    raise ValueError("hr_fusion_gamma must be non-negative")
result["hr_fusion_gamma"] = gamma
if result["hr_fusion_mode"] == "off" and result["hr_fusion_gamma"] != 0.0:
    raise ValueError("disabled HR fusion requires zero gamma")
```

Do not add the old query-family mode to the accepted values.

- [ ] **Step 4: Replace the HR question-family output block**

Use the exact CVSearch response as anchor and the already selected state only as semantic evidence:

```python
if policy["answer_type"] == "option_list":
    if method_config["hr_fusion_mode"] == "global_soft":
        final_record = soft_fuse_hr(
            policy["options"], raw_response, final_record,
            method_config["hr_fusion_gamma"],
        )
    else:
        final_record.output = copy.deepcopy(raw_response)
        final_record.selected_from = "cvsearch_anchor"
output = copy.deepcopy(final_record.output)
```

The old `_hr_semantic_projection_allowed` helper may remain for historical tests but must have no call site in the main runtime.

- [ ] **Step 5: Add the frozen development config**

```json
{
  "config_id": "unified-gamma210-v1",
  "mode": "root_search_fallback",
  "rerank_enabled": false,
  "beta": 0.6,
  "alpha": 0.65,
  "visual_lambda": 0.5,
  "quick_gate": 0.6,
  "root_fallback_tolerance": 0.05,
  "enable_zoom": false,
  "enable_split": false,
  "enable_expand": false,
  "enable_certified_stop": false,
  "hr_fusion_mode": "global_soft",
  "hr_fusion_gamma": 2.1,
  "max_mllm_calls": 512,
  "max_processed_pixels": 10000000000,
  "pixel_accounting": "source_image_area_per_logical_forward_approximation"
}
```

- [ ] **Step 6: Run method tests and full CPU suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_method -v
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest discover -s tests -v
git diff --check
```

Expected: all tests pass; no main-runtime query-family call remains.

- [ ] **Step 7: Commit**

```bash
git add cvsearch/evidence_gap/method.py tests/test_evidence_gap_method.py \
  reproduction/evidence_gap/configs/dev_unified_gate060_gamma210_budget512.json
git commit -m "feat: use unified HR evidence fusion"
```

---

### Task 3: Label-aware development replay with topic-level statistics

**Files:**
- Create: `cvsearch/eval/replay_unified_fusion.py`
- Create: `tests/test_replay_unified_fusion.py`

**Interfaces:**
- Consumes: completed evidence-gap JSONL containing exact raw output and `method_trace.history`.
- Produces: JSON report containing config hash, topic counts, per-resolution accuracy/delta, `min_delta`, changed-topic count, and selected `gamma`; it must not modify source JSONL.

- [ ] **Step 1: Write failing replay tests**

```python
def test_replay_scores_topics_not_shuffle_rows(tmp_path):
    path = write_two_topic_fixture(tmp_path)
    report = replay_paths({"hr-bench_4k": path}, gammas=(0.0, 2.1))
    self.assertEqual(report["hr-bench_4k"]["n_topics"], 2)
    self.assertEqual(report["hr-bench_4k"]["n_cycles"], 8)

def test_replay_gamma_zero_matches_exact_record_output(tmp_path):
    path = write_two_topic_fixture(tmp_path)
    report = replay_paths({"hr-bench_4k": path}, gammas=(0.0,))
    self.assertEqual(report["hr-bench_4k"]["candidates"]["0.0"]["delta"], 0.0)

def test_replay_rejects_duplicate_ordinals_and_missing_trace(tmp_path):
    with self.assertRaises(ValueError):
        replay_paths({"hr-bench_4k": write_invalid_fixture(tmp_path)}, gammas=(2.1,))
```

- [ ] **Step 2: Run replay tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_replay_unified_fusion -v
```

Expected: import failure.

- [ ] **Step 3: Implement strict read-only replay**

The replay must:

```python
from copy import deepcopy

GAMMAS = (0.0, 1.1, 1.5, 2.1, 4.1)

def topic_accuracy(row, output):
    return sum(a == official_letter(o) for a, o in zip(row["answer"], output)) / 4.0

def select_evidence(row):
    history = row["method_trace"]["history"]
    selected = row["method_trace"]["final_answer"]["selected_from"]
    matches = [item["answer"] for item in history
               if item["answer"]["selected_from"] == selected]
    payload = copy.deepcopy(matches[-1] if matches else row["method_trace"]["final_answer"])
    payload["raw_outputs"] = tuple(payload.get("raw_outputs", ()))
    payload["losses"] = tuple(payload.get("losses", ()))
    return AnswerRecord(**payload)
```

It must validate unique `_eg_ordinal`, exactly four answers/options/outputs per HR topic, strict JSON, and unchanged source file size/hash before and after replay.

- [ ] **Step 4: Implement the grouped selection objective**

```python
joint[gamma] = min(
    reports["hr-bench_4k"][gamma]["delta"],
    reports["hr-bench_8k"][gamma]["delta"],
)
selected = max(GAMMAS, key=lambda gamma: (joint[gamma], -gamma))
```

Report the result as exploratory development calibration, never frozen evidence.

- [ ] **Step 5: Run focused and full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest \
  tests.test_replay_unified_fusion tests.test_evidence_gap_fusion -v
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add cvsearch/eval/replay_unified_fusion.py tests/test_replay_unified_fusion.py
git commit -m "feat: replay unified fusion by HR topic"
```

---

### Task 4: Conservative query-aware ranking that preserves CVSearch as a strong expert

**Files:**
- Modify: `cvsearch/evidence_gap/ranking.py`
- Modify: `tests/test_evidence_gap_ranking.py`
- Modify: `cvsearch/evidence_gap/method.py`

**Interfaces:**
- Consumes: candidates already sorted by CVSearch and `CandidateScore.rank` from the existing query-aware scorer.
- Produces: `ConservativeQueryRanker(base_ranker, rho: float, max_displacement: int)` returning the same `(ranked_nodes, details)` hook schema.

- [ ] **Step 1: Write failing identity, displacement, and tie tests**

```python
def test_conservative_ranker_preserves_identity_and_max_displacement(self):
    ranked, details = ConservativeQueryRanker(
        reversing_ranker(), rho=0.25, max_displacement=1
    )(self.nodes, self.image, "question", ["evidence"])
    self.assertCountEqual(map(id, ranked), map(id, self.nodes))
    original = {id(node): index for index, node in enumerate(self.nodes)}
    self.assertTrue(all(abs(index - original[id(node)]) <= 1
                        for index, node in enumerate(ranked)))

def test_zero_rho_is_exact_cvsearch_order(self):
    ranked, _ = ConservativeQueryRanker(reversing_ranker(), rho=0.0,
                                        max_displacement=4)(
        self.nodes, self.image, "question", ["evidence"]
    )
    self.assertEqual(ranked, self.nodes)

def test_ties_keep_original_order_and_details_record_both_ranks(self):
    ranked, details = ConservativeQueryRanker(tied_ranker(), rho=0.5,
                                              max_displacement=2)(
        self.nodes, self.image, "question", ["evidence"]
    )
    self.assertEqual(ranked, self.nodes)
    self.assertIn("cvsearch_rank", details[0])
    self.assertIn("query_rank", details[0])
    self.assertIn("fused_rank_score", details[0])
```

- [ ] **Step 2: Run ranking tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_ranking -v
```

Expected: `ConservativeQueryRanker` is undefined.

- [ ] **Step 3: Implement bounded rank fusion**

Use reciprocal-rank fusion with the original position as the first expert:

```python
base_score = 1.0 / (60.0 + cvsearch_rank)
query_score = 1.0 / (60.0 + query_rank)
fused = (1.0 - rho) * base_score + rho * query_score
```

Sort by `(-fused, cvsearch_rank)`, then enforce the displacement constraint with a stable insertion algorithm.  Reject bool/non-finite `rho`, negative displacement, duplicate nodes, or details that do not align with query-ranked nodes.

- [ ] **Step 4: Add strict config keys**

Add:

```python
"ranking_mode": "cvsearch",
"ranking_rho": 0.0,
"ranking_max_displacement": 0,
```

Allowed modes are `cvsearch`, `query_linear`, and `conservative_rrf`.  `cvsearch` requires reranking disabled; `conservative_rrf` requires reranking enabled and uses the wrapper.  Do not sweep `beta`, `alpha`, or `visual_lambda` in this phase.

- [ ] **Step 5: Run focused, hook, and full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest \
  tests.test_evidence_gap_ranking tests.test_evidence_gap_hooks tests.test_evidence_gap_method -v
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest discover -s tests -v
git diff --check
```

Expected: all tests pass, and the zero-rho integration output is byte-equivalent to rerank-off output excluding trace metadata.

- [ ] **Step 6: Commit**

```bash
git add cvsearch/evidence_gap/ranking.py cvsearch/evidence_gap/method.py \
  tests/test_evidence_gap_ranking.py tests/test_evidence_gap_method.py
git commit -m "feat: preserve CVSearch prior in query ranking"
```

---

### Task 5: Evidence-state trace and replacement audit

**Files:**
- Modify: `cvsearch/evidence_gap/types.py`
- Create: `cvsearch/evidence_gap/state.py`
- Create: `tests/test_evidence_gap_state.py`
- Modify: `cvsearch/evidence_gap/method.py`

**Interfaces:**
- Produces: immutable `EvidenceStateScore(uncertainty, support_avg, support_min, coverage, normalized_cost)`,
  `EvidenceState(answer: AnswerRecord, features: EvidenceStateScore)`, and
  `score_state(state: EvidenceStateScore) -> float`.
- Extends `MethodTrace` with `anchor_answer`, `anchor_state_score`, `selected_state_score`, and `replacement_margin`.

- [ ] **Step 1: Write failing score and serialization tests**

```python
def test_state_score_rewards_support_and_penalizes_uncertainty_and_cost(self):
    strong = EvidenceStateScore(.1, .8, .7, .75, .2)
    weak = EvidenceStateScore(.4, .5, .3, .50, .4)
    self.assertGreater(score_state(strong), score_state(weak))

def test_equal_state_score_keeps_anchor(self):
    features = EvidenceStateScore(.2, .5, .5, .5, .2)
    anchor = EvidenceState(AnswerRecord(output="A"), features)
    candidate = EvidenceState(AnswerRecord(output="B"), features)
    selected, margin = select_state(anchor, candidate, tau=0.0)
    self.assertIs(selected, anchor)
    self.assertEqual(margin, 0.0)

def test_trace_serializes_anchor_and_replacement_without_labels(self):
    trace = MethodTrace(
        anchor_answer=AnswerRecord(output="A"),
        anchor_state_score=0.4,
        selected_state_score=0.5,
        replacement_margin=0.1,
    )
    encoded = json.dumps(trace.to_dict(), allow_nan=False)
    self.assertIn("anchor_state_score", encoded)
    self.assertNotIn("category", encoded)
```

- [ ] **Step 2: Run state tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_state -v
```

Expected: import failure.

- [ ] **Step 3: Implement one frozen composite**

```python
def score_state(state):
    return (
        0.35 * (1.0 - state.uncertainty)
        + 0.25 * state.support_min
        + 0.15 * state.support_avg
        + 0.25 * state.coverage
        - 0.10 * state.normalized_cost
    )

def select_state(anchor, candidate, tau):
    margin = score_state(candidate.features) - score_state(anchor.features)
    return (candidate, margin) if margin > tau else (anchor, margin)
```

All inputs must be finite and within `[0, 1]`; `tau` is finite and non-negative.  Phase 1 logs this score but does not tune its coefficients or use unavailable support to overwrite answers.

- [ ] **Step 4: Wire audit-only state fields into runtime**

Until the verifier exists, set `support_avg=support_min=coverage=0` and mark the trace source `support_status="not_observed"`.  Do not interpret zero as evidence against the candidate and do not enable state replacement from this incomplete score.

- [ ] **Step 5: Run focused and full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest \
  tests.test_evidence_gap_state tests.test_evidence_gap_types tests.test_evidence_gap_method -v
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest discover -s tests -v
```

Expected: all tests pass and existing output schemas remain evaluator-compatible.

- [ ] **Step 6: Commit**

```bash
git add cvsearch/evidence_gap/state.py cvsearch/evidence_gap/types.py \
  cvsearch/evidence_gap/method.py tests/test_evidence_gap_state.py
git commit -m "feat: audit unified evidence state selection"
```

---

### Task 6: Replay phase-1 ablations and freeze the first GPU candidates

**Files:**
- Create: `reproduction/evidence_gap/configs/dev_unified_rrf_gamma210_budget512.json`
- Create: `reproduction/evidence_gap/reports/phase1-dev-replay.json`
- Modify: `docs/superpowers/specs/2026-08-09-hr-exposure-ledger.md`

**Interfaces:**
- Consumes: Tasks 1–5 and existing exposed development artifacts only.
- Produces: immutable config IDs for `P1` global fusion and `P1R` conservative ranking plus fusion.

- [ ] **Step 1: Run the fixed gamma replay on exposed development artifacts**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python \
  cvsearch/eval/replay_unified_fusion.py \
  --hr4 /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/evidence_gap/hr_gate060_dev/7b34a4efc368/hr-bench_4k/dev_gate060.jsonl \
  --hr8 /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/evidence_gap/hr_gate060_dev/7b34a4efc368/hr-bench_8k/dev_gate060.jsonl \
  --gammas 0,1.1,1.5,2.1,4.1 \
  --output reproduction/evidence_gap/reports/phase1-dev-replay.json
```

Expected: `gamma=0` exactly matches anchor; selection uses topic counts `30` and `28`, not cycle count `232`; no Recovery file is read.

- [ ] **Step 2: Freeze gamma with the max-min rule**

Choose the smallest gamma within one topic-level standard error of the maximum `min(delta_4k, delta_8k)`.  Record every candidate, not only the winner.  If no gamma has non-negative delta on both resolutions, freeze `gamma=0` and classify fusion as rejected.

- [ ] **Step 3: Create conservative RRF config**

Copy the Task 2 config and change only:

```json
{
  "config_id": "unified-rrf025-d1-gamma210-v1",
  "rerank_enabled": true,
  "ranking_mode": "conservative_rrf",
  "ranking_rho": 0.25,
  "ranking_max_displacement": 1
}
```

The actual gamma suffix must match Step 2.  Do not create a grid over RRF parameters; `rho=.25,D=1` is the single predeclared conservative candidate.

- [ ] **Step 4: Amend the exposure ledger before new outcomes**

Record that manifest v1 is superseded without opening Recovery-A/B, list the phase-1 code/config hashes, and state that full-set results remain exploratory.

- [ ] **Step 5: Commit the frozen development decision**

```bash
git add reproduction/evidence_gap/configs/dev_unified_* \
  reproduction/evidence_gap/reports/phase1-dev-replay.json \
  docs/superpowers/specs/2026-08-09-hr-exposure-ledger.md
git commit -m "docs: freeze unified phase-one candidates"
```

---

### Task 7: GPU development ablation with V* preservation gate

**Files:**
- Produce: `reproduction/evidence_gap/phase1_dev/v1/{vstar,hr-bench_4k,hr-bench_8k}/unified-gamma210-v1.jsonl`
- Produce: `reproduction/evidence_gap/reports/phase1-gpu-dev.json`

**Interfaces:**
- Consumes: frozen `P1` and `P1R` configs from Task 6.
- Produces: paired development score/cost report across all three benchmarks.

- [ ] **Step 1: Verify clean code and free GPUs**

Run:

```bash
git status --short
git diff --check
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

Expected: clean worktree; selected GPUs have enough memory for one Qwen+SAM runtime each.

- [ ] **Step 2: Run P1 concurrently on V*, HR-4K, and HR-8K development splits**

Use one GPU per benchmark and the exact paths from the design:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python \
  cvsearch/perform_EGSearch.py --root-path / \
  --model-path /home/xingrui/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --annotation-path /mnt/data3/data_xingrui/lueq/CVSearch/datasets/hr_data \
  --sam-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/sam3/sam3.pt \
  --nlp-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/en_core_web_sm-3.8.0 \
  --clip-model-path /home/xingrui/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41 \
  --benchmark vstar --split dev --split-seed 260809 \
  --config reproduction/evidence_gap/configs/dev_unified_gate060_gamma210_budget512.json \
  --answers-file reproduction/evidence_gap/phase1_dev/v1/vstar/unified-gamma210-v1.jsonl

CUDA_VISIBLE_DEVICES=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python \
  cvsearch/perform_EGSearch.py --root-path / \
  --model-path /home/xingrui/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --annotation-path /mnt/data3/data_xingrui/lueq/CVSearch/datasets/hr_data \
  --sam-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/sam3/sam3.pt \
  --nlp-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/en_core_web_sm-3.8.0 \
  --clip-model-path /home/xingrui/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41 \
  --benchmark hr-bench_4k --split dev --split-seed 260809 \
  --config reproduction/evidence_gap/configs/dev_unified_gate060_gamma210_budget512.json \
  --answers-file reproduction/evidence_gap/phase1_dev/v1/hr-bench_4k/unified-gamma210-v1.jsonl

CUDA_VISIBLE_DEVICES=2 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python \
  cvsearch/perform_EGSearch.py --root-path / \
  --model-path /home/xingrui/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --annotation-path /mnt/data3/data_xingrui/lueq/CVSearch/datasets/hr_data \
  --sam-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/sam3/sam3.pt \
  --nlp-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/en_core_web_sm-3.8.0 \
  --clip-model-path /home/xingrui/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41 \
  --benchmark hr-bench_8k --split dev --split-seed 260809 \
  --config reproduction/evidence_gap/configs/dev_unified_gate060_gamma210_budget512.json \
  --answers-file reproduction/evidence_gap/phase1_dev/v1/hr-bench_8k/unified-gamma210-v1.jsonl
```

Expected counts are the frozen development counts for each benchmark.  Resume only when fingerprint and code revision match.

- [ ] **Step 3: Score P1 and enforce V* gate**

Expected acceptance:

```text
V* P1 output == verified root-fallback output on the same partition
HR-4K delta >= 0
HR-8K delta >= 0
min(HR deltas) > 0 for promotion
```

If V* differs, stop the candidate and fix cross-benchmark isolation before any HR interpretation.

- [ ] **Step 4: Run P1R only if P1 passes the V* gate**

Run the same three commands with the single RRF config.  Retain RRF only if V* does not regress and both HR resolutions are non-negative relative to P1, with a positive joint minimum or lower cost.

- [ ] **Step 5: Independently review traces and code**

The reviewer must verify no type routing, exact anchor preservation, candidate identity/displacement, budget accounting, and absence of evaluator-only policy fields.  Any Critical or Important finding blocks promotion.

- [ ] **Step 6: Write and commit the development report**

Report accuracy, paired changes, changed topics, MLLM calls, processed pixels, p50/p95 latency, search mode, selected source, and forced-return counts.  Commit only the compact report/config/docs; do not stage JSONL or logs.

---

### Task 8: Decide whether to enter active-action Phase 2 or locked evaluation

**Files:**
- Modify: `docs/superpowers/specs/2026-08-09-hr-exposure-ledger.md`
- Create when warranted: `docs/superpowers/plans/2026-08-10-unified-evidence-gap-actions.md`
- Create when warranted: `reproduction/evidence_gap/manifests/unified-v2.json`

**Interfaces:**
- Consumes: independently reviewed Task 7 report.
- Produces exactly one next state: active-action implementation or a frozen manifest.

- [ ] **Step 1: Apply the promotion rule**

If either HR resolution is non-positive, do not tune more fusion/ranking weights.  Write the active-action plan in PDF order: `NEXT`, general `ZOOM`, `EXPAND`, `SPLIT`, verifier, certified stop.  The first active-action task must introduce only `NEXT` and one gap threshold.

If both HR resolutions are positive and V* is preserved, freeze the winning Phase-1 configuration; do not open Recovery outcomes yet.

- [ ] **Step 2: Audit provenance before manifest v2**

Hash exact code revision, config, Qwen/SAM/spaCy/CLIP artifacts, annotations, baselines, scorer, split ledger, and commands.  The scorer must accept the new frozen config rather than `local-perceptual-v1`.

- [ ] **Step 3: Run full CPU verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest discover -s tests -v
git diff --check
git status --short
```

Expected: all tests pass, no diff errors, no unexplained files.

- [ ] **Step 4: Request independent manifest review**

The reviewer checks that manifest v1 is explicitly superseded before outcomes, A/B remain unopened, v2 binds the new formula/config, and aggregate-only scoring cannot reveal per-topic changes.

- [ ] **Step 5: Commit the decision artifact**

```bash
git add docs/superpowers/specs/2026-08-09-hr-exposure-ledger.md \
  docs/superpowers/plans/2026-08-10-unified-evidence-gap-actions.md \
  reproduction/evidence_gap/manifests/unified-v2.json
git commit -m "docs: gate unified controller evaluation"
```

Add only files that exist for the selected branch; never create an empty placeholder plan or manifest.
