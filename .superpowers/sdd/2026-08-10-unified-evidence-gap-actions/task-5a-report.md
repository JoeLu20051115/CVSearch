# Task 5a report: frozen Phase-2 oracle and launch provenance

## Scope

- Added the disabled P2 sibling config. Its loaded effective settings differ
  from the enabled config only in `config_id`, `next_enabled`, and
  `next_admission_mode`.
- Added a strict evaluator-only scorer for paired V*, HR-4K, and HR-8K P2A
  artifacts, including whole-topic HR selection, exact official A--D parsing,
  deterministic 10,000-topic bootstrap intervals, the preregistered dev gate,
  and the conditional full-V* `171/191` preservation gate.
- Added content-addressed launch manifests and made their canonical SHA256 the
  JSONL run fingerprint. Resume validates the sidecar before opening the
  checkpoint writer.
- Bound loaded and file config, selected annotations and source images, Qwen
  and processor files, SAM checkpoint, spaCy model, optional CLIP, environment
  package versions, visible GPU UUIDs, runner/Qwen/SAM/tree/utils/scorer code,
  all `cvsearch/evidence_gap/**/*.py`, and all local `sam3/**/*.py` sources.
- Recovery/Vault and GPU inference were not opened or run.

## TDD evidence

- Initial RED: the focused modules failed to import because
  `cvsearch.eval.phase2_oracle` and `cvsearch.evidence_gap.provenance` did not
  exist.
- Provenance RED: blanket symlink rejection failed the explicit Hugging Face
  snapshot-link test; the implementation now permits only referents inside the
  explicit `models--*/` store and binds link target, resolved path, and content.
  Broken and escaping links fail closed. Arbitrary config/annotation/image
  symlinks remain rejected.
- Resume RED: changing a selected image after a partial run initially did not
  have a launch-sidecar check; the runner now rejects resume before inference.
- Code coverage RED: mutations to `modeling_sam3.py`, `tree.py`, `utils.py`,
  the Phase-2 scorer, and local SAM source initially left the revision
  unchanged; all now change it.
- Bootstrap RED: the exact first-eight-digest-bytes draw rule was absent; the
  known `vstar, replicate=0, draw=0, N=37` draw is frozen to index `36`.

## Verification

- Focused scorer/provenance/launcher: `26/26` passed.
- Full CPU suite: `324/324` passed.
- Both runner and scorer `--help` entry points passed.
- `py_compile` passed for all changed Python files and tests.
- `git diff --check` passed.
- Actual Qwen snapshot content preflight (read-only, no model load/GPU
  inference): 14 controlled symlinks, 16,595,961,188 bytes, manifest SHA256
  `42fa5e17cdd97ffadb316cd32a8f9f2f587c768db5bd35dab31a51527e19f487`,
  completed in 8.04 seconds.

## Gate semantics

- Labels are accessed only after pair, revision, launch manifest, config,
  P0-anchor, trace, status, and budget validation.
- V* oracle correctness is P0-correct OR feasible-candidate-winner-correct.
- HR candidate raw strings are scored as one four-shuffle state per topic;
  ties retain P0 and `candidate_stability.output` is never scored.
- P2B is warranted only when both HR deltas are non-negative and their minimum
  is strictly positive. CIs are reported but cannot change the point gate.
- If warranted, full gate-0.6 V* disabled and enabled artifacts must be paired
  at the same launch identity, preserve emitted outputs exactly, and score
  `171/191`.
