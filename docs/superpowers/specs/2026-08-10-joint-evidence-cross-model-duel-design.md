# Joint-Evidence Cross-Model Duel Design

## Decision status

Approved under the user's continuing autonomous experiment mandate on
2026-08-10. This stage is B4. It follows the PDF's requirement to redesign a
failed verifier instead of loosening thresholds, while changing only the
verification module so that the next ablation remains interpretable.

## Motivation

B3's independent Yes/No support improved HR-4K under one rule but regressed V*
and HR-8K. The two answers were scored independently, leaving the result
exposed to the verifier's global Yes/No bias and to one crop dominating the
decision. B4 replaces absolute support with a direct, order-balanced answer
duel over all B2 evidence at once.

## Joint evidence observation

For every feasible B2 answer that differs from v2 P0, reconstruct the exact
three frozen B2 tiles. Render a deterministic `2x2` RGB sheet with fixed
448-pixel panels and eight-pixel separators:

1. top-left: first tile;
2. top-right: second tile;
3. bottom-left: third tile;
4. bottom-right: the full source image with all three tile boxes marked red.

Every panel is letterboxed without stretching. Tile identity, boxes, panel
hashes, source hash, and final sheet hash are bound to the B2 record.

## Order-balanced independent duel

Convert P0 and DENSE outputs to the same label-blind canonical answer text used
by B3. Cosmos-Reason1-7B receives the joint sheet and the original question in
two prompts. The first assigns P0 to A and DENSE to B; the second swaps the
assignments. At the final prompt position, compare only the exact A/B token
logits. DENSE must win both orders: B in the first prompt and A in the second.
No generated rationale or free-form parsing is used.

## Generator cross-check

The frozen Qwen2.5-VL-7B answer generator independently scores the two exact
answer texts with its existing multiple-choice loss on the same joint sheet.
DENSE must have the smaller loss. A B4 replacement therefore requires all
three decisions to agree:

- Cosmos, P0-first order: DENSE;
- Cosmos, DENSE-first order: DENSE;
- Qwen option loss: DENSE.

This is one fixed rule with no accuracy-tuned threshold. A tie, unavailable
canonical answer, tokenization mismatch, nonfinite logit/loss, render mismatch,
model failure, or any disagreement retains exact v2.

## Evaluation integrity and cost

B4 consumes only frozen B2 records and validated source images. The runner has
no annotation argument and forbids benchmark, resolution, category, ordinal,
truth, and evaluator metadata from the selector DTO. Existing v2 ZOOM/EXPAND
decisions cannot be vetoed.

Each changed candidate costs two Cosmos forwards and one Qwen loss forward:
30 calls for V*, 48 for HR-4K, and 48 for HR-8K. Raw observations and the one
complete decision vector are committed before development scoring. The rule
qualifies only if all three development deltas are strictly positive. Failure
leads to B5 generated localization queries plus lazy SPLIT/BACKTRACK; full
labels remain sealed.

