# Independent Paired Evidence Verifier Design

## Decision status

Approved under the user's continuing autonomous experiment mandate on
2026-08-10. This stage is B3. It implements the independent, answer-conditioned
visual verifier from pages 6--8 of the PDF and consumes the immutable B2 dense
candidate records without regenerating or relabelling them.

## Motivation

B2 produced feasible answer-changing candidates on every development
benchmark and a positive topic-replacement oracle of `+1 / +9 / +3` corrected
cycles. Its confidence-only selector nevertheless regressed V* and HR-8K.
Candidate generation therefore has useful headroom, while replacement is not
cross-resolution calibrated. B3 changes the selection evidence rather than
tuning the B2 confidence thresholds.

## Independent verifier

The answer generator remains the frozen Qwen2.5-VL-7B-Instruct checkpoint.
The verifier is the already-local, immutable
`nvidia/Cosmos-Reason1-7B` checkpoint. Its configuration is Qwen2.5-VL, but its
weights and checkpoint revision differ from the generator. The full verifier
artifact, processor, prompt, source, and output are content-addressed.

The fixed prompt is:

```text
Assess only whether the displayed visual observation directly supports the
proposed answer to the question. Use visible evidence only. Do not use prior
knowledge, do not infer missing details, and do not replace the proposed
answer.
Question: {question}
Proposed answer: {answer}
Does this visual observation directly support the proposed answer? Answer Yes
or No.
```

Support is the softmax probability of the final-position `Yes` logit within
the exact `Yes`/`No` pair. No generated rationale, self-reported confidence,
benchmark name, resolution, category, label, or evaluator metadata is visible
to the verifier.

## Paired observations and answer text

B3 processes only B2 records whose feasible dense answer differs from P0. It
reconstructs the three exact B2 evidence sheets from their source image and
frozen tile boxes and requires their pixel hashes to match B2. Both proposed
answers are scored on every one of the same three sheets, yielding six verifier
forwards per record.

For V*, the proposed answer is the exact option text at the answer index. For
HR, the four shuffled raw outputs are projected to one canonical option text
with the existing label-blind semantic aggregator. Both P0 and DENSE must have
an available canonical projection; otherwise B3 fails closed to v2.

For answer `a`, define:

- `avg(a)` as the mean support across the three sheets;
- `min(a)` as the minimum support across the three sheets;
- `wins(a,b)` as the number of sheets on which `support(a) > support(b)`.

This follows the PDF's separate average/minimum support roles. Comparing two
answers on identical sheets also cancels much of the verifier's global Yes/No
bias.

## Frozen global selector grid

A DENSE answer may supplement only a topic where v2 retained P0. It can never
veto an existing v2 ZOOM or EXPAND decision. It is selected only when all of
the following hold:

1. DENSE wins on at least two of the three paired sheets;
2. `avg(DENSE) - avg(P0)` is strictly greater than an average margin in
   `{0.00, 0.05, 0.10}`;
3. `min(DENSE) - min(P0)` is at least a minimum margin in `{0.00, 0.05}`.

The six combinations are frozen before development scoring. One identical rule
is used for V*, HR-4K, and HR-8K. A rule qualifies only if it strictly improves
all three development benchmarks. Selection maximizes the minimum
accuracy-point delta, then total corrected cycles, then the stricter average
and minimum margins.

## Integrity and cost

The runner never imports or accepts annotation paths. It validates the complete
B2 manifest, pair identities, candidate projection, rank digest, tile boxes,
and rendered sheet hashes before inference. Per record it stores only proposed
answers, exact support logits/probabilities and provenance, decisions, hashes,
and cost. The maximum planned development load is 60 verifier calls on V*, 96
on HR-4K, and 96 on HR-8K, below the existing 512-call per-benchmark ceiling.

Raw B3 outputs and all six complete decision vectors are frozen and committed
before development labels are reopened. Full/holdout labels remain sealed
unless one rule qualifies. If B3 fails, its immutable support records will be
used only for aggregate failure diagnosis before proceeding to generated
localization queries and lazy SPLIT/BACKTRACK in B4.

