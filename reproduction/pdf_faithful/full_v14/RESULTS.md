# Full cross-backbone result (v14)

## Protocol

- Six label-blind full runs: InternVL2.5-8B and LLaVA-OV-7B on V*, HR-Bench 4K, and HR-Bench 8K.
- Decisions were completed, trace-audited, hash-sealed, and committed at `36fc7c4` before score access.
- Scoring was performed once against the true locally reproduced CVSearch vectors.
- No policy or threshold was changed after scoring, and the score was not rerun.

## One-shot result

| Backbone | Benchmark | CVSearch | PDF-faithful | Delta | Corrections | Regressions | Net |
|---|---|---:|---:|---:|---:|---:|---:|
| InternVL2.5-8B | V* | 169/191 (88.48%) | 159/191 (83.25%) | -5.24 pp | 6 | 16 | -10 |
| InternVL2.5-8B | HR-Bench 4K | 617/800 (77.12%) | 631/800 (78.88%) | +1.75 pp | 36 | 22 | +14 |
| InternVL2.5-8B | HR-Bench 8K | 613/800 (76.62%) | 592/800 (74.00%) | -2.62 pp | 13 | 34 | -21 |
| LLaVA-OV-7B | V* | 121/191 (63.35%) | 120/191 (62.83%) | -0.52 pp | 0 | 1 | -1 |
| LLaVA-OV-7B | HR-Bench 4K | 599/800 (74.88%) | 607/800 (75.88%) | +1.00 pp | 17 | 9 | +8 |
| LLaVA-OV-7B | HR-Bench 8K | 580/800 (72.50%) | 578/800 (72.25%) | -0.25 pp | 12 | 14 | -2 |

| Pool | CVSearch | PDF-faithful | Delta | Corrections | Regressions | Net |
|---|---:|---:|---:|---:|---:|---:|
| InternVL2.5-8B | 1399/1791 (78.11%) | 1382/1791 (77.16%) | -0.95 pp | 55 | 72 | -17 |
| LLaVA-OV-7B | 1300/1791 (72.59%) | 1305/1791 (72.86%) | +0.28 pp | 29 | 24 | +5 |
| Combined | 2699/3582 (75.35%) | 2687/3582 (75.01%) | -0.34 pp | 84 | 96 | -12 |

## Pre-registered gate

The primary gate failed. Only LLaVA's pooled delta was positive. Four of six individual units regressed, InternVL pooled delta was negative, and total regressions exceeded total corrections. The result therefore does not support a claim that the frozen policy reliably improves both transferred backbones or all six backbone-dataset units.

The defensible positive result is narrower: both backbones improved on HR-Bench 4K, and LLaVA had a small positive pooled result. These observations are reported as transfer-boundary evidence, not as grounds for post-hoc tuning.

## Provenance

- Method seal: `cross-backbone-sealed-manifest.json`, SHA-256 `862c81ac1a4cf68932759d6816d5f737010a76487c0c9450d9c6f2b2e0641f2d`.
- Decision seal: `cross-backbone-decision-seal.json`, SHA-256 `ab34a124fb61a2c069dbb89790a6bd06f4f2757dfe8d9e53b46f7db4b610a0ef`.
- Scorer: `cvsearch/eval/pdf_scores.py`, SHA-256 `9dd9bbc9bd06266fa10d8f6357a70c29ce795cd59a8cc1f9b7d2f6e5e283f707`.
- All 1,182 decision rows and six launch manifests passed the frozen row-count, ordinal, runner, revision, config, GPU, and trace checks; runtime failures and independent-verifier fallbacks were both zero.
