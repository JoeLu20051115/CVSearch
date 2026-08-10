# Phase 15 Selected-Bundle Publisher

The publisher converts the committed B8 Vstar and held-fixed B7 HR vectors
into one auditable LOGIV_V2 release bundle.  It is not a selector and may not
change any output.

Before parsing predictions, it authenticates the Phase-15 full-freeze report
against its source-embedded SHA-256 trust root and authenticates every raw
prediction/manifest byte stream against that report.  Trusted annotation
files are likewise authenticated before parsing.  Exact record schemas reject
unknown metadata.

The publisher scores Vstar by its frozen canonical option-index protocol and
HR with `official_letter`, emits canonical `selected.jsonl` and per-benchmark
manifests, then publishes all three directories plus one suite manifest using
a no-replace atomic directory rename.  The suite gate requires strict gains
over 171/191, 616/800, and 618/800.
