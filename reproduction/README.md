# CVSearch Main-Results Reproduction

This directory records the Qwen2.5-VL-7B reproduction of the direct-answer and CVSearch results on V* Bench, HR-Bench 4K, and HR-Bench 8K.

Generated outputs are organized as follows:

- `answers/qwen2.5-vl-7b/<benchmark>/direct_answer.jsonl`: raw direct-answer predictions.
- `answers/qwen2.5-vl-7b/<benchmark>/cvsearch.jsonl`: raw CVSearch predictions.
- `logs/<benchmark>-direct.log`: direct-answer inference log.
- `logs/<benchmark>-cvsearch.log`: CVSearch inference log.
- `logs/eval-*.log`: official evaluator output.
- `scores.json`: machine-readable paper/reproduction comparison.
- `report.md`: environment, commands, results, deviations, anomalies, and verdict.

The large raw answers and logs remain available locally but are ignored by Git. The compact score and report files are versioned.
