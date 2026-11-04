# eval/

This folder holds the real evaluation questions and detailed results. Git ignores everything here except this README, because the labels quote corpus text and the questions reveal what the corpus contains.

| Path | Contents |
| --- | --- |
| `dev.jsonl` | Labeled development questions, used while tuning |
| `heldout.jsonl` | Held-out questions, written after tuning and run once |
| `results/` | Per-question output, including misses |

Aggregate metrics without question or corpus text go to `../docs/eval/`, which is committed.
