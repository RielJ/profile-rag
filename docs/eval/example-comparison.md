# Retrieval comparison: example set

Rebuilt by `profile-rag eval` from the per-run reports in this folder.
Recall@k and MRR are means over answerable questions (n); unanswerable
questions are counted separately. A miss is a question with recall@k below 1.

| model | mode | k | recall@k | MRR | n | unanswerable | misses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nomic-v15 | dense | 3 | 0.667 | 0.708 | 12 | 2 | ex-04, ex-07, ex-08, ex-09, ex-10 |
| nomic-v15 | dense | 8 | 0.958 | 0.755 | 12 | 2 | ex-08 |
| nomic-v15 | hybrid | 3 | 0.667 | 0.708 | 12 | 2 | ex-04, ex-07, ex-08, ex-09, ex-10 |
| nomic-v15 | hybrid | 8 | 0.958 | 0.755 | 12 | 2 | ex-08 |
| nomic-v15 | mmr | 3 | 0.583 | 0.667 | 12 | 2 | ex-04, ex-07, ex-08, ex-09, ex-10, ex-11 |
| nomic-v15 | mmr | 8 | 0.625 | 0.667 | 12 | 2 | ex-04, ex-07, ex-08, ex-10, ex-11 |
| qwen3-06b | dense | 3 | 0.583 | 0.514 | 12 | 2 | ex-01, ex-04, ex-07, ex-08, ex-09, ex-10 |
| qwen3-06b | dense | 8 | 0.958 | 0.578 | 12 | 2 | ex-08 |
| qwen3-06b | hybrid | 3 | 0.583 | 0.514 | 12 | 2 | ex-01, ex-04, ex-07, ex-08, ex-09, ex-10 |
| qwen3-06b | hybrid | 8 | 0.958 | 0.578 | 12 | 2 | ex-08 |
| qwen3-06b | mmr | 3 | 0.625 | 0.528 | 12 | 2 | ex-04, ex-06, ex-07, ex-08, ex-11 |
| qwen3-06b | mmr | 8 | 1.000 | 0.614 | 12 | 2 | none |

<!-- Notes below this line are kept when the table is rebuilt. -->
## Notes

The example set has 14 questions over the fictional corpus in
`corpus-example/` (3 sources, 15 chunks). 12 questions are answerable and
2 are unanswerable. At k = 8 each question gets more than half of the
corpus, so k = 8 recall says little; the k = 3 rows separate the
retrievers more. One question moves recall by 1/12 = 0.083, so none of the
differences above is meaningful on its own.

Do not pick the default model from this table. The example corpus is
tiny and was written to exercise the pipeline, not to rank models. The
model decision needs the private dev set (`eval/dev.jsonl`) run on approved
sources, and a final check on the held-out set.

Hybrid rows equal dense rows because the keyword half found nothing: keyword
search uses `plainto_tsquery`, which needs every question word in one
chunk, and no chunk of `corpus-example/` matched any of the 14 questions
(0 of 14, counted with the same `plainto_tsquery('english', ...)` query against
`chunks_nomic_v15`). On this set, hybrid search is dense search.

### Heading path as the first line of each chunk

Measured once with a throwaway script: each chunk of `corpus-example/` was
indexed into a temporary table (`<table>_hp`) with its heading path, joined
by ` > `, as the first line of its content. Chunk IDs and the question set
were unchanged. The temporary tables were dropped afterwards. Splitting was
not changed. Baseline is the table above.

| model | mode | k | recall@k (baseline -> prefixed) | MRR (baseline -> prefixed) | misses with prefix |
| --- | --- | --- | --- | --- | --- |
| nomic-v15 | dense | 8 | 0.958 -> 1.000 | 0.755 -> 0.783 | none |
| nomic-v15 | hybrid | 8 | 0.958 -> 1.000 | 0.755 -> 0.783 | none |
| nomic-v15 | mmr | 8 | 0.625 -> 1.000 | 0.667 -> 0.794 | none |
| nomic-v15 | dense | 3 | 0.667 -> 0.750 | 0.708 -> 0.750 | ex-07, ex-08, ex-09, ex-11 |
| nomic-v15 | hybrid | 3 | 0.667 -> 0.750 | 0.708 -> 0.750 | ex-07, ex-08, ex-09, ex-11 |
| nomic-v15 | mmr | 3 | 0.583 -> 0.750 | 0.667 -> 0.750 | ex-07, ex-08, ex-09, ex-10 |
| qwen3-06b | dense | 8 | 0.958 -> 0.958 | 0.578 -> 0.816 | ex-08 |
| qwen3-06b | hybrid | 8 | 0.958 -> 0.958 | 0.578 -> 0.816 | ex-08 |
| qwen3-06b | mmr | 8 | 1.000 -> 1.000 | 0.614 -> 0.840 | none |
| qwen3-06b | dense | 3 | 0.583 -> 0.792 | 0.514 -> 0.792 | ex-07, ex-08, ex-11 |
| qwen3-06b | hybrid | 3 | 0.583 -> 0.792 | 0.514 -> 0.792 | ex-07, ex-08, ex-11 |
| qwen3-06b | mmr | 3 | 0.625 -> 0.792 | 0.528 -> 0.778 | ex-07, ex-08, ex-10 |

The prefix raised or kept every number in every row (n = 12). Because the
keyword half matched nothing, the gain comes from the embeddings, not from
keyword search. Whether the prefix also lets keyword search match heading
words is not measured here. Repeat the measurement on the dev set before
changing the splitter.
