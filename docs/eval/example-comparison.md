# Retrieval comparison: example set

Rebuilt by `profile-rag eval` from the per-run reports in this folder.
Recall@k and MRR are means over answerable questions (n); unanswerable
questions are counted separately. A miss is a question with recall@k below 1.

| model | mode | k | recall@k | MRR | n | unanswerable | misses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nomic-v15 | dense | 3 | 0.750 | 0.750 | 12 | 2 | ex-07, ex-08, ex-09, ex-11 |
| nomic-v15 | dense | 8 | 1.000 | 0.783 | 12 | 2 | none |
| nomic-v15 | hybrid | 3 | 0.833 | 0.778 | 12 | 2 | ex-07, ex-08, ex-09 |
| nomic-v15 | hybrid | 8 | 1.000 | 0.794 | 12 | 2 | none |
| nomic-v15 | mmr | 3 | 0.750 | 0.750 | 12 | 2 | ex-07, ex-08, ex-09, ex-10 |
| nomic-v15 | mmr | 8 | 1.000 | 0.794 | 12 | 2 | none |
| qwen3-06b | dense | 3 | 0.792 | 0.792 | 12 | 2 | ex-07, ex-08, ex-11 |
| qwen3-06b | dense | 8 | 0.958 | 0.816 | 12 | 2 | ex-08 |
| qwen3-06b | hybrid | 3 | 0.833 | 0.736 | 12 | 2 | ex-07, ex-08, ex-09 |
| qwen3-06b | hybrid | 8 | 1.000 | 0.748 | 12 | 2 | none |
| qwen3-06b | mmr | 3 | 0.750 | 0.778 | 12 | 2 | ex-07, ex-08, ex-09, ex-10 |
| qwen3-06b | mmr | 8 | 0.917 | 0.812 | 12 | 2 | ex-07 |

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

### Splitter v2 and OR keyword search: before and after

The table above is splitter v2 with the new keyword search. Two changes
went in together:

1. Splitter v2. Each chunk's text starts with its heading path, joined by
   ` > `, then a blank line, then the body. Before (v1), headings were
   metadata only, so job titles and company names were invisible to both
   searches. Chunk IDs changed (`:v1:` to `:v2:`); `make index` replaced
   all 15 rows per table (15 added, 15 deleted).
2. Keyword search (hybrid mode only). `PGVectorStore` 0.0.19 builds its
   keyword query with `plainto_tsquery`, which ANDs every question word,
   and has no option to change that. The keyword half is now
   `retrievers/keyword.py`: the question's lexemes joined with OR, ranked
   by `ts_rank_cd`, fused with the dense list by reciprocal rank fusion
   (k = 60).

Dense and MMR rows change only through (1); hybrid rows change through both.

| model | mode | k | recall@k (before -> after) | MRR (before -> after) | misses after |
| --- | --- | --- | --- | --- | --- |
| nomic-v15 | dense | 3 | 0.667 -> 0.750 | 0.708 -> 0.750 | ex-07, ex-08, ex-09, ex-11 |
| nomic-v15 | dense | 8 | 0.958 -> 1.000 | 0.755 -> 0.783 | none |
| nomic-v15 | hybrid | 3 | 0.667 -> 0.833 | 0.708 -> 0.778 | ex-07, ex-08, ex-09 |
| nomic-v15 | hybrid | 8 | 0.958 -> 1.000 | 0.755 -> 0.794 | none |
| nomic-v15 | mmr | 3 | 0.583 -> 0.750 | 0.667 -> 0.750 | ex-07, ex-08, ex-09, ex-10 |
| nomic-v15 | mmr | 8 | 0.625 -> 1.000 | 0.667 -> 0.794 | none |
| qwen3-06b | dense | 3 | 0.583 -> 0.792 | 0.514 -> 0.792 | ex-07, ex-08, ex-11 |
| qwen3-06b | dense | 8 | 0.958 -> 0.958 | 0.578 -> 0.816 | ex-08 |
| qwen3-06b | hybrid | 3 | 0.583 -> 0.833 | 0.514 -> 0.736 | ex-07, ex-08, ex-09 |
| qwen3-06b | hybrid | 8 | 0.958 -> 1.000 | 0.578 -> 0.748 | none |
| qwen3-06b | mmr | 3 | 0.625 -> 0.750 | 0.528 -> 0.778 | ex-07, ex-08, ex-09, ex-10 |
| qwen3-06b | mmr | 8 | 1.000 -> 0.917 | 0.614 -> 0.812 | ex-07 |

Keyword hits (top 20, visibility filter applied), counted on both tables:

| keyword query | rows | questions with >= 1 keyword hit | answerable questions with a relevant chunk among the hits |
| --- | --- | --- | --- |
| `plainto_tsquery` (AND, before) | v1 | 0 of 14 | not measured |
| `plainto_tsquery` (AND) | v2 | 0 of 14 | not measured |
| OR of lexemes (after) | v1 | 14 of 14 | not measured |
| OR of lexemes (after) | v2 | 14 of 14 | 11 of 12 |

What the numbers say, with n = 12:

- Before, hybrid equalled dense in every row. Now hybrid differs from
  dense in all four hybrid rows, so the keyword half contributes.
- Hybrid k = 3 recall is the best k = 3 recall for both models (0.833).
  For both models it finds ex-11, which dense k = 3 misses, and ex-10,
  which MMR k = 3 misses. For qwen3-06b it misses ex-09, which dense
  k = 3 finds.
- For qwen3-06b, hybrid has a lower MRR than dense (k = 3: 0.736 vs 0.792;
  k = 8: 0.748 vs 0.816). For nomic-v15, hybrid MRR is higher than dense.
  Fusion gives keyword rank the same weight as dense rank. When the dense
  list already has the relevant chunk first, a different keyword leader
  can push it down.
- Every OR query matches something, also for the 2 unanswerable questions,
  because any shared word counts. Keyword search does not judge
  answerability.
- qwen3-06b MMR k = 8 lost one question (ex-07; recall 1.000 -> 0.917).
  The earlier throwaway measurement with heading prefixes (below) had
  1.000 for this row. Its dense rows match the v2 dense rows exactly; its
  two qwen3-06b MMR rows do not (k = 3: 0.792 / 0.778 there, 0.750 / 0.778
  here). The cause is not confirmed. That measurement used its own prefix
  format, and MMR reacts to small changes in vector distances.

The private dev set (39 questions, 40 labels) still resolves with
splitter v2: 0 label errors over 272 chunks from 32 sources. Its sources
are not approved yet, so it was checked with the files loaded in memory,
without changing `corpus/sources.yaml`, and not run against an index.

### Earlier measurement: heading path as the first line of each chunk

This measurement came before splitter v2 and led to it. Each chunk of
`corpus-example/` was indexed into a temporary table (`<table>_hp`) with
its heading path, joined by ` > `, as the first line of its content. Chunk IDs and the question set
were unchanged. The temporary tables were dropped afterwards. Baseline is
the v1 table ("before" above).

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
