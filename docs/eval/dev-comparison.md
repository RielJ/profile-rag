# Retrieval comparison: dev set

Rebuilt by `profile-rag eval` from the per-run reports in this folder.
Recall@k and MRR are means over answerable questions (n); unanswerable
questions are counted separately. A miss is a question with recall@k below 1.

| model | mode | k | recall@k | MRR | n | unanswerable | misses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bge-m3 | dense | 8 | 0.838 | 0.577 | 34 | 5 | dev-01, dev-10, dev-12, dev-21, dev-23, dev-26, dev-29 |
| bge-m3 | hybrid | 8 | 0.779 | 0.603 | 34 | 5 | dev-01, dev-10, dev-11, dev-12, dev-17, dev-19, dev-21, dev-23, dev-26 |
| nomic-v15 | dense | 8 | 0.750 | 0.499 | 34 | 5 | dev-01, dev-10, dev-11, dev-12, dev-13, dev-17, dev-21, dev-26, dev-29 |
| nomic-v15 | hybrid | 8 | 0.750 | 0.570 | 34 | 5 | dev-01, dev-10, dev-11, dev-12, dev-13, dev-17, dev-21, dev-23, dev-26, dev-29 |
| nomic-v15 | mmr | 8 | 0.632 | 0.474 | 34 | 5 | dev-06, dev-10, dev-11, dev-12, dev-13, dev-17, dev-20, dev-21, dev-23, dev-26, dev-29, dev-31, dev-38 |

<!-- Notes below this line are kept when the table is rebuilt. -->
