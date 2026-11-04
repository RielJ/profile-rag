# Decision: the answer chain, not the agent graph

The default answer path stays the chain (`profile-rag ask`,
generation/chain.py). The agent graph (`profile-rag agent`, agent/graph.py)
stays in the code, but nothing uses it by default.

## Evidence

From [agent-vs-chain-example-nomic-v15.md](agent-vs-chain-example-nomic-v15.md):
the example set (14 questions, 12 answerable, 2 unanswerable), embedding
model nomic-v15, hybrid retrieval, k 3.

| measure | chain | agent |
| --- | --- | --- |
| answerable answered (of 12) | 10 | 10 |
| unanswerable answered unknown (of 2) | 2 | 2 |
| answers citing a labeled chunk | 9 | 9 |
| citation precision vs labels | 0.643 | 0.600 |
| Pi calls, total | 14 | 36 |
| questions rewritten | 0 | 4 |

- Both pipelines answered the same questions and returned `unknown` for the
  same questions. Neither rejected a reply.
- The agent rewrote 4 questions: the 2 unanswerable ones and the 2
  attribution traps (ex-10, ex-11). A rewrite did not change the outcome
  of any question: each of the 4 was graded not relevant twice and ended
  `unknown`, as with the chain.
- The agent made 2.57 times as many Pi calls (36 against 14) and took
  longer for each question.
- The precision difference comes from one question (ex-07), where the
  agent's answer cited one more chunk. At this sample size, that is noise.

## Reason

On this set the grader and the rewrite add Pi calls and latency, but they
change no answer. The rewrite can only help when the first retrieval
misses evidence that a second query finds. Here, every question it
rewrote had nothing more to find (unanswerable), or had evidence the
answer step treats as not enough (the attribution traps).

## Limits, and when to revisit

The example set is small (14 questions) and its corpus is fictional and
tiny (15 chunks), so k 3 already holds most of the evidence. Revisit this
decision on the dev set (`profile-rag compare-agent --set dev`) once the
real corpus has approved sources. The dev set has more paraphrase and
multi-source questions, where a rewrite has a chance to retrieve missing
evidence.
