"""Write the Python hybrid retriever's results for the dev set, for the parity check.

Each dev question is embedded once with bge-m3 (Workers AI). The Python
`HybridRetriever` then runs with that fixed vector against the local
database, and the script writes, per question, the question, the vector, and
the chunk IDs Python returned. worker/scripts/retrieval-parity.ts runs the
Worker's search with the same vectors and compares.

Using one vector for both sides keeps the comparison about the search, not
about small differences between two embedding calls.

The output holds questions, vectors, and chunk IDs, no corpus text. Write it
under eval/results/ (gitignored). Local database only: the script refuses a
DATABASE_URL whose host is not 127.0.0.1 or localhost.

    PROFILE_RAG_CF_ACCOUNT_ID=... PROFILE_RAG_CF_API_TOKEN=... \
      uv --directory backend run python scripts/retrieval_fixture.py OUT
"""

import argparse
import json
from pathlib import Path
from typing import override
from urllib.parse import urlsplit

from langchain_core.embeddings import Embeddings

from profile_rag.retrievers.factory import build_retriever
from profile_rag.settings import get_settings
from profile_rag.vectorstore import embedders, store
from profile_rag.vectorstore.models import BGE_M3

DEV_SET = Path(__file__).resolve().parents[2] / "eval" / "dev.jsonl"
LOCAL_HOSTS = {"127.0.0.1", "localhost"}


class FixedQuery(Embeddings):
    """Returns one precomputed vector for any query."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    @override
    def embed_query(self, text: str) -> list[float]:
        return self.vector

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vector for _ in texts]


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("-k", type=int, default=8)
    args = parser.parse_args()

    settings = get_settings()
    database_url = str(settings.database_url)
    host = urlsplit(database_url).hostname
    if host not in LOCAL_HOSTS:
        raise SystemExit(f"refusing to run against {host!r}: local database only")

    try:
        lines = DEV_SET.read_text(encoding="utf-8").splitlines()
        questions = [json.loads(line) for line in lines if line]
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read the dev set {DEV_SET}: {error}") from error
    vectors = embedders.for_model(BGE_M3, settings).embed_documents(
        [BGE_M3.query_prefix + q["question"] for q in questions]
    )
    cases = []
    for question, vector in zip(questions, vectors, strict=True):
        with store.searching(database_url, BGE_M3, FixedQuery(vector)) as search_store:
            retriever = build_retriever(
                "hybrid", search_store, args.k, conninfo=database_url, model=BGE_M3
            )
            found = retriever.invoke(question["question"])
        cases.append(
            {
                "id": question["id"],
                "question": question["question"],
                "vector": vector,
                "chunk_ids": [doc.metadata["chunk_id"] for doc in found],
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"k": args.k, "cases": cases}) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
