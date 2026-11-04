"""The embedding models this project can index with.

One `EmbeddingModel` per model. Each one gets its own table, because a
`vector(N)` column holds vectors of one size only, and vectors of two models
cannot be compared even when the sizes match.

To check a digest after `ollama pull`, compare it with
`curl -s $OLLAMA_URL/api/tags`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModel:
    """Everything the index command needs to know about one model."""

    key: str  # the name on the command line: `profile-rag index --model KEY`
    ollama_tag: str  # always with an explicit tag: a bare name means `:latest`
    digest: str  # the full SHA-256 of `ollama_tag` from /api/tags, 64 hex characters
    dimensions: int  # values per vector, the N of the table's vector(N)
    table_name: str
    document_prefix: str  # added to every chunk before `embed_documents`
    query_prefix: str  # added to the question before `embed_query`
    max_words: int  # longer texts are refused, not sent (Ollama would cut them)


# Both digests are the SHA-256 of the tag's manifest, read on 2026-10-29 from
# https://registry.ollama.ai/v2/library/<name>/manifests/<tag>. Ollama stores
# that manifest when you pull, and `ollama list` shows the first 12
# characters of its SHA-256 as the ID. Confirm them after `ollama pull`.

# nomic-embed-text v1.5: prefixes from the nomic-ai/nomic-embed-text-v1.5 model
# card. Trained on 2048 tokens. 1000 words stays below that for English text.
NOMIC_V15 = EmbeddingModel(
    key="nomic-v15",
    ollama_tag="nomic-embed-text:v1.5",
    digest="0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f",
    dimensions=768,
    table_name="chunks_nomic_v15",
    document_prefix="search_document: ",
    query_prefix="search_query: ",
    max_words=1000,
)

# Qwen3-Embedding 0.6B: the query format is `get_detailed_instruct` from the
# Qwen/Qwen3-Embedding-0.6B model card, with no space after "Query:".
# Documents get no instruction. The context is 32k tokens; the same word
# limit keeps every chunk comparable across the two tables.
QWEN3_TASK = "Given a web search query, retrieve relevant passages that answer the query"
QWEN3_06B = EmbeddingModel(
    key="qwen3-06b",
    ollama_tag="qwen3-embedding:0.6b",
    digest="ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d",
    dimensions=1024,
    table_name="chunks_qwen3_06b",
    document_prefix="",
    query_prefix=f"Instruct: {QWEN3_TASK}\nQuery:",
    max_words=1000,
)

MODELS: dict[str, EmbeddingModel] = {model.key: model for model in (NOMIC_V15, QWEN3_06B)}
DEFAULT_MODEL = NOMIC_V15.key
