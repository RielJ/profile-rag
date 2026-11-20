"""The embedding models this project can index with.

One `EmbeddingModel` per model. Each one gets its own table, because a
`vector(N)` column holds vectors of one size only, and vectors of two models
cannot be compared even when the sizes match.

A model runs on one of two services, and its class says which:

- `OllamaModel`, on the local Ollama server: `ollama_tag` and `digest`. To
  check a digest after `ollama pull`, compare it with
  `curl -s $OLLAMA_URL/api/tags`.
- `WorkersAIModel`, on Cloudflare Workers AI (workers_ai_embeddings.py):
  `workers_ai_name`, the model ID after `/ai/run/`. Workers AI has no
  digest; the ID names the weights. A Cloudflare Worker can call Workers
  AI but not Ollama, so a model the deployed Worker uses must be one of these.

`EmbeddingModel` is the union of the two. vectorstore/embedders.py picks the
service from the class.
"""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class _Model:
    """What every model has, whichever service runs it."""

    key: str  # the name on the command line: `profile-rag index --model KEY`
    dimensions: int  # values per vector, the N of the table's vector(N)
    table_name: str
    document_prefix: str  # added to every chunk before `embed_documents`
    query_prefix: str  # added to the question before `embed_query`
    # Longer texts are refused before the call (prefixed.py). Set it so the
    # service never cuts a text without an error: Ollama does, at the
    # model's context length.
    max_words: int
    # The same weights in fastembed (fastembed_embeddings.py), for a server
    # without Ollama. None when fastembed has no model whose vectors match.
    fastembed_name: str | None = None


@dataclass(frozen=True, kw_only=True)
class OllamaModel(_Model):
    """A model the local Ollama server runs."""

    # Always with an explicit tag: a bare name means `:latest`.
    ollama_tag: str
    # The full SHA-256 of that tag, as /api/tags lists it: 64 hex characters.
    digest: str


@dataclass(frozen=True, kw_only=True)
class WorkersAIModel(_Model):
    """A model Cloudflare Workers AI runs."""

    workers_ai_name: str  # the model ID, such as "@cf/baai/bge-m3"


type EmbeddingModel = OllamaModel | WorkersAIModel


# Both digests are the SHA-256 of the tag's manifest, read on 2026-10-29 from
# https://registry.ollama.ai/v2/library/<name>/manifests/<tag>. Ollama stores
# that manifest when you pull, and `ollama list` shows the first 12
# characters of its SHA-256 as the ID. Confirm them after `ollama pull`.

# nomic-embed-text v1.5: prefixes from the nomic-ai/nomic-embed-text-v1.5 model
# card. Trained on 2048 tokens. 1000 words stays below that for English text.
NOMIC_V15 = OllamaModel(
    key="nomic-v15",
    ollama_tag="nomic-embed-text:v1.5",
    digest="0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f",
    dimensions=768,
    table_name="chunks_nomic_v15",
    document_prefix="search_document: ",
    query_prefix="search_query: ",
    max_words=1000,
    fastembed_name="nomic-ai/nomic-embed-text-v1.5",
)

# Qwen3-Embedding 0.6B: the query format is `get_detailed_instruct` from the
# Qwen/Qwen3-Embedding-0.6B model card, with no space after "Query:".
# Documents get no instruction. The context is 32k tokens; the same word
# limit keeps every chunk comparable across the two tables.
QWEN3_TASK = "Given a web search query, retrieve relevant passages that answer the query"
QWEN3_06B = OllamaModel(
    key="qwen3-06b",
    ollama_tag="qwen3-embedding:0.6b",
    digest="ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d",
    dimensions=1024,
    table_name="chunks_qwen3_06b",
    document_prefix="",
    query_prefix=f"Instruct: {QWEN3_TASK}\nQuery:",
    max_words=1000,
)

# BGE-M3 on Workers AI. The BAAI/bge-m3 model card: dense retrieval needs no
# instruction on either side. 1024 dimensions, checked with a live call on
# 2026-11-20. Its limit is 8192 tokens, and Workers AI refuses a longer text
# (HTTP 400 "Sequence too long") instead of cutting it, so the word limit is
# the same as the others'. The longest corpus chunk was 533 tokens.
#
# Not here, both checked live on 2026-11-20: @cf/baai/bge-base-en-v1.5 cuts
# text at 512 tokens without an error, and a 221-word chunk of the corpus is
# 519 tokens; @cf/google/embeddinggemma-300m also cuts without an error, and
# Cloudflare deprecated it on 2026-11-20.
BGE_M3 = WorkersAIModel(
    key="bge-m3",
    workers_ai_name="@cf/baai/bge-m3",
    dimensions=1024,
    table_name="chunks_bge_m3",
    document_prefix="",
    query_prefix="",
    max_words=1000,
)

MODELS: dict[str, EmbeddingModel] = {model.key: model for model in (NOMIC_V15, QWEN3_06B, BGE_M3)}
DEFAULT_MODEL = NOMIC_V15.key
