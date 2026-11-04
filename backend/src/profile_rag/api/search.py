"""The answer chains the API serves, opened once at startup.

`open_chains(settings)` runs the checks `profile-rag ask` runs, opens the
search store of the configured embedding model, and returns a
`rag.ChainCache` and the engine to close at shutdown. Any failed check
raises `SearchSetupError`, and the server does not start:

- `PROFILE_RAG_API_MODEL` must name a model in vectorstore/models.py.
- Ollama must have that model, with the digest models.py expects.
- The model's chunk table must exist.
- With tracing on, the table must hold exactly the chunks of
  corpus-example/ (as cli.py's `_require_traceable`), because traces may
  only come from the example corpus.

After startup, a database or Ollama outage fails single requests with
`retrieval_failed`; `GET /api/health` reports the database.
"""

from dataclasses import dataclass
from typing import Protocol

import ollama
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGEngine

from profile_rag import tracing
from profile_rag.corpus import documents, registry, splitting
from profile_rag.generation.pi_chat_model import PiChatModel
from profile_rag.generation.pi_rpc import PiConfig
from profile_rag.rag import ChainCache
from profile_rag.retrievers.factory import SearchMode, build_retriever
from profile_rag.settings import Settings
from profile_rag.vectorstore import store
from profile_rag.vectorstore.models import MODELS, EmbeddingModel
from profile_rag.vectorstore.prefixed import PrefixedEmbeddings


class SearchSetupError(RuntimeError):
    """A startup check failed. The message says which and how to fix it."""


class AsyncClosable(Protocol):
    async def close(self) -> None: ...


@dataclass
class OpenChains:
    chains: ChainCache
    engine: AsyncClosable  # the PGEngine of the search store

    async def close(self) -> None:
        await self.engine.close()


def pi_model(settings: Settings) -> BaseChatModel:
    config = PiConfig(
        model=settings.pi_model,
        executable=(settings.pi_bin,),
        timeout=settings.pi_timeout_seconds,
    )
    return PiChatModel(config=config)


def open_chains(settings: Settings) -> OpenChains:
    """Check the setup and open the search store. Blocks; run it in a thread."""
    model = MODELS.get(settings.api_model)
    if model is None:
        raise SearchSetupError(
            f"PROFILE_RAG_API_MODEL={settings.api_model!r} is not one of: {', '.join(MODELS)}."
        )
    database_url = str(settings.database_url)
    _check_ollama(model, settings.ollama_url)
    if not store.table_exists(database_url, model.table_name):
        raise SearchSetupError(
            f"no table {model.table_name}. Run `profile-rag index --model {model.key}` first."
        )
    if tracing.tracing_enabled():
        _check_example_table(database_url, model)

    embeddings = PrefixedEmbeddings(
        OllamaEmbeddings(model=model.ollama_tag, base_url=settings.ollama_url), model
    )
    engine = PGEngine.from_connection_string(url=database_url)
    search_store = store.open_search_store(engine, model, embeddings)

    def make_retriever(mode: SearchMode, k: int) -> Runnable[str, list[Document]]:
        return build_retriever(mode, search_store, k, conninfo=database_url, model=model)

    chains = ChainCache(
        make_retriever,
        pi_model(settings),
        default_mode=settings.api_mode,
        default_k=settings.api_k,
    )
    chains.get()  # the default chain, built now rather than on the first request
    return OpenChains(chains=chains, engine=engine)


def _check_ollama(model: EmbeddingModel, base_url: str) -> None:
    try:
        listed = ollama.Client(host=base_url).list()
    except ConnectionError as error:
        raise SearchSetupError(f"cannot reach Ollama at {base_url}.") from error
    digests = {
        entry.model: (entry.digest or "").removeprefix("sha256:")
        for entry in listed.models
        if entry.model
    }
    installed = digests.get(model.ollama_tag)
    if installed is None:
        raise SearchSetupError(f"Ollama has no {model.ollama_tag}. Run `ollama pull`.")
    if installed != model.digest:
        raise SearchSetupError(
            f"Ollama's {model.ollama_tag} has digest {installed[:12]}, but"
            f" vectorstore/models.py expects {model.digest[:12]}."
        )


def _check_example_table(database_url: str, model: EmbeddingModel) -> None:
    root = tracing.EXAMPLE_CORPUS
    loaded = documents.load_documents(
        registry.load_registry(root / registry.REGISTRY_FILENAME), root
    )
    chunks = splitting.split_documents(
        loaded,
        chunk_words=splitting.DEFAULT_CHUNK_WORDS,
        overlap_words=splitting.DEFAULT_OVERLAP_WORDS,
    )
    if store.chunk_ids(database_url, model) != {c.metadata["chunk_id"] for c in chunks}:
        raise SearchSetupError(
            f"LANGSMITH_TRACING=true, but {model.table_name} does not hold exactly the"
            " example corpus. Run `make index CORPUS_DIR=corpus-example`, or turn tracing off."
        )
