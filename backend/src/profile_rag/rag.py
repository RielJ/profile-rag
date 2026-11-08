"""Answer one API question with the grounded answer chain.

`ask(question, chain, request_id=...)` runs the chain from
generation/chain.py and maps its result to the API's `AskResponse`
(api/models.py). It adds no answer logic of its own:

- `Answered` becomes `AnsweredResponse`, with one `Citation` per cited
  chunk, taken from the chunk's metadata. The excerpt starts after the
  chunk's heading line, and `section` gives those headings instead.
- `Unknown` becomes `UnknownResponse`. An empty retrieval is an `Unknown`
  made by the chain without a model call.
- Each error the chain raises becomes a `FailedResponse` with a fixed code
  and message (`FAILURES`). The message never holds the error's text: an
  `OutputParserException` contains the model's whole reply, and a
  `PiProcessError` holds Pi's stderr. The log line names only the
  exception class and the request ID.

`model` in a response is the `model_name` the chat model put in its reply's
`response_metadata` (PiChatModel: Pi's `provider/model`). `ModelName`, a
callback, reads it, so the chain does not change. It is None when no model
ran or the model reported none.

`ChainCache` builds one chain per (mode, k) and keeps it. The API makes one
at startup (api/search.py); tests make one from a fake retriever and a fake
chat model.

Tests: backend/tests/test_api_contract.py.
"""

import logging
import re
import threading
from collections.abc import Callable
from typing import Any, override

import httpx
import ollama
import psycopg
import sqlalchemy.exc
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import Runnable

from profile_rag.api.models import (
    AnsweredResponse,
    AskResponse,
    Citation,
    FailedCode,
    FailedResponse,
    UnknownResponse,
)
from profile_rag.corpus.splitting import HEADING_SEPARATOR, heading_line
from profile_rag.generation.answer import InventedCitation, Unknown
from profile_rag.generation.chain import GroundedAnswer, build_chain
from profile_rag.generation.pi_rpc import PiError, PiTimeout
from profile_rag.retrievers.factory import SearchMode
from profile_rag.vectorstore.prefixed import EmbeddingError

logger = logging.getLogger(__name__)

AnswerChain = Runnable[dict[str, str], GroundedAnswer]
MakeRetriever = Callable[[SearchMode, int], Runnable[str, list[Document]]]

EXCERPT_CHARS = 280

# What a visitor sees for each failure. Fixed text: never the error's own.
FAILURES: dict[FailedCode, str] = {
    "retrieval_failed": "The evidence could not be searched right now. Try again later.",
    "pi_timeout": "The answer model took too long to reply. Try again later.",
    "pi_failed": "The answer model could not be reached. Try again later.",
    "bad_reply": "The answer model's reply was not a valid answer. Try again.",
    "invented_citation": "The answer model cited evidence it was not given, so the answer"
    " was discarded. Try again.",
    "internal_error": "Something went wrong on the server.",
}

# Errors of the search step: the embedding call to Ollama and the queries.
RETRIEVAL_ERRORS: tuple[type[Exception], ...] = (
    EmbeddingError,
    ConnectionError,
    httpx.HTTPError,
    ollama.ResponseError,
    psycopg.Error,
    sqlalchemy.exc.SQLAlchemyError,
)


class ModelName(BaseCallbackHandler):
    """Records the `model_name` of the last chat model reply in a run."""

    def __init__(self) -> None:
        self.value: str | None = None

    @override
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                if isinstance(generation, ChatGeneration):
                    name = generation.message.response_metadata.get("model_name")
                    if isinstance(name, str) and name:
                        self.value = name


def failure_code(error: Exception) -> FailedCode:
    """The `FailedCode` of an error the chain raised."""
    if isinstance(error, PiTimeout):
        return "pi_timeout"
    if isinstance(error, PiError):
        return "pi_failed"
    if isinstance(error, OutputParserException):
        return "bad_reply"
    if isinstance(error, InventedCitation):
        return "invented_citation"
    if isinstance(error, RETRIEVAL_ERRORS):
        return "retrieval_failed"
    return "internal_error"


def ask(question: str, chain: AnswerChain, *, request_id: str) -> AskResponse:
    """Run `chain` on `question` and return its result as an `AskResponse`.

    Never raises for an error of the chain: it returns a `FailedResponse`
    and logs one line with the exception class and `request_id`.
    """
    model = ModelName()
    try:
        result = chain.invoke({"question": question}, {"callbacks": [model]})
        return to_response(result, model.value)
    except Exception as error:
        code = failure_code(error)
        # The class name only: str(error) can hold the model's reply or Pi's stderr.
        logger.warning(
            "ask failed: request_id=%s error=%s code=%s",
            request_id,
            type(error).__name__,
            code,
        )
        return AskResponse(
            FailedResponse(
                status="failed", error=code, message=FAILURES[code], request_id=request_id
            )
        )


def to_response(result: GroundedAnswer, model: str | None) -> AskResponse:
    answer = result.answer.root
    if isinstance(answer, Unknown):
        return AskResponse(UnknownResponse(status="unknown", reason=answer.reason, model=model))
    citations = [to_citation(doc) for doc in result.cited_docs()]
    return AskResponse(
        AnsweredResponse(status="answered", answer=answer.answer, citations=citations, model=model)
    )


def to_citation(doc: Document) -> Citation:
    metadata = doc.metadata
    path: list[str] = metadata.get("heading_path") or []
    return Citation(
        chunk_id=metadata["chunk_id"],
        source_id=metadata["source_id"],
        source_title=metadata["title"],
        section=section(path),
        url=public_url(metadata.get("url")),
        attribution=metadata["attribution"],
        excerpt=excerpt(body(doc.page_content, path)),
    )


def section(heading_path: list[str]) -> str | None:
    """The headings below the top one, joined by " > ", or None if there are none.

    The top heading is usually the document's title, which the citation
    already has as `source_title`.
    """
    return HEADING_SEPARATOR.join(heading_path[1:]) or None


def body(text: str, heading_path: list[str]) -> str:
    """`text` without the heading line that splitting.chunk_text put first.

    Text that does not start with that line is returned whole, so a chunk
    from an older splitter still gets an excerpt.
    """
    line = heading_line(heading_path)
    return text.removeprefix(f"{line}\n\n") if line else text


def public_url(value: object) -> str | None:
    """`value` if it is an http(s) URL, else None.

    The registry's `url` is the public link of a source, or None. Anything
    else (a local path, an empty string) is not a link a visitor can open.
    """
    if isinstance(value, str) and value.startswith(("https://", "http://")):
        return value
    return None


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """`text` as plain lines, cut at a word boundary to at most `limit` characters.

    See `lines` for how Markdown becomes lines. A cut text ends with an
    ellipsis, which counts toward `limit`.
    """
    flat = "\n".join(lines(text))
    if len(flat) <= limit:
        return flat
    cut = flat[: limit - 1]
    space = max(cut.rfind(" "), cut.rfind("\n"))
    if space > 0:
        cut = cut[:space]
    return cut.rstrip() + "…"


LIST_ITEM = re.compile(r"([-*+]|\d+[.)])\s")
FENCE = re.compile(r"(```|~~~)")


def lines(text: str) -> list[str]:
    """The paragraphs, list items, and code lines of Markdown `text`, one per line.

    Spaces collapse to one. The lines of a wrapped paragraph join with a
    space. Code fence markers are dropped, and each line inside a fence
    stays its own line.
    """
    out: list[str] = []
    joinable = False  # whether the next line continues the last one
    in_fence = False
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if FENCE.match(line):
            in_fence = not in_fence
            joinable = False
        elif not line:
            joinable = False
        elif in_fence:
            out.append(line)
        elif joinable and not LIST_ITEM.match(line):
            out[-1] += " " + line
        else:
            out.append(line)
            joinable = True
    return out


class ChainCache:
    """One answer chain per (mode, k), built on first use and kept.

    `make_retriever(mode, k)` returns the retriever for that mode and `k`;
    `model` is the chat model every chain uses. Chains hold no request
    state, so concurrent requests share them. Safe to call from several
    threads.
    """

    def __init__(
        self,
        make_retriever: MakeRetriever,
        model: BaseChatModel,
        *,
        default_mode: SearchMode,
        default_k: int,
    ) -> None:
        self._make_retriever = make_retriever
        self._model = model
        self.default_mode: SearchMode = default_mode
        self.default_k = default_k
        self._chains: dict[tuple[SearchMode, int], AnswerChain] = {}
        self._lock = threading.Lock()

    def get(self, mode: SearchMode | None = None, k: int | None = None) -> AnswerChain:
        key = (mode or self.default_mode, k or self.default_k)
        with self._lock:
            chain = self._chains.get(key)
            if chain is None:
                chain = build_chain(self._make_retriever(*key), self._model)
                self._chains[key] = chain
            return chain
