"""Request and response models of the public API.

These classes are the API contract. FastAPI turns them into the OpenAPI
schema (`/openapi.json`), and `make types` turns that schema into the
frontend's TypeScript types (frontend/src/api/schema.d.ts). Every model
forbids unknown fields.

`POST /api/ask` takes an `AskRequest` and returns an `AskResponse`: one of
three variants, chosen by `status`.

    {"status": "answered", "answer": ..., "citations": [Citation, ...], "model": ...}
    {"status": "unknown", "reason": ..., "model": ...}
    {"status": "failed", "error": FailedCode, "message": ..., "request_id": ...}

`POST /api/ask/stream` takes the same `AskRequest` and returns
newline-delimited JSON: one `AskStreamEvent` per line. Zero or more `draft`
events carry the answer text as the model writes it, unchecked; the last
line is one `result` event with the `AskResponse` that `POST /api/ask` would
have returned.

    {"type": "draft", "text": "He cut"}
    {"type": "draft", "text": " build time"}
    {"type": "result", "response": AskResponse}

`GET /api/sources/{id}` returns a `SourceResponse`: the metadata of one
approved source, never its text.

Tests: backend/tests/test_api_contract.py.
"""

from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, RootModel, StringConstraints

from profile_rag.corpus.registry import Attribution, Kind, Visibility
from profile_rag.retrievers.factory import SearchMode

QUESTION_MAX_LENGTH = 1000
K_MAX = 20

# At least one character that is not whitespace.
NonBlank = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]

FORBID = ConfigDict(extra="forbid")


class AskRequest(BaseModel):
    model_config = FORBID

    question: Annotated[NonBlank, StringConstraints(max_length=QUESTION_MAX_LENGTH)] = Field(
        description="The question, in plain text."
    )
    mode: SearchMode | None = Field(
        default=None, description="Which retriever to run. Default: the server's setting."
    )
    k: int | None = Field(
        default=None,
        ge=1,
        le=K_MAX,
        description="How many chunks to give the model. Default: the server's setting.",
    )


class Citation(BaseModel):
    """One chunk the answer relies on, with its source."""

    model_config = FORBID

    chunk_id: str
    source_id: str
    source_title: str
    section: str | None = Field(
        description="The headings above the chunk, without the source's top heading, "
        "joined by ' > '. Null when the chunk has no heading below the top one."
    )
    url: str | None = Field(
        description="The source's public http(s) link, or null when it has none."
    )
    attribution: Attribution
    excerpt: str = Field(
        description="The start of the chunk's text after its headings, shortened. "
        "Each paragraph and list item is on its own line."
    )


class AnsweredResponse(BaseModel):
    model_config = FORBID

    status: Literal["answered"]
    answer: str
    citations: list[Citation] = Field(min_length=1)
    model: str | None = Field(
        description="The model that wrote the answer, as provider/model, or null if unreported."
    )


class UnknownResponse(BaseModel):
    model_config = FORBID

    status: Literal["unknown"]
    reason: str
    model: str | None = Field(
        description="The model that gave the reason, or null when no model was called"
        " (nothing was retrieved) or it reported none."
    )


FailedCode = Literal[
    "retrieval_failed",
    "pi_timeout",
    "pi_failed",
    "bad_reply",
    "invented_citation",
    "internal_error",
]


class FailedResponse(BaseModel):
    """The question could not be answered. `message` is safe to show to a visitor.

    It never contains the model's reply or Pi's output. `request_id` matches
    the server log line for the failure.
    """

    model_config = FORBID

    status: Literal["failed"]
    error: FailedCode
    message: str
    request_id: str


class AskResponse(
    RootModel[
        Annotated[
            AnsweredResponse | UnknownResponse | FailedResponse, Field(discriminator="status")
        ]
    ]
):
    """An answered, unknown, or failed response, chosen by the `status` field."""


class DraftEvent(BaseModel):
    """The next piece of the answer (or reason) text, sent while the model writes.

    Unchecked: show it as a draft and replace it with the `result` event.
    """

    model_config = FORBID

    type: Literal["draft"]
    text: str = Field(description="Text to append to the draft so far.")


class ResultEvent(BaseModel):
    """The checked response. Always the last line of the stream."""

    model_config = FORBID

    type: Literal["result"]
    response: AskResponse


class AskStreamEvent(RootModel[Annotated[DraftEvent | ResultEvent, Field(discriminator="type")]]):
    """One line of `POST /api/ask/stream`, chosen by the `type` field."""


class SourceResponse(BaseModel):
    """The metadata of one approved source: what a citation links to.

    Only the columns a visitor may see. `revision`, `content_sha256`, and
    `approved_by` stay internal, and the source text is never returned.
    """

    model_config = FORBID

    id: str
    kind: Kind
    title: str
    url: str | None = Field(description="The public http(s) link, or null when it has none.")
    attribution: Attribution
    visibility: Visibility
    approved_at: AwareDatetime
