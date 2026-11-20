"""The result of asking one question: answered, unknown, or failed.

`rag.ask` returns an `AskResponse`, one of three variants chosen by `status`.
The CLI prints it, and the Worker (worker/) returns the same shape as JSON;
`rag.to_response` is the reference the Worker is checked against. Every
model forbids unknown fields.

    {"status": "answered", "answer": ..., "citations": [Citation, ...], "model": ...}
    {"status": "unknown", "reason": ..., "model": ...}
    {"status": "failed", "error": FailedCode, "message": ..., "request_id": ...}
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel

from profile_rag.corpus.registry import Attribution

FORBID = ConfigDict(extra="forbid")


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
    "rate_limited",
]


class FailedResponse(BaseModel):
    """The question could not be answered. `message` is safe to show to a visitor.

    It never contains the model's reply or Pi's output. `request_id` matches
    the log line for the failure.
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
