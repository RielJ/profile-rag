"""The answer the model must return, and the check of its citations.

The model's reply is untrusted input. `PydanticOutputParser(pydantic_object=
Answer)` parses it and validates its shape: exactly one of

    {"status": "answered", "answer": "...", "citations": ["chunk_id", ...]}
    {"status": "unknown", "reason": "..."}

with no other keys and no blank text. `check_citations` then checks its
meaning: every cited chunk_id must be one of the chunks the model was shown.

`Answer` is a `RootModel` around the union because the parser needs one model
class: `PydanticOutputParser(pydantic_object=Answered | Unknown)` is accepted
but raises TypeError on the first parse.

`draft_text` reads the answer or reason out of a reply that is still being
written, so the API can stream it. A draft is shown, never trusted: only the
complete reply goes through the parser and the citation check.

Tests: backend/tests/test_answer.py.
"""

from collections.abc import Sequence
from typing import Annotated, Literal

from langchain_core.documents import Document
from langchain_core.utils.json import parse_partial_json
from pydantic import BaseModel, ConfigDict, Field, RootModel, StringConstraints

# At least one character that is not whitespace.
NonBlank = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class Answered(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["answered"]
    answer: NonBlank = Field(description="The answer, using only facts from the evidence.")
    citations: list[NonBlank] = Field(
        min_length=1,
        description="The chunk_id of every evidence block the answer relies on.",
    )


class Unknown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["unknown"]
    reason: NonBlank = Field(description="Why the evidence does not answer the question.")


class Answer(RootModel[Annotated[Answered | Unknown, Field(discriminator="status")]]):
    """An `Answered` or an `Unknown`, chosen by the `status` field."""


class InventedCitation(ValueError):
    """The answer cites chunk IDs that are not among the retrieved chunks."""

    def __init__(self, invented: Sequence[str]) -> None:
        super().__init__(f"the answer cites chunks that were not retrieved: {', '.join(invented)}")
        self.invented = list(invented)


def check_citations(answer: Answer, docs: Sequence[Document]) -> Answer:
    """Return `answer` unchanged if every citation is a `chunk_id` of `docs`.

    Raises InventedCitation, listing each unknown ID once, otherwise. An
    `Unknown` answer has no citations and always passes.
    """
    result = answer.root
    if isinstance(result, Unknown):
        return answer
    retrieved = {doc.metadata.get("chunk_id") for doc in docs}
    invented = [cid for cid in dict.fromkeys(result.citations) if cid not in retrieved]
    if invented:
        raise InventedCitation(invented)
    return answer


def draft_text(partial_reply: str) -> str:
    """The `answer` or `reason` text in a reply that may be cut off, else "".

    Tolerates a reply cut anywhere, including inside a string or an escape
    sequence, and a leading ```json fence. Text the model has not finished
    escaping may be missing for a moment, so a draft can briefly shrink.
    """
    text = partial_reply.lstrip()
    if text.startswith("```"):
        text = text.partition("\n")[2]
    try:
        reply = parse_partial_json(text)
    except ValueError:
        return ""
    if not isinstance(reply, dict):
        return ""
    for key in ("answer", "reason"):
        value = reply.get(key)
        if isinstance(value, str):
            return value
    return ""
