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

Tests: backend/tests/test_answer.py.
"""

from collections.abc import Sequence
from typing import Annotated, Literal

from langchain_core.documents import Document
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
