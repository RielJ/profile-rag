"""The grounded answer chain: retrieve, prompt, generate, parse, check.

`build_chain(retriever, model)` returns a Runnable. Its input is
`{"question": str}` and its output is a `GroundedAnswer`: the checked
`Answer` and the Documents the model was shown, so a caller can display the
evidence behind each citation.

Steps, each adding a key to the input dict:

    question -> docs (retriever) -> route
        no docs: GroundedAnswer(Unknown, []), and the model is never called
        docs:    context (format_evidence) -> answer (prompt | model | parser)
                 -> check_citations -> GroundedAnswer

Errors pass through unchanged:

- `pi_rpc.PiError` subclasses, from the model.
- `OutputParserException`, when the reply is not a valid `Answer`. Its
  text contains the whole model reply, so do not log it outside
  eval/results/.
- `answer.InventedCitation`, when the reply cites a chunk that was not
  retrieved.

Tests: backend/tests/test_chain.py.
"""

from dataclasses import dataclass
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from profile_rag.generation.answer import Answer, Unknown, check_citations
from profile_rag.generation.prompt import build_prompt, format_evidence

NO_EVIDENCE_REASON = "No indexed evidence matched the question."


@dataclass(frozen=True)
class GroundedAnswer:
    """A checked answer and the retrieved chunks, in retrieval order.

    `docs` is everything the model saw, not only the cited chunks. It is
    empty when retrieval found nothing.
    """

    answer: Answer
    docs: list[Document]

    def cited_docs(self) -> list[Document]:
        """The chunks the answer cites, in citation order; empty for `unknown`."""
        result = self.answer.root
        if isinstance(result, Unknown):
            return []
        by_id = {doc.metadata.get("chunk_id"): doc for doc in self.docs}
        return [by_id[cid] for cid in dict.fromkeys(result.citations) if cid in by_id]


def build_answerer(model: BaseChatModel) -> Runnable[dict[str, object], GroundedAnswer]:
    """One model call from `{"question": str, "docs": [Document, ...]}` to a `GroundedAnswer`.

    The prompt, parser, and citation check of the chain, without retrieval.
    `docs` must not be empty: use `no_evidence_answer()` for that case.
    The agent graph (agent/graph.py) calls this in its `generate` node.
    """
    parser = PydanticOutputParser(pydantic_object=Answer)
    return (
        RunnablePassthrough.assign(context=lambda inputs: format_evidence(inputs["docs"]))
        | RunnablePassthrough.assign(answer=build_prompt(parser) | model | parser)
        | RunnableLambda(_checked)
    )


def no_evidence_answer() -> GroundedAnswer:
    """The `unknown` answer for an empty retrieval, made without a model call."""
    return GroundedAnswer(
        answer=Answer(Unknown(status="unknown", reason=NO_EVIDENCE_REASON)), docs=[]
    )


def build_chain(
    retriever: Runnable[str, list[Document]], model: BaseChatModel
) -> Runnable[dict[str, str], GroundedAnswer]:
    """The chain from `{"question": ...}` to a `GroundedAnswer`. See the module docstring."""
    generate = build_answerer(model)
    no_evidence = RunnableLambda(_no_evidence)

    def route(inputs: dict[str, object]) -> Runnable[dict[str, object], GroundedAnswer]:
        return generate if inputs["docs"] else no_evidence

    retrieve = RunnablePassthrough.assign(docs=itemgetter("question") | retriever)
    chain: Runnable[dict[str, str], GroundedAnswer] = retrieve | RunnableLambda(route)
    return chain.with_config(run_name="grounded_answer")


def _no_evidence(_inputs: dict[str, object]) -> GroundedAnswer:
    return no_evidence_answer()


def _checked(inputs: dict[str, object]) -> GroundedAnswer:
    answer = inputs["answer"]
    docs = inputs["docs"]
    if not isinstance(answer, Answer) or not isinstance(docs, list):
        raise TypeError("the chain lost its answer or docs")
    documents = [doc for doc in docs if isinstance(doc, Document)]
    return GroundedAnswer(answer=check_citations(answer, documents), docs=documents)
