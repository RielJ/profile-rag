"""`AgentState`, the dictionary the agent graph passes from node to node.

Each node returns only the keys it changes; LangGraph merges them into the
state. No key has a reducer, so a returned value replaces the old one.
"""

from typing import TypedDict

from langchain_core.documents import Document

from profile_rag.agent.grader import GradeLabel
from profile_rag.generation.chain import GroundedAnswer


class AgentState(TypedDict):
    # The question retrieval uses: the user's question, or its rewrite.
    question: str
    # The user's question. The answer is always to this one.
    original_question: str
    # The chunks of the latest retrieval, best first.
    docs: list[Document]
    # The grader's verdict on `docs`; None until the grade node runs.
    grade: GradeLabel | None
    # How many times the question was rewritten: 0 or 1.
    rewrites: int
    # How many Pi calls the run made so far: grade, rewrite, and generate
    # each add 1. Retrieval and an answer without evidence add nothing.
    pi_calls: int
    # The final answer; None until the generate node runs.
    result: GroundedAnswer | None


def initial_state(question: str) -> AgentState:
    """The state a run starts from."""
    return AgentState(
        question=question,
        original_question=question,
        docs=[],
        grade=None,
        rewrites=0,
        pi_calls=0,
        result=None,
    )
