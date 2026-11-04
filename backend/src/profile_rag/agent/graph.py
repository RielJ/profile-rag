"""The agent graph: retrieve, grade, rewrite at most once, generate.

`build_graph(retriever, model)` compiles a LangGraph `StateGraph` over
`AgentState` (agent/state.py). `run_agent(graph, question)` runs it and
returns an `AgentRun`: the `GroundedAnswer`, the Pi call count, the number
of rewrites, and the node path.

    START -> retrieve -> (no docs) ---------------------------> generate -> END
                      -> (docs) grade -> (relevant) ----------> generate
                                      -> (not_relevant, no rewrite yet) rewrite -> retrieve
                                      -> (not_relevant, rewritten once) -> generate

- retrieve: runs the retriever on `question` (no Pi call).
- grade: one Pi call that grades the whole retrieved set (agent/grader.py).
- rewrite: one Pi call that rephrases the original question
  (agent/rewrite.py), then retrieval runs again on the new question.
- generate: the answer chain's prompt, parser, and citation check
  (`generation.chain.build_answerer`), always on the original question. With
  no docs it returns `unknown` without a Pi call, as the chain does.

So a run makes 0 Pi calls (nothing retrieved), 2 (grade, generate), or 4
(grade, rewrite, grade, generate); 2 also when the rewrite retrieves nothing.

The longest path has six steps: retrieve, grade, rewrite, retrieve, grade,
generate. On langgraph 1.2.19 it raises `GraphRecursionError` with
`recursion_limit` 6 and finishes with 7 (tests/test_graph.py checks both).
`RECURSION_LIMIT` is 8, so a routing bug stops with `AgentLoopError`
instead of calling Pi in a loop.

Errors: `AgentLoopError` replaces `GraphRecursionError`. Everything else
passes through unchanged: `PiError` from any Pi call, `GradeParseError`,
`RewriteError`, the answer step's `OutputParserException` (its text holds
the reply) and `InventedCitation`, and the retriever's errors.

Tests: backend/tests/test_graph.py.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from profile_rag.agent.errors import AgentLoopError
from profile_rag.agent.grader import build_grader
from profile_rag.agent.rewrite import build_rewriter
from profile_rag.agent.state import AgentState, initial_state
from profile_rag.generation.chain import GroundedAnswer, build_answerer, no_evidence_answer

RECURSION_LIMIT = 8
MAX_REWRITES = 1

# Pyrefly does not match a TypedDict class against LangGraph's TypedDictLike
# protocol (keys as class attributes), so the specialization is suppressed.
AgentGraph = CompiledStateGraph[AgentState, None, AgentState, AgentState]  # pyrefly: ignore[bad-specialization]


@dataclass(frozen=True)
class AgentRun:
    result: GroundedAnswer
    pi_calls: int
    rewrites: int
    # The nodes in the order they ran.
    path: tuple[str, ...]


def build_graph(retriever: Runnable[str, list[Document]], model: BaseChatModel) -> AgentGraph:
    """Compile the graph. See the module docstring."""
    grader = build_grader(model)
    rewriter = build_rewriter(model)
    answerer = build_answerer(model)

    def retrieve(state: AgentState) -> dict[str, object]:
        return {"docs": retriever.invoke(state["question"])}

    def grade(state: AgentState) -> dict[str, object]:
        label = grader.invoke({"question": state["original_question"], "docs": state["docs"]})
        return {"grade": label, "pi_calls": state["pi_calls"] + 1}

    def rewrite(state: AgentState) -> dict[str, object]:
        question = rewriter.invoke({"question": state["original_question"], "docs": state["docs"]})
        return {
            "question": question,
            "rewrites": state["rewrites"] + 1,
            "pi_calls": state["pi_calls"] + 1,
        }

    def generate(state: AgentState) -> dict[str, object]:
        if not state["docs"]:
            return {"result": no_evidence_answer()}
        result = answerer.invoke({"question": state["original_question"], "docs": state["docs"]})
        return {"result": result, "pi_calls": state["pi_calls"] + 1}

    graph = StateGraph(AgentState)  # pyrefly: ignore[bad-specialization] (see AgentGraph)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve", route_after_retrieve, {"grade": "grade", "generate": "generate"}
    )
    graph.add_conditional_edges(
        "grade", route_after_grade, {"rewrite": "rewrite", "generate": "generate"}
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    return graph.compile(name="agent")


def route_after_retrieve(state: AgentState) -> Literal["grade", "generate"]:
    """Grade the docs; with none, answer `unknown` without a Pi call."""
    return "grade" if state["docs"] else "generate"


def route_after_grade(state: AgentState) -> Literal["rewrite", "generate"]:
    """Rewrite only when the evidence is not relevant and no rewrite happened yet."""
    if state["grade"] == "not_relevant" and state["rewrites"] < MAX_REWRITES:
        return "rewrite"
    return "generate"


# Called with the node name and the keys the node changed.
NodeCallback = Callable[[str, dict[str, Any]], None]


def run_agent(
    graph: AgentGraph,
    question: str,
    *,
    on_node: NodeCallback | None = None,
    recursion_limit: int = RECURSION_LIMIT,
    config: RunnableConfig | None = None,
) -> AgentRun:
    """Run the graph on `question` and return the answer, Pi calls, rewrites, and path.

    `on_node` is called after each node, in order (stream_mode="updates").
    Raises `AgentLoopError` if the run exceeds `recursion_limit` steps.
    """
    run_config: RunnableConfig = {**(config or {}), "recursion_limit": recursion_limit}
    path: list[str] = []
    # No state key has a reducer, so merging the updates in order rebuilds
    # the final state.
    final: dict[str, Any] = dict(initial_state(question))
    try:
        for chunk in graph.stream(initial_state(question), run_config, stream_mode="updates"):
            for node, update in chunk.items():
                changed = dict(update or {})
                path.append(node)
                final.update(changed)
                if on_node is not None:
                    on_node(node, changed)
    except GraphRecursionError:
        raise AgentLoopError(recursion_limit) from None
    result, pi_calls, rewrites = final["result"], final["pi_calls"], final["rewrites"]
    if not isinstance(result, GroundedAnswer):
        raise TypeError("the agent graph ended without a result")
    if not isinstance(pi_calls, int) or not isinstance(rewrites, int):
        raise TypeError("the agent graph lost its counters")
    return AgentRun(result=result, pi_calls=pi_calls, rewrites=rewrites, path=tuple(path))
