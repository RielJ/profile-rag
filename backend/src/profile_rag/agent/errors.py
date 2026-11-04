"""Errors the agent graph raises, besides the ones its steps pass through.

No message contains the model's reply: a reply can quote corpus text, and
these messages reach the terminal and, with tracing on, a run's error field.
"""


class AgentError(Exception):
    """Base class of the agent's own errors."""


class GradeParseError(AgentError):
    """The grader's reply is not `{"grade": "relevant"}` or `{"grade": "not_relevant"}`."""

    def __init__(self) -> None:
        super().__init__('the grader\'s reply is not JSON of the shape {"grade": ...}.')


class RewriteError(AgentError):
    """The rewrite reply is empty or too long to be one question."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"the rewritten question was rejected: {reason}.")
        self.reason = reason


class AgentLoopError(AgentError):
    """The graph took more steps than its recursion limit allows.

    The longest legal path takes six steps, so this means a routing bug, not
    a hard question. Raised in place of LangGraph's `GraphRecursionError`.
    """

    def __init__(self, limit: int) -> None:
        super().__init__(f"the agent did not finish within its recursion limit of {limit} steps.")
        self.limit = limit
