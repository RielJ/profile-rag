"""The errors every answer-model backend raises.

`rag.failure_code` sorts errors by these two classes, so the API and the CLI
report a failed model call the same way whichever backend made it:
`ModelTimeout` becomes `pi_timeout` and any other `ModelError` becomes
`pi_failed`. The wire codes keep their `pi_` names because they are part of
the API contract (api/models.py).

Backends: Pi (generation/pi_rpc.py, whose `PiError` subclasses these) and an
OpenAI-compatible HTTP API (generation/openai_chat_model.py).
"""


class ModelError(Exception):
    """The answer model did not return a complete reply."""


class ModelTimeout(ModelError):
    """The answer model did not reply in time."""
