"""The errors every answer-model backend raises.

`rag.failure_code` sorts errors by these two classes: `ModelTimeout`
becomes `pi_timeout` and any other `ModelError` becomes `pi_failed`. The
codes are part of the answer contract (answers.py, `FailedCode`), which the
Worker shares.

The backend is Pi (generation/pi_rpc.py, whose `PiError` subclasses these).
"""


class ModelError(Exception):
    """The answer model did not return a complete reply."""


class ModelTimeout(ModelError):
    """The answer model did not reply in time."""
