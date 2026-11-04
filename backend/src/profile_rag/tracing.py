"""LangSmith tracing: off unless asked for, and never with corpus text.

`configure_tracing(settings, corpus_dir)` runs once at startup (the CLI's
root callback, the API's `create_app`), before any LangChain object runs.

Off: unless `LANGSMITH_TRACING=true` (environment or .env), tracing is
off. It sets langsmith's global switch to off, so the other names the
library also reads (`LANGSMITH_TRACING_V2`, `LANGCHAIN_TRACING_V2`) cannot
turn it on behind the app's back.

On: it refuses to start, with `TracingRefused`, unless

- `corpus_dir` is this repository's `corpus-example/` (the fictional
  corpus; the CLI passes the `CORPUS_DIR` environment variable, or
  `corpus/` if it is unset), and
- `LANGSMITH_API_KEY` is set.

Then it sets `LANGSMITH_HIDE_INPUTS=true` and `LANGSMITH_HIDE_OUTPUTS=true`
in the environment, and installs a client (`build_client`) that hides
inputs, outputs, and metadata, and scrubs every run's error field.

Why the error field needs its own scrubbing: in langsmith 0.14.15,
`Client._hide_run_error` (langsmith/client.py:2864, called from
`_run_transform` at client.py:2398 and `update_run` at client.py:3762)
applies only the `anonymizer`. `hide_outputs` and
`LANGSMITH_HIDE_OUTPUTS` do not touch it. LangChain fills the field with
`repr(exception)` plus the traceback (langchain_core/tracers/core.py,
`_get_stacktrace`, line 141), and an `OutputParserException`'s text holds the model's
whole reply. `scrub_error` keeps only the exception's class name.

The commands that read the database also check, when tracing is on, that
the data they read comes from `corpus-example/` (cli.py).

Tests: backend/tests/test_tracing.py.
"""

import os
import re
from pathlib import Path
from typing import Any

import langsmith
import requests
from langsmith import utils as ls_utils

from profile_rag.settings import REPO_ROOT, TracingSettings

EXAMPLE_CORPUS = REPO_ROOT / "corpus-example"
DEFAULT_CORPUS = REPO_ROOT / "corpus"

# Every environment variable langsmith reads as a tracing switch.
TRACING_SWITCHES = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
)

REDACTED = "[redacted]"
_EXCEPTION_NAME = re.compile(r"\s*([A-Za-z_][\w.]*)\(")


class TracingRefused(Exception):
    """Tracing is on, but a condition for it is not met."""


def redact_error(error: str) -> str:
    """The exception class name from an error string, and nothing else."""
    match = _EXCEPTION_NAME.match(error)
    return f"{match.group(1)}: {REDACTED}" if match else REDACTED


def scrub_error(payload: dict[Any, Any]) -> dict[Any, Any]:
    """The client's anonymizer: replaces an `error` value, passes the rest through.

    langsmith calls it with `{"error": text}` for a run's error. Inputs,
    outputs, and metadata never reach it: `hide_inputs`, `hide_outputs`, and
    `hide_metadata` set to True return `{}` first.
    """
    error = payload.get("error")
    if error is None:
        return payload
    return {**payload, "error": redact_error(error) if isinstance(error, str) else REDACTED}


def build_client(
    api_key: str,
    *,
    api_url: str | None = None,
    session: requests.Session | None = None,
    auto_batch_tracing: bool = True,
) -> langsmith.Client:
    """A LangSmith client that sends no inputs, outputs, error text, or runtime info.

    `session`, `api_url`, and `auto_batch_tracing` are for tests, which
    capture the requests instead of sending them.
    """
    return langsmith.Client(
        api_key=api_key,
        api_url=api_url,
        session=session,
        auto_batch_tracing=auto_batch_tracing,
        hide_inputs=True,
        hide_outputs=True,
        # LangChain copies a model's invocation parameters into the metadata.
        hide_metadata=True,
        anonymizer=scrub_error,
        omit_traced_runtime_info=True,
    )


def corpus_dir_from_env() -> Path:
    """The corpus directory the CLI uses: `CORPUS_DIR`, else `corpus/`.

    Read from the environment only, as the CLI's `--root` options do: a
    CORPUS_DIR in .env is not seen by them, so it does not count here either.
    """
    value = os.environ.get("CORPUS_DIR")
    return Path(value) if value else DEFAULT_CORPUS


def is_example_corpus(corpus_dir: Path) -> bool:
    return corpus_dir.resolve() == EXAMPLE_CORPUS.resolve()


def configure_tracing(
    settings: TracingSettings,
    corpus_dir: Path,
    *,
    client: langsmith.Client | None = None,
) -> bool:
    """Turn tracing off, or on with the guards. Returns whether it is on.

    Raises `TracingRefused` if tracing is asked for and a condition fails.
    `client` replaces `build_client(...)`, for tests.
    """
    if not settings.langsmith_tracing:
        langsmith.configure(enabled=False)
        return False
    if not is_example_corpus(corpus_dir):
        raise TracingRefused(
            f"LANGSMITH_TRACING=true needs the example corpus, but the corpus directory"
            f" is {corpus_dir}. Traces may only come from corpus-example/: set"
            " CORPUS_DIR=corpus-example in the environment, or turn tracing off."
        )
    if client is None:
        if settings.langsmith_api_key is None:
            raise TracingRefused("LANGSMITH_TRACING=true needs LANGSMITH_API_KEY.")
        client = build_client(settings.langsmith_api_key.get_secret_value())
    os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"
    langsmith.configure(client=client, enabled=True)
    return True


def tracing_enabled() -> bool:
    """Whether langsmith will trace the next run."""
    return ls_utils.tracing_is_enabled() is True
