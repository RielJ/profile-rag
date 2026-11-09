"""Send one prompt to Pi in RPC mode and return the reply text.

Pi (https://pi.dev) runs as a child process: `pi --mode rpc` reads JSON
commands on stdin and writes JSON records on stdout, one per LF-terminated
line. `run_prompt` sends one `prompt` command and reads records until Pi
reports `agent_settled`. The process never outlives the call: it is killed
and reaped on success, on error, and on timeout.

Streaming. If `on_text` is given, it is called with each piece of reply text
as Pi reports it (`message_update` records with a `text_delta` event). The
pieces are a draft: the return value, taken from the final assistant message,
is the only reply to trust. `on_text` runs on the calling thread; an
exception it raises ends the call, and the process is still killed.

Isolation. Pi is a coding agent; here it only turns a prompt into text.
`ISOLATION_FLAGS` turn off everything else: tools, extensions, skills,
prompt templates, themes, context files (AGENTS.md, CLAUDE.md), MCP servers,
project-local files, startup network calls, and session files. Without
`--no-session`, Pi would save each prompt, evidence included, under
~/.pi/agent/sessions/. The process also runs in a new, empty temporary
directory, so no project file is near it. The prompt goes through stdin, not
the command line, so other users cannot read it with `ps`. Pi reports the
sections of its system prompt at the start of a run; any section other than
`ALLOWED_SYSTEM_SECTIONS` stops the run with PiProtocolError, so a loaded
context file or skill fails closed.

Tested with Pi `TESTED_PI_VERSION`. `tests/test_pi_live.py --live` checks
that the installed Pi lists every flag in `ISOLATION_FLAGS`.

Errors: every failure raises a subclass of `PiError`. None of their messages
contains the prompt or the reply.

Tests: backend/tests/test_pi_rpc.py, with the fake Pi in tests/fake_pi.py.
"""

import contextlib
import json
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import IO, NoReturn

TESTED_PI_VERSION = "1.5.0"

# The lowest thinking level of claude-opus-5-5. Pi 1.5.0's model table marks
# "off" and "minimal" as unsupported for it (thinkingLevelMap null).
DEFAULT_THINKING = "low"

ISOLATION_FLAGS = (
    "--no-session",
    "--no-tools",
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-themes",
    "--no-context-files",
    "--no-mcp",
    "--no-approve",
    "--offline",
)

# Replaces Pi's default system prompt, which describes a coding agent and its
# tools. The task rules travel in the prompt itself (generation/prompt.py).
SYSTEM_PROMPT = (
    "You are a text-only answering service. You have no tools and no files."
    " Follow the instructions in the user message and reply with text only."
)

# The sections of Pi's system prompt, as Pi 1.5.0 reports them in the
# `message_start` record of the system message, when every isolation flag is
# set. Any other section (such as context files, skills, or tool notes) means
# something was loaded that should not have been, and the run is stopped.
ALLOWED_SYSTEM_SECTIONS = frozenset({"preamble", "cwd"})

# Environment overrides for the Pi process.
PI_ENVIRONMENT = {"PI_OFFLINE": "1", "PI_TELEMETRY": "0"}

# Variables a parent Pi session exports to the commands it runs: its session
# file and ID, its model, and subagent state. Pi 1.5.0 does not read them, but
# the child is a separate, sessionless run, so none of them is passed on.
PARENT_SESSION_VARIABLES = frozenset(
    {"PI_SESSION_ID", "PI_SESSION_FILE", "PI_PROVIDER", "PI_MODEL", "PI_REASONING_LEVEL"}
)
PARENT_SESSION_PREFIXES = ("PI_SUBAGENT_",)


def child_environment(parent: Mapping[str, str]) -> dict[str, str]:
    """`parent` without the parent session's variables, plus `PI_ENVIRONMENT`."""
    kept = {
        name: value
        for name, value in parent.items()
        if name not in PARENT_SESSION_VARIABLES and not name.startswith(PARENT_SESSION_PREFIXES)
    }
    return {**kept, **PI_ENVIRONMENT}


REQUEST_ID = "profile-rag-1"

# How much of Pi's stderr a PiProcessError keeps, in bytes.
STDERR_TAIL_BYTES = 2000

# After `agent_settled`, how long Pi may take to exit on its own once stdin
# closes, before it is killed.
EXIT_GRACE_SECONDS = 2.0


@dataclass(frozen=True)
class PiConfig:
    """How to start Pi and how long to wait for it.

    `executable` is the command prefix: `("pi",)` normally, or an
    interpreter and script for the fake Pi in tests.
    """

    model: str
    executable: tuple[str, ...] = ("pi",)
    timeout: float = 120.0
    thinking: str = DEFAULT_THINKING
    system_prompt: str = SYSTEM_PROMPT


@dataclass(frozen=True)
class PiReply:
    """The final assistant text, the model that wrote it, and Pi's system-prompt sections.

    `model` is `provider/model` as Pi's assistant message reports it, and
    `response_model` the concrete model the provider reported (empty if none).
    Pi reports its system prompt as named sections. With the isolation flags
    they are only `preamble` and `cwd` (`ALLOWED_SYSTEM_SECTIONS`).
    """

    text: str
    model: str = ""
    response_model: str = ""
    system_sections: frozenset[str] = field(default_factory=frozenset)


class PiError(Exception):
    """Base class of every failure in this module."""


class PiTimeout(PiError):
    def __init__(self, seconds: float) -> None:
        super().__init__(f"Pi did not finish within {seconds:g} seconds")
        self.seconds = seconds


class PiProcessError(PiError):
    """Pi could not start, or exited before it finished the prompt.

    `returncode` is None when the process never started.
    """

    def __init__(self, message: str, *, returncode: int | None, stderr_tail: str) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr_tail = stderr_tail


class PiProtocolError(PiError):
    """Pi wrote something this module does not understand, or rejected the prompt."""


class PiModelError(PiError):
    """The model call ended without a complete reply (provider error, abort, or length)."""


def pi_command(config: PiConfig) -> list[str]:
    """The argument list that starts Pi. No shell is involved."""
    return [
        *config.executable,
        "--mode",
        "rpc",
        *ISOLATION_FLAGS,
        "--model",
        config.model,
        "--thinking",
        config.thinking,
        "--system-prompt",
        config.system_prompt,
    ]


OnText = Callable[[str], None]


def run_prompt(prompt: str, config: PiConfig, on_text: OnText | None = None) -> PiReply:
    """Send `prompt` to a fresh Pi process and return its final reply.

    `on_text`, if given, receives the reply text piece by piece as it is
    generated (see the module docstring).
    """
    command = pi_command(config)
    environment = child_environment(os.environ)
    with tempfile.TemporaryDirectory(prefix="profile-rag-pi-") as workdir:
        try:
            process = subprocess.Popen(
                command,
                cwd=workdir,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # its own process group, killed as a whole
            )
        except OSError as error:
            raise PiProcessError(
                f"cannot start {config.executable[0]!r}: {error.strerror or error}",
                returncode=None,
                stderr_tail="",
            ) from error
        session = _Session(process)
        try:
            return session.converse(prompt, config.timeout, on_text)
        finally:
            session.stop()


class _Session:
    """One Pi process: its reader threads, the event loop, and cleanup."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self.process = process
        self.lines: queue.Queue[bytes | None] = queue.Queue()
        self.stderr: deque[bytes] = deque()
        self.stderr_size = 0
        self.lock = threading.Lock()
        if process.stdout is None or process.stderr is None:
            raise ValueError("start Pi with stdout=PIPE and stderr=PIPE")
        self.threads = [
            threading.Thread(target=self._read_stdout, args=(process.stdout,), daemon=True),
            threading.Thread(target=self._read_stderr, args=(process.stderr,), daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    # --- reader threads: Pi must never block on a full pipe ---------------

    def _read_stdout(self, stream: IO[bytes]) -> None:
        # A binary readline splits on LF only, as the RPC framing requires.
        for line in iter(stream.readline, b""):
            self.lines.put(line)
        self.lines.put(None)  # end of stream

    def _read_stderr(self, stream: IO[bytes]) -> None:
        # Line by line: Pi's stderr is diagnostics, one message per line.
        for chunk in iter(stream.readline, b""):
            with self.lock:
                self.stderr.append(chunk)
                self.stderr_size += len(chunk)
                while self.stderr_size - len(self.stderr[0]) >= STDERR_TAIL_BYTES:
                    self.stderr_size -= len(self.stderr.popleft())

    def stderr_tail(self) -> str:
        with self.lock:
            data = b"".join(self.stderr)[-STDERR_TAIL_BYTES:]
        return data.decode("utf-8", errors="replace").strip()

    # --- the conversation -------------------------------------------------

    def converse(self, prompt: str, timeout: float, on_text: OnText | None = None) -> PiReply:
        deadline = time.monotonic() + timeout
        command = {"id": REQUEST_ID, "type": "prompt", "message": prompt}
        self._send(json.dumps(command).encode("utf-8") + b"\n", deadline, timeout)

        assistant: dict[str, object] | None = None
        sections: frozenset[str] = frozenset()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PiTimeout(timeout)
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty:
                raise PiTimeout(timeout) from None
            if line is None:
                self._raise_exited(deadline)
            record = _decode(line)
            kind = record.get("type")
            if kind == "response" and record.get("id") == REQUEST_ID:
                _check_accepted(record)
            elif kind == "message_start":
                message = _as_dict(record.get("message"))
                if message.get("role") == "system":
                    sections = frozenset(_as_dict(message.get("sections")))
                    _check_sections(sections)
            elif kind == "message_update" and on_text is not None:
                if delta := _text_delta(record):
                    on_text(delta)
            elif kind == "message_end":
                message = _as_dict(record.get("message"))
                if message.get("role") == "assistant":
                    assistant = message
            elif kind == "extension_ui_request":
                raise PiProtocolError("Pi asked for user input, which this client never gives")
            elif kind == "agent_settled":
                if record.get("aborted") is True:
                    raise PiModelError("Pi aborted the run")
                return PiReply(
                    text=_final_text(assistant),
                    model=_reported_model(assistant),
                    response_model=_text_field(assistant, "responseModel"),
                    system_sections=sections,
                )

    def _send(self, data: bytes, deadline: float, timeout: float) -> None:
        """Write `data` to stdin in a thread, so a stalled Pi cannot block past the deadline."""
        stdin = self.process.stdin
        if stdin is None:
            raise ValueError("start Pi with stdin=PIPE")

        def write() -> None:
            try:
                stdin.write(data)
                stdin.flush()
            except OSError:
                pass  # Pi exited; the event loop sees end of stream and reports it

        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        writer.join(max(0.0, deadline - time.monotonic()))
        if writer.is_alive():
            raise PiTimeout(timeout)

    def _raise_exited(self, deadline: float) -> NoReturn:
        try:
            returncode = self.process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            returncode = None
        self.threads[1].join(timeout=1.0)  # collect the rest of stderr
        described = "is still running" if returncode is None else f"exited with code {returncode}"
        raise PiProcessError(
            f"Pi closed its output before the reply was complete; the process {described}",
            returncode=returncode,
            stderr_tail=self.stderr_tail(),
        )

    def stop(self) -> None:
        """Close stdin, give Pi a moment to exit, then kill its process group and reap it."""
        process = self.process
        if process.stdin is not None:
            with contextlib.suppress(OSError):
                process.stdin.close()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=EXIT_GRACE_SECONDS)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)  # also any child Pi started
        if process.poll() is None:
            process.kill()
        process.wait()
        for thread in self.threads:
            thread.join(timeout=1.0)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def _decode(line: bytes) -> dict[str, object]:
    text = line.rstrip(b"\n").rstrip(b"\r")
    try:
        record: object = json.loads(text)
    except UnicodeDecodeError, json.JSONDecodeError:
        raise PiProtocolError("Pi wrote a line that is not JSON") from None
    if not isinstance(record, dict):
        raise PiProtocolError("Pi wrote a JSON value that is not an object")
    return {str(key): value for key, value in record.items()}


def _as_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _check_sections(sections: frozenset[str]) -> None:
    """Fail closed if Pi's system prompt has more than the preamble and the cwd."""
    extra = sorted(sections - ALLOWED_SYSTEM_SECTIONS)
    if extra:
        raise PiProtocolError(
            f"Pi's system prompt has sections this client does not allow: {', '.join(extra)}."
            " A context file, skill, or tool may have been loaded."
        )


def _check_accepted(response: dict[str, object]) -> None:
    if response.get("success") is not True:
        error = response.get("error")
        raise PiProtocolError(f"Pi rejected the prompt: {error if isinstance(error, str) else '?'}")
    disposition = _as_dict(response.get("data")).get("disposition")
    if disposition != "started":
        raise PiProtocolError(
            f"Pi did not start a run for the prompt (disposition {disposition!r})"
        )


def _text_delta(record: dict[str, object]) -> str:
    """The reply text of a `message_update` record, or "" if it carries none."""
    event = _as_dict(record.get("assistantMessageEvent"))
    delta = event.get("delta")
    return delta if event.get("type") == "text_delta" and isinstance(delta, str) else ""


def _text_field(message: dict[str, object] | None, key: str) -> str:
    value = (message or {}).get(key)
    return value if isinstance(value, str) else ""


def _reported_model(message: dict[str, object] | None) -> str:
    provider = _text_field(message, "provider")
    model = _text_field(message, "model")
    return f"{provider}/{model}" if provider and model else model


def _final_text(message: dict[str, object] | None) -> str:
    if message is None:
        raise PiProtocolError("Pi settled without an assistant message")
    stop_reason = message.get("stopReason")
    if stop_reason != "stop":
        error = message.get("errorMessage")
        detail = f": {error[:300]}" if isinstance(error, str) and error else ""
        raise PiModelError(f"the model call ended with stop reason {stop_reason!r}{detail}")
    content = message.get("content")
    parts: Sequence[object] = content if isinstance(content, list) else []
    text = "".join(
        str(part.get("text", ""))
        for part in parts
        if isinstance(part, dict) and part.get("type") == "text"
    )
    if not text.strip():
        raise PiProtocolError("Pi's assistant message has no text")
    return text
