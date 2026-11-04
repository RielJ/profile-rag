"""Opt-in live tests: the installed Pi, and one real prompt through it.

Skipped unless pytest runs with `--live`:

    uv --directory backend run pytest tests/test_pi_live.py --live

They use PROFILE_RAG_PI_BIN and PROFILE_RAG_PI_MODEL from the settings. The prompt test makes one
model call. It checks that Pi loads no context file or skill, and that it
writes no session file for its working directory.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from profile_rag.generation.pi_chat_model import PiChatModel
from profile_rag.generation.pi_rpc import ISOLATION_FLAGS, TESTED_PI_VERSION, PiConfig
from profile_rag.settings import get_settings

pytestmark = pytest.mark.live

# Every option pi_rpc.pi_command passes.
REQUIRED_FLAGS = ("--mode", "--model", "--thinking", "--system-prompt", *ISOLATION_FLAGS)


def pi(*args: str) -> str:
    completed = subprocess.run(
        [get_settings().pi_bin, *args], capture_output=True, text=True, timeout=60, check=True
    )
    return completed.stdout + completed.stderr


def sessions_dir() -> Path:
    """Where Pi stores sessions when no setting overrides it."""
    if override := os.environ.get("PI_CODING_AGENT_SESSION_DIR"):
        return Path(override)
    agent_dir = os.environ.get("PI_CODING_AGENT_DIR") or Path.home() / ".pi" / "agent"
    return Path(agent_dir) / "sessions"


def test_the_installed_pi_is_the_tested_version() -> None:
    assert pi("--version").strip() == TESTED_PI_VERSION


def test_pi_help_lists_every_flag_the_adapter_relies_on() -> None:
    help_text = pi("--help")
    # A flag is followed by a space, or by a comma and its short alias.
    missing = [flag for flag in REQUIRED_FLAGS if not re.search(rf"{flag}[ ,]", help_text)]
    assert not missing, f"pi --help does not list {missing}"
    assert "rpc" in help_text


def test_one_prompt_answers_without_context_files_or_session_files() -> None:
    settings = get_settings()
    config = PiConfig(
        model=settings.pi_model, executable=(settings.pi_bin,), timeout=settings.pi_timeout_seconds
    )
    sessions = sessions_dir()
    before = set(sessions.rglob("*")) if sessions.exists() else set()

    reply = PiChatModel(config=config).invoke(
        [HumanMessage("Reply with the single word OK and nothing else.")]
    )

    assert reply.text.strip().strip(".").upper() == "OK"
    # The model from the settings, not one inherited from a parent Pi session.
    assert reply.response_metadata["model_name"] == settings.pi_model
    # Only Pi's own preamble and the cwd line: no context files, skills, or tools.
    assert reply.response_metadata["pi_system_sections"] == ["cwd", "preamble"]
    after = set(sessions.rglob("*")) if sessions.exists() else set()
    ours = [path for path in after - before if "profile-rag-pi-" in str(path)]
    assert not ours, f"Pi wrote session files: {ours}"
