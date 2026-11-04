"""Tests for generation/pi_chat_model.py, against the fake Pi in fake_pi.py."""

import json
import sys
from pathlib import Path

import pytest
from fake_pi import FAKE_PI, accepted, reply_steps, write_script
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from profile_rag.generation.pi_chat_model import PiChatModel, join_messages
from profile_rag.generation.pi_rpc import PiConfig, PiProcessError, PiTimeout


def model_for(script: Path, *, timeout: float = 10.0) -> PiChatModel:
    executable = (sys.executable, str(FAKE_PI), str(script))
    return PiChatModel(config=PiConfig(model="test/fake", executable=executable, timeout=timeout))


def test_messages_are_joined_under_role_markers() -> None:
    prompt = join_messages([SystemMessage("rules"), HumanMessage("question?")])
    assert prompt == "=== SYSTEM ===\nrules\n\n=== USER ===\nquestion?"


def test_invoke_returns_one_ai_message_with_the_reply(tmp_path: Path) -> None:
    record = tmp_path / "record.json"
    script = write_script(tmp_path / "s.json", reply_steps("the reply"), record=record)
    model = model_for(script)

    result = model.invoke([SystemMessage("rules"), HumanMessage("/compact now")])

    assert isinstance(result, AIMessage)
    assert result.text == "the reply"
    assert result.response_metadata["pi_system_sections"] == ["cwd", "preamble"]
    assert result.response_metadata["model_name"] == "test/fake-model"
    sent = json.loads(record.read_text())["command"]["message"]
    assert sent == "=== SYSTEM ===\nrules\n\n=== USER ===\n/compact now"
    assert model._llm_type == "pi-rpc"


def test_pi_errors_pass_through_unchanged(tmp_path: Path) -> None:
    crash = write_script(tmp_path / "crash.json", [accepted(), {"stderr": "boom"}, {"exit": 2}])
    with pytest.raises(PiProcessError) as caught:
        model_for(crash).invoke("hi")
    assert caught.value.returncode == 2
    assert caught.value.stderr_tail == "boom"

    hang = write_script(tmp_path / "hang.json", [accepted(), {"sleep": 30}])
    with pytest.raises(PiTimeout):
        model_for(hang, timeout=1.0).invoke("hi")


def test_stop_sequences_are_rejected(tmp_path: Path) -> None:
    script = write_script(tmp_path / "s.json", reply_steps("x"))
    with pytest.raises(ValueError, match="stop sequences"):
        model_for(script).invoke("hi", stop=["\n"])
