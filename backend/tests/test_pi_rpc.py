"""Tests for generation/pi_rpc.py, against the fake Pi in fake_pi.py.

Each test writes a script for the fake, runs `run_prompt` with
`executable=(python, fake_pi.py, script)`, and checks the reply or the typed
error. No test starts the real Pi; test_pi_live.py does, with --live.
"""

import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from fake_pi import (
    FAKE_PI,
    Step,
    accepted,
    assistant_end,
    reply_steps,
    system_start,
    write_script,
)

from profile_rag.generation.pi_rpc import (
    ISOLATION_FLAGS,
    PARENT_SESSION_VARIABLES,
    PiConfig,
    PiModelError,
    PiProcessError,
    PiProtocolError,
    PiTimeout,
    pi_command,
    run_prompt,
)


def fake_config(script: Path, *, timeout: float = 10.0) -> PiConfig:
    executable = (sys.executable, str(FAKE_PI), str(script))
    return PiConfig(model="test/fake-model", executable=executable, timeout=timeout)


def run(
    tmp_path: Path,
    steps: Sequence[Mapping[str, object]],
    prompt: str = "the prompt",
    **kwargs: float,
) -> str:
    script = write_script(tmp_path / "script.json", steps)
    return run_prompt(prompt, fake_config(script, **kwargs)).text


# --- the command line ---------------------------------------------------------


def test_the_command_has_every_isolation_flag_and_no_prompt() -> None:
    command = pi_command(PiConfig(model="anthropic/claude-opus-5-5"))
    assert command[:3] == ["pi", "--mode", "rpc"]
    for flag in ISOLATION_FLAGS:
        assert flag in command
    assert command[command.index("--model") + 1] == "anthropic/claude-opus-5-5"
    assert {"--no-session", "--no-tools", "--no-context-files", "--no-extensions"} <= set(command)


def test_the_prompt_goes_through_stdin_not_the_command_line(tmp_path: Path) -> None:
    record = tmp_path / "record.json"
    script = write_script(
        tmp_path / "script.json",
        [*reply_steps("ok"), {"emit": {"type": "never-read"}}],
        record=record,
    )
    run_prompt("What is {this}?\nLine two", fake_config(script))

    received = json.loads(record.read_text())
    assert received["command"]["type"] == "prompt"
    assert received["command"]["message"] == "What is {this}?\nLine two"
    assert "What is {this}?\nLine two" not in received["argv"]
    assert "--no-session" in received["argv"]


def test_parent_pi_session_variables_do_not_reach_pi_or_change_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "PI_MODEL": "parent-session-model",
        "PI_PROVIDER": "parent-provider",
        "PI_SESSION_FILE": str(tmp_path / "parent-session.jsonl"),
        "PI_SESSION_ID": "parent-id",
        "PI_REASONING_LEVEL": "high",
        "PI_SUBAGENT_ID": "parent-subagent",
    }.items():
        monkeypatch.setenv(name, value)
    record = tmp_path / "record.json"
    script = write_script(tmp_path / "script.json", reply_steps("ok"), record=record)

    run_prompt("p", fake_config(script))

    received = json.loads(record.read_text())
    passed = set(received["env"])
    assert not passed & PARENT_SESSION_VARIABLES
    assert not [name for name in passed if name.startswith("PI_SUBAGENT_")]
    assert {"PI_OFFLINE", "PI_TELEMETRY"} <= passed
    argv = received["argv"]
    assert argv[argv.index("--model") + 1] == "test/fake-model"


def test_pi_runs_in_an_empty_directory_that_is_removed_afterwards(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, os, sys\n"
        "sys.stdin.readline()\n"
        f"open({str(tmp_path / 'cwd.json')!r}, 'w').write("
        "json.dumps({'cwd': os.getcwd(), 'files': os.listdir('.')}))\n"
    )
    config = PiConfig(model="m", executable=(sys.executable, str(probe)), timeout=10)
    with pytest.raises(PiProcessError):
        run_prompt("p", config)  # the probe exits without a reply
    seen = json.loads((tmp_path / "cwd.json").read_text())
    assert seen["files"] == []
    assert "profile-rag-pi-" in seen["cwd"]
    assert not Path(seen["cwd"]).exists()


# --- replies ------------------------------------------------------------------


def test_a_normal_run_returns_the_text_parts_of_the_last_assistant_message(
    tmp_path: Path,
) -> None:
    script = write_script(tmp_path / "script.json", reply_steps('{"status": "unknown"}'))
    reply = run_prompt("p", fake_config(script))
    assert reply.text == '{"status": "unknown"}'
    assert (reply.model, reply.response_model) == ("test/fake-model", "fake-model-2026")
    assert reply.system_sections == {"preamble", "cwd"}


def test_events_after_agent_end_are_read_until_agent_settled(tmp_path: Path) -> None:
    steps = [
        accepted(),
        assistant_end("first try", stop_reason="error", error="529 overloaded"),
        {"emit": {"type": "agent_end", "willRetry": True}},
        {"emit": {"type": "auto_retry_start", "attempt": 1}},
        assistant_end("second try"),
        {"emit": {"type": "agent_end", "willRetry": False}},
        {"emit": {"type": "agent_settled", "aborted": False}},
    ]
    assert run(tmp_path, steps) == "second try"


def test_a_system_prompt_with_a_context_file_section_raises_pi_protocol_error(
    tmp_path: Path,
) -> None:
    steps = reply_steps("ok")
    steps[2] = system_start(["preamble", "cwd", "contextFiles"])
    with pytest.raises(PiProtocolError, match="contextFiles"):
        run(tmp_path, steps)


# --- errors -------------------------------------------------------------------


def test_a_hung_pi_raises_pi_timeout_and_is_killed(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    hang = tmp_path / "hang.py"
    hang.write_text(
        f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n"
    )
    config = PiConfig(model="m", executable=(sys.executable, str(hang)), timeout=1.0)
    started = time.monotonic()
    with pytest.raises(PiTimeout) as caught:
        run_prompt("p", config)
    assert time.monotonic() - started < 6
    assert caught.value.seconds == 1.0
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)  # reaped: the pid no longer exists


def test_a_pi_that_stops_writing_mid_run_raises_pi_timeout(tmp_path: Path) -> None:
    steps = [accepted(), {"emit": {"type": "agent_start"}}, {"sleep": 30}]
    with pytest.raises(PiTimeout):
        run(tmp_path, steps, timeout=1.0)


def test_a_crash_raises_pi_process_error_with_exit_code_and_stderr_tail(tmp_path: Path) -> None:
    steps = [
        accepted(),
        {"stderr": "x" * 5000},
        {"stderr": "Error: model not found: test/fake-model"},
        {"exit": 3},
    ]
    with pytest.raises(PiProcessError) as caught:
        run(tmp_path, steps)
    assert caught.value.returncode == 3
    assert caught.value.stderr_tail.endswith("Error: model not found: test/fake-model")
    assert len(caught.value.stderr_tail) <= 2000
    assert "the prompt" not in str(caught.value)


def test_a_missing_binary_raises_pi_process_error_without_exit_code(tmp_path: Path) -> None:
    config = PiConfig(model="m", executable=(str(tmp_path / "no-such-pi"),))
    with pytest.raises(PiProcessError) as caught:
        run_prompt("p", config)
    assert caught.value.returncode is None
    assert "no-such-pi" in str(caught.value)


def test_a_line_that_is_not_json_raises_pi_protocol_error(tmp_path: Path) -> None:
    steps = [accepted(), {"raw": "Debugger attached. {not json"}]
    with pytest.raises(PiProtocolError, match="not JSON"):
        run(tmp_path, steps)


def test_a_json_line_that_is_not_an_object_raises_pi_protocol_error(tmp_path: Path) -> None:
    with pytest.raises(PiProtocolError, match="not an object"):
        run(tmp_path, [accepted(), {"raw": "[1, 2]"}])


def test_a_rejected_prompt_raises_pi_protocol_error(tmp_path: Path) -> None:
    rejected: Step = {
        "emit": {
            "id": "$ID",
            "type": "response",
            "command": "prompt",
            "success": False,
            "error": "Agent is busy",
        }
    }
    with pytest.raises(PiProtocolError, match="Agent is busy"):
        run(tmp_path, [rejected])


def test_a_prompt_handled_without_a_run_raises_pi_protocol_error(tmp_path: Path) -> None:
    handled = accepted()
    emitted = handled["emit"]
    assert isinstance(emitted, dict)
    emitted["data"] = {"disposition": "handled"}
    with pytest.raises(PiProtocolError, match="handled"):
        run(tmp_path, [handled])


def test_settling_without_an_assistant_message_raises_pi_protocol_error(tmp_path: Path) -> None:
    steps = [accepted(), {"emit": {"type": "agent_settled", "aborted": False}}]
    with pytest.raises(PiProtocolError, match="without an assistant message"):
        run(tmp_path, steps)


def test_a_provider_error_raises_pi_model_error(tmp_path: Path) -> None:
    steps = [
        accepted(),
        assistant_end("", stop_reason="error", error="401 invalid x-api-key"),
        {"emit": {"type": "agent_settled", "aborted": False}},
    ]
    with pytest.raises(PiModelError, match="401 invalid x-api-key"):
        run(tmp_path, steps)


def test_a_reply_cut_off_at_the_token_limit_raises_pi_model_error(tmp_path: Path) -> None:
    steps = [
        accepted(),
        assistant_end('{"status": "answ', stop_reason="length"),
        {"emit": {"type": "agent_settled", "aborted": False}},
    ]
    with pytest.raises(PiModelError, match="length") as caught:
        run(tmp_path, steps)
    assert "answ" not in str(caught.value)


def test_an_aborted_run_raises_pi_model_error(tmp_path: Path) -> None:
    steps = [accepted(), {"emit": {"type": "agent_settled", "aborted": True}}]
    with pytest.raises(PiModelError, match="aborted"):
        run(tmp_path, steps)


def test_an_extension_ui_request_raises_pi_protocol_error(tmp_path: Path) -> None:
    steps = [accepted(), {"emit": {"type": "extension_ui_request", "id": "u1"}}]
    with pytest.raises(PiProtocolError, match="user input"):
        run(tmp_path, steps)
