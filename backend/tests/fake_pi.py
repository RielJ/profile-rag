"""A fake `pi --mode rpc` for tests: replays a scripted list of steps.

Run it as `python fake_pi.py SCRIPT [pi arguments...]`. SCRIPT is a JSON
file:

    {"record": "PATH or null", "steps": [STEP, ...]}

The fake reads one command line from stdin, writes the command, its own
arguments, and its PI_* environment variables to `record` (if set), then runs the steps in order:

    {"emit": {...}}      write the object as one JSON line; the value "$ID"
                         anywhere at the top level becomes the command's id
    {"raw": "text"}      write the text and a newline as is
    {"stderr": "text"}   write the text and a newline to stderr
    {"sleep": seconds}   wait
    {"exit": code}       exit with that code

After the last step, the fake waits for stdin to close and exits with 0, like
Pi does.

`write_script` builds the file; the `*_steps` functions build common scripts.
"""

import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

FAKE_PI = Path(__file__).resolve()

Step = dict[str, object]


def write_script(
    path: Path, steps: Sequence[Mapping[str, object]], *, record: Path | None = None
) -> Path:
    data = {"record": str(record) if record else None, "steps": list(steps)}
    path.write_text(json.dumps(data))
    return path


def accepted() -> Step:
    return {
        "emit": {
            "id": "$ID",
            "type": "response",
            "command": "prompt",
            "success": True,
            "data": {"disposition": "started"},
        }
    }


def assistant_end(text: str, *, stop_reason: str = "stop", error: str | None = None) -> Step:
    message: dict[str, object] = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "not part of the reply"},
            {"type": "text", "text": text},
        ],
        "provider": "test",
        "model": "fake-model",
        "responseModel": "fake-model-2026",
        "stopReason": stop_reason,
    }
    if error is not None:
        message["errorMessage"] = error
    return {"emit": {"type": "message_end", "message": message}}


def system_start(sections: list[str]) -> Step:
    message = {"role": "system", "content": "", "sections": dict.fromkeys(sections, "...")}
    return {"emit": {"type": "message_start", "message": message}}


def reply_steps(text: str) -> list[Step]:
    """A normal run: accepted, the events Pi sends, one reply, settled."""
    return [
        accepted(),
        {"emit": {"type": "agent_start"}},
        system_start(["preamble", "cwd"]),
        {"emit": {"type": "message_update", "assistantMessageEvent": {"type": "text_delta"}}},
        assistant_end(text),
        {"emit": {"type": "agent_end", "willRetry": False}},
        {"emit": {"type": "agent_settled", "aborted": False}},
    ]


def main() -> None:
    script = json.loads(Path(sys.argv[1]).read_text())
    line = sys.stdin.readline()
    command = json.loads(line) if line.strip() else {}
    if script["record"]:
        pi_variables = {name: value for name, value in os.environ.items() if name.startswith("PI_")}
        record = {"argv": sys.argv[2:], "command": command, "env": pi_variables}
        Path(script["record"]).write_text(json.dumps(record))
    for step in script["steps"]:
        if "emit" in step:
            record = {k: command.get("id") if v == "$ID" else v for k, v in step["emit"].items()}
            sys.stdout.write(json.dumps(record) + "\n")
            sys.stdout.flush()
        elif "raw" in step:
            sys.stdout.write(step["raw"] + "\n")
            sys.stdout.flush()
        elif "stderr" in step:
            sys.stderr.write(step["stderr"] + "\n")
            sys.stderr.flush()
        elif "sleep" in step:
            time.sleep(step["sleep"])
        elif "exit" in step:
            sys.exit(step["exit"])
    sys.stdin.read()  # Pi exits when stdin closes


if __name__ == "__main__":
    main()
