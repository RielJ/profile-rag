"""A fake `pi --mode rpc` for the end-to-end tests (frontend/tests/e2e/).

fake_pi.py replays a fixed script, which suits one test and one reply.
The browser tests ask several questions of one server, so this fake reads
the prompt and picks its reply from the question:

- a question that contains `UNKNOWN_TRIGGER` gets an unknown answer;
- a question that contains `FAIL_TRIGGER` makes the fake exit with code 3
  after accepting the prompt, which the API reports as `pi_failed`;
- any other question gets an answer that cites the first two evidence
  blocks of the prompt, so the citations are always chunks the model was
  given.

The reply is the JSON object the answer parser expects. It never calls a
model. Run it as `python e2e_fake_pi.py [pi arguments...]`; the arguments
are ignored. Tests: backend/tests/test_e2e_server.py.
"""

import json
import re
from collections.abc import Sequence
from pathlib import Path

from fake_pi import Step, accepted, read_command, reply_steps, run_steps

E2E_FAKE_PI = Path(__file__).resolve()

UNKNOWN_TRIGGER = "salary"
FAIL_TRIGGER = "E2E-FAIL"

UNKNOWN_REASON = "The evidence does not mention a salary."

_CHUNK_ID = re.compile(r"^chunk_id: (\S+)$", re.MULTILINE)


def question_of(prompt: str) -> str:
    """The question: the text after the last `Question: ` (prompt.py's HUMAN_TEMPLATE)."""
    return prompt.rpartition("Question: ")[2].strip()


def reply_for(prompt: str) -> dict[str, object]:
    """The answer object for `prompt`, as described above."""
    question = question_of(prompt)
    cited = list(dict.fromkeys(_CHUNK_ID.findall(prompt)))[:2]
    if UNKNOWN_TRIGGER in question.lower() or not cited:
        return {"status": "unknown", "reason": UNKNOWN_REASON}
    return {
        "status": "answered",
        "answer": f"A scripted answer from the e2e fake model, citing {len(cited)} chunks.",
        "citations": cited,
    }


def steps_for(prompt: str) -> Sequence[Step]:
    if FAIL_TRIGGER in question_of(prompt):
        return [accepted(), {"stderr": "fake Pi failure"}, {"exit": 3}]
    return reply_steps(json.dumps(reply_for(prompt)))


def main() -> None:
    command = read_command()
    message = command.get("message")
    run_steps(command, steps_for(message if isinstance(message, str) else ""))


if __name__ == "__main__":
    main()
