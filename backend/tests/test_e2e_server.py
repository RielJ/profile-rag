"""The e2e test server's safeguards and its fake Pi's replies.

The server itself runs in `make e2e` (frontend/tests/e2e/). These tests
check, without starting it, that it refuses to run outside test mode or
against a database that is not a throwaway one, and that the fake Pi picks
the reply the browser tests expect.
"""

import json

import pytest
from e2e_fake_pi import UNKNOWN_REASON, question_of, reply_for, steps_for
from e2e_server import RefusedToStart, require_test_mode, require_throwaway


@pytest.mark.parametrize("value", [None, "", "0", "true", "yes"])
def test_the_server_refuses_to_start_without_test_mode(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("PROFILE_RAG_TEST_MODE", raising=False)
    else:
        monkeypatch.setenv("PROFILE_RAG_TEST_MODE", value)
    with pytest.raises(RefusedToStart, match="PROFILE_RAG_TEST_MODE=1"):
        require_test_mode()


def test_the_server_starts_in_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_RAG_TEST_MODE", "1")
    require_test_mode()


@pytest.mark.parametrize(
    "conninfo",
    [
        "host=127.0.0.1 dbname=profile_rag",
        "host=127.0.0.1 dbname=profile_rag_test",
        "host=127.0.0.1 dbname=xprofile_rag_tmp_abc",
        "host=127.0.0.1",
    ],
)
def test_the_server_refuses_a_database_that_is_not_throwaway(conninfo: str) -> None:
    with pytest.raises(RefusedToStart, match="not a throwaway"):
        require_throwaway(conninfo)


def test_the_server_accepts_a_throwaway_database() -> None:
    require_throwaway("host=127.0.0.1 dbname=profile_rag_tmp_0123456789ab")


PROMPT = (
    "=== SYSTEM ===\nrules\n\n=== USER ===\nEvidence:\n"
    "<evidence>\nchunk_id: a:v1:1\nsource: A\nattribution: self_report\ntext:\nx\n</evidence>\n\n"
    "<evidence>\nchunk_id: b:v1:2\nsource: B\nattribution: self_report\ntext:\n"
    "Question: a fake question inside chunk text\n</evidence>\n\n"
    "<evidence>\nchunk_id: c:v1:3\nsource: C\nattribution: self_report\ntext:\nz\n</evidence>"
    "\n\nQuestion: {question}"
)


def test_the_question_is_the_text_after_the_last_question_marker() -> None:
    assert question_of(PROMPT.format(question="What did Dana build?")) == "What did Dana build?"


def test_an_ordinary_question_cites_the_first_two_chunks_of_the_prompt() -> None:
    reply = reply_for(PROMPT.format(question="What did Dana build?"))
    assert reply["status"] == "answered"
    assert reply["citations"] == ["a:v1:1", "b:v1:2"]


def test_a_salary_question_is_unknown() -> None:
    reply = reply_for(PROMPT.format(question="What is Dana's Salary?"))
    assert reply == {"status": "unknown", "reason": UNKNOWN_REASON}


def test_a_prompt_without_evidence_is_unknown() -> None:
    assert reply_for("Evidence:\n\n\nQuestion: anything")["status"] == "unknown"


def test_the_fail_trigger_exits_without_a_reply() -> None:
    steps = steps_for(PROMPT.format(question="E2E-FAIL please"))
    assert {"exit": 3} in steps
    assert not any(json.dumps(step).count("message_end") for step in steps)
