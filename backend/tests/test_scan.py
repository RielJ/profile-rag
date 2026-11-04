"""Tests for corpus/scan.py: the secret and personal-data flagger.

Fake secrets are assembled from pieces at run time, so this file itself
never contains a string that looks like a real token to a secret scanner.
"""

from pathlib import Path

import pytest

from profile_rag.corpus.scan import Finding, main, scan_file, scan_text

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_APPROVED = REPO_ROOT / "corpus-example" / "approved"


def rules(text: str) -> list[str]:
    return [finding.rule for finding in scan_text(text)]


def test_flags_an_email_with_its_position() -> None:
    assert scan_text("line one\nmail dana@example.com now") == [
        Finding("email", "personal_data", 2, 6, "dana@example.com")
    ]


@pytest.mark.parametrize(
    "phone",
    ["+63 917 555 0100", "+1-555-010-0199", "(555) 010-0100", "555.010.0100", "0917 555 0100"],
)
def test_flags_phone_numbers(phone: str) -> None:
    assert rules(f"call {phone} today") == ["phone"]


@pytest.mark.parametrize(
    "text",
    [
        "version 1.2.3",
        "released 2026-10-01",
        "about 2,000 jobs per second",
        "from 50 minutes to 6",
        "listening on 127.0.0.1:8000",
        "public DNS at 8.8.8.8",
        "commit 3f9a1c2e7b4d5a6f8e0c1b2a3d4e5f60718293a4",
    ],
)
def test_ignores_ordinary_numbers_and_public_addresses(text: str) -> None:
    assert rules(text) == []


@pytest.mark.parametrize(
    ("address", "flagged"),
    [
        ("10.0.0.5", True),
        ("172.16.4.4", True),
        ("172.31.255.1", True),
        ("172.32.0.1", False),
        ("192.168.1.20", True),
        ("100.64.1.1", True),
        ("127.0.0.1", False),
        ("999.1.1.1", False),
    ],
)
def test_flags_private_ipv4_addresses(address: str, flagged: bool) -> None:
    assert rules(f"host {address} up") == (["private_ip"] if flagged else [])


@pytest.mark.parametrize(
    ("secret", "rule"),
    [
        ("AKIA" + "IOSFODNN7EXAMPLE", "aws_access_key"),
        ("ghp_" + "a1" * 18, "github_token"),
        ("github_pat_" + "B2" * 12, "github_token"),
        ("xoxb-" + "1234567890-abc", "slack_token"),
        ("sk-" + "x" * 24, "api_key"),
        ("sk-ant-" + "y" * 24, "api_key"),
        ("eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12, "jwt"),
        ("-----BEGIN OPENSSH " + "PRIVATE KEY-----", "private_key"),
        ("-----BEGIN " + "PRIVATE KEY-----", "private_key"),
    ],
)
def test_flags_key_patterns(secret: str, rule: str) -> None:
    assert rule in rules(f"value {secret} end")


def test_flags_credential_assignments_and_urls_with_passwords() -> None:
    assert rules('api_key = "abcd1234efgh"') == ["credential_assignment"]
    assert rules("PASSWORD: correct-horse-battery") == ["credential_assignment"]
    assert rules("token: short") == []  # under 8 characters: probably prose
    found = rules("postgresql://app:" + "hunter2secret@db.example.com/app")
    # `secret@db.example.com` also looks like an email. Two flags for one
    # problem is fine: the scan only points a reviewer at the line.
    assert found == ["url_with_credentials", "email"]


def test_flags_internal_hostnames() -> None:
    assert rules("ssh build01.corp") == ["internal_hostname"]
    assert rules("see wiki.intranet for details") == ["internal_hostname"]
    assert rules("open http://grafana.internal:3000") == ["internal_hostname"]
    assert rules("mirror at pkg.corp.example.com") == ["internal_hostname"]
    assert rules("see example.com/docs") == []


def test_findings_are_sorted_by_position() -> None:
    text = "b@example.com and 10.0.0.1\nfirst a@example.com"
    positions = [(f.line, f.column) for f in scan_text(text)]
    assert positions == sorted(positions)
    assert len(positions) == 3


def test_display_hides_secrets_but_not_personal_data() -> None:
    token = "ghp_" + "Z" * 36
    (secret,) = scan_text(token)
    assert secret.display() == "1:1 github_token: ghp_…"
    (email,) = scan_text("a@example.com")
    assert email.display() == "1:1 email: a@example.com"


def test_scan_file_survives_bytes_that_are_not_utf8(tmp_path: Path) -> None:
    path = tmp_path / "mixed.txt"
    path.write_bytes(b"caf\xe9 at 10.0.0.9\r\n")
    assert [f.rule for f in scan_file(path)] == ["private_ip"]


def test_example_corpus_has_no_secrets_or_internal_details() -> None:
    files = sorted(EXAMPLE_APPROVED.iterdir())
    assert files, "corpus-example/approved/ is empty"
    for path in files:
        flagged = [f for f in scan_file(path) if f.category != "personal_data"]
        assert flagged == [], f"{path.name}: {[f.display() for f in flagged]}"


def test_main_exit_code_reports_whether_anything_was_flagged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clean, dirty = tmp_path / "clean.md", tmp_path / "dirty.md"
    clean.write_text("nothing here\n")
    dirty.write_text("ping 192.168.0.7\n")

    assert main([str(clean)]) == 0
    assert main([str(clean), str(dirty)]) == 1
    assert capsys.readouterr().out == f"{dirty}:1:6 private_ip: 192.168.0.7\n"
