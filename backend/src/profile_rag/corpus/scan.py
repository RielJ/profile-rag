"""Flag text that may be a secret or personal data, for a human to review.

This is a reviewer's aid, not a gate. It looks for patterns (email
addresses, phone numbers, API keys, private IP addresses, internal
hostnames) and reports where they are. It cannot know whether a match is
fine to publish: your own public email on a resume is, a colleague's phone
number is not. So it flags; it never approves, and an empty result does not
mean a source is safe. The review checklist (`profile-rag corpus review`) still applies.

`profile-rag corpus review` runs it on the snapshot before asking for a
decision. To scan any file by hand:

    uv --directory backend run python -m profile_rag.corpus.scan FILE...
"""

import ipaddress
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Category = Literal["personal_data", "secret", "internal"]


@dataclass(frozen=True)
class Finding:
    """One match. `line` and `column` are 1-based."""

    rule: str
    category: Category
    line: int
    column: int
    match: str

    def display(self) -> str:
        """The match as safe to print: secrets show only their first 4 characters."""
        shown = f"{self.match[:4]}…" if self.category == "secret" else self.match
        return f"{self.line}:{self.column} {self.rule}: {shown}"


@dataclass(frozen=True)
class _Rule:
    name: str
    category: Category
    pattern: re.Pattern[str]


_RULES: tuple[_Rule, ...] = (
    _Rule(
        "email",
        "personal_data",
        # The top-level domain must be letters, so `user:pass@127.0.0.1` in a
        # connection string is not reported as an email (it is reported below).
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
    ),
    _Rule(
        "phone",
        "personal_data",
        re.compile(
            r"(?<![\w+])(?:"
            # International: +63 917 555 0100, +1-555-010-0199
            r"\+\d{1,3}[ .-]?\(?\d{1,4}\)?(?:[ .-]?\d{2,4}){2,4}"
            # North American: (555) 010-0100, 555.010.0100
            r"|\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}"
            # Local with a trunk prefix: 0917 555 0100
            r"|0\d{3}[ .-]?\d{3}[ .-]?\d{4}"
            r")(?![\w])"
        ),
    ),
    _Rule(
        "private_key",
        "secret",
        re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"),
    ),
    _Rule("aws_access_key", "secret", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    _Rule(
        "github_token",
        "secret",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})\b"),
    ),
    _Rule("slack_token", "secret", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    _Rule("api_key", "secret", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}")),
    _Rule(
        "jwt",
        "secret",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ),
    _Rule(
        "credential_assignment",
        "secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*['\"]?[^\s'\"]{8,}"
        ),
    ),
    _Rule(
        "url_with_credentials",
        "secret",
        re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@[^\s/]+"),
    ),
    _Rule(
        "internal_hostname",
        "internal",
        re.compile(
            # Also inside longer names: build.corp.example.com is flagged.
            r"(?i)\b[a-z0-9-]+(?:\.[a-z0-9-]+)*?\.(?:internal|corp|lan|local|intranet)\b"
        ),
    ),
)

_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")

# RFC 1918 private ranges plus 100.64.0.0/10 (carrier-grade NAT, also used by
# Tailscale). Loopback (127.0.0.0/8) is left out: `127.0.0.1:8000` in a README
# reveals nothing.
_PRIVATE_NETWORKS = tuple(
    ipaddress.IPv4Network(network)
    for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10")
)


def _is_private_ipv4(text: str) -> bool:
    try:
        address = ipaddress.IPv4Address(text)
    except ipaddress.AddressValueError:
        return False
    return any(address in network for network in _PRIVATE_NETWORKS)


def scan_text(text: str) -> list[Finding]:
    """Return every finding in `text`, ordered by position."""
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for rule in _RULES:
            for match in rule.pattern.finditer(line):
                findings.append(
                    Finding(rule.name, rule.category, number, match.start() + 1, match.group())
                )
        for match in _IPV4.finditer(line):
            if _is_private_ipv4(match.group()):
                findings.append(
                    Finding("private_ip", "internal", number, match.start() + 1, match.group())
                )
    return sorted(findings, key=lambda f: (f.line, f.column, f.rule))


def scan_file(path: Path) -> list[Finding]:
    """Scan a file's text. Bytes that are not UTF-8 are replaced, not fatal."""
    return scan_text(path.read_bytes().decode("utf-8", errors="replace"))


def main(argv: list[str]) -> int:
    """Print findings for each file. Exit 1 if anything was flagged."""
    flagged = False
    for name in argv:
        for finding in scan_file(Path(name)):
            flagged = True
            print(f"{name}:{finding.display()}")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
