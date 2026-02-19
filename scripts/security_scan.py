"""Static security scan for common vulnerability patterns.

Scans all Python files for patterns flagged in the v0.7.0 security audit
(torch.load without weights_only, np.load with allow_pickle, bare hashlib
usage without HMAC, UTC-naive datetime.now(), and bare ``except:`` clauses).

DISCLAIMER: RESEARCH USE ONLY — Not approved for clinical deployment
without independent validation and regulatory review.

VERSION: 0.1.0
LICENSE: MIT
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Patterns to detect ───────────────────────────────────────────────────────

_CHECKS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "SEC-001: torch.load without weights_only=True",
        re.compile(r"torch\.load\((?!.*weights_only\s*=\s*True)"),
        "Use torch.load(path, weights_only=True) to prevent arbitrary code execution.",
    ),
    (
        "SEC-002: np.load with allow_pickle=True",
        re.compile(r"np\.load\([^)]*allow_pickle\s*=\s*True"),
        "Use allow_pickle=False unless pickle is explicitly required and validated.",
    ),
    (
        "SEC-003: hashlib.md5 usage",
        re.compile(r"hashlib\.md5\("),
        "Replace MD5 with SHA-256 or HMAC-SHA256 for all hashing operations.",
    ),
    (
        "SEC-004: UTC-naive datetime.now()",
        re.compile(r"datetime\.now\(\)(?!\s*#\s*noqa)"),
        "Use datetime.now(timezone.utc) for consistent UTC timestamps.",
    ),
    (
        "SEC-005: bare except clause",
        re.compile(r"^\s*except\s*:\s*$", re.MULTILINE),
        "Always catch a specific exception type (never bare 'except:').",
    ),
    (
        "SEC-006: hardcoded secret-like literal",
        re.compile(r"""(?:_KEY|_SECRET|_TOKEN|_PASSWORD)\s*[:=]\s*[br]?["'][^"']{8,}["']"""),
        "Move secrets to environment variables or a secrets manager.",
    ),
]


def scan_file(path: Path) -> list[tuple[str, int, str]]:
    """Return list of (check_id, line_number, matched_text) for a file."""
    findings: list[tuple[str, int, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return findings

    lines = text.splitlines()
    for check_id, pattern, _advice in _CHECKS:
        for i, line in enumerate(lines, start=1):
            stripped = line.lstrip()
            # Skip comments and docstrings/strings (simple heuristic)
            if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                continue
            if pattern.search(line):
                # Allow inline suppression with justification
                if "SEC-006" in check_id and "# noqa: SEC006" in line:
                    continue
                findings.append((check_id, i, line.strip()))
    return findings


def main() -> int:
    """Scan all .py files and report findings."""
    py_files = sorted(PROJECT_ROOT.rglob("*.py"))
    total_findings = 0

    print("Security pattern scan")
    print("=" * 60)
    print(f"  Scanning {len(py_files)} Python files ...\n")

    for path in py_files:
        if path.name == "security_scan.py":
            continue  # Skip self — contains pattern descriptions
        hits = scan_file(path)
        if hits:
            rel = path.relative_to(PROJECT_ROOT)
            for check_id, lineno, matched in hits:
                print(f"  {rel}:{lineno}  {check_id}")
                print(f"    {matched}\n")
                total_findings += 1

    print("-" * 60)
    if total_findings == 0:
        print("  No security findings detected.")
        return 0
    else:
        print(f"  {total_findings} finding(s) detected. Review and remediate.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
