"""Log sanitizer — strips PHI patterns from log messages.

Ensures that error logs do not accidentally leak Protected Health
Information by scrubbing common PHI patterns before emission.

RESEARCH USE ONLY — Not approved for clinical deployment.
"""

from __future__ import annotations

import re

# Patterns that may indicate PHI in log messages
_PHI_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("phone", re.compile(r"\b(?:\+1[-.\\s]?)?\(?\d{3}\)?[-.\\s]?\d{3}[-.\\s]?\d{4}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    ("MRN", re.compile(r"\bMRN[:\s#]*\d{6,10}\b", re.IGNORECASE)),
    ("IP", re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    ("date", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
]


def sanitize_log_message(message: str) -> str:
    """Remove PHI-like patterns from a log message.

    Args:
        message: Raw log message that may contain PHI.

    Returns:
        Sanitized message with PHI patterns replaced by ``[REDACTED-<type>]``.
    """
    result = message
    for phi_type, pattern in _PHI_PATTERNS:
        result = pattern.sub(f"[REDACTED-{phi_type}]", result)
    return result
