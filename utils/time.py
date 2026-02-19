"""Shared UTC timestamp utilities.

Provides consistent ISO 8601 UTC timestamp formatting across all
modules, eliminating format drift between subsystems.

RESEARCH USE ONLY — Not approved for clinical deployment.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Return the current UTC time as a full ISO 8601 string with offset.

    Example: ``2026-02-19T14:30:00+00:00``
    """
    return datetime.now(timezone.utc).isoformat()


def utc_now_date() -> str:
    """Return today's UTC date in ISO 8601 format.

    Example: ``2026-02-19``
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
