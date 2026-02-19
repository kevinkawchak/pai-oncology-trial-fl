"""Centralized cryptographic key management.

Loads HMAC keys from environment variables to avoid hardcoded secrets
in source code. Falls back to development defaults when PAI_HMAC_KEY
is not set.

RESEARCH USE ONLY — Not approved for clinical deployment.
"""

from __future__ import annotations

import os


def get_hmac_key() -> bytes:
    """Load HMAC key from environment; fall back to dev default.

    In production, set the ``PAI_HMAC_KEY`` environment variable.
    The development fallback is intentionally weak and must not
    be used in any deployment handling real patient data.
    """
    key = os.environ.get("PAI_HMAC_KEY")
    if key is not None:
        return key.encode("utf-8")
    return b"pai-oncology-dev-default-key"  # dev/test default only
