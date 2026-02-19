"""Check that version strings are consistent across project files.

Compares the canonical version in ``pyproject.toml`` against
``CITATION.cff`` and ``README.md`` badge, reporting any mismatches.

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

# ── Source files to compare ──────────────────────────────────────────────────

_FILES = {
    "pyproject.toml": re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE),
    "CITATION.cff": re.compile(r'^version:\s*"([^"]+)"', re.MULTILINE),
}


def _extract_version(file_name: str, pattern: re.Pattern[str]) -> str | None:
    """Return the first version match from *file_name*, or None."""
    path = PROJECT_ROOT / file_name
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = pattern.search(text)
    return m.group(1) if m else None


def main() -> int:
    """Compare versions and return 0 (aligned) or 1 (mismatch)."""
    versions: dict[str, str | None] = {}
    for name, pattern in _FILES.items():
        versions[name] = _extract_version(name, pattern)

    # Also check README badge
    readme = PROJECT_ROOT / "README.md"
    if readme.exists():
        badge = re.search(r"release-v([0-9]+\.[0-9]+\.[0-9]+)", readme.read_text(encoding="utf-8"))
        versions["README.md (badge)"] = badge.group(1) if badge else None

    print("Version alignment check")
    print("=" * 50)

    canonical = versions.get("pyproject.toml")
    all_ok = True

    for source, version in versions.items():
        status = "OK" if version == canonical else "MISMATCH"
        if version != canonical:
            all_ok = False
        print(f"  {source:30s}  {version or 'NOT FOUND':12s}  [{status}]")

    print()
    if all_ok:
        print("All version strings are aligned.")
        return 0
    else:
        print(f"Expected canonical version: {canonical}")
        print("FAIL: Version strings are not aligned.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
