"""Generate a release-metrics snapshot for the PAI Oncology Trial FL repository.

Counts Python files, lines of code, test files, Markdown files,
YAML files, directories, and other useful metrics. Output is printed
to stdout in a human-readable table and optionally as JSON (``--json``).

DISCLAIMER: RESEARCH USE ONLY — Not approved for clinical deployment
without independent validation and regulatory review.

VERSION: 0.1.0
LICENSE: MIT
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".ruff_cache", "node_modules", ".tox", ".eggs"}


def _walk_files(root: Path) -> list[Path]:
    """Return all tracked files, skipping hidden/cache dirs."""
    files: list[Path] = []
    for item in sorted(root.rglob("*")):
        if any(part in _SKIP_DIRS for part in item.parts):
            continue
        if item.is_file():
            files.append(item)
    return files


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a text file."""
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except (OSError, UnicodeDecodeError):
        return 0


def compute_metrics() -> dict[str, int | str]:
    """Compute repository-wide metrics and return as a dictionary."""
    files = _walk_files(PROJECT_ROOT)

    py_files = [f for f in files if f.suffix == ".py"]
    test_files = [f for f in py_files if f.name.startswith("test_") or "/tests/" in str(f)]
    md_files = [f for f in files if f.suffix == ".md"]
    yaml_files = [f for f in files if f.suffix in (".yaml", ".yml")]
    dirs = {f.parent for f in files}

    py_loc = sum(_count_lines(f) for f in py_files)
    test_loc = sum(_count_lines(f) for f in test_files)

    return {
        "python_files": len(py_files),
        "python_loc": py_loc,
        "test_files": len(test_files),
        "test_loc": test_loc,
        "markdown_files": len(md_files),
        "yaml_files": len(yaml_files),
        "total_files": len(files),
        "directories": len(dirs),
    }


def main() -> int:
    """Print metrics and return 0."""
    use_json = "--json" in sys.argv

    metrics = compute_metrics()

    if use_json:
        print(json.dumps(metrics, indent=2))
    else:
        print("Release Metrics Snapshot")
        print("=" * 50)
        print(f"  Python files:    {metrics['python_files']}")
        print(f"  Python LOC:      {metrics['python_loc']:,}")
        print(f"  Test files:      {metrics['test_files']}")
        print(f"  Test LOC:        {metrics['test_loc']:,}")
        print(f"  Markdown files:  {metrics['markdown_files']}")
        print(f"  YAML files:      {metrics['yaml_files']}")
        print(f"  Total files:     {metrics['total_files']}")
        print(f"  Directories:     {metrics['directories']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
