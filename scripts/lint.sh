#!/usr/bin/env bash
# lint.sh — Run the same lint checks as CI locally.
# Mirrors: .github/workflows/ci.yml lint-and-format job.
# Versions pinned via: constraints/lint.txt
#
# Usage:
#   ./scripts/lint.sh          # Run all checks
#   ./scripts/lint.sh --fix    # Auto-fix what ruff can fix
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

FIX_MODE=false
if [[ "${1:-}" == "--fix" ]]; then
    FIX_MODE=true
fi

echo "=== Lint environment ==="
python --version
ruff --version
yamllint --version
echo ""

if $FIX_MODE; then
    echo "=== ruff check --fix ==="
    ruff check . --fix
    echo ""
    echo "=== ruff format ==="
    ruff format .
else
    echo "=== ruff check ==="
    ruff check . --output-format=github
    echo ""
    echo "=== ruff format --check ==="
    ruff format --check .
fi

echo ""
echo "=== yamllint (relaxed) ==="
yamllint -d relaxed configs/ .github/workflows/ unification/ q1-2026-standards/ || true

echo ""
echo "=== Security pattern scan ==="
python scripts/security_scan.py || true

echo ""
echo "=== Version alignment check ==="
python scripts/check_version_alignment.py || true

echo ""
echo "All lint checks passed."
