# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Federated machine learning framework for physical AI oncology trials. Python monorepo (Python 3.10+) combining federated ML orchestration, physical AI simulation/digital twins, privacy/compliance controls, regulatory submission tooling, and cross-framework interoperability. **Research use only** — not for clinical deployment without independent validation.

## Build & Install

```bash
pip install -e ".[dev]"          # editable install with dev tools (pytest, ruff, yamllint)
pip install -e ".[all]"          # dev + docs extras
python scripts/verify_installation.py  # validate environment
```

## Lint & Format

Lint tool versions are pinned in `constraints/lint.txt` (ruff==0.15.1, yamllint==1.38.0). CI enforces these exact versions across Python 3.10/3.11/3.12.

```bash
ruff check .                     # lint (errors = CI failure)
ruff check . --fix               # auto-fix
ruff format --check .            # format check (errors = CI failure)
ruff format .                    # auto-format
yamllint -d relaxed configs/ .github/workflows/ unification/ q1-2026-standards/  # YAML lint (warnings only, non-blocking)
```

Ruff config is in `ruff.toml` (line-length=120, target py310, selects E/F/W). Many per-file ignores for F401/F811 across example and domain directories.

## Tests

```bash
pytest tests/ -v --tb=short      # full suite
pytest tests/test_client.py      # single file
pytest tests/test_client.py::test_function_name -v  # single test
```

CI runs tests only after lint-and-format passes (the `test` job has `needs: lint-and-format`).

## CI Workflow (.github/workflows/ci.yml)

Three jobs run on push to main/master/claude/* and on PRs:

1. **lint-and-format** (matrix: 3.10, 3.11, 3.12) — `ruff check`, `ruff format --check`, `yamllint`
2. **validate-scripts** (3.10 only) — `py_compile scripts/verify_installation.py`, `python scripts/check_version_alignment.py`
3. **test** (matrix: 3.10, 3.11, 3.12, needs lint-and-format) — `pip install -r requirements.txt && pip install -e . && pytest tests/ -v --tb=short`

## Version Management

The canonical version lives in `pyproject.toml`. These files must stay aligned:

- `pyproject.toml` — `version = "X.Y.Z"`
- `CITATION.cff` — `version: "X.Y.Z"` and `date-released`
- `README.md` — badge `release-vX.Y.Z` and bibtex citation block
- `federated/__init__.py` — `__version__ = "X.Y.Z"`
- `docs/repository_summaries.md` — maturity snapshot version

Run `python scripts/check_version_alignment.py` to verify alignment (checks pyproject.toml, CITATION.cff, README.md badge).

## Architecture

Domain modules (each a top-level directory with `__init__.py`):
- **federated/** — coordinator, clients, model, secure aggregation, differential privacy, data harmonization, site enrollment
- **physical_ai/** — digital twin modeling, surgical tasks, sensor fusion, simulation bridge
- **privacy/** — de-identification, PHI detection, access control, DUA tooling, breach response (subdirs: `core/`, `access-control/`, `compliance/`)
- **regulatory/** + **regulatory-submissions/** — compliance checks, protocol/submission lifecycle, document generation
- **unification/** — framework detection, model conversion, agent interface unification, physics mapping
- **clinical-analytics/** — trial data management, risk stratification, survival/PKPD analytics
- **utils/** — shared utilities (HMAC keys, timestamps, PHI-safe logging, config, error codes)

Supporting directories:
- **tools/** — CLI tools (DICOM inspector, dose calculator, deployment readiness, trial site monitor, MCP server)
- **examples/** and **examples-physical-ai/** — progressive scenario scripts
- **tests/** — mirrors domain structure (test_federated/, test_privacy/, test_regulatory/, etc.)
- **scripts/** — CI helpers (version alignment check, security scan, release metrics, lint runner)
- **constraints/** — pinned lint tool versions for CI reproducibility
- **releases.md/** — per-version release notes (one .md file per release)

## GitHub Limitations

Claude Code in this environment does not have write access to the GitHub API. Git push works through the local proxy, but creating PRs, editing releases, or commenting on issues requires the repository owner. When tasks need GitHub write operations, push the branch and provide the user with the compare URL or `gh` CLI commands to run manually.

## Release Workflow

Each release adds: version bump across all files above, `CHANGELOG.md` entry, `releases.md/vX.Y.Z.md` release notes, prompt documentation in `prompts.md`. The release notes format uses no hash for the title, `##` for Summary/Features/Contributors/Notes sections.
