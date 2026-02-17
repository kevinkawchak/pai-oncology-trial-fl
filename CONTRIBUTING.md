# Contributing to PAI Oncology Trial FL

Thank you for your interest in contributing to the PAI Oncology Trial FL
platform. This project sits at the intersection of federated learning, physical
AI, and oncology clinical trials, so contributions carry an above-average
responsibility for correctness, safety, and regulatory awareness.

Please read this guide in full before opening your first pull request.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Types of Contributions](#types-of-contributions)
- [Contribution Requirements](#contribution-requirements)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Benchmark Data Policy](#benchmark-data-policy)
- [Safety and Compliance](#safety-and-compliance)
- [Licensing](#licensing)

## Code of Conduct

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md). In
particular, note the clinical context provisions regarding patient safety, data
privacy, and scientific integrity.

## Types of Contributions

| Type | Label | Description |
|------|-------|-------------|
| Bug fix | `bug` | Fix incorrect behavior, logic errors, or crashes |
| Feature | `enhancement` | Add new functionality (module, strategy, integration) |
| Documentation | `docs` | Improve or add docstrings, guides, tutorials, or notebooks |
| Test | `test` | Add or improve unit tests, integration tests, or end-to-end tests |
| Security | `security` | Fix vulnerabilities, harden privacy or access-control code |
| Compliance | `compliance` | Update regulatory templates, compliance checks, or audit tooling |
| Example | `example` | Add or improve example scripts and notebooks |

If your contribution does not fit neatly into one of these categories, open a
discussion issue first to agree on scope and approach.

## Contribution Requirements

Every contribution, regardless of type, must meet the following requirements:

### Recency

References, techniques, and dependencies must reflect the current state of the
field. As a guideline, core algorithmic references should have been published or
updated within the **last 3 months** unless they are established foundational
works (e.g., FedAvg, differential privacy). If you cite a paper or tool, verify
that it has not been superseded.

### Oncology Relevance

This project is purpose-built for oncology clinical trials. Contributions must
be directly relevant to this domain or clearly generalizable to it. Features
that are purely generic ML infrastructure without a clear connection to
federated oncology workflows will be asked to re-scope.

### Reproducibility

All code contributions must be reproducible:

- Provide a clear description of what the code does and how to test it.
- Include or update unit tests that exercise the new or changed behavior.
- Use fixed random seeds in examples and tests where stochastic behavior is
  involved.
- Pin any new dependencies to minimum compatible versions in `pyproject.toml`.

### Cross-Platform Awareness

The platform supports simulation interoperability across NVIDIA Isaac Lab,
MuJoCo, Gazebo, and PyBullet. Contributions to `physical_ai/` modules should:

- Avoid hard dependencies on a single simulation framework.
- Use the `simulation_bridge.py` abstraction layer where applicable.
- Test with the framework detection module to verify graceful fallback when a
  framework is not installed.

## Development Workflow

### 1. Fork and Clone

```bash
# Fork via GitHub, then:
git clone https://github.com/<your-username>/pai-oncology-trial-fl.git
cd pai-oncology-trial-fl
git remote add upstream https://github.com/kevinkawchak/pai-oncology-trial-fl.git
```

### 2. Create a Feature Branch

```bash
git checkout -b <type>/<short-description>
# Examples:
#   fix/dp-budget-overflow
#   feat/scaffold-momentum
#   docs/sensor-fusion-guide
```

### 3. Set Up the Development Environment

```bash
pip install -e ".[dev]"
```

### 4. Make Your Changes

- Write clear, well-documented code.
- Add or update tests in `tests/`.
- Follow the coding standards described below.

### 5. Run Linters and Tests

```bash
# Lint and format Python code
ruff check .
ruff format --check .

# Lint YAML files (if you changed any)
yamllint .github/ docker-compose.yml

# Run the test suite
pytest tests/ -v
```

All checks must pass before opening a pull request.

### 6. Commit

Write clear, imperative-mood commit messages:

```
Add Gompertz growth model to digital twin

Implement the Gompertz differential equation for tumor volume
prediction. Add unit tests and update the digital twin docstring.
```

### 7. Open a Pull Request

- Push your branch to your fork.
- Open a PR against the `main` branch of the upstream repository.
- Fill out the pull request template completely.
- Link any related issues.
- Ensure CI passes (ruff, pytest).

A maintainer will review your PR. Please be responsive to feedback; most PRs
require at least one round of revision.

## Coding Standards

### Python Style

- **Formatter/Linter**: [Ruff](https://docs.astral.sh/ruff/) (configured in
  `pyproject.toml` and `ruff.toml`).
- **Line length**: 100 characters.
- **Target version**: Python 3.10+.
- **Imports**: Sorted by `isort` rules (handled by Ruff `I` rules).
- **Quotes**: Double quotes.
- **Type hints**: Encouraged for all public function signatures.

### Docstrings

Use Google-style docstrings for all public modules, classes, and functions:

```python
def compute_fedavg(updates: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Compute the Federated Averaging weighted sum.

    Args:
        updates: List of model parameter arrays from each client.
        weights: Relative weight for each client (typically sample counts).

    Returns:
        Weighted average of the model parameters.

    Raises:
        ValueError: If updates and weights have different lengths.
    """
```

### Testing

- Test files live in `tests/` and are named `test_<module>.py`.
- Use `pytest` fixtures for shared setup.
- Aim for clear, isolated tests that each verify one behavior.
- Include both positive and negative (error-path) tests.

### YAML

- All YAML files (CI workflows, `docker-compose.yml`, templates) must pass
  `yamllint` with default settings.

## Benchmark Data Policy

### Published Data

When a contribution references benchmark results (model accuracy, convergence
rates, privacy budgets, etc.), the data must come from one of:

1. **Reproducible experiments** run using this repository's own tooling, with
   the exact command and configuration documented.
2. **Peer-reviewed publications** with full citations.
3. **Official framework benchmarks** (e.g., MONAI, Flower) with version numbers.

### Illustrative Data

Example scripts and notebooks may use **illustrative** (synthetic) data to
demonstrate API usage. Such data must be clearly labeled as illustrative and
must not be presented as clinical evidence. Use comments or markdown cells such
as:

```python
# NOTE: This is illustrative synthetic data for demonstration purposes only.
# It does not represent real patient outcomes or clinical trial results.
```

### Prohibited Data

The following must never appear in any contribution:

- Real patient data or PHI.
- Data derived from proprietary clinical trials without explicit written
  authorization.
- Benchmark numbers that cannot be independently reproduced.

## Safety and Compliance

### Patient Safety

- Contributions to `physical_ai/` must not weaken clinical accuracy thresholds
  defined in `surgical_tasks.py` without explicit maintainer approval and a
  documented clinical rationale.
- Digital twin models must clearly document their assumptions and limitations.
- No contribution should claim clinical validity without appropriate evidence.

### Privacy and Data Protection

- Never commit real PHI, PII, or clinical trial identifiers.
- Contributions to `privacy/` modules must not reduce the strength of
  de-identification, differential privacy, or access-control mechanisms.
- If your change modifies the privacy budget, noise calibration, or secure
  aggregation protocol, include a clear explanation of the impact.

### Regulatory Awareness

- Changes to `regulatory/` modules (compliance checks, FDA submission tracking,
  templates) must reference the specific regulation or guidance document that
  motivates the change.
- Template changes (IRB, DUA, informed consent) should note which regulatory
  framework they address (HIPAA, GDPR, FDA 21 CFR Part 11, ICH-GCP).

### Security Vulnerabilities

If your contribution fixes a security vulnerability, follow the process in
[SECURITY.md](SECURITY.md). Do not open a public issue for security
vulnerabilities; use private reporting.

## Licensing

By contributing to this repository, you agree that your contributions will be
licensed under the [MIT License](LICENSE), the same license that covers the
project.

---

Thank you for contributing to PAI Oncology Trial FL. Your work helps advance
privacy-preserving, federated AI for oncology clinical trials.
