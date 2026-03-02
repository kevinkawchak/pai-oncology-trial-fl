# Repository Summaries (Standardized Formats)

## 1) Senior AI Engineer Standard: **ML System Card / AI Program Brief**

### Problem Definition
`pai-oncology-trial-fl` is a research-grade platform for building and validating federated learning workflows for oncology trials, with emphasis on privacy, simulation-assisted decision support, and regulatory readiness.

### System Scope
- **Core objective:** enable multi-site oncology model training and analysis without centralizing raw patient data.
- **Extended objective:** unify simulation frameworks (physical AI + digital twins + agentic orchestration) for preclinical and workflow validation.
- **Operational boundary:** repository explicitly marks modules as research/development tooling and not direct clinical deployment artifacts.

### Data + Privacy Posture
- Federated data handling includes ingestion, harmonization, and site enrollment modules.
- Privacy stack includes de-identification, PHI/PII detection, RBAC-style access control, and breach-response workflows.
- Security and privacy capabilities are reinforced with secure aggregation and differential privacy support in the federated pipeline.

### Learning/Reasoning Components
- Federated training module supports multiple aggregation strategies (FedAvg/FedProx/SCAFFOLD).
- Physical AI modules provide digital twin disease progression/treatment simulation and surgical robotics integration/sensor fusion.
- Agentic/unification modules bridge tool/protocol boundaries across AI agent frameworks and simulation ecosystems.

### Evaluation + Validation
- Broad pytest suite covers federated learning, privacy, regulatory workflows, clinical analytics, agentic AI, digital twins, tools, standards conversion, and end-to-end integration.
- Examples directory provides progressive scenario scripts for federated training, digital twin planning, cross-framework validation, and outcome prediction.

### Governance + Compliance Readiness
- Regulatory-focused components include FDA submission tracking, IRB management, ICH-GCP checks, and regulatory intelligence tracking.
- Repository includes explicit responsible-use guidance, emphasizing IRB approval, independent validation, and safety before clinical use.

### AI Peer Review Assurance
- Triple AI peer review completed (v0.9.4–v0.9.9): Codex (OpenAI) reviewed, Claude Code (Anthropic) fixed. 31/31 recommendations resolved (100%). Combined with v0.7.0 security audit, 92 total issues resolved.
- Dual-manufacturer review reduces single-AI bias and establishes a verifiable trust chain. DOI: [10.5281/zenodo.17774559](https://doi.org/10.5281/zenodo.17774559).

### Maturity Snapshot
- Package is versioned as `1.1.1` (Production/Stable classifier) with Python 3.10+ support and a modular package layout spanning federated, physical AI, privacy, regulatory, unification, clinical analytics, and regulatory submissions domains.
- 235 Python files, ~86,800 LOC, 82 test files, 105+ Markdown docs, 31 example scripts, 5 CLI tools. All 21 GitHub releases promoted to stable (non-pre-release).

---

## 2) Senior Software Engineer Standard: **C4/arc42-Style Architecture Summary**

### Context (Level 1)
This repository is a Python monorepo for oncology trial R&D that combines:
1. federated ML orchestration,
2. physical AI simulation/digital twins,
3. privacy/compliance controls,
4. regulatory submission tooling,
5. cross-framework interoperability utilities.

Primary external stakeholders are ML engineers, simulation engineers, privacy/compliance teams, and clinical/research operations.

### Containers (Level 2)
- **`federated/`**: coordinator, clients, model, secure aggregation, differential privacy, data harmonization.
- **`physical_ai/`**: digital twin modeling, surgical tasks, sensor fusion, simulation bridge, robot integration.
- **`privacy/`**: de-id, PHI detection, access control, DUA tooling, breach response.
- **`regulatory/` + `regulatory-submissions/`**: compliance checks, protocol/submission lifecycle, document generation.
- **`unification/`**: framework detection, model conversion, agent interface unification, physics mapping.
- **`clinical-analytics/`**: trial data management, risk stratification, survival/PKPD analytics, consortium reporting.
- **`tools/`**: operational CLIs (e.g., DICOM inspector, dose calculator, deployment readiness, trial site monitor).

### Components (Level 3 highlights)
- Federated coordination pipeline with pluggable algorithms and privacy controls.
- Simulation/robotics adapters for cross-platform model and state translation.
- Compliance and submission compilers for regulated-document workflows.
- Validation/conversion suites for portability and reproducibility across frameworks.

### Code & Dependency View
- Build system uses `setuptools` via `pyproject.toml`.
- Baseline dependencies emphasize numerical/ML primitives (`numpy`, `pandas`, `scikit-learn`) plus crypto/config support.
- Optional extras separate developer tooling (`pytest`, `ruff`, `pre-commit`) from docs and torch ecosystems.

### Quality Attributes
- **Security/privacy:** secure aggregation, DP, audit and de-id tooling.
- **Modularity:** clear domain directories with examples and focused tests.
- **Interoperability:** dedicated unification layer and standards artifacts.
- **Testability:** extensive domain and integration test coverage using pytest.

### Build/Run/Test Workflow
- Install with editable mode and dev extras.
- Validate environment using `scripts/verify_installation.py`.
- Execute tests through pytest configuration in `pyproject.toml` (`tests/`, `test_*.py`, verbose short tracebacks).

### Code Assurance
- **Security audit (v0.7.0):** 61 findings remediated (12 security, 14 logic, 35 compliance).
- **Triple AI peer review (v0.9.4–v0.9.9):** 31 recommendations from Codex (OpenAI), all 31 implemented by Claude Code (Anthropic). 100% resolution rate across 3 cycles.
- **Shared utilities (v0.9.7–v0.9.9):** `utils/` package provides environment-driven HMAC keys, UTC timestamps, PHI-safe logging, centralized config, and structured error codes.
- **CI validation:** ruff check + ruff format across Python 3.10/3.11/3.12 matrix, yamllint, version alignment check, security scan.

### Risks / Technical Debt (visible from structure)
- Large monorepo breadth increases coupling risk across regulated/privacy/simulation domains.
- Presence of legacy and newer parallel modules in privacy/regulatory areas suggests ongoing migration burden.
- Advanced simulation/agentic integrations likely have optional dependency and environment complexity not required for core federated paths.
