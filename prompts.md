# Prompts Used in Development

This document records the prompts used to develop the PAI Oncology Trial FL platform.

---

## Prompt 1: v0.1.0 — Integration and Implementation

```
Your goal is to integrate physical ai with the following federated learning platform instructions below into kevinkawchak/pai-oncology-trial-fl. Utilize kevinkawchak/physical-ai-oncology-trials for assistance regarding physical ai implementation and also for how to simultaneously unify the new field. Target the product as an aid for the customer, and not how the user can make money. The GitHub repository should result in a sellable product that is comprehensive and addresses key unmet needs for performing federated learning alongside robots in upcoming oncology clinical trials. Make sure the Readme and other files under main are complete, such as changelog.md (v0.1.0). Also include a prompts.md that includes the prompt(s) used in this conversation.

Be sure to fix and address errors that would cause failed checks for the pull request (such as Python environment issues to avoid the following error during final checks): "3 failing checks
x Cl / lint-and-format (3.10) (pull...
x Cl / lint-and-format (3.11) (pull...
x Cl / lint-and-format (3.12) (pull... " When you are finished, provide a list of new additions and what changed from old to new files. The user will then review your lists prior to committing changes. Separately, at the end of this conversation provide new release notes (v0.1.0) in the format below (keep release notes separate from the pull request, and keep hashtag characters as is, not formatting the text bold).
```

### Federated Learning Platform Instructions

```
Concept: A Federated Learning platform for oncology that unifies data ingestion, privacy, and multi-institution collaboration. It lets multiple hospitals or research centers jointly train AI models (e.g. tumor progression, imaging analysis) without exchanging raw patient data. This product builds on the repository's federation/ and privacy/ modules. It includes secure aggregation, differential privacy, consent management (DUAs), and monitoring.

Claude Code Prompt:
From the kevinkawchak/pai-oncology-trial-fl repo implement a federated machine learning framework for physical ai oncology trials. The repo should include:
- A Python module `federated/` with submodules: `coordinator.py` (orchestrates training rounds), `client.py` (simulated hospital nodes), `secure_aggregation.py`, and `differential_privacy.py`.
- Data ingestion scripts to load synthetic oncology datasets (e.g. tumor images or patient records) and split them across sites.
- Privacy modules for PHI de-identification (reuse phi_detector/deidentification code).
- CI/CD setup (GitHub Actions) running unit tests and code style checks.
- Automated tests in `tests/` verifying model training works across two or more simulated sites.
- Jupyter notebooks in `docs/` demonstrating federated training of a simple model (e.g. tumor classification) with at least two clients.
- Example data or scripts to generate dummy data.
- A deployment script or docker-compose to launch federated learning simulation.
- Detailed `README.md` explaining usage, architecture, and installation.
Ensure reproducibility (Conda env or Dockerfile), include license (MIT), and a descriptive CITATION.cff.
Deliverables & Acceptance Criteria:
* Code Modules: A working physical ai federated learning codebase (Python) with modules for coordinator, clients, privacy, and secure aggregation. The code should build on concepts from federation/.
* CI/CD: GitHub Actions that automatically lint (e.g. flake8 or ruff), test federation logic, and verify example runs.
* Tests: Unit tests in tests/ confirming that model parameters update correctly across sites and privacy safeguards work. For example, a test that adding Gaussian noise in differential_privacy.py still yields a reasonable aggregated model.
* Documentation/Examples: Jupyter notebooks and examples (in docs/ or examples-federation/) that show how two or more synthetic "site" clients coordinate to train a model, with scripts for data splitting.
* Containers: Dockerfile or Conda environment supporting federated execution across containers or processes, with instructions for setup.
* Compliance Components: Example consent forms or DUA templates (leveraging privacy/dua-templates/) that accompany the federated process.
* Acceptance: The platform must run end-to-end: simulate at least two clients joining a study, train an AI model (e.g. CNN classifier), and produce aggregate model accuracy comparable to centralized training. Privacy checks (no raw data sharing) should be enforced. Documentation must guide a user through setup and interpretation of results.
Effort & Expertise: High. Requires expertise in machine learning, physical ai, federated algorithms, security, and healthcare data compliance. Development involves advanced topics like distributed systems and privacy engineering. Estimate ~3-6 person-months for an initial MVP.
Monetization & Customers: Target customers include research consortia, hospitals, and pharma companies collaborating on oncology AI models. Monetization could be via enterprise licensing or cloud-based SaaS (managed federated training). Premium features (advanced analytics, support) could be add-ons. The product leverages open MIT-licensed core but charges for value-added services (certifications, deployments).
Regulatory & Ethical: Must adhere to HIPAA/GDPR: all patient data must be de-identified and encrypted in transit. Differential privacy or secure aggregation should ensure no individual can be re-identified. Ethical concerns include data bias: consortium should audit model fairness across subgroups. Regulatory docs (e.g. IRB approvals) may be needed before federated trials. The product should include logging and audit trails (perhaps leveraging privacy/breach-response/).
```

---

## Prompt 2: v0.2.0 — Comprehensive Enhancement

```
Your goal is to make the existing kevinkawchak/pai-oncology-trial-fl more comprehensive. Utilize kevinkawchak/physical-ai-oncology-trials for assistance regarding physical ai implementation and also for how to simultaneously unify the new field. Make sure no duplicate and/or repetitive information is used from the other repo. Code should be longer, have greater context, and be more comprehensive. Add, modify, or remove code as needed. Target the product as an aid for the customer, and not how the user can make money. The GitHub repository should result in a sellable end to end product that is substantial and addresses key unmet needs for performing federated learning alongside robots in upcoming oncology clinical trials. Make sure the Readme and other files under main are complete, such as changelog.md (v0.2.0). Also update the prompts.md that includes the prompt(s) used in this conversation.

Be sure to fix and address errors that would cause failed checks for the pull request (such as Python environment issues to avoid the following error during final checks): "3 failing checks
x Cl / lint-and-format (3.10) (pull...
x Cl / lint-and-format (3.11) (pull...
x Cl / lint-and-format (3.12) (pull... " When you are finished, provide a list of new additions and what changed from old to new files. The user will then review your lists prior to committing changes. Separately, at the end of this conversation provide new release notes (v0.2.0) in the format below (keep release notes separate from the pull request, and keep hashtag characters as is, not formatting the text bold).
```

### Federated Learning Platform Instructions (Same as v0.1.0)

```
Concept: A Federated Learning platform for oncology that unifies data ingestion, privacy, and multi-institution collaboration. It lets multiple hospitals or research centers jointly train AI models (e.g. tumor progression, imaging analysis) without exchanging raw patient data. This product builds on the repository's federation/ and privacy/ modules. It includes secure aggregation, differential privacy, consent management (DUAs), and monitoring.

Claude Code Prompt:
From the "pai-oncology-trial-fl" repo implement a federated machine learning framework for physical ai oncology trials. The repo should include:
- A Python module `federated/` with submodules: `coordinator.py` (orchestrates training rounds), `client.py` (simulated hospital nodes), `secure_aggregation.py`, and `differential_privacy.py`.
- Data ingestion scripts to load synthetic oncology datasets (e.g. tumor images or patient records) and split them across sites.
- Privacy modules for PHI de-identification (reuse phi_detector/deidentification code).
- CI/CD setup (GitHub Actions) running unit tests and code style checks.
- Automated tests in `tests/` verifying model training works across two or more simulated sites.
- Jupyter notebooks in `docs/` demonstrating federated training of a simple model (e.g. tumor classification) with at least two clients.
- Example data or scripts to generate dummy data.
- A deployment script or docker-compose to launch federated learning simulation.
- Detailed `README.md` explaining usage, architecture, and installation.
Ensure reproducibility (Conda env or Dockerfile), include license (MIT), and a descriptive CITATION.cff.
Deliverables & Acceptance Criteria:
* Code Modules: A working physical ai federated learning codebase (Python) with modules for coordinator, clients, privacy, and secure aggregation. The code should build on concepts from federation/.
* CI/CD: GitHub Actions that automatically lint (e.g. flake8 or ruff), test federation logic, and verify example runs.
* Tests: Unit tests in tests/ confirming that model parameters update correctly across sites and privacy safeguards work. For example, a test that adding Gaussian noise in differential_privacy.py still yields a reasonable aggregated model.
* Documentation/Examples: Jupyter notebooks and examples (in docs/ or examples-federation/) that show how two or more synthetic "site" clients coordinate to train a model, with scripts for data splitting.
* Containers: Dockerfile or Conda environment supporting federated execution across containers or processes, with instructions for setup.
* Compliance Components: Example consent forms or DUA templates (leveraging privacy/dua-templates/) that accompany the federated process.
* Acceptance: The platform must run end-to-end: simulate at least two clients joining a study, train an AI model (e.g. CNN classifier), and produce aggregate model accuracy comparable to centralized training. Privacy checks (no raw data sharing) should be enforced. Documentation must guide a user through setup and interpretation of results.
Effort & Expertise: High. Requires expertise in machine learning, physical ai, federated algorithms, security, and healthcare data compliance. Development involves advanced topics like distributed systems and privacy engineering. Estimate ~3–6 person-months for an initial MVP.
Monetization & Customers: Target customers include research consortia, hospitals, and pharma companies collaborating on oncology AI models. Monetization could be via enterprise licensing or cloud-based SaaS (managed federated training). Premium features (advanced analytics, support) could be add-ons. The product leverages open MIT-licensed core but charges for value-added services (certifications, deployments).
Regulatory & Ethical: Must adhere to HIPAA/GDPR: all patient data must be de-identified and encrypted in transit. Differential privacy or secure aggregation should ensure no individual can be re-identified. Ethical concerns include data bias: consortium should audit model fairness across subgroups. Regulatory docs (e.g. IRB approvals) may be needed before federated trials. The product should include logging and audit trails (perhaps leveraging privacy/breach-response/).
```
