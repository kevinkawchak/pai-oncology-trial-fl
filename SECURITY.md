# Security Policy

PAI Oncology Trial FL provides privacy-preserving federated learning
infrastructure for oncology clinical trials. Because the platform is designed to
operate in environments where protected health information (PHI) and sensitive
clinical data may be present, we treat security with the highest priority.

## Scope

This security policy covers vulnerabilities in the code, configurations, and
documentation published in the
[pai-oncology-trial-fl](https://github.com/kevinkawchak/pai-oncology-trial-fl)
repository. This includes, but is not limited to:

- Federated learning coordination and aggregation (`federated/`)
- Differential privacy and secure aggregation (`federated/`, `privacy/`)
- PHI detection, de-identification, and access control (`privacy/`)
- Audit logging and breach response (`privacy/`)
- Regulatory compliance checks and FDA submission tracking (`regulatory/`)
- Simulation bridge and physical AI modules (`physical_ai/`)
- CI/CD workflows, Docker configurations, and deployment scripts
- Example scripts and documentation that could lead to insecure usage patterns

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

We offer two private reporting channels:

### 1. Email

Send a detailed report to **kevin@ceodatascience.com** with the subject line:

```
[SECURITY] pai-oncology-trial-fl: <brief description>
```

### 2. GitHub Private Vulnerability Reporting

Use GitHub's built-in private vulnerability reporting feature:

1. Navigate to the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Fill out the advisory form with as much detail as possible.

### What to Include

- A clear description of the vulnerability and its potential impact.
- The affected module(s) and version(s).
- Steps to reproduce the issue, including code snippets or configuration if
  applicable.
- Any suggested fix or mitigation, if you have one.
- Your contact information for follow-up (optional, but appreciated).

### What to Expect

| Milestone | Target |
|-----------|--------|
| Acknowledgment of report | **7 calendar days** |
| Initial assessment and severity classification | 14 calendar days |
| Fix developed and tested | **30 calendar days** |
| Patch release published | Within 7 days of fix verification |
| Public disclosure (coordinated) | After patch release, or 90 days from report, whichever comes first |

If a reported vulnerability is determined to affect patient safety or could lead
to PHI exposure, we will prioritize it above the standard timeline and aim for
resolution within 14 calendar days.

We will keep you informed of progress throughout the process. If you do not
receive an acknowledgment within 7 days, please follow up via the alternate
reporting channel.

## Supported Versions

| Version | Supported | Notes |
|---------|-----------|-------|
| 0.3.x | Yes | Current development branch; receives all security fixes |
| 0.2.x | Yes | Latest stable release; receives critical and high-severity fixes |
| 0.1.x | Limited | Receives fixes only for critical vulnerabilities with PHI exposure risk |
| < 0.1.0 | No | Unsupported; please upgrade |

We strongly recommend running the latest patch release of a supported version.

## Deployer Responsibilities

PAI Oncology Trial FL provides tooling and infrastructure for federated learning
in clinical trial environments, but **deployers bear ultimate responsibility**
for the security and compliance of their specific deployment. The following
areas require deployer attention:

### Protected Health Information (PHI)

- The platform includes PHI detection (`privacy/phi_detector.py`) and
  de-identification (`privacy/deidentification.py`) modules. These tools are
  provided as aids and **do not guarantee** complete PHI removal in all cases.
- Deployers must validate de-identification results against their own data and
  institutional policies before using outputs in any context where PHI
  restrictions apply.
- Deployers are responsible for ensuring that raw patient data never leaves
  institutional boundaries, consistent with the federated architecture.

### Regulatory Compliance

- The compliance checker (`regulatory/compliance_checker.py`) and FDA submission
  tracker (`regulatory/fda_submission.py`) are informational tools. They do not
  constitute legal or regulatory advice.
- Deployers must engage qualified legal and regulatory professionals to validate
  compliance with HIPAA, GDPR, FDA 21 CFR Part 11, ICH-GCP, and any other
  applicable regulations.
- IRB protocol templates and informed consent templates are starting points, not
  finalized documents. Institutional review and approval are required.

### Infrastructure Security

- Deployers are responsible for securing the infrastructure on which the
  platform runs, including but not limited to:
  - Network security (TLS for all model parameter transport).
  - Authentication and authorization for coordinator and site endpoints.
  - Encryption at rest for any persisted model parameters, audit logs, or
    configuration files.
  - Container security for Docker-based deployments.
  - Access control to the deployment environment itself.

### Dependency Monitoring

- Deployers should monitor dependencies for known vulnerabilities using tools
  such as `pip-audit`, `safety`, Dependabot, or equivalent.
- The project's `pyproject.toml` pins minimum versions but does not pin maximum
  versions. Deployers should test upgrades in a staging environment before
  applying them to production.
- Critical dependencies to monitor include:
  - `cryptography` (used for secure aggregation and hashing)
  - `numpy` and `scikit-learn` (used in model training and evaluation)
  - `pyyaml` (used for configuration parsing)

## Security-Related Modules

For reference, the following modules contain security-critical functionality:

| Module | Function |
|--------|----------|
| `federated/secure_aggregation.py` | Mask-based secure aggregation protocol |
| `federated/differential_privacy.py` | Gaussian mechanism with budget tracking |
| `privacy/phi_detector.py` | PHI detection for 18 HIPAA identifiers |
| `privacy/deidentification.py` | De-identification pipeline |
| `privacy/access_control.py` | Role-based access control (RBAC) |
| `privacy/audit_logger.py` | Tamper-evident audit logging |
| `privacy/breach_response.py` | Breach detection and incident response |
| `regulatory/compliance_checker.py` | HIPAA/GDPR/FDA compliance validation |

Changes to these modules receive additional scrutiny during code review.

## Acknowledgments

We appreciate the security research community's efforts in responsibly
disclosing vulnerabilities. Contributors who report valid security issues will
be acknowledged in the release notes (with their permission) and in this
document.

## Contact

Security reports: **kevin@ceodatascience.com**

For non-security bugs and feature requests, please use the
[issue tracker](https://github.com/kevinkawchak/pai-oncology-trial-fl/issues).
