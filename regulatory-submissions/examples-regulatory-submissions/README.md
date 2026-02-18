# Regulatory Submissions Examples

[![Version](https://img.shields.io/badge/version-0.9.1-blue.svg)](../../CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../../LICENSE)

> **DISCLAIMER: RESEARCH USE ONLY** — These examples demonstrate research-grade regulatory
> submission tooling. They are not validated for actual regulatory submissions.

## Overview

Six progressive examples demonstrating the `regulatory-submissions/` module capabilities,
from basic submission workflow configuration through full end-to-end pipeline integration
with `clinical-analytics/` outputs.

## Examples

| # | File | Focus | Complexity |
|---|------|-------|-----------|
| 1 | `01_basic_submission_workflow.py` | Submission config, document registration, milestone tracking | Minimal |
| 2 | `02_ectd_compilation.py` | eCTD module structure compilation and validation | Single feature |
| 3 | `03_compliance_validation.py` | Multi-regulation compliance checking (FDA/ISO/IEC) | Multi-regulation |
| 4 | `04_document_generation.py` | Template-driven document generation (510(k), IEC 62304) | Multi-template |
| 5 | `05_regulatory_intelligence.py` | Guidance tracking, impact assessment, timelines | Intelligence |
| 6 | `06_full_submission_pipeline.py` | End-to-end pipeline with clinical-analytics integration | Full integration |

## Running

```bash
# Run individual examples
python regulatory-submissions/examples-regulatory-submissions/01_basic_submission_workflow.py

# Run all examples
for f in regulatory-submissions/examples-regulatory-submissions/0*.py; do python "$f"; done
```

## Requirements

- Python 3.10+
- numpy >= 1.24.0
- scipy >= 1.11.0

## License

MIT -- see [LICENSE](../../LICENSE).
