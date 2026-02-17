## Summary

<!-- Provide a concise description of the changes in this PR. -->
<!-- Link to any relevant design discussions, papers, or external references. -->



## Change Type

<!-- Check all that apply. -->

- [ ] Bug fix (non-breaking change that resolves an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Enhancement (improvement to existing functionality)
- [ ] Documentation (docstrings, guides, notebooks, or comments only)
- [ ] Refactor (code restructuring with no change in external behavior)
- [ ] Test (adding or updating tests with no production code changes)
- [ ] Compliance / regulatory (templates, compliance checks, audit tooling)

## Checklist

### Code Quality

- [ ] My code follows the project's coding standards (Ruff, 100-char lines, double quotes).
- [ ] I have added or updated type hints for all public function signatures.
- [ ] I have added or updated Google-style docstrings for all public modules, classes, and functions.
- [ ] `ruff check .` and `ruff format --check .` pass without errors.
- [ ] `yamllint` passes for any modified YAML files.

### Documentation

- [ ] I have updated the README or relevant docs if this change affects user-facing behavior.
- [ ] I have added inline comments where the logic is non-obvious.
- [ ] Any new benchmark data follows the [Benchmark Data Policy](../CONTRIBUTING.md#benchmark-data-policy) (published, reproducible, or clearly labeled as illustrative).

### Safety and Compliance

- [ ] This change does not weaken privacy protections (DP, secure aggregation, de-identification, access control).
- [ ] This change does not lower clinical accuracy thresholds without documented rationale and maintainer approval.
- [ ] No real patient data, PHI, or PII is included in this PR.
- [ ] If this PR modifies `privacy/` or `regulatory/` modules, I have described the compliance impact in the summary above.

### Testing

- [ ] I have added or updated tests that cover the new or changed behavior.
- [ ] `pytest tests/ -v` passes with all tests green.
- [ ] I have tested edge cases and error paths, not just the happy path.

## Related Issues

<!-- Link related issues below using "Closes #123" or "Relates to #456". -->
<!-- If there is no related issue, consider opening one first. -->

