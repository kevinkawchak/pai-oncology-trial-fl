"""Tests for tools/deployment-readiness/deployment_readiness.py.

Covers enums, dataclasses, checklist building, check evaluation,
report generation, export, and CLI helpers.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict

import pytest

from tests.conftest import load_module

mod = load_module("deployment_readiness", "tools/deployment-readiness/deployment_readiness.py")


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestCheckStatus:
    """Tests for the CheckStatus enum."""

    def test_members(self):
        """CheckStatus has four expected members."""
        expected = {"PASS", "FAIL", "WARNING", "REQUIRES_VERIFICATION"}
        assert set(mod.CheckStatus.__members__.keys()) == expected

    def test_values(self):
        """CheckStatus values match expected strings."""
        assert mod.CheckStatus.PASS.value == "pass"
        assert mod.CheckStatus.FAIL.value == "fail"
        assert mod.CheckStatus.WARNING.value == "warning"
        assert mod.CheckStatus.REQUIRES_VERIFICATION.value == "requires_verification"


class TestDeploymentDecision:
    """Tests for the DeploymentDecision enum."""

    def test_members(self):
        """DeploymentDecision has three expected members."""
        expected = {"APPROVED", "CONDITIONAL", "BLOCKED"}
        assert set(mod.DeploymentDecision.__members__.keys()) == expected

    def test_values(self):
        """DeploymentDecision values match expected strings."""
        assert mod.DeploymentDecision.APPROVED.value == "approved"
        assert mod.DeploymentDecision.BLOCKED.value == "blocked"


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestReadinessCheck:
    """Tests for the ReadinessCheck dataclass."""

    def test_defaults(self):
        """Default ReadinessCheck has requires_verification status."""
        rc = mod.ReadinessCheck()
        assert rc.status == mod.CheckStatus.REQUIRES_VERIFICATION.value
        assert rc.check_id == ""

    def test_custom_values(self):
        """ReadinessCheck accepts custom values."""
        rc = mod.ReadinessCheck(
            check_id="MV-001",
            category="model_validation",
            name="AUC check",
            status=mod.CheckStatus.PASS.value,
        )
        assert rc.check_id == "MV-001"
        assert rc.status == "pass"


class TestReadinessReport:
    """Tests for the ReadinessReport dataclass."""

    def test_defaults(self):
        """Default ReadinessReport is BLOCKED."""
        rr = mod.ReadinessReport()
        assert rr.decision == mod.DeploymentDecision.BLOCKED.value
        assert rr.total_checks == 0
        assert rr.checks == []

    def test_asdict(self):
        """ReadinessReport can be serialized."""
        rr = mod.ReadinessReport(model_name="test", model_version="1.0")
        d = asdict(rr)
        assert d["model_name"] == "test"
        assert d["model_version"] == "1.0"


# ---------------------------------------------------------------------------
# Checklist building
# ---------------------------------------------------------------------------
class TestBuildChecklist:
    """Tests for build_checklist."""

    def test_all_categories(self):
        """Building with all categories returns 32 checks."""
        checks = mod.build_checklist()
        assert len(checks) == 32

    def test_single_category(self):
        """Building a single category returns only those checks."""
        checks = mod.build_checklist(categories=["model_validation"])
        assert len(checks) == 7
        assert all(c.category == "model_validation" for c in checks)

    def test_multiple_categories(self):
        """Building multiple categories returns combined count."""
        checks = mod.build_checklist(categories=["safety_compliance", "regulatory"])
        assert len(checks) == 12

    def test_unknown_category_empty(self):
        """Unknown category produces no checks."""
        checks = mod.build_checklist(categories=["nonexistent"])
        assert len(checks) == 0

    def test_check_has_source(self):
        """Each check has a non-empty source field."""
        checks = mod.build_checklist(categories=["model_validation"])
        for c in checks:
            assert c.source != ""


# ---------------------------------------------------------------------------
# Check evaluation
# ---------------------------------------------------------------------------
class TestEvaluateChecks:
    """Tests for evaluate_checks."""

    def test_no_overrides_all_require_verification(self):
        """Without overrides all checks are requires_verification."""
        checks = mod.build_checklist(categories=["documentation"])
        evaluated = mod.evaluate_checks(checks)
        assert all(c.status == "requires_verification" for c in evaluated)

    def test_override_pass(self):
        """Override sets status to pass."""
        checks = mod.build_checklist(categories=["documentation"])
        overrides = {checks[0].check_id: "pass"}
        evaluated = mod.evaluate_checks(checks, overrides=overrides)
        assert evaluated[0].status == "pass"
        assert evaluated[0].verified_at != ""

    def test_override_fail(self):
        """Override sets status to fail."""
        checks = mod.build_checklist(categories=["documentation"])
        overrides = {checks[0].check_id: "fail"}
        evaluated = mod.evaluate_checks(checks, overrides=overrides)
        assert evaluated[0].status == "fail"

    def test_invalid_override_resets(self):
        """Invalid override status resets to requires_verification."""
        checks = mod.build_checklist(categories=["documentation"])
        overrides = {checks[0].check_id: "bogus_status"}
        evaluated = mod.evaluate_checks(checks, overrides=overrides)
        assert evaluated[0].status == "requires_verification"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
class TestGenerateReadinessReport:
    """Tests for generate_readiness_report."""

    def test_all_pass_approved(self):
        """All checks passing yields APPROVED."""
        checks = mod.build_checklist(categories=["documentation"])
        overrides = {c.check_id: "pass" for c in checks}
        evaluated = mod.evaluate_checks(checks, overrides=overrides)
        report = mod.generate_readiness_report("model-v1", "1.0.0", evaluated)
        assert report.decision == "approved"
        assert report.failed == 0
        assert report.passed == len(evaluated)

    def test_any_fail_blocked(self):
        """Any failing check yields BLOCKED."""
        checks = mod.build_checklist(categories=["documentation"])
        overrides = {checks[0].check_id: "fail"}
        evaluated = mod.evaluate_checks(checks, overrides=overrides)
        report = mod.generate_readiness_report("model-v1", "1.0.0", evaluated)
        assert report.decision == "blocked"
        assert report.failed >= 1

    def test_warnings_conditional(self):
        """Warnings (no fails) yield CONDITIONAL."""
        checks = mod.build_checklist(categories=["documentation"])
        overrides = {c.check_id: "pass" for c in checks}
        overrides[checks[0].check_id] = "warning"
        evaluated = mod.evaluate_checks(checks, overrides=overrides)
        report = mod.generate_readiness_report("model-v1", "1.0.0", evaluated)
        assert report.decision == "conditional"

    def test_unverified_conditional(self):
        """Unverified checks yield CONDITIONAL."""
        checks = mod.build_checklist(categories=["documentation"])
        evaluated = mod.evaluate_checks(checks)
        report = mod.generate_readiness_report("model-v1", "1.0.0", evaluated)
        assert report.decision == "conditional"
        assert report.requires_verification > 0

    def test_summary_by_category(self):
        """Report includes per-category summary."""
        checks = mod.build_checklist()
        evaluated = mod.evaluate_checks(checks)
        report = mod.generate_readiness_report("model-v1", "1.0.0", evaluated)
        assert "model_validation" in report.summary_by_category
        assert "safety_compliance" in report.summary_by_category

    def test_report_id_generated(self):
        """Report ID starts with 'DR-'."""
        checks = mod.build_checklist(categories=["documentation"])
        evaluated = mod.evaluate_checks(checks)
        report = mod.generate_readiness_report("m", "1.0", evaluated)
        assert report.report_id.startswith("DR-")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
class TestExportReport:
    """Tests for export_report."""

    def test_export_json(self):
        """Export writes valid JSON to the specified path."""
        checks = mod.build_checklist(categories=["documentation"])
        evaluated = mod.evaluate_checks(checks)
        report = mod.generate_readiness_report("m", "1.0", evaluated)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result_path = mod.export_report(report, path)
            assert result_path == path
            with open(path) as fh:
                data = json.load(fh)
            assert data["model_name"] == "m"
        finally:
            os.unlink(path)

    def test_export_unsupported_format(self):
        """Unsupported export format raises ValueError."""
        report = mod.ReadinessReport()
        with pytest.raises(ValueError, match="Unsupported"):
            mod.export_report(report, "/tmp/test.xml", fmt="xml")
