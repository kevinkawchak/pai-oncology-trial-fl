"""Tests for privacy/access_control.py — Role-based access control.

Covers AccessControlManager creation, user registration, role
assignment, permission checks, extra/denied permissions, audit log
immutability (returned copy), deactivated user denial, and the
Role/Permission enum structures.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module("privacy.access_control", "privacy/access_control.py")

AccessControlManager = mod.AccessControlManager
Role = mod.Role
Permission = mod.Permission
UserProfile = mod.UserProfile
ROLE_PERMISSIONS = mod.ROLE_PERMISSIONS


class TestRoleEnum:
    """Tests for the Role enum."""

    def test_all_roles_exist(self):
        """All six roles are defined."""
        assert Role.COORDINATOR.value == "coordinator"
        assert Role.SITE_ADMIN.value == "site_admin"
        assert Role.RESEARCHER.value == "researcher"
        assert Role.DATA_ENGINEER.value == "data_engineer"
        assert Role.AUDITOR.value == "auditor"
        assert Role.PATIENT.value == "patient"

    def test_role_from_string(self):
        """Roles can be constructed from string values."""
        assert Role("coordinator") == Role.COORDINATOR


class TestPermissionEnum:
    """Tests for the Permission enum."""

    def test_permission_count(self):
        """There should be 13 permissions defined."""
        assert len(Permission) == 13

    def test_federation_permissions(self):
        """Federation-related permissions exist."""
        assert Permission.START_FEDERATION.value == "start_federation"
        assert Permission.STOP_FEDERATION.value == "stop_federation"


class TestAccessControlManagerCreation:
    """Tests for AccessControlManager instantiation."""

    def test_creation_empty(self):
        """Newly created manager has no users."""
        mgr = AccessControlManager()
        assert mgr.get_user("nonexistent") is None

    def test_creation_empty_log(self):
        """Newly created manager has an empty access log."""
        mgr = AccessControlManager()
        log = mgr.get_access_log()
        assert log == []


class TestUserRegistration:
    """Tests for user registration."""

    def test_register_user_basic(self):
        """A registered user can be retrieved."""
        mgr = AccessControlManager()
        profile = mgr.register_user("u1", "Alice", Role.RESEARCHER)
        assert profile.user_id == "u1"
        assert profile.display_name == "Alice"
        assert profile.role == Role.RESEARCHER
        assert profile.active is True

    def test_register_user_with_string_role(self):
        """User can be registered with a string role."""
        mgr = AccessControlManager()
        profile = mgr.register_user("u2", "Bob", "auditor")
        assert profile.role == Role.AUDITOR

    def test_register_user_with_site(self):
        """User can be registered with a site_id."""
        mgr = AccessControlManager()
        profile = mgr.register_user("u3", "Carol", Role.SITE_ADMIN, site_id="SITE-A")
        assert profile.site_id == "SITE-A"

    def test_register_user_creates_audit_entry(self):
        """Registration is logged in the access log."""
        mgr = AccessControlManager()
        mgr.register_user("u1", "Alice", Role.RESEARCHER)
        log = mgr.get_access_log(user_id="u1")
        assert len(log) >= 1
        assert log[0]["action"] == "register_user"

    def test_user_has_created_at(self):
        """Registered user has a non-empty created_at timestamp."""
        mgr = AccessControlManager()
        profile = mgr.register_user("u1", "Alice", Role.RESEARCHER)
        assert profile.created_at != ""


class TestPermissionChecks:
    """Tests for permission checking logic."""

    def test_coordinator_can_start_federation(self):
        """Coordinator has START_FEDERATION permission."""
        mgr = AccessControlManager()
        mgr.register_user("c1", "Coord", Role.COORDINATOR)
        assert mgr.check_permission("c1", Permission.START_FEDERATION) is True

    def test_researcher_cannot_start_federation(self):
        """Researcher does not have START_FEDERATION permission."""
        mgr = AccessControlManager()
        mgr.register_user("r1", "Res", Role.RESEARCHER)
        assert mgr.check_permission("r1", Permission.START_FEDERATION) is False

    def test_patient_has_manage_consent(self):
        """Patient has MANAGE_CONSENT permission."""
        mgr = AccessControlManager()
        mgr.register_user("p1", "Patient", Role.PATIENT)
        assert mgr.check_permission("p1", Permission.MANAGE_CONSENT) is True

    def test_patient_has_only_manage_consent(self):
        """Patient does not have permissions beyond MANAGE_CONSENT."""
        mgr = AccessControlManager()
        mgr.register_user("p1", "Patient", Role.PATIENT)
        assert mgr.check_permission("p1", Permission.START_FEDERATION) is False
        assert mgr.check_permission("p1", Permission.VIEW_AUDIT_LOG) is False

    def test_check_permission_with_string(self):
        """Permission can be provided as a string."""
        mgr = AccessControlManager()
        mgr.register_user("c1", "Coord", Role.COORDINATOR)
        assert mgr.check_permission("c1", "start_federation") is True

    def test_check_permission_unknown_user(self):
        """Checking permission for unknown user returns False."""
        mgr = AccessControlManager()
        assert mgr.check_permission("nonexistent", Permission.START_FEDERATION) is False


class TestDeactivatedUser:
    """Tests for deactivated user behavior."""

    def test_deactivate_user_denies_access(self):
        """A deactivated user is denied all permissions."""
        mgr = AccessControlManager()
        mgr.register_user("u1", "Alice", Role.COORDINATOR)
        assert mgr.check_permission("u1", Permission.START_FEDERATION) is True
        mgr.deactivate_user("u1")
        assert mgr.check_permission("u1", Permission.START_FEDERATION) is False

    def test_deactivate_unknown_user_returns_false(self):
        """Deactivating an unknown user returns False."""
        mgr = AccessControlManager()
        assert mgr.deactivate_user("nonexistent") is False

    def test_deactivate_sets_active_false(self):
        """Deactivation sets user.active to False."""
        mgr = AccessControlManager()
        mgr.register_user("u1", "Alice", Role.RESEARCHER)
        mgr.deactivate_user("u1")
        user = mgr.get_user("u1")
        assert user is not None
        assert user.active is False


class TestExtraAndDeniedPermissions:
    """Tests for grant_permission and revoke_permission."""

    def test_grant_extra_permission(self):
        """Granting extra permission allows access beyond role defaults."""
        mgr = AccessControlManager()
        mgr.register_user("r1", "Res", Role.RESEARCHER)
        assert mgr.check_permission("r1", Permission.UPLOAD_DATA) is False
        mgr.grant_permission("r1", Permission.UPLOAD_DATA)
        assert mgr.check_permission("r1", Permission.UPLOAD_DATA) is True

    def test_revoke_permission(self):
        """Revoking a permission denies access despite role defaults."""
        mgr = AccessControlManager()
        mgr.register_user("c1", "Coord", Role.COORDINATOR)
        assert mgr.check_permission("c1", Permission.START_FEDERATION) is True
        mgr.revoke_permission("c1", Permission.START_FEDERATION)
        assert mgr.check_permission("c1", Permission.START_FEDERATION) is False

    def test_grant_clears_denied(self):
        """Granting a permission removes it from denied set."""
        mgr = AccessControlManager()
        mgr.register_user("u1", "User", Role.COORDINATOR)
        mgr.revoke_permission("u1", Permission.START_FEDERATION)
        assert mgr.check_permission("u1", Permission.START_FEDERATION) is False
        mgr.grant_permission("u1", Permission.START_FEDERATION)
        assert mgr.check_permission("u1", Permission.START_FEDERATION) is True

    def test_grant_unknown_user_returns_false(self):
        """Granting to unknown user returns False."""
        mgr = AccessControlManager()
        assert mgr.grant_permission("ghost", Permission.UPLOAD_DATA) is False

    def test_revoke_unknown_user_returns_false(self):
        """Revoking from unknown user returns False."""
        mgr = AccessControlManager()
        assert mgr.revoke_permission("ghost", Permission.UPLOAD_DATA) is False


class TestAuditLog:
    """Tests for audit log behavior and immutability."""

    def test_audit_log_records_checks(self):
        """Permission checks are recorded in the audit log."""
        mgr = AccessControlManager()
        mgr.register_user("u1", "Alice", Role.RESEARCHER)
        mgr.check_permission("u1", Permission.VIEW_GLOBAL_MODEL, resource="model_v1")
        log = mgr.get_access_log(user_id="u1")
        perm_checks = [e for e in log if e["action"] == "view_global_model"]
        assert len(perm_checks) >= 1
        assert perm_checks[0]["resource"] == "model_v1"

    def test_audit_log_returns_copy(self):
        """get_access_log returns a copy; mutation does not affect internal state."""
        mgr = AccessControlManager()
        mgr.register_user("u1", "Alice", Role.RESEARCHER)
        log1 = mgr.get_access_log()
        original_length = len(log1)
        # Mutate the returned list
        log1.append({"fake": "entry"})
        # Internal state should be unchanged
        log2 = mgr.get_access_log()
        assert len(log2) == original_length

    def test_audit_log_filter_granted(self):
        """Audit log can be filtered by granted=True/False."""
        mgr = AccessControlManager()
        mgr.register_user("u1", "Alice", Role.RESEARCHER)
        mgr.check_permission("u1", Permission.VIEW_GLOBAL_MODEL)  # granted
        mgr.check_permission("u1", Permission.START_FEDERATION)  # denied
        granted_log = mgr.get_access_log(granted=True)
        denied_log = mgr.get_access_log(granted=False)
        assert all(e["granted"] is True for e in granted_log)
        assert all(e["granted"] is False for e in denied_log)
