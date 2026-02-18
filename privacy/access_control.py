"""Role-based access control for federated learning operations.

Provides fine-grained access control so that different participants
(site administrators, researchers, coordinators, auditors) can only
perform actions appropriate to their role.  Access decisions are
logged to the audit trail for compliance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """Roles in the federated learning platform."""

    COORDINATOR = "coordinator"
    SITE_ADMIN = "site_admin"
    RESEARCHER = "researcher"
    DATA_ENGINEER = "data_engineer"
    AUDITOR = "auditor"
    PATIENT = "patient"


class Permission(str, Enum):
    """Granular permissions for platform operations."""

    # Federation
    START_FEDERATION = "start_federation"
    STOP_FEDERATION = "stop_federation"
    VIEW_GLOBAL_MODEL = "view_global_model"
    DOWNLOAD_MODEL = "download_model"
    # Data
    UPLOAD_DATA = "upload_data"
    VIEW_DATA_SUMMARY = "view_data_summary"
    EXPORT_DATA = "export_data"
    # Privacy
    VIEW_AUDIT_LOG = "view_audit_log"
    MANAGE_CONSENT = "manage_consent"
    RUN_DEIDENTIFICATION = "run_deidentification"
    # Administration
    MANAGE_SITES = "manage_sites"
    MANAGE_USERS = "manage_users"
    VIEW_COMPLIANCE_REPORT = "view_compliance_report"


# Default role -> permissions mapping
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.COORDINATOR: {
        Permission.START_FEDERATION,
        Permission.STOP_FEDERATION,
        Permission.VIEW_GLOBAL_MODEL,
        Permission.DOWNLOAD_MODEL,
        Permission.VIEW_DATA_SUMMARY,
        Permission.VIEW_AUDIT_LOG,
        Permission.MANAGE_SITES,
        Permission.MANAGE_USERS,
        Permission.VIEW_COMPLIANCE_REPORT,
    },
    Role.SITE_ADMIN: {
        Permission.UPLOAD_DATA,
        Permission.VIEW_DATA_SUMMARY,
        Permission.VIEW_GLOBAL_MODEL,
        Permission.MANAGE_CONSENT,
        Permission.RUN_DEIDENTIFICATION,
        Permission.VIEW_COMPLIANCE_REPORT,
    },
    Role.RESEARCHER: {
        Permission.VIEW_GLOBAL_MODEL,
        Permission.VIEW_DATA_SUMMARY,
        Permission.DOWNLOAD_MODEL,
    },
    Role.DATA_ENGINEER: {
        Permission.UPLOAD_DATA,
        Permission.VIEW_DATA_SUMMARY,
        Permission.RUN_DEIDENTIFICATION,
        Permission.EXPORT_DATA,
    },
    Role.AUDITOR: {
        Permission.VIEW_AUDIT_LOG,
        Permission.VIEW_DATA_SUMMARY,
        Permission.VIEW_COMPLIANCE_REPORT,
    },
    Role.PATIENT: {
        Permission.MANAGE_CONSENT,
    },
}


@dataclass
class UserProfile:
    """A registered user in the platform.

    Attributes:
        user_id: Unique user identifier.
        display_name: Human-readable name.
        role: Assigned role.
        site_id: Site the user belongs to (if applicable).
        active: Whether the account is active.
        created_at: ISO timestamp of account creation.
        extra_permissions: Permissions granted beyond the role default.
        denied_permissions: Permissions explicitly revoked.
    """

    user_id: str
    display_name: str
    role: Role
    site_id: str = ""
    active: bool = True
    created_at: str = ""
    extra_permissions: set[Permission] = field(default_factory=set)
    denied_permissions: set[Permission] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class AccessControlManager:
    """Manages users, roles, and permission checks.

    Every permission check is logged for HIPAA/GDPR audit compliance.
    """

    def __init__(self) -> None:
        self._users: dict[str, UserProfile] = {}
        self._access_log: list[dict] = []

    # ------------------------------------------------------------------
    # User management
    # ------------------------------------------------------------------

    def register_user(
        self,
        user_id: str,
        display_name: str,
        role: Role | str,
        site_id: str = "",
    ) -> UserProfile:
        """Register a new user.

        Args:
            user_id: Unique user identifier.
            display_name: Human-readable name.
            role: Role to assign.
            site_id: Associated site (if any).

        Returns:
            The created UserProfile.
        """
        if isinstance(role, str):
            role = Role(role)

        profile = UserProfile(
            user_id=user_id,
            display_name=display_name,
            role=role,
            site_id=site_id,
        )
        self._users[user_id] = profile
        self._log_access(user_id, "register_user", "system", granted=True)
        logger.info("User %s registered with role %s", user_id, role.value)
        return profile

    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user account."""
        user = self._users.get(user_id)
        if user is None:
            return False
        user.active = False
        self._log_access(user_id, "deactivate_user", "system", granted=True)
        return True

    def get_user(self, user_id: str) -> UserProfile | None:
        """Retrieve a user profile."""
        return self._users.get(user_id)

    # ------------------------------------------------------------------
    # Permission checks
    # ------------------------------------------------------------------

    def check_permission(
        self,
        user_id: str,
        permission: Permission | str,
        resource: str = "",
    ) -> bool:
        """Check whether a user has a specific permission.

        The decision is logged for audit purposes.

        Args:
            user_id: The user requesting access.
            permission: The permission to check.
            resource: The resource being accessed (for logging).

        Returns:
            True if access is granted.
        """
        if isinstance(permission, str):
            permission = Permission(permission)

        user = self._users.get(user_id)
        if user is None or not user.active:
            self._log_access(user_id, permission.value, resource, granted=False)
            return False

        # Check denied first
        if permission in user.denied_permissions:
            self._log_access(user_id, permission.value, resource, granted=False)
            return False

        # Check extra permissions
        if permission in user.extra_permissions:
            self._log_access(user_id, permission.value, resource, granted=True)
            return True

        # Check role defaults
        role_perms = ROLE_PERMISSIONS.get(user.role, set())
        granted = permission in role_perms
        self._log_access(user_id, permission.value, resource, granted=granted)
        return granted

    def grant_permission(self, user_id: str, permission: Permission) -> bool:
        """Grant an additional permission to a user."""
        user = self._users.get(user_id)
        if user is None:
            return False
        user.extra_permissions.add(permission)
        user.denied_permissions.discard(permission)
        return True

    def revoke_permission(self, user_id: str, permission: Permission) -> bool:
        """Explicitly revoke a permission from a user."""
        user = self._users.get(user_id)
        if user is None:
            return False
        user.denied_permissions.add(permission)
        user.extra_permissions.discard(permission)
        return True

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def get_access_log(self, user_id: str | None = None, granted: bool | None = None) -> list[dict]:
        """Query access log with optional filters."""
        results = list(self._access_log)
        if user_id is not None:
            results = [r for r in results if r["user_id"] == user_id]
        if granted is not None:
            results = [r for r in results if r["granted"] == granted]
        return results

    def _log_access(
        self,
        user_id: str,
        action: str,
        resource: str,
        granted: bool,
    ) -> None:
        self._access_log.append(
            {
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "granted": granted,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
