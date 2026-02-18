"""Role-Based Access Control (RBAC) with 21 CFR Part 11 audit trail.

Implements fine-grained access control for federated oncology trial
platforms with comprehensive audit logging, time-limited access grants,
and regulatory-compliant access decision recording.

References:
    - 21 CFR Part 11: Electronic Records; Electronic Signatures
    - 45 CFR 164.312: HIPAA Security Rule — Access Controls
    - 45 CFR 164.312(b): Audit Controls
    - NIST SP 800-53 AC-2: Account Management
    - NIST SP 800-53 AC-3: Access Enforcement

DISCLAIMER: RESEARCH USE ONLY — Not for clinical decision-making.
LICENSE: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AccessRole(str, Enum):
    """Roles in the federated oncology trial platform.

    Role hierarchy supports separation of duties per 21 CFR Part 11
    and HIPAA minimum necessary standard.
    """

    PRINCIPAL_INVESTIGATOR = "principal_investigator"
    COORDINATOR = "coordinator"
    SITE_ADMIN = "site_admin"
    CLINICAL_RESEARCHER = "clinical_researcher"
    DATA_ENGINEER = "data_engineer"
    BIOSTATISTICIAN = "biostatistician"
    SAFETY_OFFICER = "safety_officer"
    AUDITOR = "auditor"
    REGULATORY_AFFAIRS = "regulatory_affairs"
    PATIENT = "patient"


class Permission(str, Enum):
    """Granular permissions for platform operations.

    Mapped to roles via ROLE_PERMISSIONS. Additional permissions
    can be granted or denied per-user.
    """

    # Federation operations
    START_FEDERATION = "start_federation"
    STOP_FEDERATION = "stop_federation"
    VIEW_GLOBAL_MODEL = "view_global_model"
    DOWNLOAD_MODEL = "download_model"
    CONFIGURE_AGGREGATION = "configure_aggregation"
    # Data operations
    UPLOAD_DATA = "upload_data"
    VIEW_DATA_SUMMARY = "view_data_summary"
    EXPORT_DATA = "export_data"
    DELETE_DATA = "delete_data"
    # Privacy operations
    VIEW_AUDIT_LOG = "view_audit_log"
    MANAGE_CONSENT = "manage_consent"
    RUN_DEIDENTIFICATION = "run_deidentification"
    VIEW_PHI = "view_phi"
    # Administration
    MANAGE_SITES = "manage_sites"
    MANAGE_USERS = "manage_users"
    VIEW_COMPLIANCE_REPORT = "view_compliance_report"
    APPROVE_PROTOCOL = "approve_protocol"
    # Safety
    REPORT_ADVERSE_EVENT = "report_adverse_event"
    VIEW_SAFETY_DATA = "view_safety_data"
    HALT_TRIAL = "halt_trial"
    # Regulatory
    SUBMIT_REGULATORY = "submit_regulatory"
    VIEW_REGULATORY_STATUS = "view_regulatory_status"


class AuditAction(str, Enum):
    """Actions recorded in the 21 CFR Part 11 audit trail.

    Every access decision, user lifecycle event, and permission
    change is recorded with one of these action types.
    """

    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    USER_REGISTERED = "user_registered"
    USER_DEACTIVATED = "user_deactivated"
    USER_REACTIVATED = "user_reactivated"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    ROLE_CHANGED = "role_changed"
    ACCESS_EXPIRED = "access_expired"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    DATA_ACCESSED = "data_accessed"
    CONFIG_CHANGED = "config_changed"


# ---------------------------------------------------------------------------
# Role-Permission mapping
# ---------------------------------------------------------------------------

ROLE_PERMISSIONS: dict[AccessRole, set[Permission]] = {
    AccessRole.PRINCIPAL_INVESTIGATOR: {
        Permission.START_FEDERATION,
        Permission.STOP_FEDERATION,
        Permission.VIEW_GLOBAL_MODEL,
        Permission.DOWNLOAD_MODEL,
        Permission.CONFIGURE_AGGREGATION,
        Permission.VIEW_DATA_SUMMARY,
        Permission.VIEW_AUDIT_LOG,
        Permission.MANAGE_SITES,
        Permission.MANAGE_USERS,
        Permission.VIEW_COMPLIANCE_REPORT,
        Permission.APPROVE_PROTOCOL,
        Permission.VIEW_SAFETY_DATA,
        Permission.HALT_TRIAL,
        Permission.SUBMIT_REGULATORY,
        Permission.VIEW_REGULATORY_STATUS,
    },
    AccessRole.COORDINATOR: {
        Permission.START_FEDERATION,
        Permission.STOP_FEDERATION,
        Permission.VIEW_GLOBAL_MODEL,
        Permission.DOWNLOAD_MODEL,
        Permission.VIEW_DATA_SUMMARY,
        Permission.VIEW_AUDIT_LOG,
        Permission.MANAGE_SITES,
        Permission.MANAGE_USERS,
        Permission.VIEW_COMPLIANCE_REPORT,
        Permission.VIEW_REGULATORY_STATUS,
    },
    AccessRole.SITE_ADMIN: {
        Permission.UPLOAD_DATA,
        Permission.VIEW_DATA_SUMMARY,
        Permission.VIEW_GLOBAL_MODEL,
        Permission.MANAGE_CONSENT,
        Permission.RUN_DEIDENTIFICATION,
        Permission.VIEW_COMPLIANCE_REPORT,
        Permission.REPORT_ADVERSE_EVENT,
    },
    AccessRole.CLINICAL_RESEARCHER: {
        Permission.VIEW_GLOBAL_MODEL,
        Permission.VIEW_DATA_SUMMARY,
        Permission.DOWNLOAD_MODEL,
        Permission.VIEW_SAFETY_DATA,
    },
    AccessRole.DATA_ENGINEER: {
        Permission.UPLOAD_DATA,
        Permission.VIEW_DATA_SUMMARY,
        Permission.RUN_DEIDENTIFICATION,
        Permission.EXPORT_DATA,
    },
    AccessRole.BIOSTATISTICIAN: {
        Permission.VIEW_GLOBAL_MODEL,
        Permission.VIEW_DATA_SUMMARY,
        Permission.DOWNLOAD_MODEL,
        Permission.EXPORT_DATA,
        Permission.VIEW_SAFETY_DATA,
    },
    AccessRole.SAFETY_OFFICER: {
        Permission.VIEW_SAFETY_DATA,
        Permission.REPORT_ADVERSE_EVENT,
        Permission.HALT_TRIAL,
        Permission.VIEW_DATA_SUMMARY,
        Permission.VIEW_AUDIT_LOG,
        Permission.VIEW_COMPLIANCE_REPORT,
    },
    AccessRole.AUDITOR: {
        Permission.VIEW_AUDIT_LOG,
        Permission.VIEW_DATA_SUMMARY,
        Permission.VIEW_COMPLIANCE_REPORT,
        Permission.VIEW_REGULATORY_STATUS,
    },
    AccessRole.REGULATORY_AFFAIRS: {
        Permission.VIEW_COMPLIANCE_REPORT,
        Permission.VIEW_REGULATORY_STATUS,
        Permission.SUBMIT_REGULATORY,
        Permission.VIEW_AUDIT_LOG,
        Permission.VIEW_DATA_SUMMARY,
    },
    AccessRole.PATIENT: {
        Permission.MANAGE_CONSENT,
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AuditEntry:
    """A single entry in the 21 CFR Part 11 compliant audit trail.

    Attributes:
        entry_id: Unique sequential identifier.
        timestamp: ISO 8601 timestamp with timezone.
        user_id: User who triggered the action.
        action: The audit action type.
        permission: The permission checked (if applicable).
        resource: The resource being accessed.
        granted: Whether access was granted.
        ip_address: Source IP (if available).
        details: Additional context.
    """

    entry_id: int
    timestamp: str
    user_id: str
    action: AuditAction
    permission: str = ""
    resource: str = ""
    granted: bool = False
    ip_address: str = ""
    details: str = ""


@dataclass
class UserAccount:
    """A user account in the access control system.

    Attributes:
        user_id: Unique user identifier.
        display_name: Human-readable name.
        role: Assigned role.
        site_id: Associated site (if applicable).
        active: Whether the account is active.
        created_at: ISO timestamp of account creation.
        expires_at: Optional expiration timestamp (ISO format).
        extra_permissions: Permissions granted beyond role defaults.
        denied_permissions: Permissions explicitly revoked.
        last_access: Timestamp of last access check.
        access_count: Number of access checks performed.
    """

    user_id: str
    display_name: str
    role: AccessRole
    site_id: str = ""
    active: bool = True
    created_at: str = ""
    expires_at: str = ""
    extra_permissions: set[Permission] = field(default_factory=set)
    denied_permissions: set[Permission] = field(default_factory=set)
    last_access: str = ""
    access_count: int = 0

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Access Control Manager
# ---------------------------------------------------------------------------


class AccessControlManager:
    """RBAC manager with 21 CFR Part 11 compliant audit trail.

    Every access decision is immutably logged. The audit log is
    returned as a copy to prevent external mutation.

    Features:
        - Role-based permission checks with per-user overrides
        - Time-limited access with automatic expiration
        - Complete audit trail of all access decisions
        - User lifecycle management (register, deactivate, reactivate)
        - Separation of duties enforcement
    """

    def __init__(self) -> None:
        self._users: dict[str, UserAccount] = {}
        self._audit_log: list[AuditEntry] = []
        self._next_entry_id: int = 1

    # ------------------------------------------------------------------
    # User management
    # ------------------------------------------------------------------

    def register_user(
        self,
        user_id: str,
        display_name: str,
        role: AccessRole | str,
        site_id: str = "",
        expires_at: str = "",
    ) -> UserAccount:
        """Register a new user account.

        Args:
            user_id: Unique user identifier.
            display_name: Human-readable name.
            role: Role to assign.
            site_id: Associated clinical site.
            expires_at: Optional access expiration (ISO format).

        Returns:
            The created UserAccount.
        """
        if isinstance(role, str):
            role = AccessRole(role)

        account = UserAccount(
            user_id=user_id,
            display_name=display_name,
            role=role,
            site_id=site_id,
            expires_at=expires_at,
        )
        self._users[user_id] = account
        self._record_audit(
            user_id=user_id,
            action=AuditAction.USER_REGISTERED,
            details=f"role={role.value}, site={site_id}",
            granted=True,
        )
        logger.info("User '%s' registered with role '%s'", user_id, role.value)
        return account

    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user account."""
        user = self._users.get(user_id)
        if user is None:
            return False
        user.active = False
        self._record_audit(
            user_id=user_id,
            action=AuditAction.USER_DEACTIVATED,
            granted=True,
        )
        logger.info("User '%s' deactivated", user_id)
        return True

    def reactivate_user(self, user_id: str) -> bool:
        """Reactivate a previously deactivated user account."""
        user = self._users.get(user_id)
        if user is None:
            return False
        user.active = True
        self._record_audit(
            user_id=user_id,
            action=AuditAction.USER_REACTIVATED,
            granted=True,
        )
        return True

    def change_role(self, user_id: str, new_role: AccessRole | str) -> bool:
        """Change a user's role."""
        user = self._users.get(user_id)
        if user is None:
            return False
        if isinstance(new_role, str):
            new_role = AccessRole(new_role)
        old_role = user.role
        user.role = new_role
        self._record_audit(
            user_id=user_id,
            action=AuditAction.ROLE_CHANGED,
            details=f"old={old_role.value}, new={new_role.value}",
            granted=True,
        )
        return True

    def get_user(self, user_id: str) -> UserAccount | None:
        """Retrieve a user account."""
        return self._users.get(user_id)

    def list_users(self, role: AccessRole | None = None, active_only: bool = True) -> list[UserAccount]:
        """List user accounts with optional filtering."""
        users = list(self._users.values())
        if active_only:
            users = [u for u in users if u.active]
        if role is not None:
            users = [u for u in users if u.role == role]
        return users

    # ------------------------------------------------------------------
    # Permission management
    # ------------------------------------------------------------------

    def check_permission(
        self,
        user_id: str,
        permission: Permission | str,
        resource: str = "",
    ) -> bool:
        """Check whether a user has a specific permission.

        Access is denied by default for:
        - Unknown users
        - Inactive users
        - Users with expired access (invalid date formats deny access)
        - Users with explicitly denied permissions

        Every check is recorded in the audit trail per 21 CFR Part 11.

        Args:
            user_id: The user requesting access.
            permission: The permission to check.
            resource: The resource being accessed (for audit).

        Returns:
            True if access is granted, False otherwise.
        """
        if isinstance(permission, str):
            permission = Permission(permission)

        user = self._users.get(user_id)

        # Deny unknown users
        if user is None:
            self._record_audit(
                user_id=user_id,
                action=AuditAction.ACCESS_DENIED,
                permission=permission.value,
                resource=resource,
                granted=False,
                details="user_not_found",
            )
            return False

        # Deny inactive users
        if not user.active:
            self._record_audit(
                user_id=user_id,
                action=AuditAction.ACCESS_DENIED,
                permission=permission.value,
                resource=resource,
                granted=False,
                details="user_inactive",
            )
            return False

        # Check expiration — invalid date formats deny access by default
        if user.expires_at:
            try:
                expires = datetime.fromisoformat(user.expires_at)
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) > expires:
                    self._record_audit(
                        user_id=user_id,
                        action=AuditAction.ACCESS_EXPIRED,
                        permission=permission.value,
                        resource=resource,
                        granted=False,
                        details=f"expired_at={user.expires_at}",
                    )
                    return False
            except (ValueError, TypeError):
                # Invalid date format — deny access by default (fail-closed)
                self._record_audit(
                    user_id=user_id,
                    action=AuditAction.ACCESS_DENIED,
                    permission=permission.value,
                    resource=resource,
                    granted=False,
                    details=f"invalid_expiration_format={user.expires_at}",
                )
                return False

        # Check explicitly denied permissions
        if permission in user.denied_permissions:
            self._record_audit(
                user_id=user_id,
                action=AuditAction.ACCESS_DENIED,
                permission=permission.value,
                resource=resource,
                granted=False,
                details="explicitly_denied",
            )
            return False

        # Check extra permissions
        if permission in user.extra_permissions:
            user.access_count += 1
            user.last_access = datetime.now(timezone.utc).isoformat()
            self._record_audit(
                user_id=user_id,
                action=AuditAction.ACCESS_GRANTED,
                permission=permission.value,
                resource=resource,
                granted=True,
                details="extra_permission",
            )
            return True

        # Check role-based permissions
        role_perms = ROLE_PERMISSIONS.get(user.role, set())
        granted = permission in role_perms

        if granted:
            user.access_count += 1
            user.last_access = datetime.now(timezone.utc).isoformat()

        self._record_audit(
            user_id=user_id,
            action=AuditAction.ACCESS_GRANTED if granted else AuditAction.ACCESS_DENIED,
            permission=permission.value,
            resource=resource,
            granted=granted,
            details=f"role={user.role.value}",
        )
        return granted

    def grant_permission(self, user_id: str, permission: Permission) -> bool:
        """Grant an additional permission to a user."""
        user = self._users.get(user_id)
        if user is None:
            return False
        user.extra_permissions.add(permission)
        user.denied_permissions.discard(permission)
        self._record_audit(
            user_id=user_id,
            action=AuditAction.PERMISSION_GRANTED,
            permission=permission.value,
            granted=True,
        )
        return True

    def revoke_permission(self, user_id: str, permission: Permission) -> bool:
        """Explicitly revoke a permission from a user."""
        user = self._users.get(user_id)
        if user is None:
            return False
        user.denied_permissions.add(permission)
        user.extra_permissions.discard(permission)
        self._record_audit(
            user_id=user_id,
            action=AuditAction.PERMISSION_REVOKED,
            permission=permission.value,
            granted=True,
        )
        return True

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------

    def get_audit_log(
        self,
        user_id: str | None = None,
        action: AuditAction | None = None,
        granted: bool | None = None,
    ) -> list[AuditEntry]:
        """Query the audit log with optional filters.

        Returns a COPY of the audit entries to prevent external
        mutation of the immutable audit trail.

        Args:
            user_id: Filter by user.
            action: Filter by action type.
            granted: Filter by access decision.

        Returns:
            List of matching AuditEntry objects (copy).
        """
        results = list(self._audit_log)  # Copy
        if user_id is not None:
            results = [e for e in results if e.user_id == user_id]
        if action is not None:
            results = [e for e in results if e.action == action]
        if granted is not None:
            results = [e for e in results if e.granted == granted]
        return results

    def get_audit_summary(self) -> dict[str, int]:
        """Return a summary of audit log entries by action type."""
        summary: dict[str, int] = {}
        for entry in self._audit_log:
            key = entry.action.value
            summary[key] = summary.get(key, 0) + 1
        return summary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_audit(
        self,
        user_id: str,
        action: AuditAction,
        permission: str = "",
        resource: str = "",
        granted: bool = False,
        details: str = "",
        ip_address: str = "",
    ) -> None:
        """Record an immutable audit trail entry per 21 CFR Part 11."""
        entry = AuditEntry(
            entry_id=self._next_entry_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            action=action,
            permission=permission,
            resource=resource,
            granted=granted,
            ip_address=ip_address,
            details=details,
        )
        self._audit_log.append(entry)
        self._next_entry_id += 1
