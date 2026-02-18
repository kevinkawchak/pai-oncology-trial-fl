# Access Control Module

**Version 0.6.0** | `privacy/access-control/`

> **DISCLAIMER: RESEARCH USE ONLY.** This module provides research-grade role-based access
> control tooling. It is not a certified access management system and should not be relied
> upon as the sole access control mechanism in a production clinical environment. Independent
> security review and organizational IT governance approval are required before deployment.

## Overview

The Access Control module implements fine-grained Role-Based Access Control (RBAC) for
federated oncology trial platforms with a **21 CFR Part 11** compliant audit trail. It
enforces the HIPAA minimum necessary standard by mapping each platform role to a precise
set of permissions, supporting time-limited access grants, per-user permission overrides,
and comprehensive access decision logging.

Every access check -- whether granted or denied -- is recorded in a sequential, immutable
audit trail with ISO 8601 timestamps, user identity, resource, and decision outcome,
satisfying regulatory requirements for access monitoring and retrospective compliance review.

## File Structure

```
access-control/
├── README.md                     # This file
└── access_control_manager.py     # RBAC engine with 21 CFR Part 11 audit trail
```

## Regulatory References

- **21 CFR Part 11** -- Electronic Records; Electronic Signatures
  - 11.10(d): Access controls limiting system access to authorized individuals
  - 11.10(e): Audit trails for record creation, modification, and deletion
- **45 CFR 164.312** -- HIPAA Security Rule, Access Controls
- **45 CFR 164.312(b)** -- Audit Controls
- **NIST SP 800-53 AC-2** -- Account Management
- **NIST SP 800-53 AC-3** -- Access Enforcement

## API Overview

### Enums

| Enum | Values | Purpose |
|------|--------|---------|
| `AccessRole` | `PRINCIPAL_INVESTIGATOR`, `COORDINATOR`, `SITE_ADMIN`, `CLINICAL_RESEARCHER`, `DATA_ENGINEER`, `BIOSTATISTICIAN`, `SAFETY_OFFICER`, `AUDITOR`, `REGULATORY_AFFAIRS`, `PATIENT` | Platform roles with separation of duties |
| `Permission` | `START_FEDERATION`, `STOP_FEDERATION`, `VIEW_GLOBAL_MODEL`, `DOWNLOAD_MODEL`, `CONFIGURE_AGGREGATION`, `UPLOAD_DATA`, `VIEW_DATA_SUMMARY`, `EXPORT_DATA`, `DELETE_DATA`, `VIEW_AUDIT_LOG`, `MANAGE_CONSENT`, `RUN_DEIDENTIFICATION`, `VIEW_PHI`, `MANAGE_SITES`, `MANAGE_USERS`, `VIEW_COMPLIANCE_REPORT`, `APPROVE_PROTOCOL`, `REPORT_ADVERSE_EVENT`, `VIEW_SAFETY_DATA`, `HALT_TRIAL`, `SUBMIT_REGULATORY`, `VIEW_REGULATORY_STATUS` | Granular permissions mapped to roles |
| `AuditAction` | `ACCESS_GRANTED`, `ACCESS_DENIED`, `USER_REGISTERED`, `USER_DEACTIVATED`, `USER_REACTIVATED`, `PERMISSION_GRANTED`, `PERMISSION_REVOKED`, `ROLE_CHANGED`, `ACCESS_EXPIRED`, `SESSION_STARTED`, `SESSION_ENDED`, `DATA_ACCESSED`, `CONFIG_CHANGED` | Audit trail action types |

### Data Classes

| Class | Key Fields | Purpose |
|-------|-----------|---------|
| `AuditEntry` | `entry_id`, `timestamp`, `user_id`, `action`, `permission`, `resource`, `granted` | 21 CFR Part 11 compliant audit record |
| `UserProfile` | `user_id`, `display_name`, `role`, `site_id`, `active`, `extra_permissions`, `denied_permissions` | Registered user with role and permission overrides |

### AccessControlManager

The primary class for managing users, roles, and permission checks.

```python
from privacy.access_control.access_control_manager import (
    AccessControlManager,
    AccessRole,
    Permission,
)

acm = AccessControlManager()
```

#### User Management

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `register_user` | `(user_id, display_name, role, site_id) -> UserProfile` | `UserProfile` | Register a new user with a role |
| `deactivate_user` | `(user_id: str) -> bool` | `bool` | Deactivate a user account |
| `reactivate_user` | `(user_id: str) -> bool` | `bool` | Reactivate a user account |
| `change_role` | `(user_id, new_role) -> bool` | `bool` | Change a user's role |
| `get_user` | `(user_id: str) -> UserProfile` | `UserProfile` | Retrieve a user profile |

#### Permission Checks

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `check_permission` | `(user_id, permission, resource) -> bool` | `bool` | Check and log an access decision |
| `grant_permission` | `(user_id, permission) -> bool` | `bool` | Grant an extra permission beyond role default |
| `revoke_permission` | `(user_id, permission) -> bool` | `bool` | Explicitly deny a permission |

#### Audit

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `get_audit_trail` | `(user_id?, action?, limit?) -> list[AuditEntry]` | `list[AuditEntry]` | Query audit trail with optional filters |
| `get_access_summary` | `(user_id: str) -> dict` | `dict` | Access statistics for a user |

### Role-Permission Matrix

| Role | Federation | Data | Privacy | Admin | Safety | Regulatory |
|------|-----------|------|---------|-------|--------|-----------|
| Principal Investigator | Start, Stop, View, Download, Configure | View Summary | View Audit, View Compliance | Manage Sites, Manage Users, Approve Protocol | View Safety, Halt Trial | Submit, View Status |
| Coordinator | Start, Stop, View, Download | View Summary | View Audit, View Compliance | Manage Sites, Manage Users | -- | View Status |
| Site Admin | -- | Upload, View Summary | Manage Consent, Run DeID, View Compliance | -- | Report AE | -- |
| Clinical Researcher | View, Download | View Summary | -- | -- | View Safety | -- |
| Data Engineer | -- | Upload, View Summary, Export | Run DeID | -- | -- | -- |
| Biostatistician | View, Download | View Summary, Export | -- | -- | View Safety | -- |
| Safety Officer | -- | View Summary | View Audit, View Compliance | -- | View Safety, Report AE, Halt Trial | -- |
| Auditor | -- | View Summary | View Audit, View Compliance | -- | -- | View Status |
| Regulatory Affairs | -- | View Summary | View Audit, View Compliance | -- | -- | Submit, View Status |
| Patient | -- | -- | Manage Consent | -- | -- | -- |

## Usage Example

```python
from privacy.access_control.access_control_manager import (
    AccessControlManager,
    AccessRole,
    Permission,
)

acm = AccessControlManager()

# Register users
pi = acm.register_user("pi_001", "Dr. Chen", AccessRole.PRINCIPAL_INVESTIGATOR, "site_01")
researcher = acm.register_user("res_001", "Dr. Park", AccessRole.CLINICAL_RESEARCHER, "site_02")

# Check permissions (every check is logged)
can_start = acm.check_permission("pi_001", Permission.START_FEDERATION, resource="study_001")
print(f"PI can start federation: {can_start}")  # True

can_delete = acm.check_permission("res_001", Permission.DELETE_DATA, resource="site_02_data")
print(f"Researcher can delete data: {can_delete}")  # False

# Grant an extra permission
acm.grant_permission("res_001", Permission.EXPORT_DATA)

# Query audit trail
trail = acm.get_audit_trail(user_id="pi_001")
for entry in trail:
    print(f"  [{entry.timestamp}] {entry.action.value}: {entry.permission} -> {entry.granted}")
```

## License

MIT License. See [LICENSE](../../LICENSE) for details.
