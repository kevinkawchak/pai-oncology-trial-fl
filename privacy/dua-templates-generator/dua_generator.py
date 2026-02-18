"""Data Use Agreement (DUA) template generator.

Generates structured DUA documents for oncology clinical trial
data sharing, supporting intra-institutional, multi-site, and
commercial partnership agreements with appropriate data retention
periods and security requirements.

References:
    - 45 CFR 164.514(e): Limited data set and DUA requirements
    - 45 CFR 164.508: Authorization for uses and disclosures
    - NIH Data Sharing Policy (2023)
    - HIPAA Privacy Rule — minimum necessary standard

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


class DUAType(str, Enum):
    """Types of Data Use Agreements.

    Each type has different requirements for data handling,
    security controls, and retention periods.
    """

    INTRA_INSTITUTIONAL = "intra_institutional"
    MULTI_SITE = "multi_site"
    COMMERCIAL = "commercial"
    ACADEMIC_COLLABORATION = "academic_collaboration"
    GOVERNMENT = "government"


class DataCategory(str, Enum):
    """Categories of data covered by a DUA."""

    LIMITED_DATA_SET = "limited_data_set"
    DE_IDENTIFIED = "de_identified"
    CODED = "coded"
    FULLY_IDENTIFIED = "fully_identified"
    AGGREGATED = "aggregated"


class SecurityTier(str, Enum):
    """Security requirement tiers for data handling."""

    STANDARD = "standard"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"


class DUAStatus(str, Enum):
    """Status of a DUA in its lifecycle."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    UNDER_NEGOTIATION = "under_negotiation"
    APPROVED = "approved"
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SecurityRequirements:
    """Security controls required for data handling under a DUA.

    Attributes:
        encryption_at_rest: Whether data must be encrypted at rest.
        encryption_in_transit: Whether data must be encrypted in transit.
        encryption_standard: Minimum encryption standard (e.g., AES-256).
        mfa_required: Whether multi-factor authentication is required.
        access_logging: Whether access logging is required.
        data_loss_prevention: Whether DLP controls are required.
        network_segmentation: Whether network segmentation is required.
        incident_response_plan: Whether a documented IR plan is required.
        annual_security_audit: Whether annual security audits are required.
        background_checks: Whether background checks for data handlers are required.
    """

    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    encryption_standard: str = "AES-256"
    mfa_required: bool = True
    access_logging: bool = True
    data_loss_prevention: bool = False
    network_segmentation: bool = False
    incident_response_plan: bool = True
    annual_security_audit: bool = False
    background_checks: bool = False


@dataclass
class RetentionPolicy:
    """Data retention requirements.

    Attributes:
        retention_period_years: How long data may be retained.
        destruction_method: Method for data destruction.
        destruction_certification: Whether destruction certification is required.
        extension_allowed: Whether retention extension is possible.
        review_frequency_months: How often retention is reviewed.
    """

    retention_period_years: int = 3
    destruction_method: str = "cryptographic_erasure"
    destruction_certification: bool = True
    extension_allowed: bool = False
    review_frequency_months: int = 12


@dataclass
class DUAParty:
    """A party to a Data Use Agreement.

    Attributes:
        organization_name: Legal name of the organization.
        role: Role in the agreement (provider, recipient, etc.).
        contact_name: Primary contact person.
        contact_email: Contact email.
        institution_type: Type of institution.
        irb_approval_number: IRB approval number (if applicable).
    """

    organization_name: str
    role: str = "recipient"
    contact_name: str = ""
    contact_email: str = ""
    institution_type: str = ""
    irb_approval_number: str = ""


@dataclass
class DUADocument:
    """A generated Data Use Agreement document.

    Attributes:
        dua_id: Unique DUA identifier.
        dua_type: Type of agreement.
        status: Current lifecycle status.
        title: Agreement title.
        parties: Parties to the agreement.
        data_category: Category of data covered.
        data_description: Description of data being shared.
        purpose: Permitted use purpose.
        security_requirements: Required security controls.
        retention_policy: Data retention policy.
        effective_date: When the DUA becomes effective.
        expiration_date: When the DUA expires.
        created_at: When the DUA was generated.
        sections: Ordered agreement sections.
        amendments: List of amendments.
    """

    dua_id: str
    dua_type: DUAType
    status: DUAStatus = DUAStatus.DRAFT
    title: str = ""
    parties: list[DUAParty] = field(default_factory=list)
    data_category: DataCategory = DataCategory.LIMITED_DATA_SET
    data_description: str = ""
    purpose: str = ""
    security_requirements: SecurityRequirements | None = None
    retention_policy: RetentionPolicy | None = None
    effective_date: str = ""
    expiration_date: str = ""
    created_at: str = ""
    sections: list[dict[str, str]] = field(default_factory=list)
    amendments: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if self.security_requirements is None:
            self.security_requirements = SecurityRequirements()
        if self.retention_policy is None:
            self.retention_policy = RetentionPolicy()


# ---------------------------------------------------------------------------
# Default security/retention by DUA type
# ---------------------------------------------------------------------------

_DEFAULT_SECURITY: dict[DUAType, SecurityRequirements] = {
    DUAType.INTRA_INSTITUTIONAL: SecurityRequirements(
        encryption_at_rest=True,
        encryption_in_transit=True,
        encryption_standard="AES-256",
        mfa_required=True,
        access_logging=True,
        data_loss_prevention=False,
        network_segmentation=False,
        incident_response_plan=True,
        annual_security_audit=False,
        background_checks=False,
    ),
    DUAType.MULTI_SITE: SecurityRequirements(
        encryption_at_rest=True,
        encryption_in_transit=True,
        encryption_standard="AES-256",
        mfa_required=True,
        access_logging=True,
        data_loss_prevention=True,
        network_segmentation=True,
        incident_response_plan=True,
        annual_security_audit=True,
        background_checks=False,
    ),
    DUAType.COMMERCIAL: SecurityRequirements(
        encryption_at_rest=True,
        encryption_in_transit=True,
        encryption_standard="AES-256",
        mfa_required=True,
        access_logging=True,
        data_loss_prevention=True,
        network_segmentation=True,
        incident_response_plan=True,
        annual_security_audit=True,
        background_checks=True,
    ),
    DUAType.ACADEMIC_COLLABORATION: SecurityRequirements(
        encryption_at_rest=True,
        encryption_in_transit=True,
        encryption_standard="AES-256",
        mfa_required=True,
        access_logging=True,
        data_loss_prevention=False,
        network_segmentation=False,
        incident_response_plan=True,
        annual_security_audit=False,
        background_checks=False,
    ),
    DUAType.GOVERNMENT: SecurityRequirements(
        encryption_at_rest=True,
        encryption_in_transit=True,
        encryption_standard="AES-256",
        mfa_required=True,
        access_logging=True,
        data_loss_prevention=True,
        network_segmentation=True,
        incident_response_plan=True,
        annual_security_audit=True,
        background_checks=True,
    ),
}

_DEFAULT_RETENTION: dict[DUAType, RetentionPolicy] = {
    DUAType.INTRA_INSTITUTIONAL: RetentionPolicy(
        retention_period_years=3,
        destruction_method="cryptographic_erasure",
        destruction_certification=True,
        review_frequency_months=12,
    ),
    DUAType.MULTI_SITE: RetentionPolicy(
        retention_period_years=5,
        destruction_method="cryptographic_erasure",
        destruction_certification=True,
        review_frequency_months=6,
    ),
    DUAType.COMMERCIAL: RetentionPolicy(
        retention_period_years=7,
        destruction_method="nist_800_88_purge",
        destruction_certification=True,
        extension_allowed=True,
        review_frequency_months=6,
    ),
    DUAType.ACADEMIC_COLLABORATION: RetentionPolicy(
        retention_period_years=5,
        destruction_method="cryptographic_erasure",
        destruction_certification=True,
        review_frequency_months=12,
    ),
    DUAType.GOVERNMENT: RetentionPolicy(
        retention_period_years=10,
        destruction_method="nist_800_88_purge",
        destruction_certification=True,
        review_frequency_months=6,
    ),
}


# ---------------------------------------------------------------------------
# DUA Generator
# ---------------------------------------------------------------------------


class DUAGenerator:
    """Generates Data Use Agreement templates for clinical trial data sharing.

    Produces structured DUA documents with appropriate security
    requirements, retention policies, and regulatory clauses
    based on the agreement type and data category.

    Features:
        - Template generation for 5 DUA types
        - Security requirements scaled by data sensitivity
        - Retention policies per agreement type
        - Section-based document structure
        - Amendment tracking
    """

    def __init__(self) -> None:
        self._documents: dict[str, DUADocument] = {}
        self._next_id: int = 1

    def generate(
        self,
        dua_type: DUAType | str,
        title: str = "",
        parties: list[DUAParty] | None = None,
        data_category: DataCategory = DataCategory.LIMITED_DATA_SET,
        data_description: str = "",
        purpose: str = "",
        effective_date: str = "",
        expiration_date: str = "",
    ) -> DUADocument:
        """Generate a new DUA document from template.

        Args:
            dua_type: Type of agreement.
            title: Agreement title.
            parties: Parties to the agreement.
            data_category: Category of data covered.
            data_description: Description of shared data.
            purpose: Permitted use purpose.
            effective_date: Effective date (ISO format).
            expiration_date: Expiration date (ISO format).

        Returns:
            Generated DUADocument with all sections populated.
        """
        if isinstance(dua_type, str):
            dua_type = DUAType(dua_type)

        dua_id = f"DUA-{self._next_id:04d}"
        self._next_id += 1

        if not title:
            title = f"{dua_type.value.replace('_', ' ').title()} Data Use Agreement"

        security = _DEFAULT_SECURITY.get(dua_type, SecurityRequirements())
        retention = _DEFAULT_RETENTION.get(dua_type, RetentionPolicy())

        # Scale security for fully identified data
        if data_category == DataCategory.FULLY_IDENTIFIED:
            security.mfa_required = True
            security.data_loss_prevention = True
            security.network_segmentation = True
            security.annual_security_audit = True
            security.background_checks = True

        doc = DUADocument(
            dua_id=dua_id,
            dua_type=dua_type,
            title=title,
            parties=list(parties or []),
            data_category=data_category,
            data_description=data_description,
            purpose=purpose,
            security_requirements=security,
            retention_policy=retention,
            effective_date=effective_date,
            expiration_date=expiration_date,
        )

        doc.sections = self._generate_sections(doc)
        self._documents[dua_id] = doc

        logger.info(
            "Generated DUA '%s' (type=%s, category=%s)",
            dua_id,
            dua_type.value,
            data_category.value,
        )
        return doc

    def get_document(self, dua_id: str) -> DUADocument | None:
        """Retrieve a DUA document by ID."""
        return self._documents.get(dua_id)

    def update_status(self, dua_id: str, status: DUAStatus | str) -> bool:
        """Update the status of a DUA document."""
        doc = self._documents.get(dua_id)
        if doc is None:
            return False
        if isinstance(status, str):
            status = DUAStatus(status)
        doc.status = status
        logger.info("DUA '%s' status updated to '%s'", dua_id, status.value)
        return True

    def add_amendment(self, dua_id: str, description: str, approved_by: str = "") -> bool:
        """Add an amendment to a DUA document."""
        doc = self._documents.get(dua_id)
        if doc is None:
            return False
        doc.amendments.append(
            {
                "description": description,
                "approved_by": approved_by,
                "date": datetime.now(timezone.utc).isoformat(),
            }
        )
        return True

    def list_documents(self, status: DUAStatus | None = None) -> list[DUADocument]:
        """List all DUA documents with optional status filter."""
        docs = list(self._documents.values())
        if status is not None:
            docs = [d for d in docs if d.status == status]
        return docs

    # ------------------------------------------------------------------
    # Section generation
    # ------------------------------------------------------------------

    def _generate_sections(self, doc: DUADocument) -> list[dict[str, str]]:
        """Generate the standard sections for a DUA document."""
        sections: list[dict[str, str]] = []

        # 1. Preamble
        party_names = ", ".join(p.organization_name for p in doc.parties) if doc.parties else "[PARTIES]"
        sections.append(
            {
                "title": "Preamble",
                "content": (
                    f'This Data Use Agreement ("Agreement") is entered into by and between '
                    f"{party_names} for the purpose of sharing {doc.data_category.value.replace('_', ' ')} "
                    f"data in connection with oncology clinical trial research."
                ),
            }
        )

        # 2. Definitions
        sections.append(
            {
                "title": "Definitions",
                "content": (
                    '"Covered Entity" means an entity subject to HIPAA. '
                    '"Limited Data Set" means PHI excluding direct identifiers per 45 CFR 164.514(e)(2). '
                    '"De-identified Data" means data meeting Safe Harbor requirements per 45 CFR 164.514(b). '
                    '"Data Recipient" means the party receiving data under this Agreement. '
                    '"Data Provider" means the party furnishing data under this Agreement.'
                ),
            }
        )

        # 3. Permitted uses
        sections.append(
            {
                "title": "Permitted Uses and Disclosures",
                "content": (
                    f"Data may be used solely for: {doc.purpose or 'oncology clinical trial research'}. "
                    "Data Recipient shall not use or disclose the data for any purpose other than "
                    "as permitted by this Agreement or as required by law. Data Recipient shall not "
                    "attempt to re-identify any de-identified data or contact any individuals whose "
                    "data is included in the dataset."
                ),
            }
        )

        # 4. Security requirements
        sec = doc.security_requirements
        if sec is not None:
            security_items: list[str] = []
            if sec.encryption_at_rest:
                security_items.append(f"Encryption at rest ({sec.encryption_standard})")
            if sec.encryption_in_transit:
                security_items.append("Encryption in transit (TLS 1.2+)")
            if sec.mfa_required:
                security_items.append("Multi-factor authentication for all data access")
            if sec.access_logging:
                security_items.append("Comprehensive access logging with 21 CFR Part 11 audit trail")
            if sec.data_loss_prevention:
                security_items.append("Data loss prevention controls")
            if sec.network_segmentation:
                security_items.append("Network segmentation for data processing environment")
            if sec.incident_response_plan:
                security_items.append("Documented incident response plan")
            if sec.annual_security_audit:
                security_items.append("Annual third-party security audit")
            if sec.background_checks:
                security_items.append("Background checks for all personnel with data access")

            sections.append(
                {
                    "title": "Security Requirements",
                    "content": (
                        "Data Recipient shall implement and maintain the following security controls: "
                        + "; ".join(security_items)
                        + "."
                    ),
                }
            )

        # 5. Data retention and destruction
        ret = doc.retention_policy
        if ret is not None:
            sections.append(
                {
                    "title": "Data Retention and Destruction",
                    "content": (
                        f"Data shall be retained for no longer than {ret.retention_period_years} years "
                        f"from the effective date of this Agreement. Upon expiration, all data and copies "
                        f"shall be destroyed using {ret.destruction_method.replace('_', ' ')} method. "
                        + ("Written certification of destruction is required." if ret.destruction_certification else "")
                        + f" Retention shall be reviewed every {ret.review_frequency_months} months."
                    ),
                }
            )

        # 6. Breach notification
        sections.append(
            {
                "title": "Breach Notification",
                "content": (
                    "Data Recipient shall notify Data Provider of any breach or suspected breach "
                    "of this Agreement within 24 hours of discovery. Notification shall include: "
                    "description of the breach, types of data involved, number of individuals affected, "
                    "steps taken to mitigate harm, and corrective actions planned. Breach notification "
                    "shall comply with HIPAA Breach Notification Rule (45 CFR 164.400-414)."
                ),
            }
        )

        # 7. Term and termination
        sections.append(
            {
                "title": "Term and Termination",
                "content": (
                    f"This Agreement shall be effective from {doc.effective_date or '[EFFECTIVE_DATE]'} "
                    f"until {doc.expiration_date or '[EXPIRATION_DATE]'}. Either party may terminate "
                    "this Agreement upon 30 days written notice. Upon termination, Data Recipient "
                    "shall return or destroy all data as specified in the Data Retention section."
                ),
            }
        )

        # 8. Regulatory compliance
        sections.append(
            {
                "title": "Regulatory Compliance",
                "content": (
                    "Both parties shall comply with all applicable federal and state laws including "
                    "HIPAA Privacy Rule (45 CFR Parts 160, 164), HIPAA Security Rule, HITECH Act, "
                    "and applicable state privacy laws. For international data transfers, compliance "
                    "with GDPR and applicable cross-border data transfer mechanisms is required."
                ),
            }
        )

        # 9. Signatures
        sections.append(
            {
                "title": "Signatures",
                "content": (
                    "By signing below, the authorized representatives of each party agree to "
                    "the terms and conditions of this Data Use Agreement. Electronic signatures "
                    "are acceptable per 21 CFR Part 11 requirements."
                ),
            }
        )

        return sections
