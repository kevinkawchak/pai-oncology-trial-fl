"""Multi-jurisdiction regulatory intelligence tracker.

Monitors regulatory updates, guidance documents, and compliance
deadlines across major health authorities: FDA (US), EMA (EU),
PMDA (Japan), TGA (Australia), and Health Canada.

References:
    - FDA 21 CFR Parts 11, 50, 56, 312, 820
    - EU MDR (2017/745), EU AI Act (2024/1689)
    - ICH E6(R3) published September 2025
    - PMDA Guidance on AI-Based Medical Devices (2024)
    - TGA Regulatory Framework for AI Medical Devices (2023)

DISCLAIMER: RESEARCH USE ONLY — Not for clinical decision-making.
LICENSE: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Jurisdiction(str, Enum):
    """Regulatory jurisdictions monitored."""

    FDA = "fda"  # US Food and Drug Administration
    EMA = "ema"  # European Medicines Agency
    PMDA = "pmda"  # Japan Pharmaceuticals and Medical Devices Agency
    TGA = "tga"  # Australian Therapeutic Goods Administration
    HEALTH_CANADA = "health_canada"  # Health Canada


class UpdateType(str, Enum):
    """Types of regulatory updates."""

    GUIDANCE_DRAFT = "guidance_draft"
    GUIDANCE_FINAL = "guidance_final"
    REGULATION_PROPOSED = "regulation_proposed"
    REGULATION_FINAL = "regulation_final"
    ENFORCEMENT_ACTION = "enforcement_action"
    ADVISORY = "advisory"
    STANDARD_UPDATE = "standard_update"
    RECALL = "recall"
    SAFETY_COMMUNICATION = "safety_communication"


class Priority(str, Enum):
    """Priority level for regulatory updates."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(str, Enum):
    """Status of compliance with a regulatory requirement."""

    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"
    UNDER_REVIEW = "under_review"
    OVERDUE = "overdue"
    IMMINENT = "imminent"


class ActionStatus(str, Enum):
    """Status of a compliance action item."""

    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RegulatoryUpdate:
    """A regulatory update or guidance document.

    Attributes:
        update_id: Unique update identifier.
        jurisdiction: Issuing authority.
        update_type: Type of update.
        title: Title of the update.
        summary: Brief summary.
        reference_number: Official reference number.
        published_date: Publication date.
        effective_date: When the update takes effect.
        comment_deadline: Deadline for public comments (if applicable).
        priority: Assessed priority for the organization.
        url: URL to the official document.
        impact_areas: Areas of the organization affected.
        tags: Classification tags.
    """

    update_id: str
    jurisdiction: Jurisdiction
    update_type: UpdateType
    title: str
    summary: str = ""
    reference_number: str = ""
    published_date: str = ""
    effective_date: str = ""
    comment_deadline: str = ""
    priority: Priority = Priority.MEDIUM
    url: str = ""
    impact_areas: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.published_date:
            self.published_date = datetime.now(timezone.utc).isoformat()


@dataclass
class ComplianceRequirement:
    """A specific regulatory compliance requirement.

    Attributes:
        requirement_id: Unique requirement identifier.
        jurisdiction: Applicable jurisdiction.
        regulation: Source regulation or guidance.
        description: Requirement description.
        deadline: Compliance deadline.
        status: Current compliance status.
        assigned_to: Person or team responsible.
        evidence: Evidence of compliance.
        last_assessed: Last assessment date.
        notes: Additional notes.
    """

    requirement_id: str
    jurisdiction: Jurisdiction
    regulation: str
    description: str
    deadline: str = ""
    status: ComplianceStatus = ComplianceStatus.UNDER_REVIEW
    assigned_to: str = ""
    evidence: str = ""
    last_assessed: str = ""
    notes: str = ""


@dataclass
class ActionItem:
    """A compliance action item.

    Attributes:
        action_id: Unique action identifier.
        requirement_id: Linked compliance requirement.
        description: Action description.
        assigned_to: Responsible person or team.
        due_date: Action due date.
        status: Current status.
        completed_date: Completion date.
        notes: Additional context.
    """

    action_id: str
    requirement_id: str
    description: str
    assigned_to: str = ""
    due_date: str = ""
    status: ActionStatus = ActionStatus.PLANNED
    completed_date: str = ""
    notes: str = ""


@dataclass
class JurisdictionProfile:
    """Profile of a regulatory jurisdiction.

    Attributes:
        jurisdiction: The jurisdiction.
        authority_name: Full name of the regulatory authority.
        ai_ml_framework: Current AI/ML regulatory framework.
        key_regulations: Key applicable regulations.
        recent_updates: Count of recent updates tracked.
        compliance_requirements: Number of active requirements.
    """

    jurisdiction: Jurisdiction
    authority_name: str
    ai_ml_framework: str = ""
    key_regulations: list[str] = field(default_factory=list)
    recent_updates: int = 0
    compliance_requirements: int = 0


# ---------------------------------------------------------------------------
# Built-in jurisdiction profiles
# ---------------------------------------------------------------------------

_JURISDICTION_PROFILES: dict[Jurisdiction, JurisdictionProfile] = {
    Jurisdiction.FDA: JurisdictionProfile(
        jurisdiction=Jurisdiction.FDA,
        authority_name="U.S. Food and Drug Administration",
        ai_ml_framework="FDA AI/ML-Based SaMD Framework (2021), PCCP Guidance (Aug 2025), QMSR (Feb 2026)",
        key_regulations=[
            "21 CFR Part 11 — Electronic Records; Electronic Signatures",
            "21 CFR Part 820 — Quality Management System Regulation (QMSR effective Feb 2026)",
            "21 CFR Part 812 — Investigational Device Exemptions",
            "FDA Draft Guidance — AI/ML-Enabled Device Software Functions (Jan 2025)",
            "FDA PCCP Guidance — Predetermined Change Control Plans (Aug 2025)",
        ],
    ),
    Jurisdiction.EMA: JurisdictionProfile(
        jurisdiction=Jurisdiction.EMA,
        authority_name="European Medicines Agency",
        ai_ml_framework="EU AI Act (2024/1689), EU MDR (2017/745)",
        key_regulations=[
            "EU AI Act (Regulation 2024/1689) — High-risk AI systems",
            "EU MDR (Regulation 2017/745) — Medical Device Regulation",
            "GDPR (Regulation 2016/679) — Data protection",
            "EU IVDR (Regulation 2017/746) — In Vitro Diagnostic Regulation",
        ],
    ),
    Jurisdiction.PMDA: JurisdictionProfile(
        jurisdiction=Jurisdiction.PMDA,
        authority_name="Pharmaceuticals and Medical Devices Agency (Japan)",
        ai_ml_framework="PMDA AI-Based Medical Device Guidance (2024)",
        key_regulations=[
            "Pharmaceutical and Medical Device Act (PMD Act)",
            "PMDA Guidance on AI-Based Medical Devices (2024)",
            "Japan Good Clinical Practice (J-GCP)",
        ],
    ),
    Jurisdiction.TGA: JurisdictionProfile(
        jurisdiction=Jurisdiction.TGA,
        authority_name="Therapeutic Goods Administration (Australia)",
        ai_ml_framework="TGA AI/ML Medical Device Framework (2023)",
        key_regulations=[
            "Therapeutic Goods Act 1989",
            "TGA Regulatory Framework for AI Medical Devices",
            "Australian Privacy Act 1988",
        ],
    ),
    Jurisdiction.HEALTH_CANADA: JurisdictionProfile(
        jurisdiction=Jurisdiction.HEALTH_CANADA,
        authority_name="Health Canada",
        ai_ml_framework="Health Canada AI/ML SaMD Guidance (2024)",
        key_regulations=[
            "Medical Devices Regulations (SOR/98-282)",
            "Health Canada Pre-Market Guidance for ML-Enabled Devices",
            "PIPEDA — Personal Information Protection",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Regulatory Tracker
# ---------------------------------------------------------------------------


class RegulatoryTracker:
    """Multi-jurisdiction regulatory intelligence tracker.

    Monitors regulatory updates, manages compliance requirements,
    and tracks action items across FDA, EMA, PMDA, TGA, and
    Health Canada.
    """

    def __init__(self) -> None:
        self._updates: dict[str, RegulatoryUpdate] = {}
        self._requirements: dict[str, ComplianceRequirement] = {}
        self._actions: dict[str, ActionItem] = {}
        self._next_update_id: int = 1
        self._next_req_id: int = 1
        self._next_action_id: int = 1

    # ------------------------------------------------------------------
    # Regulatory updates
    # ------------------------------------------------------------------

    def add_update(
        self,
        jurisdiction: Jurisdiction | str,
        update_type: UpdateType | str,
        title: str,
        summary: str = "",
        reference_number: str = "",
        effective_date: str = "",
        priority: Priority | str = Priority.MEDIUM,
        impact_areas: list[str] | None = None,
    ) -> RegulatoryUpdate:
        """Add a regulatory update to the tracker.

        Args:
            jurisdiction: Issuing authority.
            update_type: Type of regulatory update.
            title: Title of the update.
            summary: Brief summary.
            reference_number: Official reference number.
            effective_date: When the update takes effect.
            priority: Assessed priority level.
            impact_areas: Organizational areas affected.

        Returns:
            The created RegulatoryUpdate.
        """
        if isinstance(jurisdiction, str):
            jurisdiction = Jurisdiction(jurisdiction)
        if isinstance(update_type, str):
            update_type = UpdateType(update_type)
        if isinstance(priority, str):
            priority = Priority(priority)

        update_id = f"RU-{self._next_update_id:04d}"
        self._next_update_id += 1

        update = RegulatoryUpdate(
            update_id=update_id,
            jurisdiction=jurisdiction,
            update_type=update_type,
            title=title,
            summary=summary,
            reference_number=reference_number,
            effective_date=effective_date,
            priority=priority,
            impact_areas=list(impact_areas or []),
        )
        self._updates[update_id] = update
        logger.info("Regulatory update added: %s — %s (%s)", update_id, title, jurisdiction.value)
        return update

    def get_update(self, update_id: str) -> RegulatoryUpdate | None:
        """Retrieve a regulatory update."""
        return self._updates.get(update_id)

    def get_recent_updates(
        self,
        days: int = 90,
        jurisdiction: Jurisdiction | None = None,
    ) -> list[RegulatoryUpdate]:
        """Get updates from the last N days.

        Args:
            days: Number of days to look back.
            jurisdiction: Optional jurisdiction filter.

        Returns:
            List of recent updates, sorted by date (newest first).
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days)
        recent: list[RegulatoryUpdate] = []

        for update in self._updates.values():
            if jurisdiction is not None and update.jurisdiction != jurisdiction:
                continue
            try:
                pub_date = datetime.fromisoformat(update.published_date)
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=timezone.utc)
                if pub_date >= cutoff:
                    recent.append(update)
            except (ValueError, TypeError):
                continue

        recent.sort(key=lambda u: u.published_date, reverse=True)
        return recent

    # ------------------------------------------------------------------
    # Compliance requirements
    # ------------------------------------------------------------------

    def add_requirement(
        self,
        jurisdiction: Jurisdiction | str,
        regulation: str,
        description: str,
        deadline: str = "",
        assigned_to: str = "",
    ) -> ComplianceRequirement:
        """Add a compliance requirement.

        Args:
            jurisdiction: Applicable jurisdiction.
            regulation: Source regulation.
            description: Requirement description.
            deadline: Compliance deadline.
            assigned_to: Responsible person or team.

        Returns:
            The created ComplianceRequirement.
        """
        if isinstance(jurisdiction, str):
            jurisdiction = Jurisdiction(jurisdiction)

        req_id = f"CR-{self._next_req_id:04d}"
        self._next_req_id += 1

        req = ComplianceRequirement(
            requirement_id=req_id,
            jurisdiction=jurisdiction,
            regulation=regulation,
            description=description,
            deadline=deadline,
            assigned_to=assigned_to,
        )
        self._requirements[req_id] = req
        return req

    def update_requirement_status(
        self,
        requirement_id: str,
        status: ComplianceStatus | str,
        evidence: str = "",
    ) -> bool:
        """Update the compliance status of a requirement."""
        req = self._requirements.get(requirement_id)
        if req is None:
            return False
        if isinstance(status, str):
            status = ComplianceStatus(status)
        req.status = status
        req.last_assessed = datetime.now(timezone.utc).isoformat()
        if evidence:
            req.evidence = evidence
        return True

    def get_overdue_requirements(self) -> list[ComplianceRequirement]:
        """Get requirements that are past their deadline.

        Checks ordering: OVERDUE (past deadline) is checked BEFORE
        IMMINENT (approaching deadline) to ensure overdue status
        is always reachable.
        """
        overdue: list[ComplianceRequirement] = []
        now = datetime.now(timezone.utc)

        for req in self._requirements.values():
            if req.status in (ComplianceStatus.COMPLIANT, ComplianceStatus.NOT_APPLICABLE):
                continue
            if not req.deadline:
                continue
            try:
                deadline = datetime.fromisoformat(req.deadline)
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                # Check OVERDUE first (past-due before imminent)
                if now > deadline:
                    req.status = ComplianceStatus.OVERDUE
                    overdue.append(req)
                elif (deadline - now).days <= 30:
                    req.status = ComplianceStatus.IMMINENT
            except (ValueError, TypeError):
                continue

        return overdue

    def get_requirements_by_jurisdiction(
        self,
        jurisdiction: Jurisdiction,
    ) -> list[ComplianceRequirement]:
        """Get all requirements for a specific jurisdiction."""
        return [r for r in self._requirements.values() if r.jurisdiction == jurisdiction]

    # ------------------------------------------------------------------
    # Action items
    # ------------------------------------------------------------------

    def add_action(
        self,
        requirement_id: str,
        description: str,
        assigned_to: str = "",
        due_date: str = "",
    ) -> str | None:
        """Add an action item linked to a requirement.

        Returns:
            The action ID, or None if the requirement was not found.
        """
        if requirement_id not in self._requirements:
            return None

        action_id = f"ACT-{self._next_action_id:04d}"
        self._next_action_id += 1

        action = ActionItem(
            action_id=action_id,
            requirement_id=requirement_id,
            description=description,
            assigned_to=assigned_to,
            due_date=due_date,
        )
        self._actions[action_id] = action
        return action_id

    def complete_action(self, action_id: str, notes: str = "") -> bool:
        """Mark an action item as completed."""
        action = self._actions.get(action_id)
        if action is None:
            return False
        action.status = ActionStatus.COMPLETED
        action.completed_date = datetime.now(timezone.utc).isoformat()
        if notes:
            action.notes = notes
        return True

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_jurisdiction_profile(self, jurisdiction: Jurisdiction) -> JurisdictionProfile:
        """Get the profile for a jurisdiction with current counts."""
        profile = _JURISDICTION_PROFILES.get(jurisdiction)
        if profile is None:
            return JurisdictionProfile(jurisdiction=jurisdiction, authority_name="Unknown")

        # Create a copy with current counts
        return JurisdictionProfile(
            jurisdiction=profile.jurisdiction,
            authority_name=profile.authority_name,
            ai_ml_framework=profile.ai_ml_framework,
            key_regulations=list(profile.key_regulations),
            recent_updates=sum(1 for u in self._updates.values() if u.jurisdiction == jurisdiction),
            compliance_requirements=sum(1 for r in self._requirements.values() if r.jurisdiction == jurisdiction),
        )

    def get_dashboard(self) -> dict[str, Any]:
        """Generate a regulatory intelligence dashboard summary."""
        total_updates = len(self._updates)
        total_requirements = len(self._requirements)
        total_actions = len(self._actions)

        overdue = self.get_overdue_requirements()

        by_jurisdiction: dict[str, int] = {}
        for req in self._requirements.values():
            key = req.jurisdiction.value
            by_jurisdiction[key] = by_jurisdiction.get(key, 0) + 1

        by_status: dict[str, int] = {}
        for req in self._requirements.values():
            key = req.status.value
            by_status[key] = by_status.get(key, 0) + 1

        return {
            "total_updates": total_updates,
            "total_requirements": total_requirements,
            "total_actions": total_actions,
            "overdue_requirements": len(overdue),
            "requirements_by_jurisdiction": by_jurisdiction,
            "requirements_by_status": by_status,
            "jurisdictions_monitored": [j.value for j in Jurisdiction],
        }
