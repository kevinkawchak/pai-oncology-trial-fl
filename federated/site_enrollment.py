"""Site enrollment management for federated oncology trials.

Tracks which hospital sites have enrolled in a federated study,
validates their readiness (data availability, consent, compliance),
and manages the lifecycle of site participation including enrollment,
suspension, and withdrawal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class SiteStatus(str, Enum):
    """Lifecycle status of an enrolled site."""

    PENDING = "pending"
    ENROLLED = "enrolled"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    WITHDRAWN = "withdrawn"


@dataclass
class SiteProfile:
    """Profile of a participating site in a federated study.

    Attributes:
        site_id: Unique identifier for the site.
        institution_name: Human-readable institution name.
        status: Current enrollment status.
        enrolled_at: ISO timestamp of enrollment.
        patient_count: Number of consented patients at the site.
        data_ready: Whether the site has passed data-readiness checks.
        compliance_passed: Whether the site passed regulatory compliance.
        capabilities: List of procedure types the site can contribute
            (e.g. ``["imaging", "biopsy", "surgery"]``).
        contact_email: De-identified contact for the site PI.
        rounds_participated: Number of federation rounds participated in.
        last_active: ISO timestamp of last participation.
        quality_score: Rolling quality score based on contribution
            consistency and data integrity checks [0, 1].
    """

    site_id: str
    institution_name: str
    status: SiteStatus = SiteStatus.PENDING
    enrolled_at: str | None = None
    patient_count: int = 0
    data_ready: bool = False
    compliance_passed: bool = False
    capabilities: list[str] = field(default_factory=list)
    contact_email: str = ""
    rounds_participated: int = 0
    last_active: str | None = None
    quality_score: float = 1.0


class SiteEnrollmentManager:
    """Manages multi-site enrollment for a federated oncology study.

    Provides an enrollment workflow with readiness validation,
    quality-weighted site selection for each training round, and
    lifecycle management (suspend/withdraw).

    Args:
        study_id: Identifier for the federated study.
        min_patients_per_site: Minimum patients required for enrollment.
        min_quality_score: Minimum quality score to remain active.
    """

    def __init__(
        self,
        study_id: str,
        min_patients_per_site: int = 10,
        min_quality_score: float = 0.3,
    ):
        self.study_id = study_id
        self.min_patients_per_site = min_patients_per_site
        self.min_quality_score = min_quality_score
        self._sites: dict[str, SiteProfile] = {}
        self._audit_trail: list[dict] = []

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll_site(
        self,
        site_id: str,
        institution_name: str,
        patient_count: int = 0,
        capabilities: list[str] | None = None,
        contact_email: str = "",
    ) -> SiteProfile:
        """Enroll a new site into the federated study.

        The site starts in ``PENDING`` status until it passes readiness
        checks.

        Args:
            site_id: Unique site identifier.
            institution_name: Human-readable name.
            patient_count: Initial patient count.
            capabilities: Procedure types the site supports.
            contact_email: Site PI contact.

        Returns:
            The created SiteProfile.
        """
        now = datetime.now(timezone.utc).isoformat()
        profile = SiteProfile(
            site_id=site_id,
            institution_name=institution_name,
            status=SiteStatus.PENDING,
            enrolled_at=now,
            patient_count=patient_count,
            capabilities=capabilities or [],
            contact_email=contact_email,
        )
        self._sites[site_id] = profile
        self._log("site_enrolled", site_id)
        logger.info("Site %s enrolled in study %s", site_id, self.study_id)
        return profile

    def activate_site(self, site_id: str) -> bool:
        """Activate a site after it passes readiness checks.

        A site must have ``data_ready=True``, ``compliance_passed=True``,
        and at least ``min_patients_per_site`` patients.

        Returns:
            True if activation succeeded.
        """
        profile = self._sites.get(site_id)
        if profile is None:
            logger.warning("Site %s not found", site_id)
            return False

        if not profile.data_ready:
            logger.warning("Site %s data not ready", site_id)
            return False

        if not profile.compliance_passed:
            logger.warning("Site %s compliance not passed", site_id)
            return False

        if profile.patient_count < self.min_patients_per_site:
            logger.warning(
                "Site %s has %d patients (min %d)",
                site_id,
                profile.patient_count,
                self.min_patients_per_site,
            )
            return False

        profile.status = SiteStatus.ACTIVE
        self._log("site_activated", site_id)
        logger.info("Site %s activated", site_id)
        return True

    def mark_data_ready(self, site_id: str, ready: bool = True) -> None:
        """Mark a site's data as ready (or not) for training."""
        profile = self._sites.get(site_id)
        if profile:
            profile.data_ready = ready

    def mark_compliance_passed(self, site_id: str, passed: bool = True) -> None:
        """Mark a site's compliance status."""
        profile = self._sites.get(site_id)
        if profile:
            profile.compliance_passed = passed

    def update_patient_count(self, site_id: str, count: int) -> None:
        """Update the patient count for a site."""
        profile = self._sites.get(site_id)
        if profile:
            profile.patient_count = count

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def suspend_site(self, site_id: str, reason: str = "") -> bool:
        """Temporarily suspend a site from participating."""
        profile = self._sites.get(site_id)
        if profile is None:
            return False
        profile.status = SiteStatus.SUSPENDED
        self._log("site_suspended", site_id, reason=reason)
        logger.info("Site %s suspended: %s", site_id, reason)
        return True

    def withdraw_site(self, site_id: str, reason: str = "") -> bool:
        """Permanently withdraw a site from the study."""
        profile = self._sites.get(site_id)
        if profile is None:
            return False
        profile.status = SiteStatus.WITHDRAWN
        self._log("site_withdrawn", site_id, reason=reason)
        logger.info("Site %s withdrawn: %s", site_id, reason)
        return True

    def reactivate_site(self, site_id: str) -> bool:
        """Reactivate a previously suspended site."""
        profile = self._sites.get(site_id)
        if profile is None or profile.status != SiteStatus.SUSPENDED:
            return False
        profile.status = SiteStatus.ACTIVE
        self._log("site_reactivated", site_id)
        return True

    # ------------------------------------------------------------------
    # Site selection and quality
    # ------------------------------------------------------------------

    def get_active_sites(self) -> list[SiteProfile]:
        """Return all currently active sites."""
        return [s for s in self._sites.values() if s.status == SiteStatus.ACTIVE]

    def select_sites_for_round(
        self,
        max_sites: int | None = None,
        required_capabilities: list[str] | None = None,
    ) -> list[SiteProfile]:
        """Select sites for a training round based on quality scores.

        Active sites are sorted by quality score (descending).  Sites
        whose quality has dropped below ``min_quality_score`` are
        automatically suspended.

        Args:
            max_sites: Maximum number of sites to select.
            required_capabilities: Only include sites supporting all
                listed capabilities.

        Returns:
            List of selected SiteProfiles.
        """
        active = self.get_active_sites()

        # Filter by capabilities
        if required_capabilities:
            active = [s for s in active if all(c in s.capabilities for c in required_capabilities)]

        # Auto-suspend low-quality sites
        eligible: list[SiteProfile] = []
        for site in active:
            if site.quality_score < self.min_quality_score:
                self.suspend_site(site.site_id, reason="quality below threshold")
            else:
                eligible.append(site)

        # Sort by quality (best first)
        eligible.sort(key=lambda s: s.quality_score, reverse=True)

        if max_sites is not None:
            eligible = eligible[:max_sites]

        return eligible

    def record_round_participation(self, site_id: str) -> None:
        """Record that a site participated in a training round."""
        profile = self._sites.get(site_id)
        if profile:
            profile.rounds_participated += 1
            profile.last_active = datetime.now(timezone.utc).isoformat()

    def update_quality_score(self, site_id: str, new_score: float) -> None:
        """Update a site's quality score.

        Scores are clamped to [0, 1].
        """
        profile = self._sites.get(site_id)
        if profile:
            profile.quality_score = max(0.0, min(1.0, new_score))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_site(self, site_id: str) -> SiteProfile | None:
        """Retrieve a site profile by ID."""
        return self._sites.get(site_id)

    def get_enrollment_summary(self) -> dict:
        """Return a summary of all enrolled sites."""
        status_counts: dict[str, int] = {}
        total_patients = 0
        for s in self._sites.values():
            key = s.status.value
            status_counts[key] = status_counts.get(key, 0) + 1
            total_patients += s.patient_count

        return {
            "study_id": self.study_id,
            "total_sites": len(self._sites),
            "by_status": status_counts,
            "total_patients": total_patients,
        }

    def get_audit_trail(self) -> list[dict]:
        """Return the full enrollment audit trail."""
        return list(self._audit_trail)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log(self, event: str, site_id: str, **kwargs: str) -> None:
        entry = {
            "event": event,
            "site_id": site_id,
            "study_id": self.study_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self._audit_trail.append(entry)
