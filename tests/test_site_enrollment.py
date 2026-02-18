"""Tests for site enrollment management.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.
"""

from federated.site_enrollment import SiteEnrollmentManager, SiteStatus


class TestSiteEnrollmentManager:
    def test_enroll_site(self):
        mgr = SiteEnrollmentManager("STUDY_01")
        profile = mgr.enroll_site("site_A", "Hospital A", patient_count=100)
        assert profile.status == SiteStatus.PENDING
        assert profile.institution_name == "Hospital A"

    def test_activate_site_success(self):
        mgr = SiteEnrollmentManager("STUDY_01", min_patients_per_site=10)
        mgr.enroll_site("site_A", "Hospital A", patient_count=50)
        mgr.mark_data_ready("site_A")
        mgr.mark_compliance_passed("site_A")
        assert mgr.activate_site("site_A") is True
        assert mgr.get_site("site_A").status == SiteStatus.ACTIVE

    def test_activate_site_insufficient_patients(self):
        mgr = SiteEnrollmentManager("STUDY_01", min_patients_per_site=100)
        mgr.enroll_site("site_A", "Hospital A", patient_count=10)
        mgr.mark_data_ready("site_A")
        mgr.mark_compliance_passed("site_A")
        assert mgr.activate_site("site_A") is False

    def test_activate_site_not_compliant(self):
        mgr = SiteEnrollmentManager("STUDY_01")
        mgr.enroll_site("site_A", "Hospital A", patient_count=50)
        mgr.mark_data_ready("site_A")
        # compliance not passed
        assert mgr.activate_site("site_A") is False

    def test_suspend_and_reactivate(self):
        mgr = SiteEnrollmentManager("STUDY_01")
        mgr.enroll_site("site_A", "Hospital A", patient_count=50)
        mgr.mark_data_ready("site_A")
        mgr.mark_compliance_passed("site_A")
        mgr.activate_site("site_A")

        assert mgr.suspend_site("site_A", "quality issue") is True
        assert mgr.get_site("site_A").status == SiteStatus.SUSPENDED
        assert mgr.reactivate_site("site_A") is True
        assert mgr.get_site("site_A").status == SiteStatus.ACTIVE

    def test_withdraw_site(self):
        mgr = SiteEnrollmentManager("STUDY_01")
        mgr.enroll_site("site_A", "Hospital A")
        assert mgr.withdraw_site("site_A") is True
        assert mgr.get_site("site_A").status == SiteStatus.WITHDRAWN

    def test_get_active_sites(self):
        mgr = SiteEnrollmentManager("STUDY_01", min_patients_per_site=5)
        for sid in ["s1", "s2", "s3"]:
            mgr.enroll_site(sid, f"Hospital {sid}", patient_count=50)
            mgr.mark_data_ready(sid)
            mgr.mark_compliance_passed(sid)
            mgr.activate_site(sid)

        active = mgr.get_active_sites()
        assert len(active) == 3

    def test_select_sites_for_round(self):
        mgr = SiteEnrollmentManager("STUDY_01", min_patients_per_site=5)
        for sid in ["s1", "s2", "s3"]:
            mgr.enroll_site(sid, f"Hospital {sid}", patient_count=50)
            mgr.mark_data_ready(sid)
            mgr.mark_compliance_passed(sid)
            mgr.activate_site(sid)

        selected = mgr.select_sites_for_round(max_sites=2)
        assert len(selected) == 2

    def test_quality_score_auto_suspend(self):
        mgr = SiteEnrollmentManager("STUDY_01", min_patients_per_site=5, min_quality_score=0.5)
        mgr.enroll_site("s1", "Hospital A", patient_count=50)
        mgr.mark_data_ready("s1")
        mgr.mark_compliance_passed("s1")
        mgr.activate_site("s1")
        mgr.update_quality_score("s1", 0.1)

        selected = mgr.select_sites_for_round()
        assert len(selected) == 0
        assert mgr.get_site("s1").status == SiteStatus.SUSPENDED

    def test_enrollment_summary(self):
        mgr = SiteEnrollmentManager("STUDY_01", min_patients_per_site=5)
        mgr.enroll_site("s1", "Hospital A", patient_count=50)
        mgr.enroll_site("s2", "Hospital B", patient_count=30)
        summary = mgr.get_enrollment_summary()
        assert summary["total_sites"] == 2
        assert summary["total_patients"] == 80

    def test_audit_trail(self):
        mgr = SiteEnrollmentManager("STUDY_01")
        mgr.enroll_site("s1", "Hospital A")
        trail = mgr.get_audit_trail()
        assert len(trail) >= 1
        assert trail[0]["event"] == "site_enrolled"
