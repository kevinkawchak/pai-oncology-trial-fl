"""
Real-Time Adaptive Monitoring Agent with Streaming Multi-Modal Clinical Data.

CLINICAL CONTEXT:
    This module implements a real-time adaptive monitoring agent that processes
    streaming multi-modal clinical data (vital signs, laboratory results,
    imaging reports) for oncology clinical trial patients. The agent performs
    cross-modal correlation detection, maintains adaptive alert thresholds
    that adjust based on patient baseline and treatment phase, and generates
    prioritized clinical alerts with escalation pathways.

    The agent operates as an event-driven system that continuously ingests
    data streams, applies sliding-window analytics, detects anomalies via
    statistical and rule-based methods, and correlates findings across data
    modalities for early detection of adverse events.

USE CASES COVERED:
    1. Vital sign stream processing with adaptive normal ranges based on
       patient demographics, treatment phase, and historical baseline.
    2. Laboratory result monitoring with trend detection, critical value
       alerting, and CTCAE v5.0 toxicity grading.
    3. Imaging report ingestion with RECIST 1.1 response assessment
       tracking and progression detection.
    4. Cross-modal correlation detection linking vital sign changes with
       lab abnormalities and imaging findings.
    5. Adaptive threshold management that evolves based on accumulated
       patient data and treatment response patterns.

FRAMEWORK REQUIREMENTS:
    Required:
        - numpy >= 1.24.0        (https://numpy.org/)
    Optional:
        - anthropic >= 0.39.0    (https://docs.anthropic.com/)
        - openai >= 1.0.0        (https://platform.openai.com/)
        - scikit-learn >= 1.3.0  (https://scikit-learn.org/)

REFERENCES:
    - NCI (2017). CTCAE v5.0. URL: https://ctep.cancer.gov/
    - Eisenhauer et al. (2009). RECIST 1.1. Eur J Cancer 45(2).
    - FDA Guidance: Conduct of Clinical Trials of Medical Products
      during COVID-19 Public Health Emergency (2020, updated 2021).
    - ICH E6(R3) Good Clinical Practice (2023).

DISCLAIMER:
    RESEARCH USE ONLY. This software is provided for research and educational
    purposes only. It has NOT been validated for clinical use, is NOT approved
    by the FDA or any other regulatory body, and MUST NOT be used to make
    clinical decisions or direct patient care. All alerts must be reviewed
    by qualified clinical professionals before any action is taken.

LICENSE: MIT
VERSION: 0.5.0
LAST UPDATED: 2026-02-17
Copyright (c) 2026 PAI Oncology Trial FL Contributors
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports for optional dependencies
# ---------------------------------------------------------------------------
try:
    import anthropic  # type: ignore[import-untyped]

    HAS_ANTHROPIC = True
except ImportError:
    anthropic = None  # type: ignore[assignment]
    HAS_ANTHROPIC = False

try:
    import openai  # type: ignore[import-untyped]

    HAS_OPENAI = True
except ImportError:
    openai = None  # type: ignore[assignment]
    HAS_OPENAI = False

try:
    from sklearn.ensemble import IsolationForest  # type: ignore[import-untyped]

    HAS_SKLEARN = True
except ImportError:
    IsolationForest = None  # type: ignore[assignment,misc]
    HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class DataModality(Enum):
    """Types of clinical data streams."""

    VITAL_SIGNS = "vital_signs"
    LABORATORY = "laboratory"
    IMAGING = "imaging"
    MEDICATION = "medication"
    PATIENT_REPORTED = "patient_reported"


class AlertSeverity(Enum):
    """Clinical alert severity levels."""

    INFO = "info"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Lifecycle status of a clinical alert."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class TreatmentPhase(Enum):
    """Clinical trial treatment phases affecting monitoring thresholds."""

    SCREENING = "screening"
    BASELINE = "baseline"
    INDUCTION = "induction"
    MAINTENANCE = "maintenance"
    FOLLOW_UP = "follow_up"
    OFF_TREATMENT = "off_treatment"


class ToxicityGrade(Enum):
    """CTCAE v5.0 toxicity grading."""

    GRADE_0 = 0
    GRADE_1 = 1
    GRADE_2 = 2
    GRADE_3 = 3
    GRADE_4 = 4
    GRADE_5 = 5


class VitalType(Enum):
    """Types of vital sign measurements."""

    HEART_RATE = "heart_rate"
    SYSTOLIC_BP = "systolic_bp"
    DIASTOLIC_BP = "diastolic_bp"
    TEMPERATURE = "temperature"
    RESPIRATORY_RATE = "respiratory_rate"
    SPO2 = "spo2"
    PAIN_SCORE = "pain_score"


class LabType(Enum):
    """Types of laboratory measurements."""

    WBC = "wbc"
    ANC = "anc"
    HEMOGLOBIN = "hemoglobin"
    PLATELETS = "platelets"
    CREATININE = "creatinine"
    ALT = "alt"
    AST = "ast"
    BILIRUBIN = "bilirubin"
    ALBUMIN = "albumin"
    LDH = "ldh"
    TSH = "tsh"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class VitalSignReading:
    """A single vital sign measurement from a patient monitoring stream."""

    reading_id: str = field(default_factory=lambda: f"VS-{uuid.uuid4().hex[:8].upper()}")
    patient_id: str = ""
    vital_type: VitalType = VitalType.HEART_RATE
    value: float = 0.0
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    source: str = "bedside_monitor"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "reading_id": self.reading_id,
            "patient_id": self.patient_id,
            "vital_type": self.vital_type.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "source": self.source,
        }


@dataclass
class LabResult:
    """A laboratory test result."""

    result_id: str = field(default_factory=lambda: f"LAB-{uuid.uuid4().hex[:8].upper()}")
    patient_id: str = ""
    lab_type: LabType = LabType.WBC
    value: float = 0.0
    unit: str = ""
    reference_low: float = 0.0
    reference_high: float = 0.0
    timestamp: float = field(default_factory=time.time)
    is_critical: bool = False

    def is_abnormal(self) -> bool:
        """Check if the result is outside normal range."""
        return self.value < self.reference_low or self.value > self.reference_high

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "result_id": self.result_id,
            "patient_id": self.patient_id,
            "lab_type": self.lab_type.value,
            "value": self.value,
            "unit": self.unit,
            "reference_low": self.reference_low,
            "reference_high": self.reference_high,
            "is_abnormal": self.is_abnormal(),
            "is_critical": self.is_critical,
            "timestamp": self.timestamp,
        }


@dataclass
class ImagingReport:
    """An imaging assessment report."""

    report_id: str = field(default_factory=lambda: f"IMG-{uuid.uuid4().hex[:8].upper()}")
    patient_id: str = ""
    modality: str = "CT"
    target_lesion_sum_mm: float = 0.0
    baseline_sum_mm: float = 0.0
    nadir_sum_mm: float = 0.0
    new_lesions: bool = False
    response_category: str = "stable_disease"
    timestamp: float = field(default_factory=time.time)
    findings_summary: str = ""

    def compute_response(self) -> str:
        """Compute RECIST 1.1 response category."""
        if self.new_lesions:
            return "progressive_disease"

        if self.baseline_sum_mm <= 0:
            return "not_evaluable"

        change_from_baseline = (self.target_lesion_sum_mm - self.baseline_sum_mm) / self.baseline_sum_mm
        change_from_nadir = (
            (self.target_lesion_sum_mm - self.nadir_sum_mm) / max(self.nadir_sum_mm, 0.01)
            if self.nadir_sum_mm > 0
            else 0.0
        )

        if self.target_lesion_sum_mm <= 0.001:
            return "complete_response"
        elif change_from_baseline <= -0.30:
            return "partial_response"
        elif change_from_nadir >= 0.20:
            return "progressive_disease"
        else:
            return "stable_disease"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "report_id": self.report_id,
            "patient_id": self.patient_id,
            "modality": self.modality,
            "target_lesion_sum_mm": self.target_lesion_sum_mm,
            "baseline_sum_mm": self.baseline_sum_mm,
            "change_from_baseline_pct": (
                (self.target_lesion_sum_mm - self.baseline_sum_mm) / max(self.baseline_sum_mm, 0.01) * 100.0
            ),
            "response_category": self.compute_response(),
            "new_lesions": self.new_lesions,
            "timestamp": self.timestamp,
        }


@dataclass
class ClinicalAlert:
    """A clinical alert generated by the monitoring agent."""

    alert_id: str = field(default_factory=lambda: f"ALT-{uuid.uuid4().hex[:10].upper()}")
    patient_id: str = ""
    severity: AlertSeverity = AlertSeverity.LOW
    status: AlertStatus = AlertStatus.ACTIVE
    title: str = ""
    description: str = ""
    source_modality: DataModality = DataModality.VITAL_SIGNS
    source_readings: list[str] = field(default_factory=list)
    correlation_ids: list[str] = field(default_factory=list)
    recommended_action: str = ""
    escalation_path: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    acknowledged_by: str = ""
    resolved_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "alert_id": self.alert_id,
            "patient_id": self.patient_id,
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "source_modality": self.source_modality.value,
            "source_readings": self.source_readings,
            "correlation_ids": self.correlation_ids,
            "recommended_action": self.recommended_action,
            "escalation_path": self.escalation_path,
            "timestamp": self.timestamp,
        }


@dataclass
class AdaptiveThreshold:
    """An adaptive monitoring threshold that adjusts based on patient data."""

    parameter: str = ""
    baseline_low: float = 0.0
    baseline_high: float = 0.0
    current_low: float = 0.0
    current_high: float = 0.0
    adaptation_factor: float = 1.0
    samples_seen: int = 0
    running_mean: float = 0.0
    running_std: float = 0.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class CrossModalCorrelation:
    """A detected correlation across data modalities."""

    correlation_id: str = field(default_factory=lambda: f"COR-{uuid.uuid4().hex[:8].upper()}")
    modalities: list[str] = field(default_factory=list)
    description: str = ""
    strength: float = 0.0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    clinical_significance: str = ""
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Adaptive threshold manager
# ---------------------------------------------------------------------------
class AdaptiveThresholdManager:
    """Manages adaptive thresholds that evolve with patient data.

    Thresholds start from clinical normal ranges and adapt based on
    observed patient data using Welford's online algorithm for running
    statistics. Treatment phase transitions can trigger threshold resets.
    """

    # Default clinical normal ranges
    _DEFAULT_RANGES: dict[str, tuple[float, float]] = {
        VitalType.HEART_RATE.value: (60.0, 100.0),
        VitalType.SYSTOLIC_BP.value: (90.0, 140.0),
        VitalType.DIASTOLIC_BP.value: (60.0, 90.0),
        VitalType.TEMPERATURE.value: (36.1, 37.5),
        VitalType.RESPIRATORY_RATE.value: (12.0, 20.0),
        VitalType.SPO2.value: (95.0, 100.0),
        LabType.WBC.value: (4.0, 11.0),
        LabType.ANC.value: (1.5, 8.0),
        LabType.HEMOGLOBIN.value: (12.0, 17.0),
        LabType.PLATELETS.value: (150.0, 400.0),
        LabType.CREATININE.value: (0.6, 1.2),
        LabType.ALT.value: (7.0, 56.0),
        LabType.AST.value: (10.0, 40.0),
        LabType.BILIRUBIN.value: (0.1, 1.2),
    }

    def __init__(self, sigma_multiplier: float = 2.5) -> None:
        self._thresholds: dict[str, AdaptiveThreshold] = {}
        self._sigma_multiplier = sigma_multiplier
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Initialize thresholds from default clinical ranges."""
        for param, (low, high) in self._DEFAULT_RANGES.items():
            self._thresholds[param] = AdaptiveThreshold(
                parameter=param,
                baseline_low=low,
                baseline_high=high,
                current_low=low,
                current_high=high,
                running_mean=(low + high) / 2.0,
                running_std=(high - low) / 4.0,
            )

    def update(self, parameter: str, value: float) -> AdaptiveThreshold:
        """Update the adaptive threshold with a new observation.

        Uses Welford's online algorithm for numerically stable running
        mean and standard deviation computation.
        """
        if parameter not in self._thresholds:
            self._thresholds[parameter] = AdaptiveThreshold(
                parameter=parameter,
                baseline_low=value * 0.8,
                baseline_high=value * 1.2,
                current_low=value * 0.8,
                current_high=value * 1.2,
                running_mean=value,
            )

        threshold = self._thresholds[parameter]
        threshold.samples_seen += 1
        n = threshold.samples_seen

        # Welford's online algorithm
        delta = value - threshold.running_mean
        threshold.running_mean += delta / n
        if n > 1:
            delta2 = value - threshold.running_mean
            m2 = threshold.running_std**2 * (n - 2) + delta * delta2 if n > 2 else delta * delta2
            threshold.running_std = float(np.sqrt(max(m2 / (n - 1), 0.0001)))

        # Adapt thresholds (blend baseline with learned range)
        if n >= 5:
            alpha = min(n / 50.0, 0.7)
            learned_low = threshold.running_mean - self._sigma_multiplier * threshold.running_std
            learned_high = threshold.running_mean + self._sigma_multiplier * threshold.running_std
            threshold.current_low = (1 - alpha) * threshold.baseline_low + alpha * learned_low
            threshold.current_high = (1 - alpha) * threshold.baseline_high + alpha * learned_high
            threshold.adaptation_factor = alpha

        threshold.last_updated = time.time()
        return threshold

    def check_value(self, parameter: str, value: float) -> tuple[bool, float]:
        """Check if a value is within adaptive thresholds.

        Returns:
            Tuple of (is_within_threshold, deviation_score).
            deviation_score: 0.0 = at mean, >1.0 = outside threshold.
        """
        threshold = self._thresholds.get(parameter)
        if threshold is None:
            return True, 0.0

        mid = (threshold.current_low + threshold.current_high) / 2.0
        half_range = max((threshold.current_high - threshold.current_low) / 2.0, 0.001)
        deviation = abs(value - mid) / half_range
        within = threshold.current_low <= value <= threshold.current_high

        return within, deviation

    def get_threshold(self, parameter: str) -> Optional[AdaptiveThreshold]:
        """Get the current threshold for a parameter."""
        return self._thresholds.get(parameter)

    def reset_for_phase(self, phase: TreatmentPhase) -> None:
        """Reset adaptation factors for a treatment phase change."""
        phase_adjustments = {
            TreatmentPhase.INDUCTION: 1.2,
            TreatmentPhase.MAINTENANCE: 1.0,
            TreatmentPhase.FOLLOW_UP: 0.9,
        }
        factor = phase_adjustments.get(phase, 1.0)

        for threshold in self._thresholds.values():
            range_width = threshold.baseline_high - threshold.baseline_low
            center = (threshold.baseline_low + threshold.baseline_high) / 2.0
            threshold.current_low = center - (range_width / 2.0) * factor
            threshold.current_high = center + (range_width / 2.0) * factor
            threshold.adaptation_factor = 0.0
            threshold.samples_seen = 0

        logger.info("Thresholds reset for phase %s (factor=%.2f)", phase.value, factor)


# ---------------------------------------------------------------------------
# Cross-modal correlation detector
# ---------------------------------------------------------------------------
class CrossModalCorrelationDetector:
    """Detects correlations across clinical data modalities.

    Maintains sliding windows of recent readings per modality and checks
    for known clinical correlation patterns (e.g., fever + neutropenia,
    rising creatinine + falling urine output).
    """

    # Known cross-modal correlation patterns
    _CORRELATION_PATTERNS: list[dict[str, Any]] = [
        {
            "name": "febrile_neutropenia",
            "modalities": [DataModality.VITAL_SIGNS.value, DataModality.LABORATORY.value],
            "conditions": {"temperature_above": 38.3, "anc_below": 0.5},
            "severity": AlertSeverity.CRITICAL,
            "action": "Initiate febrile neutropenia protocol; obtain blood cultures; start empiric antibiotics",
        },
        {
            "name": "hepatotoxicity_signal",
            "modalities": [DataModality.LABORATORY.value],
            "conditions": {"alt_above_ratio": 3.0, "bilirubin_above_ratio": 2.0},
            "severity": AlertSeverity.HIGH,
            "action": "Hold hepatotoxic medications; obtain hepatology consultation",
        },
        {
            "name": "nephrotoxicity_signal",
            "modalities": [DataModality.LABORATORY.value],
            "conditions": {"creatinine_above_ratio": 1.5},
            "severity": AlertSeverity.HIGH,
            "action": "Assess hydration status; hold nephrotoxic agents; nephrology consult",
        },
        {
            "name": "tumor_lysis_syndrome",
            "modalities": [DataModality.LABORATORY.value, DataModality.VITAL_SIGNS.value],
            "conditions": {"ldh_above_ratio": 2.0, "heart_rate_above": 110},
            "severity": AlertSeverity.CRITICAL,
            "action": "Initiate TLS protocol; aggressive hydration; rasburicase per protocol",
        },
        {
            "name": "immune_related_ae",
            "modalities": [DataModality.LABORATORY.value, DataModality.VITAL_SIGNS.value],
            "conditions": {"tsh_outside_range": True, "temperature_above": 37.8},
            "severity": AlertSeverity.MODERATE,
            "action": "Assess for immune-related adverse event; endocrinology consultation",
        },
    ]

    def __init__(self, window_size: int = 50) -> None:
        self._vitals_window: deque[VitalSignReading] = deque(maxlen=window_size)
        self._labs_window: deque[LabResult] = deque(maxlen=window_size)
        self._imaging_window: deque[ImagingReport] = deque(maxlen=10)
        self._detected_correlations: list[CrossModalCorrelation] = []

    def ingest_vital(self, reading: VitalSignReading) -> None:
        """Ingest a vital sign reading into the sliding window."""
        self._vitals_window.append(reading)

    def ingest_lab(self, result: LabResult) -> None:
        """Ingest a lab result into the sliding window."""
        self._labs_window.append(result)

    def ingest_imaging(self, report: ImagingReport) -> None:
        """Ingest an imaging report into the sliding window."""
        self._imaging_window.append(report)

    def detect_correlations(self, patient_id: str) -> list[CrossModalCorrelation]:
        """Check for cross-modal correlations in recent data.

        Returns:
            List of newly detected correlations.
        """
        new_correlations: list[CrossModalCorrelation] = []

        # Get latest readings per type
        latest_vitals: dict[str, float] = {}
        for v in self._vitals_window:
            if v.patient_id == patient_id:
                latest_vitals[v.vital_type.value] = v.value

        latest_labs: dict[str, tuple[float, float, float]] = {}
        for lab in self._labs_window:
            if lab.patient_id == patient_id:
                latest_labs[lab.lab_type.value] = (lab.value, lab.reference_low, lab.reference_high)

        # Check febrile neutropenia
        temp = latest_vitals.get(VitalType.TEMPERATURE.value, 37.0)
        anc_data = latest_labs.get(LabType.ANC.value)
        if temp > 38.3 and anc_data is not None and anc_data[0] < 0.5:
            corr = CrossModalCorrelation(
                modalities=[DataModality.VITAL_SIGNS.value, DataModality.LABORATORY.value],
                description="Febrile neutropenia detected: temperature > 38.3C with ANC < 0.5 x10^9/L",
                strength=0.95,
                evidence=[
                    {"type": "temperature", "value": temp, "threshold": 38.3},
                    {"type": "anc", "value": anc_data[0], "threshold": 0.5},
                ],
                clinical_significance="CRITICAL - Requires immediate intervention per febrile neutropenia protocol",
            )
            new_correlations.append(corr)

        # Check hepatotoxicity
        alt_data = latest_labs.get(LabType.ALT.value)
        bili_data = latest_labs.get(LabType.BILIRUBIN.value)
        if alt_data is not None and bili_data is not None:
            alt_ratio = alt_data[0] / max(alt_data[2], 1.0)
            bili_ratio = bili_data[0] / max(bili_data[2], 0.1)
            if alt_ratio > 3.0 and bili_ratio > 2.0:
                corr = CrossModalCorrelation(
                    modalities=[DataModality.LABORATORY.value],
                    description=f"Hepatotoxicity signal: ALT {alt_ratio:.1f}x ULN, bilirubin {bili_ratio:.1f}x ULN",
                    strength=0.85,
                    evidence=[
                        {"type": "alt_ratio", "value": alt_ratio, "threshold": 3.0},
                        {"type": "bilirubin_ratio", "value": bili_ratio, "threshold": 2.0},
                    ],
                    clinical_significance="HIGH - Hy's Law criteria may be met; assess for DILI",
                )
                new_correlations.append(corr)

        # Check nephrotoxicity
        creat_data = latest_labs.get(LabType.CREATININE.value)
        if creat_data is not None:
            creat_ratio = creat_data[0] / max(creat_data[2], 0.1)
            if creat_ratio > 1.5:
                corr = CrossModalCorrelation(
                    modalities=[DataModality.LABORATORY.value],
                    description=f"Nephrotoxicity signal: creatinine {creat_ratio:.1f}x ULN",
                    strength=0.80,
                    evidence=[{"type": "creatinine_ratio", "value": creat_ratio, "threshold": 1.5}],
                    clinical_significance="HIGH - Assess renal function; hold nephrotoxic agents",
                )
                new_correlations.append(corr)

        self._detected_correlations.extend(new_correlations)
        return new_correlations

    @property
    def correlation_count(self) -> int:
        """Total correlations detected."""
        return len(self._detected_correlations)


# ---------------------------------------------------------------------------
# Real-time adaptive monitoring agent
# ---------------------------------------------------------------------------
class RealtimeAdaptiveMonitoringAgent:
    """Real-time adaptive monitoring agent for oncology clinical trials.

    Processes streaming multi-modal clinical data, applies adaptive
    thresholds, detects cross-modal correlations, and generates
    prioritized clinical alerts with escalation pathways.
    """

    def __init__(
        self,
        patient_id: str = "",
        treatment_phase: TreatmentPhase = TreatmentPhase.INDUCTION,
        seed: int = 42,
    ) -> None:
        self._patient_id = patient_id
        self._treatment_phase = treatment_phase
        self._thresholds = AdaptiveThresholdManager()
        self._correlator = CrossModalCorrelationDetector()
        self._alerts: list[ClinicalAlert] = []
        self._readings_processed = 0
        self._rng = np.random.default_rng(seed)

        # Set thresholds for treatment phase
        self._thresholds.reset_for_phase(treatment_phase)
        logger.info(
            "RealtimeAdaptiveMonitoringAgent initialized: patient=%s, phase=%s",
            patient_id,
            treatment_phase.value,
        )

    def process_vital_sign(self, reading: VitalSignReading) -> Optional[ClinicalAlert]:
        """Process a vital sign reading and generate alerts if needed.

        Args:
            reading: Vital sign measurement.

        Returns:
            ClinicalAlert if thresholds are exceeded, else None.
        """
        self._readings_processed += 1
        self._correlator.ingest_vital(reading)

        # Update adaptive threshold
        self._thresholds.update(reading.vital_type.value, reading.value)

        # Check against threshold
        within, deviation = self._thresholds.check_value(reading.vital_type.value, reading.value)

        if not within:
            severity = self._compute_alert_severity(deviation)
            alert = ClinicalAlert(
                patient_id=reading.patient_id,
                severity=severity,
                title=f"Vital sign out of range: {reading.vital_type.value}",
                description=(
                    f"{reading.vital_type.value} = {reading.value:.1f} {reading.unit} "
                    f"(threshold: [{self._thresholds.get_threshold(reading.vital_type.value).current_low:.1f}, "
                    f"{self._thresholds.get_threshold(reading.vital_type.value).current_high:.1f}], "
                    f"deviation score: {deviation:.2f})"
                ),
                source_modality=DataModality.VITAL_SIGNS,
                source_readings=[reading.reading_id],
                recommended_action=self._get_vital_action(reading.vital_type, deviation),
                escalation_path=self._get_escalation_path(severity),
            )
            self._alerts.append(alert)
            logger.warning(
                "ALERT [%s] %s: %s = %.1f (deviation=%.2f)",
                severity.value.upper(),
                reading.patient_id,
                reading.vital_type.value,
                reading.value,
                deviation,
            )
            return alert

        return None

    def process_lab_result(self, result: LabResult) -> Optional[ClinicalAlert]:
        """Process a laboratory result and generate alerts if needed.

        Args:
            result: Laboratory test result.

        Returns:
            ClinicalAlert if abnormal, else None.
        """
        self._readings_processed += 1
        self._correlator.ingest_lab(result)

        # Update adaptive threshold
        self._thresholds.update(result.lab_type.value, result.value)

        alert = None
        if result.is_critical:
            alert = ClinicalAlert(
                patient_id=result.patient_id,
                severity=AlertSeverity.CRITICAL,
                title=f"Critical lab value: {result.lab_type.value}",
                description=(
                    f"{result.lab_type.value} = {result.value:.2f} {result.unit} "
                    f"(reference: [{result.reference_low:.2f}, {result.reference_high:.2f}])"
                ),
                source_modality=DataModality.LABORATORY,
                source_readings=[result.result_id],
                recommended_action="Immediately notify treating physician; repeat test to confirm",
                escalation_path=["nurse_coordinator", "attending_oncologist", "principal_investigator"],
            )
        elif result.is_abnormal():
            severity = (
                AlertSeverity.MODERATE
                if abs(result.value - result.reference_high) > result.reference_high * 0.3
                else AlertSeverity.LOW
            )
            toxicity = self._grade_lab_toxicity(result)
            alert = ClinicalAlert(
                patient_id=result.patient_id,
                severity=severity,
                title=f"Abnormal lab: {result.lab_type.value} (CTCAE Grade {toxicity.value})",
                description=(
                    f"{result.lab_type.value} = {result.value:.2f} {result.unit} "
                    f"(reference: [{result.reference_low:.2f}, {result.reference_high:.2f}], "
                    f"toxicity: {toxicity.name})"
                ),
                source_modality=DataModality.LABORATORY,
                source_readings=[result.result_id],
                recommended_action=self._get_lab_action(result.lab_type, toxicity),
                escalation_path=self._get_escalation_path(severity),
            )

        if alert is not None:
            self._alerts.append(alert)
            logger.warning(
                "ALERT [%s] %s: %s = %.2f",
                alert.severity.value.upper(),
                result.patient_id,
                result.lab_type.value,
                result.value,
            )

        return alert

    def process_imaging_report(self, report: ImagingReport) -> Optional[ClinicalAlert]:
        """Process an imaging report and generate alerts for progression.

        Args:
            report: Imaging assessment report.

        Returns:
            ClinicalAlert if progression detected, else None.
        """
        self._readings_processed += 1
        self._correlator.ingest_imaging(report)

        response = report.compute_response()
        report.response_category = response

        if response == "progressive_disease":
            alert = ClinicalAlert(
                patient_id=report.patient_id,
                severity=AlertSeverity.HIGH,
                title="Disease progression detected on imaging",
                description=(
                    f"{report.modality} assessment: {response}. "
                    f"Target lesion sum: {report.target_lesion_sum_mm:.1f}mm "
                    f"(baseline: {report.baseline_sum_mm:.1f}mm). "
                    f"New lesions: {report.new_lesions}"
                ),
                source_modality=DataModality.IMAGING,
                source_readings=[report.report_id],
                recommended_action="Schedule tumor board discussion; consider treatment modification",
                escalation_path=["attending_oncologist", "principal_investigator", "sponsor_medical_monitor"],
            )
            self._alerts.append(alert)
            logger.warning("ALERT [HIGH] %s: Disease progression on %s", report.patient_id, report.modality)
            return alert

        return None

    def check_cross_modal_correlations(self) -> list[ClinicalAlert]:
        """Check for cross-modal correlations and generate alerts.

        Returns:
            List of alerts from detected correlations.
        """
        correlations = self._correlator.detect_correlations(self._patient_id)
        alerts = []

        for corr in correlations:
            severity = AlertSeverity.HIGH
            if "CRITICAL" in corr.clinical_significance:
                severity = AlertSeverity.CRITICAL

            alert = ClinicalAlert(
                patient_id=self._patient_id,
                severity=severity,
                title=f"Cross-modal correlation: {corr.description[:60]}",
                description=corr.description,
                source_modality=DataModality.VITAL_SIGNS,
                correlation_ids=[corr.correlation_id],
                recommended_action=corr.clinical_significance,
                escalation_path=self._get_escalation_path(severity),
            )
            alerts.append(alert)
            self._alerts.append(alert)

        return alerts

    def get_monitoring_summary(self) -> dict[str, Any]:
        """Get a summary of the monitoring agent state."""
        active_alerts = [a for a in self._alerts if a.status == AlertStatus.ACTIVE]
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]

        return {
            "patient_id": self._patient_id,
            "treatment_phase": self._treatment_phase.value,
            "readings_processed": self._readings_processed,
            "total_alerts": len(self._alerts),
            "active_alerts": len(active_alerts),
            "critical_alerts": len(critical_alerts),
            "correlations_detected": self._correlator.correlation_count,
            "alert_summary": {
                severity.value: len([a for a in active_alerts if a.severity == severity]) for severity in AlertSeverity
            },
        }

    def _compute_alert_severity(self, deviation: float) -> AlertSeverity:
        """Compute alert severity from deviation score."""
        if deviation > 3.0:
            return AlertSeverity.CRITICAL
        elif deviation > 2.0:
            return AlertSeverity.HIGH
        elif deviation > 1.5:
            return AlertSeverity.MODERATE
        else:
            return AlertSeverity.LOW

    def _grade_lab_toxicity(self, result: LabResult) -> ToxicityGrade:
        """Grade laboratory toxicity per CTCAE v5.0 simplified criteria."""
        if not result.is_abnormal():
            return ToxicityGrade.GRADE_0

        # Compute how far outside normal range
        if result.value < result.reference_low:
            deviation = (result.reference_low - result.value) / max(result.reference_low, 0.01)
        else:
            deviation = (result.value - result.reference_high) / max(result.reference_high, 0.01)

        if deviation > 1.0:
            return ToxicityGrade.GRADE_4
        elif deviation > 0.5:
            return ToxicityGrade.GRADE_3
        elif deviation > 0.25:
            return ToxicityGrade.GRADE_2
        else:
            return ToxicityGrade.GRADE_1

    def _get_vital_action(self, vital_type: VitalType, deviation: float) -> str:
        """Get recommended action for a vital sign alert."""
        actions = {
            VitalType.HEART_RATE: "Assess for arrhythmia; check medications; obtain ECG if HR > 120",
            VitalType.SYSTOLIC_BP: "Assess for hypotension/hypertension; check medications; fluid status",
            VitalType.TEMPERATURE: "Assess for infection; obtain blood cultures if T > 38.3C; check ANC",
            VitalType.SPO2: "Apply supplemental oxygen; assess respiratory status; consider imaging",
            VitalType.RESPIRATORY_RATE: "Assess respiratory effort; auscultate lungs; check SpO2",
        }
        base_action = actions.get(vital_type, "Assess clinical status; notify treating physician")
        if deviation > 2.5:
            base_action += " [URGENT]"
        return base_action

    def _get_lab_action(self, lab_type: LabType, toxicity: ToxicityGrade) -> str:
        """Get recommended action for a lab alert based on toxicity grade."""
        if toxicity.value >= 3:
            return f"Grade {toxicity.value} toxicity for {lab_type.value}; consider dose modification per protocol"
        elif toxicity.value >= 2:
            return f"Grade {toxicity.value} toxicity for {lab_type.value}; monitor closely; repeat in 48h"
        else:
            return f"Grade {toxicity.value} toxicity for {lab_type.value}; routine monitoring"

    @staticmethod
    def _get_escalation_path(severity: AlertSeverity) -> list[str]:
        """Get the escalation path for an alert severity."""
        paths = {
            AlertSeverity.INFO: ["data_manager"],
            AlertSeverity.LOW: ["nurse_coordinator"],
            AlertSeverity.MODERATE: ["nurse_coordinator", "attending_oncologist"],
            AlertSeverity.HIGH: ["attending_oncologist", "principal_investigator"],
            AlertSeverity.CRITICAL: ["attending_oncologist", "principal_investigator", "sponsor_medical_monitor"],
        }
        return paths.get(severity, ["nurse_coordinator"])


# ---------------------------------------------------------------------------
# Synthetic data generator for demonstration
# ---------------------------------------------------------------------------
class ClinicalDataStreamGenerator:
    """Generates synthetic clinical data streams for demonstration.

    Produces realistic vital signs, lab results, and imaging reports
    with optional abnormal values to trigger alerts and correlations.
    """

    def __init__(self, patient_id: str, seed: int = 42) -> None:
        self._patient_id = patient_id
        self._rng = np.random.default_rng(seed)
        self._time_cursor = time.time()

    def generate_vital_signs(self, count: int = 10, abnormal_rate: float = 0.2) -> list[VitalSignReading]:
        """Generate a sequence of vital sign readings."""
        readings = []
        for i in range(count):
            self._time_cursor += self._rng.uniform(60, 300)
            is_abnormal = self._rng.random() < abnormal_rate

            vital_type = self._rng.choice(list(VitalType))
            value, unit = self._generate_vital_value(vital_type, abnormal=is_abnormal)

            reading = VitalSignReading(
                patient_id=self._patient_id,
                vital_type=vital_type,
                value=value,
                unit=unit,
                timestamp=self._time_cursor,
            )
            readings.append(reading)
        return readings

    def generate_lab_results(self, count: int = 5, abnormal_rate: float = 0.3) -> list[LabResult]:
        """Generate a batch of laboratory results."""
        results = []
        for i in range(count):
            self._time_cursor += self._rng.uniform(3600, 14400)
            is_abnormal = self._rng.random() < abnormal_rate

            lab_type = self._rng.choice(list(LabType))
            value, unit, ref_low, ref_high, is_critical = self._generate_lab_value(lab_type, abnormal=is_abnormal)

            result = LabResult(
                patient_id=self._patient_id,
                lab_type=lab_type,
                value=value,
                unit=unit,
                reference_low=ref_low,
                reference_high=ref_high,
                timestamp=self._time_cursor,
                is_critical=is_critical,
            )
            results.append(result)
        return results

    def generate_imaging_report(self, baseline_mm: float = 45.0, progression: bool = False) -> ImagingReport:
        """Generate an imaging assessment report."""
        self._time_cursor += self._rng.uniform(86400, 604800)

        if progression:
            target_sum = baseline_mm * float(self._rng.uniform(1.2, 1.5))
            new_lesions = self._rng.random() < 0.3
        else:
            target_sum = baseline_mm * float(self._rng.uniform(0.5, 0.95))
            new_lesions = False

        return ImagingReport(
            patient_id=self._patient_id,
            modality="CT",
            target_lesion_sum_mm=target_sum,
            baseline_sum_mm=baseline_mm,
            nadir_sum_mm=baseline_mm * 0.7,
            new_lesions=new_lesions,
            timestamp=self._time_cursor,
            findings_summary="Routine restaging CT chest/abdomen/pelvis with IV contrast",
        )

    def _generate_vital_value(self, vital_type: VitalType, abnormal: bool = False) -> tuple[float, str]:
        """Generate a vital sign value with optional abnormality."""
        normals: dict[VitalType, tuple[float, float, str]] = {
            VitalType.HEART_RATE: (72.0, 8.0, "bpm"),
            VitalType.SYSTOLIC_BP: (120.0, 10.0, "mmHg"),
            VitalType.DIASTOLIC_BP: (75.0, 8.0, "mmHg"),
            VitalType.TEMPERATURE: (36.8, 0.3, "C"),
            VitalType.RESPIRATORY_RATE: (16.0, 2.0, "breaths/min"),
            VitalType.SPO2: (97.0, 1.0, "%"),
            VitalType.PAIN_SCORE: (2.0, 1.5, "NRS"),
        }
        mean, std, unit = normals.get(vital_type, (0.0, 1.0, ""))
        if abnormal:
            offset = self._rng.choice([-1, 1]) * std * float(self._rng.uniform(2.5, 4.0))
            value = mean + offset
        else:
            value = float(self._rng.normal(mean, std))
        return float(np.clip(value, 0.0, 300.0)), unit

    def _generate_lab_value(self, lab_type: LabType, abnormal: bool = False) -> tuple[float, str, float, float, bool]:
        """Generate a lab value with reference range."""
        lab_params: dict[LabType, tuple[float, float, float, float, str]] = {
            LabType.WBC: (7.0, 1.5, 4.0, 11.0, "x10^9/L"),
            LabType.ANC: (4.0, 1.0, 1.5, 8.0, "x10^9/L"),
            LabType.HEMOGLOBIN: (13.5, 1.0, 12.0, 17.0, "g/dL"),
            LabType.PLATELETS: (250.0, 50.0, 150.0, 400.0, "x10^9/L"),
            LabType.CREATININE: (0.9, 0.15, 0.6, 1.2, "mg/dL"),
            LabType.ALT: (25.0, 10.0, 7.0, 56.0, "U/L"),
            LabType.AST: (22.0, 8.0, 10.0, 40.0, "U/L"),
            LabType.BILIRUBIN: (0.7, 0.3, 0.1, 1.2, "mg/dL"),
            LabType.ALBUMIN: (4.0, 0.3, 3.5, 5.0, "g/dL"),
            LabType.LDH: (180.0, 30.0, 120.0, 246.0, "U/L"),
            LabType.TSH: (2.5, 1.0, 0.4, 4.0, "mIU/L"),
        }
        mean, std, ref_low, ref_high, unit = lab_params.get(lab_type, (1.0, 0.5, 0.5, 1.5, ""))
        is_critical = False

        if abnormal:
            offset = self._rng.choice([-1, 1]) * std * float(self._rng.uniform(2.0, 4.0))
            value = mean + offset
            if abs(offset) > std * 3.5:
                is_critical = True
        else:
            value = float(self._rng.normal(mean, std))

        return float(np.clip(value, 0.001, 10000.0)), unit, ref_low, ref_high, is_critical


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------
def main() -> None:
    """Demonstrate the real-time adaptive monitoring agent."""
    logger.info("=" * 80)
    logger.info("Real-Time Adaptive Monitoring Agent Demonstration")
    logger.info("Version: 0.5.0 | RESEARCH USE ONLY")
    logger.info("=" * 80)

    logger.info(
        "Optional dependencies: HAS_ANTHROPIC=%s, HAS_OPENAI=%s, HAS_SKLEARN=%s",
        HAS_ANTHROPIC,
        HAS_OPENAI,
        HAS_SKLEARN,
    )

    patient_id = "PAT-MON-001"

    # Initialize the monitoring agent
    agent = RealtimeAdaptiveMonitoringAgent(
        patient_id=patient_id,
        treatment_phase=TreatmentPhase.INDUCTION,
        seed=42,
    )

    # Generate synthetic data streams
    generator = ClinicalDataStreamGenerator(patient_id=patient_id, seed=42)

    # Process vital signs
    logger.info("-" * 60)
    logger.info("Processing vital sign stream...")
    vitals = generator.generate_vital_signs(count=20, abnormal_rate=0.25)
    vital_alerts = 0
    for reading in vitals:
        alert = agent.process_vital_sign(reading)
        if alert is not None:
            vital_alerts += 1
    logger.info("Vital signs processed: %d readings, %d alerts", len(vitals), vital_alerts)

    # Process lab results
    logger.info("-" * 60)
    logger.info("Processing laboratory results...")
    labs = generator.generate_lab_results(count=10, abnormal_rate=0.35)
    lab_alerts = 0
    for result in labs:
        alert = agent.process_lab_result(result)
        if alert is not None:
            lab_alerts += 1
    logger.info("Lab results processed: %d results, %d alerts", len(labs), lab_alerts)

    # Process imaging report
    logger.info("-" * 60)
    logger.info("Processing imaging report...")
    imaging = generator.generate_imaging_report(baseline_mm=45.0, progression=False)
    img_alert = agent.process_imaging_report(imaging)
    logger.info(
        "Imaging: %s (%s)",
        imaging.compute_response(),
        "alert generated" if img_alert else "no alert",
    )

    # Check cross-modal correlations
    logger.info("-" * 60)
    logger.info("Checking cross-modal correlations...")
    correlation_alerts = agent.check_cross_modal_correlations()
    logger.info("Cross-modal correlations: %d alerts generated", len(correlation_alerts))

    # Display monitoring summary
    logger.info("-" * 60)
    summary = agent.get_monitoring_summary()
    logger.info("MONITORING SUMMARY:")
    for key, value in summary.items():
        logger.info("  %s: %s", key, value)

    logger.info("=" * 80)
    logger.info("Demonstration complete.")


if __name__ == "__main__":
    main()
