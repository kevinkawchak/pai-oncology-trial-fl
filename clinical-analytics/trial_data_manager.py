"""Trial data manager — clinical trial data lifecycle for federated analytics.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.

Manages the full lifecycle of clinical trial datasets within a federated
oncology analytics platform.  Responsibilities include dataset registration,
schema validation, data-quality assessment, cross-site reconciliation,
retention-policy enforcement, and regulatory manifest generation.

Each operation is recorded in an append-only audit trail that satisfies
21 CFR Part 11 requirements for electronic records.  Dataset and patient
identifiers are protected with HMAC-SHA256 to prevent leakage of
identifying information across site boundaries.

Quality metrics (completeness, accuracy, consistency, timeliness) are
computed per-dataset and aggregated per-site, enabling the platform to
flag datasets that fall below acceptable thresholds before they enter
a federated training round.

IEC 62304 traceability is maintained through structured audit entries
that link every data-lifecycle transition (registration, quality check,
archival, destruction) to its originating requirement and operator.

DISCLAIMER: RESEARCH USE ONLY — This software is provided for research
and educational purposes.  It has not been validated for clinical
deployment or patient care decisions.  Independent validation against
applicable regulations is required before any clinical use.

LICENSE: MIT
VERSION: 0.9.0
LAST UPDATED: 2026-02-18
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security constants
# ---------------------------------------------------------------------------

_HMAC_KEY = b"pai-oncology-trial-data-integrity-key"
_HASH_TRUNCATION = 32

# ---------------------------------------------------------------------------
# IEC 62304 requirement identifiers referenced in audit entries
# ---------------------------------------------------------------------------

_REQ_DATA_REGISTRATION = "IEC62304-SWR-001"
_REQ_DATA_QUALITY = "IEC62304-SWR-002"
_REQ_SCHEMA_VALIDATION = "IEC62304-SWR-003"
_REQ_RETENTION_POLICY = "IEC62304-SWR-004"
_REQ_CROSS_SITE_RECON = "IEC62304-SWR-005"
_REQ_MANIFEST_GENERATION = "IEC62304-SWR-006"


# ===================================================================
# Enumerations
# ===================================================================


class DataDomain(str, Enum):
    """Clinical data domain classifications for oncology trials."""

    DEMOGRAPHICS = "demographics"
    VITALS = "vitals"
    LABORATORY = "laboratory"
    IMAGING = "imaging"
    GENOMICS = "genomics"
    PATHOLOGY = "pathology"
    TREATMENT = "treatment"
    OUTCOMES = "outcomes"
    ADVERSE_EVENTS = "adverse_events"
    BIOSPECIMENS = "biospecimens"


class DataQualityLevel(str, Enum):
    """Ordinal quality tiers for clinical datasets.

    Thresholds (based on composite quality score):
        EXCELLENT     >= 0.90
        GOOD          >= 0.75
        ACCEPTABLE    >= 0.60
        POOR          >= 0.40
        UNACCEPTABLE  <  0.40
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


class RetentionPolicy(str, Enum):
    """Data retention lifecycle states."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    PENDING_DESTRUCTION = "pending_destruction"
    DESTROYED = "destroyed"


class ConsentScope(str, Enum):
    """Scope of patient consent for data usage."""

    FULL_RESEARCH = "full_research"
    LIMITED_DISEASE = "limited_disease"
    ANONYMIZED_ONLY = "anonymized_only"
    WITHDRAWN = "withdrawn"


# ===================================================================
# Dataclasses
# ===================================================================


@dataclass
class ClinicalDataset:
    """Metadata record for a registered clinical dataset."""

    dataset_id: str
    domain: DataDomain
    site_id: str
    record_count: int = 0
    quality_level: DataQualityLevel = DataQualityLevel.ACCEPTABLE
    schema_version: str = "1.0.0"
    consent_scope: ConsentScope = ConsentScope.FULL_RESEARCH
    created_at: str = ""
    updated_at: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


@dataclass
class DataQualityReport:
    """Quantitative quality assessment — all scores bounded to [0.0, 1.0]."""

    dataset_id: str = ""
    completeness: float = 0.0
    accuracy: float = 0.0
    consistency: float = 0.0
    timeliness: float = 0.0
    composite_score: float = 0.0
    quality_level: DataQualityLevel = DataQualityLevel.UNACCEPTABLE
    assessed_at: str = ""
    field_issues: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Clamp all scores to [0, 1]
        self.completeness = _clamp01(self.completeness)
        self.accuracy = _clamp01(self.accuracy)
        self.consistency = _clamp01(self.consistency)
        self.timeliness = _clamp01(self.timeliness)
        self.composite_score = _clamp01(self.composite_score)
        if not self.assessed_at:
            self.assessed_at = datetime.now(timezone.utc).isoformat()


@dataclass
class SchemaDefinition:
    """Expected schema for a clinical dataset (fields, types, constraints)."""

    fields: list[str] = field(default_factory=list)
    types: dict[str, str] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    constraints: dict[str, dict[str, Any]] = field(default_factory=dict)
    version: str = "1.0.0"


@dataclass
class DataRetentionRecord:
    """Tracks the retention lifecycle of a dataset with HMAC certificates."""

    dataset_id: str = ""
    retention_policy: RetentionPolicy = RetentionPolicy.ACTIVE
    creation_date: str = ""
    expiry_date: str = ""
    destruction_certificate_hash: str = ""
    policy_applied_at: str = ""
    policy_applied_by: str = ""

    def __post_init__(self) -> None:
        if not self.creation_date:
            self.creation_date = datetime.now(timezone.utc).isoformat()
        if not self.policy_applied_at:
            self.policy_applied_at = datetime.now(timezone.utc).isoformat()


@dataclass
class SiteDataInventory:
    """Aggregate data inventory for a single federated site."""

    site_id: str = ""
    datasets_by_domain: dict[str, list[str]] = field(default_factory=dict)
    total_records: int = 0
    quality_summary: dict[str, int] = field(default_factory=dict)
    domain_coverage: float = 0.0
    last_updated: str = ""

    def __post_init__(self) -> None:
        if not self.last_updated:
            self.last_updated = datetime.now(timezone.utc).isoformat()


# ===================================================================
# Helper utilities
# ===================================================================


def _clamp01(value: float) -> float:
    """Clamp a numeric value to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


def _safe_divide(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 when the denominator is near zero."""
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def _hmac_identifier(raw_id: str) -> str:
    """Produce an HMAC-SHA256 pseudonym for a raw identifier."""
    return hmac.new(
        _HMAC_KEY, raw_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()[:_HASH_TRUNCATION]


def _quality_level_from_score(score: float) -> DataQualityLevel:
    """Map a composite quality score to a quality tier."""
    if score >= 0.90:
        return DataQualityLevel.EXCELLENT
    if score >= 0.75:
        return DataQualityLevel.GOOD
    if score >= 0.60:
        return DataQualityLevel.ACCEPTABLE
    if score >= 0.40:
        return DataQualityLevel.POOR
    return DataQualityLevel.UNACCEPTABLE


# ===================================================================
# Retention policy transition matrix (IEC 62304 traceability)
# ===================================================================

_VALID_RETENTION_TRANSITIONS: dict[RetentionPolicy, set[RetentionPolicy]] = {
    RetentionPolicy.ACTIVE: {RetentionPolicy.ARCHIVED, RetentionPolicy.PENDING_DESTRUCTION},
    RetentionPolicy.ARCHIVED: {RetentionPolicy.ACTIVE, RetentionPolicy.PENDING_DESTRUCTION},
    RetentionPolicy.PENDING_DESTRUCTION: {RetentionPolicy.DESTROYED},
    RetentionPolicy.DESTROYED: set(),
}


# ===================================================================
# Main class
# ===================================================================


class TrialDataManager:
    """Manages the clinical trial data lifecycle for federated analytics.

    Provides dataset registration, quality assessment, schema validation,
    cross-site reconciliation, retention-policy enforcement, and regulatory
    manifest generation.  Every mutation is recorded in a 21 CFR Part 11
    compliant audit trail with HMAC integrity hashes.

    Args:
        trial_id: Unique identifier for the clinical trial.
        quality_weights: Optional dict of dimension weights for composite
            quality scoring.  Keys are ``completeness``, ``accuracy``,
            ``consistency``, ``timeliness``; values are positive floats
            that are normalised internally.  Defaults to equal weighting.
        operator_id: Default operator identifier for audit entries.
    """

    # Quality dimension weights (default: equal)
    _DEFAULT_WEIGHTS: dict[str, float] = {
        "completeness": 0.30,
        "accuracy": 0.30,
        "consistency": 0.25,
        "timeliness": 0.15,
    }

    def __init__(
        self,
        trial_id: str,
        quality_weights: dict[str, float] | None = None,
        operator_id: str = "system",
    ) -> None:
        self.trial_id = trial_id
        self._operator_id = operator_id

        # Normalise quality weights
        raw_weights = quality_weights or dict(self._DEFAULT_WEIGHTS)
        total_w = sum(raw_weights.values()) or 1.0
        self._quality_weights = {k: v / total_w for k, v in raw_weights.items()}

        # Internal registries
        self._datasets: dict[str, ClinicalDataset] = {}
        self._quality_reports: dict[str, DataQualityReport] = {}
        self._retention_records: dict[str, DataRetentionRecord] = {}
        self._schemas: dict[str, SchemaDefinition] = {}

        # Simulated data store for quality assessment (dataset_id -> records)
        self._data_store: dict[str, list[dict[str, Any]]] = {}

        # 21 CFR Part 11 audit trail
        self._audit_trail: list[dict[str, Any]] = []

        logger.info(
            "TrialDataManager initialised for trial %s (operator=%s)",
            trial_id,
            operator_id,
        )

    # ------------------------------------------------------------------
    # Dataset registration
    # ------------------------------------------------------------------

    def register_dataset(
        self,
        dataset: ClinicalDataset,
        records: list[dict[str, Any]] | None = None,
    ) -> ClinicalDataset:
        """Validate and register a clinical dataset with audit logging.

        Raises ValueError if the dataset fails validation.  An initial
        retention record is created in ACTIVE state.
        """
        # Validation
        if not dataset.dataset_id:
            raise ValueError("dataset_id must not be empty")
        if not dataset.site_id:
            raise ValueError("site_id must not be empty")
        if dataset.record_count < 0:
            raise ValueError("record_count must be non-negative")
        if dataset.dataset_id in self._datasets:
            raise ValueError(
                f"Dataset {dataset.dataset_id} is already registered"
            )

        # Populate timestamps
        now = datetime.now(timezone.utc).isoformat()
        dataset.created_at = now
        dataset.updated_at = now

        # Store
        self._datasets[dataset.dataset_id] = dataset

        if records is not None:
            self._data_store[dataset.dataset_id] = list(records)
            dataset.record_count = len(records)

        # Create initial retention record
        retention = DataRetentionRecord(
            dataset_id=dataset.dataset_id,
            retention_policy=RetentionPolicy.ACTIVE,
            creation_date=now,
            policy_applied_by=self._operator_id,
        )
        self._retention_records[dataset.dataset_id] = retention

        # Audit
        self._record_audit(
            action="dataset_registered",
            resource=dataset.dataset_id,
            details={
                "domain": dataset.domain.value,
                "site_id": _hmac_identifier(dataset.site_id),
                "record_count": dataset.record_count,
                "schema_version": dataset.schema_version,
                "consent_scope": dataset.consent_scope.value,
            },
            requirement=_REQ_DATA_REGISTRATION,
        )

        logger.info(
            "Dataset %s registered (domain=%s, site=%s, records=%d)",
            dataset.dataset_id,
            dataset.domain.value,
            dataset.site_id,
            dataset.record_count,
        )
        return dataset

    # ------------------------------------------------------------------
    # Data-quality assessment
    # ------------------------------------------------------------------

    def assess_data_quality(
        self,
        dataset_id: str,
        schema: SchemaDefinition | None = None,
    ) -> DataQualityReport:
        """Compute completeness, accuracy, consistency, and timeliness.

        The composite score is the weighted average of the four dimensions.
        Raises KeyError if the dataset_id is not registered.
        """
        if dataset_id not in self._datasets:
            raise KeyError(f"Dataset {dataset_id} is not registered")

        dataset = self._datasets[dataset_id]
        records = self._data_store.get(dataset_id, [])
        total_records = max(len(records), dataset.record_count, 1)

        # --- Completeness ---
        completeness = self._compute_completeness(records, schema)

        # --- Accuracy ---
        accuracy = self._compute_accuracy(records, schema)

        # --- Consistency ---
        consistency = self._compute_consistency(records)

        # --- Timeliness ---
        timeliness = self._compute_timeliness(records)

        # --- Composite ---
        composite = (
            self._quality_weights.get("completeness", 0.25) * completeness
            + self._quality_weights.get("accuracy", 0.25) * accuracy
            + self._quality_weights.get("consistency", 0.25) * consistency
            + self._quality_weights.get("timeliness", 0.25) * timeliness
        )
        composite = _clamp01(composite)

        quality_level = _quality_level_from_score(composite)

        report = DataQualityReport(
            dataset_id=dataset_id,
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            composite_score=composite,
            quality_level=quality_level,
        )

        # Update the dataset's quality level
        dataset.quality_level = quality_level
        dataset.updated_at = datetime.now(timezone.utc).isoformat()

        # Cache the report
        self._quality_reports[dataset_id] = report

        # Audit
        self._record_audit(
            action="quality_assessed",
            resource=dataset_id,
            details={
                "completeness": round(completeness, 4),
                "accuracy": round(accuracy, 4),
                "consistency": round(consistency, 4),
                "timeliness": round(timeliness, 4),
                "composite": round(composite, 4),
                "quality_level": quality_level.value,
            },
            requirement=_REQ_DATA_QUALITY,
        )

        logger.info(
            "Quality assessment for %s: composite=%.3f (%s)",
            dataset_id,
            composite,
            quality_level.value,
        )
        return report

    def _compute_completeness(
        self,
        records: list[dict[str, Any]],
        schema: SchemaDefinition | None,
    ) -> float:
        """Compute the completeness score for a set of records.

        Completeness measures the fraction of required fields that are
        present and non-null across all records.
        """
        if not records:
            return 0.0

        required_fields: list[str] = []
        if schema and schema.required:
            required_fields = list(schema.required)
        else:
            # Infer required fields from the union of all keys
            all_keys: set[str] = set()
            for rec in records:
                all_keys.update(rec.keys())
            required_fields = sorted(all_keys)

        if not required_fields:
            return 1.0

        present_count = 0
        total_checks = len(records) * len(required_fields)

        for rec in records:
            for fld in required_fields:
                value = rec.get(fld)
                if value is not None and value != "" and value != []:
                    present_count += 1

        return _safe_divide(float(present_count), float(total_checks))

    def _compute_accuracy(
        self,
        records: list[dict[str, Any]],
        schema: SchemaDefinition | None,
    ) -> float:
        """Compute the accuracy score for a set of records.

        Accuracy measures the fraction of field values that fall within
        their declared constraints (type, range, pattern).
        """
        if not records:
            return 0.0

        if schema is None or not schema.constraints:
            # Without constraints, perform basic type/range heuristics
            return self._heuristic_accuracy(records)

        valid_count = 0
        total_checks = 0

        for rec in records:
            for fld, constraint in schema.constraints.items():
                value = rec.get(fld)
                if value is None:
                    continue  # Missing values handled by completeness

                total_checks += 1
                if self._check_constraint(value, constraint):
                    valid_count += 1

        return _safe_divide(float(valid_count), float(total_checks))

    @staticmethod
    def _check_constraint(value: Any, constraint: dict[str, Any]) -> bool:
        """Check a single value against a constraint dictionary.

        Supported constraint keys: ``min``, ``max``, ``type``,
        ``allowed_values``.
        """
        try:
            if "min" in constraint and float(value) < constraint["min"]:
                return False
            if "max" in constraint and float(value) > constraint["max"]:
                return False
            if "type" in constraint:
                expected = constraint["type"]
                if expected == "int" and not isinstance(value, (int, np.integer)):
                    return False
                if expected == "float" and not isinstance(value, (int, float, np.floating)):
                    return False
                if expected == "str" and not isinstance(value, str):
                    return False
            if "allowed_values" in constraint:
                if value not in constraint["allowed_values"]:
                    return False
        except (TypeError, ValueError):
            return False
        return True

    def _heuristic_accuracy(self, records: list[dict[str, Any]]) -> float:
        """Estimate accuracy without an explicit schema.

        Collects numeric values across all records and checks for NaN/Inf
        and extreme outliers (> 5 sigma from mean).
        """
        numeric_values: dict[str, list[float]] = {}
        for rec in records:
            for key, val in rec.items():
                if isinstance(val, (int, float, np.integer, np.floating)):
                    numeric_values.setdefault(key, []).append(float(val))

        if not numeric_values:
            return 1.0

        valid_count = 0
        total_count = 0

        for fld, vals in numeric_values.items():
            arr = np.array(vals, dtype=np.float64)
            total_count += len(arr)

            # Check for NaN/Inf
            finite_mask = np.isfinite(arr)
            valid_count += int(np.sum(finite_mask))

            # Check for extreme outliers among finite values
            finite_vals = arr[finite_mask]
            if len(finite_vals) > 2:
                mean = np.mean(finite_vals)
                std = np.std(finite_vals)
                if std > 1e-12:
                    z_scores = np.abs((finite_vals - mean) / std)
                    outlier_count = int(np.sum(z_scores > 5.0))
                    valid_count -= outlier_count

        return _safe_divide(float(max(valid_count, 0)), float(total_count))

    def _compute_consistency(self, records: list[dict[str, Any]]) -> float:
        """Compute the consistency score for a set of records.

        Checks cross-field logical consistency rules:
        - Dates are chronologically ordered (e.g. start <= end).
        - Numeric fields that should be non-negative are checked.
        - Duplicate record detection via hash-based deduplication.
        """
        if not records:
            return 0.0

        consistent_count = 0

        for rec in records:
            is_consistent = True

            # Check date ordering if both start and end dates present
            start_date = rec.get("start_date") or rec.get("date_start")
            end_date = rec.get("end_date") or rec.get("date_end")
            if start_date and end_date:
                if str(start_date) > str(end_date):
                    is_consistent = False

            # Check non-negative fields
            for key in ("age", "dose", "count", "record_count", "volume"):
                val = rec.get(key)
                if val is not None:
                    try:
                        if float(val) < 0:
                            is_consistent = False
                    except (TypeError, ValueError):
                        is_consistent = False

            if is_consistent:
                consistent_count += 1

        # Duplicate detection penalty
        record_hashes: set[str] = set()
        duplicate_count = 0
        for rec in records:
            rec_str = str(sorted(rec.items()))
            h = hashlib.sha256(rec_str.encode()).hexdigest()[:16]
            if h in record_hashes:
                duplicate_count += 1
            else:
                record_hashes.add(h)

        base_score = _safe_divide(float(consistent_count), float(len(records)))
        dup_penalty = _safe_divide(float(duplicate_count), float(len(records)))
        return _clamp01(base_score - dup_penalty)

    def _compute_timeliness(self, records: list[dict[str, Any]]) -> float:
        """Compute the timeliness score for a set of records.

        Examines timestamp fields to determine how current the data is.
        Records updated within the last 30 days score 1.0; the score
        decays linearly to 0.0 at 365 days.
        """
        if not records:
            return 0.0

        now = datetime.now(timezone.utc)
        timestamp_fields = ("updated_at", "timestamp", "date", "created_at")
        scores: list[float] = []

        for rec in records:
            ts_str = None
            for tf in timestamp_fields:
                if tf in rec and rec[tf]:
                    ts_str = str(rec[tf])
                    break

            if ts_str is None:
                # No timestamp: assume moderate timeliness
                scores.append(0.5)
                continue

            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_days = (now - ts).total_seconds() / 86400.0
                if age_days <= 30:
                    scores.append(1.0)
                elif age_days >= 365:
                    scores.append(0.0)
                else:
                    scores.append(_clamp01(1.0 - (age_days - 30) / 335.0))
            except (ValueError, TypeError):
                scores.append(0.5)

        if not scores:
            return 0.5

        return _clamp01(float(np.mean(scores)))

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    def validate_schema(
        self,
        data: list[dict[str, Any]],
        schema: SchemaDefinition,
    ) -> list[str]:
        """Validate records against a schema; return list of violation strings."""
        violations: list[str] = []

        if not data:
            violations.append("No records provided for validation")
            self._record_audit(
                action="schema_validation",
                resource="unknown",
                details={"violations": 1, "reason": "empty_data"},
                requirement=_REQ_SCHEMA_VALIDATION,
            )
            return violations

        for idx, record in enumerate(data):
            row_label = f"record[{idx}]"

            # Check required fields
            for req_field in schema.required:
                if req_field not in record:
                    violations.append(
                        f"{row_label}: missing required field '{req_field}'"
                    )
                elif record[req_field] is None:
                    violations.append(
                        f"{row_label}: required field '{req_field}' is null"
                    )

            # Check declared field types
            for fld, expected_type in schema.types.items():
                if fld not in record:
                    continue
                value = record[fld]
                if value is None:
                    continue

                type_ok = self._validate_type(value, expected_type)
                if not type_ok:
                    violations.append(
                        f"{row_label}: field '{fld}' expected type "
                        f"'{expected_type}', got {type(value).__name__}"
                    )

            # Check constraints
            for fld, constraint in schema.constraints.items():
                if fld not in record or record[fld] is None:
                    continue

                value = record[fld]
                constraint_violations = self._validate_constraint(
                    value, fld, constraint, row_label
                )
                violations.extend(constraint_violations)

            # Check for unexpected fields
            if schema.fields:
                known = set(schema.fields)
                for fld in record:
                    if fld not in known:
                        violations.append(
                            f"{row_label}: unexpected field '{fld}' "
                            f"not in schema v{schema.version}"
                        )

        # Audit
        self._record_audit(
            action="schema_validation",
            resource=f"batch_{len(data)}_records",
            details={
                "schema_version": schema.version,
                "records_validated": len(data),
                "violation_count": len(violations),
            },
            requirement=_REQ_SCHEMA_VALIDATION,
        )

        logger.info(
            "Schema validation: %d records, %d violations (schema v%s)",
            len(data),
            len(violations),
            schema.version,
        )
        return violations

    @staticmethod
    def _validate_type(value: Any, expected_type: str) -> bool:
        """Check whether a value conforms to the expected type string."""
        type_map: dict[str, tuple[type, ...]] = {
            "int": (int, np.integer),
            "float": (int, float, np.integer, np.floating),
            "str": (str,),
            "bool": (bool,),
            "list": (list,),
            "dict": (dict,),
        }
        allowed = type_map.get(expected_type)
        if allowed is None:
            return True  # Unknown type, skip check
        return isinstance(value, allowed)

    @staticmethod
    def _validate_constraint(
        value: Any,
        field_name: str,
        constraint: dict[str, Any],
        row_label: str,
    ) -> list[str]:
        """Validate a value against its constraint dictionary."""
        issues: list[str] = []
        try:
            if "min" in constraint:
                if float(value) < constraint["min"]:
                    issues.append(
                        f"{row_label}: field '{field_name}' value "
                        f"{value} < min {constraint['min']}"
                    )
            if "max" in constraint:
                if float(value) > constraint["max"]:
                    issues.append(
                        f"{row_label}: field '{field_name}' value "
                        f"{value} > max {constraint['max']}"
                    )
            if "allowed_values" in constraint:
                if value not in constraint["allowed_values"]:
                    issues.append(
                        f"{row_label}: field '{field_name}' value "
                        f"'{value}' not in allowed values"
                    )
            if "min_length" in constraint:
                if hasattr(value, "__len__") and len(value) < constraint["min_length"]:
                    issues.append(
                        f"{row_label}: field '{field_name}' length "
                        f"{len(value)} < min_length {constraint['min_length']}"
                    )
        except (TypeError, ValueError) as exc:
            issues.append(
                f"{row_label}: field '{field_name}' constraint check "
                f"failed: {exc}"
            )
        return issues

    # ------------------------------------------------------------------
    # Site inventory
    # ------------------------------------------------------------------

    def compute_site_inventory(self, site_id: str) -> SiteDataInventory:
        """Aggregate dataset information for a single site.

        Args:
            site_id: The hospital site to compute inventory for.

        Returns:
            A SiteDataInventory summarising all datasets at the site.
        """
        datasets_by_domain: dict[str, list[str]] = {}
        quality_summary: dict[str, int] = {}
        total_records = 0

        for ds in self._datasets.values():
            if ds.site_id != site_id:
                continue

            domain_key = ds.domain.value
            datasets_by_domain.setdefault(domain_key, []).append(ds.dataset_id)
            total_records += ds.record_count

            ql_key = ds.quality_level.value
            quality_summary[ql_key] = quality_summary.get(ql_key, 0) + 1

        total_domains = len(DataDomain)
        covered_domains = len(datasets_by_domain)
        domain_coverage = _safe_divide(float(covered_domains), float(total_domains))

        inventory = SiteDataInventory(
            site_id=site_id,
            datasets_by_domain=datasets_by_domain,
            total_records=total_records,
            quality_summary=quality_summary,
            domain_coverage=domain_coverage,
        )

        logger.info(
            "Site %s inventory: %d datasets, %d records, %.0f%% domain coverage",
            site_id,
            sum(len(v) for v in datasets_by_domain.values()),
            total_records,
            domain_coverage * 100,
        )
        return inventory

    # ------------------------------------------------------------------
    # Cross-site reconciliation
    # ------------------------------------------------------------------

    def reconcile_cross_site(
        self,
        site_inventories: list[SiteDataInventory],
    ) -> dict[str, Any]:
        """Find discrepancies in domain coverage, record counts, and quality."""
        if len(site_inventories) < 2:
            return {
                "status": "insufficient_sites",
                "message": "At least two site inventories are required",
                "discrepancies": [],
            }

        discrepancies: list[dict[str, Any]] = []

        # --- Domain coverage comparison ---
        all_domains_present: dict[str, set[str]] = {}
        for inv in site_inventories:
            for domain in inv.datasets_by_domain:
                all_domains_present.setdefault(domain, set()).add(inv.site_id)

        for domain, sites_with in all_domains_present.items():
            missing_sites = [
                inv.site_id for inv in site_inventories
                if inv.site_id not in sites_with
            ]
            if missing_sites:
                discrepancies.append({
                    "type": "missing_domain",
                    "domain": domain,
                    "missing_sites": missing_sites,
                    "severity": "warning",
                })

        # --- Record count outlier detection ---
        record_counts = np.array(
            [inv.total_records for inv in site_inventories], dtype=np.float64
        )
        if len(record_counts) > 1:
            mean_count = float(np.mean(record_counts))
            std_count = float(np.std(record_counts))
            for inv in site_inventories:
                if std_count > 1e-12:
                    z = abs(inv.total_records - mean_count) / std_count
                    if z > 2.0:
                        discrepancies.append({
                            "type": "record_count_outlier",
                            "site_id": inv.site_id,
                            "record_count": inv.total_records,
                            "mean": round(mean_count, 1),
                            "z_score": round(z, 2),
                            "severity": "warning" if z <= 3.0 else "error",
                        })

        # --- Quality distribution comparison ---
        quality_distributions: dict[str, dict[str, int]] = {}
        for inv in site_inventories:
            quality_distributions[inv.site_id] = dict(inv.quality_summary)

        sites_with_poor = [
            sid for sid, qdist in quality_distributions.items()
            if qdist.get("poor", 0) + qdist.get("unacceptable", 0) > 0
        ]
        if sites_with_poor:
            discrepancies.append({
                "type": "quality_concern",
                "sites": sites_with_poor,
                "severity": "warning",
                "message": "Sites have datasets rated POOR or UNACCEPTABLE",
            })

        # Audit
        self._record_audit(
            action="cross_site_reconciliation",
            resource=f"{len(site_inventories)}_sites",
            details={
                "site_count": len(site_inventories),
                "discrepancy_count": len(discrepancies),
            },
            requirement=_REQ_CROSS_SITE_RECON,
        )

        result = {
            "status": "completed",
            "sites_compared": [inv.site_id for inv in site_inventories],
            "total_discrepancies": len(discrepancies),
            "discrepancies": discrepancies,
            "quality_distributions": quality_distributions,
            "record_count_statistics": {
                "mean": round(float(np.mean(record_counts)), 1),
                "std": round(float(np.std(record_counts)), 1),
                "min": int(np.min(record_counts)),
                "max": int(np.max(record_counts)),
            },
        }

        logger.info(
            "Cross-site reconciliation: %d sites, %d discrepancies",
            len(site_inventories),
            len(discrepancies),
        )
        return result

    # ------------------------------------------------------------------
    # Retention-policy management
    # ------------------------------------------------------------------

    def apply_retention_policy(
        self,
        dataset_id: str,
        policy: RetentionPolicy,
        operator: str | None = None,
    ) -> DataRetentionRecord:
        """Transition a dataset's retention state per IEC 62304 lifecycle.

        Only valid transitions are permitted.  Destruction generates an
        HMAC-SHA256 certificate hash.  Raises KeyError if the dataset is
        not registered; ValueError if the transition is invalid.
        """
        if dataset_id not in self._datasets:
            raise KeyError(f"Dataset {dataset_id} is not registered")

        operator = operator or self._operator_id
        record = self._retention_records.get(dataset_id)

        if record is None:
            # Create a new retention record if one doesn't exist
            record = DataRetentionRecord(
                dataset_id=dataset_id,
                retention_policy=RetentionPolicy.ACTIVE,
                policy_applied_by=operator,
            )
            self._retention_records[dataset_id] = record

        current_policy = record.retention_policy
        valid_targets = _VALID_RETENTION_TRANSITIONS.get(current_policy, set())

        if policy not in valid_targets:
            raise ValueError(
                f"Invalid retention transition: {current_policy.value} -> "
                f"{policy.value}.  Valid targets: "
                f"{[t.value for t in valid_targets]}"
            )

        now = datetime.now(timezone.utc).isoformat()
        record.retention_policy = policy
        record.policy_applied_at = now
        record.policy_applied_by = operator

        # Handle destruction
        if policy == RetentionPolicy.DESTROYED:
            cert_payload = f"{dataset_id}:{now}:{operator}"
            record.destruction_certificate_hash = hmac.new(
                _HMAC_KEY, cert_payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            # Purge data store
            self._data_store.pop(dataset_id, None)

        if policy == RetentionPolicy.PENDING_DESTRUCTION:
            record.expiry_date = now  # Mark for imminent destruction

        # Audit
        self._record_audit(
            action="retention_policy_applied",
            resource=dataset_id,
            details={
                "previous_policy": current_policy.value,
                "new_policy": policy.value,
                "operator": operator,
                "destruction_cert": record.destruction_certificate_hash or None,
            },
            requirement=_REQ_RETENTION_POLICY,
        )

        logger.info(
            "Retention policy for %s: %s -> %s (operator=%s)",
            dataset_id,
            current_policy.value,
            policy.value,
            operator,
        )
        return record

    # ------------------------------------------------------------------
    # Regulatory manifest
    # ------------------------------------------------------------------

    def generate_data_manifest(self) -> dict[str, Any]:
        """Generate a complete data inventory for regulatory submission.

        Satisfies 21 CFR Part 11 requirements for data accountability.
        """
        datasets_manifest: list[dict[str, Any]] = []

        for ds_id, ds in self._datasets.items():
            retention = self._retention_records.get(ds_id)
            quality = self._quality_reports.get(ds_id)

            entry: dict[str, Any] = {
                "dataset_id": ds_id,
                "dataset_id_hmac": _hmac_identifier(ds_id),
                "domain": ds.domain.value,
                "site_id_hmac": _hmac_identifier(ds.site_id),
                "record_count": ds.record_count,
                "quality_level": ds.quality_level.value,
                "schema_version": ds.schema_version,
                "consent_scope": ds.consent_scope.value,
                "created_at": ds.created_at,
                "updated_at": ds.updated_at,
            }

            if retention:
                entry["retention_policy"] = retention.retention_policy.value
                entry["retention_expiry"] = retention.expiry_date
                entry["destruction_certificate"] = (
                    retention.destruction_certificate_hash or None
                )

            if quality:
                entry["quality_scores"] = {
                    "completeness": round(quality.completeness, 4),
                    "accuracy": round(quality.accuracy, 4),
                    "consistency": round(quality.consistency, 4),
                    "timeliness": round(quality.timeliness, 4),
                    "composite": round(quality.composite_score, 4),
                }

            datasets_manifest.append(entry)

        # Summary statistics
        quality_distribution: dict[str, int] = {}
        domain_distribution: dict[str, int] = {}
        total_records = 0

        for ds in self._datasets.values():
            ql = ds.quality_level.value
            quality_distribution[ql] = quality_distribution.get(ql, 0) + 1
            dm = ds.domain.value
            domain_distribution[dm] = domain_distribution.get(dm, 0) + 1
            total_records += ds.record_count

        manifest: dict[str, Any] = {
            "trial_id": self.trial_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "TrialDataManager",
            "version": "0.9.0",
            "summary": {
                "total_datasets": len(self._datasets),
                "total_records": total_records,
                "quality_distribution": quality_distribution,
                "domain_distribution": domain_distribution,
                "unique_sites": len(
                    {ds.site_id for ds in self._datasets.values()}
                ),
            },
            "datasets": datasets_manifest,
            "audit_trail_size": len(self._audit_trail),
            "manifest_integrity_hash": "",
        }

        # Compute integrity hash over the manifest content
        manifest_str = str(sorted(manifest["summary"].items()))
        manifest["manifest_integrity_hash"] = hmac.new(
            _HMAC_KEY, manifest_str.encode("utf-8"), hashlib.sha256
        ).hexdigest()[:_HASH_TRUNCATION]

        # Audit
        self._record_audit(
            action="manifest_generated",
            resource=self.trial_id,
            details={
                "total_datasets": len(self._datasets),
                "total_records": total_records,
                "integrity_hash": manifest["manifest_integrity_hash"],
            },
            requirement=_REQ_MANIFEST_GENERATION,
        )

        logger.info(
            "Data manifest generated for trial %s: %d datasets, %d records",
            self.trial_id,
            len(self._datasets),
            total_records,
        )
        return manifest

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def register_schema(
        self,
        domain: DataDomain,
        schema: SchemaDefinition,
    ) -> None:
        """Register a schema definition for a data domain.

        Args:
            domain: The clinical data domain this schema applies to.
            schema: The schema definition.
        """
        self._schemas[domain.value] = schema
        self._record_audit(
            action="schema_registered",
            resource=domain.value,
            details={
                "version": schema.version,
                "field_count": len(schema.fields),
                "required_count": len(schema.required),
            },
            requirement=_REQ_SCHEMA_VALIDATION,
        )
        logger.info(
            "Schema registered for domain %s (v%s, %d fields)",
            domain.value,
            schema.version,
            len(schema.fields),
        )

    def get_schema(self, domain: DataDomain) -> SchemaDefinition | None:
        """Retrieve the registered schema for a domain."""
        return self._schemas.get(domain.value)

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------

    def get_audit_trail(self) -> list[dict[str, Any]]:
        """Return a defensive deep copy of the 21 CFR Part 11 audit trail."""
        return copy.deepcopy(self._audit_trail)

    def _record_audit(
        self,
        action: str,
        resource: str,
        details: dict[str, Any] | None = None,
        requirement: str = "",
    ) -> None:
        """Append an entry to the internal audit trail.

        Each entry is stamped with a UTC timestamp and an HMAC-SHA256
        integrity hash for tamper detection (21 CFR Part 11).
        """
        now = datetime.now(timezone.utc).isoformat()
        entry: dict[str, Any] = {
            "action": action,
            "resource": resource,
            "operator": self._operator_id,
            "trial_id": self.trial_id,
            "timestamp": now,
            "details": details or {},
            "iec62304_requirement": requirement,
            "integrity_hash": "",
        }

        # Compute integrity hash over key fields
        hash_payload = (
            f"{action}:{resource}:{self._operator_id}:"
            f"{self.trial_id}:{now}:{requirement}"
        )
        entry["integrity_hash"] = hmac.new(
            _HMAC_KEY, hash_payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()[:_HASH_TRUNCATION]

        self._audit_trail.append(entry)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_dataset(self, dataset_id: str) -> ClinicalDataset | None:
        """Retrieve a registered dataset by ID."""
        return self._datasets.get(dataset_id)

    def get_quality_report(self, dataset_id: str) -> DataQualityReport | None:
        """Retrieve the most recent quality report for a dataset."""
        return self._quality_reports.get(dataset_id)

    def get_retention_record(self, dataset_id: str) -> DataRetentionRecord | None:
        """Retrieve the retention record for a dataset."""
        return self._retention_records.get(dataset_id)

    def list_datasets(
        self,
        site_id: str | None = None,
        domain: DataDomain | None = None,
        quality_level: DataQualityLevel | None = None,
    ) -> list[ClinicalDataset]:
        """List datasets with optional filtering by site, domain, or quality."""
        results: list[ClinicalDataset] = []
        for ds in self._datasets.values():
            if site_id is not None and ds.site_id != site_id:
                continue
            if domain is not None and ds.domain != domain:
                continue
            if quality_level is not None and ds.quality_level != quality_level:
                continue
            results.append(ds)
        return results

    @property
    def dataset_count(self) -> int:
        """Total number of registered datasets."""
        return len(self._datasets)

    @property
    def total_records(self) -> int:
        """Sum of record_count across all registered datasets."""
        return sum(ds.record_count for ds in self._datasets.values())
