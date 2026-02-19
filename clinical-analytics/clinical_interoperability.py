"""Clinical interoperability — DICOM/FHIR/ICD-10/SNOMED/LOINC harmonization.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.

Cross-site schema alignment and standard vocabulary mapping for federated
oncology clinical trials.  Translates data between DICOM SR, HL7 FHIR R4,
ICD-10-CM, SNOMED CT, LOINC and CDISC CDASH/SDTM/ADaM.  Includes Jaccard
similarity and edit-distance metrics for fuzzy concept matching, and
HMAC-SHA256 pseudonymisation for patient identifiers.

DISCLAIMER: RESEARCH USE ONLY.  Not validated for clinical use and MUST NOT
    be used to make clinical decisions or guide patient treatment.
LICENSE: MIT
VERSION: 0.9.0
LAST UPDATED: 2026-02-18
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from utils.crypto import get_hmac_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_HMAC_KEY: bytes = get_hmac_key()
_MIN_JACCARD_THRESHOLD: float = 0.25
_FHIR_R4_RESOURCE_TYPES: list[str] = [
    "Patient",
    "Condition",
    "Observation",
    "MedicationAdministration",
    "MedicationRequest",
    "Procedure",
    "DiagnosticReport",
    "ImagingStudy",
    "Specimen",
    "ResearchStudy",
    "ResearchSubject",
    "AdverseEvent",
]


class ClinicalStandard(str, Enum):
    """Supported clinical interoperability standards."""

    DICOM = "DICOM"
    FHIR_R4 = "FHIR_R4"
    ICD_10_CM = "ICD-10-CM"
    SNOMED_CT = "SNOMED_CT"
    LOINC = "LOINC"
    CDISC_CDASH = "CDISC_CDASH"
    CDISC_SDTM = "CDISC_SDTM"
    CDISC_ADaM = "CDISC_ADaM"
    RxNorm = "RxNorm"
    MedDRA = "MedDRA"


class MappingConfidence(str, Enum):
    """Confidence level of a vocabulary mapping."""

    EXACT = "exact"
    BROADER = "broader"
    NARROWER = "narrower"
    RELATED = "related"
    NO_MATCH = "no_match"


class DataExchangeFormat(str, Enum):
    """Wire / file formats for clinical data exchange."""

    JSON_FHIR = "json_fhir"
    XML_CDA = "xml_cda"
    CSV_SDTM = "csv_sdtm"
    PARQUET_ADaM = "parquet_adam"
    DICOM_SR = "dicom_sr"


class HarmonizationStatus(str, Enum):
    """Status of a concept or record in the harmonization pipeline."""

    PENDING = "pending"
    MAPPED = "mapped"
    VALIDATED = "validated"
    CONFLICT = "conflict"
    UNMAPPABLE = "unmappable"


@dataclass
class VocabularyMapping:
    """A single concept mapping between two coding systems."""

    source_code: str
    source_system: ClinicalStandard
    target_code: str
    target_system: ClinicalStandard
    confidence: MappingConfidence = MappingConfidence.EXACT
    display_name: str = ""
    last_validated: float = 0.0

    def __post_init__(self) -> None:
        if self.last_validated <= 0.0:
            self.last_validated = time.time()


@dataclass
class SchemaAlignment:
    """Result of aligning two site data-dictionary schemas."""

    source_schema: str
    target_schema: str
    field_mappings: dict[str, str] = field(default_factory=dict)
    alignment_score: float = 0.0
    unmapped_source: list[str] = field(default_factory=list)
    unmapped_target: list[str] = field(default_factory=list)


@dataclass
class InteroperabilityConfig:
    """Configuration for a ClinicalInteroperabilityEngine instance."""

    standards: list[ClinicalStandard] = field(
        default_factory=lambda: [
            ClinicalStandard.FHIR_R4,
            ClinicalStandard.ICD_10_CM,
            ClinicalStandard.SNOMED_CT,
            ClinicalStandard.LOINC,
        ]
    )
    mapping_files: list[str] = field(default_factory=list)
    validation_rules: dict[str, Any] = field(default_factory=dict)
    hmac_key: bytes = _HMAC_KEY
    fuzzy_threshold: float = 0.60


@dataclass
class HarmonizationResult:
    """Outcome summary produced by ``harmonize_dataset``."""

    records_processed: int = 0
    mapped: int = 0
    unmapped: int = 0
    conflicts: int = 0
    quality_score: float = 0.0
    elapsed_seconds: float = 0.0
    audit_trail: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CrossSiteSchemaReport:
    """Cross-site interoperability assessment."""

    sites: list[str] = field(default_factory=list)
    schema_versions: dict[str, str] = field(default_factory=dict)
    alignment_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    recommendations: list[str] = field(default_factory=list)
    generated_at: float = 0.0

    def __post_init__(self) -> None:
        if self.generated_at <= 0.0:
            self.generated_at = time.time()


_ICD10_TO_SNOMED: dict[str, dict[str, str]] = {
    "C34.1": {"snomed": "254637007", "display": "Non-small cell lung cancer"},
    "C34.9": {"snomed": "93880001", "display": "Primary malignant neoplasm of lung"},
    "C50.9": {"snomed": "254837009", "display": "Malignant neoplasm of breast"},
    "C18.9": {"snomed": "363406005", "display": "Malignant neoplasm of colon"},
    "C61": {"snomed": "399068003", "display": "Malignant neoplasm of prostate"},
    "C56.9": {"snomed": "363443007", "display": "Malignant neoplasm of ovary"},
    "C71.9": {"snomed": "428061005", "display": "Malignant neoplasm of brain"},
    "C43.9": {"snomed": "372244006", "display": "Malignant melanoma of skin"},
    "C25.9": {"snomed": "363418001", "display": "Malignant neoplasm of pancreas"},
    "C64.9": {"snomed": "363358000", "display": "Malignant neoplasm of kidney"},
    "C22.0": {"snomed": "187769009", "display": "Hepatocellular carcinoma"},
    "C90.0": {"snomed": "109989006", "display": "Multiple myeloma"},
    "C91.0": {"snomed": "91857003", "display": "Acute lymphoblastic leukaemia"},
    "C92.0": {"snomed": "91861009", "display": "Acute myeloid leukaemia"},
}

_LOINC_REFERENCE: dict[str, dict[str, str]] = {
    "26464-8": {"display": "Leukocytes [#/volume] in Blood", "component": "WBC", "unit": "10^9/L"},
    "718-7": {"display": "Hemoglobin [Mass/volume] in Blood", "component": "Hemoglobin", "unit": "g/dL"},
    "777-3": {"display": "Platelets [#/volume] in Blood", "component": "Platelets", "unit": "10^9/L"},
    "2160-0": {"display": "Creatinine [Mass/volume] in Serum", "component": "Creatinine", "unit": "mg/dL"},
    "1742-6": {"display": "ALT [Enzymatic activity/volume]", "component": "ALT", "unit": "U/L"},
    "1920-8": {"display": "AST [Enzymatic activity/volume]", "component": "AST", "unit": "U/L"},
    "83064-6": {"display": "PD-L1 by clone 22C3", "component": "PD-L1 TPS", "unit": "%"},
    "85337-4": {"display": "Estrogen receptor Ag [Presence]", "component": "ER Status", "unit": "categorical"},
    "48676-1": {"display": "HER2 [Interpretation] in Tissue", "component": "HER2 Status", "unit": "categorical"},
    "29463-7": {"display": "Body weight", "component": "Weight", "unit": "kg"},
    "8302-2": {"display": "Body height", "component": "Height", "unit": "cm"},
}

_RXNORM_REFERENCE: dict[str, dict[str, str]] = {
    "1946832": {"display": "Pembrolizumab 25 MG/ML Injectable Solution", "ingredient": "pembrolizumab"},
    "1946840": {"display": "Nivolumab 10 MG/ML Injectable Solution", "ingredient": "nivolumab"},
    "1719773": {"display": "Atezolizumab 60 MG/ML Injectable Solution", "ingredient": "atezolizumab"},
    "583218": {"display": "Cisplatin 1 MG/ML Injectable Solution", "ingredient": "cisplatin"},
    "597195": {"display": "Carboplatin 10 MG/ML Injectable Solution", "ingredient": "carboplatin"},
    "1740692": {"display": "Paclitaxel 6 MG/ML Injectable Suspension", "ingredient": "paclitaxel"},
    "262105": {"display": "Doxorubicin 2 MG/ML Injectable Solution", "ingredient": "doxorubicin"},
    "1308738": {"display": "Trastuzumab 150 MG Injection", "ingredient": "trastuzumab"},
}

_SITE_LOCAL_LAB_CODES: dict[str, dict[str, str]] = {
    "SITE-001": {"718-7": "HGB", "777-3": "PLT", "26464-8": "WBC", "2160-0": "CREAT"},
    "SITE-002": {"718-7": "HB", "777-3": "PLTS", "26464-8": "LEUK", "2160-0": "CRE"},
    "SITE-003": {"718-7": "Hgb", "777-3": "Plt", "26464-8": "Wbc", "2160-0": "Cr"},
}


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            curr_row.append(min(prev_row[j + 1] + 1, curr_row[j] + 1, prev_row[j] + (0 if c1 == c2 else 1)))
        prev_row = curr_row
    return prev_row[-1]


def _normalised_edit_similarity(s1: str, s2: str) -> float:
    """Return similarity in [0, 1] based on normalised edit distance."""
    max_len = max(len(s1), len(s2))
    return 1.0 if max_len == 0 else 1.0 - _levenshtein_distance(s1, s2) / max_len


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard index between two sets of strings."""
    if not set_a and not set_b:
        return 1.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union else 0.0


def _tokenise(text: str) -> set[str]:
    """Tokenise a clinical term into lowercase alphanumeric tokens."""
    cleaned = text.lower().replace("-", " ").replace("_", " ").replace("/", " ")
    return {tok.strip() for tok in cleaned.split() if tok.strip()}


def _pseudonymise_id(raw_id: str, key: bytes = _HMAC_KEY) -> str:
    """HMAC-SHA256 pseudonymisation for patient identifiers."""
    digest = hmac.new(key, raw_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"PSDM-{digest[:16].upper()}"


def _fuzzy_score(query_tokens: set[str], query_lower: str, display: str) -> float:
    """Combined Jaccard + edit-distance similarity score."""
    display_tokens = _tokenise(display)
    jaccard = _jaccard_similarity(query_tokens, display_tokens)
    edit_sim = _normalised_edit_similarity(query_lower, display.lower())
    return 0.6 * jaccard + 0.4 * edit_sim


class ClinicalInteroperabilityEngine:
    """Unified engine for cross-standard vocabulary mapping and schema alignment.

    Maintains in-memory reference vocabularies for common oncology concepts
    and exposes methods for concept mapping, schema alignment, dataset
    harmonization, FHIR validation, and CDISC export.
    """

    def __init__(
        self,
        config: InteroperabilityConfig | None = None,
        site_id: str = "SITE-001",
    ) -> None:
        self.config = config or InteroperabilityConfig()
        self.site_id = site_id
        self._audit_log: list[dict[str, Any]] = []
        self._icd10_snomed = dict(_ICD10_TO_SNOMED)
        self._loinc_ref = dict(_LOINC_REFERENCE)
        self._rxnorm_ref = dict(_RXNORM_REFERENCE)
        self._site_local_labs = dict(_SITE_LOCAL_LAB_CODES)

        # Build reverse indexes
        self._snomed_to_icd10: dict[str, list[str]] = {}
        for icd_code, info in self._icd10_snomed.items():
            self._snomed_to_icd10.setdefault(info["snomed"], []).append(icd_code)
        self._loinc_by_component: dict[str, str] = {}
        for loinc_code, info in self._loinc_ref.items():
            comp = info.get("component", "").lower()
            if comp:
                self._loinc_by_component[comp] = loinc_code

        logger.info(
            "ClinicalInteroperabilityEngine initialised — site=%s, standards=%s, ICD-10=%d, LOINC=%d, RxNorm=%d",
            self.site_id,
            [s.value for s in self.config.standards],
            len(self._icd10_snomed),
            len(self._loinc_ref),
            len(self._rxnorm_ref),
        )

    def _record_audit(self, action: str, details: dict[str, Any]) -> None:
        """Append an entry to the in-memory audit trail."""
        self._audit_log.append(
            {
                "audit_id": f"AUD-{uuid.uuid4().hex[:10].upper()}",
                "timestamp": time.time(),
                "site_id": self.site_id,
                "action": action,
                **details,
            }
        )
        logger.debug("Audit: %s — %s", action, json.dumps(details, default=str))

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        """Return a copy of the full audit trail."""
        return list(self._audit_log)

    def map_vocabulary(
        self,
        source_code: str,
        source_system: ClinicalStandard,
        target_system: ClinicalStandard,
    ) -> VocabularyMapping:
        """Map *source_code* from *source_system* to *target_system*.

        Exact lookup first, then fuzzy fallback (Jaccard + edit distance).
        """
        self._record_audit(
            "map_vocabulary",
            {
                "source_code": source_code,
                "source_system": source_system.value,
                "target_system": target_system.value,
            },
        )

        # ICD-10-CM -> SNOMED CT
        if source_system == ClinicalStandard.ICD_10_CM and target_system == ClinicalStandard.SNOMED_CT:
            result = self._lookup_icd10_to_snomed(source_code)
            if result is not None:
                return result

        # SNOMED CT -> ICD-10-CM (reverse)
        if source_system == ClinicalStandard.SNOMED_CT and target_system == ClinicalStandard.ICD_10_CM:
            icd_codes = self._snomed_to_icd10.get(source_code, [])
            if icd_codes:
                return VocabularyMapping(
                    source_code=source_code,
                    source_system=source_system,
                    target_code=icd_codes[0],
                    target_system=target_system,
                    confidence=MappingConfidence.BROADER if len(icd_codes) > 1 else MappingConfidence.EXACT,
                    display_name=self._icd10_snomed.get(icd_codes[0], {}).get("display", ""),
                )

        # LOINC enrichment
        if source_system == ClinicalStandard.LOINC:
            ref = self._loinc_ref.get(source_code)
            if ref is not None:
                return VocabularyMapping(
                    source_code=source_code,
                    source_system=source_system,
                    target_code=source_code,
                    target_system=target_system,
                    confidence=MappingConfidence.EXACT,
                    display_name=ref.get("display", ""),
                )

        # RxNorm lookup
        if source_system == ClinicalStandard.RxNorm:
            ref = self._rxnorm_ref.get(source_code)
            if ref is not None:
                return VocabularyMapping(
                    source_code=source_code,
                    source_system=source_system,
                    target_code=source_code,
                    target_system=target_system,
                    confidence=MappingConfidence.EXACT,
                    display_name=ref.get("display", ""),
                )

        # Fuzzy fallback
        best = self._fuzzy_search_all(source_code, target_system)
        if best is not None:
            return best

        logger.warning("No mapping found: %s (%s) -> %s", source_code, source_system.value, target_system.value)
        return VocabularyMapping(
            source_code=source_code,
            source_system=source_system,
            target_code="",
            target_system=target_system,
            confidence=MappingConfidence.NO_MATCH,
        )

    def _lookup_icd10_to_snomed(self, icd_code: str) -> VocabularyMapping | None:
        """Exact + prefix ICD-10-CM to SNOMED CT lookup."""
        info = self._icd10_snomed.get(icd_code)
        if info is not None:
            return VocabularyMapping(
                source_code=icd_code,
                source_system=ClinicalStandard.ICD_10_CM,
                target_code=info["snomed"],
                target_system=ClinicalStandard.SNOMED_CT,
                confidence=MappingConfidence.EXACT,
                display_name=info.get("display", ""),
            )
        for ref_code, ref_info in self._icd10_snomed.items():
            if ref_code.startswith(icd_code) or icd_code.startswith(ref_code.split(".")[0]):
                return VocabularyMapping(
                    source_code=icd_code,
                    source_system=ClinicalStandard.ICD_10_CM,
                    target_code=ref_info["snomed"],
                    target_system=ClinicalStandard.SNOMED_CT,
                    confidence=MappingConfidence.BROADER,
                    display_name=ref_info.get("display", ""),
                )
        return None

    def _fuzzy_search_all(
        self,
        query: str,
        target_system: ClinicalStandard,
    ) -> VocabularyMapping | None:
        """Search all reference tables with fuzzy matching on display names."""
        query_tokens = _tokenise(query)
        if not query_tokens:
            return None

        best_score, best_code, best_display = 0.0, "", ""
        best_src = ClinicalStandard.FHIR_R4
        query_lower = query.lower()

        # Build (code, display, source_system) candidates from all tables
        candidates: list[tuple[str, str, ClinicalStandard]] = []
        for icd_code, info in self._icd10_snomed.items():
            code = info["snomed"] if target_system == ClinicalStandard.SNOMED_CT else icd_code
            candidates.append((code, info.get("display", ""), ClinicalStandard.ICD_10_CM))
        for loinc_code, info in self._loinc_ref.items():
            candidates.append((loinc_code, info.get("display", ""), ClinicalStandard.LOINC))
        for rxn_code, info in self._rxnorm_ref.items():
            candidates.append((rxn_code, info.get("display", ""), ClinicalStandard.RxNorm))

        for code, display, src_sys in candidates:
            score = _fuzzy_score(query_tokens, query_lower, display)
            if score > best_score:
                best_score, best_code, best_display, best_src = score, code, display, src_sys

        if best_score >= self.config.fuzzy_threshold:
            if best_score >= 0.90:
                conf = MappingConfidence.EXACT
            elif best_score >= 0.70:
                conf = MappingConfidence.RELATED
            else:
                conf = MappingConfidence.NARROWER
            return VocabularyMapping(
                source_code=query,
                source_system=best_src,
                target_code=best_code,
                target_system=target_system,
                confidence=conf,
                display_name=best_display,
            )
        return None

    def align_schemas(
        self,
        schema_a: dict[str, str],
        schema_b: dict[str, str],
    ) -> SchemaAlignment:
        """Compute field-level alignment between two data-dictionary schemas.

        Combines Jaccard similarity on tokens with normalised edit distance.
        """
        self._record_audit(
            "align_schemas",
            {
                "schema_a_fields": len(schema_a),
                "schema_b_fields": len(schema_b),
            },
        )
        field_mappings: dict[str, str] = {}
        match_scores: list[float] = []
        unmapped_source: list[str] = []
        used_targets: set[str] = set()

        for field_a, desc_a in schema_a.items():
            tokens_a = _tokenise(field_a) | _tokenise(desc_a)
            best_match, best_score = "", 0.0
            for field_b, desc_b in schema_b.items():
                if field_b in used_targets:
                    continue
                tokens_b = _tokenise(field_b) | _tokenise(desc_b)
                combined = 0.5 * _normalised_edit_similarity(
                    field_a.lower(), field_b.lower()
                ) + 0.5 * _jaccard_similarity(tokens_a, tokens_b)
                if combined > best_score:
                    best_score, best_match = combined, field_b

            if best_score >= _MIN_JACCARD_THRESHOLD and best_match:
                field_mappings[field_a] = best_match
                match_scores.append(best_score)
                used_targets.add(best_match)
            else:
                unmapped_source.append(field_a)
                match_scores.append(0.0)

        alignment_score = float(np.mean(match_scores)) if match_scores else 0.0
        result = SchemaAlignment(
            source_schema="schema_a",
            target_schema="schema_b",
            field_mappings=field_mappings,
            alignment_score=float(np.clip(alignment_score, 0.0, 1.0)),
            unmapped_source=unmapped_source,
            unmapped_target=[f for f in schema_b if f not in used_targets],
        )
        logger.info(
            "Schema alignment: %d/%d fields mapped, score=%.3f",
            len(field_mappings),
            len(schema_a),
            result.alignment_score,
        )
        return result

    def harmonize_dataset(
        self,
        data: list[dict[str, Any]],
        source_standard: ClinicalStandard,
        target_standard: ClinicalStandard,
    ) -> tuple[list[dict[str, Any]], HarmonizationResult]:
        """Transform *data* from *source_standard* to *target_standard*.

        Maps coded fields via vocabulary engine and pseudonymises patient IDs.
        """
        start_time = time.time()
        self._record_audit(
            "harmonize_dataset",
            {
                "record_count": len(data),
                "source": source_standard.value,
                "target": target_standard.value,
            },
        )
        harmonized: list[dict[str, Any]] = []
        result = HarmonizationResult(records_processed=len(data))
        _id_keys = {"patient_id", "subject_id", "mrn", "patientid"}
        _code_keys = {"diagnosis_code", "icd_code", "condition_code", "code"}
        _lab_keys = {"loinc_code", "lab_code", "observation_code"}

        for record in data:
            mapped_rec: dict[str, Any] = {}
            status = HarmonizationStatus.PENDING
            notes: list[str] = []

            for key, value in record.items():
                k_low = key.lower()
                if k_low in _id_keys:
                    mapped_rec[key] = _pseudonymise_id(str(value), self.config.hmac_key)
                elif k_low in _code_keys:
                    m = self.map_vocabulary(str(value), source_standard, target_standard)
                    if m.confidence != MappingConfidence.NO_MATCH:
                        mapped_rec[key] = m.target_code
                        mapped_rec[f"{key}_display"] = m.display_name
                        mapped_rec[f"{key}_confidence"] = m.confidence.value
                        notes.append(f"{key}: {value} -> {m.target_code} ({m.confidence.value})")
                    else:
                        mapped_rec[key] = value
                        mapped_rec[f"{key}_status"] = HarmonizationStatus.UNMAPPABLE.value
                        status = HarmonizationStatus.CONFLICT
                        notes.append(f"{key}: {value} -> UNMAPPABLE")
                elif k_low in _lab_keys:
                    m = self.map_vocabulary(str(value), ClinicalStandard.LOINC, target_standard)
                    mapped_rec[key] = m.target_code if m.confidence != MappingConfidence.NO_MATCH else value
                    if m.confidence != MappingConfidence.NO_MATCH:
                        mapped_rec[f"{key}_display"] = m.display_name
                else:
                    mapped_rec[key] = value

            if status == HarmonizationStatus.CONFLICT:
                result.conflicts += 1
            elif any("UNMAPPABLE" in n for n in notes):
                result.unmapped += 1
                status = HarmonizationStatus.UNMAPPABLE
            else:
                result.mapped += 1
                status = HarmonizationStatus.MAPPED
            mapped_rec["_harmonization_status"] = status.value
            harmonized.append(mapped_rec)
            result.audit_trail.append({"status": status.value, "notes": notes})

        result.elapsed_seconds = time.time() - start_time
        total = result.mapped + result.unmapped + result.conflicts
        result.quality_score = float(result.mapped / total) if total > 0 else 0.0
        logger.info(
            "Harmonized %d records (%s -> %s): mapped=%d, unmapped=%d, conflicts=%d, quality=%.3f",
            result.records_processed,
            source_standard.value,
            target_standard.value,
            result.mapped,
            result.unmapped,
            result.conflicts,
            result.quality_score,
        )
        return harmonized, result

    def validate_fhir_resource(self, resource: dict[str, Any]) -> dict[str, Any]:
        """Simplified FHIR R4 structural validation.

        Returns dict with ``valid``, ``errors``, and ``warnings`` keys.
        """
        self._record_audit(
            "validate_fhir_resource",
            {
                "resource_type": resource.get("resourceType", "unknown"),
            },
        )
        errors: list[str] = []
        warnings: list[str] = []

        if not isinstance(resource, dict):
            return {"valid": False, "errors": ["Resource is not a JSON object"], "warnings": []}

        rtype = resource.get("resourceType", "")
        if not rtype:
            errors.append("Missing required field: resourceType")
        elif rtype not in _FHIR_R4_RESOURCE_TYPES:
            errors.append(f"Unsupported resource type: {rtype}")
        if "id" not in resource:
            errors.append("Missing required field: id")

        if rtype == "Patient":
            for fld in ("gender", "birthDate"):
                if fld not in resource:
                    warnings.append(f"Patient missing recommended field: {fld}")
        elif rtype == "Condition":
            if "code" not in resource:
                errors.append("Condition missing required field: code")
            else:
                codings = resource["code"].get("coding", []) if isinstance(resource["code"], dict) else []
                if not codings:
                    errors.append("Condition.code.coding is empty")
                for c in codings:
                    sys = c.get("system", "")
                    if sys and "icd" not in sys.lower() and "snomed" not in sys.lower():
                        warnings.append(f"Non-standard coding system: {sys}")
            if "subject" not in resource:
                errors.append("Condition missing required field: subject")
        elif rtype == "Observation":
            if "code" not in resource:
                errors.append("Observation missing required field: code")
            if "status" not in resource:
                errors.append("Observation missing required field: status")
            elif resource["status"] not in (
                "registered",
                "preliminary",
                "final",
                "amended",
                "corrected",
                "cancelled",
                "entered-in-error",
                "unknown",
            ):
                errors.append(f"Invalid Observation.status: {resource['status']}")
            if "valueQuantity" not in resource and "valueCodeableConcept" not in resource:
                warnings.append("Observation has no value[x] element")
        elif rtype == "MedicationAdministration":
            if "medicationCodeableConcept" not in resource and "medicationReference" not in resource:
                errors.append("MedicationAdministration missing medication element")
            for fld in ("subject", "status"):
                if fld not in resource:
                    errors.append(f"MedicationAdministration missing required field: {fld}")
        elif rtype in ("Procedure", "DiagnosticReport"):
            for fld in ("code", "status"):
                if fld not in resource:
                    errors.append(f"{rtype} missing required field: {fld}")
            if rtype == "Procedure" and "subject" not in resource:
                errors.append("Procedure missing required field: subject")
        elif rtype == "AdverseEvent":
            if "event" not in resource and "code" not in resource:
                warnings.append("AdverseEvent missing event/code element")
            if "subject" not in resource:
                errors.append("AdverseEvent missing required field: subject")

        is_valid = len(errors) == 0
        logger.info(
            "FHIR validation for %s/%s: valid=%s, errors=%d, warnings=%d",
            rtype,
            resource.get("id", "?"),
            is_valid,
            len(errors),
            len(warnings),
        )
        return {
            "valid": is_valid,
            "resource_type": rtype,
            "errors": errors,
            "warnings": warnings,
            "error_count": len(errors),
            "warning_count": len(warnings),
        }

    def map_icd10_to_snomed(self, icd_code: str) -> VocabularyMapping:
        """Convenience: map an ICD-10-CM *icd_code* to SNOMED CT."""
        return self.map_vocabulary(icd_code, ClinicalStandard.ICD_10_CM, ClinicalStandard.SNOMED_CT)

    def map_loinc_to_local(self, loinc_code: str, site_id: str) -> VocabularyMapping:
        """Map a LOINC code to a site-local lab code, with component fallback."""
        self._record_audit("map_loinc_to_local", {"loinc_code": loinc_code, "site_id": site_id})
        local_code = self._site_local_labs.get(site_id, {}).get(loinc_code)
        if local_code is not None:
            return VocabularyMapping(
                source_code=loinc_code,
                source_system=ClinicalStandard.LOINC,
                target_code=local_code,
                target_system=ClinicalStandard.LOINC,
                confidence=MappingConfidence.EXACT,
                display_name=self._loinc_ref.get(loinc_code, {}).get("display", ""),
            )
        ref = self._loinc_ref.get(loinc_code)
        if ref is not None:
            return VocabularyMapping(
                source_code=loinc_code,
                source_system=ClinicalStandard.LOINC,
                target_code=ref.get("component", loinc_code),
                target_system=ClinicalStandard.LOINC,
                confidence=MappingConfidence.RELATED,
                display_name=ref.get("display", ""),
            )
        logger.warning("LOINC code %s not found in reference table", loinc_code)
        return VocabularyMapping(
            source_code=loinc_code,
            source_system=ClinicalStandard.LOINC,
            target_code="",
            target_system=ClinicalStandard.LOINC,
            confidence=MappingConfidence.NO_MATCH,
        )

    def generate_cross_site_alignment_report(
        self,
        sites: dict[str, dict[str, str]],
    ) -> CrossSiteSchemaReport:
        """Pairwise alignment report across *sites* (site_id -> schema dict)."""
        self._record_audit("cross_site_report", {"site_count": len(sites)})
        site_ids = sorted(sites.keys())
        n = len(site_ids)
        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            matrix[i, i] = 1.0
            for j in range(i + 1, n):
                score = self.align_schemas(sites[site_ids[i]], sites[site_ids[j]]).alignment_score
                matrix[i, j] = matrix[j, i] = score

        recs: list[str] = []
        mean_score = float(np.mean(matrix[np.triu_indices(n, k=1)])) if n > 1 else 1.0
        if mean_score < 0.5:
            recs.append("Overall alignment LOW (< 0.50). Adopt a common data dictionary.")
        elif mean_score < 0.75:
            recs.append("Alignment MODERATE. Review unmapped high-priority fields.")
        else:
            recs.append("Alignment GOOD (>= 0.75). Proceed with automated harmonization.")
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i, j] < 0.40:
                    recs.append(
                        f"Sites {site_ids[i]} and {site_ids[j]} have low alignment "
                        f"({matrix[i, j]:.2f}). Manual review recommended."
                    )

        logger.info("Cross-site report: %d sites, mean=%.3f", n, mean_score)
        return CrossSiteSchemaReport(
            sites=site_ids,
            schema_versions={s: f"v1.0-{s}" for s in site_ids},
            alignment_matrix=matrix,
            recommendations=recs,
        )

    def export_to_cdisc(
        self,
        data: list[dict[str, Any]],
        target_format: DataExchangeFormat = DataExchangeFormat.CSV_SDTM,
    ) -> dict[str, Any]:
        """Export *data* as CDISC SDTM (DM/AE/LB) or ADaM (ADSL/ADAE)."""
        self._record_audit("export_to_cdisc", {"record_count": len(data), "format": target_format.value})
        export_id = f"CDISC-{uuid.uuid4().hex[:10].upper()}"
        if target_format == DataExchangeFormat.PARQUET_ADaM:
            return self._export_adam(data, export_id)
        return self._export_sdtm(data, export_id)

    def _export_sdtm(self, data: list[dict[str, Any]], export_id: str) -> dict[str, Any]:
        """Build SDTM domain tables (DM, AE, LB) from harmonized records."""
        dm_rows: list[dict[str, str]] = []
        ae_rows: list[dict[str, str]] = []
        lb_rows: list[dict[str, str]] = []

        for idx, rec in enumerate(data):
            study = rec.get("study_id", "ONCO-FL-001")
            subj = _pseudonymise_id(
                str(rec.get("patient_id", rec.get("subject_id", f"SUBJ-{idx:04d}"))), self.config.hmac_key
            )
            dm_rows.append(
                {
                    "STUDYID": str(study),
                    "DOMAIN": "DM",
                    "USUBJID": subj,
                    "SUBJID": subj,
                    "SITEID": rec.get("site_id", self.site_id),
                    "AGE": str(rec.get("age", "")),
                    "AGEU": "YEARS",
                    "SEX": str(rec.get("sex", rec.get("gender", ""))).upper()[:1],
                    "RACE": str(rec.get("race", "")),
                    "ARMCD": str(rec.get("arm_code", rec.get("treatment_arm", ""))),
                    "ARM": str(rec.get("arm", rec.get("treatment_arm_description", ""))),
                }
            )
            ae_term = rec.get("adverse_event", rec.get("ae_term", ""))
            if ae_term:
                ae_rows.append(
                    {
                        "STUDYID": str(study),
                        "DOMAIN": "AE",
                        "USUBJID": subj,
                        "AETERM": str(ae_term),
                        "AEDECOD": str(rec.get("ae_preferred_term", ae_term)),
                        "AESEV": str(rec.get("ae_severity", "")),
                        "AESER": str(rec.get("ae_serious", "")),
                        "AEREL": str(rec.get("ae_relatedness", "")),
                        "AESTDTC": str(rec.get("ae_start_date", "")),
                        "AEENDTC": str(rec.get("ae_end_date", "")),
                    }
                )
            lab_code = rec.get("loinc_code", rec.get("lab_code", ""))
            lab_val = rec.get("lab_value", rec.get("value", ""))
            if lab_code or lab_val:
                li = self._loinc_ref.get(str(lab_code), {})
                lb_rows.append(
                    {
                        "STUDYID": str(study),
                        "DOMAIN": "LB",
                        "USUBJID": subj,
                        "LBTESTCD": str(li.get("component", lab_code)),
                        "LBTEST": str(li.get("display", lab_code)),
                        "LBORRES": str(lab_val),
                        "LBORRESU": str(rec.get("unit", li.get("unit", ""))),
                        "LBSTRESN": str(lab_val),
                        "LBSTRESU": str(rec.get("unit", li.get("unit", ""))),
                        "LBDTC": str(rec.get("collection_date", "")),
                    }
                )

        domains: dict[str, str] = {}
        for name, rows in [("DM", dm_rows), ("AE", ae_rows), ("LB", lb_rows)]:
            if rows:
                hdrs = list(rows[0].keys())
                domains[name] = "\n".join([",".join(hdrs)] + [",".join(r.get(h, "") for h in hdrs) for r in rows])

        logger.info("SDTM export %s: DM=%d, AE=%d, LB=%d", export_id, len(dm_rows), len(ae_rows), len(lb_rows))
        return {
            "export_id": export_id,
            "format": DataExchangeFormat.CSV_SDTM.value,
            "domains": list(domains.keys()),
            "domain_data": domains,
            "row_counts": {"DM": len(dm_rows), "AE": len(ae_rows), "LB": len(lb_rows)},
            "total_rows": len(dm_rows) + len(ae_rows) + len(lb_rows),
            "timestamp": time.time(),
        }

    def _export_adam(self, data: list[dict[str, Any]], export_id: str) -> dict[str, Any]:
        """Build ADaM-style analysis datasets (ADSL, ADAE) from harmonized records."""
        adsl: list[dict[str, Any]] = []
        adae: list[dict[str, Any]] = []
        for idx, rec in enumerate(data):
            study = rec.get("study_id", "ONCO-FL-001")
            subj = _pseudonymise_id(
                str(rec.get("patient_id", rec.get("subject_id", f"SUBJ-{idx:04d}"))), self.config.hmac_key
            )
            age_val = rec.get("age")
            adsl.append(
                {
                    "STUDYID": str(study),
                    "USUBJID": subj,
                    "SUBJID": subj,
                    "SITEID": rec.get("site_id", self.site_id),
                    "AGE": float(age_val) if age_val is not None else np.nan,
                    "AGEU": "YEARS",
                    "SEX": str(rec.get("sex", rec.get("gender", ""))).upper()[:1],
                    "RACE": str(rec.get("race", "")),
                    "TRT01P": str(rec.get("treatment_arm", "")),
                    "TRT01A": str(rec.get("actual_treatment", rec.get("treatment_arm", ""))),
                    "SAFFL": "Y" if rec.get("safety_population", True) else "N",
                    "ITTFL": "Y" if rec.get("itt_population", True) else "N",
                }
            )
            ae_term = rec.get("adverse_event", rec.get("ae_term", ""))
            if ae_term:
                adae.append(
                    {
                        "STUDYID": str(study),
                        "USUBJID": subj,
                        "AETERM": str(ae_term),
                        "AEDECOD": str(rec.get("ae_preferred_term", ae_term)),
                        "AESEV": str(rec.get("ae_severity", "")),
                        "AESER": str(rec.get("ae_serious", "")),
                        "TRTEMFL": "Y",
                    }
                )

        logger.info("ADaM export %s: ADSL=%d, ADAE=%d", export_id, len(adsl), len(adae))
        return {
            "export_id": export_id,
            "format": DataExchangeFormat.PARQUET_ADaM.value,
            "datasets": ["ADSL", "ADAE"],
            "row_counts": {"ADSL": len(adsl), "ADAE": len(adae)},
            "total_rows": len(adsl) + len(adae),
            "adsl_data": adsl,
            "adae_data": adae,
            "timestamp": time.time(),
        }

    def get_supported_standards(self) -> list[str]:
        """Return the list of enabled clinical standards."""
        return [s.value for s in self.config.standards]

    def get_vocabulary_stats(self) -> dict[str, int]:
        """Return counts of entries in each reference vocabulary."""
        return {
            "icd10_to_snomed": len(self._icd10_snomed),
            "loinc_reference": len(self._loinc_ref),
            "rxnorm_reference": len(self._rxnorm_ref),
            "site_local_lab_maps": len(self._site_local_labs),
        }

    def lookup_loinc_by_component(self, component_name: str) -> str | None:
        """Find a LOINC code by its component name (case-insensitive)."""
        return self._loinc_by_component.get(component_name.lower())

    def register_custom_mapping(self, icd_code: str, snomed_code: str, display: str) -> None:
        """Register a custom ICD-10-CM to SNOMED CT mapping at runtime."""
        self._icd10_snomed[icd_code] = {"snomed": snomed_code, "display": display}
        self._snomed_to_icd10.setdefault(snomed_code, []).append(icd_code)
        self._record_audit(
            "register_custom_mapping", {"icd_code": icd_code, "snomed_code": snomed_code, "display": display}
        )
        logger.info("Registered custom mapping: %s -> %s (%s)", icd_code, snomed_code, display)

    def register_site_lab_codes(self, site_id: str, loinc_to_local: dict[str, str]) -> None:
        """Register site-local lab code aliases for LOINC codes."""
        existing = self._site_local_labs.setdefault(site_id, {})
        existing.update(loinc_to_local)
        self._record_audit("register_site_lab_codes", {"site_id": site_id, "codes_registered": len(loinc_to_local)})
        logger.info(
            "Registered %d lab code aliases for site %s (total=%d)", len(loinc_to_local), site_id, len(existing)
        )
