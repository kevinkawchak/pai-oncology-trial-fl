"""Tests for digital-twins/clinical-integration/clinical_integrator.py.

Covers ClinicalIntegrator creation, system registration, connection,
data ingestion with de-identification, HMAC patient ID hashing,
FHIR validation, CSV parsing, export, and FederatedClinicalBridge.

RESEARCH USE ONLY.
"""

from __future__ import annotations

import hashlib
import hmac

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module(
    "clinical_integrator",
    "digital-twins/clinical-integration/clinical_integrator.py",
)

ClinicalSystem = mod.ClinicalSystem
IntegrationStatus = mod.IntegrationStatus
DataFormat = mod.DataFormat
DataCategory = mod.DataCategory
ClinicalDataPoint = mod.ClinicalDataPoint
IntegrationConfig = mod.IntegrationConfig
SyncResult = mod.SyncResult
ClinicalReport = mod.ClinicalReport
ClinicalIntegrator = mod.ClinicalIntegrator
FederatedClinicalBridge = mod.FederatedClinicalBridge
_deidentify_record = mod._deidentify_record
_validate_fhir_resource = mod._validate_fhir_resource
_compute_data_hash = mod._compute_data_hash
_parse_csv_records = mod._parse_csv_records


class TestClinicalSystemEnum:
    """Tests for ClinicalSystem enum."""

    def test_expected_members(self):
        """All expected clinical system types exist."""
        expected = {
            "electronic_health_record",
            "picture_archiving_communication_system",
            "laboratory_information_system",
            "radiology_information_system",
            "treatment_planning_system",
        }
        actual = {s.value for s in ClinicalSystem}
        assert expected == actual


class TestIntegrationStatusEnum:
    """Tests for IntegrationStatus enum."""

    def test_expected_members(self):
        """All expected statuses exist."""
        expected = {"connected", "disconnected", "syncing", "error"}
        actual = {s.value for s in IntegrationStatus}
        assert expected == actual


class TestClinicalDataPoint:
    """Tests for ClinicalDataPoint dataclass."""

    def test_auto_id_generation(self):
        """Data ID is auto-generated if not provided."""
        dp = ClinicalDataPoint()
        assert dp.data_id.startswith("CDP-")
        assert len(dp.data_id) > 4

    def test_auto_patient_id(self):
        """Patient ID is auto-generated if empty."""
        dp = ClinicalDataPoint()
        assert dp.patient_id.startswith("UNKNOWN-")

    def test_explicit_values(self):
        """Explicit values are preserved."""
        dp = ClinicalDataPoint(
            data_id="CDP-TEST",
            patient_id="PAT-001",
            name="CEA",
            value=5.2,
            unit="ng/mL",
        )
        assert dp.data_id == "CDP-TEST"
        assert dp.patient_id == "PAT-001"


class TestIntegrationConfig:
    """Tests for IntegrationConfig dataclass."""

    def test_auto_config_id(self):
        """Config ID is auto-generated."""
        config = IntegrationConfig()
        assert config.config_id.startswith("INTG-")

    def test_batch_size_clamped(self):
        """Batch size is clamped to [1, MAX_RECORDS_PER_SYNC]."""
        config = IntegrationConfig(batch_size=0)
        assert config.batch_size >= 1

    def test_polling_interval_clamped(self):
        """Polling interval is clamped to [10, 86400]."""
        config = IntegrationConfig(polling_interval_seconds=1.0)
        assert config.polling_interval_seconds >= 10.0


class TestDeidentifyRecord:
    """Tests for _deidentify_record."""

    def test_phi_fields_removed(self):
        """PHI fields like name, address, ssn are removed."""
        record = {
            "name": "John Doe",
            "address": "123 Main St",
            "ssn": "123-45-6789",
            "diagnosis": "NSCLC",
        }
        cleaned = _deidentify_record(record)
        assert "name" not in cleaned
        assert "address" not in cleaned
        assert "ssn" not in cleaned
        assert cleaned["diagnosis"] == "NSCLC"

    def test_patient_id_hashed_with_hmac(self):
        """Patient ID is replaced with HMAC-based hash."""
        record = {"patient_id": "REAL-123"}
        cleaned = _deidentify_record(record)
        assert cleaned["patient_id"].startswith("DI-")
        # Verify the HMAC hash matches
        expected_hash = (
            hmac.new(
                b"pai-oncology-patient-id-key",
                b"REAL-123",
                hashlib.sha256,
            )
            .hexdigest()[:12]
            .upper()
        )
        assert cleaned["patient_id"] == f"DI-{expected_hash}"

    def test_same_id_produces_same_hash(self):
        """Same patient ID always yields the same de-identified ID."""
        r1 = _deidentify_record({"patient_id": "PAT-A"})
        r2 = _deidentify_record({"patient_id": "PAT-A"})
        assert r1["patient_id"] == r2["patient_id"]

    def test_different_ids_produce_different_hashes(self):
        """Different patient IDs yield different hashes."""
        r1 = _deidentify_record({"patient_id": "PAT-A"})
        r2 = _deidentify_record({"patient_id": "PAT-B"})
        assert r1["patient_id"] != r2["patient_id"]


class TestValidateFhirResource:
    """Tests for _validate_fhir_resource."""

    def test_valid_fhir_resource(self):
        """A valid FHIR resource passes validation."""
        resource = {"resourceType": "Patient", "id": "p-001"}
        assert _validate_fhir_resource(resource) is True

    def test_unsupported_resource_type(self):
        """An unsupported resource type fails."""
        resource = {"resourceType": "UnsupportedType", "id": "x-001"}
        assert _validate_fhir_resource(resource) is False

    def test_missing_id_fails(self):
        """A resource without 'id' fails."""
        resource = {"resourceType": "Patient"}
        assert _validate_fhir_resource(resource) is False

    def test_non_dict_fails(self):
        """A non-dict input fails."""
        assert _validate_fhir_resource("not a dict") is False


class TestParseCsvRecords:
    """Tests for _parse_csv_records."""

    def test_valid_csv(self):
        """Valid CSV text is parsed into records."""
        csv_text = "name,value\nCEA,5.0\nLDH,200.0"
        records = _parse_csv_records(csv_text)
        assert len(records) == 2
        assert records[0]["name"] == "CEA"
        assert records[1]["value"] == "200.0"

    def test_single_line_csv(self):
        """CSV with only a header returns empty list."""
        records = _parse_csv_records("name,value")
        assert records == []


class TestClinicalIntegrator:
    """Tests for ClinicalIntegrator class."""

    def _make_integrator_with_config(self):
        """Helper to create an integrator with a registered config."""
        config = IntegrationConfig(
            config_id="CFG-001",
            system_type=ClinicalSystem.EHR,
            data_format=DataFormat.JSON,
        )
        integrator = ClinicalIntegrator(site_id="SITE-TEST", configs=[config])
        return integrator, config

    def test_creation(self):
        """Integrator is created with correct site_id."""
        integrator = ClinicalIntegrator(site_id="SITE-A")
        assert integrator.site_id == "SITE-A"
        assert integrator.ingestion_count == 0

    def test_register_system(self):
        """Registering a config adds it to the integrator."""
        integrator = ClinicalIntegrator(site_id="SITE-B")
        config = IntegrationConfig(config_id="CFG-002")
        integrator.register_system(config)
        status = integrator.get_connection_status()
        assert "CFG-002" in status

    def test_connect_known_config(self):
        """Connecting a known config returns CONNECTED."""
        integrator, config = self._make_integrator_with_config()
        status = integrator.connect("CFG-001")
        assert status == IntegrationStatus.CONNECTED

    def test_connect_unknown_config(self):
        """Connecting an unknown config returns ERROR."""
        integrator = ClinicalIntegrator(site_id="SITE-C")
        status = integrator.connect("UNKNOWN")
        assert status == IntegrationStatus.ERROR

    def test_ingest_clinical_data(self):
        """Ingesting JSON data increments ingestion count."""
        integrator, config = self._make_integrator_with_config()
        integrator.connect("CFG-001")
        raw = [{"patient_id": "P1", "value": 5.0}, {"patient_id": "P2", "value": 6.0}]
        result = integrator.ingest_clinical_data("CFG-001", raw)
        assert result.records_received == 2
        assert result.records_accepted == 2
        assert integrator.ingestion_count == 2

    def test_ingest_unknown_config_returns_error(self):
        """Ingesting with unknown config returns ERROR result."""
        integrator = ClinicalIntegrator(site_id="SITE-D")
        result = integrator.ingest_clinical_data("UNKNOWN", [{"val": 1}])
        assert result.status == IntegrationStatus.ERROR

    def test_export_results_json(self):
        """Exporting results as JSON returns expected keys."""
        integrator, _ = self._make_integrator_with_config()
        export = integrator.export_results({"key": "value"}, target_format=DataFormat.JSON)
        assert "export_id" in export
        assert export["format"] == "json"
        assert export["size_bytes"] > 0


class TestFederatedClinicalBridge:
    """Tests for FederatedClinicalBridge."""

    def test_creation(self):
        """Bridge is created with correct defaults."""
        bridge = FederatedClinicalBridge(site_id="SITE-X")
        assert bridge.site_id == "SITE-X"
        assert bridge.global_model_version == 0
        assert bridge.submitted_count == 0

    def test_prepare_federated_payload(self):
        """Preparing payload returns valid structure."""
        bridge = FederatedClinicalBridge(site_id="SITE-Y")
        vectors = [np.random.rand(10) for _ in range(5)]
        payload = bridge.prepare_federated_payload(vectors)
        assert payload["site_id"] == "SITE-Y"
        assert payload["n_samples"] == 5
        assert payload["n_features"] == 10

    def test_prepare_empty_payload_returns_error(self):
        """Empty feature vectors returns error payload."""
        bridge = FederatedClinicalBridge(site_id="SITE-Z")
        payload = bridge.prepare_federated_payload([])
        assert "error" in payload

    def test_submit_increments_count(self):
        """Submitting a payload increments submitted_count."""
        bridge = FederatedClinicalBridge(site_id="SITE-W")
        payload = {"payload_id": "FED-001"}
        bridge.submit_to_federation(payload)
        assert bridge.submitted_count == 1

    def test_receive_global_update_increments_version(self):
        """Receiving a global update increases model version."""
        bridge = FederatedClinicalBridge(site_id="SITE-V")
        result = bridge.receive_global_update(model_version=5)
        assert bridge.global_model_version >= 5
        assert result["previous_version"] == 0
