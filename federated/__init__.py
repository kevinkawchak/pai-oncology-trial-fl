"""Federated learning framework for physical AI oncology trials.

RESEARCH USE ONLY — Not intended for clinical deployment without validation.
"""

from __future__ import annotations

import importlib

__version__ = "1.1.1"

__all__ = [
    "AggregationStrategy",
    "DataHarmonizer",
    "DataPartitioner",
    "DifferentialPrivacy",
    "FederatedClient",
    "FederatedModel",
    "FederationCoordinator",
    "ModelConfig",
    "SecureAggregator",
    "SiteEnrollmentManager",
    "generate_synthetic_oncology_data",
]

# Lazy imports to avoid circular dependency when submodules are loaded
# individually via importlib.util.spec_from_file_location (e.g. in tests).
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FederatedClient": ("federated.client", "FederatedClient"),
    "AggregationStrategy": ("federated.coordinator", "AggregationStrategy"),
    "FederationCoordinator": ("federated.coordinator", "FederationCoordinator"),
    "DataHarmonizer": ("federated.data_harmonization", "DataHarmonizer"),
    "DataPartitioner": ("federated.data_ingestion", "DataPartitioner"),
    "generate_synthetic_oncology_data": ("federated.data_ingestion", "generate_synthetic_oncology_data"),
    "DifferentialPrivacy": ("federated.differential_privacy", "DifferentialPrivacy"),
    "FederatedModel": ("federated.model", "FederatedModel"),
    "ModelConfig": ("federated.model", "ModelConfig"),
    "SecureAggregator": ("federated.secure_aggregation", "SecureAggregator"),
    "SiteEnrollmentManager": ("federated.site_enrollment", "SiteEnrollmentManager"),
}


def __getattr__(name: str):  # noqa: ANN001
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
