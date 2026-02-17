"""Federated learning framework for physical AI oncology trials."""

__version__ = "0.2.0"

from federated.client import FederatedClient
from federated.coordinator import AggregationStrategy, FederationCoordinator
from federated.data_harmonization import DataHarmonizer
from federated.data_ingestion import DataPartitioner, generate_synthetic_oncology_data
from federated.differential_privacy import DifferentialPrivacy
from federated.model import FederatedModel, ModelConfig
from federated.secure_aggregation import SecureAggregator
from federated.site_enrollment import SiteEnrollmentManager

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
