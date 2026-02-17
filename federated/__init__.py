"""Federated learning framework for physical AI oncology trials."""

__version__ = "0.1.0"

from federated.client import FederatedClient
from federated.coordinator import FederationCoordinator
from federated.data_ingestion import DataPartitioner, generate_synthetic_oncology_data
from federated.differential_privacy import DifferentialPrivacy
from federated.model import FederatedModel, ModelConfig
from federated.secure_aggregation import SecureAggregator

__all__ = [
    "FederatedClient",
    "FederationCoordinator",
    "DataPartitioner",
    "DifferentialPrivacy",
    "FederatedModel",
    "ModelConfig",
    "SecureAggregator",
    "generate_synthetic_oncology_data",
]
