"""Data ingestion utilities for federated oncology trials.

Provides functions to generate synthetic oncology datasets and
partition them across multiple simulated hospital sites using
IID or non-IID strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SiteData:
    """Data partition assigned to a single federated site."""

    site_id: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def num_train(self) -> int:
        return len(self.x_train)

    @property
    def num_test(self) -> int:
        return len(self.x_test)


def generate_synthetic_oncology_data(
    n_samples: int = 1000,
    n_features: int = 30,
    n_classes: int = 2,
    class_balance: list[float] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic oncology feature data for classification.

    Creates realistic-looking clinical feature vectors (e.g. tumor
    measurements, biomarker levels, patient demographics) drawn from
    class-conditional Gaussian distributions.

    Args:
        n_samples: Total number of patient records.
        n_features: Number of clinical features per record.
        n_classes: Number of diagnostic classes.
        class_balance: Proportion of samples per class.
            Defaults to uniform.
        seed: Random seed.

    Returns:
        Tuple of (X, y) where X has shape (n_samples, n_features) and
        y has shape (n_samples,) with integer class labels.
    """
    rng = np.random.default_rng(seed)

    if class_balance is None:
        class_balance = [1.0 / n_classes] * n_classes
    class_balance_arr = np.array(class_balance)
    class_balance_arr = class_balance_arr / class_balance_arr.sum()

    samples_per_class = (class_balance_arr * n_samples).astype(int)
    # Adjust rounding
    samples_per_class[-1] = n_samples - samples_per_class[:-1].sum()

    all_x = []
    all_y = []
    for cls in range(n_classes):
        n = samples_per_class[cls]
        # Each class has a shifted mean to make them separable
        mean = rng.uniform(-1, 1, size=n_features) * (cls + 1) * 0.5
        cov_diag = rng.uniform(0.5, 1.5, size=n_features)
        x = rng.normal(loc=mean, scale=np.sqrt(cov_diag), size=(n, n_features))
        all_x.append(x)
        all_y.append(np.full(n, cls, dtype=int))

    x = np.vstack(all_x)
    y = np.concatenate(all_y)

    # Shuffle
    perm = rng.permutation(n_samples)
    return x[perm], y[perm]


class DataPartitioner:
    """Split a dataset across multiple federated sites.

    Supports IID (uniform random) and non-IID (label-skewed)
    partitioning strategies.

    Args:
        num_sites: Number of hospital sites.
        strategy: ``"iid"`` or ``"non_iid"``.
        test_fraction: Fraction of each site's data reserved for testing.
        seed: Random seed.
    """

    def __init__(
        self,
        num_sites: int = 2,
        strategy: str = "iid",
        test_fraction: float = 0.2,
        seed: int = 42,
    ):
        if num_sites < 1:
            raise ValueError("num_sites must be >= 1")
        if strategy not in ("iid", "non_iid"):
            raise ValueError("strategy must be 'iid' or 'non_iid'")

        self.num_sites = num_sites
        self.strategy = strategy
        self.test_fraction = test_fraction
        self.seed = seed

    def partition(self, x: np.ndarray, y: np.ndarray) -> list[SiteData]:
        """Partition dataset across sites.

        Args:
            x: Feature matrix (n_samples, n_features).
            y: Label vector (n_samples,).

        Returns:
            List of SiteData, one per site.
        """
        if self.strategy == "iid":
            return self._partition_iid(x, y)
        return self._partition_non_iid(x, y)

    def _partition_iid(self, x: np.ndarray, y: np.ndarray) -> list[SiteData]:
        """Uniform random split."""
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(x))
        chunks = np.array_split(indices, self.num_sites)

        sites = []
        for i, chunk in enumerate(chunks):
            sx, sy = x[chunk], y[chunk]
            site = self._train_test_split(f"site_{i}", sx, sy)
            sites.append(site)
        return sites

    def _partition_non_iid(self, x: np.ndarray, y: np.ndarray) -> list[SiteData]:
        """Label-skewed split — each site gets a disproportionate share of certain classes."""
        rng = np.random.default_rng(self.seed)
        classes = np.unique(y)

        # Sort by label, then assign shards to sites
        sorted_idx = np.argsort(y)
        n_shards = self.num_sites * 2
        shards = np.array_split(sorted_idx, n_shards)
        rng.shuffle(shards)

        site_indices: list[list[int]] = [[] for _ in range(self.num_sites)]
        for shard_i, shard in enumerate(shards):
            site_i = shard_i % self.num_sites
            site_indices[site_i].extend(shard.tolist())

        sites = []
        for i in range(self.num_sites):
            idx = np.array(site_indices[i])
            rng.shuffle(idx)
            site = self._train_test_split(f"site_{i}", x[idx], y[idx])
            sites.append(site)

        logger.info(
            "Non-IID partition: %d sites, classes=%s",
            self.num_sites,
            classes.tolist(),
        )
        return sites

    def _train_test_split(self, site_id: str, x: np.ndarray, y: np.ndarray) -> SiteData:
        """Split a site's data into train/test."""
        n = len(x)
        n_test = max(1, int(n * self.test_fraction))
        return SiteData(
            site_id=site_id,
            x_train=x[n_test:],
            y_train=y[n_test:],
            x_test=x[:n_test],
            y_test=y[:n_test],
        )
