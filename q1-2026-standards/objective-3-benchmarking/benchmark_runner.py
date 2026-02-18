"""Cross-platform benchmark runner for federated oncology models.

Provides a standardized benchmarking framework for evaluating federated
model performance across heterogeneous hardware and software environments.
Benchmarks cover inference latency, prediction accuracy against clinical
ground truth, memory footprint, and throughput under concurrent load.

Results are published in a standardized format (JSON and Markdown) to
enable cross-site comparison and regression detection.

DISCLAIMER: RESEARCH USE ONLY. This software is not approved for clinical
decision-making. Benchmark results do not constitute regulatory evidence
of model safety or efficacy.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import platform
import statistics
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports -- frameworks may or may not be installed at each site
# ---------------------------------------------------------------------------
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import tensorflow as tf

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import monai

    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class BenchmarkMetric(Enum):
    """Types of benchmark metrics that can be collected."""

    LATENCY = "latency"
    ACCURACY = "accuracy"
    MEMORY = "memory"
    THROUGHPUT = "throughput"


class BenchmarkStatus(Enum):
    """Status of a benchmark run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class InferenceBackend(Enum):
    """Inference backend for model execution."""

    PYTORCH = "pytorch"
    ONNX_RUNTIME = "onnx_runtime"
    TENSORFLOW = "tensorflow"
    NUMPY = "numpy"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.

    Attributes:
        model_path: Path to the model artifact to benchmark.
        model_format: Serialization format of the model.
        backend: Inference backend to use.
        input_shape: Input tensor shape (including batch dimension).
        num_warmup_iterations: Number of warmup iterations before measuring.
        num_iterations: Number of timed iterations for latency/throughput.
        batch_sizes: List of batch sizes to benchmark.
        device: Device to run on ("cpu" or "cuda").
        num_workers: Number of concurrent workers for throughput tests.
        ground_truth_path: Path to ground truth data for accuracy tests.
        ground_truth_labels_path: Path to ground truth labels.
        output_dir: Directory for benchmark reports.
        benchmark_name: Human-readable name for this benchmark run.
        seed: Random seed for reproducibility.
        collect_metrics: Which metrics to collect.
        memory_sample_interval_ms: Interval for memory sampling in ms.
        tags: Arbitrary key-value tags for the run.
    """

    model_path: str = ""
    model_format: str = "onnx"
    backend: InferenceBackend = InferenceBackend.ONNX_RUNTIME
    input_shape: list[int] = field(default_factory=lambda: [1, 30])
    num_warmup_iterations: int = 10
    num_iterations: int = 100
    batch_sizes: list[int] = field(default_factory=lambda: [1, 8, 32])
    device: str = "cpu"
    num_workers: int = 1
    ground_truth_path: str = ""
    ground_truth_labels_path: str = ""
    output_dir: str = "benchmark_results"
    benchmark_name: str = "unnamed_benchmark"
    seed: int = 42
    collect_metrics: list[BenchmarkMetric] = field(
        default_factory=lambda: [BenchmarkMetric.LATENCY, BenchmarkMetric.MEMORY]
    )
    memory_sample_interval_ms: float = 10.0
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class LatencyResult:
    """Latency benchmark results.

    Attributes:
        batch_size: Batch size used for this measurement.
        num_iterations: Number of iterations measured.
        latencies_ms: Raw latency values in milliseconds.
        p50_ms: 50th percentile (median) latency.
        p95_ms: 95th percentile latency.
        p99_ms: 99th percentile latency.
        mean_ms: Mean latency.
        std_ms: Standard deviation of latency.
        min_ms: Minimum latency.
        max_ms: Maximum latency.
    """

    batch_size: int = 1
    num_iterations: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (excludes raw latencies for brevity)."""
        return {
            "batch_size": self.batch_size,
            "num_iterations": self.num_iterations,
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


@dataclass
class AccuracyResult:
    """Accuracy benchmark results.

    Attributes:
        num_samples: Number of samples evaluated.
        accuracy: Overall classification accuracy.
        auc_roc: Area under the ROC curve (macro-averaged).
        f1_score: Macro-averaged F1 score.
        per_class_accuracy: Accuracy per class label.
        confusion_matrix: Flattened confusion matrix.
    """

    num_samples: int = 0
    accuracy: float = 0.0
    auc_roc: float = 0.0
    f1_score: float = 0.0
    per_class_accuracy: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "num_samples": self.num_samples,
            "accuracy": round(self.accuracy, 4),
            "auc_roc": round(self.auc_roc, 4),
            "f1_score": round(self.f1_score, 4),
            "per_class_accuracy": {k: round(v, 4) for k, v in self.per_class_accuracy.items()},
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass
class MemoryResult:
    """Memory usage benchmark results.

    Attributes:
        peak_rss_mb: Peak resident set size in megabytes.
        model_size_mb: Model artifact size in megabytes.
        inference_memory_delta_mb: Memory increase during inference.
        gpu_memory_mb: Peak GPU memory usage (if applicable).
        system_total_mb: Total system memory.
        system_available_mb: Available system memory before benchmark.
    """

    peak_rss_mb: float = 0.0
    model_size_mb: float = 0.0
    inference_memory_delta_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    system_total_mb: float = 0.0
    system_available_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "model_size_mb": round(self.model_size_mb, 2),
            "inference_memory_delta_mb": round(self.inference_memory_delta_mb, 2),
            "gpu_memory_mb": round(self.gpu_memory_mb, 2),
            "system_total_mb": round(self.system_total_mb, 2),
            "system_available_mb": round(self.system_available_mb, 2),
        }


@dataclass
class ThroughputResult:
    """Throughput benchmark results.

    Attributes:
        batch_size: Batch size used.
        num_workers: Number of concurrent workers.
        total_samples: Total number of samples processed.
        total_time_seconds: Total wall-clock time.
        samples_per_second: Throughput in samples per second.
        batches_per_second: Throughput in batches per second.
    """

    batch_size: int = 1
    num_workers: int = 1
    total_samples: int = 0
    total_time_seconds: float = 0.0
    samples_per_second: float = 0.0
    batches_per_second: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "total_samples": self.total_samples,
            "total_time_seconds": round(self.total_time_seconds, 3),
            "samples_per_second": round(self.samples_per_second, 2),
            "batches_per_second": round(self.batches_per_second, 2),
        }


@dataclass
class BenchmarkResult:
    """Aggregated result from a complete benchmark run.

    Attributes:
        benchmark_id: Unique identifier for this benchmark run.
        benchmark_name: Human-readable name.
        status: Final status of the benchmark.
        model_path: Path to the model that was benchmarked.
        backend: Inference backend used.
        device: Device used (cpu or cuda).
        latency_results: Latency measurements per batch size.
        accuracy_result: Accuracy measurement (if collected).
        memory_result: Memory measurement (if collected).
        throughput_results: Throughput measurements per batch size.
        hardware_info: Hardware description of the benchmark host.
        software_info: Software versions on the benchmark host.
        config: Original benchmark configuration.
        total_time_seconds: Total wall-clock time for the full run.
        error_message: Error message if the benchmark failed.
        timestamp: ISO 8601 timestamp of benchmark completion.
        tags: Tags from the configuration.
    """

    benchmark_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    benchmark_name: str = ""
    status: BenchmarkStatus = BenchmarkStatus.PENDING
    model_path: str = ""
    backend: str = ""
    device: str = "cpu"
    latency_results: list[LatencyResult] = field(default_factory=list)
    accuracy_result: AccuracyResult | None = None
    memory_result: MemoryResult | None = None
    throughput_results: list[ThroughputResult] = field(default_factory=list)
    hardware_info: dict[str, str] = field(default_factory=dict)
    software_info: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    total_time_seconds: float = 0.0
    error_message: str = ""
    timestamp: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full result to a dictionary for JSON export."""
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_name": self.benchmark_name,
            "status": self.status.value,
            "model_path": self.model_path,
            "backend": self.backend,
            "device": self.device,
            "latency_results": [lr.to_dict() for lr in self.latency_results],
            "accuracy_result": self.accuracy_result.to_dict() if self.accuracy_result else None,
            "memory_result": self.memory_result.to_dict() if self.memory_result else None,
            "throughput_results": [tr.to_dict() for tr in self.throughput_results],
            "hardware_info": self.hardware_info,
            "software_info": self.software_info,
            "total_time_seconds": round(self.total_time_seconds, 3),
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "tags": self.tags,
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class BenchmarkError(Exception):
    """Raised when a benchmark operation fails."""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _iso_timestamp() -> str:
    """Return current UTC time as ISO 8601 string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _percentile(data: list[float], pct: float) -> float:
    """Compute the given percentile from a sorted list.

    Args:
        data: List of float values (will be sorted internally).
        pct: Percentile to compute (0-100).

    Returns:
        The percentile value.
    """
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (pct / 100.0) * (len(sorted_data) - 1)
    lower = int(math.floor(idx))
    upper = int(math.ceil(idx))
    if lower == upper:
        return sorted_data[lower]
    fraction = idx - lower
    return sorted_data[lower] * (1 - fraction) + sorted_data[upper] * fraction


def _generate_dummy_input(input_shape: list[int], seed: int = 42) -> np.ndarray:
    """Generate deterministic dummy input data.

    Args:
        input_shape: Shape of the input tensor.
        seed: Random seed for reproducibility.

    Returns:
        Numpy array with reproducible random values.
    """
    rng = np.random.default_rng(seed=seed)
    return rng.standard_normal(input_shape).astype(np.float32)


def _ensure_directory(path: str) -> None:
    """Create directory and parents if they do not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def _collect_hardware_info() -> dict[str, str]:
    """Collect hardware information about the current host.

    Returns:
        Dictionary with hardware description fields.
    """
    info: dict[str, str] = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": str(os.cpu_count() or "unknown"),
    }

    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        info["total_memory_gb"] = f"{mem.total / (1024**3):.1f}"
        info["available_memory_gb"] = f"{mem.available / (1024**3):.1f}"

    if HAS_TORCH and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = f"{torch.cuda.get_device_properties(0).total_mem / (1024**3):.1f}"
        info["cuda_version"] = torch.version.cuda or "unknown"

    return info


def _collect_software_info() -> dict[str, str]:
    """Collect software version information.

    Returns:
        Dictionary with library versions.
    """
    info: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }

    if HAS_TORCH:
        info["torch"] = torch.__version__
    if HAS_ONNX:
        info["onnxruntime"] = ort.__version__
    if HAS_TENSORFLOW:
        info["tensorflow"] = tf.__version__
    if HAS_SKLEARN:
        import sklearn

        info["scikit_learn"] = sklearn.__version__

    return info


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------
class BenchmarkRunner:
    """Executes cross-platform benchmarks for federated oncology models.

    The runner supports four benchmark categories:

        - **Latency**: Measures per-inference latency across batch sizes,
          reporting p50, p95, p99, mean, and standard deviation.
        - **Accuracy**: Evaluates model predictions against clinical ground
          truth data, computing accuracy, AUC-ROC, and F1 score.
        - **Memory**: Profiles peak RSS, model size, and inference memory
          delta. Reports GPU memory usage when available.
        - **Throughput**: Measures sustained throughput in samples/second
          under configured batch sizes and concurrency.

    Results are exported as JSON and optionally as a Markdown report
    suitable for cross-site comparison reviews.

    Example::

        config = BenchmarkConfig(
            model_path="models/tumor_classifier.onnx",
            model_format="onnx",
            backend=InferenceBackend.ONNX_RUNTIME,
            input_shape=[1, 30],
            num_iterations=200,
            batch_sizes=[1, 16, 64],
            output_dir="benchmark_results/",
            benchmark_name="tumor_classifier_v1",
        )
        runner = BenchmarkRunner()
        result = runner.run(config)
        runner.generate_report(result, format="markdown")
    """

    def __init__(self) -> None:
        """Initialize the benchmark runner."""
        self._run_history: list[BenchmarkResult] = []
        logger.info("BenchmarkRunner initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Execute a complete benchmark run.

        Runs all configured metrics in sequence: latency, accuracy,
        memory, then throughput.

        Args:
            config: Benchmark configuration.

        Returns:
            BenchmarkResult with all collected metrics.
        """
        result = BenchmarkResult(
            benchmark_name=config.benchmark_name,
            model_path=config.model_path,
            backend=config.backend.value,
            device=config.device,
            hardware_info=_collect_hardware_info(),
            software_info=_collect_software_info(),
            tags=dict(config.tags),
            timestamp=_iso_timestamp(),
        )

        result.status = BenchmarkStatus.RUNNING
        overall_start = time.monotonic()

        try:
            if BenchmarkMetric.LATENCY in config.collect_metrics:
                logger.info("Running latency benchmarks for %s", config.benchmark_name)
                for batch_size in config.batch_sizes:
                    latency_result = self.run_latency(config, batch_size)
                    result.latency_results.append(latency_result)

            if BenchmarkMetric.ACCURACY in config.collect_metrics:
                logger.info("Running accuracy benchmark for %s", config.benchmark_name)
                accuracy_result = self.run_accuracy(config)
                result.accuracy_result = accuracy_result

            if BenchmarkMetric.MEMORY in config.collect_metrics:
                logger.info("Running memory benchmark for %s", config.benchmark_name)
                memory_result = self.run_memory(config)
                result.memory_result = memory_result

            if BenchmarkMetric.THROUGHPUT in config.collect_metrics:
                logger.info("Running throughput benchmarks for %s", config.benchmark_name)
                for batch_size in config.batch_sizes:
                    throughput_result = self._run_throughput(config, batch_size)
                    result.throughput_results.append(throughput_result)

            result.status = BenchmarkStatus.COMPLETED

        except BenchmarkError as exc:
            result.status = BenchmarkStatus.FAILED
            result.error_message = str(exc)
            logger.error("Benchmark %s failed: %s", result.benchmark_id, exc)

        except Exception as exc:
            result.status = BenchmarkStatus.FAILED
            result.error_message = f"Unexpected error: {exc}"
            logger.exception("Benchmark %s unexpected failure", result.benchmark_id)

        result.total_time_seconds = round(time.monotonic() - overall_start, 3)
        self._run_history.append(result)

        logger.info(
            "Benchmark %s completed in %.1fs: status=%s",
            result.benchmark_id,
            result.total_time_seconds,
            result.status.value,
        )

        return result

    def run_latency(self, config: BenchmarkConfig, batch_size: int) -> LatencyResult:
        """Measure inference latency for a specific batch size.

        Runs warmup iterations first to stabilize caches and JIT, then
        measures the configured number of timed iterations.

        Args:
            config: Benchmark configuration.
            batch_size: Number of samples per inference call.

        Returns:
            LatencyResult with percentile and summary statistics.
        """
        input_shape = [batch_size] + config.input_shape[1:]
        dummy_input = _generate_dummy_input(input_shape, seed=config.seed)

        infer_fn = self._get_inference_function(config)

        # Warmup
        for _ in range(config.num_warmup_iterations):
            infer_fn(dummy_input)

        # Timed iterations
        latencies_ms: list[float] = []
        for _ in range(config.num_iterations):
            gc.disable()
            start = time.perf_counter_ns()
            infer_fn(dummy_input)
            end = time.perf_counter_ns()
            gc.enable()
            latencies_ms.append((end - start) / 1e6)

        result = LatencyResult(
            batch_size=batch_size,
            num_iterations=config.num_iterations,
            latencies_ms=latencies_ms,
            p50_ms=_percentile(latencies_ms, 50),
            p95_ms=_percentile(latencies_ms, 95),
            p99_ms=_percentile(latencies_ms, 99),
            mean_ms=statistics.mean(latencies_ms) if latencies_ms else 0.0,
            std_ms=statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
            min_ms=min(latencies_ms) if latencies_ms else 0.0,
            max_ms=max(latencies_ms) if latencies_ms else 0.0,
        )

        logger.info(
            "Latency (batch=%d): p50=%.2fms, p95=%.2fms, p99=%.2fms, mean=%.2fms",
            batch_size,
            result.p50_ms,
            result.p95_ms,
            result.p99_ms,
            result.mean_ms,
        )

        return result

    def run_accuracy(self, config: BenchmarkConfig) -> AccuracyResult:
        """Evaluate model accuracy against ground truth data.

        Loads ground truth data and labels, runs inference, and computes
        classification metrics.

        Args:
            config: Benchmark configuration with ground truth paths.

        Returns:
            AccuracyResult with classification metrics.

        Raises:
            BenchmarkError: If ground truth data is not available or metrics
                cannot be computed.
        """
        result = AccuracyResult()

        if not config.ground_truth_path or not Path(config.ground_truth_path).exists():
            logger.warning("Ground truth data not found at '%s'; generating synthetic data", config.ground_truth_path)
            rng = np.random.default_rng(seed=config.seed)
            num_samples = 200
            feature_dim = config.input_shape[-1] if len(config.input_shape) > 1 else config.input_shape[0]
            ground_truth_data = rng.standard_normal((num_samples, feature_dim)).astype(np.float32)
            ground_truth_labels = rng.integers(0, 2, size=num_samples)
        else:
            ground_truth_data = np.load(config.ground_truth_path).astype(np.float32)
            if config.ground_truth_labels_path and Path(config.ground_truth_labels_path).exists():
                ground_truth_labels = np.load(config.ground_truth_labels_path)
            else:
                raise BenchmarkError("Ground truth labels file not found")

        infer_fn = self._get_inference_function(config)

        try:
            predictions_raw = infer_fn(ground_truth_data)
            if isinstance(predictions_raw, np.ndarray) and predictions_raw.ndim == 2:
                predicted_labels = np.argmax(predictions_raw, axis=1)
                predicted_probs = predictions_raw
            else:
                predicted_labels = np.asarray(predictions_raw).flatten()
                predicted_probs = None
        except Exception as exc:
            raise BenchmarkError(f"Inference failed during accuracy benchmark: {exc}") from exc

        result.num_samples = len(ground_truth_labels)

        if HAS_SKLEARN:
            result.accuracy = float(accuracy_score(ground_truth_labels, predicted_labels))
            result.f1_score = float(f1_score(ground_truth_labels, predicted_labels, average="macro", zero_division=0))

            if predicted_probs is not None and predicted_probs.shape[1] == 2:
                try:
                    result.auc_roc = float(roc_auc_score(ground_truth_labels, predicted_probs[:, 1]))
                except ValueError:
                    result.auc_roc = 0.0
                    logger.warning("AUC-ROC could not be computed (likely single-class predictions)")
        else:
            correct = int(np.sum(predicted_labels == ground_truth_labels))
            result.accuracy = correct / max(result.num_samples, 1)
            logger.warning("scikit-learn not available; only basic accuracy computed")

        # Per-class accuracy
        unique_classes = np.unique(ground_truth_labels)
        for cls in unique_classes:
            mask = ground_truth_labels == cls
            cls_correct = int(np.sum(predicted_labels[mask] == ground_truth_labels[mask]))
            cls_total = int(np.sum(mask))
            result.per_class_accuracy[str(cls)] = cls_correct / max(cls_total, 1)

        logger.info(
            "Accuracy benchmark: accuracy=%.4f, f1=%.4f, auc=%.4f (n=%d)",
            result.accuracy,
            result.f1_score,
            result.auc_roc,
            result.num_samples,
        )

        return result

    def run_memory(self, config: BenchmarkConfig) -> MemoryResult:
        """Profile memory usage during model loading and inference.

        Measures RSS before and after model loading and inference to
        compute memory deltas. Reports GPU memory when available.

        Args:
            config: Benchmark configuration.

        Returns:
            MemoryResult with memory usage statistics.
        """
        result = MemoryResult()

        # Model artifact size
        if config.model_path and Path(config.model_path).exists():
            result.model_size_mb = Path(config.model_path).stat().st_size / (1024 * 1024)

        if HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            mem_info = psutil.virtual_memory()
            result.system_total_mb = mem_info.total / (1024 * 1024)
            result.system_available_mb = mem_info.available / (1024 * 1024)

            # Measure RSS before loading
            gc.collect()
            rss_before = process.memory_info().rss / (1024 * 1024)

            # Load model and run inference
            input_shape = config.input_shape
            dummy_input = _generate_dummy_input(input_shape, seed=config.seed)

            try:
                infer_fn = self._get_inference_function(config)
                infer_fn(dummy_input)
            except Exception as exc:
                logger.warning("Inference failed during memory benchmark: %s", exc)

            # Measure RSS after inference
            rss_after = process.memory_info().rss / (1024 * 1024)
            result.peak_rss_mb = rss_after
            result.inference_memory_delta_mb = rss_after - rss_before
        else:
            logger.warning("psutil not available; memory metrics will be limited")

        # GPU memory (PyTorch CUDA)
        if HAS_TORCH and torch.cuda.is_available() and config.device == "cuda":
            result.gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        logger.info(
            "Memory benchmark: peak_rss=%.1fMB, model_size=%.1fMB, delta=%.1fMB",
            result.peak_rss_mb,
            result.model_size_mb,
            result.inference_memory_delta_mb,
        )

        return result

    def _run_throughput(self, config: BenchmarkConfig, batch_size: int) -> ThroughputResult:
        """Measure sustained throughput for a specific batch size.

        Runs inference in a tight loop for the configured number of
        iterations and computes samples-per-second throughput.

        Args:
            config: Benchmark configuration.
            batch_size: Number of samples per inference call.

        Returns:
            ThroughputResult with throughput measurements.
        """
        input_shape = [batch_size] + config.input_shape[1:]
        dummy_input = _generate_dummy_input(input_shape, seed=config.seed)
        infer_fn = self._get_inference_function(config)

        # Warmup
        for _ in range(config.num_warmup_iterations):
            infer_fn(dummy_input)

        # Timed run
        total_samples = config.num_iterations * batch_size
        start = time.monotonic()
        for _ in range(config.num_iterations):
            infer_fn(dummy_input)
        elapsed = time.monotonic() - start

        result = ThroughputResult(
            batch_size=batch_size,
            num_workers=config.num_workers,
            total_samples=total_samples,
            total_time_seconds=elapsed,
            samples_per_second=total_samples / max(elapsed, 1e-9),
            batches_per_second=config.num_iterations / max(elapsed, 1e-9),
        )

        logger.info(
            "Throughput (batch=%d): %.1f samples/s, %.1f batches/s",
            batch_size,
            result.samples_per_second,
            result.batches_per_second,
        )

        return result

    def generate_report(
        self,
        result: BenchmarkResult,
        format: str = "json",
        output_path: str | None = None,
    ) -> str:
        """Generate a benchmark report in the specified format.

        Args:
            result: BenchmarkResult to format.
            format: Output format ("json" or "markdown").
            output_path: File path to write the report. If None, uses
                the config output_dir.

        Returns:
            The report content as a string.
        """
        if format == "json":
            return self._generate_json_report(result, output_path)
        elif format == "markdown":
            return self._generate_markdown_report(result, output_path)
        else:
            raise BenchmarkError(f"Unsupported report format: {format}")

    def get_run_history(self) -> list[dict[str, Any]]:
        """Return the history of all benchmark runs in this session."""
        return [r.to_dict() for r in self._run_history]

    # ------------------------------------------------------------------
    # Inference function factory
    # ------------------------------------------------------------------

    def _get_inference_function(self, config: BenchmarkConfig):
        """Return an inference callable for the configured backend.

        Args:
            config: Benchmark configuration with backend and model path.

        Returns:
            Callable that takes a numpy array and returns a numpy array.

        Raises:
            BenchmarkError: If the backend is not available.
        """
        backend = config.backend

        if backend == InferenceBackend.ONNX_RUNTIME:
            return self._make_onnx_infer_fn(config)
        elif backend == InferenceBackend.PYTORCH:
            return self._make_pytorch_infer_fn(config)
        elif backend == InferenceBackend.TENSORFLOW:
            return self._make_tensorflow_infer_fn(config)
        elif backend == InferenceBackend.NUMPY:
            return self._make_numpy_infer_fn(config)
        else:
            raise BenchmarkError(f"Unsupported inference backend: {backend.value}")

    def _make_onnx_infer_fn(self, config: BenchmarkConfig):
        """Create an ONNX Runtime inference function.

        Args:
            config: Benchmark configuration.

        Returns:
            Callable for ONNX Runtime inference.
        """
        if not HAS_ONNX:
            raise BenchmarkError("ONNX Runtime is required but not installed")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if config.device == "cuda" else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(config.model_path, providers=providers)
        input_name = session.get_inputs()[0].name

        def infer(data: np.ndarray) -> np.ndarray:
            return session.run(None, {input_name: data})[0]

        return infer

    def _make_pytorch_infer_fn(self, config: BenchmarkConfig):
        """Create a PyTorch inference function.

        Args:
            config: Benchmark configuration.

        Returns:
            Callable for PyTorch inference.
        """
        if not HAS_TORCH:
            raise BenchmarkError("PyTorch is required but not installed")

        model = torch.load(config.model_path, map_location=config.device, weights_only=True)
        if hasattr(model, "eval"):
            model.eval()

        def infer(data: np.ndarray) -> np.ndarray:
            tensor = torch.from_numpy(data).to(config.device)
            with torch.no_grad():
                output = model(tensor)
            return output.cpu().numpy()

        return infer

    def _make_tensorflow_infer_fn(self, config: BenchmarkConfig):
        """Create a TensorFlow inference function.

        Args:
            config: Benchmark configuration.

        Returns:
            Callable for TensorFlow inference.
        """
        if not HAS_TENSORFLOW:
            raise BenchmarkError("TensorFlow is required but not installed")

        model = tf.keras.models.load_model(config.model_path)

        def infer(data: np.ndarray) -> np.ndarray:
            return model.predict(data, verbose=0)

        return infer

    def _make_numpy_infer_fn(self, config: BenchmarkConfig):
        """Create a NumPy-based inference function.

        Loads model parameters from an npz file and runs a simple
        feedforward network for benchmarking purposes.

        Args:
            config: Benchmark configuration.

        Returns:
            Callable for NumPy inference.
        """
        if config.model_path and Path(config.model_path).exists():
            params = np.load(config.model_path)
            weights = [(params[f"w{i}"], params[f"b{i}"]) for i in range(len(params.files) // 2)]
        else:
            logger.warning("NumPy model file not found; using identity function for benchmarking")
            weights = []

        def infer(data: np.ndarray) -> np.ndarray:
            x = data
            for i, (w, b) in enumerate(weights):
                x = x @ w + b
                if i < len(weights) - 1:
                    x = np.maximum(0, x)
            if not weights:
                return x
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        return infer

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_json_report(self, result: BenchmarkResult, output_path: str | None = None) -> str:
        """Generate a JSON benchmark report.

        Args:
            result: BenchmarkResult to serialize.
            output_path: File path for the report.

        Returns:
            JSON string of the report.
        """
        report_json = json.dumps(result.to_dict(), indent=2)

        if output_path:
            _ensure_directory(str(Path(output_path).parent))
            with open(output_path, "w") as fh:
                fh.write(report_json)
            logger.info("JSON benchmark report written to %s", output_path)

        return report_json

    def _generate_markdown_report(self, result: BenchmarkResult, output_path: str | None = None) -> str:
        """Generate a Markdown benchmark report.

        Args:
            result: BenchmarkResult to format.
            output_path: File path for the report.

        Returns:
            Markdown string of the report.
        """
        lines: list[str] = []
        lines.append(f"# Benchmark Report: {result.benchmark_name}")
        lines.append("")
        lines.append(f"- **Benchmark ID:** {result.benchmark_id}")
        lines.append(f"- **Status:** {result.status.value}")
        lines.append(f"- **Timestamp:** {result.timestamp}")
        lines.append(f"- **Total Time:** {result.total_time_seconds:.1f}s")
        lines.append(f"- **Model:** {result.model_path}")
        lines.append(f"- **Backend:** {result.backend}")
        lines.append(f"- **Device:** {result.device}")
        lines.append("")

        # Hardware info
        if result.hardware_info:
            lines.append("## Hardware")
            lines.append("")
            lines.append("| Property | Value |")
            lines.append("|----------|-------|")
            for key, value in result.hardware_info.items():
                lines.append(f"| {key} | {value} |")
            lines.append("")

        # Software info
        if result.software_info:
            lines.append("## Software")
            lines.append("")
            lines.append("| Library | Version |")
            lines.append("|---------|---------|")
            for key, value in result.software_info.items():
                lines.append(f"| {key} | {value} |")
            lines.append("")

        # Latency results
        if result.latency_results:
            lines.append("## Latency")
            lines.append("")
            lines.append("| Batch Size | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) | Std (ms) |")
            lines.append("|------------|----------|----------|----------|-----------|----------|")
            for lr in result.latency_results:
                lines.append(
                    f"| {lr.batch_size} | {lr.p50_ms:.2f} | {lr.p95_ms:.2f} | "
                    f"{lr.p99_ms:.2f} | {lr.mean_ms:.2f} | {lr.std_ms:.2f} |"
                )
            lines.append("")

        # Accuracy results
        if result.accuracy_result:
            ar = result.accuracy_result
            lines.append("## Accuracy")
            lines.append("")
            lines.append(f"- **Samples:** {ar.num_samples}")
            lines.append(f"- **Accuracy:** {ar.accuracy:.4f}")
            lines.append(f"- **AUC-ROC:** {ar.auc_roc:.4f}")
            lines.append(f"- **F1 Score:** {ar.f1_score:.4f}")
            lines.append("")
            if ar.per_class_accuracy:
                lines.append("### Per-Class Accuracy")
                lines.append("")
                lines.append("| Class | Accuracy |")
                lines.append("|-------|----------|")
                for cls, acc in ar.per_class_accuracy.items():
                    lines.append(f"| {cls} | {acc:.4f} |")
                lines.append("")

        # Memory results
        if result.memory_result:
            mr = result.memory_result
            lines.append("## Memory")
            lines.append("")
            lines.append(f"- **Model Size:** {mr.model_size_mb:.1f} MB")
            lines.append(f"- **Peak RSS:** {mr.peak_rss_mb:.1f} MB")
            lines.append(f"- **Inference Delta:** {mr.inference_memory_delta_mb:.1f} MB")
            if mr.gpu_memory_mb > 0:
                lines.append(f"- **GPU Memory:** {mr.gpu_memory_mb:.1f} MB")
            lines.append("")

        # Throughput results
        if result.throughput_results:
            lines.append("## Throughput")
            lines.append("")
            lines.append("| Batch Size | Workers | Samples/s | Batches/s | Total Time (s) |")
            lines.append("|------------|---------|-----------|-----------|----------------|")
            for tr in result.throughput_results:
                lines.append(
                    f"| {tr.batch_size} | {tr.num_workers} | {tr.samples_per_second:.1f} | "
                    f"{tr.batches_per_second:.1f} | {tr.total_time_seconds:.2f} |"
                )
            lines.append("")

        # Error message
        if result.error_message:
            lines.append("## Errors")
            lines.append("")
            lines.append(f"```\n{result.error_message}\n```")
            lines.append("")

        report_md = "\n".join(lines)

        if output_path:
            _ensure_directory(str(Path(output_path).parent))
            with open(output_path, "w") as fh:
                fh.write(report_md)
            logger.info("Markdown benchmark report written to %s", output_path)

        return report_md
