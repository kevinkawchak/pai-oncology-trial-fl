"""Tests for q1-2026-standards/objective-3-benchmarking/benchmark_runner.py.

Covers enums (BenchmarkMetric, BenchmarkStatus, InferenceBackend),
dataclasses (BenchmarkConfig, LatencyResult, AccuracyResult, MemoryResult,
ThroughputResult, BenchmarkResult), helper functions, and BenchmarkRunner.
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pytest

from tests.conftest import load_module

mod = load_module(
    "benchmark_runner",
    "q1-2026-standards/objective-3-benchmarking/benchmark_runner.py",
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------
class TestBenchmarkMetric:
    """Tests for the BenchmarkMetric enum."""

    def test_members(self):
        """BenchmarkMetric has four members."""
        expected = {"LATENCY", "ACCURACY", "MEMORY", "THROUGHPUT"}
        assert set(mod.BenchmarkMetric.__members__.keys()) == expected

    def test_values(self):
        """Values are lowercase."""
        assert mod.BenchmarkMetric.LATENCY.value == "latency"
        assert mod.BenchmarkMetric.THROUGHPUT.value == "throughput"


class TestBenchmarkStatus:
    """Tests for the BenchmarkStatus enum."""

    def test_members(self):
        """BenchmarkStatus has four members."""
        expected = {"PENDING", "RUNNING", "COMPLETED", "FAILED"}
        assert set(mod.BenchmarkStatus.__members__.keys()) == expected


class TestInferenceBackend:
    """Tests for the InferenceBackend enum."""

    def test_members(self):
        """InferenceBackend has four members."""
        expected = {"PYTORCH", "ONNX_RUNTIME", "TENSORFLOW", "NUMPY"}
        assert set(mod.InferenceBackend.__members__.keys()) == expected

    def test_values(self):
        """InferenceBackend values match expected strings."""
        assert mod.InferenceBackend.ONNX_RUNTIME.value == "onnx_runtime"
        assert mod.InferenceBackend.NUMPY.value == "numpy"


# ---------------------------------------------------------------------------
# BenchmarkConfig dataclass
# ---------------------------------------------------------------------------
class TestBenchmarkConfig:
    """Tests for the BenchmarkConfig dataclass."""

    def test_defaults(self):
        """Default config uses ONNX_RUNTIME backend."""
        cfg = mod.BenchmarkConfig()
        assert cfg.backend == mod.InferenceBackend.ONNX_RUNTIME
        assert cfg.num_warmup_iterations == 10
        assert cfg.num_iterations == 100
        assert cfg.device == "cpu"
        assert cfg.seed == 42

    def test_custom_values(self):
        """BenchmarkConfig accepts custom values."""
        cfg = mod.BenchmarkConfig(
            model_path="/models/test.onnx",
            benchmark_name="perf_test",
            num_iterations=500,
            batch_sizes=[1, 4, 16],
        )
        assert cfg.model_path == "/models/test.onnx"
        assert cfg.num_iterations == 500
        assert cfg.batch_sizes == [1, 4, 16]

    def test_default_metrics(self):
        """Default collect_metrics includes LATENCY and MEMORY."""
        cfg = mod.BenchmarkConfig()
        assert mod.BenchmarkMetric.LATENCY in cfg.collect_metrics
        assert mod.BenchmarkMetric.MEMORY in cfg.collect_metrics


# ---------------------------------------------------------------------------
# LatencyResult dataclass
# ---------------------------------------------------------------------------
class TestLatencyResult:
    """Tests for the LatencyResult dataclass."""

    def test_defaults(self):
        """Default LatencyResult has zero values."""
        lr = mod.LatencyResult()
        assert lr.batch_size == 1
        assert lr.p50_ms == 0.0
        assert lr.latencies_ms == []

    def test_to_dict(self):
        """to_dict excludes raw latencies."""
        lr = mod.LatencyResult(
            batch_size=8,
            p50_ms=1.5,
            p95_ms=2.5,
            p99_ms=3.5,
            mean_ms=1.8,
        )
        d = lr.to_dict()
        assert d["batch_size"] == 8
        assert d["p50_ms"] == 1.5
        assert "latencies_ms" not in d  # excluded for brevity


# ---------------------------------------------------------------------------
# AccuracyResult dataclass
# ---------------------------------------------------------------------------
class TestAccuracyResult:
    """Tests for the AccuracyResult dataclass."""

    def test_defaults(self):
        """Default AccuracyResult has zero accuracy."""
        ar = mod.AccuracyResult()
        assert ar.accuracy == 0.0
        assert ar.num_samples == 0

    def test_to_dict(self):
        """to_dict rounds values to four decimals."""
        ar = mod.AccuracyResult(accuracy=0.91234567, f1_score=0.88765432)
        d = ar.to_dict()
        assert d["accuracy"] == 0.9123
        assert d["f1_score"] == 0.8877


# ---------------------------------------------------------------------------
# MemoryResult dataclass
# ---------------------------------------------------------------------------
class TestMemoryResult:
    """Tests for the MemoryResult dataclass."""

    def test_defaults(self):
        """Default MemoryResult has zero values."""
        mr = mod.MemoryResult()
        assert mr.peak_rss_mb == 0.0
        assert mr.model_size_mb == 0.0

    def test_to_dict(self):
        """to_dict rounds to two decimals."""
        mr = mod.MemoryResult(peak_rss_mb=512.3456, model_size_mb=10.7891)
        d = mr.to_dict()
        assert d["peak_rss_mb"] == 512.35
        assert d["model_size_mb"] == 10.79


# ---------------------------------------------------------------------------
# ThroughputResult dataclass
# ---------------------------------------------------------------------------
class TestThroughputResult:
    """Tests for the ThroughputResult dataclass."""

    def test_defaults(self):
        """Default ThroughputResult has zero throughput."""
        tr = mod.ThroughputResult()
        assert tr.samples_per_second == 0.0

    def test_to_dict(self):
        """to_dict includes batch_size and workers."""
        tr = mod.ThroughputResult(
            batch_size=16,
            num_workers=4,
            samples_per_second=5000.123,
        )
        d = tr.to_dict()
        assert d["batch_size"] == 16
        assert d["num_workers"] == 4
        assert d["samples_per_second"] == 5000.12


# ---------------------------------------------------------------------------
# BenchmarkResult dataclass
# ---------------------------------------------------------------------------
class TestBenchmarkResult:
    """Tests for the BenchmarkResult dataclass."""

    def test_defaults(self):
        """Default result is PENDING."""
        br = mod.BenchmarkResult()
        assert br.status == mod.BenchmarkStatus.PENDING
        assert br.benchmark_id != ""

    def test_to_dict(self):
        """to_dict includes all fields."""
        br = mod.BenchmarkResult(
            benchmark_name="test_bench",
            status=mod.BenchmarkStatus.COMPLETED,
        )
        d = br.to_dict()
        assert d["benchmark_name"] == "test_bench"
        assert d["status"] == "completed"
        assert "latency_results" in d
        assert "hardware_info" in d

    def test_unique_ids(self):
        """Each BenchmarkResult gets a unique ID."""
        r1 = mod.BenchmarkResult()
        r2 = mod.BenchmarkResult()
        assert r1.benchmark_id != r2.benchmark_id


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
class TestPercentile:
    """Tests for the _percentile helper."""

    def test_median(self):
        """Percentile 50 of [1, 2, 3, 4, 5] is 3."""
        result = mod._percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50)
        assert result == pytest.approx(3.0)

    def test_p99(self):
        """Percentile 99 of range is close to max."""
        data = list(range(1, 101))
        result = mod._percentile([float(x) for x in data], 99)
        assert result > 98.0

    def test_empty_list(self):
        """Percentile of empty list is 0."""
        assert mod._percentile([], 50) == 0.0

    def test_single_element(self):
        """Percentile of single element returns that element."""
        assert mod._percentile([42.0], 99) == 42.0


class TestGenerateDummyInput:
    """Tests for the _generate_dummy_input helper."""

    def test_shape(self):
        """Output shape matches input_shape."""
        arr = mod._generate_dummy_input([4, 30])
        assert arr.shape == (4, 30)

    def test_deterministic(self):
        """Same seed produces same output."""
        a = mod._generate_dummy_input([1, 10], seed=42)
        b = mod._generate_dummy_input([1, 10], seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seed(self):
        """Different seeds produce different output."""
        a = mod._generate_dummy_input([1, 10], seed=42)
        b = mod._generate_dummy_input([1, 10], seed=99)
        assert not np.array_equal(a, b)

    def test_float32(self):
        """Output dtype is float32."""
        arr = mod._generate_dummy_input([1, 5])
        assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------
class TestBenchmarkRunner:
    """Tests for the BenchmarkRunner class."""

    def test_creation(self):
        """BenchmarkRunner can be created."""
        runner = mod.BenchmarkRunner()
        assert runner is not None

    def test_run_history_empty(self):
        """Run history starts empty."""
        runner = mod.BenchmarkRunner()
        assert runner.get_run_history() == []

    def test_run_numpy_backend(self):
        """Run with numpy backend completes without model file."""
        runner = mod.BenchmarkRunner()
        cfg = mod.BenchmarkConfig(
            backend=mod.InferenceBackend.NUMPY,
            model_path="",
            benchmark_name="numpy_test",
            num_iterations=5,
            num_warmup_iterations=1,
            batch_sizes=[1],
            collect_metrics=[mod.BenchmarkMetric.LATENCY],
        )
        result = runner.run(cfg)
        assert result.status == mod.BenchmarkStatus.COMPLETED
        assert len(result.latency_results) == 1
        assert result.latency_results[0].p50_ms > 0

    def test_run_populates_history(self):
        """Run adds to history."""
        runner = mod.BenchmarkRunner()
        cfg = mod.BenchmarkConfig(
            backend=mod.InferenceBackend.NUMPY,
            model_path="",
            num_iterations=2,
            num_warmup_iterations=0,
            batch_sizes=[1],
            collect_metrics=[mod.BenchmarkMetric.LATENCY],
        )
        runner.run(cfg)
        history = runner.get_run_history()
        assert len(history) == 1

    def test_generate_report_json(self):
        """generate_report produces valid JSON."""
        runner = mod.BenchmarkRunner()
        result = mod.BenchmarkResult(benchmark_name="report_test")
        report = runner.generate_report(result, format="json")
        parsed = json.loads(report)
        assert parsed["benchmark_name"] == "report_test"

    def test_generate_report_markdown(self):
        """generate_report produces Markdown with header."""
        runner = mod.BenchmarkRunner()
        result = mod.BenchmarkResult(benchmark_name="md_test")
        report = runner.generate_report(result, format="markdown")
        assert "# Benchmark Report" in report
        assert "md_test" in report

    def test_generate_report_unsupported(self):
        """Unsupported format raises BenchmarkError."""
        runner = mod.BenchmarkRunner()
        result = mod.BenchmarkResult()
        with pytest.raises(mod.BenchmarkError, match="Unsupported"):
            runner.generate_report(result, format="xml")
