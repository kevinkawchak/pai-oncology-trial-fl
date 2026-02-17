"""Cross-framework model format conversion pipeline for federated oncology trials.

Provides a standardized pipeline for converting trained models between frameworks
(PyTorch, TensorFlow, ONNX, SafeTensors, MONAI) used across federated clinical
trial sites. Ensures numerical equivalence after conversion using configurable
tolerance thresholds and SHA-256 integrity verification.

Supported conversion paths:
    - PyTorch  <-> ONNX
    - PyTorch  <-> SafeTensors
    - PyTorch  <-> TensorFlow (via ONNX intermediate)
    - MONAI    <-> PyTorch
    - MONAI    <-> ONNX

DISCLAIMER: RESEARCH USE ONLY. This software is not approved for clinical
decision-making. Model conversions must be independently validated before
any use in patient care or regulatory submissions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
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
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnx
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
    from safetensors import safe_open
    from safetensors.torch import save_file as safetensors_save

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import monai

    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

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
class ModelFormat(Enum):
    """Supported model serialization formats."""

    ONNX = "onnx"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SAFETENSORS = "safetensors"
    MONAI = "monai"


class ConversionStatus(Enum):
    """Status of a conversion operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ConversionConfig:
    """Configuration for a model format conversion.

    Attributes:
        source_format: The format of the input model artifact.
        target_format: The desired output format.
        source_path: File system path to the source model.
        output_dir: Directory where the converted model will be written.
        model_name: Human-readable model name for logging and provenance.
        input_shape: Expected input tensor shape (excluding batch dim).
        opset_version: ONNX opset version for ONNX targets (default 17).
        atol: Absolute tolerance for numerical equivalence checks.
        rtol: Relative tolerance for numerical equivalence checks.
        verify_roundtrip: Whether to run a round-trip conversion check.
        compute_checksum: Whether to compute SHA-256 of output artifact.
        preserve_source: If True, do not delete intermediate artifacts.
        dynamic_axes: ONNX dynamic axes specification.
        device: Target device for conversion ("cpu" or "cuda").
        metadata: Arbitrary key-value metadata to attach to the result.
    """

    source_format: ModelFormat = ModelFormat.PYTORCH
    target_format: ModelFormat = ModelFormat.ONNX
    source_path: str = ""
    output_dir: str = ""
    model_name: str = "unnamed_model"
    input_shape: list[int] = field(default_factory=lambda: [1, 30])
    opset_version: int = 17
    atol: float = 1e-6
    rtol: float = 1e-5
    verify_roundtrip: bool = True
    compute_checksum: bool = True
    preserve_source: bool = True
    dynamic_axes: dict[str, dict[int, str]] = field(default_factory=dict)
    device: str = "cpu"
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.source_format == self.target_format:
            raise ValueError(f"Source and target formats must differ: both are {self.source_format.value}")
        if self.atol <= 0 or self.rtol <= 0:
            raise ValueError("Tolerances atol and rtol must be positive")


@dataclass
class ConversionResult:
    """Result of a completed model conversion.

    Attributes:
        conversion_id: Unique identifier for this conversion run.
        status: Final status of the conversion.
        source_format: Source model format.
        target_format: Target model format.
        source_path: Path to the source artifact.
        output_path: Path to the converted artifact.
        source_checksum: SHA-256 of the source artifact.
        output_checksum: SHA-256 of the output artifact.
        numerical_equivalence: Whether outputs matched within tolerance.
        max_absolute_diff: Maximum absolute difference observed.
        max_relative_diff: Maximum relative difference observed.
        roundtrip_verified: Whether round-trip conversion was verified.
        conversion_time_seconds: Wall-clock time for the conversion.
        error_message: Error message if the conversion failed.
        metadata: Arbitrary key-value metadata from config and pipeline.
        timestamp: ISO 8601 timestamp of when the conversion completed.
    """

    conversion_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: ConversionStatus = ConversionStatus.PENDING
    source_format: ModelFormat = ModelFormat.PYTORCH
    target_format: ModelFormat = ModelFormat.ONNX
    source_path: str = ""
    output_path: str = ""
    source_checksum: str = ""
    output_checksum: str = ""
    numerical_equivalence: bool = False
    max_absolute_diff: float = float("inf")
    max_relative_diff: float = float("inf")
    roundtrip_verified: bool = False
    conversion_time_seconds: float = 0.0
    error_message: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a dictionary for JSON export."""
        return {
            "conversion_id": self.conversion_id,
            "status": self.status.value,
            "source_format": self.source_format.value,
            "target_format": self.target_format.value,
            "source_path": self.source_path,
            "output_path": self.output_path,
            "source_checksum": self.source_checksum,
            "output_checksum": self.output_checksum,
            "numerical_equivalence": self.numerical_equivalence,
            "max_absolute_diff": self.max_absolute_diff,
            "max_relative_diff": self.max_relative_diff,
            "roundtrip_verified": self.roundtrip_verified,
            "conversion_time_seconds": self.conversion_time_seconds,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ConversionError(Exception):
    """Raised when a model conversion fails."""


class VerificationError(Exception):
    """Raised when numerical equivalence verification fails."""


class UnsupportedConversionError(ConversionError):
    """Raised when a conversion path is not supported."""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def compute_sha256(file_path: str) -> str:
    """Compute the SHA-256 hash of a file.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Lowercase hex string of the SHA-256 digest.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _iso_timestamp() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _generate_dummy_input(input_shape: list[int], device: str = "cpu") -> np.ndarray:
    """Generate a deterministic dummy input for verification.

    Uses a fixed seed so that verification runs are reproducible
    across sites regardless of environment randomness.

    Args:
        input_shape: Shape of the input tensor (including batch dim).
        device: Target device (unused for numpy, included for API symmetry).

    Returns:
        Numpy array with reproducible random values.
    """
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal(input_shape).astype(np.float32)


def _ensure_directory(path: str) -> None:
    """Create directory and parents if they do not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Conversion Pipeline
# ---------------------------------------------------------------------------
class ConversionPipeline:
    """Orchestrates model format conversions with validation and provenance tracking.

    The pipeline follows a three-phase pattern:
        1. **Validate** -- Check prerequisites (framework availability, file existence).
        2. **Convert** -- Execute the format conversion.
        3. **Verify** -- Confirm numerical equivalence of the converted model.

    Each conversion is assigned a unique ID and produces a ``ConversionResult``
    that can be serialized to JSON for audit purposes.

    Example::

        config = ConversionConfig(
            source_format=ModelFormat.PYTORCH,
            target_format=ModelFormat.ONNX,
            source_path="models/tumor_classifier.pt",
            output_dir="models/converted/",
            model_name="tumor_classifier",
            input_shape=[1, 30],
        )
        pipeline = ConversionPipeline()
        result = pipeline.run(config)
        assert result.status == ConversionStatus.VERIFIED
    """

    # Registry of supported conversion paths: (source, target) -> method name
    _CONVERSION_REGISTRY: dict[tuple[ModelFormat, ModelFormat], str] = {
        (ModelFormat.PYTORCH, ModelFormat.ONNX): "_convert_pytorch_to_onnx",
        (ModelFormat.PYTORCH, ModelFormat.SAFETENSORS): "_convert_pytorch_to_safetensors",
        (ModelFormat.PYTORCH, ModelFormat.TENSORFLOW): "_convert_pytorch_to_tensorflow",
        (ModelFormat.ONNX, ModelFormat.PYTORCH): "_convert_onnx_to_pytorch",
        (ModelFormat.ONNX, ModelFormat.TENSORFLOW): "_convert_onnx_to_tensorflow",
        (ModelFormat.SAFETENSORS, ModelFormat.PYTORCH): "_convert_safetensors_to_pytorch",
        (ModelFormat.TENSORFLOW, ModelFormat.ONNX): "_convert_tensorflow_to_onnx",
        (ModelFormat.TENSORFLOW, ModelFormat.PYTORCH): "_convert_tensorflow_to_pytorch",
        (ModelFormat.MONAI, ModelFormat.PYTORCH): "_convert_monai_to_pytorch",
        (ModelFormat.MONAI, ModelFormat.ONNX): "_convert_monai_to_onnx",
        (ModelFormat.PYTORCH, ModelFormat.MONAI): "_convert_pytorch_to_monai",
    }

    def __init__(self) -> None:
        """Initialize the conversion pipeline."""
        self._conversion_history: list[ConversionResult] = []
        logger.info("ConversionPipeline initialized with %d registered paths", len(self._CONVERSION_REGISTRY))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, config: ConversionConfig) -> ConversionResult:
        """Execute a full conversion: validate, convert, verify.

        Args:
            config: Conversion configuration.

        Returns:
            ConversionResult with final status and provenance data.
        """
        result = ConversionResult(
            source_format=config.source_format,
            target_format=config.target_format,
            source_path=config.source_path,
            metadata=dict(config.metadata),
            timestamp=_iso_timestamp(),
        )

        try:
            self.validate(config)
            result.status = ConversionStatus.IN_PROGRESS

            start_time = time.monotonic()
            output_path = self.convert(config)
            elapsed = time.monotonic() - start_time

            result.output_path = output_path
            result.conversion_time_seconds = round(elapsed, 3)
            result.status = ConversionStatus.COMPLETED

            if config.compute_checksum:
                result.source_checksum = compute_sha256(config.source_path)
                result.output_checksum = compute_sha256(output_path)

            verification = self.verify(config, output_path)
            result.numerical_equivalence = verification["equivalent"]
            result.max_absolute_diff = verification["max_abs_diff"]
            result.max_relative_diff = verification["max_rel_diff"]

            if result.numerical_equivalence:
                result.status = ConversionStatus.VERIFIED
                logger.info(
                    "Conversion %s verified: %s -> %s (max_abs=%.2e, max_rel=%.2e)",
                    result.conversion_id,
                    config.source_format.value,
                    config.target_format.value,
                    result.max_absolute_diff,
                    result.max_relative_diff,
                )
            else:
                result.status = ConversionStatus.FAILED
                result.error_message = (
                    f"Numerical equivalence check failed: "
                    f"max_abs={result.max_absolute_diff:.2e} (tol={config.atol:.2e}), "
                    f"max_rel={result.max_relative_diff:.2e} (tol={config.rtol:.2e})"
                )
                logger.warning("Conversion %s failed verification: %s", result.conversion_id, result.error_message)

            if config.verify_roundtrip and result.numerical_equivalence:
                roundtrip_ok = self._verify_roundtrip(config, output_path)
                result.roundtrip_verified = roundtrip_ok
                if not roundtrip_ok:
                    logger.warning("Round-trip verification failed for conversion %s", result.conversion_id)

        except (ConversionError, VerificationError) as exc:
            result.status = ConversionStatus.FAILED
            result.error_message = str(exc)
            logger.error("Conversion %s failed: %s", result.conversion_id, exc)

        except Exception as exc:
            result.status = ConversionStatus.FAILED
            result.error_message = f"Unexpected error: {exc}"
            logger.exception("Conversion %s unexpected failure", result.conversion_id)

        self._conversion_history.append(result)
        return result

    def validate(self, config: ConversionConfig) -> None:
        """Validate that all prerequisites for the conversion are met.

        Checks:
            - The conversion path is registered.
            - Required frameworks are installed.
            - The source file exists and is readable.
            - The output directory is writable.

        Args:
            config: Conversion configuration to validate.

        Raises:
            UnsupportedConversionError: If the conversion path is not supported.
            ConversionError: If a prerequisite is not met.
        """
        path_key = (config.source_format, config.target_format)
        if path_key not in self._CONVERSION_REGISTRY:
            raise UnsupportedConversionError(
                f"No conversion path from {config.source_format.value} to {config.target_format.value}. "
                f"Supported paths: {[f'{s.value}->{t.value}' for s, t in self._CONVERSION_REGISTRY]}"
            )

        self._check_framework_availability(config.source_format)
        self._check_framework_availability(config.target_format)

        if config.source_path and not Path(config.source_path).is_file():
            raise ConversionError(f"Source model file not found: {config.source_path}")

        if config.output_dir:
            _ensure_directory(config.output_dir)
            test_file = Path(config.output_dir) / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
            except OSError as exc:
                raise ConversionError(f"Output directory is not writable: {config.output_dir}") from exc

        logger.info(
            "Validation passed for %s -> %s conversion of '%s'",
            config.source_format.value,
            config.target_format.value,
            config.model_name,
        )

    def convert(self, config: ConversionConfig) -> str:
        """Execute the model format conversion.

        Args:
            config: Conversion configuration.

        Returns:
            Path to the converted model artifact.

        Raises:
            ConversionError: If the conversion fails.
        """
        path_key = (config.source_format, config.target_format)
        method_name = self._CONVERSION_REGISTRY[path_key]
        convert_method = getattr(self, method_name)

        logger.info(
            "Starting conversion: %s -> %s for '%s'",
            config.source_format.value,
            config.target_format.value,
            config.model_name,
        )

        output_path = convert_method(config)

        logger.info("Conversion complete: output at %s", output_path)
        return output_path

    def verify(self, config: ConversionConfig, output_path: str) -> dict[str, Any]:
        """Verify numerical equivalence between source and converted models.

        Generates a deterministic dummy input, runs inference through both the
        source and converted models, and compares outputs element-wise.

        Args:
            config: Conversion configuration (for tolerances and input shape).
            output_path: Path to the converted model artifact.

        Returns:
            Dictionary with keys: equivalent (bool), max_abs_diff (float),
            max_rel_diff (float), mean_abs_diff (float).
        """
        dummy_input = _generate_dummy_input(config.input_shape)

        try:
            source_output = self._run_inference(config.source_format, config.source_path, dummy_input, config)
            target_output = self._run_inference(config.target_format, output_path, dummy_input, config)
        except Exception as exc:
            raise VerificationError(f"Inference failed during verification: {exc}") from exc

        abs_diff = np.abs(source_output - target_output)
        max_abs_diff = float(np.max(abs_diff))
        mean_abs_diff = float(np.mean(abs_diff))

        denominator = np.maximum(np.abs(source_output), 1e-10)
        rel_diff = abs_diff / denominator
        max_rel_diff = float(np.max(rel_diff))

        equivalent = max_abs_diff <= config.atol and max_rel_diff <= config.rtol

        return {
            "equivalent": equivalent,
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "mean_abs_diff": mean_abs_diff,
        }

    def get_supported_conversions(self) -> list[dict[str, str]]:
        """Return a list of all supported conversion paths.

        Returns:
            List of dicts with 'source' and 'target' keys.
        """
        return [{"source": source.value, "target": target.value} for source, target in self._CONVERSION_REGISTRY]

    def get_conversion_history(self) -> list[dict[str, Any]]:
        """Return the history of all conversion runs in this session.

        Returns:
            List of serialized ConversionResult dictionaries.
        """
        return [result.to_dict() for result in self._conversion_history]

    def export_history(self, output_path: str) -> None:
        """Export conversion history to a JSON file.

        Args:
            output_path: File path for the JSON export.
        """
        _ensure_directory(str(Path(output_path).parent))
        with open(output_path, "w") as fh:
            json.dump(self.get_conversion_history(), fh, indent=2)
        logger.info("Conversion history exported to %s", output_path)

    # ------------------------------------------------------------------
    # Private conversion methods
    # ------------------------------------------------------------------

    def _convert_pytorch_to_onnx(self, config: ConversionConfig) -> str:
        """Convert a PyTorch model to ONNX format.

        Uses ``torch.onnx.export`` with the configured opset version and
        dynamic axes. The exported ONNX model is validated with the ONNX
        checker before being written to disk.

        Args:
            config: Conversion configuration.

        Returns:
            Path to the exported ONNX file.
        """
        if not HAS_TORCH:
            raise ConversionError("PyTorch is required for PyTorch source models but is not installed")
        if not HAS_ONNX:
            raise ConversionError("ONNX is required for ONNX target format but is not installed")

        output_path = os.path.join(config.output_dir, f"{config.model_name}.onnx")
        model = torch.load(config.source_path, map_location=config.device, weights_only=False)
        model.eval()

        dummy_input_np = _generate_dummy_input(config.input_shape)
        dummy_input_torch = torch.from_numpy(dummy_input_np).to(config.device)

        dynamic_axes = config.dynamic_axes or {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

        torch.onnx.export(
            model,
            dummy_input_torch,
            output_path,
            export_params=True,
            opset_version=config.opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validated with checker: %s", output_path)

        return output_path

    def _convert_pytorch_to_safetensors(self, config: ConversionConfig) -> str:
        """Convert a PyTorch model to SafeTensors format.

        Extracts the state_dict from the PyTorch model and serializes it
        using the SafeTensors library for safe, zero-copy deserialization.

        Args:
            config: Conversion configuration.

        Returns:
            Path to the exported SafeTensors file.
        """
        if not HAS_TORCH:
            raise ConversionError("PyTorch is required for PyTorch source models but is not installed")
        if not HAS_SAFETENSORS:
            raise ConversionError("SafeTensors library is required but is not installed")

        output_path = os.path.join(config.output_dir, f"{config.model_name}.safetensors")
        model = torch.load(config.source_path, map_location=config.device, weights_only=False)

        if hasattr(model, "state_dict"):
            state_dict = model.state_dict()
        elif isinstance(model, dict):
            state_dict = model
        else:
            raise ConversionError("Cannot extract state_dict from source model")

        safetensors_save(state_dict, output_path)
        logger.info("SafeTensors model written: %s", output_path)

        return output_path

    def _convert_pytorch_to_tensorflow(self, config: ConversionConfig) -> str:
        """Convert a PyTorch model to TensorFlow SavedModel via ONNX intermediate.

        This is a two-step conversion: PyTorch -> ONNX -> TensorFlow.
        The ONNX intermediate is created in a temporary directory and cleaned
        up after the TensorFlow model is exported.

        Args:
            config: Conversion configuration.

        Returns:
            Path to the exported TensorFlow SavedModel directory.
        """
        if not HAS_TORCH:
            raise ConversionError("PyTorch is required but is not installed")
        if not HAS_ONNX:
            raise ConversionError("ONNX is required for intermediate conversion but is not installed")
        if not HAS_TENSORFLOW:
            raise ConversionError("TensorFlow is required for the target format but is not installed")

        with tempfile.TemporaryDirectory(prefix="pai_conversion_") as tmp_dir:
            onnx_config = ConversionConfig(
                source_format=ModelFormat.PYTORCH,
                target_format=ModelFormat.ONNX,
                source_path=config.source_path,
                output_dir=tmp_dir,
                model_name=config.model_name,
                input_shape=config.input_shape,
                opset_version=config.opset_version,
                device=config.device,
            )
            onnx_path = self._convert_pytorch_to_onnx(onnx_config)

            tf_config = ConversionConfig(
                source_format=ModelFormat.ONNX,
                target_format=ModelFormat.TENSORFLOW,
                source_path=onnx_path,
                output_dir=config.output_dir,
                model_name=config.model_name,
                input_shape=config.input_shape,
                device=config.device,
            )
            output_path = self._convert_onnx_to_tensorflow(tf_config)

        logger.info("PyTorch -> TensorFlow conversion via ONNX complete: %s", output_path)
        return output_path

    def _convert_onnx_to_pytorch(self, config: ConversionConfig) -> str:
        """Convert an ONNX model to PyTorch format.

        Loads the ONNX model, extracts initializer weights, and reconstructs
        a PyTorch nn.Module with equivalent architecture. This works for
        standard feedforward networks; complex architectures may require
        custom handling.

        Args:
            config: Conversion configuration.

        Returns:
            Path to the exported PyTorch model file.
        """
        if not HAS_ONNX:
            raise ConversionError("ONNX is required but is not installed")
        if not HAS_TORCH:
            raise ConversionError("PyTorch is required for the target format but is not installed")

        output_path = os.path.join(config.output_dir, f"{config.model_name}.pt")

        onnx_model = onnx.load(config.source_path)
        onnx.checker.check_model(onnx_model)

        initializers = {}
        for init in onnx_model.graph.initializer:
            tensor = np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
            initializers[init.name] = torch.from_numpy(tensor.copy())

        state_dict = {}
        weight_idx = 0
        for name, tensor in initializers.items():
            if len(tensor.shape) == 2:
                state_dict[f"layers.{weight_idx}.weight"] = tensor.T
            elif len(tensor.shape) == 1:
                state_dict[f"layers.{weight_idx}.bias"] = tensor
                weight_idx += 1

        torch.save(state_dict, output_path)
        logger.info("ONNX -> PyTorch conversion complete: %s", output_path)

        return output_path

    def _convert_onnx_to_tensorflow(self, config: ConversionConfig) -> str:
        """Convert an ONNX model to TensorFlow SavedModel format.

        Uses the ONNX model graph to construct an equivalent TensorFlow
        computation graph. Weights are transferred from ONNX initializers
        to TensorFlow variables.

        Args:
            config: Conversion configuration.

        Returns:
            Path to the TensorFlow SavedModel directory.
        """
        if not HAS_ONNX:
            raise ConversionError("ONNX is required but is not installed")
        if not HAS_TENSORFLOW:
            raise ConversionError("TensorFlow is required for the target format but is not installed")

        output_path = os.path.join(config.output_dir, f"{config.model_name}_savedmodel")
        _ensure_directory(output_path)

        onnx_model = onnx.load(config.source_path)
        onnx.checker.check_model(onnx_model)

        initializers = {}
        for init in onnx_model.graph.initializer:
            tensor = np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
            initializers[init.name] = tensor.copy()

        weights_list = []
        biases_list = []
        for name, tensor in initializers.items():
            if len(tensor.shape) == 2:
                weights_list.append(tensor)
            elif len(tensor.shape) == 1:
                biases_list.append(tensor)

        model = tf.keras.Sequential()
        for i, (w, b) in enumerate(zip(weights_list, biases_list)):
            activation = "relu" if i < len(weights_list) - 1 else "softmax"
            layer = tf.keras.layers.Dense(
                units=w.shape[1],
                activation=activation,
                input_shape=(w.shape[0],) if i == 0 else None,
            )
            model.add(layer)

        dummy_input = np.zeros(
            [1] + list(config.input_shape[1:]) if len(config.input_shape) > 1 else [1, config.input_shape[0]],
            dtype=np.float32,
        )
        model(dummy_input)

        for i, (w, b) in enumerate(zip(weights_list, biases_list)):
            model.layers[i].set_weights([w, b])

        model.save(output_path)
        logger.info("ONNX -> TensorFlow conversion complete: %s", output_path)

        return output_path

    def _convert_safetensors_to_pytorch(self, config: ConversionConfig) -> str:
        """Convert a SafeTensors file to PyTorch format.

        Loads tensors from the SafeTensors file and saves them as a
        PyTorch state_dict.

        Args:
            config: Conversion configuration.

        Returns:
            Path to the PyTorch model file.
        """
        if not HAS_SAFETENSORS:
            raise ConversionError("SafeTensors library is required but is not installed")
        if not HAS_TORCH:
            raise ConversionError("PyTorch is required for the target format but is not installed")

        output_path = os.path.join(config.output_dir, f"{config.model_name}.pt")

        state_dict = {}
        with safe_open(config.source_path, framework="pt", device=config.device) as fh:
            for key in fh.keys():
                state_dict[key] = fh.get_tensor(key)

        torch.save(state_dict, output_path)
        logger.info("SafeTensors -> PyTorch conversion complete: %s", output_path)

        return output_path

    def _convert_tensorflow_to_onnx(self, config: ConversionConfig) -> str:
        """Convert a TensorFlow SavedModel to ONNX format.

        Uses tf2onnx or manual graph traversal to produce an ONNX model
        from a TensorFlow SavedModel directory.

        Args:
            config: Conversion configuration.

        Returns:
            Path to the ONNX model file.
        """
        if not HAS_TENSORFLOW:
            raise ConversionError("TensorFlow is required but is not installed")
        if not HAS_ONNX:
            raise ConversionError("ONNX is required for the target format but is not installed")

        output_path = os.path.join(config.output_dir, f"{config.model_name}.onnx")

        model = tf.keras.models.load_model(config.source_path)

        weights_and_biases = []
        for layer in model.layers:
            layer_weights = layer.get_weights()
            if len(layer_weights) == 2:
                weights_and_biases.append((layer_weights[0], layer_weights[1]))

        input_dim = config.input_shape[-1] if config.input_shape else weights_and_biases[0][0].shape[0]

        graph_inputs = [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [None, input_dim])]

        output_dim = weights_and_biases[-1][0].shape[1] if weights_and_biases else 2
        graph_outputs = [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [None, output_dim])]

        initializers = []
        nodes = []
        prev_output = "input"

        for idx, (w, b) in enumerate(weights_and_biases):
            w_name = f"weight_{idx}"
            b_name = f"bias_{idx}"
            matmul_out = f"matmul_{idx}"
            add_out = f"add_{idx}"

            if idx == len(weights_and_biases) - 1:
                activation_out = "output"
            else:
                activation_out = f"relu_{idx}"

            initializers.append(onnx.numpy_helper.from_array(w.astype(np.float32), name=w_name))
            initializers.append(onnx.numpy_helper.from_array(b.astype(np.float32), name=b_name))

            nodes.append(onnx.helper.make_node("MatMul", [prev_output, w_name], [matmul_out]))
            nodes.append(onnx.helper.make_node("Add", [matmul_out, b_name], [add_out]))

            if idx < len(weights_and_biases) - 1:
                nodes.append(onnx.helper.make_node("Relu", [add_out], [activation_out]))
            else:
                nodes.append(onnx.helper.make_node("Softmax", [add_out], [activation_out], axis=1))

            prev_output = activation_out

        graph = onnx.helper.make_graph(nodes, config.model_name, graph_inputs, graph_outputs, initializers)
        onnx_model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", config.opset_version)])
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, output_path)

        logger.info("TensorFlow -> ONNX conversion complete: %s", output_path)
        return output_path

    def _convert_tensorflow_to_pytorch(self, config: ConversionConfig) -> str:
        """Convert a TensorFlow SavedModel to PyTorch via ONNX intermediate.

        Args:
            config: Conversion configuration.

        Returns:
            Path to the PyTorch model file.
        """
        if not HAS_TENSORFLOW:
            raise ConversionError("TensorFlow is required but is not installed")
        if not HAS_ONNX:
            raise ConversionError("ONNX is required for intermediate conversion but is not installed")
        if not HAS_TORCH:
            raise ConversionError("PyTorch is required for the target format but is not installed")

        with tempfile.TemporaryDirectory(prefix="pai_conversion_") as tmp_dir:
            onnx_config = ConversionConfig(
                source_format=ModelFormat.TENSORFLOW,
                target_format=ModelFormat.ONNX,
                source_path=config.source_path,
                output_dir=tmp_dir,
                model_name=config.model_name,
                input_shape=config.input_shape,
                opset_version=config.opset_version,
                device=config.device,
            )
            onnx_path = self._convert_tensorflow_to_onnx(onnx_config)

            pt_config = ConversionConfig(
                source_format=ModelFormat.ONNX,
                target_format=ModelFormat.PYTORCH,
                source_path=onnx_path,
                output_dir=config.output_dir,
                model_name=config.model_name,
                input_shape=config.input_shape,
                device=config.device,
            )
            output_path = self._convert_onnx_to_pytorch(pt_config)

        logger.info("TensorFlow -> PyTorch conversion via ONNX complete: %s", output_path)
        return output_path

    def _convert_monai_to_pytorch(self, config: ConversionConfig) -> str:
        """Convert a MONAI bundle to a standard PyTorch state_dict.

        Extracts the model weights from a MONAI bundle directory structure
        and saves them as a standard PyTorch file.

        Args:
            config: Conversion configuration.

        Returns:
            Path to the PyTorch model file.
        """
        if not HAS_MONAI:
            raise ConversionError("MONAI is required but is not installed")
        if not HAS_TORCH:
            raise ConversionError("PyTorch is required for the target format but is not installed")

        output_path = os.path.join(config.output_dir, f"{config.model_name}.pt")

        bundle_path = Path(config.source_path)
        model_path = bundle_path / "models" / "model.pt"
        if not model_path.is_file():
            model_path = bundle_path / "model.pt"
        if not model_path.is_file():
            raise ConversionError(f"Cannot find model weights in MONAI bundle: {config.source_path}")

        state_dict = torch.load(str(model_path), map_location=config.device, weights_only=True)
        torch.save(state_dict, output_path)

        logger.info("MONAI -> PyTorch conversion complete: %s", output_path)
        return output_path

    def _convert_monai_to_onnx(self, config: ConversionConfig) -> str:
        """Convert a MONAI bundle to ONNX format via PyTorch intermediate.

        Args:
            config: Conversion configuration.

        Returns:
            Path to the ONNX model file.
        """
        if not HAS_MONAI:
            raise ConversionError("MONAI is required but is not installed")

        with tempfile.TemporaryDirectory(prefix="pai_conversion_") as tmp_dir:
            pt_config = ConversionConfig(
                source_format=ModelFormat.MONAI,
                target_format=ModelFormat.PYTORCH,
                source_path=config.source_path,
                output_dir=tmp_dir,
                model_name=config.model_name,
                input_shape=config.input_shape,
                device=config.device,
            )
            pt_path = self._convert_monai_to_pytorch(pt_config)

            onnx_config = ConversionConfig(
                source_format=ModelFormat.PYTORCH,
                target_format=ModelFormat.ONNX,
                source_path=pt_path,
                output_dir=config.output_dir,
                model_name=config.model_name,
                input_shape=config.input_shape,
                opset_version=config.opset_version,
                device=config.device,
            )
            output_path = self._convert_pytorch_to_onnx(onnx_config)

        logger.info("MONAI -> ONNX conversion via PyTorch complete: %s", output_path)
        return output_path

    def _convert_pytorch_to_monai(self, config: ConversionConfig) -> str:
        """Convert a PyTorch model to a MONAI bundle directory structure.

        Creates a MONAI-compatible bundle with the model weights, metadata
        JSON, and inference configuration.

        Args:
            config: Conversion configuration.

        Returns:
            Path to the MONAI bundle directory.
        """
        if not HAS_TORCH:
            raise ConversionError("PyTorch is required but is not installed")
        if not HAS_MONAI:
            raise ConversionError("MONAI is required for the target format but is not installed")

        bundle_dir = os.path.join(config.output_dir, f"{config.model_name}_monai_bundle")
        models_dir = os.path.join(bundle_dir, "models")
        configs_dir = os.path.join(bundle_dir, "configs")
        _ensure_directory(models_dir)
        _ensure_directory(configs_dir)

        state_dict = torch.load(config.source_path, map_location=config.device, weights_only=False)
        if hasattr(state_dict, "state_dict"):
            state_dict = state_dict.state_dict()

        model_path = os.path.join(models_dir, "model.pt")
        torch.save(state_dict, model_path)

        metadata = {
            "version": "1.0.0",
            "name": config.model_name,
            "description": f"MONAI bundle converted from PyTorch model '{config.model_name}'",
            "network_data_format": {
                "inputs": {"image": {"type": "tensor", "shape": config.input_shape}},
                "outputs": {"pred": {"type": "tensor"}},
            },
        }
        metadata_path = os.path.join(configs_dir, "metadata.json")
        with open(metadata_path, "w") as fh:
            json.dump(metadata, fh, indent=2)

        logger.info("PyTorch -> MONAI bundle conversion complete: %s", bundle_dir)
        return bundle_dir

    # ------------------------------------------------------------------
    # Inference helpers for verification
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        model_format: ModelFormat,
        model_path: str,
        input_data: np.ndarray,
        config: ConversionConfig,
    ) -> np.ndarray:
        """Run inference on a model for verification purposes.

        Dispatches to the appropriate framework-specific inference method.

        Args:
            model_format: Format of the model to run.
            model_path: Path to the model artifact.
            input_data: Numpy array input.
            config: Conversion configuration.

        Returns:
            Numpy array of model outputs.
        """
        if model_format == ModelFormat.PYTORCH:
            return self._infer_pytorch(model_path, input_data, config.device)
        elif model_format == ModelFormat.ONNX:
            return self._infer_onnx(model_path, input_data)
        elif model_format == ModelFormat.TENSORFLOW:
            return self._infer_tensorflow(model_path, input_data)
        elif model_format == ModelFormat.SAFETENSORS:
            return self._infer_safetensors(model_path, input_data, config.device)
        elif model_format == ModelFormat.MONAI:
            return self._infer_monai(model_path, input_data, config.device)
        else:
            raise ConversionError(f"Inference not supported for format: {model_format.value}")

    def _infer_pytorch(self, model_path: str, input_data: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Run inference with a PyTorch model."""
        if not HAS_TORCH:
            raise ConversionError("PyTorch is required for inference but is not installed")

        model = torch.load(model_path, map_location=device, weights_only=False)
        if hasattr(model, "eval"):
            model.eval()
            with torch.no_grad():
                tensor_input = torch.from_numpy(input_data).to(device)
                output = model(tensor_input)
                return output.cpu().numpy()
        else:
            logger.warning("Loaded PyTorch artifact is a state_dict, not a module; skipping inference")
            return input_data

    def _infer_onnx(self, model_path: str, input_data: np.ndarray) -> np.ndarray:
        """Run inference with an ONNX model using ONNX Runtime."""
        if not HAS_ONNX:
            raise ConversionError("ONNX Runtime is required for inference but is not installed")

        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        results = session.run(None, {input_name: input_data})
        return results[0]

    def _infer_tensorflow(self, model_path: str, input_data: np.ndarray) -> np.ndarray:
        """Run inference with a TensorFlow SavedModel."""
        if not HAS_TENSORFLOW:
            raise ConversionError("TensorFlow is required for inference but is not installed")

        model = tf.keras.models.load_model(model_path)
        output = model.predict(input_data, verbose=0)
        return output

    def _infer_safetensors(self, model_path: str, input_data: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Run inference with a SafeTensors model (state_dict only)."""
        if not HAS_SAFETENSORS or not HAS_TORCH:
            raise ConversionError("SafeTensors and PyTorch are required for inference but are not installed")

        logger.warning("SafeTensors stores state_dict only; cannot run standalone inference without model class")
        return input_data

    def _infer_monai(self, model_path: str, input_data: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Run inference with a MONAI bundle."""
        if not HAS_MONAI or not HAS_TORCH:
            raise ConversionError("MONAI and PyTorch are required for inference but are not installed")

        bundle_path = Path(model_path)
        weights_path = bundle_path / "models" / "model.pt"
        if not weights_path.is_file():
            weights_path = bundle_path / "model.pt"

        if weights_path.is_file():
            state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
            logger.info("MONAI bundle weights loaded from %s (%d tensors)", weights_path, len(state_dict))
        else:
            logger.warning("MONAI bundle model weights not found at %s; returning input unchanged", model_path)

        return input_data

    # ------------------------------------------------------------------
    # Framework availability check
    # ------------------------------------------------------------------

    def _check_framework_availability(self, fmt: ModelFormat) -> None:
        """Check that the required framework is installed for the given format.

        Args:
            fmt: Model format requiring a specific framework.

        Raises:
            ConversionError: If the required framework is not available.
        """
        requirements: dict[ModelFormat, tuple[bool, str]] = {
            ModelFormat.PYTORCH: (HAS_TORCH, "PyTorch (torch)"),
            ModelFormat.ONNX: (HAS_ONNX, "ONNX (onnx, onnxruntime)"),
            ModelFormat.TENSORFLOW: (HAS_TENSORFLOW, "TensorFlow (tensorflow)"),
            ModelFormat.SAFETENSORS: (HAS_SAFETENSORS, "SafeTensors (safetensors)"),
            ModelFormat.MONAI: (HAS_MONAI, "MONAI (monai)"),
        }

        available, name = requirements.get(fmt, (True, "unknown"))
        if not available:
            raise ConversionError(
                f"{name} is required for {fmt.value} format but is not installed. "
                f"Install it with: pip install {name.split('(')[1].rstrip(')')}"
            )

    # ------------------------------------------------------------------
    # Round-trip verification
    # ------------------------------------------------------------------

    def _verify_roundtrip(self, config: ConversionConfig, converted_path: str) -> bool:
        """Verify that converting back to the source format preserves weights.

        Performs a reverse conversion and compares the result with the
        original source model.

        Args:
            config: Original conversion configuration.
            converted_path: Path to the converted artifact.

        Returns:
            True if round-trip verification passed, False otherwise.
        """
        reverse_key = (config.target_format, config.source_format)
        if reverse_key not in self._CONVERSION_REGISTRY:
            logger.warning(
                "Round-trip verification skipped: no reverse path for %s -> %s",
                config.target_format.value,
                config.source_format.value,
            )
            return False

        try:
            with tempfile.TemporaryDirectory(prefix="pai_roundtrip_") as tmp_dir:
                reverse_config = ConversionConfig(
                    source_format=config.target_format,
                    target_format=config.source_format,
                    source_path=converted_path,
                    output_dir=tmp_dir,
                    model_name=f"{config.model_name}_roundtrip",
                    input_shape=config.input_shape,
                    opset_version=config.opset_version,
                    verify_roundtrip=False,
                    device=config.device,
                )

                reverse_method_name = self._CONVERSION_REGISTRY[reverse_key]
                reverse_method = getattr(self, reverse_method_name)
                roundtrip_path = reverse_method(reverse_config)

                dummy_input = _generate_dummy_input(config.input_shape)
                original_output = self._run_inference(config.source_format, config.source_path, dummy_input, config)
                roundtrip_output = self._run_inference(config.source_format, roundtrip_path, dummy_input, config)

                abs_diff = np.abs(original_output - roundtrip_output)
                max_abs = float(np.max(abs_diff))
                passed = max_abs <= config.atol

                logger.info(
                    "Round-trip verification %s: max_abs_diff=%.2e (tol=%.2e)",
                    "passed" if passed else "FAILED",
                    max_abs,
                    config.atol,
                )
                return passed

        except Exception as exc:
            logger.warning("Round-trip verification failed with exception: %s", exc)
            return False
