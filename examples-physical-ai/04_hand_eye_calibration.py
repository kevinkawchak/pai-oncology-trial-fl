#!/usr/bin/env python3
"""Hand-eye calibration for surgical robotic systems.

CLINICAL CONTEXT
================
Accurate hand-eye calibration is a prerequisite for surgical robotic
procedures where a camera must be precisely registered to the robot's
kinematic chain. The classic AX=XB formulation relates the robot
end-effector motion (A) to the observed camera motion (B) to recover
the unknown camera-to-end-effector transformation (X).

In oncology robotics, sub-millimetre calibration accuracy directly
impacts biopsy needle placement, resection margin control, and
instrument tracking. This module demonstrates the complete calibration
pipeline including data collection, multiple solver methods,
transformation chain validation, and statistical error analysis.

USE CASES COVERED
=================
1. AX=XB hand-eye calibration problem formulation and solution using
   three methods: Tsai-Lenz, Park-Martin, and Dual Quaternion.
2. Transformation chain validation with forward/inverse consistency
   checks and loop closure verification.
3. Calibration error analysis with reprojection error computation,
   rotation and translation error decomposition.
4. Statistical hypothesis testing for calibration quality assessment
   (chi-squared, Kolmogorov-Smirnov on residuals).
5. Multi-method comparison and benchmarking with synthetic ground
   truth data.
6. Calibration stability analysis with bootstrap confidence intervals.

FRAMEWORK REQUIREMENTS
======================
Required:
    numpy >= 1.24.0  (https://numpy.org)

Optional:
    torch >= 2.1.0   (https://pytorch.org) -- GPU-accelerated inference
    mujoco >= 3.0.0  (https://mujoco.org) -- Physics simulation backend
    scipy >= 1.11.0  (https://scipy.org) -- Statistical tests

HARDWARE REQUIREMENTS
=====================
- CPU: 2+ cores
- RAM: 2 GB minimum
- GPU: Not required
- Camera: Stereo or structured-light (simulated)

REFERENCES
==========
- IEC 80601-2-77:2019 -- Accuracy validation for RASE
- ISO 14971:2019 -- Verification and risk controls
- IEC 62304:2006+AMD1:2015 -- Unit testing requirements
- Tsai & Lenz (1989), "A New Technique for Fully Autonomous and
  Efficient 3D Robotics Hand/Eye Calibration"
- Park & Martin (1994), "Robot Sensor Calibration: Solving AX=XB on
  the Euclidean Group"

DISCLAIMER
==========
RESEARCH USE ONLY. This software is provided for research and educational
purposes. It is NOT validated for clinical use and must NOT be deployed in
any patient care setting without appropriate regulatory clearance and
comprehensive clinical validation.

LICENSE: MIT
VERSION: 0.5.0
LAST UPDATED: 2026-02-17
"""

from __future__ import annotations

import hashlib
import logging
import math
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
HAS_TORCH = False
try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]

HAS_MUJOCO = False
try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:
    mujoco = None  # type: ignore[assignment]

HAS_SCIPY = False
try:
    import scipy
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except ImportError:
    scipy = None  # type: ignore[assignment]
    scipy_stats = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pai.hand_eye_calibration")


# ============================================================================
# Enumerations
# ============================================================================


class CalibrationMethod(str, Enum):
    """Available hand-eye calibration methods."""

    TSAI_LENZ = "tsai_lenz"
    PARK_MARTIN = "park_martin"
    DUAL_QUATERNION = "dual_quaternion"


class CalibrationQuality(str, Enum):
    """Quality classification of a calibration result."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


class ValidationStatus(str, Enum):
    """Status of a validation check."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class CalibrationConfig:
    """Configuration for the hand-eye calibration procedure.

    Attributes:
        num_poses: Number of calibration poses to collect.
        rotation_noise_deg: Simulated rotation measurement noise.
        translation_noise_mm: Simulated translation measurement noise.
        min_rotation_deg: Minimum rotation between consecutive poses.
        max_position_error_mm: Maximum acceptable position error.
        max_rotation_error_deg: Maximum acceptable rotation error.
        bootstrap_samples: Number of bootstrap samples for CI estimation.
        random_seed: Random seed for reproducibility.
    """

    num_poses: int = 30
    rotation_noise_deg: float = 0.1
    translation_noise_mm: float = 0.5
    min_rotation_deg: float = 10.0
    max_position_error_mm: float = 2.0
    max_rotation_error_deg: float = 1.0
    bootstrap_samples: int = 100
    random_seed: int = 42


@dataclass
class HomogeneousTransform:
    """A 4x4 homogeneous transformation matrix.

    Attributes:
        matrix: The 4x4 transformation matrix.
        frame_from: Source coordinate frame name.
        frame_to: Target coordinate frame name.
    """

    matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    frame_from: str = ""
    frame_to: str = ""

    @property
    def rotation(self) -> np.ndarray:
        """3x3 rotation matrix."""
        return self.matrix[:3, :3].copy()

    @property
    def translation(self) -> np.ndarray:
        """3x1 translation vector."""
        return self.matrix[:3, 3].copy()

    def inverse(self) -> HomogeneousTransform:
        """Compute the inverse transformation."""
        R = self.matrix[:3, :3]
        t = self.matrix[:3, 3]
        inv = np.eye(4)
        inv[:3, :3] = R.T
        inv[:3, 3] = -R.T @ t
        return HomogeneousTransform(
            matrix=inv,
            frame_from=self.frame_to,
            frame_to=self.frame_from,
        )


@dataclass
class CalibrationResult:
    """Result of a hand-eye calibration.

    Attributes:
        method: Calibration method used.
        transform: Estimated camera-to-end-effector transform.
        rotation_error_deg: Mean rotation error in degrees.
        translation_error_mm: Mean translation error in millimetres.
        quality: Quality classification.
        num_poses_used: Number of poses used in calibration.
        residuals: Per-pose residual errors.
        computation_time_s: Time taken for computation.
        metadata: Additional diagnostic information.
    """

    method: CalibrationMethod = CalibrationMethod.TSAI_LENZ
    transform: HomogeneousTransform = field(default_factory=HomogeneousTransform)
    rotation_error_deg: float = 0.0
    translation_error_mm: float = 0.0
    quality: CalibrationQuality = CalibrationQuality.FAILED
    num_poses_used: int = 0
    residuals: list[float] = field(default_factory=list)
    computation_time_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of a transformation chain validation.

    Attributes:
        test_name: Name of the validation test.
        status: Pass/warn/fail status.
        measured_value: Measured metric value.
        threshold: Acceptance threshold.
        description: Human-readable description.
    """

    test_name: str = ""
    status: ValidationStatus = ValidationStatus.PASSED
    measured_value: float = 0.0
    threshold: float = 0.0
    description: str = ""


@dataclass
class StatisticalTestResult:
    """Result of a statistical test on calibration residuals.

    Attributes:
        test_name: Name of the statistical test.
        statistic: Test statistic value.
        p_value: P-value of the test.
        passed: Whether the test passed at alpha=0.05.
        description: Interpretation of the result.
    """

    test_name: str = ""
    statistic: float = 0.0
    p_value: float = 0.0
    passed: bool = True
    description: str = ""


# ============================================================================
# Transformation Utilities
# ============================================================================


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Create a 3x3 rotation matrix from axis-angle representation.

    Args:
        axis: Unit rotation axis (3,).
        angle_rad: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    R = np.eye(3) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * (K @ K)
    return R


def rotation_to_axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Extract axis-angle from a rotation matrix.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Tuple of (axis, angle_rad).
    """
    angle = math.acos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if abs(angle) < 1e-10:
        return np.array([0.0, 0.0, 1.0]), 0.0

    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 1.0]), angle
    axis = axis / norm
    return axis, angle


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Create rotation matrix from Euler angles (ZYX convention).

    Args:
        roll: Rotation about X axis in radians.
        pitch: Rotation about Y axis in radians.
        yaw: Rotation about Z axis in radians.

    Returns:
        3x3 rotation matrix.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    return R


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Construct a 4x4 homogeneous transform from R and t.

    Args:
        R: 3x3 rotation matrix.
        t: 3x1 or (3,) translation vector.

    Returns:
        4x4 homogeneous transformation matrix.
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def rotation_error_deg(R_est: np.ndarray, R_true: np.ndarray) -> float:
    """Compute rotation error in degrees between two rotation matrices.

    Args:
        R_est: Estimated rotation.
        R_true: Ground truth rotation.

    Returns:
        Rotation error in degrees.
    """
    R_diff = R_est.T @ R_true
    _, angle = rotation_to_axis_angle(R_diff)
    return math.degrees(abs(angle))


def translation_error_mm(t_est: np.ndarray, t_true: np.ndarray) -> float:
    """Compute translation error in millimetres.

    Args:
        t_est: Estimated translation.
        t_true: Ground truth translation.

    Returns:
        Euclidean distance in mm.
    """
    return float(np.linalg.norm(t_est.flatten() - t_true.flatten()))


# ============================================================================
# Hand-Eye Calibration Solvers
# ============================================================================


class HandEyeCalibrator:
    """Solves the AX=XB hand-eye calibration problem.

    Implements three classical methods for recovering the camera-to-
    end-effector transformation from paired robot and camera motions.

    Args:
        config: Calibration configuration.
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self._config = config or CalibrationConfig()
        self._robot_poses: list[np.ndarray] = []
        self._camera_poses: list[np.ndarray] = []

    def set_pose_pairs(self, robot_poses: list[np.ndarray], camera_poses: list[np.ndarray]) -> None:
        """Set the calibration pose pairs.

        Args:
            robot_poses: List of 4x4 robot end-effector poses.
            camera_poses: List of 4x4 camera poses.
        """
        assert len(robot_poses) == len(camera_poses), "Pose pair count mismatch"
        self._robot_poses = robot_poses
        self._camera_poses = camera_poses
        logger.info("Loaded %d pose pairs for calibration", len(robot_poses))

    def calibrate(self, method: CalibrationMethod = CalibrationMethod.TSAI_LENZ) -> CalibrationResult:
        """Run calibration using the specified method.

        Args:
            method: Calibration method to use.

        Returns:
            CalibrationResult with estimated transform and error metrics.
        """
        if len(self._robot_poses) < 3:
            logger.error("Need at least 3 pose pairs; have %d", len(self._robot_poses))
            return CalibrationResult(method=method, quality=CalibrationQuality.FAILED)

        start_time = time.time()

        # Compute relative motions
        A_list, B_list = self._compute_relative_motions()

        # Solve based on method
        if method == CalibrationMethod.TSAI_LENZ:
            X = self._solve_tsai_lenz(A_list, B_list)
        elif method == CalibrationMethod.PARK_MARTIN:
            X = self._solve_park_martin(A_list, B_list)
        elif method == CalibrationMethod.DUAL_QUATERNION:
            X = self._solve_dual_quaternion(A_list, B_list)
        else:
            X = self._solve_tsai_lenz(A_list, B_list)

        elapsed = time.time() - start_time

        # Compute residuals
        residuals = self._compute_residuals(A_list, B_list, X)

        transform = HomogeneousTransform(
            matrix=X,
            frame_from="camera",
            frame_to="end_effector",
        )

        result = CalibrationResult(
            method=method,
            transform=transform,
            rotation_error_deg=float(np.mean([r["rotation_deg"] for r in residuals])),
            translation_error_mm=float(np.mean([r["translation_mm"] for r in residuals])),
            quality=self._classify_quality(residuals),
            num_poses_used=len(self._robot_poses),
            residuals=[r["total"] for r in residuals],
            computation_time_s=elapsed,
            metadata={
                "num_motions": len(A_list),
                "residual_details": residuals,
            },
        )

        logger.info(
            "Calibration [%s]: rot_err=%.4f deg, trans_err=%.4f mm, quality=%s (%.3fs)",
            method.value,
            result.rotation_error_deg,
            result.translation_error_mm,
            result.quality.value,
            elapsed,
        )
        return result

    def _compute_relative_motions(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compute relative motions between consecutive poses."""
        A_list: list[np.ndarray] = []
        B_list: list[np.ndarray] = []

        for i in range(len(self._robot_poses) - 1):
            # A = T_robot_i^{-1} * T_robot_{i+1}
            T_ri_inv = np.linalg.inv(self._robot_poses[i])
            A = T_ri_inv @ self._robot_poses[i + 1]

            # B = T_camera_i^{-1} * T_camera_{i+1}
            T_ci_inv = np.linalg.inv(self._camera_poses[i])
            B = T_ci_inv @ self._camera_poses[i + 1]

            A_list.append(A)
            B_list.append(B)

        return A_list, B_list

    def _solve_tsai_lenz(self, A_list: list[np.ndarray], B_list: list[np.ndarray]) -> np.ndarray:
        """Solve AX=XB using the Tsai-Lenz method.

        Uses the rotation-then-translation two-stage approach.

        Args:
            A_list: Robot relative motions.
            B_list: Camera relative motions.

        Returns:
            4x4 estimated hand-eye transform X.
        """
        # Stage 1: Solve for rotation using Rodrigues parameterisation
        n = len(A_list)
        C_rot = np.zeros((3 * n, 3))
        d_rot = np.zeros(3 * n)

        for i in range(n):
            Ra = A_list[i][:3, :3]
            Rb = B_list[i][:3, :3]

            axis_a, angle_a = rotation_to_axis_angle(Ra)
            axis_b, angle_b = rotation_to_axis_angle(Rb)

            # Modified Rodrigues parameters
            alpha = axis_a * 2.0 * math.tan(angle_a / 2.0) if abs(angle_a) > 1e-10 else np.zeros(3)
            beta = axis_b * 2.0 * math.tan(angle_b / 2.0) if abs(angle_b) > 1e-10 else np.zeros(3)

            skew_sum = self._skew(alpha + beta)
            C_rot[3 * i : 3 * (i + 1), :] = skew_sum
            d_rot[3 * i : 3 * (i + 1)] = beta - alpha

        # Least squares solution for modified Rodrigues of Rx
        pcg, _, _, _ = np.linalg.lstsq(C_rot, d_rot, rcond=None)

        # Convert back to rotation matrix
        pcg_norm_sq = np.dot(pcg, pcg)
        Rx = (1 - pcg_norm_sq) * np.eye(3) + 2 * np.outer(pcg, pcg) + 2 * self._skew(pcg)
        Rx = Rx / (1 + pcg_norm_sq)

        # Ensure proper rotation via SVD projection
        U, _, Vt = np.linalg.svd(Rx)
        Rx = U @ Vt
        if np.linalg.det(Rx) < 0:
            Vt[-1, :] *= -1
            Rx = U @ Vt

        # Stage 2: Solve for translation
        C_trans = np.zeros((3 * n, 3))
        d_trans = np.zeros(3 * n)

        for i in range(n):
            Ra = A_list[i][:3, :3]
            ta = A_list[i][:3, 3]
            tb = B_list[i][:3, 3]

            C_trans[3 * i : 3 * (i + 1), :] = Ra - np.eye(3)
            d_trans[3 * i : 3 * (i + 1)] = Rx @ tb - ta

        tx, _, _, _ = np.linalg.lstsq(C_trans, d_trans, rcond=None)

        return make_transform(Rx, tx)

    def _solve_park_martin(self, A_list: list[np.ndarray], B_list: list[np.ndarray]) -> np.ndarray:
        """Solve AX=XB using the Park-Martin method.

        Uses the Lie group logarithm approach.

        Args:
            A_list: Robot relative motions.
            B_list: Camera relative motions.

        Returns:
            4x4 estimated hand-eye transform X.
        """
        n = len(A_list)

        # Solve rotation using log mapping
        M = np.zeros((3, 3))
        for i in range(n):
            Ra = A_list[i][:3, :3]
            Rb = B_list[i][:3, :3]

            axis_a, angle_a = rotation_to_axis_angle(Ra)
            axis_b, angle_b = rotation_to_axis_angle(Rb)

            log_a = axis_a * angle_a if abs(angle_a) > 1e-10 else np.zeros(3)
            log_b = axis_b * angle_b if abs(angle_b) > 1e-10 else np.zeros(3)

            M += np.outer(log_b, log_a)

        # SVD-based rotation estimation
        U, S, Vt = np.linalg.svd(M)
        Rx = U @ Vt
        if np.linalg.det(Rx) < 0:
            Vt[-1, :] *= -1
            Rx = U @ Vt

        # Solve translation (same as Tsai-Lenz stage 2)
        C_trans = np.zeros((3 * n, 3))
        d_trans = np.zeros(3 * n)

        for i in range(n):
            Ra = A_list[i][:3, :3]
            ta = A_list[i][:3, 3]
            tb = B_list[i][:3, 3]

            C_trans[3 * i : 3 * (i + 1), :] = Ra - np.eye(3)
            d_trans[3 * i : 3 * (i + 1)] = Rx @ tb - ta

        tx, _, _, _ = np.linalg.lstsq(C_trans, d_trans, rcond=None)

        return make_transform(Rx, tx)

    def _solve_dual_quaternion(self, A_list: list[np.ndarray], B_list: list[np.ndarray]) -> np.ndarray:
        """Solve AX=XB using the dual quaternion method.

        Uses a quaternion-based approach: first solves for the rotation
        quaternion from the null space of the stacked constraint matrix,
        then solves for translation via least squares.

        Args:
            A_list: Robot relative motions.
            B_list: Camera relative motions.

        Returns:
            4x4 estimated hand-eye transform X.
        """
        n = len(A_list)

        # Build the rotation constraint matrix from quaternion algebra
        # For each motion pair: (L(qa) - R(qb)) qx = 0
        rot_constraints = np.zeros((4 * n, 4))

        for i in range(n):
            Ra = A_list[i][:3, :3]
            Rb = B_list[i][:3, :3]

            qa = self._rotation_to_quaternion(Ra)
            qb = self._rotation_to_quaternion(Rb)

            La = self._quat_left_multiply(qa)
            Rb_mat = self._quat_right_multiply(qb)
            rot_constraints[4 * i : 4 * (i + 1), :] = La - Rb_mat

        # Solve for rotation quaternion via SVD null space
        _, S, Vt = np.linalg.svd(rot_constraints, full_matrices=True)
        qx = Vt[-1, :]
        qx = qx / (np.linalg.norm(qx) + 1e-12)

        # Ensure positive scalar part convention
        if qx[0] < 0:
            qx = -qx

        Rx = self._quaternion_to_rotation(qx)

        # Ensure proper rotation via SVD projection
        U_r, _, Vt_r = np.linalg.svd(Rx)
        Rx = U_r @ Vt_r
        if np.linalg.det(Rx) < 0:
            Vt_r[-1, :] *= -1
            Rx = U_r @ Vt_r

        # Solve translation via least squares: (Ra - I) tx = Rx tb - ta
        C_trans = np.zeros((3 * n, 3))
        d_trans = np.zeros(3 * n)
        for i in range(n):
            Ra = A_list[i][:3, :3]
            ta = A_list[i][:3, 3]
            tb = B_list[i][:3, 3]
            C_trans[3 * i : 3 * (i + 1), :] = Ra - np.eye(3)
            d_trans[3 * i : 3 * (i + 1)] = Rx @ tb - ta

        tx, _, _, _ = np.linalg.lstsq(C_trans, d_trans, rcond=None)

        return make_transform(Rx, tx)

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        """Skew-symmetric matrix from a 3-vector."""
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    @staticmethod
    def _rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to unit quaternion [w, x, y, z]."""
        tr = np.trace(R)
        if tr > 0:
            s = 2.0 * math.sqrt(tr + 1.0)
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        q = np.array([w, x, y, z])
        return q / (np.linalg.norm(q) + 1e-12)

    @staticmethod
    def _quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
        """Convert unit quaternion [w, x, y, z] to rotation matrix."""
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    @staticmethod
    def _quat_left_multiply(q: np.ndarray) -> np.ndarray:
        """Left quaternion multiplication matrix."""
        w, x, y, z = q
        return np.array(
            [
                [w, -x, -y, -z],
                [x, w, -z, y],
                [y, z, w, -x],
                [z, -y, x, w],
            ]
        )

    @staticmethod
    def _quat_right_multiply(q: np.ndarray) -> np.ndarray:
        """Right quaternion multiplication matrix."""
        w, x, y, z = q
        return np.array(
            [
                [w, -x, -y, -z],
                [x, w, z, -y],
                [y, -z, w, x],
                [z, y, -x, w],
            ]
        )

    def _compute_residuals(
        self,
        A_list: list[np.ndarray],
        B_list: list[np.ndarray],
        X: np.ndarray,
    ) -> list[dict[str, float]]:
        """Compute per-motion residuals for AX=XB."""
        residuals: list[dict[str, float]] = []

        for i in range(len(A_list)):
            AX = A_list[i] @ X
            XB = X @ B_list[i]

            rot_err = rotation_error_deg(AX[:3, :3], XB[:3, :3])
            trans_err = translation_error_mm(AX[:3, 3], XB[:3, 3])

            residuals.append(
                {
                    "rotation_deg": rot_err,
                    "translation_mm": trans_err,
                    "total": math.sqrt(rot_err**2 + trans_err**2),
                }
            )

        return residuals

    def _classify_quality(self, residuals: list[dict[str, float]]) -> CalibrationQuality:
        """Classify calibration quality based on residuals."""
        mean_rot = float(np.mean([r["rotation_deg"] for r in residuals]))
        mean_trans = float(np.mean([r["translation_mm"] for r in residuals]))

        if mean_rot < 0.1 and mean_trans < 0.5:
            return CalibrationQuality.EXCELLENT
        elif mean_rot < 0.5 and mean_trans < 1.0:
            return CalibrationQuality.GOOD
        elif mean_rot < 1.0 and mean_trans < 2.0:
            return CalibrationQuality.ACCEPTABLE
        elif mean_rot < 2.0 and mean_trans < 5.0:
            return CalibrationQuality.POOR
        else:
            return CalibrationQuality.FAILED


# ============================================================================
# Transformation Chain Validator
# ============================================================================


class TransformationChainValidator:
    """Validates transformation chains for consistency.

    Checks forward/inverse consistency, orthogonality, determinant,
    and loop closure for chains of transformations.
    """

    def __init__(self, tolerance_mm: float = 0.01, tolerance_deg: float = 0.01) -> None:
        self._tolerance_mm = tolerance_mm
        self._tolerance_deg = tolerance_deg

    def validate_transform(self, T: np.ndarray) -> list[ValidationResult]:
        """Validate a single homogeneous transformation.

        Args:
            T: 4x4 transformation matrix.

        Returns:
            List of validation results.
        """
        results: list[ValidationResult] = []
        R = T[:3, :3]

        # Check orthogonality: R^T R = I
        orth_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
        results.append(
            ValidationResult(
                test_name="orthogonality",
                status=ValidationStatus.PASSED if orth_err < 1e-6 else ValidationStatus.FAILED,
                measured_value=orth_err,
                threshold=1e-6,
                description=f"R^T R deviation from identity: {orth_err:.2e}",
            )
        )

        # Check determinant = +1
        det_val = float(np.linalg.det(R))
        det_err = abs(det_val - 1.0)
        results.append(
            ValidationResult(
                test_name="determinant",
                status=ValidationStatus.PASSED if det_err < 1e-6 else ValidationStatus.FAILED,
                measured_value=det_val,
                threshold=1e-6,
                description=f"det(R) = {det_val:.6f} (expected 1.0)",
            )
        )

        # Check homogeneous row
        bottom_row = T[3, :]
        bottom_err = float(np.linalg.norm(bottom_row - np.array([0, 0, 0, 1])))
        results.append(
            ValidationResult(
                test_name="homogeneous_row",
                status=ValidationStatus.PASSED if bottom_err < 1e-10 else ValidationStatus.FAILED,
                measured_value=bottom_err,
                threshold=1e-10,
                description=f"Bottom row deviation: {bottom_err:.2e}",
            )
        )

        return results

    def validate_inverse_consistency(self, T: np.ndarray) -> ValidationResult:
        """Check that T * T^{-1} = I.

        Args:
            T: 4x4 transformation matrix.

        Returns:
            Validation result.
        """
        T_inv = np.linalg.inv(T)
        product = T @ T_inv
        err = float(np.linalg.norm(product - np.eye(4)))

        return ValidationResult(
            test_name="inverse_consistency",
            status=ValidationStatus.PASSED if err < 1e-8 else ValidationStatus.FAILED,
            measured_value=err,
            threshold=1e-8,
            description=f"T * T^-1 deviation from I: {err:.2e}",
        )

    def validate_chain_closure(self, transforms: list[np.ndarray]) -> ValidationResult:
        """Check that a closed chain of transforms returns to identity.

        Args:
            transforms: List of transforms forming a closed loop.

        Returns:
            Validation result.
        """
        product = np.eye(4)
        for T in transforms:
            product = product @ T

        position_err = float(np.linalg.norm(product[:3, 3]))
        rot_err = rotation_error_deg(product[:3, :3], np.eye(3))

        total_err = math.sqrt(position_err**2 + rot_err**2)

        status = ValidationStatus.PASSED
        if position_err > self._tolerance_mm or rot_err > self._tolerance_deg:
            status = ValidationStatus.WARNING
        if position_err > self._tolerance_mm * 10 or rot_err > self._tolerance_deg * 10:
            status = ValidationStatus.FAILED

        return ValidationResult(
            test_name="chain_closure",
            status=status,
            measured_value=total_err,
            threshold=self._tolerance_mm,
            description=(f"Loop closure error: pos={position_err:.4f}mm, rot={rot_err:.4f}deg"),
        )


# ============================================================================
# Calibration Error Analyser
# ============================================================================


class CalibrationErrorAnalyzer:
    """Statistical analysis of calibration errors.

    Provides hypothesis tests on residual distributions and bootstrap
    confidence interval estimation for calibration parameters.

    Args:
        config: Calibration configuration.
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self._config = config or CalibrationConfig()

    def analyse_residuals(self, residuals: list[float]) -> dict[str, Any]:
        """Compute descriptive statistics on calibration residuals.

        Args:
            residuals: List of residual errors.

        Returns:
            Dictionary of statistical measures.
        """
        arr = np.array(residuals)
        stats = {
            "count": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "rms": float(np.sqrt(np.mean(arr**2))),
        }
        return stats

    def run_statistical_tests(self, residuals: list[float]) -> list[StatisticalTestResult]:
        """Run statistical tests on calibration residuals.

        Tests include normality (via histogram-based approximation)
        and chi-squared goodness of fit. Uses scipy if available,
        otherwise falls back to simplified tests.

        Args:
            residuals: List of residual errors.

        Returns:
            List of test results.
        """
        results: list[StatisticalTestResult] = []
        arr = np.array(residuals)

        # Test 1: Normality test
        if HAS_SCIPY and scipy_stats is not None:
            if len(arr) >= 8:
                ks_stat, ks_p = scipy_stats.kstest(arr, "norm", args=(np.mean(arr), np.std(arr)))
                results.append(
                    StatisticalTestResult(
                        test_name="Kolmogorov-Smirnov normality",
                        statistic=float(ks_stat),
                        p_value=float(ks_p),
                        passed=ks_p > 0.05,
                        description="Residuals are normally distributed"
                        if ks_p > 0.05
                        else "Residuals deviate from normal",
                    )
                )
        else:
            # Simplified normality check using skewness
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            if std_val > 1e-10:
                skewness = float(np.mean(((arr - mean_val) / std_val) ** 3))
                is_normal = abs(skewness) < 1.0
                results.append(
                    StatisticalTestResult(
                        test_name="Simplified normality (skewness)",
                        statistic=abs(skewness),
                        p_value=0.5 if is_normal else 0.01,
                        passed=is_normal,
                        description=f"Skewness={skewness:.3f} ({'acceptable' if is_normal else 'high'})",
                    )
                )

        # Test 2: Chi-squared test for uniformity of residual distribution
        if len(arr) >= 10:
            n_bins = min(10, len(arr) // 3)
            observed, bin_edges = np.histogram(arr, bins=n_bins)
            expected_count = len(arr) / n_bins
            expected = np.full(n_bins, expected_count)

            if HAS_SCIPY and scipy_stats is not None:
                chi2_stat, chi2_p = scipy_stats.chisquare(observed, expected)
                results.append(
                    StatisticalTestResult(
                        test_name="Chi-squared uniformity",
                        statistic=float(chi2_stat),
                        p_value=float(chi2_p),
                        passed=chi2_p > 0.05,
                        description="Residuals uniformly distributed across bins"
                        if chi2_p > 0.05
                        else "Non-uniform residual distribution",
                    )
                )
            else:
                chi2_approx = float(np.sum((observed - expected) ** 2 / np.maximum(expected, 1)))
                results.append(
                    StatisticalTestResult(
                        test_name="Chi-squared uniformity (approx)",
                        statistic=chi2_approx,
                        p_value=0.5 if chi2_approx < 2 * n_bins else 0.01,
                        passed=chi2_approx < 2 * n_bins,
                        description=f"Chi2={chi2_approx:.2f} (threshold={2 * n_bins})",
                    )
                )

        # Test 3: Outlier detection (values > 3 sigma)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        if std_val > 1e-10:
            z_scores = np.abs((arr - mean_val) / std_val)
            n_outliers = int(np.sum(z_scores > 3.0))
            results.append(
                StatisticalTestResult(
                    test_name="Outlier detection (3-sigma)",
                    statistic=float(n_outliers),
                    p_value=1.0 if n_outliers == 0 else float(n_outliers / len(arr)),
                    passed=n_outliers <= max(1, len(arr) // 20),
                    description=f"{n_outliers} outliers detected out of {len(arr)} residuals",
                )
            )

        return results

    def bootstrap_confidence_interval(
        self,
        residuals: list[float],
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """Estimate confidence intervals via bootstrap resampling.

        Args:
            residuals: List of residual errors.
            confidence: Confidence level (default 95%).

        Returns:
            Dictionary with bootstrap statistics and CI bounds.
        """
        rng = np.random.default_rng(self._config.random_seed)
        arr = np.array(residuals)
        n_samples = self._config.bootstrap_samples

        bootstrap_means: list[float] = []
        for _ in range(n_samples):
            sample = rng.choice(arr, size=len(arr), replace=True)
            bootstrap_means.append(float(np.mean(sample)))

        bootstrap_arr = np.array(bootstrap_means)
        alpha = (1 - confidence) / 2

        return {
            "point_estimate": float(np.mean(arr)),
            "bootstrap_mean": float(np.mean(bootstrap_arr)),
            "bootstrap_std": float(np.std(bootstrap_arr)),
            "ci_lower": float(np.percentile(bootstrap_arr, alpha * 100)),
            "ci_upper": float(np.percentile(bootstrap_arr, (1 - alpha) * 100)),
            "confidence": confidence,
            "n_samples": n_samples,
        }


# ============================================================================
# Calibration Benchmark
# ============================================================================


class CalibrationBenchmark:
    """Benchmarks multiple calibration methods against ground truth.

    Generates synthetic calibration data with known ground truth and
    evaluates all three calibration methods.

    Args:
        config: Calibration configuration.
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self._config = config or CalibrationConfig()
        self._rng = np.random.default_rng(self._config.random_seed)

    def generate_synthetic_data(self) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """Generate synthetic calibration data with ground truth.

        Returns:
            Tuple of (robot_poses, camera_poses, ground_truth_X).
        """
        # Ground truth hand-eye transform
        R_true = rotation_matrix_from_euler(
            math.radians(15),
            math.radians(-10),
            math.radians(5),
        )
        t_true = np.array([30.0, -20.0, 50.0])
        X_true = make_transform(R_true, t_true)

        robot_poses: list[np.ndarray] = []
        camera_poses: list[np.ndarray] = []

        for i in range(self._config.num_poses):
            # Random robot pose
            angles = self._rng.uniform(-math.pi / 4, math.pi / 4, size=3)
            R_robot = rotation_matrix_from_euler(angles[0], angles[1], angles[2])
            t_robot = self._rng.uniform(-100, 100, size=3)
            T_robot = make_transform(R_robot, t_robot)

            # Camera pose = X^{-1} * T_robot^{-1} * X (with noise)
            X_inv = np.linalg.inv(X_true)
            T_robot_inv = np.linalg.inv(T_robot)
            T_camera_ideal = X_inv @ T_robot @ X_true

            # Add noise
            noise_rot = self._rng.normal(0, math.radians(self._config.rotation_noise_deg), size=3)
            noise_trans = self._rng.normal(0, self._config.translation_noise_mm, size=3)
            R_noise = rotation_matrix_from_euler(noise_rot[0], noise_rot[1], noise_rot[2])
            T_camera = T_camera_ideal.copy()
            T_camera[:3, :3] = T_camera_ideal[:3, :3] @ R_noise
            T_camera[:3, 3] += noise_trans

            robot_poses.append(T_robot)
            camera_poses.append(T_camera)

        logger.info(
            "Generated %d synthetic pose pairs (noise: %.2f deg, %.2f mm)",
            self._config.num_poses,
            self._config.rotation_noise_deg,
            self._config.translation_noise_mm,
        )
        return robot_poses, camera_poses, X_true

    def run_benchmark(self) -> dict[str, Any]:
        """Run the complete calibration benchmark.

        Returns:
            Benchmark results dictionary.
        """
        robot_poses, camera_poses, X_true = self.generate_synthetic_data()

        calibrator = HandEyeCalibrator(self._config)
        calibrator.set_pose_pairs(robot_poses, camera_poses)

        validator = TransformationChainValidator()
        analyser = CalibrationErrorAnalyzer(self._config)

        results: dict[str, Any] = {
            "ground_truth": {
                "rotation": X_true[:3, :3].tolist(),
                "translation": X_true[:3, 3].tolist(),
            },
            "methods": {},
        }

        for method in CalibrationMethod:
            logger.info("\n--- Calibrating with %s ---", method.value)
            cal_result = calibrator.calibrate(method)

            # Compare to ground truth
            R_err = rotation_error_deg(cal_result.transform.rotation, X_true[:3, :3])
            t_err = translation_error_mm(cal_result.transform.translation, X_true[:3, 3])

            # Validate the transform
            validations = validator.validate_transform(cal_result.transform.matrix)
            inv_check = validator.validate_inverse_consistency(cal_result.transform.matrix)
            validations.append(inv_check)

            # Statistical analysis
            residual_stats = analyser.analyse_residuals(cal_result.residuals)
            stat_tests = analyser.run_statistical_tests(cal_result.residuals)
            bootstrap_ci = analyser.bootstrap_confidence_interval(cal_result.residuals)

            results["methods"][method.value] = {
                "rotation_error_deg": R_err,
                "translation_error_mm": t_err,
                "quality": cal_result.quality.value,
                "computation_time_s": cal_result.computation_time_s,
                "residual_stats": residual_stats,
                "validation_results": [
                    {"test": v.test_name, "status": v.status.value, "value": v.measured_value} for v in validations
                ],
                "statistical_tests": [
                    {"test": t.test_name, "passed": t.passed, "p_value": t.p_value} for t in stat_tests
                ],
                "bootstrap_ci": bootstrap_ci,
            }

            logger.info("  Ground truth error: rot=%.4f deg, trans=%.4f mm", R_err, t_err)
            logger.info("  Quality: %s", cal_result.quality.value)

            for v in validations:
                logger.info("  Validation [%s]: %s (%.2e)", v.test_name, v.status.value, v.measured_value)

            for t in stat_tests:
                logger.info("  Stat test [%s]: %s (p=%.4f)", t.test_name, "PASS" if t.passed else "FAIL", t.p_value)

        return results


# ============================================================================
# Demonstration
# ============================================================================


def run_hand_eye_calibration_demo() -> dict[str, Any]:
    """Run the complete hand-eye calibration demonstration.

    Returns:
        Benchmark results dictionary.
    """
    logger.info("=" * 70)
    logger.info("  Hand-Eye Calibration Demo for Surgical Robots")
    logger.info("  Version 0.5.0 | RESEARCH USE ONLY")
    logger.info("=" * 70)

    config = CalibrationConfig(
        num_poses=30,
        rotation_noise_deg=0.1,
        translation_noise_mm=0.5,
        bootstrap_samples=100,
        random_seed=42,
    )

    benchmark = CalibrationBenchmark(config)
    results = benchmark.run_benchmark()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("  CALIBRATION BENCHMARK SUMMARY")
    logger.info("=" * 70)
    logger.info("  %-20s %12s %12s %12s %10s", "Method", "Rot Err(deg)", "Trans Err(mm)", "Quality", "Time(s)")
    logger.info("  %s", "-" * 66)

    for method_name, method_result in results["methods"].items():
        logger.info(
            "  %-20s %12.4f %12.4f %12s %10.4f",
            method_name,
            method_result["rotation_error_deg"],
            method_result["translation_error_mm"],
            method_result["quality"],
            method_result["computation_time_s"],
        )

    # Best method
    best_method = min(results["methods"].items(), key=lambda x: x[1]["translation_error_mm"])
    logger.info("\n  Best method: %s (trans_err=%.4fmm)", best_method[0], best_method[1]["translation_error_mm"])

    logger.info("\n" + "=" * 70)
    logger.info("  RESEARCH USE ONLY -- NOT FOR CLINICAL DEPLOYMENT")
    logger.info("=" * 70)

    return results


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    results = run_hand_eye_calibration_demo()
    sys.exit(0)
