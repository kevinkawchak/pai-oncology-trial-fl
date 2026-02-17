"""
Verify installation of PAI Oncology Trial FL platform dependencies.

Checks for required and optional framework availability, validates version
compatibility, and reports system readiness for federated learning workflows
in oncology clinical trials.

DISCLAIMER: RESEARCH USE ONLY — Not approved for clinical deployment
without independent validation and regulatory review.

VERSION: 0.3.0
LICENSE: MIT
"""

from __future__ import annotations

import importlib
import logging
import platform
import struct
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Color output utilities
# ─────────────────────────────────────────────────────────────────────────────


def _supports_color() -> bool:
    """Check if the terminal supports ANSI color codes."""
    if sys.platform == "win32":
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_USE_COLOR = _supports_color()


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m" if _USE_COLOR else text


def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m" if _USE_COLOR else text


def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m" if _USE_COLOR else text


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _USE_COLOR else text


def _cyan(text: str) -> str:
    return f"\033[96m{text}\033[0m" if _USE_COLOR else text


# ─────────────────────────────────────────────────────────────────────────────
# Package definitions
# ─────────────────────────────────────────────────────────────────────────────

CORE_PACKAGES = [
    {"name": "numpy", "import_name": "numpy", "min_version": "1.24.0", "required": True},
    {"name": "scipy", "import_name": "scipy", "min_version": "1.11.0", "required": True},
    {"name": "scikit-learn", "import_name": "sklearn", "min_version": "1.3.0", "required": True},
    {"name": "pandas", "import_name": "pandas", "min_version": "2.0.0", "required": True},
    {"name": "cryptography", "import_name": "cryptography", "min_version": "41.0.0", "required": True},
    {"name": "PyYAML", "import_name": "yaml", "min_version": "6.0", "required": True},
]

DEEP_LEARNING_PACKAGES = [
    {"name": "PyTorch", "import_name": "torch", "min_version": "2.5.0", "required": False},
    {"name": "torchvision", "import_name": "torchvision", "min_version": "0.20.0", "required": False},
    {"name": "ONNX", "import_name": "onnx", "min_version": "1.15.0", "required": False},
    {"name": "ONNX Runtime", "import_name": "onnxruntime", "min_version": "1.17.0", "required": False},
]

MEDICAL_IMAGING_PACKAGES = [
    {"name": "MONAI", "import_name": "monai", "min_version": "1.3.0", "required": False},
    {"name": "pydicom", "import_name": "pydicom", "min_version": "2.4.0", "required": False},
    {"name": "nibabel", "import_name": "nibabel", "min_version": "5.2.0", "required": False},
]

SIMULATION_PACKAGES = [
    {"name": "Isaac Sim", "import_name": "isaacsim", "min_version": "4.2.0", "required": False},
    {"name": "MuJoCo", "import_name": "mujoco", "min_version": "3.1.0", "required": False},
    {"name": "PyBullet", "import_name": "pybullet", "min_version": "3.2.6", "required": False},
]

AGENTIC_AI_PACKAGES = [
    {"name": "LangChain", "import_name": "langchain", "min_version": "0.3.0", "required": False},
    {"name": "LangGraph", "import_name": "langgraph", "min_version": "0.2.0", "required": False},
    {"name": "CrewAI", "import_name": "crewai", "min_version": "0.80.0", "required": False},
    {"name": "Anthropic SDK", "import_name": "anthropic", "min_version": "0.39.0", "required": False},
    {"name": "OpenAI SDK", "import_name": "openai", "min_version": "1.50.0", "required": False},
]

VISUALIZATION_PACKAGES = [
    {"name": "matplotlib", "import_name": "matplotlib", "min_version": "3.8.0", "required": False},
    {"name": "plotly", "import_name": "plotly", "min_version": "5.18.0", "required": False},
]

TESTING_PACKAGES = [
    {"name": "pytest", "import_name": "pytest", "min_version": "7.4.0", "required": False},
    {"name": "ruff", "import_name": "ruff", "min_version": "0.4.0", "required": False},
    {"name": "mypy", "import_name": "mypy", "min_version": "1.8.0", "required": False},
]

ALL_PACKAGE_GROUPS = [
    ("Core Dependencies", CORE_PACKAGES),
    ("Deep Learning", DEEP_LEARNING_PACKAGES),
    ("Medical Imaging", MEDICAL_IMAGING_PACKAGES),
    ("Simulation Frameworks", SIMULATION_PACKAGES),
    ("Agentic AI", AGENTIC_AI_PACKAGES),
    ("Visualization", VISUALIZATION_PACKAGES),
    ("Testing & Development", TESTING_PACKAGES),
]


# ─────────────────────────────────────────────────────────────────────────────
# Version comparison
# ─────────────────────────────────────────────────────────────────────────────


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of integers for comparison."""
    parts = []
    for part in version_str.split("."):
        cleaned = ""
        for ch in part:
            if ch.isdigit():
                cleaned += ch
            else:
                break
        if cleaned:
            parts.append(int(cleaned))
    return tuple(parts) if parts else (0,)


def _version_at_least(current: str, minimum: str) -> bool:
    """Check if current version meets the minimum requirement."""
    return _parse_version(current) >= _parse_version(minimum)


# ─────────────────────────────────────────────────────────────────────────────
# Package checking
# ─────────────────────────────────────────────────────────────────────────────


def _check_package(pkg: dict) -> dict:
    """
    Check availability and version of a single package.

    Returns a dict with keys: name, available, version, meets_minimum, required, error.
    """
    result = {
        "name": pkg["name"],
        "import_name": pkg["import_name"],
        "min_version": pkg["min_version"],
        "required": pkg["required"],
        "available": False,
        "version": None,
        "meets_minimum": False,
        "error": None,
    }

    try:
        mod = importlib.import_module(pkg["import_name"])
        result["available"] = True

        # Try common version attributes
        version = None
        for attr in ("__version__", "VERSION", "version"):
            if hasattr(mod, attr):
                val = getattr(mod, attr)
                if isinstance(val, str):
                    version = val
                    break

        if version is None:
            # Try importlib.metadata as fallback
            try:
                import importlib.metadata

                version = importlib.metadata.version(pkg["import_name"])
            except Exception:
                pass

        result["version"] = version

        if version and pkg["min_version"]:
            result["meets_minimum"] = _version_at_least(version, pkg["min_version"])
        elif version:
            result["meets_minimum"] = True

    except ImportError as exc:
        result["error"] = str(exc)
    except Exception as exc:
        result["error"] = f"Unexpected error: {exc}"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# System information
# ─────────────────────────────────────────────────────────────────────────────


def _print_system_info() -> None:
    """Print system information relevant to the platform."""
    print(_bold("\n╔══════════════════════════════════════════════════════════════╗"))
    print(_bold("║   PAI Oncology Trial FL — Installation Verification         ║"))
    print(_bold("╚══════════════════════════════════════════════════════════════╝"))
    print()
    print(_bold("System Information:"))
    print(f"  Python:        {sys.version.split()[0]}")
    print(f"  Platform:      {platform.platform()}")
    print(f"  Architecture:  {platform.machine()}")
    print(f"  Pointer Size:  {struct.calcsize('P') * 8}-bit")
    print(f"  OS:            {platform.system()} {platform.release()}")
    print()

    # Check for GPU
    gpu_info = "Not detected"
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info = f"CUDA {torch.version.cuda} — {torch.cuda.get_device_name(0)}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info = "Apple MPS"
    except ImportError:
        pass
    print(f"  GPU:           {gpu_info}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Verification reporting
# ─────────────────────────────────────────────────────────────────────────────


def _print_group_results(group_name: str, results: list[dict]) -> tuple[int, int, int]:
    """Print results for a package group. Returns (passed, warned, failed) counts."""
    print(_bold(f"  {group_name}:"))

    passed = 0
    warned = 0
    failed = 0

    for r in results:
        if r["available"] and r["meets_minimum"]:
            status = _green("✓ PASS")
            version_info = f"v{r['version']}" if r["version"] else "installed"
            print(f"    {status}  {r['name']:20s} {version_info} (>= {r['min_version']})")
            passed += 1
        elif r["available"] and not r["meets_minimum"]:
            status = _yellow("⚠ WARN")
            version_info = f"v{r['version']}" if r["version"] else "unknown version"
            print(f"    {status}  {r['name']:20s} {version_info} (needs >= {r['min_version']})")
            warned += 1
        elif r["required"]:
            status = _red("✗ FAIL")
            print(f"    {status}  {r['name']:20s} NOT FOUND (required, >= {r['min_version']})")
            failed += 1
        else:
            status = _yellow("- SKIP")
            print(f"    {status}  {r['name']:20s} not installed (optional)")
            warned += 1

    print()
    return passed, warned, failed


def _print_internal_modules() -> None:
    """Check internal package modules are importable."""
    print(_bold("  Internal Modules:"))

    modules = [
        ("federated", "Federated Learning"),
        ("physical_ai", "Physical AI"),
        ("privacy", "Privacy"),
        ("regulatory", "Regulatory"),
    ]

    for module_name, display_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"    {_green('✓ PASS')}  {display_name:20s} ({module_name}/)")
        except ImportError:
            print(f"    {_yellow('⚠ WARN')}  {display_name:20s} ({module_name}/ — not on sys.path)")
        except Exception as exc:
            print(f"    {_red('✗ FAIL')}  {display_name:20s} ({module_name}/ — {exc})")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    """
    Run installation verification and return exit code.

    Returns:
        0 if all required dependencies are satisfied, 1 otherwise.
    """
    _print_system_info()

    total_passed = 0
    total_warned = 0
    total_failed = 0

    print(_bold("Dependency Verification:"))
    print()

    for group_name, packages in ALL_PACKAGE_GROUPS:
        results = [_check_package(pkg) for pkg in packages]
        p, w, f = _print_group_results(group_name, results)
        total_passed += p
        total_warned += w
        total_failed += f

    _print_internal_modules()

    # Summary
    print(_bold("─" * 62))
    total = total_passed + total_warned + total_failed
    print(f"  Total:   {total} packages checked")
    print(f"  Passed:  {_green(str(total_passed))}")
    print(f"  Warned:  {_yellow(str(total_warned))} (optional or version mismatch)")
    print(f"  Failed:  {_red(str(total_failed))} (required, missing)")
    print()

    if total_failed == 0:
        print(_green(_bold("  ✓ All required dependencies satisfied.")))
        print(_cyan("  Platform is ready for federated learning workflows."))
        logger.info("Installation verification passed: %d/%d packages available", total_passed, total)
        return 0
    else:
        print(_red(_bold(f"  ✗ {total_failed} required dependency(ies) missing.")))
        print("  Install missing packages with: pip install -r requirements.txt")
        logger.warning("Installation verification failed: %d required packages missing", total_failed)
        return 1


if __name__ == "__main__":
    sys.exit(main())
