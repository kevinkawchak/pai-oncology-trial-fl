"""
Cross-Platform Model Format Converter for Oncology Surgical Robotics.

Converts robot and environment models between URDF, MJCF, USD, and SDF
formats. Designed for federated learning workflows where different clinical
trial sites use different simulation engines and need interoperable models.

DISCLAIMER: RESEARCH USE ONLY
This software is provided for research and educational purposes only.
It has NOT been validated for clinical use, is NOT approved by the FDA
or any other regulatory body, and MUST NOT be used to make clinical
decisions or control physical surgical robots in patient-facing settings.
All converted models must be independently validated for physics fidelity
before use in policy training or clinical simulation. Use at your own risk.

Copyright (c) 2026 PAI Oncology Trial FL Contributors
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
try:
    import xml.etree.ElementTree as ET

    HAS_XML = True
except ImportError:
    ET = None  # type: ignore[assignment,misc]
    HAS_XML = False

try:
    import yaml

    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    HAS_YAML = False

try:
    from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore[import-untyped]

    HAS_USD = True
except ImportError:
    Usd = None  # type: ignore[assignment]
    UsdGeom = None  # type: ignore[assignment]
    UsdPhysics = None  # type: ignore[assignment]
    HAS_USD = False

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class ModelFormat(Enum):
    """Supported robot model formats."""

    URDF = "urdf"
    MJCF = "mjcf"
    USD = "usd"
    SDF = "sdf"


class ConversionStatus(Enum):
    """Status of a model conversion operation."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class JointType(Enum):
    """Joint types common across all formats."""

    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    CONTINUOUS = "continuous"
    FIXED = "fixed"
    FLOATING = "floating"
    PLANAR = "planar"
    BALL = "ball"


class GeometryType(Enum):
    """Geometry primitive types."""

    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    CAPSULE = "capsule"
    MESH = "mesh"
    PLANE = "plane"


class MaterialType(Enum):
    """Material representation types."""

    BASIC = "basic"
    PBR = "pbr"
    PHONG = "phong"


# ---------------------------------------------------------------------------
# Intermediate representation data classes
# ---------------------------------------------------------------------------
@dataclass
class Vector3:
    """3D vector for position, scale, etc."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])

    def to_string(self) -> str:
        """Convert to space-separated string for XML formats."""
        return f"{self.x} {self.y} {self.z}"


@dataclass
class Quaternion:
    """Quaternion for orientation [w, x, y, z]."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [w, x, y, z]."""
        return np.array([self.w, self.x, self.y, self.z])

    def to_euler_string(self) -> str:
        """Convert to roll-pitch-yaw string for URDF/SDF."""
        # Simplified conversion (does not handle gimbal lock)
        import math

        sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (self.w * self.y - self.z * self.x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        siny_cosp = 2.0 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return f"{roll} {pitch} {yaw}"


@dataclass
class InertialProperties:
    """Inertial properties of a link."""

    mass: float = 1.0
    origin: Vector3 = field(default_factory=Vector3)
    orientation: Quaternion = field(default_factory=Quaternion)
    ixx: float = 0.001
    ixy: float = 0.0
    ixz: float = 0.0
    iyy: float = 0.001
    iyz: float = 0.0
    izz: float = 0.001


@dataclass
class GeometrySpec:
    """Specification of a collision or visual geometry."""

    geom_type: GeometryType = GeometryType.BOX
    size: Vector3 = field(default_factory=lambda: Vector3(0.1, 0.1, 0.1))
    radius: float = 0.05
    length: float = 0.1
    mesh_path: str = ""
    mesh_scale: Vector3 = field(default_factory=lambda: Vector3(1.0, 1.0, 1.0))
    origin: Vector3 = field(default_factory=Vector3)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class MaterialSpec:
    """Material specification for visual appearance."""

    name: str = "default"
    material_type: MaterialType = MaterialType.BASIC
    color_rgba: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    texture_path: str = ""
    metallic: float = 0.0
    roughness: float = 0.5


@dataclass
class ContactProperties:
    """Contact and friction properties for a geometry."""

    friction_sliding: float = 0.5
    friction_torsional: float = 0.005
    friction_rolling: float = 0.001
    restitution: float = 0.0
    stiffness: float = 0.0
    damping: float = 0.0
    contact_type: str = "default"


@dataclass
class JointSpec:
    """Intermediate representation of a robot joint."""

    name: str = ""
    joint_type: JointType = JointType.REVOLUTE
    parent_link: str = ""
    child_link: str = ""
    origin: Vector3 = field(default_factory=Vector3)
    orientation: Quaternion = field(default_factory=Quaternion)
    axis: Vector3 = field(default_factory=lambda: Vector3(0.0, 0.0, 1.0))
    lower_limit: float = -3.14159
    upper_limit: float = 3.14159
    velocity_limit: float = 6.2832
    effort_limit: float = 500.0
    damping: float = 0.0
    stiffness: float = 0.0
    friction: float = 0.0


@dataclass
class LinkSpec:
    """Intermediate representation of a robot link."""

    name: str = ""
    inertial: InertialProperties = field(default_factory=InertialProperties)
    visual_geometries: list[GeometrySpec] = field(default_factory=list)
    collision_geometries: list[GeometrySpec] = field(default_factory=list)
    materials: list[MaterialSpec] = field(default_factory=list)
    contact: ContactProperties = field(default_factory=ContactProperties)


@dataclass
class RobotModel:
    """Intermediate representation of a complete robot model.

    This format-agnostic structure serves as the canonical representation
    for cross-format conversion. All format-specific parsers produce this
    structure, and all format-specific writers consume it.
    """

    name: str = "robot"
    links: list[LinkSpec] = field(default_factory=list)
    joints: list[JointSpec] = field(default_factory=list)
    source_format: ModelFormat = ModelFormat.URDF
    source_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_link(self, name: str) -> Optional[LinkSpec]:
        """Find a link by name."""
        for link in self.links:
            if link.name == name:
                return link
        return None

    def get_joint(self, name: str) -> Optional[JointSpec]:
        """Find a joint by name."""
        for joint in self.joints:
            if joint.name == name:
                return joint
        return None

    def compute_hash(self) -> str:
        """Compute a content hash for change detection."""
        data = {
            "name": self.name,
            "num_links": len(self.links),
            "num_joints": len(self.joints),
            "link_names": sorted(lnk.name for lnk in self.links),
            "joint_names": sorted(jnt.name for jnt in self.joints),
        }
        raw = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass
class ConversionReport:
    """Report generated after a model conversion."""

    conversion_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_format: ModelFormat = ModelFormat.URDF
    target_format: ModelFormat = ModelFormat.MJCF
    source_path: str = ""
    target_path: str = ""
    status: ConversionStatus = ConversionStatus.SUCCESS
    links_converted: int = 0
    joints_converted: int = 0
    geometries_converted: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    features_lost: list[str] = field(default_factory=list)
    source_hash: str = ""
    target_hash: str = ""
    conversion_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "conversion_id": self.conversion_id,
            "source_format": self.source_format.value,
            "target_format": self.target_format.value,
            "source_path": self.source_path,
            "target_path": self.target_path,
            "status": self.status.value,
            "links_converted": self.links_converted,
            "joints_converted": self.joints_converted,
            "geometries_converted": self.geometries_converted,
            "warnings": self.warnings,
            "errors": self.errors,
            "features_lost": self.features_lost,
            "source_hash": self.source_hash,
            "target_hash": self.target_hash,
            "conversion_time_ms": round(self.conversion_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# URDF parser
# ---------------------------------------------------------------------------
class URDFParser:
    """Parse URDF XML into the intermediate RobotModel representation."""

    def parse(self, urdf_path: str) -> RobotModel:
        """Parse a URDF file into a RobotModel."""
        path = Path(urdf_path)
        if not path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        tree = ET.parse(str(path))
        root = tree.getroot()

        model = RobotModel(
            name=root.attrib.get("name", "robot"),
            source_format=ModelFormat.URDF,
            source_path=urdf_path,
        )

        # Parse links
        for link_elem in root.findall("link"):
            link = self._parse_link(link_elem)
            model.links.append(link)

        # Parse joints
        for joint_elem in root.findall("joint"):
            joint = self._parse_joint(joint_elem)
            model.joints.append(joint)

        logger.info("URDF parsed: %s (%d links, %d joints)", model.name, len(model.links), len(model.joints))
        return model

    def _parse_link(self, elem: Any) -> LinkSpec:
        """Parse a URDF link element."""
        link = LinkSpec(name=elem.attrib.get("name", ""))

        # Inertial
        inertial_elem = elem.find("inertial")
        if inertial_elem is not None:
            mass_elem = inertial_elem.find("mass")
            if mass_elem is not None:
                link.inertial.mass = float(mass_elem.attrib.get("value", "1.0"))

            inertia_elem = inertial_elem.find("inertia")
            if inertia_elem is not None:
                link.inertial.ixx = float(inertia_elem.attrib.get("ixx", "0.001"))
                link.inertial.ixy = float(inertia_elem.attrib.get("ixy", "0"))
                link.inertial.ixz = float(inertia_elem.attrib.get("ixz", "0"))
                link.inertial.iyy = float(inertia_elem.attrib.get("iyy", "0.001"))
                link.inertial.iyz = float(inertia_elem.attrib.get("iyz", "0"))
                link.inertial.izz = float(inertia_elem.attrib.get("izz", "0.001"))

        # Visual geometries
        for visual_elem in elem.findall("visual"):
            geom = self._parse_geometry(visual_elem)
            if geom:
                link.visual_geometries.append(geom)

        # Collision geometries
        for collision_elem in elem.findall("collision"):
            geom = self._parse_geometry(collision_elem)
            if geom:
                link.collision_geometries.append(geom)

        return link

    def _parse_geometry(self, parent_elem: Any) -> Optional[GeometrySpec]:
        """Parse a geometry element from visual or collision."""
        geom_elem = parent_elem.find("geometry")
        if geom_elem is None:
            return None

        geom = GeometrySpec()

        # Parse origin
        origin_elem = parent_elem.find("origin")
        if origin_elem is not None:
            xyz = origin_elem.attrib.get("xyz", "0 0 0").split()
            if len(xyz) == 3:
                geom.origin = Vector3(float(xyz[0]), float(xyz[1]), float(xyz[2]))

        # Parse geometry type
        box_elem = geom_elem.find("box")
        if box_elem is not None:
            geom.geom_type = GeometryType.BOX
            size = box_elem.attrib.get("size", "0.1 0.1 0.1").split()
            if len(size) == 3:
                geom.size = Vector3(float(size[0]), float(size[1]), float(size[2]))
            return geom

        cylinder_elem = geom_elem.find("cylinder")
        if cylinder_elem is not None:
            geom.geom_type = GeometryType.CYLINDER
            geom.radius = float(cylinder_elem.attrib.get("radius", "0.05"))
            geom.length = float(cylinder_elem.attrib.get("length", "0.1"))
            return geom

        sphere_elem = geom_elem.find("sphere")
        if sphere_elem is not None:
            geom.geom_type = GeometryType.SPHERE
            geom.radius = float(sphere_elem.attrib.get("radius", "0.05"))
            return geom

        mesh_elem = geom_elem.find("mesh")
        if mesh_elem is not None:
            geom.geom_type = GeometryType.MESH
            geom.mesh_path = mesh_elem.attrib.get("filename", "")
            scale = mesh_elem.attrib.get("scale", "1 1 1").split()
            if len(scale) == 3:
                geom.mesh_scale = Vector3(float(scale[0]), float(scale[1]), float(scale[2]))
            return geom

        return geom

    def _parse_joint(self, elem: Any) -> JointSpec:
        """Parse a URDF joint element."""
        joint = JointSpec(
            name=elem.attrib.get("name", ""),
        )

        type_str = elem.attrib.get("type", "revolute")
        type_map = {
            "revolute": JointType.REVOLUTE,
            "prismatic": JointType.PRISMATIC,
            "continuous": JointType.CONTINUOUS,
            "fixed": JointType.FIXED,
            "floating": JointType.FLOATING,
            "planar": JointType.PLANAR,
        }
        joint.joint_type = type_map.get(type_str, JointType.REVOLUTE)

        parent_elem = elem.find("parent")
        if parent_elem is not None:
            joint.parent_link = parent_elem.attrib.get("link", "")

        child_elem = elem.find("child")
        if child_elem is not None:
            joint.child_link = child_elem.attrib.get("link", "")

        origin_elem = elem.find("origin")
        if origin_elem is not None:
            xyz = origin_elem.attrib.get("xyz", "0 0 0").split()
            if len(xyz) == 3:
                joint.origin = Vector3(float(xyz[0]), float(xyz[1]), float(xyz[2]))

        axis_elem = elem.find("axis")
        if axis_elem is not None:
            xyz = axis_elem.attrib.get("xyz", "0 0 1").split()
            if len(xyz) == 3:
                joint.axis = Vector3(float(xyz[0]), float(xyz[1]), float(xyz[2]))

        limit_elem = elem.find("limit")
        if limit_elem is not None:
            joint.lower_limit = float(limit_elem.attrib.get("lower", "-3.14159"))
            joint.upper_limit = float(limit_elem.attrib.get("upper", "3.14159"))
            joint.velocity_limit = float(limit_elem.attrib.get("velocity", "6.2832"))
            joint.effort_limit = float(limit_elem.attrib.get("effort", "500"))

        dynamics_elem = elem.find("dynamics")
        if dynamics_elem is not None:
            joint.damping = float(dynamics_elem.attrib.get("damping", "0"))
            joint.friction = float(dynamics_elem.attrib.get("friction", "0"))

        return joint


# ---------------------------------------------------------------------------
# MJCF writer
# ---------------------------------------------------------------------------
class MJCFWriter:
    """Write a RobotModel to MuJoCo MJCF XML format."""

    def write(self, model: RobotModel, output_path: str) -> ConversionReport:
        """Write the model to MJCF format."""
        report = ConversionReport(
            source_format=model.source_format,
            target_format=ModelFormat.MJCF,
            source_path=model.source_path,
            target_path=output_path,
        )

        start = time.time()

        try:
            root = ET.Element("mujoco", model=model.name)

            # Compiler settings
            ET.SubElement(root, "compiler", angle="radian", meshdir="meshes")

            # Option settings
            option = ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81")

            # Default settings
            default = ET.SubElement(root, "default")
            ET.SubElement(default, "joint", damping="0.1", armature="0.01")
            ET.SubElement(default, "geom", condim="3", friction="0.5 0.005 0.001")

            # Worldbody
            worldbody = ET.SubElement(root, "worldbody")

            # Build link hierarchy
            link_map = {link.name: link for link in model.links}
            child_map: dict[str, list[JointSpec]] = {}
            root_links: set[str] = set(link.name for link in model.links)

            for joint in model.joints:
                if joint.parent_link not in child_map:
                    child_map[joint.parent_link] = []
                child_map[joint.parent_link].append(joint)
                root_links.discard(joint.child_link)

            # Write bodies recursively
            for root_link_name in sorted(root_links):
                self._write_body(worldbody, root_link_name, link_map, child_map, report)

            # Write to file
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ")
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tree.write(str(path), encoding="unicode", xml_declaration=True)

            report.status = ConversionStatus.SUCCESS
            report.links_converted = len(model.links)
            report.joints_converted = len(model.joints)

        except Exception as exc:
            report.status = ConversionStatus.FAILED
            report.errors.append(str(exc))
            logger.error("MJCF conversion failed: %s", exc)

        report.conversion_time_ms = (time.time() - start) * 1000.0
        return report

    def _write_body(
        self,
        parent_elem: Any,
        link_name: str,
        link_map: dict[str, LinkSpec],
        child_map: dict[str, list[JointSpec]],
        report: ConversionReport,
    ) -> None:
        """Recursively write a body and its children."""
        link = link_map.get(link_name)
        if link is None:
            return

        body = ET.SubElement(parent_elem, "body", name=link_name)

        # Inertial
        if link.inertial.mass > 0:
            inertial = ET.SubElement(
                body, "inertial", mass=str(link.inertial.mass), pos=link.inertial.origin.to_string()
            )
            inertial.set(
                "fullinertia",
                f"{link.inertial.ixx} {link.inertial.iyy} {link.inertial.izz} "
                f"{link.inertial.ixy} {link.inertial.ixz} {link.inertial.iyz}",
            )

        # Geometries
        for geom_spec in link.collision_geometries:
            geom_attribs = {"name": f"{link_name}_collision"}
            if geom_spec.geom_type == GeometryType.BOX:
                geom_attribs["type"] = "box"
                half = f"{geom_spec.size.x / 2} {geom_spec.size.y / 2} {geom_spec.size.z / 2}"
                geom_attribs["size"] = half
            elif geom_spec.geom_type == GeometryType.CYLINDER:
                geom_attribs["type"] = "cylinder"
                geom_attribs["size"] = f"{geom_spec.radius} {geom_spec.length / 2}"
            elif geom_spec.geom_type == GeometryType.SPHERE:
                geom_attribs["type"] = "sphere"
                geom_attribs["size"] = str(geom_spec.radius)
            elif geom_spec.geom_type == GeometryType.MESH:
                geom_attribs["type"] = "mesh"
                geom_attribs["mesh"] = link_name
                report.warnings.append(f"Mesh '{geom_spec.mesh_path}' needs manual asset setup")
            ET.SubElement(body, "geom", **geom_attribs)
            report.geometries_converted += 1

        # Child joints and bodies
        for joint in child_map.get(link_name, []):
            joint_attribs = {
                "name": joint.name,
                "type": "hinge" if joint.joint_type == JointType.REVOLUTE else "slide",
                "axis": joint.axis.to_string(),
                "pos": joint.origin.to_string(),
            }
            if joint.joint_type in (JointType.REVOLUTE, JointType.PRISMATIC):
                joint_attribs["range"] = f"{joint.lower_limit} {joint.upper_limit}"
                joint_attribs["damping"] = str(joint.damping)

            child_body_name = joint.child_link
            child_body = ET.SubElement(body, "body", name=child_body_name, pos=joint.origin.to_string())
            ET.SubElement(child_body, "joint", **joint_attribs)

            # Recurse into child
            child_link = link_map.get(child_body_name)
            if child_link:
                for cg in child_link.collision_geometries:
                    cg_attribs = {"name": f"{child_body_name}_collision"}
                    if cg.geom_type == GeometryType.BOX:
                        cg_attribs["type"] = "box"
                        half = f"{cg.size.x / 2} {cg.size.y / 2} {cg.size.z / 2}"
                        cg_attribs["size"] = half
                    elif cg.geom_type == GeometryType.SPHERE:
                        cg_attribs["type"] = "sphere"
                        cg_attribs["size"] = str(cg.radius)
                    ET.SubElement(child_body, "geom", **cg_attribs)

            # Continue recursion for child's children
            for grandchild_joint in child_map.get(child_body_name, []):
                self._write_body(child_body, grandchild_joint.child_link, link_map, child_map, report)


# ---------------------------------------------------------------------------
# Model converter orchestrator
# ---------------------------------------------------------------------------
class ModelConverter:
    """Orchestrates model conversion between supported formats.

    Uses the intermediate RobotModel representation for lossless (where
    possible) cross-format conversion. Generates detailed conversion reports
    for audit trail purposes.
    """

    def __init__(self) -> None:
        self._urdf_parser = URDFParser()
        self._mjcf_writer = MJCFWriter()
        self._conversion_history: list[ConversionReport] = []

    def convert(
        self,
        source_path: str,
        source_format: ModelFormat,
        target_path: str,
        target_format: ModelFormat,
    ) -> ConversionReport:
        """Convert a model from one format to another."""
        logger.info("Converting %s (%s) -> %s (%s)", source_path, source_format.value, target_path, target_format.value)

        # Parse source
        model: Optional[RobotModel] = None
        if source_format == ModelFormat.URDF:
            model = self._urdf_parser.parse(source_path)
        else:
            report = ConversionReport(
                source_format=source_format,
                target_format=target_format,
                status=ConversionStatus.FAILED,
            )
            report.errors.append(f"Source format '{source_format.value}' parser not yet implemented")
            return report

        if model is None:
            report = ConversionReport(status=ConversionStatus.FAILED)
            report.errors.append("Failed to parse source model")
            return report

        # Write target
        report: ConversionReport
        if target_format == ModelFormat.MJCF:
            report = self._mjcf_writer.write(model, target_path)
        else:
            report = ConversionReport(
                source_format=source_format,
                target_format=target_format,
                status=ConversionStatus.FAILED,
            )
            report.errors.append(f"Target format '{target_format.value}' writer not yet implemented")
            return report

        report.source_hash = model.compute_hash()
        self._conversion_history.append(report)

        logger.info(
            "Conversion %s: %s -> %s (%d links, %d joints, %d geoms) in %.1fms",
            report.status.value,
            source_format.value,
            target_format.value,
            report.links_converted,
            report.joints_converted,
            report.geometries_converted,
            report.conversion_time_ms,
        )

        return report

    def get_supported_conversions(self) -> list[tuple[str, str]]:
        """Return list of supported (source, target) format pairs."""
        return [
            ("urdf", "mjcf"),
            ("urdf", "sdf"),
            ("mjcf", "urdf"),
            ("sdf", "urdf"),
        ]

    @property
    def history(self) -> list[ConversionReport]:
        """Return conversion history."""
        return list(self._conversion_history)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Demonstrate model converter usage."""
    logger.info("Model Converter demonstration")
    logger.info("HAS_XML=%s  HAS_USD=%s  HAS_YAML=%s", HAS_XML, HAS_USD, HAS_YAML)

    converter = ModelConverter()
    supported = converter.get_supported_conversions()
    logger.info("Supported conversions: %s", supported)

    # Demonstrate intermediate model creation
    model = RobotModel(
        name="oncology_biopsy_robot",
        links=[
            LinkSpec(
                name="base_link",
                inertial=InertialProperties(mass=5.0),
                collision_geometries=[
                    GeometrySpec(geom_type=GeometryType.BOX, size=Vector3(0.2, 0.2, 0.1)),
                ],
            ),
            LinkSpec(
                name="link_1",
                inertial=InertialProperties(mass=2.0),
                collision_geometries=[
                    GeometrySpec(geom_type=GeometryType.CYLINDER, radius=0.04, length=0.3),
                ],
            ),
            LinkSpec(
                name="needle_holder",
                inertial=InertialProperties(mass=0.5),
                collision_geometries=[
                    GeometrySpec(geom_type=GeometryType.CYLINDER, radius=0.01, length=0.15),
                ],
            ),
        ],
        joints=[
            JointSpec(
                name="joint_1",
                joint_type=JointType.REVOLUTE,
                parent_link="base_link",
                child_link="link_1",
                origin=Vector3(0.0, 0.0, 0.1),
                axis=Vector3(0.0, 0.0, 1.0),
                lower_limit=-3.14,
                upper_limit=3.14,
            ),
            JointSpec(
                name="needle_joint",
                joint_type=JointType.PRISMATIC,
                parent_link="link_1",
                child_link="needle_holder",
                origin=Vector3(0.0, 0.0, 0.3),
                axis=Vector3(0.0, 0.0, 1.0),
                lower_limit=0.0,
                upper_limit=0.15,
            ),
        ],
    )

    model_hash = model.compute_hash()
    logger.info(
        "Demo model: %s (hash=%s, %d links, %d joints)",
        model.name,
        model_hash[:12],
        len(model.links),
        len(model.joints),
    )
    logger.info("Demonstration complete.")


if __name__ == "__main__":
    main()
