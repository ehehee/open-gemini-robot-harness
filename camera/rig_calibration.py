"""Loads the rig's configuration and solved extrinsics from config/.

Nothing here solves a calibration; it only reads what was produced for the
rig (config/rig.yaml for CAN channels, gripper travel, gravity-comp mass,
joint offsets and camera serials; config/extrinsics/*.yaml for the solved
transforms). Set RIG_CONFIG_DIR to use another rig's copy of that layout.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import yaml

from camera.extrinsics import Extrinsics
from camera.intrinsics import Intrinsics

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def config_dir() -> Path:
    override = os.environ.get("RIG_CONFIG_DIR")
    path = Path(override) if override else DEFAULT_CONFIG_DIR
    if not (path / "rig.yaml").is_file():
        raise RuntimeError(f"no rig.yaml in {path}. Set RIG_CONFIG_DIR to a directory laid out like config/.")
    return path


def config_files(names: list[str]) -> list[Path]:
    """rig.yaml plus the named extrinsics, for archiving with a run."""
    return [config_dir() / "rig.yaml"] + [extrinsics_dir() / f"{n}.yaml" for n in names]


def extrinsics_dir() -> Path:
    return config_dir() / "extrinsics"


def load_extrinsic(name: str) -> Extrinsics:
    """name is a file stem under extrinsics/, e.g. 'T_rightbase_overhead',
    'T_leftbase_rightbase' or 'T_right_wrist_gripper_cam'."""
    base = extrinsics_dir()
    path = base / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in base.glob("T_*.yaml"))
        raise FileNotFoundError(f"{path} not found. Available: {available}")
    return Extrinsics.load_yaml(path)


def load_overhead_intrinsics(side: str = "right") -> Intrinsics:
    """The overhead camera's intrinsics, as recorded in the solved extrinsic's
    own metadata. Use this when the camera is not connected."""
    data = yaml.safe_load((extrinsics_dir() / f"T_{side}base_overhead.yaml").read_text())
    i = data["metadata"]["intrinsics"]
    return Intrinsics(width=i["width"], height=i["height"], fx=i["fx"], fy=i["fy"],
                      cx=i["cx"], cy=i["cy"], dist_coeffs=list(i["coeffs"]))


def load_rig_config() -> dict:
    return yaml.safe_load((config_dir() / "rig.yaml").read_text())


def arm_channel(side: str) -> str:
    return load_rig_config()["arms"][side]["channel"]


def arm_gripper_type(side: str) -> str:
    return load_rig_config()["arms"][side].get("gripper_type", "crank_4310")


def arm_gripper_limits(side: str) -> tuple[float, float] | None:
    """arms.<side>.gripper_limits as [closed, open] in motor-7 radians, or None
    (then i2rt measures them on connect by driving the jaws to both stops)."""
    value = load_rig_config()["arms"][side].get("gripper_limits")
    return (float(value[0]), float(value[1])) if value is not None else None


def arm_joint_offsets_rad(side: str) -> list[float]:
    """fk.joint_offsets.<side>.delta_deg in radians (q_model = q_encoder +
    delta). i2rt's driver does not apply these; yam_backend does. Zeros if
    the rig records none."""
    deltas = (load_rig_config().get("fk", {}).get("joint_offsets", {}).get(side) or {}).get("delta_deg")
    return [math.radians(float(d)) for d in deltas] if deltas else [0.0] * 6


def camera_serial(role: str) -> str:
    """cameras.<role>.serial, e.g. role 'right_wrist'."""
    return str(load_rig_config()["cameras"][role]["serial"])
