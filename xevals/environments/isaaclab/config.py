"""Portable configuration for the Isaac Lab Newton port; no simulator imports."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from xevals.environments.envs import SPLITS
from xevals.environments.manipulation import TASKS

# Upstream develop is a moving API: record the source revision used by this port.
ISAACLAB_REVISION = "18086270eecac181e4aad1ae323bde3b9c863072"


@dataclass(frozen=True)
class IsaacLabConfig:
    """One Panda scene, stepped by Newton's MJWarp solver.

    The single-environment interface matches xevals's trajectory runner. Physics
    stays on CUDA; observations cross to NumPy once per policy step.
    """

    task: str = "reach"
    split: str = "in"
    device: str = "cuda:0"
    renderer: str = "newton"
    camera: str = "pov"
    image_size: int = 96
    frame_rate: float = 20.0
    physics_dt: float = 1.0 / 240.0
    substeps: int = 2
    horizon: int | None = None
    iterations: int = 100
    ls_iterations: int = 50
    tolerance: float = 1e-6
    nconmax: int = 200
    njmax: int = 1000
    object_mass: float = 0.08
    friction: float = 1.2
    robot_usd: str | None = None

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}; choose {sorted(TASKS)}")
        if self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r}; choose {SPLITS}")
        if self.renderer not in ("none", "newton", "rtx"):
            raise ValueError("renderer must be 'none', 'newton', or 'rtx'")
        if self.camera not in ("pov", "wrist", "scene", "scene-wrist"):
            raise ValueError("camera must be 'pov', 'wrist', 'scene', or 'scene-wrist'")
        if not (
            self.device == "cuda" or (self.device.startswith("cuda:") and self.device[5:].isdigit())
        ):
            raise ValueError("Isaac Lab Newton evaluation requires a CUDA device")
        for name in ("frame_rate", "physics_dt", "tolerance", "object_mass", "friction"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("substeps", "image_size", "iterations", "ls_iterations", "nconmax", "njmax"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.horizon is not None and (
            isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1
        ):
            raise ValueError("horizon must be a positive integer")
        ratio = 1.0 / (self.frame_rate * self.physics_dt)
        if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-8):
            raise ValueError("the control period must be an integer multiple of physics_dt")

    @property
    def decimation(self) -> int:
        return round(1.0 / (self.frame_rate * self.physics_dt))

    @property
    def episode_horizon(self) -> int:
        return TASKS[self.task].horizon if self.horizon is None else self.horizon

    def describe(self) -> dict:
        return asdict(self) | {
            "backend": "isaaclab-newton",
            "solver": "mujoco_warp",
            "isaaclab_reference_revision": ISAACLAB_REVISION,
            "controller": "dls-policy-rate-v1",
            "integrator": "implicitfast",
            "cone": "elliptic",
            "impratio": 10.0,
            "num_envs": 1,
            "decimation": self.decimation,
        }
