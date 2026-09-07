"""Robot models from the MuJoCo Menagerie, and the scene they are evaluated in.

This module owns everything about *which* robot, and :mod:`xevals.sim` owns what
happens to it. The split matters because the five arms differ in exactly the
ways that break a benchmark: FR3 and UR5e ship without a hand at all, YAM and
SO-101 carry their own, and the joint counts run from five to seven. A policy
compared across robots has to see one interface, so the differences are absorbed
here, once, in a table rather than in five code paths.

Two of them are worth naming:

* **Grippers.** FR3 and UR5e get a Robotiq 2F-85 attached at their
  ``attachment_site``, so every robot can be asked to pick something up. Without
  it, half the task set would be unmeasurable on half the robots, and a missing
  number reads like a failure when it is really an absence.
* **The tool frame.** Each robot names its end effector differently, and the
  Panda does not name it at all. :attr:`RobotSpec.tcp` is the site the
  controller servos, and one is added to the Panda's hand when the scene is
  built.

Assets come from the ``robot_descriptions`` package, which caches a Menagerie
checkout under ``~/.cache/robot_descriptions``. Set ``XEVALS_MENAGERIE`` to an
existing checkout to skip the download entirely.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from xevals.core.errors import MissingExtra

__all__ = [
    "GripperSpec",
    "ROBOTS",
    "RobotSpec",
    "Workspace",
    "assets_available",
    "available",
    "compiled",
    "compose",
    "describe",
    "menagerie_path",
    "object_xml",
    "portable_xml",
]

#: Where the Menagerie lives, if it is already on disk. Set this to a checkout
#: of https://github.com/google-deepmind/mujoco_menagerie and nothing is
#: downloaded.
MENAGERIE_ENV = "XEVALS_MENAGERIE"

#: The table top is the world's z origin: every robot is bolted to it and every
#: object rests on it, so heights read as "above the table" without arithmetic.
TABLE_HEIGHT = 0.4
TABLE_TOP = 0.05


@dataclass(frozen=True)
class Workspace:
    """The region of the table an object may be placed in, in metres.

    The box is where a *task* happens, not where the arm can reach: it is
    deliberately well inside the reachable set, so that a failure is the policy's
    and not the kinematics'. ``ood/layout`` scales it by :attr:`ood_scale` about
    its centre, which is the one thing that split is allowed to change.
    """

    x: tuple[float, float]
    y: tuple[float, float]
    ood_scale: float = 1.5

    def bounds(self, *, wide: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Low and high corners, widened for the out-of-distribution layout."""
        low = np.array([self.x[0], self.y[0]], dtype=np.float64)
        high = np.array([self.x[1], self.y[1]], dtype=np.float64)
        if not wide:
            return low, high
        centre = (low + high) / 2.0
        half = (high - low) / 2.0 * self.ood_scale
        return centre - half, centre + half


@dataclass(frozen=True)
class GripperSpec:
    """The one actuator that opens and closes a hand.

    ``open_ctrl`` and ``closed_ctrl`` are raw control values and are *not*
    ordered: the Panda opens at 255 and the Robotiq closes at 255. Naming both
    ends is what lets one normalised action mean "open" on every robot.
    """

    actuator: str
    open_ctrl: float
    closed_ctrl: float
    fingers: tuple[str, ...] = ()

    def ctrl(self, opening: float) -> float:
        """Control value for a normalised opening, 0 closed and 1 open."""
        t = float(np.clip(opening, 0.0, 1.0))
        return self.closed_ctrl + t * (self.open_ctrl - self.closed_ctrl)

    def opening(self, ctrl: float) -> float:
        """The inverse of :meth:`ctrl`, for reading a hand's commanded state."""
        span = self.open_ctrl - self.closed_ctrl
        if abs(span) < 1e-12:
            return 1.0
        return float(np.clip((float(ctrl) - self.closed_ctrl) / span, 0.0, 1.0))


@dataclass(frozen=True)
class RobotSpec:
    """One arm: where to find it, how to hold it, and what to servo."""

    name: str
    description: str
    summary: str
    arm_joints: tuple[str, ...]
    home: tuple[float, ...]
    tcp: str
    workspace: Workspace
    gripper: GripperSpec | None = None
    #: A Menagerie model attached at ``attach_site``, for arms with no hand.
    attach: str | None = None
    attach_site: str = "attachment_site"
    #: Half-extents of the manipulated object. Small hands need small objects,
    #: and an arm whose shoulder sits low needs a tall one: the SO-101 cannot
    #: bring its jaws down to a flat block, so it is given an upright one.
    object_half: tuple[float, float, float] = (0.025, 0.025, 0.025)
    #: How far the grasp sits above the object's centre.
    grasp_height: float = 0.0
    #: Scales the task's lift height, for arms with less room above the table.
    lift_scale: float = 1.0
    #: Height of a riser the task happens on. The SO-101's shoulder sits a few
    #: centimetres above the table and its tool cannot get down to it, so its
    #: object stands on a platform, the way a small arm is used in practice.
    platform_height: float = 0.0
    #: How the base is turned, as a quaternion. An arm whose neutral reach
    #: points along its own -y is bolted down rotated, so that "forward" in the
    #: scene is forward for the arm and the shoulder joint stays near zero.
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    #: Which way the tool points while it works. Straight down for arms with
    #: the reach for it; tilted forward for the SO-101, whose shoulder sits a
    #: few centimetres above the table and cannot get over a block otherwise.
    approach: tuple[float, float, float] = (0.0, 0.0, -1.0)
    #: Stiffens the position servos. The SO-101 is modelled with the gains of
    #: its hobby servos, which sag several centimetres under the arm's own
    #: weight; a benchmark needs the commanded pose to be the pose.
    gain_scale: float = 1.0
    #: Metres the tool moves per unit of action, and how far the camera sits.
    step_size: float = 0.02
    camera_distance: float = 1.4
    #: Where the point-of-view camera sits, as multiples of the arm's reach:
    #: how far behind the base, how far above the work surface, and how far
    #: forward it looks. Wider than a studio shot on purpose.
    #: Where the point-of-view camera sits and what it looks at, both in
    #: multiples of the arm's reach and both relative to the middle of the
    #: workspace: behind it, to one side, and above the work surface. Over the
    #: arm's shoulder rather than straight behind it, because a single arm
    #: mounted at the back of a table stands directly in its own line of sight.
    pov_offset: tuple[float, float, float] = (-0.8, -1.5, 2.0)
    #: What it aims at, again relative to the middle of the workspace and in
    #: multiples of reach: back toward the arm, across, and up.
    pov_aim: tuple[float, float, float] = (-0.2, 0.3, 0.05)
    pov_fovy: float = 52.0
    extras: dict[str, Any] = field(default_factory=dict)

    #: The tasks this arm is offered. Not every arm can do every one of them,
    #: and a task an arm cannot physically perform is worse than absent: it
    #: would read as a policy scoring zero rather than as a robot that cannot.
    tasks: tuple[str, ...] = ("reach", "push", "pick", "place")

    @property
    def dof(self) -> int:
        """Arm joints, hand excluded."""
        return len(self.arm_joints)

    @property
    def has_gripper(self) -> bool:
        """Whether the robot can close a hand on something, attached or native."""
        return self.gripper is not None


#: The Robotiq 2F-85, as attached to the two arms that ship bare. 0 is fully
#: open and 255 fully closed, which is the opposite of the Panda's convention.
_ROBOTIQ = GripperSpec(
    actuator="gripper/fingers_actuator",
    open_ctrl=0.0,
    closed_ctrl=255.0,
    fingers=("gripper/left_pad", "gripper/right_pad"),
)


ROBOTS: dict[str, RobotSpec] = {
    "panda": RobotSpec(
        name="panda",
        description="panda_mj_description",
        summary="Franka Emika Panda, 7 DoF with the Franka Hand",
        arm_joints=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"),
        home=(0.2092, -0.1966, 0.0047, -2.3671, 0.0011, 2.1705, -1.2429),
        tcp="tcp",
        workspace=Workspace(x=(0.38, 0.54), y=(-0.14, 0.14)),
        gripper=GripperSpec(
            actuator="actuator8",
            open_ctrl=255.0,
            closed_ctrl=0.0,
            fingers=("left_finger", "right_finger"),
        ),
        camera_distance=1.6,
        extras={"tcp_body": "hand", "tcp_offset": (0.0, 0.0, 0.1034)},
    ),
    "fr3": RobotSpec(
        name="fr3",
        description="fr3_mj_description",
        summary="Franka Research 3, 7 DoF, with a Robotiq 2F-85 attached",
        arm_joints=tuple(f"fr3_joint{i}" for i in range(1, 8)),
        home=(-0.0027, -0.2544, 0.203, -2.28, 0.0566, 2.03, -0.2045),
        tcp="gripper/pinch",
        workspace=Workspace(x=(0.38, 0.54), y=(-0.14, 0.14)),
        gripper=_ROBOTIQ,
        attach="robotiq_2f85_mj_description",
        camera_distance=1.6,
    ),
    "ur5e": RobotSpec(
        name="ur5e",
        description="ur5e_mj_description",
        summary="Universal Robots UR5e, 6 DoF, with a Robotiq 2F-85 attached",
        arm_joints=(
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ),
        home=(0.2829, -2.0079, -1.3712, 1.8083, -1.5708, 0.0223),
        tcp="gripper/pinch",
        workspace=Workspace(x=(0.40, 0.56), y=(-0.14, 0.14)),
        gripper=_ROBOTIQ,
        attach="robotiq_2f85_mj_description",
        camera_distance=1.75,
    ),
    "yam": RobotSpec(
        name="yam",
        description="yam_mj_description",
        summary="i2rt YAM, 6 DoF with a parallel gripper",
        arm_joints=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        home=(0.0, 1.6004, 1.3845, -1.3548, 0.0, 0.015),
        tcp="grasp_site",
        workspace=Workspace(x=(0.22, 0.34), y=(-0.10, 0.10)),
        gripper=GripperSpec(
            actuator="gripper",
            open_ctrl=0.041,
            closed_ctrl=0.0,
            fingers=("link_left_finger", "link_right_finger"),
        ),
        object_half=(0.022, 0.022, 0.022),
        gain_scale=2.0,
        lift_scale=0.8,
        step_size=0.014,
        camera_distance=1.15,
    ),
    "so101": RobotSpec(
        name="so101",
        description="so_arm101_mj_description",
        summary="RobotStudio SO-101, 5 DoF with a jaw gripper, a low-cost arm",
        # No grasping tasks: the SO-101's wrist reaches its stop before the tool
        # can be brought down onto a block on the table, so a top-down pick is
        # outside what this arm can do in this scene. Saying so is more useful
        # than shipping a task every policy fails for a reason of ours.
        tasks=("reach", "push"),
        arm_joints=("1", "2", "3", "4", "5"),
        home=(0.0605, 0.5839, 0.544, -1.1279, 0.0),
        tcp="gripper",
        workspace=Workspace(x=(0.33, 0.39), y=(-0.05, 0.05), ood_scale=1.3),
        gripper=GripperSpec(actuator="6", open_ctrl=-0.17, closed_ctrl=1.745),
        gain_scale=8.0,
        grasp_height=0.012,
        pov_offset=(-0.6, -1.15, 1.5),
        pov_fovy=44.0,
        base_quat=(0.7071068, 0.0, 0.0, 0.7071068),
        # The SO-101's tool site looks back up the wrist, so the axis that has
        # to point at the table is the site's -z, not its +z.
        approach=(0.0, 0.0, 1.0),
        object_half=(0.014, 0.014, 0.025),
        lift_scale=0.55,
        step_size=0.012,
        camera_distance=0.72,
    ),
}


def available() -> list[str]:
    """The robots this library knows how to build a scene for."""
    return sorted(ROBOTS)


def describe(name: str | None = None) -> dict[str, Any]:
    """What a robot is, without downloading or compiling anything."""
    if name is None:
        return {n: describe(n) for n in available()}
    spec = _spec(name)
    return {
        "name": spec.name,
        "summary": spec.summary,
        "dof": spec.dof,
        "gripper": "native" if spec.gripper and not spec.attach else "robotiq_2f85",
        "tcp": spec.tcp,
        "workspace": {"x": spec.workspace.x, "y": spec.workspace.y},
    }


def _spec(name: str) -> RobotSpec:
    if name not in ROBOTS:
        raise KeyError(f"unknown robot {name!r}; have {available()}")
    return ROBOTS[name]


def menagerie_path(description: str) -> Path:
    """The MJCF for one Menagerie model, downloading it only if it has to.

    ``XEVALS_MENAGERIE`` wins when it is set and contains the model, which is how
    an offline machine, or a CI job with a warm cache, avoids a two gigabyte
    clone it does not need.
    """
    folder, xml = _MENAGERIE_FILES[description]
    local = os.environ.get(MENAGERIE_ENV)
    if local:
        candidate = Path(local).expanduser() / folder / xml
        if candidate.exists():
            return candidate
    try:
        module = __import__(f"robot_descriptions.{description}", fromlist=["MJCF_PATH"])
    except ImportError as exc:  # pragma: no cover - exercised by the extras test
        raise MissingExtra(
            "robot_descriptions", "mujoco", f"to fetch the {description} model"
        ) from exc
    return Path(module.MJCF_PATH)


#: Folder and file inside the Menagerie, so ``XEVALS_MENAGERIE`` can be resolved
#: without importing ``robot_descriptions`` at all.
_MENAGERIE_FILES: dict[str, tuple[str, str]] = {
    "panda_mj_description": ("franka_emika_panda", "panda.xml"),
    "fr3_mj_description": ("franka_fr3", "fr3.xml"),
    "ur5e_mj_description": ("universal_robots_ur5e", "ur5e.xml"),
    "yam_mj_description": ("i2rt_yam", "yam.xml"),
    "so_arm101_mj_description": ("robotstudio_so101", "so101.xml"),
    "robotiq_2f85_mj_description": ("robotiq_2f85", "2f85.xml"),
}


def assets_available(robot: str | None = None) -> bool:
    """Whether the models are on disk already, without fetching them.

    Tests use this to skip rather than to start a large download on a machine
    that never asked for one.
    """
    names = [robot] if robot else available()
    try:
        import robot_descriptions  # noqa: F401
    except ImportError:
        if not os.environ.get(MENAGERIE_ENV):
            return False
    for name in names:
        spec = _spec(str(name))
        for description in (spec.description, spec.attach):
            if description is None:
                continue
            folder, xml = _MENAGERIE_FILES[description]
            local = os.environ.get(MENAGERIE_ENV)
            if local and (Path(local).expanduser() / folder / xml).exists():
                continue
            cache = _cache_root() / folder / xml
            if not cache.exists():
                return False
    return True


def _cache_root() -> Path:
    root = os.environ.get("ROBOT_DESCRIPTIONS_CACHE")
    base = Path(root).expanduser() if root else Path.home() / ".cache" / "robot_descriptions"
    return base / "mujoco_menagerie"


# --------------------------------------------------------------------------
# Scene composition
# --------------------------------------------------------------------------


def _look_at(eye: np.ndarray, target: np.ndarray) -> list[float]:
    """MuJoCo's ``xyaxes`` for a camera at ``eye`` pointed at ``target``."""
    forward = target - eye
    forward = forward / max(float(np.linalg.norm(forward)), 1e-9)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, forward)
    return [*right.tolist(), *up.tolist()]


def _quat_multiply(a: Any, b: Any) -> list[float]:
    """Hamilton product of two ``(w, x, y, z)`` quaternions."""
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _axis_z(quat: Any) -> np.ndarray:
    """The local z axis of a ``(w, x, y, z)`` rotation, in the parent frame."""
    w, x, y, z = (float(v) for v in quat)
    return np.array(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)]
    )


def _site_body(spec: Any, name: str) -> str:
    """The body a site is defined on."""
    return str(spec.site(name).parent.name)


def compose(
    robot: str,
    *,
    object_name: str = "red cube",
    object_shape: str = "box",
) -> Any:
    """Build the evaluation scene for one robot as an uncompiled ``MjSpec``.

    The scene is deliberately plain: a table, one object on a free joint, a goal
    marker, one camera and one light. Anything more would be a confound, because
    a perturbation has to be the only thing that changed between two cells.
    """
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - exercised by the extras test
        raise MissingExtra("mujoco", "mujoco", "to build a robot scene") from exc
    from xevals.environments.envs import OBJECT_COLORS

    spec_def = _spec(robot)
    spec = mujoco.MjSpec.from_file(str(menagerie_path(spec_def.description)))
    # The robot's own keyframes describe the robot alone. Once a hand and an
    # object are attached the qpos width has changed, so a stale key would be a
    # silent shape error at reset; home poses live on the RobotSpec instead.
    for key in list(spec.keys):
        spec.delete(key)
    # Elliptic friction cones with a high impedance ratio are what make a
    # parallel gripper hold a box rather than squeeze it out sideways. Setting
    # them before the attach also stops the merge warning about the two models
    # disagreeing.
    spec.option.cone = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    spec.option.impratio = 10.0

    if spec_def.attach is not None:
        hand = mujoco.MjSpec.from_file(str(menagerie_path(spec_def.attach)))
        for key in list(hand.keys):
            hand.delete(key)
        hand.option.cone = spec.option.cone
        hand.option.impratio = spec.option.impratio
        site = spec.site(spec_def.attach_site)
        site.attach_body(hand.body("base_mount"), "gripper/", "")

    tcp_body = spec_def.extras.get("tcp_body")
    if tcp_body is not None:
        # The Panda names no site at all, so the tool frame is added where the
        # fingers meet rather than inferred from a body origin inside the wrist.
        spec.body(tcp_body).add_site(
            name=spec_def.tcp,
            pos=list(spec_def.extras.get("tcp_offset", (0.0, 0.0, 0.0))),
            group=4,
        )

    if spec_def.base_quat != (1.0, 0.0, 0.0, 0.0):
        spec.worldbody.bodies[0].quat = list(spec_def.base_quat)

    if spec_def.gain_scale != 1.0:
        for actuator in spec.actuators:
            actuator.gainprm[0] *= spec_def.gain_scale
            actuator.biasprm[1] *= spec_def.gain_scale
            actuator.biasprm[2] *= spec_def.gain_scale

    # A sky and a floor material, because the robot XMLs carry neither: they are
    # meant to be included into a scene, and this is that scene. It also stops a
    # recorded episode being a bright arm on a black rectangle.
    spec.add_texture(
        name="sky",
        type=int(mujoco.mjtTexture.mjTEXTURE_SKYBOX),
        builtin=int(mujoco.mjtBuiltin.mjBUILTIN_GRADIENT),
        rgb1=[0.62, 0.66, 0.72],
        rgb2=[0.28, 0.31, 0.36],
        width=256,
        height=256,
    )
    spec.add_texture(
        name="grid",
        type=int(mujoco.mjtTexture.mjTEXTURE_2D),
        builtin=int(mujoco.mjtBuiltin.mjBUILTIN_CHECKER),
        rgb1=[0.30, 0.32, 0.34],
        rgb2=[0.34, 0.36, 0.38],
        width=300,
        height=300,
    )
    grid = spec.add_material(name="grid", texrepeat=[6, 6], reflectance=0.05)
    grid.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "grid"

    world = spec.worldbody
    low, high = spec_def.workspace.bounds()
    reach = float((low[0] + high[0]) / 2.0)
    world.add_light(
        name="scene_light",
        pos=[0.0, 0.0, 2.0],
        dir=[0.0, 0.0, -1.0],
        type=int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL),
        diffuse=[0.7, 0.7, 0.7],
        specular=[0.2, 0.2, 0.2],
    )
    # A second, softer light from the camera's side. One directional light alone
    # leaves the near face of everything in its own shadow, which is exactly the
    # face a policy is looking at.
    world.add_light(
        name="fill_light",
        pos=[reach * 2.0, -reach * 1.2, spec_def.platform_height + reach * 1.6],
        dir=[-0.5, 0.3, -0.8],
        type=int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL),
        diffuse=[0.35, 0.35, 0.38],
        specular=[0.05, 0.05, 0.05],
        castshadow=0,
    )
    world.add_geom(
        name="floor",
        type=int(mujoco.mjtGeom.mjGEOM_PLANE),
        pos=[0.0, 0.0, -TABLE_HEIGHT],
        size=[3.0, 3.0, 0.1],
        material="grid",
    )
    # The top is a thin slab on a plinth rather than one tall box. A block
    # resting on a half-metre box gives Newton's narrow phase a penetration
    # depth measured from somewhere inside it, and the block is fired off the
    # table on the first frame.
    table = world.add_body(name="table", pos=[0.0, 0.0, -TABLE_TOP / 2.0])
    half_table = max(0.34, reach + 0.22)
    table.add_geom(
        name="table_top",
        type=int(mujoco.mjtGeom.mjGEOM_BOX),
        size=[half_table, half_table, TABLE_TOP / 2.0],
        rgba=[0.78, 0.76, 0.72, 1.0],
        friction=[1.0, 0.005, 0.0001],
    )
    base = world.add_body(
        name="table_base", pos=[0.0, 0.0, -(TABLE_HEIGHT + TABLE_TOP) / 2.0]
    )
    base.add_geom(
        name="table_base_box",
        type=int(mujoco.mjtGeom.mjGEOM_BOX),
        size=[half_table * 0.8, half_table * 0.8, (TABLE_HEIGHT - TABLE_TOP) / 2.0],
        rgba=[0.55, 0.53, 0.50, 1.0],
    )

    if spec_def.platform_height > 0.0:
        low, high = spec_def.workspace.bounds(wide=True)
        centre = (low + high) / 2.0
        half = (high - low) / 2.0 + 0.05
        platform = world.add_body(
            name="platform",
            pos=[float(centre[0]), float(centre[1]), spec_def.platform_height / 2.0],
        )
        platform.add_geom(
            name="platform_top",
            type=int(mujoco.mjtGeom.mjGEOM_BOX),
            size=[float(half[0]), float(half[1]), spec_def.platform_height / 2.0],
            rgba=[0.62, 0.60, 0.57, 1.0],
            friction=[1.0, 0.005, 0.0001],
        )

    colour = OBJECT_COLORS.get(object_name, (198, 62, 48))
    rgba = [c / 255.0 for c in colour] + [1.0]
    half = spec_def.object_half
    shapes = {
        "box": (int(mujoco.mjtGeom.mjGEOM_BOX), list(half)),
        "cylinder": (int(mujoco.mjtGeom.mjGEOM_CYLINDER), [half[0], half[2], 0.0]),
        "sphere": (int(mujoco.mjtGeom.mjGEOM_SPHERE), [half[0], 0.0, 0.0]),
    }
    if object_shape not in shapes:
        raise ValueError(f"unknown object shape {object_shape!r}; have {sorted(shapes)}")
    geom_type, geom_size = shapes[object_shape]
    body = world.add_body(
        name="object", pos=[0.4, 0.0, spec_def.platform_height + half[2]]
    )
    body.add_freejoint(name="object_free")
    body.add_geom(
        name="object_geom",
        type=geom_type,
        size=geom_size,
        rgba=rgba,
        mass=0.08,
        friction=[1.2, 0.01, 0.0005],
        condim=4,
    )
    world.add_site(
        name="goal",
        pos=[0.4, 0.2, spec_def.platform_height + 0.002],
        type=int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        size=[half[0] * 1.8, 0.001, 0.0],
        rgba=[0.20, 0.55, 0.35, 0.55],
    )

    # The camera never moves with the split: an out-of-distribution layout is
    # meant to change where things are, not how the scene is photographed. It
    # looks between the shoulder and the workspace so that both the arm and what
    # it is doing are in frame, which is what makes a recorded episode readable.
    centre = np.array([reach * 0.55, 0.0, spec_def.platform_height + 0.18])
    distance = spec_def.camera_distance
    eye = centre + np.array([distance * 0.56, -distance * 0.64, distance * 0.40])
    world.add_camera(name="scene", pos=eye.tolist(), xyaxes=_look_at(eye, centre), fovy=48.0)

    # The camera a policy is trained through: over the arm's shoulder, looking
    # out across the table with the arm reaching away into the frame. Every
    # observation, every video and every visual perturbation goes through this
    # one. It is offset to the side rather than straight behind, because an arm
    # mounted at the back of a table stands squarely in its own line of sight:
    # from directly behind, the block is visible in 7 spawns out of 30, and from
    # over the shoulder in 30.
    surface = spec_def.platform_height
    work = np.array([reach, 0.0, surface])
    pov_eye = work + reach * np.array(spec_def.pov_offset)
    pov_target = work + reach * np.array(spec_def.pov_aim)
    world.add_camera(
        name="pov",
        pos=pov_eye.tolist(),
        xyaxes=_look_at(pov_eye, pov_target),
        fovy=spec_def.pov_fovy,
    )

    # And the other camera these setups are built with: one on the wrist,
    # looking down the tool at whatever the hand is about to touch.
    tool = spec.site(spec_def.tcp)
    approach_down = spec_def.approach[2] < 0
    # A MuJoCo camera looks along its own -z. The tool's approach is the site's
    # +z on four of these arms and its -z on the SO-101, so one of the two needs
    # turning over to point the lens the same way the hand does.
    flip = (0.0, 1.0, 0.0, 0.0) if approach_down else (1.0, 0.0, 0.0, 0.0)
    sign = 1.0 if approach_down else -1.0
    wrist = spec.body(_site_body(spec, spec_def.tcp))
    wrist.add_camera(
        name="wrist",
        pos=(np.asarray(tool.pos) - sign * np.asarray(_axis_z(tool.quat)) * 0.06).tolist(),
        quat=_quat_multiply(tool.quat, flip),
        fovy=70.0,
    )
    return spec


@functools.lru_cache(maxsize=16)
def portable_xml(
    robot: str,
    *,
    object_name: str = "red cube",
    object_shape: str = "box",
    with_object: bool = True,
) -> str:
    """The composed scene as XML that resolves its own assets.

    MuJoCo finds meshes through ``meshdir``, relative to the file the model was
    read from. A scene assembled in memory from two packages has no such file
    and can have only one ``meshdir``, so every mesh and texture is rewritten to
    an absolute path here. This is what lets another simulator, Newton in
    particular, read the same scene rather than a re-description of it.
    """
    spec_def = _spec(robot)
    spec = compose(robot, object_name=object_name, object_shape=object_shape)
    if not with_object:
        # Newton imports a whole MJCF with one choice about whether its root
        # bodies are pinned, and the arm and the block need opposite answers.
        # The block is handed over separately, by object_xml.
        spec.delete(spec.body("object"))
    roots = [menagerie_path(spec_def.description).parent]
    if spec_def.attach is not None:
        roots.append(menagerie_path(spec_def.attach).parent)
    subdirs = ["", "assets"]

    def resolve(name: str) -> str | None:
        for root in roots:
            for sub in subdirs:
                candidate = root / sub / name if sub else root / name
                if candidate.exists():
                    return str(candidate)
        return None

    for asset in [*spec.meshes, *spec.textures]:
        if not asset.file:
            continue
        found = resolve(asset.file)
        if found is not None:
            asset.file = found
    spec.meshdir = ""
    spec.texturedir = ""
    return str(spec.to_xml())


def object_xml(robot: str, *, object_name: str = "red cube", object_shape: str = "box") -> str:
    """The manipulated object on its own, as a free-floating MJCF.

    Its numbers come from the same place the composed scene's do, so the two
    backends are holding the same block rather than two blocks that happen to
    have been described similarly.
    """
    from xevals.environments.envs import OBJECT_COLORS

    spec_def = _spec(robot)
    half = spec_def.object_half
    colour = OBJECT_COLORS.get(object_name, (198, 62, 48))
    rgba = " ".join(f"{c / 255.0:.4f}" for c in colour) + " 1"
    sizes = {
        "box": f"{half[0]} {half[1]} {half[2]}",
        "cylinder": f"{half[0]} {half[2]}",
        "sphere": f"{half[0]}",
    }
    if object_shape not in sizes:
        raise ValueError(f"unknown object shape {object_shape!r}; have {sorted(sizes)}")
    return f"""<mujoco model="object">
  <worldbody>
    <body name="object" pos="0.4 0 {spec_def.platform_height + half[2]}">
      <freejoint name="object_free"/>
      <geom name="object_geom" type="{object_shape}" size="{sizes[object_shape]}"
            rgba="{rgba}" mass="0.08" friction="1.2 0.01 0.0005" condim="4"/>
    </body>
  </worldbody>
</mujoco>
"""


@functools.lru_cache(maxsize=32)
def compiled(
    robot: str,
    *,
    object_name: str = "red cube",
    object_shape: str = "box",
) -> Any:
    """A compiled ``MjModel`` for one scene configuration, cached.

    The runner builds a fresh environment for every episode, so compiling a
    Menagerie model each time would dominate the wall clock it is trying to
    measure. Callers that mutate a model (``set_physics``) copy it first.
    """
    spec = compose(robot, object_name=object_name, object_shape=object_shape)
    return spec.compile()
