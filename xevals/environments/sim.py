"""Robot manipulation environments: five arms, four tasks, two physics backends.

What makes this an *evaluation* environment rather than a demo is that every arm
is driven through the same interface. A policy emits four numbers, ``[dx, dy,
dz, grip]``, and a damped-least-squares controller turns them into joint targets
for whichever robot is underneath. A Panda has seven joints and an SO-101 has
five, they hold different hands and reach different distances, but a policy
written for one can be scored on all of them, and the difference in the score is
about the robot rather than about the plumbing.

The tool is servoed top-down: the controller holds the approach axis pointing at
the table and leaves the yaw free, which is what a five-joint arm can actually
do and what a parallel jaw needs for a block on a table.

MuJoCo is the reference backend. Newton (``backend="newton"``) steps the same
scene with Warp kernels and mirrors the result back into MuJoCo state, so the
tool frame, the contacts and the camera all come from one code path and the two
backends can be compared on the numbers rather than on the pictures.

Nothing here is imported unless a robot environment is built, so a bare install
still imports :mod:`xevals` cleanly and a run that never touches a robot never
pays for MuJoCo.
"""

from __future__ import annotations

import copy
import functools
import math
import warnings
from typing import Any

import numpy as np

from xevals.core.errors import MissingExtra
from xevals.core.types import Obs, SafetyLimits
from xevals.environments import robots
from xevals.environments.envs import ENVS, IN_DISTRIBUTION_OBJECTS, OOD_OBJECTS, SPLITS, _draw_text
from xevals.environments.manipulation import TASKS, ManipulationDemonstrator, TaskSpec

__all__ = [
    "ManipulationEnv",
    "NEWTON_TASKS",
    "TASKS",
    "TaskSpec",
    "clear_caches",
    "manipulation_env",
]

#: How stiff a joint servo may be under Newton's CPU solver before it rings.
_NEWTON_MAX_KE = 800.0

#: Contact stiffness and damping for the same solver, see ``_NewtonSim``.
_NEWTON_CONTACT_KE = 500.0
_NEWTON_CONTACT_KD = 10.0

#: The approach direction the controller holds the tool axis against.
_DOWN = np.array([0.0, 0.0, -1.0])



def clear_caches() -> None:
    """Drop cached models and renderers. Tests use this; runs do not need it."""
    robots.compiled.cache_clear()
    _renderer.cache_clear()


@functools.lru_cache(maxsize=8)
def _renderer(model: Any, height: int, width: int) -> Any:
    """One offscreen renderer per model and size.

    A renderer owns a GL context, and the runner builds a fresh environment for
    every episode, so making one per episode would cost more than the physics.
    Keyed on the model object, which :func:`xevals.robots.compiled` already
    shares across environments of the same configuration.
    """
    import mujoco

    return mujoco.Renderer(model, height, width)


# --------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------


def _ik_step(
    mujoco: Any,
    model: Any,
    data: Any,
    dof: np.ndarray,
    qadr: np.ndarray,
    limits: np.ndarray,
    site: int,
    target: np.ndarray,
    *,
    approach: np.ndarray = _DOWN,
    iterations: int = 8,
    damping: float = 0.12,
    rate: float = 0.55,
    rotation_weight: float = 0.5,
) -> None:
    """Move ``data.qpos`` toward a tool position, holding the approach axis.

    Damped least squares on the site Jacobian. The orientation term is the cross
    product of the tool axis with the desired approach direction, which asks for
    the two to line up and says nothing about the yaw: a five-joint arm cannot
    honour a full pose, and a jaw closing on a block does not need it to.
    """
    task = np.zeros(6)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iterations):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        task[:3] = target - data.site_xpos[site]
        axis = data.site_xmat[site].reshape(3, 3)[:, 2]
        task[3:] = rotation_weight * np.cross(axis, approach)
        if np.linalg.norm(task[:3]) < 1e-5 and np.linalg.norm(task[3:]) < 1e-4:
            return
        mujoco.mj_jacSite(model, data, jacp, jacr, site)
        jac = np.vstack([jacp[:, dof], rotation_weight * jacr[:, dof]])
        step = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), task)
        data.qpos[qadr] = np.clip(data.qpos[qadr] + rate * step, limits[:, 0], limits[:, 1])


# --------------------------------------------------------------------------
# The environment
# --------------------------------------------------------------------------


class ManipulationEnv(ManipulationDemonstrator):
    """A Menagerie arm on a table with one object, one goal and one camera.

    Args:
        robot: which arm, one of :data:`xevals.robots.ROBOTS`.
        task: one of :data:`TASKS`.
        split: ``in`` or one of the ``ood/*`` splits, which each change exactly
            one thing about the scene or the instruction.
        backend: ``mujoco`` (the reference) or ``newton``.
        image_size: square render, in pixels.
        camera: which camera the observation comes from. ``pov`` is the
            workspace camera a policy would be trained through; ``scene`` is a
            wider third-person view, for illustrations rather than for models.
        horizon: steps before the episode is cut off; the task's own by default.
        object_name: overrides the colour the split would choose.
        frame_rate: control rate in hertz. The physics runs at the model's own
            timestep and is substepped up to this.
    """

    def __init__(
        self,
        robot: str = "panda",
        task: str = "pick",
        *,
        split: str = "in",
        backend: str = "mujoco",
        image_size: int = 96,
        camera: str = "pov",
        horizon: int | None = None,
        object_name: str | None = None,
        object_shape: str | None = None,
        frame_rate: float = 20.0,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - exercised by the extras test
            raise MissingExtra("mujoco", "mujoco", "to run a robot environment") from exc
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; have {sorted(TASKS)}")
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; have {sorted(SPLITS)}")
        if backend not in ("mujoco", "newton"):
            raise ValueError(f"backend must be 'mujoco' or 'newton', got {backend!r}")
        self._mj = mujoco
        self.robot = str(robot)
        self.spec = robots.ROBOTS[self.robot] if self.robot in robots.ROBOTS else None
        if self.spec is None:
            raise KeyError(f"unknown robot {robot!r}; have {robots.available()}")
        self.task = task
        self.task_spec = TASKS[task]
        if task not in self.spec.tasks:
            raise ValueError(
                f"{robot!r} is not offered the {task!r} task; it has "
                f"{sorted(self.spec.tasks)}"
            )
        self.split = split
        self.backend = backend
        self.image_size = int(image_size)
        self.camera = str(camera)
        self.frame_rate = float(frame_rate)
        self.horizon = int(horizon if horizon is not None else self.task_spec.horizon)

        # Splits change one thing each: the object, where it may be, or the
        # wording. Anything else would make a generalisation gap unreadable.
        self.object_name = object_name or (
            OOD_OBJECTS[0] if split == "ood/object" else IN_DISTRIBUTION_OBJECTS[0]
        )
        self.object_shape = object_shape or ("cylinder" if split == "ood/object" else "box")
        self._wide = split == "ood/layout"

        self._base = robots.compiled(
            self.robot,
            object_name=self.object_name,
            object_shape=self.object_shape,
        )
        self.model = self._base
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        self._index()

        #: Physics knobs, as absolute values so a perturbation can scale them.
        self.mass = float(self.model.body_mass[self._object_body])
        self.friction = float(self.model.geom_friction[self._object_geom, 0])
        self.gain = 1.0
        self._gain_base = np.array(self.model.actuator_gainprm[:, 0], copy=True)
        self._bias_base = np.array(self.model.actuator_biasprm[:, 1], copy=True)

        self.action_dim = 4
        self.action_low = -np.ones(self.action_dim, dtype=np.float32)
        self.action_high = np.ones(self.action_dim, dtype=np.float32)
        self.substeps = max(1, int(round((1.0 / self.frame_rate) / self.model.opt.timestep)))

        self._scene_text: str | None = None
        self._t = 0
        self._lifted = False
        self._goal = np.zeros(3)
        self._object_start = np.zeros(3)
        self._q_cmd = np.array(self.spec.home, dtype=np.float64)
        self._grip_cmd = 1.0
        self._tcp_cmd = np.zeros(3)
        self._sim: _NewtonSim | None = None
        if backend == "newton":
            self._sim = _NewtonSim(self)

    # -- model indexing ---------------------------------------------------

    def _index(self) -> None:
        mj, m = self._mj, self.model
        name2id = mj.mj_name2id

        def jid(name: str) -> int:
            found = name2id(m, mj.mjtObj.mjOBJ_JOINT, name)
            if found < 0:
                raise KeyError(f"{self.robot}: no joint {name!r} in the compiled model")
            return found

        self._arm_joints = np.array([jid(j) for j in self.spec.arm_joints])
        self._arm_qadr = np.array([m.jnt_qposadr[j] for j in self._arm_joints])
        self._arm_dof = np.array([m.jnt_dofadr[j] for j in self._arm_joints])
        self._arm_limits = np.array(m.jnt_range[self._arm_joints], dtype=np.float64)
        self._arm_actuators = np.array(
            [name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, j) for j in self.spec.arm_joints]
        )
        if np.any(self._arm_actuators < 0):
            # Some models name the actuator after the joint and some do not; fall
            # back to actuator order, which matches joint order in every
            # Menagerie arm we ship.
            self._arm_actuators = np.arange(self.spec.dof)
        self._grip_actuator = -1
        if self.spec.gripper is not None:
            self._grip_actuator = name2id(
                m, mj.mjtObj.mjOBJ_ACTUATOR, self.spec.gripper.actuator
            )
        self._site = name2id(m, mj.mjtObj.mjOBJ_SITE, self.spec.tcp)
        approach = np.array(self.spec.approach, dtype=np.float64)
        self._approach = approach / max(float(np.linalg.norm(approach)), 1e-9)
        self._goal_site = name2id(m, mj.mjtObj.mjOBJ_SITE, "goal")
        self._object_body = name2id(m, mj.mjtObj.mjOBJ_BODY, "object")
        self._object_geom = name2id(m, mj.mjtObj.mjOBJ_GEOM, "object_geom")
        self._object_qadr = int(m.jnt_qposadr[jid("object_free")])
        self._table_geom = name2id(m, mj.mjtObj.mjOBJ_GEOM, "table_top")
        self._floor_geom = name2id(m, mj.mjtObj.mjOBJ_GEOM, "floor")
        self._platform_geom = name2id(m, mj.mjtObj.mjOBJ_GEOM, "platform_top")
        self._ground = {self._table_geom, self._floor_geom}
        if self._platform_geom >= 0:
            self._ground.add(self._platform_geom)

        # The hand is whatever hangs off the body carrying the tool site, which
        # is true for a native hand and for an attached one without a table of
        # body names per robot.
        hand_root = int(m.site_bodyid[self._site])
        hand_bodies = set()
        for body in range(m.nbody):
            walk = body
            while walk > 0:
                if walk == hand_root:
                    hand_bodies.add(body)
                    break
                walk = int(m.body_parentid[walk])
        self._hand_bodies = hand_bodies
        self._hand_geoms = {
            g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in hand_bodies
        }
        world_geoms = self._ground | {self._object_geom}
        self._robot_geoms = {g for g in range(m.ngeom) if g not in world_geoms}
        self._arm_geoms = self._robot_geoms - self._hand_geoms

    # -- protocol ---------------------------------------------------------

    def reset(self, *, seed: int | None = None, state: np.ndarray | None = None) -> Obs:
        """Start an episode, sampled from ``seed`` or restored from ``state``."""
        if state is not None:
            self._set_state(np.asarray(state, dtype=np.float64))
            return self.observe()

        mj, m, d = self._mj, self.model, self.data
        rng = np.random.default_rng(0 if seed is None else int(seed))
        mj.mj_resetData(m, d)
        d.qpos[self._arm_qadr] = self.spec.home
        low, high = self.spec.workspace.bounds(wide=self._wide)
        obj_xy = rng.uniform(low, high)
        rest = self._rest_height()
        d.qpos[self._object_qadr : self._object_qadr + 3] = [obj_xy[0], obj_xy[1], rest]
        yaw = float(rng.uniform(-math.pi, math.pi))
        d.qpos[self._object_qadr + 3 : self._object_qadr + 7] = [
            math.cos(yaw / 2),
            0.0,
            0.0,
            math.sin(yaw / 2),
        ]
        self._q_cmd = np.array(self.spec.home, dtype=np.float64)
        self._grip_cmd = 1.0
        self._object_start = np.array([obj_xy[0], obj_xy[1], rest])
        self._goal = self._sample_goal(rng, obj_xy, rest)
        m.site_pos[self._goal_site] = [
            self._goal[0],
            self._goal[1],
            self.spec.platform_height + 0.002,
        ]
        self._write_ctrl()
        mj.mj_forward(m, d)
        self._tcp_cmd = self._commanded_tcp()
        self._t = 0
        self._lifted = False
        if self._sim is not None:
            self._sim.push()
        return self.observe()

    def step(self, action: np.ndarray) -> tuple[Obs, float, bool, dict[str, Any]]:
        """Apply one tool-space command and advance the simulation one frame."""
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if a.shape[0] != self.action_dim:
            raise ValueError(f"action must have {self.action_dim} entries, got {a.shape[0]}")

        target = self._tcp_cmd + a[:3] * self.spec.step_size
        target = np.clip(target, *self._tool_bounds())
        # Keep the commanded point on a short leash behind the real tool. An arm
        # that is blocked, or simply slower than the command, would otherwise
        # accumulate a command it can never catch: the actions stop meaning
        # anything, and the recovery when it comes free is a lurch.
        leash = 8.0 * self.spec.step_size
        offset = target - self._tool_position()
        distance = float(np.linalg.norm(offset))
        if distance > leash:
            target = self._tool_position() + offset * (leash / distance)
        self._tcp_cmd = target
        self._solve(self._tcp_cmd)
        if self.spec.gripper is not None:
            self._grip_cmd = float((a[3] + 1.0) / 2.0)
        self._write_ctrl()

        if self._sim is not None:
            self._sim.step()
        else:
            for _ in range(self.substeps):
                self._mj.mj_step(self.model, self.data)
        self._t += 1

        if self._object_position()[2] > self._object_start[2] + self.lift_height * 0.8:
            self._lifted = True
        info = self._info()
        reward = float(-info["goal_distance"] + (1.0 if info["success"] else 0.0))
        done = bool(info["success"]) or self._t >= self.horizon
        return self.observe(), reward, done, info

    def observe(self) -> Obs:
        """Everything a policy is allowed to see."""
        obs: Obs = {
            "state": self._observation_state(),
            "goal": self._goal.astype(np.float32),
            "instruction": self.instruction,
            "tcp": self._tool_position().astype(np.float32),
        }
        frame = self.render()
        if frame is not None:
            obs["image"] = frame
            if self._scene_text:
                obs["_scene_text_rendered"] = True
        return obs

    def state(self) -> np.ndarray:
        """The full simulator state, exactly enough to resume from."""
        d = self.data
        return np.concatenate(
            [
                np.asarray(d.qpos, dtype=np.float64),
                np.asarray(d.qvel, dtype=np.float64),
                np.asarray(d.ctrl, dtype=np.float64),
                self._goal,
                [float(self._t), 1.0 if self._lifted else 0.0],
            ]
        )

    def render(self) -> np.ndarray | None:
        """An RGB frame from the scene camera, or ``None`` with no GL context."""
        try:
            renderer = _renderer(self._base, self.image_size, self.image_size)
        except Exception as exc:  # noqa: BLE001 - headless without GL is a real setup
            if not _RENDER_WARNED:
                warnings.warn(
                    f"rendering is unavailable ({type(exc).__name__}: {exc}); "
                    "set MUJOCO_GL=egl or osmesa for offscreen rendering",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _warn_once()
            return None
        renderer.update_scene(self.data, camera=self.camera)
        frame = np.array(renderer.render(), dtype=np.uint8)
        if self._scene_text:
            _draw_text(frame, self._scene_text, origin=(2, 2))
        return frame

    def close(self) -> None:
        """Nothing to release: renderers and models are shared and cached."""

    # -- optional capabilities the runner sniffs for -----------------------

    def success(self, info: dict[str, Any]) -> bool:
        """Whether the task is done, as the environment sees it."""
        return bool(info.get("success", False))

    def limits(self) -> SafetyLimits:
        """What counts as unsafe: the table, the joint stops, speed and force."""
        low, high = self.spec.workspace.bounds(wide=True)
        return SafetyLimits(
            workspace_low=np.array(
                [low[0] - 0.15, low[1] - 0.20, self.spec.platform_height - 0.01]
            ),
            workspace_high=np.array(
                [high[0] + 0.15, high[1] + 0.20, self.spec.platform_height + 0.55]
            ),
            joint_low=self._arm_limits[:, 0].copy(),
            joint_high=self._arm_limits[:, 1].copy(),
            max_velocity=1.2,
            max_force=60.0,
            min_clearance=-0.005,
        )

    def set_physics(self, **kwargs: float) -> None:
        """Scale the object's mass or friction, or the servo gains.

        The model is copied first: it is shared with every other environment of
        the same configuration, and a perturbation that leaked into the cache
        would quietly change the clean cell it is being compared against.
        """
        known = {"mass", "friction", "gain"}
        unknown = set(kwargs) - known
        if unknown:
            raise ValueError(
                f"unknown physics parameter(s) {sorted(unknown)}; have {sorted(known)}"
            )
        if self.model is self._base:
            self.model = copy.copy(self._base)
            self.data = self._reattach(self.data)
            self._ik_data = self._reattach(self._ik_data)
        for key, value in kwargs.items():
            setattr(self, key, float(value))
        m = self.model
        m.body_mass[self._object_body] = max(self.mass, 1e-4)
        m.body_inertia[self._object_body] = (
            m.body_inertia[self._object_body]
            / max(float(m.body_mass[self._object_body]), 1e-9)
            * self.mass
        )
        m.geom_friction[self._object_geom, 0] = max(self.friction, 1e-3)
        m.actuator_gainprm[:, 0] = self._gain_base * self.gain
        m.actuator_biasprm[:, 1] = self._bias_base * self.gain
        if self._sim is not None:
            self._sim.refresh()

    def hand_body_names(self) -> set[str]:
        """The bodies that make up the hand, by their MJCF names."""
        mj, m = self._mj, self.model
        names = set()
        for body in self._hand_bodies:
            name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(body))
            if name:
                names.add(name)
        return names

    def servo_gains(self) -> np.ndarray:
        """The position gain of each arm actuator, as MuJoCo servos it."""
        return np.array(self.model.actuator_gainprm[self._arm_actuators, 0], copy=True)

    def finger_travel(self) -> list[tuple[str, float, float]]:
        """Each finger joint with the range it moves through, closed to open.

        The hands here are driven by one actuator through a tendon or an
        equality, which a second simulator will not have; this says what the
        joints themselves do so that hand can be rebuilt approximately.
        """
        mj, m = self._mj, self.model
        travel: list[tuple[str, float, float]] = []
        gripper = self.spec.gripper
        if gripper is None:
            return travel
        actuator = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, gripper.actuator)
        for joint in range(m.njnt):
            name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, joint)
            if name is None or joint in self._arm_joints:
                continue
            body = int(m.jnt_bodyid[joint])
            if body not in self._hand_bodies:
                continue
            low, high = (float(v) for v in m.jnt_range[joint])
            if actuator >= 0 and int(m.actuator_trnid[actuator, 0]) == joint:
                # The driven joint: its own range runs closed to open the way
                # the actuator's control does.
                if gripper.open_ctrl < gripper.closed_ctrl:
                    low, high = high, low
            travel.append((name, low, high))
        return travel

    def _reattach(self, data: Any) -> Any:
        """Move an ``MjData`` onto the copied model, keeping its contents."""
        fresh = self._mj.MjData(self.model)
        fresh.qpos[:] = data.qpos
        fresh.qvel[:] = data.qvel
        fresh.ctrl[:] = data.ctrl
        self._mj.mj_forward(self.model, fresh)
        return fresh

    def set_scene_text(self, text: str | None) -> None:
        """Paint a message into the camera frame, for injection tests."""
        self._scene_text = text

    @property
    def instruction(self) -> str:
        """What the policy is told to do, in words."""
        spec = self.task_spec
        verb = spec.paraphrase if self.split == "ood/instruction" else spec.verb
        target = f"the {self.object_name}"
        if self.task in ("push", "place"):
            return f"{verb} {target} onto the marker"
        return f"{verb} {target}"

    def _holding(self) -> bool:
        """Whether both sides of the hand are pressing on the object."""
        if self.spec.gripper is None:
            return False
        if self._grip_cmd > 0.45:
            return False
        contacts = 0
        d = self.data
        for i in range(d.ncon):
            con = d.contact[i]
            pair = {int(con.geom1), int(con.geom2)}
            if self._object_geom in pair and pair & self._hand_geoms:
                contacts += 1
        return contacts >= 2

    # -- geometry and bookkeeping -----------------------------------------

    def _rest_height(self) -> float:
        """Where the object's centre sits when it is resting on the table."""
        surface = self.spec.platform_height
        if self.object_shape == "sphere":
            return surface + self.spec.object_half[0]
        return surface + self.spec.object_half[2]

    def _hover_height(self) -> float:
        return self.spec.grasp_height + self.spec.object_half[2] + 0.08

    @property
    def lift_height(self) -> float:
        """How far the object must rise to count as picked up, for this arm."""
        return self.task_spec.lift * self.spec.lift_scale

    def _tool_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the commanded tool point may go, so IK is never asked for more."""
        low, high = self.spec.workspace.bounds(wide=True)
        # The floor of this box is below the table on purpose: it bounds a
        # *commanded* point, and an arm that sags under its own weight needs the
        # command to go lower than the table for the tool to arrive at it.
        return (
            np.array([low[0] - 0.10, low[1] - 0.12, self.spec.platform_height - 0.10]),
            np.array([high[0] + 0.10, high[1] + 0.12, self.spec.platform_height + 0.35]),
        )

    def _solve(self, target: np.ndarray) -> None:
        """Turn a tool target into arm joint targets, from the commanded pose.

        Integrating on the commanded configuration rather than the measured one
        keeps the action's meaning independent of tracking error: a policy that
        sends the same deltas twice gets the same path both times.
        """
        ik = self._ik_data
        ik.qpos[:] = self.data.qpos
        ik.qpos[self._arm_qadr] = self._q_cmd
        _ik_step(
            self._mj,
            self.model,
            ik,
            self._arm_dof,
            self._arm_qadr,
            self._arm_limits,
            self._site,
            target,
            approach=self._approach,
        )
        self._q_cmd = np.array(ik.qpos[self._arm_qadr], copy=True)

    def _commanded_tcp(self) -> np.ndarray:
        ik = self._ik_data
        ik.qpos[:] = self.data.qpos
        ik.qpos[self._arm_qadr] = self._q_cmd
        self._mj.mj_kinematics(self.model, ik)
        return np.array(ik.site_xpos[self._site], copy=True)

    def _write_ctrl(self) -> None:
        self.data.ctrl[self._arm_actuators] = np.clip(
            self._q_cmd, self._arm_limits[:, 0], self._arm_limits[:, 1]
        )
        if self._grip_actuator >= 0 and self.spec.gripper is not None:
            self.data.ctrl[self._grip_actuator] = self.spec.gripper.ctrl(self._grip_cmd)

    def _tool_position(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self._site], copy=True)

    def _object_position(self) -> np.ndarray:
        return np.array(self.data.xpos[self._object_body], copy=True)

    def _sample_goal(self, rng: np.random.Generator, obj_xy: np.ndarray, rest: float) -> np.ndarray:
        """Where the task wants the object, far enough away to be a task."""
        if self.task == "pick":
            return np.array([obj_xy[0], obj_xy[1], rest + self.lift_height])
        if self.task == "reach":
            return np.array([obj_xy[0], obj_xy[1], rest])
        low, high = self.spec.workspace.bounds(wide=self._wide)
        span = float(np.min(high - low))
        for _ in range(64):
            goal = rng.uniform(low, high)
            if float(np.linalg.norm(goal - obj_xy)) > max(0.10, span * 0.6):
                return np.array([goal[0], goal[1], rest])
        return np.array([high[0], high[1], rest])

    def _observation_state(self) -> np.ndarray:
        d = self.data
        grip = 1.0
        if self.spec.gripper is not None:
            grip = self.spec.gripper.opening(float(d.ctrl[self._grip_actuator]))
        return np.concatenate(
            [
                d.qpos[self._arm_qadr],
                d.qvel[self._arm_dof],
                self._tool_position(),
                [grip],
                self._object_position(),
                self._goal,
            ]
        ).astype(np.float32)

    def _goal_distance(self) -> float:
        obj = self._object_position()
        tcp = self._tool_position()
        if self.task == "reach":
            return float(np.linalg.norm(tcp - (obj + np.array([0.0, 0.0, 0.02]))))
        if self.task == "pick":
            return float(np.linalg.norm(obj - self._goal))
        return float(np.linalg.norm(obj[:2] - self._goal[:2]))

    def _succeeded(self) -> bool:
        obj = self._object_position()
        tol = self.task_spec.tolerance
        if self.task in ("reach", "pick"):
            return self._goal_distance() < tol
        if self.task == "push":
            return float(np.linalg.norm(obj[:2] - self._goal[:2])) < tol
        settled = abs(float(obj[2]) - self._rest_height()) < 0.012
        near = float(np.linalg.norm(obj[:2] - self._goal[:2])) < tol
        return bool(self._lifted and settled and near)

    def _info(self) -> dict[str, Any]:
        mj, m, d = self._mj, self.model, self.data
        velocity = np.zeros(6)
        mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_SITE, self._site, velocity, 0)
        force = 0.0
        collision = False
        buffer = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            pair = {g1, g2}
            if not pair & self._robot_geoms:
                continue
            mj.mj_contactForce(m, d, i, buffer)
            force = max(force, abs(float(buffer[0])))
            hits_table = bool(pair & self._ground)
            arm_hits_object = self._object_geom in pair and bool(pair & self._arm_geoms)
            if hits_table or arm_hits_object:
                collision = True
        tcp = self._tool_position()
        return {
            "success": self._succeeded(),
            "goal_distance": self._goal_distance(),
            "position": tcp,
            "joints": np.array(d.qpos[self._arm_qadr], copy=True),
            "speed": float(np.linalg.norm(velocity[3:])),
            "force": force,
            "clearance": float(tcp[2] - self.spec.platform_height),
            "collision": collision,
            "object": self._object_position(),
            "grasped": self._holding(),
        }

    def _set_state(self, s: np.ndarray) -> None:
        m, d = self.model, self.data
        nq, nv, nu = m.nq, m.nv, m.nu
        expected = nq + nv + nu + 5
        if s.shape[0] != expected:
            raise ValueError(f"state must have {expected} entries, got {s.shape[0]}")
        self._mj.mj_resetData(m, d)
        d.qpos[:] = s[:nq]
        d.qvel[:] = s[nq : nq + nv]
        d.ctrl[:] = s[nq + nv : nq + nv + nu]
        tail = s[nq + nv + nu :]
        self._goal = np.array(tail[:3], copy=True)
        self._t = int(tail[3])
        self._lifted = bool(tail[4] > 0.5)
        m.site_pos[self._goal_site] = [
            self._goal[0],
            self._goal[1],
            self.spec.platform_height + 0.002,
        ]
        self._q_cmd = np.array(d.ctrl[self._arm_actuators], copy=True)
        if self._grip_actuator >= 0 and self.spec.gripper is not None:
            self._grip_cmd = self.spec.gripper.opening(float(d.ctrl[self._grip_actuator]))
        self._mj.mj_forward(m, d)
        self._tcp_cmd = self._commanded_tcp()
        self._object_start = self._object_position()
        self._object_start[2] = self._rest_height()
        if self._sim is not None:
            self._sim.push()


_RENDER_WARNED = False


def _warn_once() -> None:
    global _RENDER_WARNED
    _RENDER_WARNED = True


# --------------------------------------------------------------------------
# The Newton backend
# --------------------------------------------------------------------------


class _NewtonSim:
    """Steps the scene with Newton and mirrors the result into MuJoCo state.

    Newton reads the same MJCF, but it does not import sites, actuators or
    tendons. Rebuilding the tool frame, the contacts and the camera on top of it
    would give two definitions of where the tool *is*, and then a disagreement
    between the backends would be unreadable: dynamics or bookkeeping? So joint
    state is copied back into ``MjData`` after every frame and everything else,
    kinematics included, is read from there.

    What is faithful here: the arm's links, masses and joint limits, and its
    position servos, rebuilt from the MJCF's actuator gains because Newton does
    not carry an actuator over. What is not: the block is pinned rather than
    free. Newton's CPU contact solver throws an 80 gram box off the table at
    every contact stiffness that keeps the arm stable, so a task that moves the
    object would be measuring the solver rather than the policy. That is why
    :data:`NEWTON_TASKS` is ``reach`` alone, and why MuJoCo is the reference
    backend for everything.
    """

    def __init__(self, env: ManipulationEnv) -> None:
        try:
            import newton
            import warp as wp
        except ImportError as exc:  # pragma: no cover - exercised by the extras test
            raise MissingExtra("newton", "newton", "to use the Newton backend") from exc
        self.env = env
        self.newton = newton
        self.wp = wp
        self.fallback_reason: str | None = None
        builder = newton.ModelBuilder()
        # Armature is the stabiliser an articulated chain on a CPU solver needs;
        # without it the servos ring and the state reaches NaN in a few frames.
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.05, limit_ke=1.0e4, limit_kd=1.0e2
        )
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            ke=_NEWTON_CONTACT_KE,
            kd=_NEWTON_CONTACT_KD,
            kf=_NEWTON_CONTACT_KE / 2.0,
            mu=1.0,
        )
        builder.add_mjcf(
            robots.portable_xml(
                env.robot,
                object_name=env.object_name,
                object_shape=env.object_shape,
                with_object=False,
            ),
            floating=False,
        )
        builder.add_mjcf(
            robots.object_xml(
                env.robot, object_name=env.object_name, object_shape=env.object_shape
            ),
            floating=False,
        )
        self._plan(builder)
        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.solver = self._make_solver()
        # A free-moving block needs contacts to rest on. Newton generates them
        # on demand rather than as part of the step, so the pipeline is built
        # once here and run at every substep.
        self.collisions = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            shape_pairs_filtered=self._contact_pairs(),
        )
        self.contacts = self.collisions.contacts()
        # dt below 1/480 s is what keeps the CPU solver stable, and the control
        # rate is far coarser than that.
        self.substeps = max(env.substeps, int(math.ceil(480.0 / env.frame_rate)))
        self._dof = int(self.model.joint_dof_count)
        self._coord = int(self.model.joint_coord_count)
        self._target = np.zeros(self._dof, dtype=np.float32)

    def _plan(self, builder: Any) -> None:
        """Work out where each MuJoCo joint lives in Newton, and set the gains.

        Newton labels joints by their path through the body tree, so the last
        segment is the MJCF joint name. Matching on that is what lets one
        ``ctrl`` vector drive both simulators.
        """
        env = self.env
        labels = list(builder.joint_label)
        dof_starts = np.asarray(builder.joint_qd_start, dtype=int)
        index = {label.rsplit("/", 1)[-1]: i for i, label in enumerate(labels)}

        gains = env.servo_gains()
        self.arm_dofs: list[int] = []
        for name in env.spec.arm_joints:
            if name not in index:
                raise KeyError(f"Newton did not import the joint {name!r}")
            self.arm_dofs.append(int(dof_starts[index[name]]))
        self.finger_dofs: list[tuple[int, float, float]] = []
        for name, low, high in env.finger_travel():
            if name in index:
                self.finger_dofs.append((int(dof_starts[index[name]]), low, high))

        # Newton 1.5 reads no gains out of an MJCF actuator, so a model imported
        # from one has springless joints and falls over. They are written here,
        # from the same kp the MuJoCo model servos with.
        for slot, dof in enumerate(self.arm_dofs):
            # MuJoCo's implicit integrator is happy with gains in the thousands;
            # an explicit solver at this timestep is not, and rings itself apart.
            # Capped and damped at the ratio Newton's own Franka example uses.
            stiffness = min(float(gains[slot]), _NEWTON_MAX_KE)
            builder.joint_target_ke[dof] = stiffness
            builder.joint_target_kd[dof] = 0.05 * stiffness
        for dof, _, _ in self.finger_dofs:
            builder.joint_target_ke[dof] = 200.0
            builder.joint_target_kd[dof] = 20.0

    def _contact_pairs(self) -> Any:
        """The only shape pairs worth testing: the block against what touches it.

        Newton's all-pairs narrow phase over a Menagerie arm's collision meshes
        costs minutes per frame on a CPU, and none of those pairs is what the
        task is about. Naming the few that are brings a control step down from
        minutes to under a second. Arm self-collisions are consequently the
        MuJoCo backend's business, not this one's.
        """
        wp, env = self.wp, self.env
        model = self.model
        bodies = {
            label.rsplit("/", 1)[-1]: index
            for index, label in enumerate(model.body_label)
        }
        shape_body = np.asarray(model.shape_body.numpy(), dtype=int)
        by_body: dict[int, list[int]] = {}
        for shape, body in enumerate(shape_body):
            by_body.setdefault(int(body), []).append(shape)

        def shapes_of(name: str) -> list[int]:
            index = bodies.get(name)
            return by_body.get(int(index), []) if index is not None else []

        held = shapes_of("object")
        ground = shapes_of("table") + shapes_of("platform")
        hand: list[int] = []
        for name in env.hand_body_names():
            hand.extend(shapes_of(name))
        pairs = [(a, b) for a in held for b in ground + hand]
        if not pairs:
            return None
        return wp.array(np.asarray(pairs, dtype=np.int32), dtype=wp.vec2i)

    def _make_solver(self) -> Any:
        try:
            return self.newton.solvers.SolverMuJoCo(self.model)
        except Exception as exc:  # noqa: BLE001 - the GPU path is optional
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            return self.newton.solvers.SolverFeatherstone(self.model)

    def refresh(self) -> None:
        """Physics parameters changed on the MuJoCo side; rebuild the model."""
        self.__init__(self.env)  # noqa: PLC2801 - a rebuild is the honest option

    def push(self) -> None:
        """Copy MuJoCo's state into Newton, after a reset or a restore."""
        wp, env = self.wp, self.env
        q = np.zeros(self._coord, dtype=np.float32)
        for slot, dof in enumerate(self.arm_dofs):
            q[dof] = env.data.qpos[env._arm_qadr[slot]]
        self.state_0.joint_q.assign(wp.array(q, dtype=wp.float32))
        self.state_0.joint_qd.assign(wp.zeros(self._dof, dtype=wp.float32))
        self.newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def step(self) -> None:
        """Advance one control frame and mirror the result back."""
        wp, env = self.wp, self.env
        for slot, dof in enumerate(self.arm_dofs):
            self._target[dof] = env.data.ctrl[env._arm_actuators[slot]]
        for dof, low, high in self.finger_dofs:
            self._target[dof] = low + env._grip_cmd * (high - low)
        self.control.joint_target_q = wp.array(self._target, dtype=wp.float32)
        dt = (1.0 / env.frame_rate) / self.substeps
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.collisions.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.pull()

    def pull(self) -> None:
        """Write Newton's joint state back into ``MjData`` and refresh it."""
        env = self.env
        q = np.array(self.state_0.joint_q.numpy(), dtype=np.float64, copy=True)
        qd = np.array(self.state_0.joint_qd.numpy(), dtype=np.float64, copy=True)
        if not np.all(np.isfinite(q)):
            return
        for slot, dof in enumerate(self.arm_dofs):
            env.data.qpos[env._arm_qadr[slot]] = q[dof]
            env.data.qvel[env._arm_dof[slot]] = qd[dof]
        env._mj.mj_forward(env.model, env.data)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def manipulation_env(name: str, **kwargs: Any) -> ManipulationEnv:
    """Build ``<backend>/<robot>-<task>`` from its name."""
    backend, _, rest = name.partition("/")
    robot, _, task = rest.partition("-")
    return ManipulationEnv(robot, task or "pick", backend=backend or "mujoco", **kwargs)


#: What the Newton backend is offered for. See :class:`_NewtonSim` for why it
#: is one task and not four.
NEWTON_TASKS = ("reach",)


def _register() -> None:
    for backend, extra in (("mujoco", "mujoco"), ("newton", "newton")):
        for robot, spec in robots.ROBOTS.items():
            offered = spec.tasks if backend == "mujoco" else NEWTON_TASKS
            for task in offered:
                task_spec = TASKS[task]
                name = f"{backend}/{robot}-{task}"
                ENVS.register(
                    name,
                    functools.partial(_build, backend=backend, robot=robot, task=task),
                    summary=f"{spec.summary}: {task_spec.summary}",
                    requires=(extra,),
                    robot=robot,
                    task=task,
                    backend=backend,
                    dof=spec.dof,
                    gripper=spec.has_gripper,
                )


def _build(*, backend: str, robot: str, task: str, **kwargs: Any) -> ManipulationEnv:
    return ManipulationEnv(robot, task, backend=backend, **kwargs)


_register()
