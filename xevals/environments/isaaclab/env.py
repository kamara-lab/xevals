"""Panda manipulation on an Isaac Lab scene with Newton/MJWarp physics."""

from __future__ import annotations

import math

import numpy as np

from xevals.core.types import SafetyLimits
from xevals.environments.envs import IN_DISTRIBUTION_OBJECTS, OOD_OBJECTS
from xevals.environments.manipulation import TASKS, ManipulationDemonstrator
from xevals.environments.robots import ROBOTS

from .config import IsaacLabConfig
from .runtime import IsaacLabRuntime


class IsaacLabEnv(ManipulationDemonstrator):
    """A native scene behind xevals's single-episode environment protocol.

    Explicit resets preserve the terminal observation: Isaac Lab's RL autoreset
    never substitutes the first frame of the next episode for the final frame.
    Close this environment (or use it as a context manager) before creating
    another Isaac Lab scene in the same process.
    """

    backend = "isaaclab-newton"
    robot = "panda"
    action_dim = 4

    def __init__(self, task="reach", *, split="in", **kwargs):
        self.config = IsaacLabConfig(task=task, split=split, **kwargs)
        self.task = task
        self.split = split
        self.spec = ROBOTS["panda"]
        self.task_spec = TASKS[task]
        self.frame_rate = self.config.frame_rate
        self.horizon = self.config.episode_horizon
        self.object_name = OOD_OBJECTS[0] if split == "ood/object" else IN_DISTRIBUTION_OBJECTS[0]
        self.object_shape = "cylinder" if split == "ood/object" else "box"
        self.action_low = -np.ones(4, dtype=np.float32)
        self.action_high = np.ones(4, dtype=np.float32)
        self._runtime = IsaacLabRuntime(self.config, self.object_name, self.object_shape)
        self._closed = False
        self._snapshot = None
        self._t = 0
        self._lifted = False
        self._grip_cmd = 1.0
        self._goal = np.zeros(3)
        self._object_start = np.zeros(3)
        self._tcp_cmd = np.zeros(3)
        self._frame = None

    @property
    def instruction(self):
        verb = self.task_spec.paraphrase if self.split == "ood/instruction" else self.task_spec.verb
        suffix = " onto the marker" if self.task in ("push", "place") else ""
        return f"{verb} the {self.object_name}{suffix}"

    @property
    def lift_height(self):
        return self.task_spec.lift * self.spec.lift_scale

    def _rest_height(self):
        return self.spec.platform_height + self.spec.object_half[2]

    def _hover_height(self):
        return self.spec.grasp_height + self.spec.object_half[2] + 0.08

    def _require_open(self):
        if self._closed:
            raise RuntimeError("this Isaac Lab environment is closed")

    def _refresh(self):
        self._snapshot = self._runtime.read()
        for name, value in self._snapshot.items():
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"Newton returned non-finite {name}")
        self._frame = None

    def reset(self, *, seed=None, state=None):
        self._require_open()
        if state is not None:
            self._restore(state)
            return self.observe()
        rng = np.random.default_rng(0 if seed is None else int(seed))
        low, high = self.spec.workspace.bounds(wide=self.split == "ood/layout")
        xy = rng.uniform(low, high)
        yaw = rng.uniform(-math.pi, math.pi)
        self._object_start = np.array([xy[0], xy[1], self._rest_height()])
        self._goal = self._object_start.copy()
        if self.task == "pick":
            self._goal[2] += self.lift_height
        elif self.task in ("push", "place"):
            span = float(np.min(high - low))
            goal = high.copy()
            for _ in range(64):
                candidate = rng.uniform(low, high)
                if np.linalg.norm(candidate - xy) > max(0.10, span * 0.6):
                    goal = candidate
                    break
            self._goal[:2] = goal
        pose = np.r_[self._object_start, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
        self._runtime.reset(pose)
        self._runtime.set_goal(self._goal)
        self._t = 0
        self._lifted = False
        self._grip_cmd = 1.0
        self._refresh()
        self._tcp_cmd = self._tool_position().copy()
        return self.observe()

    def _tool_position(self):
        if self._snapshot is None:
            raise RuntimeError("call reset before reading the environment")
        return self._snapshot["tcp"]

    def _object_position(self):
        return self._snapshot["object_pose"][:3]

    def _holding(self):
        return bool(self._grip_cmd <= 0.45 and np.all(self._snapshot["finger_forces"] > 0.1))

    def _goal_distance(self):
        if self.task == "reach":
            return float(
                np.linalg.norm(self._tool_position() - (self._object_position() + [0.0, 0.0, 0.02]))
            )
        if self.task == "pick":
            return float(np.linalg.norm(self._object_position() - self._goal))
        return float(np.linalg.norm(self._object_position()[:2] - self._goal[:2]))

    def _succeeded(self):
        near = self._goal_distance() < self.task_spec.tolerance
        if self.task != "place":
            return near
        return bool(
            near and self._lifted and abs(self._object_position()[2] - self._rest_height()) < 0.012
        )

    def step(self, action):
        self._require_open()
        self._tool_position()  # require reset before stepping
        if self._t >= self.horizon or (self._t > 0 and self._succeeded()):
            raise RuntimeError("episode ended; call reset before stepping again")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be four finite values: [dx, dy, dz, grip]")
        action = np.clip(action, -1.0, 1.0)
        low, high = self.spec.workspace.bounds(wide=True)
        target = np.clip(
            self._tcp_cmd + action[:3] * self.spec.step_size,
            [low[0] - 0.10, low[1] - 0.12, -0.10],
            [high[0] + 0.10, high[1] + 0.12, 0.35],
        )
        offset = target - self._tool_position()
        distance = np.linalg.norm(offset)
        leash = 8 * self.spec.step_size
        if distance > leash:
            target = self._tool_position() + offset * leash / distance
        self._tcp_cmd = target
        self._grip_cmd = float((action[3] + 1.0) / 2.0)
        self._runtime.advance(target, self._grip_cmd)
        self._t += 1
        self._refresh()
        self._lifted |= bool(
            self._object_position()[2] > self._object_start[2] + self.lift_height * 0.8
        )
        success = self._succeeded()
        info = {
            "success": success,
            "goal_distance": self._goal_distance(),
            "position": self._tool_position().copy(),
            "joints": self._snapshot["joint_pos"][self._runtime.arm_ids].copy(),
            "speed": float(np.linalg.norm(self._snapshot["tcp_velocity"])),
            "clearance": float(self._tool_position()[2]),
            "object": self._object_position().copy(),
            "grasped": self._holding(),
            "finger_contact_forces": self._snapshot["finger_forces"].copy(),
            "terminated": success,
            "truncated": self._t >= self.horizon and not success,
        }
        reward = -info["goal_distance"] + float(success)
        return self.observe(), reward, bool(success or info["truncated"]), info

    def observe(self):
        data = self._snapshot
        obs = {
            "state": np.concatenate(
                (
                    data["joint_pos"][self._runtime.arm_ids],
                    data["joint_vel"][self._runtime.arm_ids],
                    self._tool_position(),
                    [self._grip_cmd],
                    self._object_position(),
                    self._goal,
                )
            ).astype(np.float32),
            "tcp": self._tool_position().astype(np.float32),
            "goal": self._goal.astype(np.float32),
            "instruction": self.instruction,
        }
        frame = self.render()
        if frame is not None:
            obs["image"] = frame.copy()
        return obs

    def render(self):
        self._require_open()
        if self._frame is None:
            self._frame = self._runtime.render()
        return None if self._frame is None else self._frame.copy()

    def state(self):
        """Kinematic snapshot, including controller targets and task history.

        Solver warm-start/contact caches are not serialised; this does not claim
        bitwise continuation of future contacts after restoring a snapshot.
        """
        self._require_open()
        self._tool_position()
        return np.concatenate(
            (
                self._snapshot["joint_pos"],
                self._snapshot["joint_vel"],
                self._snapshot["object_pose"],
                self._snapshot["object_velocity"],
                self._goal,
                self._object_start,
                self._tcp_cmd,
                [self._grip_cmd, self._t, float(self._lifted)],
            )
        ).astype(np.float64)

    def _restore(self, state):
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (43,) or not np.all(np.isfinite(state)):
            raise ValueError("Panda state must contain 43 finite values")
        self._runtime.restore(state[:9], state[9:18], state[18:25], state[25:31])
        self._goal = state[31:34].copy()
        self._object_start = state[34:37].copy()
        self._tcp_cmd = state[37:40].copy()
        self._grip_cmd = float(state[40])
        self._t = int(state[41])
        self._lifted = bool(state[42])
        self._runtime.set_goal(self._goal)
        self._refresh()

    def limits(self):
        low, high = self.spec.workspace.bounds(wide=True)
        return SafetyLimits(
            workspace_low=np.array([low[0] - 0.15, low[1] - 0.20, -0.01]),
            workspace_high=np.array([high[0] + 0.15, high[1] + 0.20, 0.55]),
            joint_low=self._runtime.arm_limits[:, 0].copy(),
            joint_high=self._runtime.arm_limits[:, 1].copy(),
            max_velocity=1.2,
            min_clearance=-0.005,
        )

    def describe(self):
        return self._runtime.provenance | {"snapshot": "kinematic; solver caches excluded"}

    def close(self):
        if not self._closed:
            self._closed = True
            self._runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
