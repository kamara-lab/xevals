"""Task semantics and adapter contracts, independent of GPU availability.

These tests do not claim to validate contact physics; the backend's --validate
command is the separate integration gate on the pinned Isaac Lab runtime.
"""

from __future__ import annotations

import numpy as np
import pytest

import xevals
from xevals.environments.isaaclab import IsaacLabConfig, IsaacLabEnv
from xevals.environments.isaaclab import env as port
from xevals.environments.isaaclab.runtime import ik_delta, rotation_xyzw
from xevals.environments.manipulation import TASKS
from xevals.environments.sim import ManipulationEnv


class RuntimeDouble:
    arm_ids = list(range(7))
    arm_limits = np.tile([-3.0, 3.0], (7, 1))

    def __init__(self, config, object_name, object_shape):
        self.provenance = config.describe()
        self.closed = False
        self.close_calls = 0
        self.q = np.zeros(9)
        self.qd = np.zeros(9)
        self.pose = np.array([0.46, 0.0, 0.025, 0.0, 0.0, 0.0, 1.0])
        self.velocity = np.zeros(6)
        self.tcp = np.array([0.46, 0.0, 0.20])

    def read(self):
        if self.closed:
            raise RuntimeError("runtime already closed")
        return {
            "joint_pos": self.q.copy(),
            "joint_vel": self.qd.copy(),
            "object_pose": self.pose.copy(),
            "object_velocity": self.velocity.copy(),
            "tcp": self.tcp.copy(),
            "tcp_velocity": np.zeros(3),
            "finger_forces": np.zeros(2),
        }

    def set_goal(self, goal):
        pass

    def reset(self, pose):
        self.restore(np.zeros(9), np.zeros(9), pose, np.zeros(6))
        self.tcp = np.array([0.46, 0.0, 0.20])

    def restore(self, q, qd, pose, velocity):
        self.q, self.qd, self.pose, self.velocity = (
            np.array(a, copy=True) for a in (q, qd, pose, velocity)
        )

    def advance(self, target, opening):
        self.tcp = target.copy()

    def render(self):
        return None

    def close(self):
        self.closed = True
        self.close_calls += 1


@pytest.fixture
def scene(monkeypatch):
    monkeypatch.setattr(port, "IsaacLabRuntime", RuntimeDouble)
    with IsaacLabEnv(renderer="none") as env:
        yield env


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task": "bad"},
        {"split": "bad"},
        {"device": "cpu"},
        {"renderer": "bad"},
        {"camera": "bad"},
        {"frame_rate": 0},
        {"physics_dt": float("nan")},
        {"physics_dt": 0.007},
        {"substeps": 0},
        {"horizon": -1},
        {"iterations": 2.5},
    ],
)
def test_config_rejects_invalid_requests_before_loading_simulator(kwargs):
    with pytest.raises(ValueError):
        IsaacLabConfig(**kwargs)


def test_default_control_period_and_solver_settings():
    cfg = IsaacLabConfig()
    assert cfg.decimation == 12
    assert cfg.physics_dt * cfg.decimation == pytest.approx(1 / cfg.frame_rate)
    assert cfg.describe()["solver"] == "mujoco_warp"


def test_registry_lists_all_four_native_tasks():
    assert xevals.envs.available("isaaclab-newton") == [
        f"isaaclab-newton/panda-{t}" for t in sorted(TASKS)
    ]


def test_reset_is_repeatable_and_snapshot_includes_task_history(scene):
    first = scene.reset(seed=12)
    state = scene.state()
    scene.reset(seed=123)
    scene.reset(state=state)
    np.testing.assert_array_equal(scene.state(), state)
    np.testing.assert_array_equal(scene.observe()["goal"], first["goal"])
    scene.reset(seed=12)
    np.testing.assert_array_equal(scene.state(), state)


def test_action_contract_and_explicit_reset(scene):
    with pytest.raises(RuntimeError, match="reset"):
        scene.step(np.zeros(4))
    scene.reset(seed=0)
    for action in ([1, 2], [0, 0, 0, np.nan], np.zeros((1, 4))):
        with pytest.raises(ValueError):
            scene.step(action)


def test_terminal_observation_is_not_an_autoreset_frame(scene):
    scene.reset(seed=0)
    scene._runtime.tcp = scene._object_position() + [0, 0, 0.02]
    scene._refresh()
    scene._tcp_cmd = scene._tool_position().copy()
    obs, reward, done, info = scene.step(np.array([0, 0, 0, 1]))
    assert done and info["success"] and info["terminated"] and not info["truncated"]
    assert reward == pytest.approx(1.0)
    np.testing.assert_allclose(obs["tcp"], info["object"] + [0, 0, 0.02])
    with pytest.raises(RuntimeError, match="episode ended"):
        scene.step(np.zeros(4))


def test_timeout_is_distinct_from_success(scene):
    scene.horizon = 1
    scene.reset(seed=0)
    _, _, done, info = scene.step(np.array([0, 0, 0, 1]))
    assert done and info["truncated"] and not info["success"]


@pytest.mark.parametrize("task", TASKS)
def test_success_geometry_matches_reference_backend(scene, task):
    scene.task, scene.task_spec = task, TASKS[task]
    scene.reset(seed=3)
    # Invoke the reference predicates with the same geometry, including states
    # near both sides of each success threshold and place's lift-history gate.
    rng = np.random.default_rng(91)
    for _ in range(100):
        scene._snapshot["object_pose"][:3] = scene._goal + rng.uniform(-0.1, 0.1, 3)
        scene._snapshot["tcp"] = scene._object_position() + rng.uniform(-0.06, 0.06, 3)
        scene._lifted = bool(rng.integers(2))
        assert scene._goal_distance() == pytest.approx(ManipulationEnv._goal_distance(scene))
        assert scene._succeeded() == ManipulationEnv._succeeded(scene)


def test_unmeasured_contacts_are_not_reported_as_zero_collision(scene):
    scene.reset(seed=2)
    _, _, _, info = scene.step(np.zeros(4))
    assert "collision" not in info and "force" not in info
    assert scene.limits().max_force is None


def test_evaluation_does_not_close_borrowed_isaac_scene(scene):
    result = xevals.evaluate(
        lambda obs: np.zeros(4),
        scene,
        suite="smoke",
        episodes=2,
        horizon=2,
        baselines=False,
        out=None,
        record=0,
        verbose=False,
    )
    assert not scene._runtime.closed
    assert result.cells["clean"]["accuracy/success_rate"].n == 2
    assert result.env["configuration"]["solver"] == "mujoco_warp"
    scene.close()
    scene.close()
    assert scene._runtime.close_calls == 1


def test_xyzw_rotation_and_top_down_ik():
    np.testing.assert_allclose(rotation_xyzw([0, 0, 0, 1]), np.eye(3))
    np.testing.assert_allclose(rotation_xyzw([1, 0, 0, 0])[:, 2], [0, 0, -1])
    delta = ik_delta(np.eye(6), np.array([0.1, 0, 0]), np.array([0, 0, -1]))
    assert delta[0] > 0
    np.testing.assert_allclose(delta[1:], 0)


def test_recording_includes_terminal_frame_at_the_control_rate(scene, monkeypatch, tmp_path):
    from xevals.evaluation.runner import rollout
    from xevals.reporting import media

    monkeypatch.setattr(scene._runtime, "render", lambda: np.full((8, 8, 3), scene._t, np.uint8))
    trajectory = rollout(xevals.wrap(lambda obs: np.zeros(4)), scene, horizon=2, record=True)
    assert [frame[0, 0, 0] for frame in trajectory.frames] == [0, 1, 2]
    result = xevals.evaluate(
        lambda obs: np.zeros(4),
        scene,
        suite="smoke",
        episodes=1,
        horizon=2,
        control_hz=20.0,
        baselines=False,
        out=None,
        verbose=False,
    )
    rates = []
    monkeypatch.setattr(media, "write_all", lambda result, directory, **kw: rates.append(kw["fps"]))
    result._write_videos(tmp_path)
    assert rates == [20.0]


def test_joint_targets_are_held_for_one_control_period(monkeypatch):
    from types import SimpleNamespace

    from xevals.environments.isaaclab import runtime as native

    runtime = native.IsaacLabRuntime.__new__(native.IsaacLabRuntime)
    runtime.config = IsaacLabConfig(renderer="none")
    runtime.hand_id, runtime.arm_ids, runtime.finger_ids = 1, list(range(7)), [7, 8]
    runtime.arm_limits = np.tile([-3.0, 3.0], (7, 1))
    jacobian = np.zeros((1, 1, 6, 9))
    jacobian[0, 0, :, :6] = np.eye(6)
    targets, steps = [], []
    runtime.robot = SimpleNamespace(
        is_fixed_base=True,
        num_base_dofs=0,
        data=SimpleNamespace(body_link_jacobian_w=jacobian),
        set_joint_position_target_index=lambda **kw: targets.append(kw["target"]),
    )
    runtime.scene = SimpleNamespace(write_data_to_sim=lambda: None, update=lambda dt: None)
    runtime.sim = SimpleNamespace(step=lambda **kw: steps.append(kw))
    runtime._tensor = np.asarray
    runtime.read = lambda: {
        "joint_pos": np.zeros(9),
        "tcp": np.zeros(3),
        "tcp_axis": np.array([0.0, 0.0, -1.0]),
        "tcp_offset": np.zeros(3),
    }
    monkeypatch.setattr(native, "_numpy", np.asarray)
    runtime.advance(np.array([0.01, 0.0, 0.0]), 0.5)
    assert len(targets) == 1
    assert len(steps) == runtime.config.decimation
    np.testing.assert_allclose(targets[0][0, 7:], 0.02)


def test_place_lifts_before_transport_even_when_already_over_marker(monkeypatch):
    monkeypatch.setattr(port, "IsaacLabRuntime", RuntimeDouble)
    with IsaacLabEnv("place", renderer="none") as env:
        env.reset(seed=0)
        env._goal[:2] = env._object_position()[:2]
        env._grip_cmd = 0.0
        env._snapshot["finger_forces"][:] = 5.0
        action = env.optimal_action
        np.testing.assert_allclose(action[:2], 0.0)
        assert action[2] > 0
        assert action[3] == -1


def test_wrist_camera_follows_measured_hand_pose(monkeypatch):
    from types import SimpleNamespace

    from xevals.environments.isaaclab import runtime as native

    runtime = native.IsaacLabRuntime.__new__(native.IsaacLabRuntime)
    runtime.config = IsaacLabConfig(camera="wrist")
    runtime.hand_id = 0
    pose = np.array([[[0.4, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0]]])
    runtime.robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=pose))
    calls = []
    runtime.camera = SimpleNamespace(
        set_world_poses=lambda **kw: calls.append(kw),
        update=lambda *a, **kw: None,
        data=SimpleNamespace(output={"rgb": np.zeros((1, 8, 8, 3))}),
    )
    runtime.sim = SimpleNamespace(render=lambda: None)
    runtime._tensor = np.asarray
    monkeypatch.setattr(native, "_numpy", np.asarray)
    runtime.render()
    np.testing.assert_allclose(calls[-1]["positions"], [[0.4, 0.1, 0.225]])
    np.testing.assert_allclose(calls[-1]["orientations"], [[1, 0, 0, 0]])
    assert calls[-1]["convention"] == "ros"
    pose[0, 0, 0] += 0.2
    runtime.render()
    np.testing.assert_allclose(calls[-1]["positions"], [[0.6, 0.1, 0.225]])


def test_scene_wrist_inset_uses_two_views_without_stepping_physics(monkeypatch):
    from types import SimpleNamespace

    from xevals.environments.isaaclab.runtime import IsaacLabRuntime

    monkeypatch.setattr("xevals.environments.isaaclab.runtime._numpy", np.asarray)
    runtime = IsaacLabRuntime.__new__(IsaacLabRuntime)
    runtime.config = IsaacLabConfig(camera="scene-wrist", image_size=96)
    runtime.origin = np.zeros(3)
    runtime.hand_id = 0
    runtime._tensor = np.asarray
    runtime.robot = SimpleNamespace(
        data=SimpleNamespace(body_link_pose_w=np.array([[[0, 0, 0, 0, 0, 0, 1]]]))
    )
    poses = []
    runtime.camera = SimpleNamespace(
        set_world_poses_from_view=lambda **kw: poses.append("external"),
        set_world_poses=lambda **kw: poses.append("wrist"),
    )
    frames = iter([np.full((96, 96, 3), 10, np.uint8), np.full((96, 96, 3), 90, np.uint8)])
    runtime._camera_frame = lambda: next(frames)
    output = runtime.render()
    assert poses == ["external", "wrist"]
    np.testing.assert_array_equal(output[50:, :], 10)
    np.testing.assert_array_equal(output[4:36, 60:92], 90)
    np.testing.assert_array_equal(output[2:4, 58:94], 255)
