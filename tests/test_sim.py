"""The Menagerie manipulation robots, their scene, and the tasks on them."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from xevals import envs, robots  # noqa: E402
from xevals.envs import SPLITS, check_reset_invariant  # noqa: E402
from xevals.sim import TASKS, ManipulationEnv  # noqa: E402
from xevals.types import Env  # noqa: E402

pytestmark = pytest.mark.skipif(
    not robots.assets_available(),
    reason="the Menagerie models are not in the cache; set XEVALS_MENAGERIE or fetch them",
)

ROBOTS = sorted(robots.ROBOTS)


@pytest.fixture(scope="module")
def panda() -> ManipulationEnv:
    """One environment for the module: compiling a Menagerie model is not free."""
    return ManipulationEnv("panda", "pick", image_size=32)


def _run(env: ManipulationEnv, seed: int) -> dict:
    env.reset(seed=seed)
    info: dict = {}
    done = False
    while not done:
        _obs, _reward, done, info = env.step(env.optimal_action)
    return info


@pytest.mark.parametrize("robot", ROBOTS)
def test_every_robot_composes_into_a_scene_with_a_tool_and_a_hand(robot):
    env = ManipulationEnv(robot, "reach", image_size=32)
    assert env.action_dim == 4, "one action space for every arm, whatever its joint count"
    assert env._site >= 0, "the tool frame has to exist to be servoed"
    spec = robots.ROBOTS[robot]
    if spec.gripper is not None:
        assert env._grip_actuator >= 0
        assert env._hand_geoms, "the hand has to be identifiable to detect a grasp"


@pytest.mark.parametrize("robot", ["fr3", "ur5e"])
def test_the_bare_arms_are_given_a_gripper(robot):
    # Without one they could be asked to reach and to push and nothing else,
    # and half the task set would read as failure rather than as absence.
    model = robots.compiled(robot)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper/base_mount") >= 0
    assert robots.ROBOTS[robot].has_gripper


def test_the_environment_satisfies_the_environment_protocol(panda):
    assert isinstance(panda, Env)


@pytest.mark.parametrize("robot", ROBOTS)
def test_reset_from_a_recorded_state_reproduces_it(robot):
    env = ManipulationEnv(robot, "reach", image_size=32)
    assert check_reset_invariant(env) == 0.0


@pytest.mark.parametrize("robot", ROBOTS)
def test_every_scene_carries_the_cameras_a_policy_might_have_been_trained_on(robot):
    model = robots.compiled(robot)
    for camera in ("pov", "wrist", "scene"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) >= 0, camera


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_wrist_camera_looks_at_what_the_hand_is_over(robot):
    # It rides the hand, so its view has to change when the arm moves, and it
    # has to be pointing the way the tool is rather than back up the arm.
    env = ManipulationEnv(robot, "reach", image_size=48, camera="wrist")
    env.reset(seed=0)
    before = env.render()
    for _ in range(12):
        env.step(env.optimal_action)
    assert not np.array_equal(before, env.render())


def test_the_observation_comes_from_the_point_of_view_camera():
    # The one a policy would have been trained through. Scoring a model on a
    # wide studio shot would measure it on an image no robot ever produces.
    default = ManipulationEnv("panda", "pick", image_size=48)
    assert default.camera == "pov"
    wide = ManipulationEnv("panda", "pick", image_size=48, camera="scene")
    default.reset(seed=0)
    wide.reset(seed=0)
    assert not np.array_equal(default.render(), wide.render())


def test_rendering_is_deterministic_given_the_state(panda):
    panda.reset(seed=0)
    first, second = panda.render(), panda.render()
    assert first is not None
    assert first.dtype == np.uint8
    assert first.shape == (32, 32, 3)
    assert np.array_equal(first, second)


def test_the_observation_carries_what_a_policy_and_the_metrics_need(panda):
    obs = panda.reset(seed=0)
    assert {"image", "state", "goal", "instruction"} <= set(obs)
    assert obs["instruction"], "instruction perturbations are a no-op without one"
    _obs, _reward, _done, info = panda.step(np.zeros(4, dtype=np.float32))
    # SafetyLimits.check reads these by name, and the accuracy metrics read the
    # goal distance; a missing key is a metric silently reported as null.
    assert {"success", "goal_distance", "position", "joints", "speed"} <= set(info)
    assert {"force", "clearance", "collision"} <= set(info)


def test_each_split_changes_exactly_one_thing():
    base = ManipulationEnv("panda", "reach", image_size=32)
    for split in SPLITS:
        other = ManipulationEnv("panda", "reach", split=split, image_size=32)
        if split == "in":
            continue
        if split == "ood/object":
            assert other.object_name != base.object_name
            assert other.instruction != base.instruction, "a new object renames itself"
        elif split == "ood/layout":
            wide, _ = other.spec.workspace.bounds(wide=True)
            narrow, _ = other.spec.workspace.bounds()
            assert wide[0] < narrow[0], "the layout split is a wider spawn region"
            assert other.object_name == base.object_name
        elif split == "ood/instruction":
            assert other.object_name == base.object_name
            assert other.instruction != base.instruction


def test_physics_can_be_set_and_rejects_a_parameter_it_does_not_have(panda):
    panda.set_physics(mass=0.25)
    assert panda.model.body_mass[panda._object_body] == pytest.approx(0.25)
    assert panda.model is not panda._base, "the shared cached model must not be edited"
    with pytest.raises(ValueError, match="unknown physics parameter"):
        panda.set_physics(gravity=9.8)


def test_a_dynamics_perturbation_reaches_the_object():
    from xevals import perturbations

    env = ManipulationEnv("panda", "pick", image_size=32)
    before = env.mass
    wrapped = envs.perturbed(env, perturbations.create("dynamics/mass", severity=1.0), seed=0)
    assert wrapped.mass != before


def test_driving_the_tool_into_the_table_is_a_safety_violation():
    env = ManipulationEnv("panda", "reach", image_size=32)
    env.reset(seed=0)
    limits = env.limits()
    violations = []
    for _ in range(40):
        _obs, _reward, _done, info = env.step(np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32))
        violations.extend(limits.check(info))
    assert violations, "an arm pressed into the table has to be measurably unsafe"


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_demonstrator_solves_reaching_so_the_gate_can_mean_something(robot):
    env = ManipulationEnv(robot, "reach", image_size=32)
    solved = sum(bool(_run(env, seed)["success"]) for seed in range(4))
    assert solved >= 3


@pytest.mark.parametrize("robot", ["panda", "fr3", "ur5e", "yam"])
def test_the_demonstrator_clears_the_gate_on_a_grasping_task(robot):
    # The gate passes at a success rate of 0.5, which is the bar that matters:
    # below it the environment cannot tell a working policy from a broken one.
    env = ManipulationEnv(robot, "pick", image_size=32)
    solved = sum(bool(_run(env, seed)["success"]) for seed in range(6))
    assert solved >= 4 if robot != "yam" else solved >= 3


def test_the_demonstrator_is_reliable_on_the_reference_robot():
    for task in ("push", "pick", "place"):
        env = ManipulationEnv("panda", task, image_size=32)
        solved = sum(bool(_run(env, seed)["success"]) for seed in range(6))
        assert solved >= 5, f"panda/{task} solved only {solved}/6"


def test_a_noop_policy_does_not_solve_the_task(panda):
    panda.reset(seed=0)
    info: dict = {}
    done = False
    while not done:
        _obs, _reward, done, info = panda.step(np.zeros(4, dtype=np.float32))
    assert not info["success"], "a task an idle arm solves measures nothing"


def test_an_arm_is_only_offered_the_tasks_it_can_do():
    # The SO-101 cannot bring its tool down onto a block; saying so beats
    # shipping a task every policy fails for a reason that is ours.
    assert robots.ROBOTS["so101"].tasks == ("reach", "push")
    with pytest.raises(ValueError, match="not offered"):
        ManipulationEnv("so101", "pick")
    with pytest.raises(ValueError, match="unknown task"):
        ManipulationEnv("panda", "juggle")
    with pytest.raises(KeyError, match="unknown robot"):
        ManipulationEnv("stretch", "reach")


def test_the_registry_lists_every_robot_and_task_it_offers():
    from xevals.sim import NEWTON_TASKS

    names = [n for n in envs.available() if n.startswith("mujoco/")]
    assert len(names) == sum(len(spec.tasks) for spec in robots.ROBOTS.values())
    assert "mujoco/panda-pick" in names
    # Newton is offered for the one task it reproduces faithfully.
    newton = [n for n in envs.available() if n.startswith("newton/")]
    assert len(newton) == len(robots.ROBOTS) * len(NEWTON_TASKS)
    assert "newton/panda-reach" in newton
    entry = envs.describe("mujoco/panda-pick")
    assert entry["robot"] == "panda"
    assert entry["task"] == "pick"
    assert entry["requires"] == ["mujoco"]


def test_the_shorthand_builds_a_named_scene():
    env = envs.create("mujoco/yam-push", split="ood/object")
    assert env.robot == "yam"
    assert env.task == "push"
    assert env.split == "ood/object"


def test_the_tasks_are_distinct_in_what_they_ask_for():
    assert set(TASKS) == {"reach", "push", "pick", "place"}
    assert TASKS["pick"].needs_gripper and TASKS["place"].needs_gripper
    assert not TASKS["reach"].needs_gripper


def test_the_scene_can_be_described_without_downloading_anything():
    described = robots.describe()
    assert set(described) == set(ROBOTS)
    assert described["panda"]["dof"] == 7
    assert described["so101"]["dof"] == 5
    assert described["fr3"]["gripper"] == "robotiq_2f85"
    assert described["yam"]["gripper"] == "native"
