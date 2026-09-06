"""The Newton backend, checked against the MuJoCo one it has to agree with."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("newton")
pytest.importorskip("warp")

from xevals import envs, robots  # noqa: E402
from xevals.envs import check_reset_invariant  # noqa: E402
from xevals.sim import NEWTON_TASKS, ManipulationEnv  # noqa: E402

pytestmark = pytest.mark.skipif(
    not robots.assets_available(),
    reason="the Menagerie models are not in the cache; set XEVALS_MENAGERIE or fetch them",
)


@pytest.fixture(scope="module")
def pair() -> tuple[ManipulationEnv, ManipulationEnv]:
    """The same scene under both backends. Building either compiles kernels."""
    return (
        ManipulationEnv("panda", "reach", backend="newton", image_size=32),
        ManipulationEnv("panda", "reach", backend="mujoco", image_size=32),
    )


def test_the_backends_agree_on_where_the_arm_ends_up(pair):
    newton_env, mujoco_env = pair
    newton_env.reset(seed=0)
    mujoco_env.reset(seed=0)
    zero = np.zeros(4, dtype=np.float32)
    for _ in range(20):
        newton_env.step(zero)
        mujoco_env.step(zero)
    difference = np.max(
        np.abs(
            newton_env.data.qpos[newton_env._arm_qadr]
            - mujoco_env.data.qpos[mujoco_env._arm_qadr]
        )
    )
    # Two different solvers holding one pose. They will not match exactly; a
    # disagreement of more than a couple of degrees means the servo gains did
    # not survive the import, which is the failure this is here to catch.
    assert difference < 0.05


def test_the_newton_state_stays_finite_and_round_trips(pair):
    newton_env, _ = pair
    newton_env.reset(seed=1)
    for _ in range(10):
        newton_env.step(np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32))
    assert np.all(np.isfinite(newton_env.data.qpos))
    assert check_reset_invariant(newton_env) == 0.0


def test_the_demonstrator_solves_the_task_under_newton_too(pair):
    newton_env, _ = pair
    newton_env.reset(seed=2)
    info: dict = {}
    done = False
    while not done:
        _obs, _reward, done, info = newton_env.step(newton_env.optimal_action)
    assert info["success"]


def test_the_object_is_pinned_because_newton_cannot_be_trusted_to_hold_it(pair):
    newton_env, _ = pair
    newton_env.reset(seed=3)
    start = newton_env._object_position().copy()
    for _ in range(10):
        newton_env.step(np.zeros(4, dtype=np.float32))
    assert np.allclose(newton_env._object_position(), start)
    assert NEWTON_TASKS == ("reach",), "the tasks that move the object stay on MuJoCo"


def test_the_backend_is_named_in_the_registry_and_the_fingerprint():
    assert envs.describe("newton/panda-reach")["backend"] == "newton"
    assert envs.describe("newton/panda-reach")["requires"] == ["newton"]
    env = envs.create("newton/yam-reach")
    assert env.backend == "newton"


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="backend must be"):
        ManipulationEnv("panda", "reach", backend="bullet")
