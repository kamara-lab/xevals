"""Protocols, trajectories and safety limits."""

from __future__ import annotations

import numpy as np
import pytest

from xevals.types import (
    Env,
    HasConfidence,
    Planner,
    Policy,
    SafetyLimits,
    Trajectory,
    Violation,
    WorldModel,
)


def test_protocols_are_satisfied_structurally_without_inheritance():
    class Duck:
        def act(self, obs, *, instruction=None):
            return np.zeros(2, dtype=np.float32)

    assert isinstance(Duck(), Policy)
    assert not isinstance(Duck(), WorldModel)
    assert not isinstance(Duck(), Planner)


def test_the_synthetic_environment_satisfies_the_environment_protocol(env):
    assert isinstance(env, Env)


def test_a_model_without_confidence_is_not_mistaken_for_one_with_it(scripted):
    assert not isinstance(scripted, HasConfidence)


def test_a_trajectory_round_trips_through_npz_and_json(tmp_path):
    traj = Trajectory(
        obs=[{"state": np.arange(3, dtype=np.float32)} for _ in range(3)],
        actions=[np.ones(2, dtype=np.float32) for _ in range(2)],
        rewards=[0.5, 1.5],
        infos=[{"success": False}, {"success": True}],
        instruction="move to the red cube",
        seed=7,
        perturbation="visual/blur",
        split="ood/object",
        success=True,
        violations=[Violation("workspace", 0.02, 1)],
        latencies_ms=[1.0, 2.0],
    )
    traj.save(tmp_path / "episode")
    back = Trajectory.load(tmp_path / "episode")

    assert back.instruction == traj.instruction
    assert back.seed == 7
    assert back.perturbation == "visual/blur"
    assert back.split == "ood/object"
    assert back.success is True
    assert back.violations == traj.violations
    assert np.allclose(back.action_array, traj.action_array)
    assert np.allclose(back.obs[0]["state"], traj.obs[0]["state"])


def test_total_reward_is_the_undiscounted_return():
    assert Trajectory(rewards=[1.0, -0.5, 2.0]).total_reward == 2.5


def test_limits_report_the_margin_not_only_the_breach():
    limits = SafetyLimits(
        workspace_low=np.array([-1.0, -1.0]), workspace_high=np.array([1.0, 1.0])
    )
    assert limits.check({"position": np.array([0.0, 0.0])}) == []
    breach = limits.check({"position": np.array([1.3, 0.0])})
    assert len(breach) == 1
    assert breach[0].kind == "workspace"
    assert breach[0].margin == pytest.approx(0.3)


def test_a_limit_the_environment_does_not_declare_is_never_violated():
    assert SafetyLimits().check({"speed": 1e6, "force": 1e6}) == []
