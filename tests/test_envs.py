"""The synthetic world, the replay environment, and the perturbation wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from xevals import envs, perturbations
from xevals.datasets import synthetic_episodes
from xevals.envs import SPLITS, PointMass, ReplayEnv, check_reset_invariant


def test_reset_from_a_recorded_state_reproduces_it(env):
    # The invariant the whole reset-from-recorded-state protocol rests on.
    assert check_reset_invariant(env) == 0.0


def test_rendering_is_deterministic_given_the_state(env):
    first = env.render()
    second = env.render()
    assert np.array_equal(first, second)
    assert first.dtype == np.uint8


def test_the_scripted_policy_solves_the_task_so_the_gate_can_mean_something():
    solved = 0
    for seed in range(20):
        world = PointMass("reach")
        world.reset(seed=seed)
        info = {}
        for _ in range(80):
            _obs, _reward, done, info = world.step(world.optimal_action)
            if done:
                break
        solved += bool(info["success"])
    assert solved >= 18


def test_the_workspace_limit_is_reachable_so_safety_has_something_to_measure():
    world = PointMass("reach")
    world.reset(seed=0)
    limits = world.limits()
    breached = False
    for _ in range(80):
        _obs, _r, done, info = world.step(np.array([1.0, 1.0], dtype=np.float32))
        breached |= bool(limits.check(info))
        if done:
            break
    assert breached, "a world where nothing can go wrong measures nothing"


def test_each_split_changes_exactly_one_thing():
    worlds = {split: PointMass("reach", split=split) for split in SPLITS}
    assert worlds["ood/object"].object_name != worlds["in"].object_name
    assert worlds["ood/instruction"].instruction != worlds["in"].instruction
    assert worlds["ood/object"].instruction != worlds["in"].instruction
    # The layout split keeps the object and the wording, and moves the geometry.
    assert worlds["ood/layout"].object_name == worlds["in"].object_name
    assert worlds["ood/layout"].instruction == worlds["in"].instruction


def test_physics_can_be_set_and_rejects_a_parameter_it_does_not_have(env):
    env.set_physics(friction=0.5)
    assert env.friction == 0.5
    with pytest.raises(ValueError, match="unknown physics parameter"):
        env.set_physics(gravity=9.8)


def test_a_replay_environment_returns_the_recording_whatever_the_model_does():
    episode = synthetic_episodes("reach", episodes=1)[0]
    world = episode.to_env()
    first = world.reset()
    obs, _reward, _done, info = world.step(np.array([99.0, -99.0], dtype=np.float32))

    assert np.array_equal(first["state"], episode.obs[0]["state"])
    assert np.array_equal(obs["state"], episode.obs[1]["state"])
    assert info["offline"] is True
    assert "reference_action" in info


def test_a_replay_environment_needs_more_than_one_observation():
    with pytest.raises(ValueError, match="at least two"):
        ReplayEnv([{"state": np.zeros(2)}], np.zeros((0, 2)))


def test_the_perturbation_wrapper_leaves_the_true_state_alone(env):
    wrapped = envs.perturbed(env, perturbations.create("visual/gaussian_noise", severity=1.0))
    clean_obs = env.observe()
    dirty_obs = wrapped.reset(seed=0)

    assert not np.array_equal(dirty_obs["image"], clean_obs["image"])
    assert np.array_equal(wrapped.state(), env.state())


def test_the_wrapper_forwards_optional_capabilities(env):
    wrapped = envs.perturbed(env, perturbations.identity())
    # set_physics, optimal_action and limits must survive wrapping, or a
    # dynamics cell on a wrapped environment would report itself unmeasurable.
    assert wrapped.limits() is not None
    assert wrapped.optimal_action.shape == (2,)
    wrapped.set_physics(mass=2.0)
    assert env.mass == 2.0


def test_a_dynamics_perturbation_says_so_when_the_environment_cannot_take_it():
    from xevals.errors import CapabilityMissing

    class Rigid:
        action_dim = 2
        action_low = -np.ones(2)
        action_high = np.ones(2)

    with pytest.raises(CapabilityMissing, match="set_physics"):
        envs.perturbed(Rigid(), perturbations.create("dynamics/mass"))


def test_the_registry_lists_the_built_in_worlds():
    assert "synthetic/reach" in envs.available()
    assert envs.describe("synthetic/push")["family"] == "synthetic"
