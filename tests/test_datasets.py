"""Dataset readers, and the offline evaluation path they feed."""

from __future__ import annotations

import numpy as np
import pytest

import xevals
from xevals import datasets


def test_the_synthetic_reader_needs_no_download_and_no_extras():
    episodes = datasets.synthetic_episodes("reach", episodes=3)
    assert len(episodes) == 3
    assert all(e.success for e in episodes)
    assert all(len(e.obs) == len(e) + 1 for e in episodes), "one more observation than action"


def test_an_episode_becomes_an_environment_the_runner_can_drive():
    episode = datasets.synthetic_episodes("reach", episodes=1)[0]
    world = episode.to_env()
    assert world.offline is True
    assert world.action_dim == 2


def test_a_saved_trajectory_can_be_read_back_as_a_dataset(tmp_path, env, scripted):
    traj = xevals.rollout(scripted, env, seed=0, horizon=20)
    traj.save(tmp_path / "000")

    episodes = list(datasets.read_npz(tmp_path))
    assert len(episodes) == 1
    assert np.allclose(episodes[0].actions, traj.action_array)
    assert episodes[0].instruction == traj.instruction


def test_the_offline_path_measures_action_error_against_the_demonstrator():
    episodes = datasets.synthetic_episodes("reach", episodes=4)
    cycle = {"index": 0}

    def make_env(task=None, split="in"):
        episode = episodes[cycle["index"] % len(episodes)]
        cycle["index"] += 1
        return episode.to_env()

    class Lazy:
        def act(self, obs, *, instruction=None):
            return np.zeros(2, dtype=np.float32)

    result = xevals.evaluate(
        Lazy(), make_env, suite="core", episodes=2, seeds=(0,), out=None,
        verbose=False, baselines=False,
    )
    error = result.metric("accuracy/action_mse")
    assert error is not None and error.measured
    assert error.value > 0, "a zero policy does not match a moving demonstrator"


def test_the_readers_declare_what_they_need_installed():
    assert datasets.describe("lerobot")["requires"] == ["data"]
    assert datasets.describe("npz")["requires"] == []
    assert set(datasets.available()) == {"hdf5", "lerobot", "npz", "synthetic"}


def test_a_missing_format_library_names_the_extra(monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("h5py"):
            raise ImportError("no h5py")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(xevals.MissingExtra, match=r"xevals\[data\]"):
        list(datasets.read_hdf5(tmp_path / "x.hdf5"))


def test_an_offline_run_never_publishes_the_demonstrator_success_as_the_model_s():
    episodes = datasets.synthetic_episodes("reach", episodes=2)
    cycle = {"index": 0}

    def make_env(task=None, split="in"):
        episode = episodes[cycle["index"] % len(episodes)]
        cycle["index"] += 1
        return episode.to_env()

    class Lazy:
        def act(self, obs, *, instruction=None):
            return np.zeros(2, dtype=np.float32)

    result = xevals.evaluate(
        Lazy(), make_env, suite="core", episodes=2, seeds=(0,), out=None,
        verbose=False, baselines=False,
    )
    success = result.metric("accuracy/success_rate")
    # Every recorded episode succeeded; the model did nothing. Reporting 1.00
    # here would be the most flattering number this library could produce.
    assert success.value is None
    assert "unmeasurable offline" in success.reason
