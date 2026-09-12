"""CPU execution must preserve the experiment, not merely produce more steps."""
from __future__ import annotations

import numpy as np
import pytest

import xevals
from xevals import adapters
from xevals.evaluation.runner import _episode_seeds


@pytest.fixture
def sim(monkeypatch):
    pytest.importorskip("mujoco")
    from xevals.environments import robots, sim

    if not robots.assets_available("panda"):
        pytest.skip("Panda assets are not cached")

    def forbidden(*args, **kwargs):
        raise AssertionError("state-only evaluation must never initialise a renderer")

    monkeypatch.setattr(sim, "_renderer", forbidden)
    return sim


def factory(sim, cls=None):
    cls = cls or sim.ManipulationEnv
    return lambda task=None, split="in": cls(
        "panda", task or "reach", split=split, observation_mode="state",
    )


def policy():
    def act(obs):
        return np.array([0.2, -0.1, 0.1, 1.0], dtype=np.float32)

    return adapters.CallablePolicy(
        act, action_dim=4,
        batch_fn=lambda obs, instructions=None: np.stack([act(o) for o in obs]),
    )


def run(sim, *, execution="serial", **kwargs):
    options = dict(suite={"name": "cpu", "cells": [
        {"name": "clean"},
        {"name": "noise", "perturbation": "action/noise", "severity": 0.5},
        {"name": "mass", "perturbation": "dynamics/mass", "severity": 0.5},
    ], "dimensions": ["accuracy", "robustness", "safety", "efficiency"]},
        episodes=3, seeds=(0, 1), horizon=8, record=0, baselines=False, out=None,
        verbose=False, execution=execution)
    if execution == "cpu_batch":
        options.update(batch_size=4, workers=2)
    options.update(kwargs)
    return xevals.evaluate(policy(), factory(sim), **options)


def test_state_mode_is_explicit_and_models_are_isolated(sim):
    a, b = factory(sim)(), factory(sim)()
    assert a.model is not b.model
    a.reset(seed=1)
    first_goal = a.model.site_pos.copy()
    b.reset(seed=4)
    np.testing.assert_array_equal(a.model.site_pos, first_goal)
    assert "image" not in a.observe()
    assert "state" in a.observe()
    image = sim.ManipulationEnv("panda", "reach")
    assert image.observation_mode == "image"
    with pytest.raises(ValueError, match="observation_mode"):
        sim.ManipulationEnv(observation_mode="invalid")


def test_native_substeps_match_scalar_physics_calls(sim):
    a, b = factory(sim)(), factory(sim)()
    a.reset(seed=1)
    b.reset(seed=1)
    mj = b._mj

    class ScalarSteps:
        def __getattr__(self, name):
            return getattr(mj, name)

        def mj_step(self, model, data, *, nstep):
            for _ in range(nstep):
                mj.mj_step(model, data)

    b._mj = ScalarSteps()
    for _ in range(10):
        action = np.array([0.2, 0, -0.1, 1], np.float32)
        ao, ar, ad, ai = a.step(action)
        bo, br, bd, bi = b.step(action)
        np.testing.assert_array_equal(a.state(), b.state())
        assert ar == br and ad == bd
        assert ai["force"] == bi["force"]


@pytest.mark.parametrize("batch_size,workers", [(1, 1), (4, 2), (8, 4)])
def test_serial_batch_trajectory_and_metric_parity(sim, batch_size, workers):
    serial = run(sim)
    batch = run(sim, execution="cpu_batch", batch_size=batch_size, workers=workers)
    assert batch.env["observation_mode"] == "state"
    assert batch.run["execution"]["mode"] == "cpu_batch"
    for cell, trajectories in serial.trajectories.items():
        bt = batch.trajectories[cell]
        assert [t.seed for t in trajectories] == [t.seed for t in bt]
        for a, b in zip(trajectories, bt, strict=True):
            assert a.success == b.success
            assert a.violations == b.violations
            np.testing.assert_array_equal(a.actions, b.actions)
            np.testing.assert_array_equal(a.rewards, b.rewards)
            for ao, bo in zip(a.obs, b.obs, strict=True):
                for k in ao:
                    np.testing.assert_array_equal(ao[k], bo[k])
            assert not b.latencies_ms
            assert b.extra["environment_timings"]["physics_s"] > 0
        for name, value in serial.cells[cell].items():
            if not name.startswith("efficiency/"):
                assert value.value == batch.cells[cell][name].value
                assert value.ci == batch.cells[cell][name].ci
    latency = batch.metric("efficiency/latency_p50", "clean")
    assert latency.value is None and "batched inference" in latency.reason
    assert serial.config_hash() != batch.config_hash()
    saved_hash = batch.config_hash()
    batch.run["execution"]["evaluation_s"] += 100
    assert batch.config_hash() == saved_hash


def test_no_visual_robustness_is_reported_without_images(sim):
    result = run(sim, execution="cpu_batch", suite={
        "name": "visual", "cells": [{"name": "clean"},
            {"name": "blur", "perturbation": "visual/blur", "severity": 1.0}],
        "dimensions": ["accuracy", "robustness"],
    })
    assert not result.trajectories.get("blur")
    value = result.metric("robustness/retention", "blur")
    assert value.value is None and "image observations" in value.reason


def test_budget_admits_only_remaining_episodes(sim):
    result = run(sim, execution="cpu_batch", budget=xevals.Budget(episodes=5))
    assert result.partial
    assert result.run["budget"]["episodes_run"] == 5
    assert sum(len(ts) for ts in result.trajectories.values()) == 5
    assert [t.seed for t in result.trajectories["clean"]] == _episode_seeds(0, (0, 1), 3)[:5]


def test_unequal_termination_and_owned_snapshots(sim):
    class Short(sim.ManipulationEnv):
        def reset(self, *, seed=None, state=None):
            self.horizon = 1 + (int(seed or 0) % 3)
            return super().reset(seed=seed, state=state)

    def evaluate(mode):
        return xevals.evaluate(policy(), factory(sim, Short), suite="core", episodes=5,
            horizon=5, record=0, baselines=False, out=None, verbose=False, execution=mode,
            **({"batch_size": 4, "workers": 2} if mode == "cpu_batch" else {}))

    a, b = evaluate("serial"), evaluate("cpu_batch")
    ats, bts = a.trajectories["clean"], b.trajectories["clean"]
    assert len({len(t.actions) for t in bts}) > 1
    assert [len(t.actions) for t in ats] == [len(t.actions) for t in bts]
    for t in bts:
        assert len(t.obs) == len(t.actions) + 1
        assert not np.shares_memory(t.obs[0]["state"], t.obs[-1]["state"])


def test_worker_failure_does_not_fail_other_episodes(sim):
    bad_seed = _episode_seeds(0, (0,), 4)[0]

    class Failing(sim.ManipulationEnv):
        def reset(self, *, seed=None, state=None):
            self.fail = seed == bad_seed
            return super().reset(seed=seed, state=state)

        def step(self, action):
            if self.fail:
                raise RuntimeError("episode failure")
            return super().step(action)

    result = xevals.evaluate(policy(), factory(sim, Failing), suite="core", episodes=4,
        horizon=2, record=0, baselines=False, out=None, verbose=False,
        execution="cpu_batch", batch_size=4)
    ts = result.trajectories["clean"]
    assert "episode failure" in ts[0].extra["error"]
    assert all("error" not in t.extra for t in ts[1:])
    assert all(len(t.actions) == 2 for t in ts[1:])


def test_bad_batch_input_is_isolated_by_bisection():
    from xevals.evaluation.batching import _infer

    def batch(obs, instructions=None):
        if any(o["bad"] for o in obs):
            raise ValueError("bad row")
        return np.ones((len(obs), 2))

    p = adapters.CallablePolicy(lambda obs: [1, 1], batch_fn=batch)
    calls = []
    actions, errors = _infer(p, [0, 1, 2], [{"bad": False}, {"bad": True}, {"bad": False}],
                             [None] * 3, calls)
    assert set(actions) == {0, 2} and set(errors) == {1}
    assert any(c["failed"] for c in calls)


def test_preflight_rejects_unsupported_requests(sim):
    base = dict(out=None, verbose=False, episodes=1, horizon=1, record=0, baselines=False)
    with pytest.raises(ValueError, match="stateless"):
        xevals.evaluate(lambda obs: [0, 0], factory(sim), execution="cpu_batch", **base)
    with pytest.raises(ValueError, match="built-in MuJoCo"):
        xevals.evaluate(policy(), "synthetic/reach", execution="cpu_batch", **base)
    with pytest.raises(TypeError, match="Policy.act"):
        xevals.evaluate(wrap_planner(), factory(sim), **base)
    single = factory(sim)()
    with pytest.raises(ValueError, match="reused"):
        xevals.evaluate(policy(), lambda: single, execution="cpu_batch", batch_size=2,
                        **{**base, "episodes": 2})


def wrap_planner():
    return xevals.wrap(lambda instruction: "a plan", kind="planner")


def test_state_only_visual_skip_also_applies_to_serial(sim):
    result = run(sim, suite={"name": "image-required", "cells": [
        {"name": "clean"}, {"name": "text", "perturbation": "injection/scene_text"},
    ], "dimensions": ["accuracy", "security"]})
    assert not result.trajectories.get("text")
    assert all(v.value is None and "image observations" in v.reason
               for v in result.cells["text"].values())


def test_time_budget_finishes_admitted_group(sim):
    budget = xevals.Budget()

    def expire(*args):
        # Expire after the first transition, without sleeps or timing-dependent assertions.
        budget.seconds = 0

    result = run(sim, execution="cpu_batch", batch_size=4, budget=budget, on_step=expire)
    assert result.partial
    assert result.run["budget"]["episodes_run"] == 4
    assert len(result.trajectories["clean"]) == 4
    assert all(len(t.actions) == 8 for t in result.trajectories["clean"])


def test_nonfinite_batch_action_only_fails_its_row(sim):
    target = _episode_seeds(0, (0,), 3)[1]

    class Marked(sim.ManipulationEnv):
        def reset(self, *, seed=None, state=None):
            obs = super().reset(seed=seed, state=state)
            obs["bad"] = seed == target
            return obs

    def batch(obs, instructions=None):
        result = np.zeros((len(obs), 4), np.float32)
        for i, o in enumerate(obs):
            if o.get("bad"):
                result[i, 0] = np.nan
        return result

    p = adapters.CallablePolicy(lambda obs: np.zeros(4), batch_fn=batch)
    result = xevals.evaluate(p, factory(sim, Marked), suite="core", episodes=3,
        horizon=2, record=0, baselines=False, out=None, verbose=False,
        execution="cpu_batch", batch_size=3)
    ts = result.trajectories["clean"]
    assert ["error" in t.extra for t in ts] == [False, True, False]
    assert "NaN" in ts[1].extra["error"]


def test_torch_batch_matches_serial_closed_loop(sim):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    p = xevals.wrap(torch.nn.Sequential(torch.nn.Linear(24, 16), torch.nn.Tanh(),
                                      torch.nn.Linear(16, 4), torch.nn.Tanh()),
                   device="cpu", feature_keys=("state",), batch_mode="stateless")
    options = dict(suite="core", episodes=4, horizon=5, record=0,
                   baselines=False, out=None, verbose=False)
    a = xevals.evaluate(p, factory(sim), **options)
    b = xevals.evaluate(p, factory(sim), execution="cpu_batch", batch_size=4, **options)
    for ta, tb in zip(a.trajectories["clean"], b.trajectories["clean"], strict=True):
        assert ta.success == tb.success
        np.testing.assert_allclose(ta.actions, tb.actions, atol=1e-6)
        np.testing.assert_allclose(ta.rewards, tb.rewards, atol=1e-6)


def test_batch_metadata_survives_trajectory_roundtrip(sim, tmp_path):
    result = run(sim, execution="cpu_batch", episodes=2, seeds=(0,))
    first = result.trajectories["clean"][0]
    first.save(tmp_path / "episode")
    restored = xevals.Trajectory.load(tmp_path / "episode")
    assert restored.extra["batch_calls"] == first.extra["batch_calls"]
    assert restored.extra["execution"] == "cpu_batch"
    assert restored.latencies_ms == []


def test_failed_episode_has_serial_safety_semantics(sim):
    class Failing(sim.ManipulationEnv):
        def step(self, action):
            if self._t == 1:
                raise RuntimeError("failure after one recorded step")
            return super().step(action)

    options = dict(suite="core", episodes=2, horizon=3, record=0,
                   baselines=False, out=None, verbose=False)
    serial = xevals.evaluate(policy(), factory(sim, Failing), **options)
    batch = xevals.evaluate(policy(), factory(sim, Failing), execution="cpu_batch",
                            batch_size=2, **options)
    for a, b in zip(serial.trajectories["clean"], batch.trajectories["clean"], strict=True):
        assert a.success is b.success is False
        assert a.actions == b.actions == []
        assert a.obs == b.obs == []
        assert a.extra["limits_declared"] is b.extra["limits_declared"] is False
    for name, value in serial.cells["clean"].items():
        if not name.startswith("efficiency/"):
            assert value.value == batch.cells["clean"][name].value


@pytest.mark.parametrize("real_renderer", [False, True])
@pytest.mark.parametrize("torch_policy", [False, True])
def test_image_policy_parity_and_coordinator_rendering(monkeypatch, real_renderer, torch_policy):
    import threading

    from xevals.environments import robots, sim

    if not robots.assets_available("panda"):
        pytest.skip("Panda assets are not cached")
    coordinator = threading.get_ident()
    original = sim.ManipulationEnv.render

    def render(self):
        assert threading.get_ident() == coordinator
        if real_renderer:
            frame = original(self)
            assert frame is not None, "real rendering must be available for this integration test"
            return frame
        rng = np.random.default_rng(int(abs(self.data.qpos.sum()) * 1e6))
        return rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)

    monkeypatch.setattr(sim.ManipulationEnv, "render", render)
    def act(obs):
        return np.r_[obs["image"].mean(axis=(0, 1)) / 255 - 0.5, 1].astype(np.float32)

    model = adapters.CallablePolicy(act, action_dim=4, batch_fn=lambda obs, instructions=None:
                                   np.stack([act(o) for o in obs]))
    if torch_policy:
        torch = pytest.importorskip("torch")

        class ImagePolicy(torch.nn.Module):
            def forward(self, obs):
                pixels = obs["image"]
                assert pixels.dtype == torch.uint8
                rgb = pixels.float().mean(dim=(1, 2)) / 255 - 0.5
                return torch.cat((rgb, torch.ones((len(rgb), 1))), dim=1)

        model = adapters.TorchPolicy(ImagePolicy(), device="cpu", dict_input=True,
                                     feature_keys=("image",), action_dim=4,
                                     batch_mode="stateless")

    def make_env(task=None, split="in"):
        return sim.ManipulationEnv("panda", "reach", split=split, image_size=32)

    options = dict(suite={"name": "vision", "cells": [
        {"name": "clean"}, {"name": "blur", "perturbation": "visual/blur"},
        {"name": "latency", "perturbation": "sensor/latency"}],
        "dimensions": ["accuracy", "robustness", "safety"]},
        episodes=3, horizon=3, record=1, baselines=False, out=None, verbose=False)
    serial = xevals.evaluate(model, make_env, **options)
    batch = xevals.evaluate(model, make_env, execution="cpu_batch", batch_size=2,
                            workers=2, **options)
    for cell, trajectories in serial.trajectories.items():
        assert len(trajectories) == 3
        for a, b in zip(trajectories, batch.trajectories[cell], strict=True):
            assert "error" not in a.extra and "error" not in b.extra
            np.testing.assert_allclose(a.actions, b.actions, atol=1e-6)
            np.testing.assert_allclose(a.rewards, b.rewards, atol=1e-6)
            assert a.violations == b.violations
            for ao, bo in zip(a.obs, b.obs, strict=True):
                np.testing.assert_array_equal(ao["image"], bo["image"])
            if a.frames is not None:
                np.testing.assert_array_equal(a.frames, b.frames)
                np.testing.assert_array_equal(b.frames, [o["image"] for o in b.obs])
            else:
                assert b.frames is None
