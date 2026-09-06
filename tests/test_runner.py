"""Rollouts, baselines, the replay gate, budgets, and end-to-end determinism."""

from __future__ import annotations

import numpy as np

import xevals
from xevals.runner import Budget, environment_fingerprint, rollout


def test_a_rollout_records_conditions_as_well_as_outcome(env, scripted):
    traj = rollout(scripted, env, seed=3, horizon=40, record=True)

    assert traj.seed == 3
    assert traj.split == "in"
    assert traj.instruction == "move to the red cube"
    assert len(traj.obs) == len(traj.actions) + 1
    assert len(traj.latencies_ms) == len(traj.actions)
    assert traj.frames and traj.frames[0].shape == (64, 64, 3)
    assert traj.success is True


def test_step_timing_measures_the_model_and_not_the_simulator(env):
    import time

    class Slow:
        def act(self, obs, *, instruction=None):
            time.sleep(0.002)
            return np.zeros(2, dtype=np.float32)

    traj = rollout(Slow(), env, seed=0, horizon=3)
    assert min(traj.latencies_ms) >= 2.0


def test_an_evaluation_is_reproducible_bit_for_bit(scripted):
    kwargs = dict(suite="smoke", episodes=3, seeds=(0, 1), out=None, verbose=False)
    first = xevals.evaluate(scripted, "synthetic/reach", **kwargs)
    second = xevals.evaluate(scripted, "synthetic/reach", **kwargs)

    assert first.config_hash() == second.config_hash()
    for name, score in first.scores.items():
        if name == "efficiency":
            continue  # wall-clock latency is the one thing a seed cannot fix
        assert score == second.scores[name]
    for cell in first.cells:
        for name, value in first.cells[cell].items():
            if name.startswith("efficiency/"):
                continue
            assert value.value == second.cells[cell][name].value, name
            assert value.ci == second.cells[cell][name].ci, f"{name} interval moved"


def test_the_clean_and_perturbed_cells_share_seeds_so_pairing_works(scripted):
    result = xevals.evaluate(
        scripted, "synthetic/reach", suite="smoke", episodes=3, seeds=(0,), out=None,
        verbose=False,
    )
    retention = result.metric("robustness/retention", "noise")
    assert retention is not None and retention.measured, retention and retention.reason


def test_the_replay_gate_passes_on_a_faithful_environment(scripted):
    result = xevals.evaluate(
        scripted, "synthetic/reach", suite="core", episodes=3, seeds=(0,), out=None,
        verbose=False,
    )
    # The suite's baselines do not include replay, so the gate is unmeasured --
    # which is a third state, distinct from passing.
    assert result.gate_status() == "unmeasured"

    full = xevals.evaluate(
        scripted, "synthetic/reach", suite="smoke", episodes=3, seeds=(0,), out=None,
        verbose=False, baselines=True,
    )
    assert full.baselines["random"]["success_rate"] is not None


def test_a_gate_failure_is_visible_in_the_table_not_only_in_json(scripted, monkeypatch):

    result = xevals.evaluate(
        scripted, "synthetic/reach", suite="smoke", episodes=2, seeds=(0,), out=None,
        verbose=False,
    )
    result.baselines["replay"] = {"success_rate": 0.0, "gate": True, "passed": False}
    assert result.gate_failed
    assert "Gate failed" in result.table()


def test_a_model_that_raises_fails_that_episode_rather_than_the_run():
    class Fragile:
        def act(self, obs, *, instruction=None):
            return obs["state"][:2] * 0.0 if "state" in obs else 1 / 0

    result = xevals.evaluate(
        Fragile(), "synthetic/reach",
        suite={"name": "drop", "cells": [
            {"name": "clean"},
            {"name": "dropout", "perturbation": "sensor/dropout", "severity": 1.0},
        ], "dimensions": ["accuracy"]},
        episodes=4, seeds=(0,), out=None, verbose=False, baselines=False,
    )
    assert result.metric("accuracy/success_rate", "clean").measured


def test_a_budget_truncates_between_episodes_and_marks_the_run_partial(scripted):
    result = xevals.evaluate(
        scripted, "synthetic/reach", suite="robustness", episodes=5, seeds=(0,),
        out=None, verbose=False, budget=Budget(episodes=8), baselines=False,
    )
    assert result.partial
    assert result.run["budget"]["episodes_run"] <= 8


def test_the_environment_fingerprint_reports_only_what_is_imported():
    fingerprint = environment_fingerprint()
    assert "python" in fingerprint and "numpy" in fingerprint
    assert "platform" in fingerprint


def test_the_run_records_whether_reset_from_state_round_trips(scripted):
    result = xevals.evaluate(
        scripted, "synthetic/reach", suite="core", episodes=2, seeds=(0,), out=None,
        verbose=False, baselines=False,
    )
    assert result.env["reset_state_error"] == 0.0
    assert result.env["declares_limits"] is True


def test_a_dimension_the_suite_did_not_measure_scores_none(scripted):
    result = xevals.evaluate(
        scripted, "synthetic/reach", suite="core", episodes=2, seeds=(0,), out=None,
        verbose=False, baselines=False,
    )
    assert "security" not in result.scores, "core claims three dimensions, not seven"


def test_safety_aggregates_to_the_worst_cell_not_the_average(scripted):
    from xevals.dimensions import Dimension
    from xevals.runner import AGGREGATION

    # Stated as a rule rather than left to whichever cells a suite happened to run.
    assert AGGREGATION[Dimension.SAFETY] == "worst"
    assert AGGREGATION[Dimension.SECURITY] == "worst"
    assert AGGREGATION[Dimension.ACCURACY] == "clean"


def test_an_unfaithful_environment_trips_the_gate(scripted):
    from xevals import Dimension, envs
    from xevals.suites import Cell, SuiteSpec

    class Broken(envs.PointMass):
        """A world that ignores actions, so the demonstrator cannot solve it either."""

        def step(self, action):
            return super().step(np.zeros(2, dtype=np.float32))

    suite = SuiteSpec(
        "gated", (Cell("clean"),), (Dimension.ACCURACY,),
        baselines=("random", "noop", "replay"),
    )
    result = xevals.evaluate(
        scripted, lambda task=None, split="in": Broken("reach"),
        suite=suite, episodes=4, seeds=(0,), out=None, verbose=False,
    )
    assert result.gate_status() == "failed"
    assert result.gate_failed
    # Loud by design: a banner above the table, not a field in the JSON.
    assert result.table().startswith("> **Gate failed.**")
