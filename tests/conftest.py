"""Fixtures shared by the suite: a policy, an environment, and hand-built episodes.

Kept tiny on purpose. A large conftest makes every test depend on machinery that
is itself untested, and the point of these fixtures is to be obviously correct
by inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

from xevals.envs import PointMass
from xevals.types import Trajectory, Violation


@pytest.fixture
def env() -> PointMass:
    """The synthetic reach world, reset at a fixed seed."""
    world = PointMass("reach")
    world.reset(seed=0)
    return world


@pytest.fixture
def scripted():
    """A policy that solves the synthetic task from the state vector."""

    class Scripted:
        def act(self, obs, *, instruction=None):
            state, goal = obs.get("state"), obs.get("goal")
            if state is None or goal is None:
                return np.zeros(2, dtype=np.float32)
            delta = goal - state[:2]
            norm = float(np.linalg.norm(delta))
            return (delta / norm if norm > 1e-6 else delta).astype(np.float32)

    return Scripted()


@pytest.fixture
def episodes():
    """Ten trajectories with known outcomes: even seeds succeed, odd ones fail."""

    def build(success_of, *, actions=None, **kwargs):
        return [
            Trajectory(
                actions=list(
                    actions if actions is not None else np.zeros((4, 2), dtype=np.float32)
                ),
                rewards=[1.0] * 4,
                infos=[{}] * 4,
                seed=seed,
                success=success_of(seed),
                **kwargs,
            )
            for seed in range(10)
        ]

    return build


@pytest.fixture
def violation() -> Violation:
    """One workspace excursion, two centimetres past the limit, at step three."""
    return Violation("workspace", 0.02, 3)
