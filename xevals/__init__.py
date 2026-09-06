"""xevals -- multi-dimensional evaluation of robot-learning models.

Most robot-learning evaluations report one number: task success rate. That
number hides everything that decides whether a model is usable. Does it survive
a camera shift, a paraphrased instruction, a distractor on the table? Does it
leave the workspace on the way to a success? Can text painted on a box redirect
it? What does it cost to run at 10 Hz? Each of those is measured, when it is
measured at all, by a one-off script per paper, so no two numbers compare.

xevals takes **any model** as input -- a VLA, a world model, an RL policy, an LLM
planner, a URL -- and scores it along **seven named dimensions**, writing one run
directory that can be diffed, re-scored and aggregated.

Layout
------
Flat, one module per area, because a small library with deep packages is harder
to read than a small library with plain modules:

======================  ==================================================
:mod:`xevals.types`      Protocols, ``Trajectory``, ``SafetyLimits``
:mod:`xevals.dimensions` the seven dimensions and their normalisation
:mod:`xevals.adapters`   wrapping torch / jax / LeRobot / HF / HTTP / xwm
:mod:`xevals.envs`       the built-in synthetic world, Gym, replay, wrappers
:mod:`xevals.robots`     the Menagerie arms and the scene they are evaluated in
:mod:`xevals.sim`        manipulation tasks on those arms, in MuJoCo or Newton
:mod:`xevals.datasets`   readers for LeRobot, HDF5 and NPZ corpora
:mod:`xevals.perturbations`  visual, sensor, action, language, dynamics, attack
:mod:`xevals.metrics`    pure functions over trajectories, one per measurement
:mod:`xevals.judges`     rule-based verdicts on text; an optional LLM judge
:mod:`xevals.suites`     which cells to run, on which splits
:mod:`xevals.runner`     rollouts, baselines, the replay gate, budgets
:mod:`xevals.bench`     several models under identical conditions
:mod:`xevals.results`    the run directory, tables, leaderboards, diffs
:mod:`xevals.media`      videos and GIFs with a HUD
:mod:`xevals.plots`      radar, severity curves, heatmaps, in the brand palette
:mod:`xevals.report`     one self-contained HTML page
:mod:`xevals.config`     TOML configs, ``extends``, dotted overrides
:mod:`xevals.cli`        ``xevals run|compare|report|list|doctor``
======================  ==================================================

Conventions
-----------
Arrays are NumPy at every boundary; observations are ``dict[str, Any]`` so a
model reads the fields it wants. The core depends on numpy and the standard
library alone -- torch, jax, simulators, video codecs and plotting are extras
that fail *when used*, never at import. Anything unmeasurable is reported as
``null`` with a reason, never as zero. British spelling throughout.

Quick start
-----------
>>> import xevals
>>> policy = xevals.wrap(lambda obs: obs["goal"] - obs["state"][:2])
>>> result = xevals.evaluate(policy, "synthetic/reach", suite="smoke",
...                         out=None, verbose=False)
>>> sorted(result.scores)
['accuracy', 'efficiency', 'robustness']
"""

from __future__ import annotations

from . import (
    adapters,
    bench,
    config,
    datasets,
    dimensions,
    envs,
    judges,
    media,
    metrics,
    perturbations,
    plots,
    registry,
    report,
    results,
    runner,
    seeding,
    suites,
    tradeoffs,
    types,
)
from .adapters import wrap
from .bench import Benchmark, benchmark
from .dimensions import Dimension, DimensionScore
from .errors import CapabilityMissing, GateFailed, MissingExtra, XevalsError
from .metrics import MetricValue
from .results import Result
from .runner import Budget, evaluate, rollout
from .seeding import set_seed
from .suites import Cell, SuiteSpec
from .types import Env, Planner, Policy, SafetyLimits, Scorer, Trajectory, Violation, WorldModel

__version__ = "0.1.0"

__all__ = [
    "Benchmark",
    "Budget",
    "CapabilityMissing",
    "Cell",
    "Dimension",
    "DimensionScore",
    "Env",
    "GateFailed",
    "MetricValue",
    "MissingExtra",
    "Planner",
    "Policy",
    "Result",
    "SafetyLimits",
    "Scorer",
    "SuiteSpec",
    "Trajectory",
    "Violation",
    "WorldModel",
    "XevalsError",
    "__version__",
    "adapters",
    "bench",
    "benchmark",
    "config",
    "datasets",
    "dimensions",
    "envs",
    "evaluate",
    "judges",
    "media",
    "metrics",
    "perturbations",
    "plots",
    "registry",
    "report",
    "results",
    "rollout",
    "runner",
    "seeding",
    "set_seed",
    "suites",
    "tradeoffs",
    "types",
    "wrap",
]
