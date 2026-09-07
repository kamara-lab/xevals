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
Implementation modules are grouped by responsibility: ``core`` (contracts and
configuration), ``evaluation`` (suites, rollouts and scoring), ``integrations``
(model adapters and datasets), ``environments`` (worlds and simulators), and
``reporting`` (results, analysis and output). The CLI stays at the package root.
Existing imports such as ``xevals.runner`` remain supported as module aliases;
new internal code imports from the implementation packages directly.

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

import sys as _sys

from .core import config, dimensions, errors, registry, seeding, types
from .core.dimensions import Dimension, DimensionScore
from .core.errors import CapabilityMissing, GateFailed, MissingExtra, XevalsError
from .core.seeding import set_seed
from .core.types import (
    Env,
    Planner,
    Policy,
    SafetyLimits,
    Scorer,
    Trajectory,
    Violation,
    WorldModel,
)
from .environments import envs
from .evaluation import bench, judges, metrics, perturbations, runner, suites
from .evaluation.bench import Benchmark, benchmark
from .evaluation.metrics import MetricValue
from .evaluation.runner import Budget, evaluate, rollout
from .evaluation.suites import Cell, SuiteSpec
from .integrations import adapters, datasets
from .integrations.adapters import wrap
from .reporting import charts, media, plots, report, results, tradeoffs
from .reporting.results import Result

# Both import paths share one module, including registries and monkeypatches.
for _module in (
    types,
    errors,
    registry,
    seeding,
    dimensions,
    config,
    runner,
    bench,
    suites,
    metrics,
    perturbations,
    judges,
    adapters,
    datasets,
    envs,
    results,
    report,
    charts,
    plots,
    media,
    tradeoffs,
):
    _sys.modules[f"{__name__}.{_module.__name__.rsplit('.', 1)[-1]}"] = _module
del _module, _sys

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
