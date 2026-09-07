"""Suites: which cells to run, on which splits, scored by which metrics.

A *cell* is one measurement condition -- a task, a perturbation at a severity, a
split, a set of metrics, a number of episodes. A *suite* is a list of cells plus
the baselines to run beside them. Everything about what an evaluation covers
lives in these two dataclasses, which is why the runner has no opinions about
coverage and the suite has none about execution.

Three rules built into the built-in suites, each learned from a way evaluations
mislead:

**Every suite starts with a clean cell.** Not for its own sake, but because
every paired metric -- retention, injection compliance, paraphrase agreement --
divides by it. A robustness suite without a clean baseline reports ratios
against nothing.

**A perturbation family is run as a ladder, not at one severity.** One severity
gives one number and no shape, and the shape is where models differ: a policy
that is flat to 0.6 and then falls off is a different proposition from one that
decays from the start, and both can share a mean.

**Baselines run on the same seeds as the model.** ``random`` and ``noop`` bound
the task from below, and ``replay`` gates it: if replaying a demonstrator's own
actions does not succeed, the environment is not faithful and no number from it
means anything. Running baselines on different seeds would leave the comparison
to luck.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from xevals.core.dimensions import Dimension
from xevals.core.registry import Registry
from xevals.evaluation.metrics import METRICS, applies_to, for_dimension
from xevals.evaluation.perturbations import SEVERITIES

__all__ = [
    "SUITES",
    "Cell",
    "SuiteSpec",
    "available",
    "create",
    "describe",
    "from_dict",
]

SUITES: Registry[SuiteSpec] = Registry("suites")

#: The splits a generalisation suite asks about. ``in`` is the training
#: distribution; the rest change exactly one thing each, so a drop is
#: attributable.
SPLITS: tuple[str, ...] = ("in", "ood/object", "ood/layout", "ood/instruction")


@dataclass(frozen=True)
class Cell:
    """One measurement condition.

    Attributes:
        name: how the cell appears in tables, run directories and figures.
        perturbation: registered perturbation name, or ``"none"``.
        severity: where on the ladder.
        split: which distribution the episodes are drawn from.
        metrics: which metrics score it. Empty means "every metric of the cell's
            dimensions", resolved at build time so ``run.json`` records the list
            that was actually used rather than a wildcard.
        episodes: episodes per seed.
        pair_with: the cell whose episodes are the reference for paired metrics.
            Defaults to the clean cell, which is what makes retention a ratio
            against the *same model* rather than against an absolute standard.
        task: overrides the suite's task, for a suite mixing several.
    """

    name: str
    perturbation: str = "none"
    severity: float = 0.0
    split: str = "in"
    metrics: tuple[str, ...] = ()
    episodes: int | None = None
    pair_with: str | None = None
    task: str | None = None
    perturbation_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        """Whether this is the unperturbed, in-distribution reference cell."""
        return self.perturbation == "none" and self.split == "in"

    def describe(self) -> dict[str, Any]:
        """The cell as JSON, for ``run.json``."""
        return {
            "name": self.name,
            "perturbation": self.perturbation,
            "severity": self.severity,
            "split": self.split,
            "metrics": list(self.metrics),
            "episodes": self.episodes,
            "pair_with": self.pair_with,
            "task": self.task,
        }


@dataclass(frozen=True)
class SuiteSpec:
    """What an evaluation covers.

    Attributes:
        name: the suite's registered name.
        cells: the conditions, in the order they run.
        dimensions: which dimensions this suite claims to measure. A suite that
            runs no security cell must not claim the security dimension, or its
            radar shows a confident zero where there is no measurement.
        baselines: which reference policies to run on the same seeds.
        episodes: default episodes per seed, overridable per cell.
        description: one line, shown by ``xevals suites list``.
    """

    name: str
    cells: tuple[Cell, ...]
    dimensions: tuple[Dimension, ...]
    baselines: tuple[str, ...] = ("random", "noop")
    episodes: int = 20
    description: str = ""

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError(f"suite {self.name!r} has no cells")
        names = [c.name for c in self.cells]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"suite {self.name!r} has duplicate cells {sorted(duplicates)}")
        if not any(c.is_clean for c in self.cells):
            raise ValueError(
                f"suite {self.name!r} has no clean cell; every paired metric divides by it"
            )

    @property
    def clean_cell(self) -> Cell:
        """The reference cell every paired metric is measured against."""
        return next(c for c in self.cells if c.is_clean)

    def resolved(self) -> SuiteSpec:
        """The suite with wildcard metric lists expanded and pairing filled in.

        Called once before a run so that ``run.json`` records the exact metric
        list used. A suite file that says "all robustness metrics" would
        otherwise mean something different next release, and the saved run would
        not say which it meant.
        """
        clean = self.clean_cell.name
        cells = []
        for cell in self.cells:
            # An explicit list is the author's choice and is kept verbatim. The
            # default -- every metric of every dimension the suite claims -- is
            # filtered to the ones that mean something under this cell's
            # condition; see :func:`xevals.metrics.applies_to`.
            metrics = cell.metrics or tuple(
                m
                for d in self.dimensions
                for m in for_dimension(d)
                if applies_to(m, cell.perturbation, cell.split)
            )
            pair = cell.pair_with or (None if cell.is_clean else clean)
            cells.append(replace(cell, metrics=tuple(dict.fromkeys(metrics)), pair_with=pair))
        return replace(self, cells=tuple(cells))

    def describe(self) -> dict[str, Any]:
        """The suite as JSON."""
        return {
            "name": self.name,
            "description": self.description,
            "dimensions": [d.value for d in self.dimensions],
            "baselines": list(self.baselines),
            "episodes": self.episodes,
            "cells": [c.describe() for c in self.cells],
        }


def _ladder(
    perturbation: str,
    *,
    severities: Sequence[float] = SEVERITIES,
    metrics: tuple[str, ...] = (),
    split: str = "in",
) -> list[Cell]:
    """One cell per severity, named so the ladder is legible in a table."""
    return [
        Cell(
            name=f"{perturbation}@{s:g}",
            perturbation=perturbation,
            severity=float(s),
            split=split,
            metrics=metrics,
        )
        for s in severities
    ]


_CLEAN = Cell("clean", "none", 0.0, "in")

#: The visual families worth a full ladder in a default suite. Chosen because
#: each corresponds to something that demonstrably happens between a lab and a
#: deployment: the camera moves, the light changes, something gets in the way.
_ROBUSTNESS_FAMILIES = (
    "visual/camera_shift",
    "visual/brightness",
    "visual/gaussian_noise",
    "visual/occlusion",
    "sensor/latency",
    "action/noise",
)


def _core() -> SuiteSpec:
    """Accuracy, safety and efficiency on the clean condition. The cheap default."""
    return SuiteSpec(
        name="core",
        cells=(_CLEAN,),
        dimensions=(Dimension.ACCURACY, Dimension.SAFETY, Dimension.EFFICIENCY),
        description="clean-condition accuracy, safety and efficiency -- the cheap default",
    )


def _robustness() -> SuiteSpec:
    """A severity ladder over the perturbation families that occur in practice."""
    cells = [_CLEAN]
    for family in _ROBUSTNESS_FAMILIES:
        cells += _ladder(family)
    return SuiteSpec(
        name="robustness",
        cells=tuple(cells),
        dimensions=(Dimension.ACCURACY, Dimension.ROBUSTNESS),
        description="severity ladders over camera, photometry, noise, occlusion, latency",
    )


def _safety() -> SuiteSpec:
    """Limit violations under clean and disturbed conditions.

    Includes perturbed cells on purpose: a policy that stays inside the workspace
    when everything is nominal and leaves it the moment the camera moves is not
    a safe policy, and a clean-only safety suite would call it one.
    """
    cells = [_CLEAN]
    cells += _ladder("action/noise", severities=(0.4, 0.8))
    cells += _ladder("visual/camera_shift", severities=(0.4, 0.8))
    return SuiteSpec(
        name="safety",
        cells=tuple(cells),
        dimensions=(Dimension.SAFETY, Dimension.ACCURACY),
        description="limit violations, clean and under disturbance",
    )


def _security() -> SuiteSpec:
    """Adversarial patches, pixel attacks and instruction injection."""
    cells = [_CLEAN]
    for family in ("adversarial/patch", "adversarial/pixel"):
        cells += _ladder(family, severities=(0.5, 1.0))
    for family in ("injection/scene_text", "injection/instruction"):
        cells += _ladder(family, severities=(1.0,))
    return SuiteSpec(
        name="security",
        cells=tuple(cells),
        dimensions=(Dimension.SECURITY, Dimension.ACCURACY),
        description="adversarial patches, bounded pixel noise, instruction injection",
    )


def _generalization() -> SuiteSpec:
    """The same task on unseen objects, layouts and instruction templates."""
    cells = [_CLEAN]
    cells += [
        Cell(f"split/{split.split('/')[-1]}", "none", 0.0, split)
        for split in SPLITS
        if split != "in"
    ]
    return SuiteSpec(
        name="generalization",
        cells=tuple(cells),
        dimensions=(Dimension.GENERALIZATION, Dimension.ACCURACY),
        description="unseen objects, unseen layouts, unseen instruction templates",
    )


def _consistency() -> SuiteSpec:
    """Seed spread, paraphrase invariance, and determinism at a fixed seed."""
    cells = [
        _CLEAN,
        Cell("repeat", "none", 0.0, "in"),
        Cell("paraphrase", "instruction/paraphrase", 1.0, "in"),
        Cell("typo", "instruction/typo", 0.6, "in"),
    ]
    return SuiteSpec(
        name="consistency",
        cells=tuple(cells),
        dimensions=(Dimension.CONSISTENCY, Dimension.ACCURACY),
        description="seed spread, paraphrase invariance, determinism at a fixed seed",
    )


def _full() -> SuiteSpec:
    """Every dimension. What the README's radar is made of.

    Assembled from the specialised suites rather than written out, so a cell
    added to the robustness suite reaches the full suite automatically and the
    two cannot drift apart.
    """
    cells: list[Cell] = [_CLEAN]
    seen = {"clean"}
    for build in (_robustness, _safety, _security, _generalization, _consistency):
        for cell in build().cells:
            if cell.name not in seen:
                cells.append(cell)
                seen.add(cell.name)
    return SuiteSpec(
        name="full",
        cells=tuple(cells),
        dimensions=tuple(Dimension),
        baselines=("random", "noop", "replay"),
        description="all seven dimensions -- the complete evaluation",
    )


def _smoke() -> SuiteSpec:
    """Two cells, three episodes. For CI and for checking a wiring change."""
    return SuiteSpec(
        name="smoke",
        cells=(_CLEAN, Cell("noise", "visual/gaussian_noise", 0.6)),
        dimensions=(Dimension.ACCURACY, Dimension.ROBUSTNESS, Dimension.EFFICIENCY),
        episodes=3,
        description="two cells, three episodes -- a wiring check, not an evaluation",
    )


for _build in (_core, _robustness, _safety, _security, _generalization, _consistency, _full,
               _smoke):
    SUITES.register(
        _build().name,
        _build,
        summary=_build().description,
        dimensions=[d.value for d in _build().dimensions],
        cells=len(_build().cells),
    )


def from_dict(data: dict[str, Any]) -> SuiteSpec:
    """Build a suite from a mapping -- what a TOML ``[suite]`` section becomes.

    Raises:
        ValueError: naming the unknown key or metric. A silently-ignored typo in
            a suite file produces an evaluation that skips a whole dimension and
            reports it as a confident blank.
    """
    known = {"name", "cells", "dimensions", "baselines", "episodes", "description"}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"suite has no field(s) {unknown}; valid keys are {sorted(known)}")
    cells = []
    for raw in data.get("cells", []):
        cell_keys = {f for f in Cell.__dataclass_fields__}
        bad = sorted(set(raw) - cell_keys)
        if bad:
            raise ValueError(f"cell {raw.get('name')!r} has no field(s) {bad}")
        for name in raw.get("metrics", ()):
            if name not in METRICS:
                raise ValueError(f"unknown metric {name!r} in cell {raw.get('name')!r}")
        cells.append(Cell(**{**raw, "metrics": tuple(raw.get("metrics", ()))}))
    return SuiteSpec(
        name=data.get("name", "custom"),
        cells=tuple(cells),
        dimensions=tuple(
            Dimension(d) for d in (data.get("dimensions") or [d.value for d in Dimension])
        ),
        baselines=tuple(data.get("baselines", ("random", "noop"))),
        episodes=int(data.get("episodes", 20)),
        description=data.get("description", ""),
    )


def available() -> list[str]:
    """Registered suite names."""
    return SUITES.available()


def describe(name: str | None = None) -> dict[str, Any]:
    """What a suite covers, without building the run."""
    return SUITES.describe(name)  # type: ignore[return-value]


def create(name: str | SuiteSpec | dict[str, Any]) -> SuiteSpec:
    """Resolve a suite from a name, a mapping, or an already-built spec."""
    if isinstance(name, SuiteSpec):
        return name
    if isinstance(name, dict):
        return from_dict(name)
    return SUITES.create(name)
