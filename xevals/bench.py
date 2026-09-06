"""Evaluating several models in one run, against one environment and one suite.

A single evaluation is a fact about a single model. Almost every question anyone
actually asks -- is the new checkpoint better? does the bigger model buy
robustness or only accuracy? which of these three is safe enough to deploy? --
is a comparison, and comparisons made by running ``evaluate`` three times and
putting the numbers in a spreadsheet are weaker than they look.

Three things this module does that three separate runs do not:

**Every model sees the same episodes.** Seeds derive from
``(root_seed, seed, index)`` and never from the model, so model A's episode 7 and
model B's episode 7 start from the same state under the same perturbation. That
turns a comparison of two averages into a paired comparison, which resolves a
difference with roughly a tenth of the episodes.

**The baselines and the gate run once.** ``random``, ``noop`` and the replay gate
are properties of the *environment*, not of any model, so running them per model
wastes time and -- worse -- lets three rows of a leaderboard disagree about what
the floor was. Here they are measured once and quoted identically in every row.

**The comparison is written down.** One benchmark directory holds every model's
run directory plus the leaderboard, the overlaid radar and a combined report, so
the thing that gets shared is the comparison rather than three files and a claim
about them.

What it deliberately does not do is run models concurrently. Two models sharing a
GPU report each other's contention as their own latency, and the efficiency
dimension would become a measurement of the benchmark harness.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import adapters, suites
from .dimensions import DIMENSION_ORDER
from .results import Result, markdown_table, save_table
from .runner import Budget, _close, _env_factory, _env_fingerprint, environment_fingerprint
from .suites import SuiteSpec
from .types import Env

__all__ = ["Benchmark", "benchmark"]


@dataclass
class Benchmark:
    """Several models evaluated under identical conditions.

    Attributes:
        results: model name -> its :class:`~xevals.results.Result`, in the order
            the models were given. Insertion order is kept rather than sorted by
            score, because the caller's order is usually meaningful (baseline
            first, candidate last) and a table can sort itself.
        suite: the resolved suite every model ran.
        baselines: the reference policies, measured once for the environment.
        env: the environment fingerprint.
        run: seeds, episodes, budget, library versions.
        directory: where it was written, once it has been.
    """

    results: dict[str, Result] = field(default_factory=dict)
    suite: SuiteSpec | None = None
    baselines: dict[str, dict[str, Any]] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)
    directory: Path | None = None

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, name: str) -> Result:
        return self.results[name]

    @property
    def gate_failed(self) -> bool:
        """Whether the shared replay gate failed. If it did, no row means anything."""
        gate = self.baselines.get("replay")
        return bool(gate and gate.get("passed") is False)

    def gate_status(self) -> str:
        """``"passed"``, ``"failed"`` or ``"unmeasured"``, for the whole benchmark."""
        gate = self.baselines.get("replay")
        if not gate or gate.get("passed") is None:
            return "unmeasured"
        return "passed" if gate["passed"] else "failed"

    # -- tables -----------------------------------------------------------

    def rows(self, *, sort: bool = True) -> tuple[list[str], list[list[Any]]]:
        """The leaderboard: one row per model, one column per dimension.

        The ``mean`` column is a summary and not a ranking anyone should defend --
        a model excellent everywhere except security and one mediocre everywhere
        can tie -- which is exactly why the per-dimension columns sit beside it.
        """
        claimed = {d.value for d in (self.suite.dimensions if self.suite else ())}
        dimensions = [d for d in DIMENSION_ORDER if d.value in claimed]
        headers = ["model", *[d.value for d in dimensions], "mean"]
        rows: list[list[Any]] = []
        for name, result in self.results.items():
            scores = [result.scores.get(d.value) for d in dimensions]
            measured = [s for s in scores if s is not None]
            rows.append([name, *scores, float(np.mean(measured)) if measured else None])
        if sort:
            rows.sort(key=lambda r: (r[-1] is None, -(r[-1] or 0.0)))
        return headers, rows

    def cell_rows(self, metric: str = "accuracy/success_rate") -> tuple[list[str], list[list]]:
        """One row per cell, one column per model. Where the models differ, and by how much.

        The most useful table in the module, and the one a leaderboard cannot
        replace: two models with the same robustness score routinely fail under
        *different* perturbations, and only this view shows it.
        """
        names = list(self.results)
        headers = ["cell", "perturbation", "severity", *names]
        rows = []
        for cell in (self.suite.cells if self.suite else ()):
            values = []
            for name in names:
                value = self.results[name].cells.get(cell.name, {}).get(metric)
                values.append(None if value is None else value.value)
            if all(v is None for v in values):
                continue
            rows.append([cell.name, cell.perturbation, cell.severity, *values])
        return headers, rows

    def table(self, kind: str = "leaderboard", format: str = "md") -> str:
        """Render a table. ``kind`` is ``leaderboard`` or ``cells``."""
        headers, rows = self.rows() if kind == "leaderboard" else self.cell_rows()
        from .results import csv_table, latex_table

        text = {
            "md": lambda: markdown_table(headers, rows),
            "csv": lambda: csv_table(headers, rows),
            "tex": lambda: latex_table(headers, rows),
        }[format]()
        if self.gate_failed and format == "md":
            return (
                "> **Gate failed.** Replaying the demonstrator's own actions did not "
                "solve the task in this environment, so no row below measures a model.\n\n"
                + text
            )
        return text

    def summary(self) -> str:
        """A few lines for a terminal."""
        lines = [
            f"benchmark: {len(self)} models, suite {self.suite.name!r}, "
            f"gate {self.gate_status()}"
        ]
        headers, rows = self.rows()
        for row in rows:
            mean = "--" if row[-1] is None else f"{row[-1]:.3f}"
            lines.append(f"  {str(row[0]):20s} {mean}")
        return "\n".join(lines)

    def best(self, dimension: str | None = None) -> str | None:
        """The highest-scoring model on one dimension, or on the mean.

        ``None`` when nothing was measurable. Provided because it is what a CI
        job wants, and because writing the ``max`` by hand is where a tie or a
        ``None`` quietly becomes a wrong answer.
        """
        scored = [
            (name, result.scores.get(dimension) if dimension else _mean(result))
            for name, result in self.results.items()
        ]
        scored = [(n, s) for n, s in scored if s is not None]
        return max(scored, key=lambda item: item[1])[0] if scored else None

    def disagreements(self, threshold: float = 0.2) -> list[tuple[str, str, str, float, float]]:
        """Conditions that separate the models *beyond* how they already differ.

        Returns ``(cell, best, worst, gap, excess)``, sorted by ``excess``.

        The raw gap is the wrong thing to rank by, and obviously so once you look
        at one: a model that is simply worse is worse in every cell, so a plain
        gap ranking returns the whole suite in arbitrary order and buries the one
        condition that matters. ``excess`` subtracts the gap the same two models
        already show on the clean cell, and so answers the question a benchmark is
        actually run to answer -- *given* that these models perform as they do
        nominally, which condition pulls them apart further?

        A uniformly weaker model produces an excess near zero everywhere and
        correctly drops out of this list. It is already visible in the
        leaderboard, which is where a uniform difference belongs.
        """
        headers, rows = self.cell_rows()
        names = headers[3:]
        clean_name = self.suite.clean_cell.name if self.suite else "clean"
        clean = {
            n: v
            for row in rows
            if row[0] == clean_name
            for n, v in zip(names, row[3:], strict=True)
        }
        out = []
        for row in rows:
            if row[0] == clean_name:
                continue
            values = [(n, v) for n, v in zip(names, row[3:], strict=True) if v is not None]
            if len(values) < 2:
                continue
            best = max(values, key=lambda item: item[1])
            worst = min(values, key=lambda item: item[1])
            gap = best[1] - worst[1]
            baseline = clean.get(best[0]), clean.get(worst[0])
            excess = (
                gap - (baseline[0] - baseline[1])
                if None not in baseline
                else gap
            )
            if excess >= threshold:
                out.append((row[0], best[0], worst[0], gap, excess))
        out.sort(key=lambda item: -item[4])
        return out

    # -- output -----------------------------------------------------------

    def save(self, out: str | Path = "runs", *, name: str = "benchmark", **kwargs: Any) -> Path:
        """Write one benchmark directory: every model's run, plus the comparison.

        ::

            <out>/<name>/
              <model>/...          one full run directory per model
              leaderboard.{md,tex,csv,json}
              cells.{md,tex,csv,json}
              radar.png            every model on one radar
              index.html           the comparison, linking to each model's report
        """
        directory = Path(out) / name
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        for model_name, result in self.results.items():
            result.save(directory, name=_slug(model_name), **kwargs)
        save_table(directory / "leaderboard", *self.rows(), caption="xevals benchmark")
        save_table(directory / "cells", *self.cell_rows())
        self._write_tradeoffs(directory)
        self._write_radar(directory)
        self._write_index(directory)
        return directory

    def tradeoffs(self, name: str | None = None, *, scope: str = "models") -> list[Any]:
        """Test the registered trade-offs across these models.

        Model scope is where a frontier would live, and where most benchmarks
        will honestly report ``undetermined``: a handful of models is not enough
        points for a correlation, and the analysis says so rather than drawing a
        curve through four dots. Pass ``scope="cells"`` for the within-run
        evidence, which has an order of magnitude more points.
        """
        from . import tradeoffs as module

        if name is not None:
            return [module.analyse(self, name, scope=scope)]
        return module.analyse_all(self, scope=scope)

    def _write_tradeoffs(self, directory: Path) -> None:
        """One table and one scatter per trade-off, at model scope."""
        from .results import save_table

        for analysis in self.tradeoffs():
            stem = directory / "tradeoffs" / analysis.tradeoff.name
            headers, rows = analysis.rows()
            if rows:
                save_table(stem, headers, rows, caption=analysis.tradeoff.title)
            headers, rows = analysis.group_rows()
            if rows:
                save_table(stem.with_name(f"{analysis.tradeoff.name}-groups"), headers, rows)
            try:
                from .plots import tradeoff as draw

                draw(analysis, stem.with_suffix(".png"))
            except Exception:  # noqa: BLE001 - a missing extra, or nothing to draw
                pass

    def _write_radar(self, directory: Path) -> None:
        try:
            from .plots import leaderboard_radar

            leaderboard_radar(list(self.results.values()), directory / "radar.png")
        except Exception as exc:  # noqa: BLE001 - a missing extra must not lose a run
            print(f"xevals: radar skipped ({type(exc).__name__}: {exc})", flush=True)

    def _write_index(self, directory: Path) -> None:
        from .report import write_benchmark

        write_benchmark(self, directory)

    def radar(self, path: str | Path | None = None) -> Path:
        """Every model on one radar. Capped at five, where the palette runs out."""
        from .plots import leaderboard_radar

        target = Path(path) if path else (self.directory or Path(".")) / "radar.png"
        return leaderboard_radar(list(self.results.values()), target)

    def report(self, path: str | Path | None = None) -> Path:
        """(Re)build the comparison pages. Returns the leaderboard page."""
        from .report import write_benchmark

        target = Path(path) if path else (self.directory or Path(".")) / "index.html"
        return write_benchmark(self, target.parent)

    @classmethod
    def load(cls, path: str | Path) -> Benchmark:
        """Read a benchmark directory back, model by model."""
        directory = Path(path)
        runs = sorted(p for p in directory.iterdir() if p.is_dir() and (p / "run.json").exists())
        if not runs:
            raise FileNotFoundError(f"no run directories under {directory}")
        results = {p.name.rsplit("-", 1)[0]: Result.load(p) for p in runs}
        first = next(iter(results.values()))
        return cls(
            results=results,
            suite=first.suite,
            baselines=first.baselines,
            env=first.env,
            run=first.run,
            directory=directory,
        )


def _mean(result: Result) -> float | None:
    measured = [s for s in result.scores.values() if s is not None]
    return float(np.mean(measured)) if measured else None


def _slug(name: str) -> str:
    """A filesystem-safe directory name for a model."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name).strip("-") or "model"


def benchmark(
    models: Mapping[str, Any] | Sequence[tuple[str, Any]],
    env: Env | Callable[..., Env] | str,
    *,
    suite: str | SuiteSpec | dict[str, Any] = "core",
    episodes: int | None = None,
    seeds: Iterable[int] = (0,),
    horizon: int = 200,
    record: int = 2,
    out: str | Path | None = "runs",
    name: str = "benchmark",
    root_seed: int = 0,
    budget: Budget | float | None = None,
    baselines: bool = True,
    control_hz: float = 10.0,
    save_trajectories: bool = False,
    verbose: bool = True,
) -> Benchmark:
    """Evaluate several models under identical conditions.

    Args:
        models: name -> model, or a sequence of ``(name, model)`` pairs. Anything
            :func:`xevals.wrap` accepts. Order is kept in the output.
        env: as :func:`xevals.evaluate` -- an environment, a registered name, or a
            factory taking ``(task, split)``.
        suite: the suite every model runs. One suite, by construction: models
            evaluated under different conditions are not comparable, and making
            that impossible is most of the point of this function.
        budget: seconds, or a :class:`~xevals.runner.Budget`, **shared** across the
            models. A benchmark that spends its whole allowance on the first
            model has not produced a comparison.
        out: where the benchmark directory goes. ``None`` computes without writing.

    Returns:
        A :class:`Benchmark`.

    Raises:
        ValueError: if two models are given the same name, or none are given.

    Examples:
        >>> import xevals                                          # doctest: +SKIP
        >>> bench = xevals.benchmark(
        ...     {"baseline": old, "candidate": new},
        ...     "synthetic/reach", suite="robustness", episodes=20, seeds=range(3),
        ... )
        >>> print(bench.table())                                  # doctest: +SKIP
        >>> bench.disagreements()                                 # doctest: +SKIP
    """
    from .runner import evaluate

    pairs = list(models.items()) if isinstance(models, Mapping) else list(models)
    if not pairs:
        raise ValueError("a benchmark needs at least one model")
    names = [n for n, _ in pairs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate model name(s) {duplicates}; every row needs its own name")

    spec = suites.create(suite).resolved()
    episodes = int(episodes if episodes is not None else spec.episodes)
    seeds = list(seeds)
    budget = Budget(seconds=float(budget)) if isinstance(budget, (int, float)) else budget
    if budget is not None:
        budget.start()

    if verbose:
        print(
            f"xevals benchmark: {len(pairs)} models, suite {spec.name!r}, "
            f"{len(spec.cells)} cells, {episodes} episodes x {len(seeds)} seeds",
            flush=True,
        )

    # The baselines and the gate belong to the environment, so they are measured
    # once and quoted in every row. Running them per model would waste the time
    # and let three rows of one leaderboard disagree about what the floor was.
    shared_baselines: dict[str, dict[str, Any]] = {}
    make_env = _env_factory(env)
    probe = make_env("", "in")
    env_info = _env_fingerprint(probe)
    action_dim = int(getattr(probe, "action_dim", 2))
    _close(probe)
    if baselines:
        from .runner import _run_baselines

        shared_baselines = _run_baselines(
            spec,
            make_env,
            action_dim,
            root_seed=root_seed,
            seeds=seeds,
            episodes=min(episodes, 10),
            horizon=horizon,
            verbose=verbose,
        )

    results: dict[str, Result] = {}
    for index, (model_name, model) in enumerate(pairs, start=1):
        if verbose:
            print(f"\n[{index}/{len(pairs)}] {model_name}", flush=True)
        started = time.perf_counter()
        result = evaluate(
            model,
            env,
            suite=spec,
            episodes=episodes,
            seeds=seeds,
            horizon=horizon,
            record=record,
            out=None,
            root_seed=root_seed,
            budget=budget,
            baselines=False,
            control_hz=control_hz,
            verbose=verbose,
        )
        # Every row quotes the same floor and the same gate.
        result.baselines = dict(shared_baselines)
        results[model_name] = result
        if verbose:
            print(f"  {model_name}: {time.perf_counter() - started:.1f}s", flush=True)

    bench = Benchmark(
        results=results,
        suite=spec,
        baselines=shared_baselines,
        env=env_info,
        run={
            "seeds": seeds,
            "episodes": episodes,
            "horizon": horizon,
            "root_seed": root_seed,
            "control_hz": control_hz,
            "models": [{"name": n, **adapters.fingerprint(adapters.wrap(m))} for n, m in pairs],
            "partial": bool(budget is not None and budget.exhausted),
            "budget": budget.describe() if budget is not None else None,
            "environment": environment_fingerprint(),
        },
    )
    if out is not None:
        bench.save(out, name=name, trajectories=save_trajectories)
        if verbose:
            print(f"\nxevals: wrote {bench.directory}", flush=True)
    if verbose:
        print()
        print(bench.table(), flush=True)
    return bench
