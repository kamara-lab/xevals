"""Running a suite against a model, and the baselines that make it readable.

The runner is deliberately the least clever module in the library. It builds a
cell's environment, runs episodes, times each ``act``, records what happened, and
hands trajectories to :mod:`xevals.metrics`. Everything interpretive lives
elsewhere, so the measurement path is short enough to read in one sitting -- and
so the clean and perturbed cells go through *exactly* the same code, which is the
precondition for their ratio meaning anything.

Baselines
---------
Three, and they answer different questions:

``random``  Uniform actions. The floor. A model that does not clear it has not
            been shown to do anything, and on a short-horizon task the floor is
            often surprisingly high.
``noop``    Zero actions. Separates "the task rewards motion" from "the task
            rewards the right motion" -- on some benchmarks doing nothing scores
            well, and that is a fact about the benchmark.
``replay``  The demonstrator's own actions, and a **gate** rather than a
            baseline. If replaying the data does not succeed in this
            environment, the environment does not match the data, and every
            number measured in it is uninterpretable. A failed gate is recorded
            in ``run.json`` and printed at the top of the table rather than
            being left for a reader to infer.

Budgets
-------
An evaluation that runs out of time should return partial results marked
``partial``, not nothing. The budget is checked between episodes, so a cell is
either complete or absent -- never half-measured, which would bias its mean
toward whichever episodes happen to be fast.
"""

from __future__ import annotations

import platform
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import adapters, envs, metrics, perturbations, suites
from .dimensions import Dimension, DimensionScore, score_dimension
from .errors import CapabilityMissing
from .seeding import seeds_for, set_seed
from .suites import Cell, SuiteSpec
from .types import Env, Obs, SafetyLimits, Trajectory

__all__ = [
    "AGGREGATION",
    "Budget",
    "CellResult",
    "environment_fingerprint",
    "evaluate",
    "noop_policy",
    "random_policy",
    "replay_policy",
    "rollout",
]


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def random_policy(action_dim: int, *, seed: int = 0) -> Any:
    """Uniform actions in ``[-1, 1]``. The floor any model must clear."""

    class RandomPolicy:
        def __init__(self) -> None:
            self.rng = np.random.default_rng(seed)

        def reset(self) -> None:
            self.rng = np.random.default_rng(seed)

        def act(self, obs: Obs, *, instruction: str | None = None) -> np.ndarray:
            return self.rng.uniform(-1.0, 1.0, action_dim).astype(np.float32)

        def describe(self) -> dict[str, Any]:
            return {"adapter": "baseline", "policy": "random", "seed": seed}

    return RandomPolicy()


def noop_policy(action_dim: int) -> Any:
    """Zero actions. Separates rewarding motion from rewarding the right motion."""

    class NoopPolicy:
        def act(self, obs: Obs, *, instruction: str | None = None) -> np.ndarray:
            return np.zeros(action_dim, dtype=np.float32)

        def describe(self) -> dict[str, Any]:
            return {"adapter": "baseline", "policy": "noop"}

    return NoopPolicy()


def replay_policy(env: Any) -> Any | None:
    """The demonstrator's own actions, when the environment can supply them.

    Returns ``None`` when it cannot -- a synthetic environment with a scripted
    solution supplies ``optimal_action``, a replayed dataset supplies the
    recorded action, and a bare Gym environment supplies neither. ``None`` means
    the gate is *unmeasured*, which is recorded as such rather than passed.
    """
    source = (
        "optimal_action"
        if hasattr(env, "optimal_action")
        else "dataset"
        if getattr(env, "offline", False)
        else None
    )
    if source is None:
        return None

    class ReplayBaseline:
        """Reads the demonstrator from whichever environment is *currently* live.

        The binding matters: the runner builds a fresh environment per episode,
        and a policy closing over the probe environment would replay a state the
        episode never visited -- which looks exactly like a failed gate, and was.
        """

        def __init__(self) -> None:
            self.env = env

        def bind(self, live: Any) -> None:
            self.env = live

        def act(self, obs: Obs, *, instruction: str | None = None) -> np.ndarray:
            if source == "optimal_action":
                return np.asarray(self.env.optimal_action, dtype=np.float32)
            index = min(self.env._t, len(self.env._actions) - 1)
            return np.asarray(self.env._actions[index], dtype=np.float32)

        def describe(self) -> dict[str, Any]:
            return {"adapter": "baseline", "policy": "replay", "source": source}

    return ReplayBaseline()


# --------------------------------------------------------------------------
# Budgets
# --------------------------------------------------------------------------


@dataclass
class Budget:
    """A wall-clock or episode cap that truncates between episodes, never inside one.

    Args:
        seconds: total wall-clock allowance for the run.
        episodes: total episodes across all cells.
    """

    seconds: float | None = None
    episodes: int | None = None
    _start: float = field(default_factory=time.perf_counter, repr=False)
    _spent: int = field(default=0, repr=False)
    _started: bool = field(default=False, repr=False)

    def start(self) -> None:
        """Start the clock, once.

        Idempotent, and that is the whole point: :func:`xevals.benchmark` hands
        one budget to every model's :func:`evaluate`, and a ``start`` that reset
        the clock would give each model the full allowance -- turning a shared
        cap into a per-model one, silently, and only on the runs long enough for
        anyone to have cared about the cap.
        """
        if self._started:
            return
        self._start = time.perf_counter()
        self._spent = 0
        self._started = True

    def reset(self) -> None:
        """Restart the clock deliberately, for a caller reusing one budget object."""
        self._started = False
        self.start()

    def spend(self, episodes: int = 1) -> None:
        """Record episodes run."""
        self._spent += episodes

    @property
    def exhausted(self) -> bool:
        """Whether the run should stop after the current episode."""
        if self.seconds is not None and time.perf_counter() - self._start >= self.seconds:
            return True
        return self.episodes is not None and self._spent >= self.episodes

    def describe(self) -> dict[str, Any]:
        """The budget and what was spent, for ``run.json``."""
        return {
            "seconds": self.seconds,
            "episodes": self.episodes,
            "elapsed_s": round(time.perf_counter() - self._start, 3),
            "episodes_run": self._spent,
        }


# --------------------------------------------------------------------------
# One episode
# --------------------------------------------------------------------------


def rollout(
    model: Any,
    env: Env,
    *,
    seed: int = 0,
    horizon: int = 200,
    record: bool = False,
    split: str = "in",
    perturbation: str | None = None,
    instruction: str | None = None,
    on_step: Callable[[int, Obs, np.ndarray, dict[str, Any]], None] | None = None,
) -> Trajectory:
    """Run one episode and record everything a metric might later want.

    Timing is measured around the model's ``act`` alone, not around the whole
    step: including the simulator would make a fast model on a slow simulator
    look slow, and the efficiency dimension is a claim about the model.
    """
    reset = getattr(model, "reset", None)
    if callable(reset):
        reset()
    obs = env.reset(seed=seed)
    limits = envs.limits_of(env)
    traj = Trajectory(
        instruction=instruction or obs.get("instruction"),
        seed=seed,
        perturbation=perturbation,
        split=split,
        frames=[] if record else None,
        extra={"limits_declared": limits is not None},
    )
    confidence_fn = getattr(model, "confidence", None)
    success_fn = getattr(env, "success", None)
    succeeded = False

    for step in range(horizon):
        traj.obs.append(obs)
        if record:
            frame = env.render()
            if frame is not None:
                traj.frames.append(np.asarray(frame, dtype=np.uint8))  # type: ignore[union-attr]
        start = time.perf_counter()
        action = model.act(obs, instruction=traj.instruction)
        traj.latencies_ms.append((time.perf_counter() - start) * 1e3)
        if callable(confidence_fn):
            try:
                traj.confidences.append(float(confidence_fn(obs)))
            except CapabilityMissing:
                pass
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, done, info = env.step(action)
        traj.actions.append(action)
        traj.rewards.append(float(reward))
        traj.infos.append(dict(info))
        if limits is not None:
            traj.violations.extend(
                type(v)(v.kind, v.margin, step) for v in limits.check(info)
            )
        succeeded = bool(success_fn(info)) if callable(success_fn) else bool(info.get("success"))
        if on_step is not None:
            on_step(step, obs, action, info)
        if done:
            break

    traj.obs.append(obs)
    if getattr(env, "offline", False):
        # A replayed episode does not respond to the model's actions, so whether
        # *this model* would have succeeded is unknowable. The recording's own
        # outcome is kept, clearly labelled, and the success rate reports null
        # with that reason -- because an offline run that quietly published the
        # demonstrator's success as the model's would be the single most
        # flattering number this library could produce.
        traj.success = None
        traj.extra["offline"] = True
        traj.extra["recorded_success"] = succeeded
    else:
        traj.success = succeeded
    return traj


def _episode_seeds(root: int, seeds: Sequence[int], episodes: int) -> list[int]:
    """Every episode seed, identical in every cell of the run.

    Derived from ``(root, seed, index)`` and deliberately *not* from the cell
    name. Every paired metric in the library -- retention, injection compliance,
    paraphrase agreement, trigger delta -- compares a perturbed episode with the
    clean episode that started from the same state. Seeding cells independently
    would leave that comparison to pooled averages, which need roughly ten times
    the episodes to resolve the same difference, and would silently report every
    paired metric as unmeasurable because no two cells share a seed.

    Adding a cell to a suite still does not renumber anything, because the seeds
    never depended on which cells exist.
    """
    out: list[int] = []
    for seed in seeds:
        out.extend(seeds_for(root, episodes, seed))
    return out


# --------------------------------------------------------------------------
# One cell
# --------------------------------------------------------------------------


@dataclass
class CellResult:
    """Everything one measurement condition produced."""

    cell: Cell
    trajectories: list[Trajectory] = field(default_factory=list)
    values: dict[str, metrics.MetricValue] = field(default_factory=dict)
    skipped: str = ""
    #: Errors raised by the model during this cell's episodes. A non-empty list
    #: with a success rate beside it means the rate counts the crashes as
    #: failures, which is what they are.
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float | None:
        """Convenience: the cell's success rate, or ``None`` if unmeasured."""
        value = self.values.get("accuracy/success_rate")
        return None if value is None else value.value


def _build_env(
    make_env: Callable[[str, str], Env],
    cell: Cell,
    *,
    seed: int,
) -> tuple[Env, str]:
    """The environment for one cell, wrapped in its perturbation.

    Returns the environment and a skip reason. A cell whose perturbation needs a
    capability the environment lacks is skipped *with the reason*, never silently
    dropped: "we could not measure dynamics robustness on this simulator" and
    "this model is robust to dynamics" must not look the same in a table.
    """
    env = make_env(cell.task or "", cell.split)
    if cell.perturbation in ("", "none"):
        return env, ""
    perturbation = perturbations.create(
        cell.perturbation, severity=cell.severity, **cell.perturbation_kwargs
    )
    try:
        return envs.perturbed(env, perturbation, seed=seed), ""
    except CapabilityMissing as exc:
        return env, str(exc)


def run_cell(
    model: Any,
    make_env: Callable[[str, str], Env],
    cell: Cell,
    *,
    root_seed: int = 0,
    seeds: Sequence[int] = (0,),
    episodes: int = 20,
    horizon: int = 200,
    record: int = 0,
    budget: Budget | None = None,
    on_step: Callable[..., None] | None = None,
) -> CellResult:
    """Run every episode of one cell and score it.

    ``record`` is the number of episodes to keep frames for, not a boolean:
    keeping frames for all of them is how a run directory reaches ten gigabytes,
    and the first few are what a report shows.
    """
    result = CellResult(cell)
    episode_seeds = _episode_seeds(root_seed, seeds, episodes)
    for index, seed in enumerate(episode_seeds):
        if budget is not None and budget.exhausted:
            result.skipped = "budget exhausted"
            break
        env, reason = _build_env(make_env, cell, seed=seed)
        if reason:
            result.skipped = reason
            _close(env)
            break
        bind = getattr(model, "bind", None)
        if callable(bind):
            bind(env)
        try:
            traj = rollout(
                model,
                env,
                seed=seed,
                horizon=horizon,
                record=index < record,
                split=cell.split,
                perturbation=None if cell.perturbation == "none" else cell.perturbation,
                on_step=on_step,
            )
        except Exception as exc:  # noqa: BLE001 - a model raising is a result
            # A model that crashes under a perturbation has failed that
            # condition, and that is the finding. Ending the whole run would
            # lose every cell already measured and tell nobody why -- the
            # classic case being a sensor-dropout cell handing a policy an
            # observation without the field it indexes unconditionally.
            traj = Trajectory(
                seed=seed,
                split=cell.split,
                perturbation=None if cell.perturbation == "none" else cell.perturbation,
                success=False,
                extra={"error": f"{type(exc).__name__}: {exc}", "limits_declared": False},
            )
            result.errors.append(traj.extra["error"])
        finally:
            _close(env)
        result.trajectories.append(traj)
        if budget is not None:
            budget.spend()
    return result


def _close(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


# --------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------


def environment_fingerprint() -> dict[str, Any]:
    """What the numbers were produced on.

    Versions of the libraries that are *actually imported*, not of everything
    installed: a result depends on the torch that ran, and importing torch to
    ask its version would change what a numpy-only run costs.
    """
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    for name in ("numpy", "torch", "jax", "gymnasium", "matplotlib", "imageio", "PIL", "xwm"):
        module = sys.modules.get(name)
        if module is not None:
            info[name] = getattr(module, "__version__", "unknown")
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch, "cuda", None) is not None:  # pragma: no cover
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    return info


# --------------------------------------------------------------------------
# The entry point
# --------------------------------------------------------------------------


def evaluate(
    model: Any,
    env: Env | Callable[..., Env] | str,
    *,
    suite: str | SuiteSpec | dict[str, Any] = "core",
    episodes: int | None = None,
    seeds: Iterable[int] = (0,),
    horizon: int = 200,
    record: int = 2,
    out: str | Path | None = "runs",
    name: str | None = None,
    root_seed: int = 0,
    budget: Budget | float | None = None,
    baselines: bool = True,
    control_hz: float = 10.0,
    on_step: Callable[..., None] | None = None,
    save_trajectories: bool = False,
    verbose: bool = True,
) -> Any:
    """Evaluate a model along every dimension the suite covers.

    Args:
        model: anything :func:`xevals.wrap` accepts, or something already
            satisfying a protocol in :mod:`xevals.types`.
        env: an environment, a registered environment name, or a factory taking
            ``(task, split)`` -- the factory form is what a generalisation suite
            needs, since each split is a different environment.
        suite: a registered suite name, a :class:`~xevals.suites.SuiteSpec`, or a
            mapping in the shape a TOML ``[suite]`` section produces.
        episodes: episodes per seed; defaults to the suite's own figure.
        seeds: the run's seeds. Three or more make the consistency dimension
            measurable; fewer and it reports null with that reason.
        record: how many episodes per cell to keep frames for.
        out: where the run directory goes. ``None`` computes without writing.
        budget: seconds, or a :class:`Budget`. Truncates between episodes and
            marks the run ``partial``.
        baselines: run the suite's reference policies on the same seeds.
        on_step: called as ``(step, obs, action, info)``; for Rerun or a progress
            bar. ``None``-tolerant, and never required.

    Returns:
        A :class:`xevals.results.Result`.

    Examples:
        >>> import xevals
        >>> result = xevals.evaluate(          # doctest: +SKIP
        ...     xevals.wrap(my_model), "synthetic/reach", suite="full", out=None
        ... )
    """
    from .results import build_result

    spec = suites.create(suite).resolved()
    episodes = int(episodes if episodes is not None else spec.episodes)
    seeds = list(seeds)
    budget = Budget(seconds=float(budget)) if isinstance(budget, (int, float)) else budget
    if budget is not None:
        budget.start()
    set_seed(root_seed)

    model = adapters.wrap(model)
    make_env = _env_factory(env)
    probe = make_env("", "in")
    action_dim = int(getattr(probe, "action_dim", 2))
    env_info = _env_fingerprint(probe)
    _close(probe)

    if verbose:
        print(
            f"xevals: suite {spec.name!r}, {len(spec.cells)} cells, "
            f"{episodes} episodes x {len(seeds)} seeds",
            flush=True,
        )

    cells: dict[str, CellResult] = {}
    for cell in spec.cells:
        started = time.perf_counter()
        cells[cell.name] = run_cell(
            model,
            make_env,
            cell,
            root_seed=root_seed,
            seeds=seeds,
            episodes=episodes,
            horizon=horizon,
            record=record,
            budget=budget,
            on_step=on_step,
        )
        if verbose:
            _print_cell(cells[cell.name], time.perf_counter() - started)

    baseline_results = (
        _run_baselines(
            spec,
            make_env,
            action_dim,
            root_seed=root_seed,
            seeds=seeds,
            episodes=min(episodes, 10),
            horizon=horizon,
            verbose=verbose,
        )
        if baselines
        else {}
    )

    result = build_result(
        spec=spec,
        cells=cells,
        baselines=baseline_results,
        model_info=adapters.fingerprint(model),
        env_info=env_info,
        run_info={
            "seeds": seeds,
            "episodes": episodes,
            "horizon": horizon,
            "root_seed": root_seed,
            "control_hz": control_hz,
            "partial": bool(budget is not None and budget.exhausted),
            "budget": budget.describe() if budget is not None else None,
            "environment": environment_fingerprint(),
        },
        control_hz=control_hz,
    )
    if out is not None:
        result.save(out, name=name, trajectories=save_trajectories)
        if verbose:
            print(f"xevals: wrote {result.directory}", flush=True)
    if verbose:
        print(result.table(), flush=True)
    return result


def _print_cell(result: CellResult, seconds: float) -> None:
    """One progress line per cell. ``print(flush=True)``, no logging framework."""
    if result.skipped and not result.trajectories:
        print(f"  {result.cell.name:28s} skipped: {result.skipped}", flush=True)
        return
    rate = result.trajectories and np.mean([bool(t.success) for t in result.trajectories])
    tail = f"success {rate:.2f}" if result.trajectories else "no episodes"
    if result.errors:
        tail += f"  {len(result.errors)} raised: {result.errors[0][:48]}"
    print(
        f"  {result.cell.name:28s} {len(result.trajectories):3d} eps  {tail}  {seconds:5.1f}s",
        flush=True,
    )


def _env_factory(env: Any) -> Callable[[str, str], Env]:
    """Normalise the three ways an environment can be supplied into one factory.

    The split argument is passed to a factory that accepts it and dropped for one
    that does not, so a caller who does not care about generalisation splits does
    not have to write a signature accommodating them.
    """
    if isinstance(env, str):
        name = env
        return lambda task, split: envs.create(name, split=split)
    if callable(env) and not hasattr(env, "step"):
        import inspect

        try:
            params = inspect.signature(env).parameters
        except (TypeError, ValueError):  # pragma: no cover
            params = {}
        if "split" in params:
            return lambda task, split: env(task=task or None, split=split)  # type: ignore[misc]
        return lambda task, split: env()
    # A single live environment: reused across cells, which is why every cell
    # resets it with its own seed rather than trusting the previous cell's state.
    return lambda task, split: env  # type: ignore[return-value]


def _env_fingerprint(env: Any) -> dict[str, Any]:
    """What was evaluated *in*, including whether reset-from-state round-trips."""
    info: dict[str, Any] = {
        "type": type(env).__name__,
        "action_dim": int(getattr(env, "action_dim", 0)),
        "offline": bool(getattr(env, "offline", False)),
        "declares_limits": envs.limits_of(env) is not None,
        "supports_physics": hasattr(env, "set_physics"),
    }
    for attr in ("env_id", "task", "object_name", "robot", "backend"):
        if hasattr(env, attr):
            info[attr] = getattr(env, attr)
    try:
        info["reset_state_error"] = envs.check_reset_invariant(env)
    except Exception as exc:  # noqa: BLE001 - a fingerprint must never end a run
        info["reset_state_error"] = None
        info["reset_state_note"] = f"{type(exc).__name__}: {exc}"
    return info


def _run_baselines(
    spec: SuiteSpec,
    make_env: Callable[[str, str], Env],
    action_dim: int,
    *,
    root_seed: int,
    seeds: Sequence[int],
    episodes: int,
    horizon: int,
    verbose: bool,
) -> dict[str, dict[str, Any]]:
    """Run the reference policies on the clean cell, on the model's own seeds."""
    clean = spec.clean_cell
    out: dict[str, dict[str, Any]] = {}
    for name in spec.baselines:
        probe = make_env(clean.task or "", clean.split)
        if name == "random":
            policy = random_policy(action_dim, seed=root_seed)
        elif name == "noop":
            policy = noop_policy(action_dim)
        else:
            policy = replay_policy(probe)
        _close(probe)
        if policy is None:
            out[name] = {
                "success_rate": None,
                "reason": "the environment supplies no demonstrator actions",
                "gate": True,
                "passed": None,
            }
            continue
        result = run_cell(
            policy,
            make_env,
            clean,
            root_seed=root_seed,
            seeds=seeds,
            episodes=episodes,
            horizon=horizon,
            record=0,
        )
        rate = (
            float(np.mean([bool(t.success) for t in result.trajectories]))
            if result.trajectories
            else None
        )
        out[name] = {
            "success_rate": rate,
            "episodes": len(result.trajectories),
            "gate": name == "replay",
            # The gate's threshold is deliberately low. It is not asking whether
            # the demonstrator is good; it is asking whether the environment is
            # the one the demonstrator acted in at all.
            "passed": None if rate is None else (rate >= 0.5 if name == "replay" else True),
        }
        if verbose:
            shown = "n/a" if rate is None else f"{rate:.2f}"
            print(f"  baseline {name:20s} success {shown}", flush=True)
    return out


#: How a metric measured in several cells collapses to one number per dimension.
#: Stated here rather than left implicit, because the choice changes what the
#: dimension *means* -- and averaging everything, the obvious default, is wrong
#: for four of the seven.
#:
#: ``clean``  the unperturbed cell's value. "Does it do the task?" is a question
#:            about the nominal condition; averaging in the perturbed cells would
#:            make accuracy a second, worse robustness score.
#: ``mean``   the average over the cells where the metric applies. Right for
#:            robustness, which is a claim about behaviour across conditions, and
#:            for efficiency, which barely varies between them.
#: ``worst``  the worst value over those cells. Right for safety and security,
#:            which are worst-case properties: a policy that stays inside the
#:            workspace on average and leaves it whenever the camera moves is not
#:            safe, and an attack that works is not averaged away by ones that do
#:            not.
AGGREGATION: dict[Dimension, str] = {
    Dimension.ACCURACY: "clean",
    Dimension.ROBUSTNESS: "mean",
    Dimension.SAFETY: "worst",
    Dimension.SECURITY: "worst",
    Dimension.EFFICIENCY: "mean",
    Dimension.GENERALIZATION: "clean",
    Dimension.CONSISTENCY: "clean",
}


def _aggregate(values: list[float], how: str, *, higher_is_better: bool) -> float:
    """Collapse one metric's per-cell values according to its dimension's rule."""
    if how == "worst":
        return min(values) if higher_is_better else max(values)
    return float(np.mean(values))


def dimension_scores(
    spec: SuiteSpec,
    cells: dict[str, CellResult],
    *,
    control_hz: float = 10.0,
) -> dict[str, DimensionScore]:
    """Collapse every cell's metrics into one score per dimension.

    A metric measured in several cells is collapsed by its dimension's rule in
    :data:`AGGREGATION`, so a suite that happens to schedule twelve robustness
    cells and one accuracy cell does not thereby weight robustness twelve times.
    """
    collected: dict[Dimension, dict[str, list[float]]] = {}
    clean_values: dict[Dimension, dict[str, float]] = {}
    skipped: dict[Dimension, dict[str, str]] = {}
    directions: dict[str, bool] = {}
    clean_name = spec.clean_cell.name
    for result in cells.values():
        for name, value in result.values.items():
            # A bracketed name -- ``robustness/severity_auc[visual/blur]`` -- is a
            # per-family detail kept for the table. The pooled metric of the same
            # base name already represents it, so counting both would weight a
            # suite's dimension score by how many families it happened to run.
            if "[" in name:
                continue
            entry = metrics.METRICS.entry(name)
            dimension = Dimension(entry.meta["dimension"])
            directions[name] = bool(entry.meta.get("higher_is_better", True))
            if not entry.meta.get("contributes", True):
                continue  # reported everywhere, but not folded into the score
            if value.measured:
                collected.setdefault(dimension, {}).setdefault(name, []).append(
                    float(value.value)  # type: ignore[arg-type]
                )
                if result.cell.name == clean_name:
                    clean_values.setdefault(dimension, {})[name] = float(value.value)
            else:
                skipped.setdefault(dimension, {}).setdefault(name, value.reason)
    out: dict[str, DimensionScore] = {}
    for dimension in spec.dimensions:
        how = AGGREGATION.get(dimension, "mean")
        values = {}
        for name, per_cell in collected.get(dimension, {}).items():
            if how == "clean" and name in clean_values.get(dimension, {}):
                values[name] = clean_values[dimension][name]
            else:
                values[name] = _aggregate(
                    per_cell, how, higher_is_better=directions.get(name, True)
                )
        reasons = {
            n: r for n, r in skipped.get(dimension, {}).items() if n not in values
        }
        out[dimension.value] = score_dimension(
            dimension, values, higher_is_better=directions, skipped=reasons
        )
    return out


def score_cells(
    spec: SuiteSpec,
    cells: dict[str, CellResult],
    *,
    model_info: dict[str, Any] | None = None,
    control_hz: float = 10.0,
) -> None:
    """Compute every cell's metrics in place, including the paired ones.

    Two passes rather than one: the severity ladder's area-under-curve and the
    worst-case success both need every cell's success rate, which is only known
    once the first pass is done.
    """
    for result in cells.values():
        if not result.trajectories:
            continue
        reference = cells.get(result.cell.pair_with or "")
        result.values = metrics.compute(
            result.cell.metrics,
            result.trajectories,
            ref=reference.trajectories if reference and reference is not result else None,
            model_info=model_info or {},
            control_hz=control_hz,
        )
    # Second pass: the summaries that span cells rather than living inside one.
    # A generalisation gap needs the in-distribution *and* out-of-distribution
    # episodes together, and determinism needs a seed that two cells both ran;
    # computed per cell, each would see only half its own input and report
    # itself unmeasurable. They are attached to the clean cell because they
    # describe the suite, not any single condition.
    clean = cells.get(spec.clean_cell.name)
    if clean is None or not clean.trajectories:
        return
    _cross_cell_metrics(spec, cells, clean)
    per_cell = {
        name: r.success_rate for name, r in cells.items() if r.success_rate is not None
    }
    if len(per_cell) > 1:
        clean.values["robustness/worst_case"] = metrics.worst_case([], per_cell=per_cell)
    for family, curve in _severity_curves(cells).items():
        value = metrics.severity_auc([], curve=curve)
        if value.measured:
            clean.values[f"robustness/severity_auc[{family}]"] = value
    curves = _severity_curves(cells)
    if curves:
        pooled = {
            severity: float(np.mean([c[severity] for c in curves.values() if severity in c]))
            for severity in sorted({s for c in curves.values() for s in c})
        }
        clean.values["robustness/severity_auc"] = metrics.severity_auc([], curve=pooled)



def _cross_cell_metrics(spec: SuiteSpec, cells: dict[str, CellResult], clean: CellResult) -> None:
    """Metrics whose input is several cells pooled, not one cell's episodes.

    Only unperturbed cells are pooled: a generalisation gap measured against a
    blurred out-of-distribution cell would confound the two dimensions, and the
    resulting number would be attributed to whichever heading it was filed under.
    """
    unperturbed = [
        t
        for result in cells.values()
        if result.cell.perturbation in ("", "none")
        for t in result.trajectories
    ]
    # Determinism and seed spread pool only cells that are the *same condition*
    # -- clean and its deliberate repeat. Pooling the generalisation splits in
    # would compare two episodes that share a seed but not a world, and report a
    # deterministic model as non-deterministic because the layout differed.
    identical = [
        t
        for result in cells.values()
        if result.cell.perturbation in ("", "none") and result.cell.split == "in"
        for t in result.trajectories
    ]
    for names, pool in (
        (("generalization/ood_success", "generalization/gap", "generalization/per_split"),
         unperturbed),
        (("consistency/determinism", "consistency/seed_std"), identical),
    ):
        if not pool:
            continue
        for name in names:
            if name not in clean.cell.metrics:
                continue
            value = metrics.compute([name], pool)[name]
            if value.measured or not clean.values.get(name, value).measured:
                clean.values[name] = value


def _severity_curves(cells: dict[str, CellResult]) -> dict[str, dict[float, float]]:
    """Success against severity, per perturbation family.

    Anchored at severity 0 with the clean cell's success rate, so a curve always
    starts from the model's own baseline rather than from an assumed 1.0.
    """
    clean = next((r for r in cells.values() if r.cell.is_clean), None)
    base = clean.success_rate if clean else None
    curves: dict[str, dict[float, float]] = {}
    for result in cells.values():
        cell = result.cell
        if cell.perturbation in ("", "none") or result.success_rate is None:
            continue
        family = cell.perturbation
        curve = curves.setdefault(family, {})
        if base is not None:
            curve[0.0] = base
        curve[float(cell.severity)] = float(result.success_rate)
    return {f: c for f, c in curves.items() if len(c) >= 2}


def check_capabilities(model: Any) -> dict[str, bool]:
    """Which optional capabilities a model exposes. Drives null-with-reason."""
    return {
        name: callable(getattr(model, name, None))
        for name in ("reset", "confidence", "encode", "cost", "predict", "plan", "score")
    }


def limits_summary(env: Any) -> SafetyLimits | None:
    """The environment's declared limits, re-exported for callers building reports."""
    return envs.limits_of(env)
