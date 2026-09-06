"""``xevals run|compare|report|list|describe|doctor`` -- argparse, no framework.

Standard-library ``argparse``, and ``main(argv) -> int`` so the CLI is testable
without a subprocess. Every handler imports what it needs *inside* the function,
so ``xevals --help`` and ``xevals doctor`` are fast on a bare install -- which is
exactly the install on which someone is most likely to be running ``doctor``.

This module is not imported by ``xevals/__init__.py``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["main"]

#: The registries ``xevals <name> list|describe`` can walk. One line each, and
#: adding one here is all it takes for the CLI to know about it.
_REGISTRIES = {
    "suites": "xevals.suites",
    "metrics": "xevals.metrics",
    "perturbations": "xevals.perturbations",
    "adapters": "xevals.adapters",
    "envs": "xevals.envs",
    "datasets": "xevals.datasets",
    "judges": "xevals.judges",
    "dimensions": "xevals.dimensions",
    "tradeoffs": "xevals.tradeoffs",
}

#: extra -> (import name, one line on what it unlocks). Read by ``doctor``.
_EXTRAS = {
    "torch": ("torch", "wrap torch models; gradient attacks"),
    "jax": ("jax", "wrap JAX and Equinox models"),
    "sim": ("gymnasium", "Gymnasium environments, and LIBERO / ManiSkill / SimplerEnv"),
    "mujoco": ("mujoco", "the Menagerie manipulation robots"),
    "mujoco (assets)": ("robot_descriptions", "fetching the Menagerie models"),
    "newton": ("newton", "the Newton physics backend for those robots"),
    "data": ("pyarrow", "LeRobot parquet datasets"),
    "data (hdf5)": ("h5py", "RoboMimic and LIBERO datasets"),
    "video": ("imageio", "mp4 recordings"),
    "video (gif)": ("PIL", "GIF fallback and image handling"),
    "plots": ("matplotlib", "radars, severity curves, heatmaps"),
    "judge": ("anthropic", "the optional LLM judge"),
    "xwm": ("xwm", "the sibling world-model library"),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xevals",
        description="Multi-dimensional evaluation of robot-learning models.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser(
        "run",
        help="run an evaluation, or a benchmark if the config has [[models]]",
    )
    run.add_argument("config", type=Path, help="a TOML config")
    run.add_argument(
        "overrides", nargs="*", help="dotted overrides, e.g. run.episodes=50 suite.name=full"
    )
    run.add_argument("--quiet", action="store_true", help="suppress per-cell progress")

    compare = sub.add_parser("compare", help="build a leaderboard over run directories")
    compare.add_argument("runs", nargs="+", type=Path)
    compare.add_argument("--out", type=Path, default=None, help="write the table and radar here")
    compare.add_argument(
        "--diff", action="store_true", help="diff exactly two runs, metric by metric"
    )

    report = sub.add_parser("report", help="rebuild figures and report.html for a saved run")
    report.add_argument("run", type=Path)

    for name in _REGISTRIES:
        registry = sub.add_parser(name, help=f"list or describe {name}")
        registry.add_argument("action", choices=["list", "describe"], nargs="?", default="list")
        registry.add_argument("name", nargs="?", help="an entry to describe")
        registry.add_argument("--family", default=None, help="restrict to one family")
        registry.add_argument("--json", action="store_true", help="machine-readable output")

    sub.add_parser("doctor", help="report which optional extras are installed")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from xevals import __version__

        print(__version__)
        return 0
    if args.command is None:
        parser.print_help()
        return 0
    handler = {
        "run": _run,
        "compare": _compare,
        "report": _report,
        "doctor": _doctor,
    }.get(args.command, _registry)
    try:
        return handler(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"xevals: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    """One config, one command -- whether it names one model or several."""
    from xevals import config as config_module

    cfg = config_module.load(args.config, overrides=args.overrides)
    return _benchmark(cfg, args) if cfg.is_benchmark else _single(cfg, args)


def _target(cfg: Any) -> Any:
    """The environment name, or the factory a ``[target] dataset`` describes."""
    from xevals import datasets

    if not cfg.target.dataset:
        return cfg.target.env
    episodes = datasets.DatasetSpec(
        name=cfg.target.dataset.split(":", 1)[0],
        path=cfg.target.dataset.split(":", 1)[-1],
        **cfg.target.kwargs,
    ).read()
    if not episodes:
        raise ValueError(f"no episodes read from {cfg.target.dataset!r}")
    # An offline evaluation cycles through the recorded episodes, one per cell
    # build, so every cell sees the same corpus in the same order -- and in a
    # benchmark, so does every model, which is what makes their rows comparable.
    counter = {"index": 0}

    def make_env(task: str | None = None, split: str = "in") -> Any:
        episode = episodes[counter["index"] % len(episodes)]
        counter["index"] += 1
        return episode.to_env()

    return make_env


def _suite(cfg: Any) -> Any:
    """The suite a config names, or the one it spells out inline."""
    from xevals import suites

    if not cfg.suite.cells:
        return cfg.suite.name
    return suites.from_dict(
        {
            "name": cfg.suite.name,
            "cells": cfg.suite.cells,
            "dimensions": cfg.suite.dimensions or None,
            "baselines": cfg.suite.baselines or None,
        }
    )


def _shared_kwargs(cfg: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Everything ``evaluate`` and ``benchmark`` take identically."""
    return {
        "suite": _suite(cfg),
        "episodes": cfg.run.episodes,
        "seeds": cfg.run.seeds,
        "horizon": cfg.run.horizon,
        "record": cfg.run.record,
        "out": cfg.output.dir,
        "root_seed": cfg.run.root_seed,
        "budget": cfg.run.budget_seconds,
        "baselines": cfg.run.baselines,
        "control_hz": cfg.run.control_hz,
        "save_trajectories": cfg.run.save_trajectories,
        "verbose": not args.quiet,
    }


def _single(cfg: Any, args: argparse.Namespace) -> int:
    """One model."""
    from xevals import config as config_module
    from xevals import evaluate

    result = evaluate(
        _load_model(cfg.model),
        _target(cfg),
        name=cfg.output.name or None,
        **_shared_kwargs(cfg, args),
    )
    if result.directory is not None:
        config_module.save(cfg, result.directory, source=args.config)
    # A failed gate is an exit code, not a footnote: a pipeline that publishes
    # these numbers should stop rather than publish them.
    return 1 if result.gate_failed else 0


def _benchmark(cfg: Any, args: argparse.Namespace) -> int:
    """Several models, one suite, one set of seeds, one leaderboard."""
    from xevals import benchmark
    from xevals import config as config_module

    result = benchmark(
        [(m.name, _load_model(m)) for m in cfg.models],
        _target(cfg),
        name=cfg.output.name or "benchmark",
        **_shared_kwargs(cfg, args),
    )
    if result.directory is not None:
        config_module.save(cfg, result.directory, source=args.config)
    return 1 if result.gate_failed else 0


def _load_model(model_config: Any) -> Any:
    """Resolve ``model.target`` into something :func:`xevals.wrap` accepts.

    ``module:attribute`` imports and calls it if it is a factory; a URL becomes a
    remote policy; anything else is passed to the named adapter as-is, which is
    how a checkpoint path or a Hub id reaches ``hf_vla``.
    """
    from xevals import adapters

    target = model_config.target
    if not target:
        raise ValueError("model.target is required: 'module:attribute', a URL, or a model id")
    if target.startswith(("http://", "https://")):
        return adapters.RemotePolicy(target, **model_config.kwargs)
    if ":" in target and not target.startswith(("/", ".")):
        module_name, _, attribute = target.partition(":")
        obj = getattr(importlib.import_module(module_name), attribute)
        if callable(obj) and not hasattr(obj, "act"):
            try:
                obj = obj(**model_config.kwargs)
            except TypeError:
                pass
        return adapters.wrap(obj, adapter=model_config.adapter or None)
    if not model_config.adapter:
        raise ValueError(
            f"model.target {target!r} is not 'module:attribute' or a URL, so "
            "model.adapter must say which adapter loads it (e.g. 'hf_vla')"
        )
    return adapters.create(model_config.adapter, target, **model_config.kwargs)


def _compare(args: argparse.Namespace) -> int:
    from xevals.results import Result, compare, leaderboard, markdown_table, save_table

    if args.diff:
        if len(args.runs) != 2:
            raise ValueError("--diff takes exactly two run directories")
        headers, rows = compare(args.runs[0], args.runs[1])
    else:
        headers, rows = leaderboard(args.runs)
    print(markdown_table(headers, rows))
    if args.out:
        save_table(Path(args.out) / ("diff" if args.diff else "leaderboard"), headers, rows)
        if not args.diff:
            try:
                from xevals.plots import leaderboard_radar

                results = [Result.load(p) for p in args.runs]
                path = leaderboard_radar(results, Path(args.out) / "radar.png")
                print(f"xevals: wrote {path}")
            except Exception as exc:  # noqa: BLE001 - a missing extra must not fail compare
                print(f"xevals: radar skipped ({type(exc).__name__}: {exc})")
        print(f"xevals: wrote {args.out}")
    return 0


def _report(args: argparse.Namespace) -> int:
    from xevals import plots, report
    from xevals.results import Result

    result = Result.load(args.run)
    directory = result.directory or Path(args.run)
    try:
        plots.write_all(result, directory / "figures")
    except Exception as exc:  # noqa: BLE001
        print(f"xevals: figures skipped ({type(exc).__name__}: {exc})")
    path = report.write(result, directory / "report.html")
    print(f"xevals: wrote {path}")
    return 0


def _registry(args: argparse.Namespace) -> int:
    module = importlib.import_module(_REGISTRIES[args.command])
    if args.command == "dimensions":
        return _dimensions(args)
    if args.action == "describe" or args.name:
        payload = module.describe(args.name)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    names = (
        module.available(args.family)
        if args.family and "family" in module.available.__code__.co_varnames
        else module.available()
    )
    if args.json:
        print(json.dumps(names, indent=2))
        return 0
    described = module.describe()
    width = max((len(n) for n in names), default=0)
    for name in names:
        summary = described.get(name, {}).get("summary", "")
        requires = described.get(name, {}).get("requires") or []
        tail = f"  [needs {', '.join(requires)}]" if requires else ""
        print(f"{name.ljust(width)}  {summary}{tail}")
    return 0


def _dimensions(args: argparse.Namespace) -> int:
    from xevals.dimensions import DIMENSION_ORDER, NORMALISERS
    from xevals.metrics import for_dimension

    if args.json:
        print(
            json.dumps(
                {
                    d.value: {
                        "question": d.question,
                        "metrics": for_dimension(d),
                        "normalisation": {
                            m: {"kind": n.kind, "reference": n.reference, "why": n.why}
                            for m, n in NORMALISERS.items()
                            if m in for_dimension(d)
                        },
                    }
                    for d in DIMENSION_ORDER
                },
                indent=2,
            )
        )
        return 0
    for dimension in DIMENSION_ORDER:
        print(f"{dimension.value:16s} {dimension.question}")
        for name in for_dimension(dimension):
            print(f"  {name}")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    """Report which extras are installed. Never raises: that is the whole job."""
    from xevals import __version__

    print(f"xevals {__version__}  python {sys.version.split()[0]}")
    width = max(len(name) for name in _EXTRAS)
    for extra, (module_name, purpose) in _EXTRAS.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            mark = "ok"
        except Exception:  # noqa: BLE001 - a broken install must still report
            version, mark = "--", "missing"
        print(f"  {extra.ljust(width)}  {mark:8s} {version:12s} {purpose}")
    print("\nThe core needs numpy only; everything above fails when used, not at import.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
