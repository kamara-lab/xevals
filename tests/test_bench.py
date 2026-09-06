"""Benchmarks: several models under conditions that are identical by construction."""

from __future__ import annotations

import json

import numpy as np
import pytest

import xevals
from xevals.bench import Benchmark


class Lazy:
    """A policy that does nothing, so a benchmark has something to be better than."""

    def act(self, obs, *, instruction=None):
        return np.zeros(2, dtype=np.float32)

    def describe(self):
        return {"adapter": "test", "module": "Lazy"}


@pytest.fixture
def bench(scripted, tmp_path):
    return xevals.benchmark(
        {"scripted": scripted, "lazy": Lazy()},
        "synthetic/reach",
        suite="smoke",
        episodes=4,
        seeds=(0, 1),
        out=tmp_path,
        verbose=False,
    )


def test_every_model_sees_the_same_episodes(scripted):
    result = xevals.benchmark(
        {"a": scripted, "b": scripted},
        "synthetic/reach", suite="smoke", episodes=3, seeds=(0,), out=None, verbose=False,
    )
    # Seeds derive from (root, seed, index) and never from the model, so two runs
    # of the same policy are identical -- which is what makes a comparison of two
    # *different* policies a paired one.
    left, right = result["a"], result["b"]
    for cell in left.cells:
        for name, value in left.cells[cell].items():
            if name.startswith("efficiency/"):
                continue
            assert value.value == right.cells[cell][name].value, f"{cell}/{name}"


def test_the_baselines_and_gate_are_measured_once_and_quoted_identically(bench):
    # They are properties of the environment, not of any model. Measuring them
    # per model wastes the time and lets rows of one leaderboard disagree about
    # what the floor was.
    rates = {name: result.baselines for name, result in bench.results.items()}
    assert len({json.dumps(r, sort_keys=True, default=str) for r in rates.values()}) == 1
    assert bench.baselines["random"]["success_rate"] is not None


def test_the_leaderboard_ranks_and_keeps_the_per_dimension_columns(bench):
    headers, rows = bench.rows()
    assert headers[0] == "model" and headers[-1] == "mean"
    assert "accuracy" in headers and "robustness" in headers
    assert [r[0] for r in rows] == ["scripted", "lazy"]
    assert rows[0][-1] > rows[1][-1]


def test_insertion_order_is_kept_but_the_table_sorts(scripted):
    result = xevals.benchmark(
        [("lazy", Lazy()), ("scripted", scripted)],
        "synthetic/reach", suite="smoke", episodes=3, seeds=(0,), out=None, verbose=False,
    )
    assert list(result.results) == ["lazy", "scripted"], "the caller's order is meaningful"
    _headers, rows = result.rows()
    assert rows[0][0] == "scripted", "but the table ranks"


def test_the_cell_table_puts_the_models_side_by_side(bench):
    headers, rows = bench.cell_rows()
    assert headers[:3] == ["cell", "perturbation", "severity"]
    assert headers[3:] == ["scripted", "lazy"]
    clean = next(r for r in rows if r[0] == "clean")
    assert clean[3] > clean[4]


def test_disagreements_ignore_a_model_that_is_simply_worse(scripted):
    # A uniformly weaker model differs in every cell, so ranking by the raw gap
    # returns the whole suite. The excess -- gap minus the clean-cell gap -- is
    # what answers "which condition pulls them apart further".
    result = xevals.benchmark(
        {"scripted": scripted, "lazy": Lazy()},
        "synthetic/reach", suite="robustness", episodes=3, seeds=(0, 1),
        out=None, verbose=False, baselines=False,
    )
    for _cell, _best, _worst, gap, excess in result.disagreements():
        assert excess <= gap + 1e-9
    assert all(row[0] != "clean" for row in result.disagreements())


def test_best_handles_a_dimension_and_an_unmeasurable_one(bench):
    assert bench.best() == "scripted"
    assert bench.best("accuracy") == "scripted"
    assert bench.best("security") is None, "the smoke suite claims no security dimension"


def test_a_benchmark_directory_holds_every_run_plus_the_comparison(bench):
    directory = bench.directory
    assert (directory / "leaderboard.md").exists()
    assert (directory / "cells.json").exists()
    assert (directory / "index.html").exists()
    runs = sorted(p.name for p in directory.iterdir() if p.is_dir())
    assert len(runs) == 2
    assert all((directory / r / "run.json").exists() for r in runs)


def test_the_comparison_page_links_to_directories_that_exist(bench):
    import re

    html = (bench.directory / "index.html").read_text()
    links = re.findall(r'href="([^"]+report\.html)"', html)
    assert len(links) == 2
    # A slug-only link 404s from a shared page, and does it silently.
    assert all((bench.directory / link).exists() for link in links)


def test_a_benchmark_round_trips_through_disk(bench):
    back = Benchmark.load(bench.directory)
    assert set(back.results) == set(bench.results)
    assert back.gate_status() == bench.gate_status()
    assert back.rows()[1][0][-1] == pytest.approx(bench.rows()[1][0][-1])


def test_duplicate_model_names_are_rejected(scripted):
    with pytest.raises(ValueError, match="duplicate model name"):
        xevals.benchmark(
            [("same", scripted), ("same", Lazy())], "synthetic/reach", out=None, verbose=False
        )


def test_a_benchmark_needs_at_least_one_model():
    with pytest.raises(ValueError, match="at least one model"):
        xevals.benchmark({}, "synthetic/reach", out=None, verbose=False)


def test_the_budget_is_shared_across_models_not_per_model(scripted):
    from xevals.runner import Budget

    result = xevals.benchmark(
        {"a": scripted, "b": Lazy(), "c": scripted},
        "synthetic/reach", suite="robustness", episodes=5, seeds=(0,),
        out=None, verbose=False, baselines=False, budget=Budget(episodes=12),
    )
    # A benchmark that spends its whole allowance on the first model has not
    # produced a comparison, so the cap counts episodes across all of them.
    total = sum(
        sum(v.n for name, v in cells.items() if name == "accuracy/success_rate")
        for r in result.results.values()
        for cells in r.cells.values()
    )
    assert total <= 12


def test_a_failed_gate_invalidates_the_whole_benchmark(bench):
    bench.baselines["replay"] = {"success_rate": 0.0, "gate": True, "passed": False}
    assert bench.gate_failed
    assert bench.table().startswith("> **Gate failed.**")
