"""The run directory, tables, leaderboards and diffs."""

from __future__ import annotations

import json

import pytest

import xevals
from xevals.results import SCHEMA, Result, compare, csv_table, format_cell, leaderboard


@pytest.fixture
def saved(tmp_path, scripted):
    result = xevals.evaluate(
        scripted, "synthetic/reach", suite="smoke", episodes=3, seeds=(0, 1),
        out=tmp_path, verbose=False,
    )
    return result


def test_a_run_directory_carries_everything_needed_to_read_it(saved):
    directory = saved.directory
    for name in ("run.json", "config.json", "table.md", "table.tex", "table.csv",
                 "metrics.json", "cells.md"):
        assert (directory / name).exists(), name


def test_the_run_file_records_the_normalisation_so_scores_can_be_recomputed(saved):
    payload = json.loads((saved.directory / "run.json").read_text())
    assert payload["schema"] == SCHEMA
    normalisation = payload["normalisation"]["accuracy/success_rate"]
    assert normalisation["kind"] == "unit"
    assert payload["run"]["environment"]["python"]


def test_a_run_round_trips_through_disk(saved):
    back = Result.load(saved.directory)
    assert back.scores.keys() == saved.scores.keys()
    assert back.gate_status() == saved.gate_status()
    for cell in saved.cells:
        for name, value in saved.cells[cell].items():
            assert back.cells[cell][name].value == value.value
            assert back.cells[cell][name].reason == value.reason


def test_a_future_schema_refuses_to_load_rather_than_meaning_something_else(saved):
    path = saved.directory / "run.json"
    payload = json.loads(path.read_text())
    payload["schema"] = SCHEMA + 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="upgrade xevals"):
        Result.load(saved.directory)


def test_json_keeps_full_precision_and_the_table_rounds(saved):
    payload = json.loads((saved.directory / "metrics.json").read_text())
    values = [row["value"] for row in payload["rows"] if isinstance(row["value"], float)]
    assert any(len(repr(v).split(".")[-1]) > 4 for v in values), "storage must not round"


def test_a_missing_number_prints_as_a_dash_not_a_zero():
    assert format_cell(None) == "--"
    assert format_cell(float("nan")) == "--"
    assert format_cell(True) == "yes"
    assert format_cell(3) == "3"


def test_csv_keeps_full_precision():
    text = csv_table(["a"], [[1 / 3]])
    assert "0.3333333333333333" in text


def test_the_config_hash_ignores_where_the_run_was_written(tmp_path, scripted):
    kwargs = dict(suite="smoke", episodes=2, seeds=(0,), verbose=False)
    a = xevals.evaluate(scripted, "synthetic/reach", out=tmp_path / "a", **kwargs)
    b = xevals.evaluate(scripted, "synthetic/reach", out=tmp_path / "b", **kwargs)
    assert a.config_hash() == b.config_hash()


def test_a_leaderboard_orders_runs_and_keeps_the_per_dimension_columns(tmp_path, scripted):
    class Lazy:
        def act(self, obs, *, instruction=None):
            import numpy as np

            return np.zeros(2, dtype=np.float32)

    good = xevals.evaluate(scripted, "synthetic/reach", suite="smoke", episodes=3,
                          seeds=(0,), out=tmp_path / "good", verbose=False)
    bad = xevals.evaluate(Lazy(), "synthetic/reach", suite="smoke", episodes=3,
                         seeds=(0,), out=tmp_path / "bad", verbose=False)

    headers, rows = leaderboard([good.directory, bad.directory])
    assert headers[:3] == ["run", "model", "gate"]
    assert "accuracy" in headers
    assert rows[0][-1] >= rows[1][-1], "sorted by mean dimension score"


def test_a_diff_flags_only_differences_the_error_bars_support(tmp_path, scripted):
    class Lazy:
        def act(self, obs, *, instruction=None):
            import numpy as np

            return np.zeros(2, dtype=np.float32)

    good = xevals.evaluate(scripted, "synthetic/reach", suite="smoke", episodes=6,
                          seeds=(0, 1), out=tmp_path / "good", verbose=False)
    bad = xevals.evaluate(Lazy(), "synthetic/reach", suite="smoke", episodes=6,
                         seeds=(0, 1), out=tmp_path / "bad", verbose=False)

    headers, rows = compare(good.directory, bad.directory)
    assert headers == ["cell", "metric", "a", "b", "delta", "score delta", "significant"]
    success = [r for r in rows if r[1] == "accuracy/success_rate" and r[0] == "clean"]
    assert success and success[0][-1] is True, "a total collapse must read as significant"

    # Ordered by the normalised change, not the raw one. Raw deltas are not
    # comparable across metrics, and an unbounded one would take every top row.
    bounded = {"accuracy/success_rate", "robustness/retention"}
    assert rows[0][1] in bounded, rows[0]
    assert all(abs(rows[i][5]) >= abs(rows[i + 1][5]) for i in range(len(rows) - 1)
               if rows[i][-1] == rows[i + 1][-1])
