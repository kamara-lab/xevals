"""Trade-offs, and above all the refusal to claim one that is not there."""

from __future__ import annotations

import numpy as np
import pytest

import xevals
from xevals.tradeoffs import (
    COMPETENCE_CAUTION,
    MIN_POINTS,
    Point,
    TradeOffAnalysis,
    _exchange_rate,
    _knee,
    _pareto,
    _tension,
    analyse,
    spearman,
)


def line(n=14, slope=-0.8, noise=0.05, seed=0):
    """Points on a genuine frontier: more x, less y."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.1, 0.95, n)
    ys = np.clip(1.0 + slope * xs + rng.normal(0, noise, n), 0, 1)
    return [Point(f"c{i}", float(x), float(y), float(x), float(y), 20)
            for i, (x, y) in enumerate(zip(xs, ys, strict=True))]


# -- the statistic ---------------------------------------------------------


def test_spearman_is_rank_based_not_linear():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # A monotone but very non-linear relation is a perfect rank correlation.
    assert spearman(x, x**5) == pytest.approx(1.0)
    assert spearman(x, -(x**5)) == pytest.approx(-1.0)


def test_ties_get_average_ranks():
    # Naive argsort-of-argsort invents an ordering among equal values and gets
    # this wrong, which is how a tied axis produces a fictitious correlation.
    x = np.array([1.0, 1.0, 1.0, 2.0])
    y = np.array([5.0, 5.0, 5.0, 9.0])
    assert spearman(x, y) == pytest.approx(1.0)


def test_a_constant_axis_has_no_correlation():
    assert np.isnan(spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


# -- the verdict -----------------------------------------------------------


def test_a_real_trade_off_is_detected():
    tension, rho, ci, _reason = _tension(line())
    assert tension == "present"
    assert rho < 0 and ci[1] < 0


def test_axes_that_move_together_are_reported_as_absent():
    points = [Point(f"c{i}", x, x, x, x, 20) for i, x in enumerate(np.linspace(0.1, 0.9, 12))]
    tension, _rho, ci, reason = _tension(points)
    assert tension == "absent"
    assert ci[0] > 0
    assert "rise and fall together" in reason


def test_noise_is_undetermined_rather_than_a_trade_off():
    rng = np.random.default_rng(1)
    points = [
        Point(f"c{i}", float(rng.random()), float(rng.random()), 0.0, 0.0, 20)
        for i in range(12)
    ]
    tension, _rho, ci, reason = _tension(points)
    # The failure this whole module exists to avoid: a frontier drawn through
    # scatter. The interval spans zero, so no claim is made.
    assert tension == "undetermined"
    assert ci[0] < 0 < ci[1]
    assert "spans zero" in reason


def test_too_few_points_is_insufficient_not_undetermined():
    tension, rho, ci, reason = _tension(line(n=MIN_POINTS - 1))
    assert tension == "insufficient"
    assert rho is None and ci is None
    assert str(MIN_POINTS) in reason


def test_a_heavily_tied_axis_is_flagged_in_the_reason():
    points = [Point(f"c{i}", float(i) / 12, 1.0, 0.0, 0.0, 20) for i in range(12)]
    points += [Point(f"d{i}", float(i) / 12, 0.2, 0.0, 0.0, 20) for i in range(3)]
    _tension_, _rho, _ci, reason = _tension(points)
    assert "takes one value in" in reason


def test_the_verdict_is_reproducible():
    points = line()
    assert _tension(points) == _tension(points)


# -- the frontier ----------------------------------------------------------


def test_the_frontier_keeps_only_non_dominated_points():
    points = [
        Point("best-x", 1.0, 0.2, 1.0, 0.2),
        Point("best-y", 0.2, 1.0, 0.2, 1.0),
        Point("middle", 0.6, 0.6, 0.6, 0.6),
        Point("dominated", 0.5, 0.5, 0.5, 0.5),
    ]
    assert set(_pareto(points)) == {"best-x", "best-y", "middle"}


def test_identical_points_are_both_kept():
    # Dropping one would make the frontier depend on iteration order.
    points = [Point("a", 0.5, 0.5, 0.5, 0.5), Point("b", 0.5, 0.5, 0.5, 0.5)]
    assert set(_pareto(points)) == {"a", "b"}


def test_the_exchange_rate_is_the_frontier_slope():
    points = line(slope=-0.8, noise=0.0)
    rate = _exchange_rate(points, _pareto(points))
    assert rate == pytest.approx(-0.8, abs=0.05)


def test_the_knee_needs_three_frontier_points():
    assert _knee(line(n=2), ["c0", "c1"]) is None
    assert _knee(line(), _pareto(line())) is not None


# -- end to end ------------------------------------------------------------


class Proportional:
    """A plain proportional controller: competent, but not immune to the suite.

    The conftest policy normalises its step, which on a task this forgiving
    solves every cell of the suite: a success rate of 1.0 everywhere is a
    constant axis, and the honest verdict on a constant axis is the degenerate
    "insufficient". Dropping the normalisation costs real steps under stale
    observations and action noise, so competence has something to vary against.
    """

    def act(self, obs, *, instruction=None):
        state, goal = obs.get("state"), obs.get("goal")
        if state is None or goal is None:
            return np.zeros(2, dtype=np.float32)
        return np.clip(goal - state[:2], -1.0, 1.0).astype(np.float32)


@pytest.fixture
def result():
    return xevals.evaluate(
        Proportional(), "synthetic/reach", suite="full", episodes=3, seeds=(0, 1),
        out=None, verbose=False, baselines=False,
    )


def test_both_built_ins_are_registered():
    assert xevals.tradeoffs.available() == [
        "competence-caution",
        "generality-attackability",
    ]
    described = xevals.tradeoffs.describe("competence-caution")
    assert described["scopes"] == ["cells", "models"]


def test_every_trade_off_states_why_it_might_be_one():
    for name in xevals.tradeoffs.available():
        spec = xevals.tradeoffs.create(name)
        # Without a stated mechanism, "no tension found" is uninterpretable.
        assert len(spec.mechanism) > 80
        assert spec.question.endswith("?")


def test_a_run_analyses_its_cells(result):
    analyses = {a.tradeoff.name: a for a in result.tradeoffs()}
    caution = analyses["competence-caution"]
    assert caution.scope == "cells"
    assert len(caution.points) > MIN_POINTS
    assert caution.tension in ("present", "absent", "undetermined")


def test_no_frontier_is_drawn_without_a_demonstrated_tension(result):
    for analysis in result.tradeoffs():
        if analysis.tension != "present":
            assert analysis.frontier == []
            assert analysis.exchange_rate is None
            assert analysis.knee is None


def test_the_group_contrast_uses_each_groups_own_metric(result):
    generality = next(
        a for a in result.tradeoffs() if a.tradeoff.name == "generality-attackability"
    )
    assert set(generality.groups) == {"nuisance", "chosen"}
    metrics = {g["metric"] for g in generality.groups.values()}
    # Two retentions against the same clean cell, not two raw success rates,
    # which would compare a hard perturbation with an easy one.
    assert metrics == {"robustness/retention", "security/attack_retention"}
    assert generality.gap is not None


def test_a_contrast_only_trade_off_still_produces_something(result):
    generality = next(
        a for a in result.tradeoffs() if a.tradeoff.name == "generality-attackability"
    )
    assert generality.points == []
    assert generality.group_rows()[1], "the contrast is the output at cell scope"
    assert generality.summary()


def test_model_scope_needs_a_benchmark(result):
    analysis = analyse(result, "competence-caution", scope="models")
    assert analysis.tension == "insufficient"
    assert "benchmark" in analysis.reason


def test_a_benchmark_analyses_its_models(scripted):
    class Lazy:
        def act(self, obs, *, instruction=None):
            return np.zeros(2, dtype=np.float32)

    bench = xevals.benchmark(
        {"scripted": scripted, "lazy": Lazy()}, "synthetic/reach", suite="core",
        episodes=3, seeds=(0,), out=None, verbose=False, baselines=False,
    )
    analysis = analyse(bench, "competence-caution", scope="models")
    assert analysis.scope == "models"
    assert len(analysis.points) == 2
    # Two models is a line, not a frontier, and the analysis says so.
    assert analysis.tension == "insufficient"


def test_the_analysis_serialises(result):
    for analysis in result.tradeoffs():
        payload = analysis.describe()
        assert payload["tension"] == analysis.tension
        assert payload["reason"]
        assert "mechanism" in payload


def test_the_run_directory_carries_the_verdicts(tmp_path, scripted):
    import json

    run = xevals.evaluate(
        scripted, "synthetic/reach", suite="full", episodes=3, seeds=(0, 1),
        out=tmp_path, verbose=False, baselines=False,
    )
    payload = json.loads((run.directory / "run.json").read_text())
    names = {t["name"] for t in payload["tradeoffs"]}
    assert names == set(xevals.tradeoffs.available())
    assert (run.directory / "tradeoffs" / "competence-caution.md").exists()


def test_the_report_states_the_verdict_and_the_mechanism(tmp_path, scripted):
    run = xevals.evaluate(
        scripted, "synthetic/reach", suite="full", episodes=3, seeds=(0, 1),
        out=tmp_path, verbose=False, baselines=False,
    )
    html = (run.directory / "tradeoffs.html").read_text()
    assert "Trade-offs" in html
    assert "Why these might trade" in html
    assert (
        "Undetermined" in html
        or "No trade-off here" in html
        or "trade-off is present" in html
    )
    # And the overview links to it rather than burying the verdict.
    assert 'href="tradeoffs.html"' in (run.directory / "report.html").read_text()


def test_the_plot_refuses_a_frontier_it_was_not_given(tmp_path):
    pytest.importorskip("matplotlib")
    from xevals.plots import tradeoff as draw

    points = line()
    undetermined = TradeOffAnalysis(COMPETENCE_CAUTION, "cells", points, "undetermined")
    path = draw(undetermined, tmp_path / "t.png")
    assert path.exists(), "the scatter is still drawn; only the frontier is withheld"
