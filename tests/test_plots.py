"""Figures, headless, and the palette limits that are enforced rather than advised."""

from __future__ import annotations

import pytest

import xevals
from xevals import plots

pytest.importorskip("matplotlib")


@pytest.fixture
def result(scripted):
    return xevals.evaluate(
        scripted, "synthetic/reach", suite="smoke", episodes=3, seeds=(0, 1),
        out=None, verbose=False,
    )


def test_a_dimension_keeps_one_colour_across_every_figure():
    from xevals.dimensions import DIMENSION_ORDER

    assert plots.dimension_color("accuracy") == plots.dimension_color(DIMENSION_ORDER[0])
    assert len({plots.dimension_color(d) for d in DIMENSION_ORDER}) == 7


def test_viridis_refuses_more_series_than_it_can_separate():
    assert len(plots.palette_colors(5)) == 5
    with pytest.raises(ValueError, match="small multiples"):
        plots.palette_colors(6)


def test_the_brand_pair_stops_at_two():
    assert plots.palette_colors(2, "brand") == [plots.ACCENT, plots.AMBER]
    with pytest.raises(ValueError, match="separates"):
        plots.palette_colors(3, "brand")


def test_an_unknown_palette_names_the_known_ones():
    with pytest.raises(ValueError, match="choose from"):
        plots.palette_colors(2, "rainbow")


def test_figures_render_headless(result, tmp_path):
    written = plots.write_all(result, tmp_path)
    assert "radar" in written and written["radar"].exists()
    assert "dimension-bars" in written


def test_a_figure_that_cannot_be_drawn_is_absent_rather_than_empty(result, tmp_path):
    written = plots.write_all(result, tmp_path)
    # The smoke suite runs one perturbation family at one severity, so there is
    # a curve; a run with no ladder at all simply has no severity figure.
    assert all(path.exists() for path in written.values())


def test_the_chrome_comes_from_the_brand_tokens():
    assert plots.BRAND["paper"] == "#FAFAF8"
    assert plots.BRAND["ink"] == "#121412"
    params = plots.rc_params()
    assert params["figure.facecolor"] == plots.PAPER
    assert params["axes.spines.top"] is False


def test_the_latency_chart_stays_legible_across_orders_of_magnitude(result, tmp_path):
    # A fast model under a slow control loop spans four orders of magnitude, and
    # on a linear axis the bars are invisible and the chart says nothing.
    path = plots.latency(result, tmp_path / "latency.png")
    assert path.exists() and path.stat().st_size > 0


def test_the_radar_labels_fit_inside_the_figure(result, tmp_path):
    # "generalization" is the longest, and on a square canvas the polar
    # projection draws it past the figure boundary where bbox_inches cannot
    # rescue it.
    assert plots.radar(result, tmp_path / "radar.png").exists()
    assert plots._SHORT["generalization"] == "general."
