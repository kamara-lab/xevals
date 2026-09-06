"""The HTML report: self-contained, and caveats above the numbers."""

from __future__ import annotations

import xevals
from xevals import report
from xevals.results import Result


def test_the_report_builds_from_a_saved_run_without_the_model(tmp_path, scripted):
    result = xevals.evaluate(
        scripted, "synthetic/reach", suite="smoke", episodes=3, seeds=(0,),
        out=tmp_path, verbose=False,
    )
    reloaded = Result.load(result.directory)
    path = report.write(reloaded, tmp_path / "rebuilt.html")

    html = path.read_text()
    assert html.startswith("<!doctype html>")
    assert "accuracy" in html and "robustness" in html


def test_the_pages_share_one_stylesheet(tmp_path, scripted):
    result = xevals.evaluate(
        scripted, "synthetic/reach", suite="smoke", episodes=2, seeds=(0,),
        out=tmp_path, verbose=False,
    )
    # A handful of pages in one directory, not one very long one. The CSS is
    # downloaded once instead of being inlined into each of them.
    assert (result.directory / "report.css").exists()
    for page in ("report.html", "conditions.html", "episodes.html", "metrics.html"):
        html = (result.directory / page).read_text()
        assert 'href="report.css"' in html, page
        # Figures stay base64 and nothing is fetched from the network, so the
        # directory still opens offline.
        assert "http://" not in html.replace("http://www.w3.org", ""), page


def test_render_alone_still_produces_a_standalone_page(tmp_path, scripted):
    result = xevals.evaluate(
        scripted, "synthetic/reach", suite="smoke", episodes=2, seeds=(0,),
        out=None, verbose=False,
    )
    html = report.render(result, directory=tmp_path)
    assert "<style>" in html, "inlined, so render() alone gives something that opens"
    assert 'href="report.css"' not in html


def test_a_failed_gate_appears_above_the_numbers(tmp_path, scripted):
    result = xevals.evaluate(
        scripted, "synthetic/reach", suite="smoke", episodes=2, seeds=(0,),
        out=None, verbose=False,
    )
    result.baselines["replay"] = {"success_rate": 0.0, "gate": True, "passed": False}
    html = report.render(result, directory=tmp_path)

    assert html.index("Replay gate failed") < html.index("Dimensions")


def test_an_unmeasured_dimension_says_so_rather_than_showing_a_zero(tmp_path, scripted):
    result = xevals.evaluate(
        scripted, "synthetic/reach", suite="full", episodes=2, seeds=(0,),
        out=None, verbose=False, baselines=False,
    )
    html = report.render(result, directory=tmp_path)
    if any(score.score is None for score in result.dimensions.values()):
        assert "not measured" in html


def test_the_provenance_page_says_how_to_reproduce_the_run(tmp_path, scripted):
    result = xevals.evaluate(
        scripted, "synthetic/reach", suite="smoke", episodes=3, seeds=(0, 1),
        out=tmp_path, verbose=False,
    )
    html = (result.directory / "provenance.html").read_text()
    assert "xevals run" in html
    assert "run.root_seed=0" in html
    assert "config hash" in html


def test_every_page_carries_the_same_nav(tmp_path, scripted):
    result = xevals.evaluate(
        scripted, "synthetic/reach", suite="smoke", episodes=2, seeds=(0,),
        out=tmp_path, verbose=False,
    )
    pages = [p.name for p in result.directory.glob("*.html")]
    for page in pages:
        html = (result.directory / page).read_text()
        for other in pages:
            assert f'href="{other}"' in html, f"{page} cannot reach {other}"
        # Exactly one tab is marked current, and it is this page's.
        assert html.count('aria-current="page"') == 1


def test_the_episodes_page_groups_clips_by_condition(tmp_path, scripted):
    result = xevals.evaluate(
        scripted, "synthetic/reach", suite="smoke", episodes=3, seeds=(0,),
        out=tmp_path, record=2, verbose=False,
    )
    html = (result.directory / "episodes.html").read_text()
    assert "Episodes by condition" in html
    # Clips are shown, not linked: a gallery of links is a gallery nobody opens.
    assert "<img" in html and 'src="videos/' in html
    assert "<div class=\"cellblock\">" in html or 'class="cellblock"' in html
