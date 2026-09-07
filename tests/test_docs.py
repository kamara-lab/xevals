"""The dimension pages must document exactly the metrics that are registered.

A metric added to the registry and not to its page is a metric nobody finds, and
a page describing a metric that no longer exists is worse than one that omits it.
Neither shows up in `mkdocs build --strict`, which checks links and not content,
so it is checked here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from xevals.dimensions import DIMENSION_ORDER, NORMALISERS
from xevals.metrics import METRICS, for_dimension

DOCS = Path(__file__).resolve().parents[1] / "docs" / "concepts" / "dimensions"

#: Metrics recorded under a bracketed per-family name are variants of a metric
#: that is documented, not metrics of their own.
DOCUMENTED = {name for name in METRICS.available() if "[" not in name}


def page(dimension) -> str:
    return (DOCS / f"{dimension.value}.md").read_text()


@pytest.mark.parametrize("dimension", DIMENSION_ORDER, ids=lambda d: d.value)
def test_every_dimension_has_a_page(dimension):
    assert (DOCS / f"{dimension.value}.md").exists()
    assert dimension.question in page(dimension), "the page states the question it answers"


@pytest.mark.parametrize("dimension", DIMENSION_ORDER, ids=lambda d: d.value)
def test_a_page_documents_exactly_its_own_metrics(dimension):
    text = page(dimension)
    headings = set(re.findall(r"^## `([\w/]+)`$", text, flags=re.M))
    assert headings == set(for_dimension(dimension)), (
        f"{dimension.value}.md documents {sorted(headings)}, "
        f"registry has {sorted(for_dimension(dimension))}"
    )


@pytest.mark.parametrize("name", sorted(DOCUMENTED))
def test_every_metric_is_documented_once(name):
    owner = METRICS.entry(name).meta["dimension"]
    found = [d.value for d in DIMENSION_ORDER if f"## `{name}`" in page(d)]
    assert found == [owner], f"{name} documented in {found}, belongs to {owner}"


@pytest.mark.parametrize("name", sorted(DOCUMENTED))
def test_every_metric_section_states_a_definition_target_and_example(name):
    dimension = METRICS.entry(name).meta["dimension"]
    text = page(next(d for d in DIMENSION_ORDER if d.value == dimension))
    section = text.split(f"## `{name}`", 1)[1].split("\n## ", 1)[0]
    for required in ("**Definition.**", "**Target.**", "**Example.**"):
        assert required in section, f"{name} has no {required}"
    assert "$$" in section or "$" in section, f"{name} has no equation"


def _spellings(value: float) -> set[str]:
    """How a reference may legitimately be written in prose.

    Prose is not `repr`. A reference of 1e9 reads as $10^9$ on the page and as
    `1e+09` from the code, and a test that only accepted the latter would be
    demanding worse writing rather than catching drift.
    """
    out = {f"{value:g}", f"{value:.2f}".rstrip("0").rstrip("."), f"{int(value)}"}
    exponent = round(__import__("math").log10(value)) if value > 0 else 0
    if abs(value - 10.0**exponent) < 1e-9 and abs(exponent) >= 3:
        out |= {f"10^{exponent}", f"10^{{{exponent}}}"}
    return out


def test_the_overview_links_to_every_page():
    index = (DOCS / "index.md").read_text()
    for dimension in DIMENSION_ORDER:
        assert f"({dimension.value}.md)" in index, dimension.value


def test_a_scale_normalisers_reference_appears_on_its_page():
    # The references are the library's only magic numbers. A page that states a
    # different one from the code is the drift that matters most here.
    for name, normaliser in NORMALISERS.items():
        if normaliser.kind != "scale" or "[" in name:
            continue
        dimension = METRICS.entry(name).meta["dimension"]
        text = page(next(d for d in DIMENSION_ORDER if d.value == dimension))
        section = text.split(f"## `{name}`", 1)[1].split("\n## ", 1)[0]
        assert any(s in section for s in _spellings(normaliser.reference)), (
            f"{name}: page does not state the normaliser reference "
            f"{normaliser.reference:g}"
        )


# --------------------------------------------------------------------------
# The example artefacts the docs embed
# --------------------------------------------------------------------------

EXAMPLE = Path(__file__).resolve().parents[1] / "docs" / "assets" / "example"


def test_the_example_output_is_committed():
    # The page claims these are real files rather than screenshots. If they are
    # missing the page renders empty iframes and says something untrue.
    report_pages = (
        "report.html",
        "conditions.html",
        "episodes.html",
        "tradeoffs.html",
        "metrics.html",
        "provenance.html",
        "report.css",
    )
    keep_video = ("clean", "occlusion@0.75", "patch@1")

    for name in (*report_pages, "radar.png", "leaderboard.md", "dimensions.md",
                 "cells.md"):
        assert (EXAMPLE / name).exists(), name
    for name in ("index.html", "conditions.html", "provenance.html", "report.css"):
        assert (EXAMPLE / "benchmark" / name).exists(), name
    # The clips keep the layout the report links them under. Flattening them into
    # a renamed set silently broke every link in the embedded copy once already.
    for cell in keep_video:
        assert list((EXAMPLE / "videos" / cell).glob("*.gif")), cell


def test_the_example_report_pages_link_to_clips_that_exist():
    import re

    for page in EXAMPLE.glob("*.html"):
        for src in re.findall(r'src="(videos/[^"]+)"', page.read_text()):
            assert (EXAMPLE / src).exists(), f"{page.name} -> {src}"


def test_the_example_stays_inside_its_budget():
    total = sum(p.stat().st_size for p in EXAMPLE.rglob("*") if p.is_file()) // 1024
    budget_kb = 2200
    assert total <= budget_kb, f"{total} kB committed, budget is {budget_kb} kB"


def test_the_example_report_draws_its_charts_inline():
    html = (EXAMPLE / "report.html").read_text()
    assert html.startswith("<!doctype html>")
    assert 'href="report.css"' in html, "the shared stylesheet"
    # Charts are SVG in the markup rather than a loaded image, so they follow the
    # page into dark mode, stay sharp at any zoom, and need no matplotlib.
    assert '<svg class="chart"' in html, "charts are inline SVG, not loaded images"
    assert "data:image/png;base64" not in html, "no figure is embedded as a PNG"
    # The set travels as a directory rather than one file, but still fetches
    # nothing from the network.
    assert "http://" not in html.replace("http://www.w3.org", "")


def test_the_example_page_embeds_what_it_claims_to():
    page = (DOCS.parent.parent / "guides" / "example-output.md").read_text()
    for asset in ("report.html", "episodes.html", "benchmark/index.html", "radar.png",
                  "videos/clean/000.gif", "videos/occlusion@0.75/000.gif",
                  "videos/patch@1/000.gif"):
        assert asset in page, asset
    # The tables are included from the generated files, not pasted, so they
    # cannot drift from the run that produced them.
    for include in ("leaderboard.md", "dimensions.md", "cells.md"):
        assert f'--8<-- "assets/example/{include}"' in page, include


def test_the_example_leaderboard_still_makes_its_point():
    # The page's argument is that the mean is not the answer: a policy that does
    # nothing outranks one that works, because the dimensions an idle policy
    # cannot fail are scored 1.00. If a change to the metrics flattens that, the
    # prose above the table becomes wrong and this catches it.
    table = (EXAMPLE / "leaderboard.md").read_text()
    rows = {
        line.split("|")[1].strip(): [c.strip() for c in line.split("|")[2:-1]]
        for line in table.splitlines()
        if line.startswith("|") and not set(line) <= set("|- ")
    }
    header = rows.pop("model")
    scripted = dict(zip(header, rows["scripted"], strict=True))
    visual = dict(zip(header, rows["visual"], strict=True))
    random = dict(zip(header, rows["random"], strict=True))
    assert float(random["mean"]) > float(visual["mean"]), "the page's second claim"
    assert float(random["security"]) == 1.0, "nothing can degrade a policy that fails anyway"
    # And the first: two working policies that agree on accuracy and agree on
    # nothing else. If a change to the metrics flattens either half of that, the
    # prose above the table becomes wrong.
    assert abs(float(scripted["accuracy"]) - float(visual["accuracy"])) < 0.2
    assert float(scripted["robustness"]) - float(visual["robustness"]) > 0.4
    assert float(scripted["security"]) - float(visual["security"]) > 0.4
