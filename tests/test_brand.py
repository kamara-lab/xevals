"""The committed brand assets must match the tools that generate them.

Every SVG in ``assets/`` and ``docs/assets/`` is generated, and generated files
that are committed are exactly the files someone eventually hand-edits. These
tests are cheap and they catch the drift that matters: a mark whose cells no
longer trace the bars, a kicker that was changed in one of six files, a diagram
whose palette has moved on from the library's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import make_banner, make_logos  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted(make_logos.FILES))
def test_every_committed_svg_matches_the_generator(path):
    # Regenerating and diffing, rather than checking for a substring: the whole
    # file is the artefact, and a hand edit anywhere in it is what this catches.
    assert (ROOT / path).read_text() == make_logos.FILES[path]()


def test_the_wordmark_carries_the_kicker():
    for path in ("assets/logo.svg", "assets/logo-inverse.svg", "docs/assets/logo.svg"):
        assert make_logos.KICKER in (ROOT / path).read_text(), path


def test_the_mark_traces_three_bars_of_unequal_height():
    heights = [sum(1 for row in range(3) if (row, col) in make_logos.FILLED) for col in range(3)]
    assert heights == [3, 1, 2]
    # Unequal on purpose: three equal bars would be a grid, and the mark is
    # supposed to say that a model's dimensions disagree.
    assert len(set(heights)) == 3


def test_the_banner_draws_the_same_mark_as_the_svgs():
    # The banner cannot import the SVGs, so it repeats the cell set -- and this
    # is what stops the two copies parting company.
    assert make_banner.FILLED == make_logos.FILLED
    assert make_banner.POS == make_logos.POS
    assert make_banner.KICKER == make_logos.KICKER


def test_the_diagram_palette_is_the_librarys_own():
    from tools import make_diagram

    # Raises with the offending theme named if the LaTeX has drifted.
    make_diagram.check_palette()


def test_the_favicon_is_the_mark_on_a_tile():
    favicon = (ROOT / "docs/assets/favicon.svg").read_text()
    assert 'rx="20"' in favicon, "the tile"
    assert favicon.count("<rect") == 10, "one tile plus nine cells"
