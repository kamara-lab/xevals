"""Structural checks for the committed brand assets.

The generator scripts were intentionally removed. These tests validate the
meaningful invariants directly, without importing deleted maintenance tooling.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KICKER = "ROBOTICS + AI EVALUATIONS"
WORDMARKS = ("assets/logo.svg", "assets/logo-inverse.svg", "docs/assets/logo.svg")
MARKS = (
    "assets/logo-mark.svg",
    "assets/logo-mark-inverse.svg",
    "docs/assets/logo-mark-ink.svg",
    "docs/assets/logo-mark-inverse.svg",
)


def test_every_wordmark_carries_the_kicker():
    for path in WORDMARKS:
        assert KICKER in (ROOT / path).read_text(), path


def test_every_mark_traces_three_bars_of_unequal_height():
    for path in (*WORDMARKS, *MARKS):
        text = (ROOT / path).read_text()
        filled = re.findall(r'<rect x="(14|39|64)" y="(14|39|64)"[^>]+fill="(?!none)', text)
        heights = [sum(1 for x, _ in filled if x == str(col)) for col in (14, 39, 64)]
        assert heights == [3, 1, 2], path


def test_the_diagram_uses_the_library_dimension_palette():
    source = (ROOT / "docs/assets/method.tex").read_text()
    colours = ("440154", "46327E", "365C8D", "277F8E", "1FA187", "4AC16D", "A0DA39")
    for colour in colours:
        assert f"{{HTML}}{{{colour}}}" in source


def test_the_favicon_is_the_mark_on_a_tile():
    favicon = (ROOT / "docs/assets/favicon.svg").read_text()
    assert 'rx="20"' in favicon, "the tile"
    assert favicon.count("<rect") == 10, "one tile plus nine cells"
