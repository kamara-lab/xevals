"""Build the method diagram from ``docs/assets/method.tex``, both themes.

The drawing is TikZ, as xwm's is, so that the mathematics is set by LaTeX rather
than faked in SVG text -- an equation drawn with ``<text>`` elements and a hand
positioned fraction bar is legible at exactly one size and wrong at every other.

This script exists for two reasons beyond convenience. It **checks the dimension
palette in the LaTeX against** :mod:`xevals.dimensions`, so the diagram cannot
outlive a palette change; and it runs both themes and both conversions in one
command, so nobody ships a light SVG that has moved on from its dark twin.

The SVGs are committed, so neither the docs build nor CI needs LaTeX. Only
someone editing the diagram does.

Run from the repository root::

    python -m tools.make_diagram
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ASSETS = Path("docs/assets")
SOURCE = ASSETS / "method.tex"
THEMES = ("light", "dark")

#: Intermediates pdflatex leaves beside the output. None of them are sources.
LEAVINGS = (".aux", ".log", ".out", ".pdf", ".fls", ".fdb_latexmk", ".synctex.gz")


def check_palette() -> None:
    """The seven ``dim*`` colours in the LaTeX must be the library's own.

    Raises:
        AssertionError: naming the theme that has drifted. A diagram whose bars
            are a different palette from every figure beside it is worse than no
            diagram: it teaches the reader a colour code that is then wrong.
    """
    from xevals.dimensions import DIMENSION_ORDER
    from xevals.dimensions import color as dimension_color

    text = SOURCE.read_text()
    for theme, dark in (("light", False), ("dark", True)):
        block = text.split(r"\ifx\xevalstheme\xevalslight")[1]
        block = block.split(r"\else")[0] if theme == "light" else block.split(r"\else")[1]
        found = [c.lower() for c in re.findall(r"\\definecolor\{dim[A-G]\}\{HTML\}\{(\w{6})\}",
                                               block)]
        expected = [dimension_color(d, dark=dark).lstrip("#").lower() for d in DIMENSION_ORDER]
        assert found == expected, (
            f"the {theme} dimension palette in {SOURCE} has drifted from "
            f"xevals.dimensions:\n  in tex: {found}\n  expected: {expected}"
        )


def build(theme: str) -> Path:
    """Compile one theme and convert it to SVG. Returns the SVG's path."""
    job = f"method-{theme}"
    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-jobname={job}",
            rf"\def\xevalstheme{{{theme}}}\input{{method.tex}}",
        ],
        cwd=ASSETS,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["pdftocairo", "-svg", f"{job}.pdf", f"{job}.svg"], cwd=ASSETS, check=True
    )
    return ASSETS / f"{job}.svg"


def clean() -> None:
    """Remove pdflatex's intermediates. ``exclude_docs`` hides them from the site."""
    for theme in THEMES:
        for suffix in LEAVINGS:
            (ASSETS / f"method-{theme}{suffix}").unlink(missing_ok=True)


def main() -> int:
    for tool in ("pdflatex", "pdftocairo"):
        if shutil.which(tool) is None:
            print(f"{tool} is not installed; the committed SVGs are already current")
            return 0
    check_palette()
    for theme in THEMES:
        print(f"wrote {build(theme)}")
    clean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
