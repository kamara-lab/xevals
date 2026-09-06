"""Generate the xevals logo family from one description of the mark.

The geometry is kamara's, shared with xwm: a 96x96 viewBox holding a 3x3 grid of
18-unit cells with a 4.5-unit radius, at x/y in {14, 39, 64}. Only the *pattern*
of filled cells differs between the two libraries, and it is the pattern that
carries the meaning, so it lives here as one set of coordinates rather than being
duplicated across six hand-edited SVGs.

xevals's mark is a **bar chart of three unequal bars** -- three columns rising to
three, one and two cells. Filled cells are what a model scored; the outlined cells
above the short bars are the headroom it did not reach. It is the library's thesis
as a mark: not one number, several, and they disagree.

    #  .  .        column 0: 3    <- a dimension it is strong on
    #  .  #        column 1: 1    <- one it is not
    #  #  #        column 2: 2

Run from the repository root::

    python -m tools.make_logos

To re-check the lockup after changing the kicker (needs librsvg)::

    python -m tools.make_logos --measure
"""

from __future__ import annotations

from pathlib import Path

INK = "#121412"
PAPER = "#FAFAF8"

#: (row, col) of the filled cells, row 0 at the top. Bar heights 3, 1, 2.
FILLED = {(0, 0), (1, 0), (2, 0), (1, 2), (2, 1), (2, 2)}

#: Cell origins along each axis. 18-unit cells on a 25-unit pitch, inset 14.
POS = (14, 39, 64)

DESC = (
    "Three bars of unequal height: filled cells are what a model scored on each "
    "dimension, outlined cells the headroom it did not reach."
)

#: The kicker under the wordmark. Twenty-five characters against "xevals"' six,
#: so it cannot be set at xwm's 13px/2.6 tracking -- at that size it would run
#: half again past the mark. The size and tracking below put it at about 1.2x the
#: wordmark's width, which is the proportion xwm's own kicker sits at, measured
#: off its banner. Numbers checked by rendering, not estimated: see the module
#: docstring for the command.
KICKER = "ROBOTICS + AI EVALUATIONS"
KICKER_SIZE = 9.0
KICKER_TRACKING = 1.3

#: Wide enough for the kicker plus a right margin matching the mark's 14-unit
#: inset. The wordmark ends well before this; the kicker is what sets the width.
VIEWBOX_WIDTH = 306

FONT = (
    "'Hanken Grotesk', ui-sans-serif, system-ui, -apple-system, "
    "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
)


def cells(colour: str, outline_opacity: float, indent: str = "    ") -> str:
    """The nine rects, filled or outlined, in reading order."""
    out = []
    for row in range(3):
        for col in range(3):
            x, y = POS[col], POS[row]
            common = f'x="{x}" y="{y}" width="18" height="18" rx="4.5"'
            if (row, col) in FILLED:
                out.append(f'{indent}<rect {common} fill="{colour}"/>')
            else:
                out.append(
                    f'{indent}<rect {common} fill="none" stroke="{colour}" '
                    f'stroke-width="2.5" opacity="{outline_opacity}"/>'
                )
    return "\n".join(out)


def wordmark(colour: str, outline_opacity: float) -> str:
    """Mark plus wordmark plus kicker, on a transparent ground.

    The viewBox is wider than xwm's 262 for two reasons: "xevals" is three glyphs
    longer than "xwm" at the same 52px size, and the kicker is more than twice as
    long as "WORLD MODELS". The kicker, not the wordmark, is what sets the width
    -- see :data:`KICKER_SIZE`.
    """
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEWBOX_WIDTH} 96"
     width="{VIEWBOX_WIDTH}" height="96" role="img" aria-label="xevals">
  <title>xevals</title>
  <desc>{DESC}</desc>
  <g>
{cells(colour, outline_opacity)}
  </g>
  <text x="118" y="63" font-family="{FONT}" font-size="52" font-weight="600"
        letter-spacing="-1.5" fill="{colour}">xevals</text>
  <text x="119" y="84" font-family="{FONT}" font-size="{KICKER_SIZE:g}" font-weight="500"
        letter-spacing="{KICKER_TRACKING:g}" fill="{colour}" opacity="0.75">{KICKER}</text>
</svg>
"""


def mark(colour: str, outline_opacity: float, tile: str | None = None) -> str:
    """The mark alone, optionally on a rounded tile."""
    background = f'\n  <rect width="96" height="96" rx="20" fill="{tile}"/>' if tile else ""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96"
     width="96" height="96" role="img" aria-label="xevals">
  <title>xevals</title>
  <desc>{DESC}</desc>{background}
  <g>
{cells(colour, outline_opacity)}
  </g>
</svg>
"""


#: path -> the SVG it holds. The inverse variants raise the outline opacity,
#: because a light stroke on a dark ground reads fainter than the reverse at the
#: same value -- 0.3 becomes 0.45 on the wordmark and 0.38 on the tile, as xwm's do.
FILES = {
    "assets/logo.svg": lambda: wordmark(INK, 0.3),
    "assets/logo-inverse.svg": lambda: wordmark(PAPER, 0.45),
    "assets/logo-mark.svg": lambda: mark(INK, 0.3, tile=PAPER),
    "assets/logo-mark-inverse.svg": lambda: mark(PAPER, 0.38, tile=INK),
    "docs/assets/logo.svg": lambda: wordmark(INK, 0.3),
    "docs/assets/logo-mark-ink.svg": lambda: mark(INK, 0.3),
    "docs/assets/logo-mark-inverse.svg": lambda: mark(PAPER, 0.38, tile=INK),
    "docs/assets/favicon.svg": lambda: mark(INK, 0.3, tile=PAPER),
}


def measure() -> int:
    """Render the wordmark and report the ink extents of its three parts.

    The kicker's size and tracking are a typographic judgement about a *rendered*
    result, and the font that renders it is not necessarily the one installed
    here, so the numbers are checked rather than computed. Prints the mark, the
    wordmark and the kicker, and complains if the kicker overruns the viewBox.
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which("rsvg-convert") is None:
        print("rsvg-convert is not installed; skipping the measurement")
        return 0
    import numpy as np
    from PIL import Image

    scale = 4
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "logo.png"
        subprocess.run(
            ["rsvg-convert", "-w", str(VIEWBOX_WIDTH * scale), "-b", "#FFFFFF",
             "assets/logo.svg", "-o", str(png)],
            check=True,
        )
        ink = np.asarray(Image.open(png).convert("L")) < 200

    def extent(y0: float, y1: float, x0: float = 0.0) -> tuple[float, float]:
        band = ink[int(y0 * scale) : int(y1 * scale), int(x0 * scale) :]
        cols = np.nonzero(band.any(axis=0))[0]
        return (x0 + cols[0] / scale, x0 + cols[-1] / scale) if len(cols) else (0.0, 0.0)

    for label, (y0, y1, x0) in {
        "mark": (10, 90, 0),
        "wordmark": (24, 68, 100),
        "kicker": (70, 92, 100),
    }.items():
        lo, hi = extent(y0, y1, x0)
        print(f"  {label:9s} x {lo:6.1f} .. {hi:6.1f}   width {hi - lo:6.1f}")

    word = extent(24, 68, 100)
    kick = extent(70, 92, 100)
    print(f"  kicker is {(kick[1] - kick[0]) / (word[1] - word[0]):.2f}x the wordmark")
    margin = VIEWBOX_WIDTH - max(word[1], kick[1])
    print(f"  right margin {margin:.1f} (the mark's left inset is 14)")
    if margin < 10:
        print("  ! the lockup overruns its viewBox; widen VIEWBOX_WIDTH")
        return 1
    return 0


def main() -> int:
    import sys

    for path, build in FILES.items():
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(build())
        print(f"wrote {path}")
    return measure() if "--measure" in sys.argv else 0


if __name__ == "__main__":
    raise SystemExit(main())
