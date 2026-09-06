"""Render ``assets/banner.webp`` in the kamara house style, using Pillow and numpy.

Same construction as xwm's banner, so the two sit together in a README or on the
organisation page: a deep blue-to-amber field carrying the inverse mark, the
wordmark and the kicker, centred, with nothing else on it. Both halves are the
brand hues at about 55 % value -- :data:`ACCENT` and :data:`AMBER` from
``xevals.plots``, darkened -- rather than sampled numbers, so the banner stays tied
to the palette if the palette ever moves.

Three layers, in order:

**The field.** A horizontal blue-to-amber ramp with a smoothstep transition at
0.68 of the width, plus a gentle vertical fall-off so the centre carries the
logo and the edges recede.

**The glow.** Two soft radial lifts and one long diagonal streak. Without them a
two-colour ramp reads as a PowerPoint background; with them it reads as light on
a surface. They are wide and low-contrast on purpose -- the mark is the subject.

**The grain.** Gaussian noise at sigma 5.6, matched to xwm's. It is what keeps
the ramp from banding when the file is compressed, and it is most of why the
result looks like a photograph of a surface rather than a gradient fill.

No dependency on a headless browser, Inkscape or cairosvg: the mark is drawn from
the same cell coordinates :mod:`tools.make_logos` uses, so the banner cannot drift
from the SVGs. The wordmark uses Hanken Grotesk when the font is installed and
falls back to whatever Pillow can find otherwise; the committed banner was
rendered with the real font.

Run from the repository root::

    python tools/make_banner.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1600, 300

#: The brand accent and its warm counterpart, from ``xevals.plots``, plus the ink
#: they are blended toward and the paper the logo is drawn in.
ACCENT = (54, 127, 201)
AMBER = (201, 127, 54)
INK = (18, 20, 18)
PAPER = (250, 250, 248)

#: How far both hues are blended toward ink for the field.
#:
#: Solved against xwm's banner rather than guessed, and the two halves come back
#: with the *same* constant to three decimal places -- which is the tell that its
#: field is the brand colours mixed toward ink, not scaled in value. A blend keeps
#: the hue; a value scale drifts it, and the drift is what makes a "matching"
#: banner look subtly wrong beside the one it is matching.
INK_MIX = 0.54

#: Where the blue gives way to the amber, as a fraction of the width, how wide
#: the handover is, and how far the boundary tilts from vertical. Off-centre so
#: the mark sits in the blue; tilted so the seam does not read as a straight edge
#: drawn down the image.
TRANSITION, TRANSITION_WIDTH, TRANSITION_TILT = 0.70, 0.34, 0.16

#: Matched to xwm's banner. Also what stops the ramp banding under compression.
GRAIN_SIGMA = 5.6

#: (row, col) of the filled cells -- bar heights 3, 1, 2. Kept identical to
#: :data:`tools.make_logos.FILLED`; the assertion in :func:`main` enforces it.
FILLED = {(0, 0), (1, 0), (2, 0), (1, 2), (2, 1), (2, 2)}
POS = (14, 39, 64)

#: The kicker, and how wide it should sit relative to the wordmark. Pillow has no
#: letter-spacing, so the tracking is *solved* for this ratio in :func:`_tracked`
#: rather than faked with literal spaces -- which is the only way to hold the
#: proportion when the wordmark's width depends on which font is installed.
KICKER = "ROBOTICS + AI EVALUATIONS"
KICKER_RATIO = 1.2


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Hermite ease, so the two halves meet without a visible seam."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _toward_ink(colour: tuple[int, int, int], amount: float) -> np.ndarray:
    """``colour`` blended ``amount`` of the way to ink, keeping its hue."""
    return np.asarray(colour, dtype=float) * (1.0 - amount) + np.asarray(INK, float) * amount


def field() -> np.ndarray:
    """The blue-to-amber ground, before any light is put on it."""
    u = np.linspace(0.0, 1.0, WIDTH)[None, :, None]
    v = np.linspace(0.0, 1.0, HEIGHT)[:, None, None]

    blue = _toward_ink(ACCENT, INK_MIX)
    amber = _toward_ink(AMBER, INK_MIX)
    # The boundary leans, and wanders slightly, so that it reads as two fields
    # meeting rather than as a gradient stop.
    edge = TRANSITION + TRANSITION_TILT * (v - 0.5) + 0.02 * np.sin(6.0 * np.pi * v)
    mix = _smoothstep((u - edge) / TRANSITION_WIDTH + 0.5)
    image = blue * (1.0 - mix) + amber * mix

    # A shallow vertical fall-off: brightest across the middle third, where the
    # logo sits, and receding at the top and bottom edges.
    return image * (0.90 + 0.13 * np.sin(np.pi * v) ** 0.7)


def _blob(cx: float, cy: float, rx: float, ry: float, angle: float = 0.0) -> np.ndarray:
    """A soft elliptical mask centred on ``(cx, cy)`` in fractional coordinates."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    x = (x / WIDTH - cx) / rx
    y = (y / HEIGHT - cy) / ry
    cos, sin = np.cos(angle), np.sin(angle)
    return np.exp(-((cos * x + sin * y) ** 2 + (-sin * x + cos * y) ** 2))[..., None]


def light(image: np.ndarray) -> np.ndarray:
    """Two radial lifts and a long diagonal streak, all wide and low-contrast."""
    image = image + 26.0 * _blob(0.34, 0.30, 0.28, 0.80) * np.asarray([0.55, 0.85, 1.0])
    image = image + 18.0 * _blob(0.90, 0.38, 0.20, 0.90) * np.asarray([1.0, 0.72, 0.40])
    # The streak: a very elongated ellipse rotated a few degrees off horizontal.
    image = image + 16.0 * _blob(0.56, 0.28, 0.60, 0.15, angle=-0.10) * np.asarray(
        [0.9, 0.95, 1.0]
    )
    return image


def grain(image: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Gaussian noise, one draw per channel so the grain is not grey speckle."""
    rng = np.random.default_rng(seed)
    return image + rng.normal(0.0, GRAIN_SIGMA, image.shape)


def _font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    """Hanken Grotesk if it is installed, else whatever Pillow can find."""
    for candidate in (
        f"HankenGrotesk-{weight}.ttf",
        f"/Library/Fonts/HankenGrotesk-{weight}.ttf",
        f"{Path.home()}/Library/Fonts/HankenGrotesk-{weight}.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def draw_mark(draw: ImageDraw.ImageDraw, x: int, y: int, cell: int) -> None:
    """The 3x3 mark in paper, at ``(x, y)``, with ``cell`` pixels per cell.

    Pitch and radius are scaled from the logo's own ratios -- a 25:18 pitch and
    the 1:4 radius that also gives ``--xevals-radius`` its 6 px -- so the banner's
    mark is the SVG's mark at another size rather than a redrawing of it.
    """
    pitch = round(cell * 25 / 18)
    radius = max(2, round(cell / 4))
    stroke = max(2, round(cell * 2.5 / 18))
    for row in range(3):
        for col in range(3):
            left, top = x + col * pitch, y + row * pitch
            box = (left, top, left + cell, top + cell)
            if (row, col) in FILLED:
                draw.rounded_rectangle(box, radius=radius, fill=PAPER)
            else:
                draw.rounded_rectangle(box, radius=radius, outline=(*PAPER, 115), width=stroke)


def _tracked(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    width: float,
    fill: tuple,
) -> None:
    """Draw ``text`` letter by letter, spaced to occupy exactly ``width``.

    Pillow has no tracking, and the usual workaround -- putting literal spaces
    between the characters -- fixes the spacing to one font's space advance and
    to one string length. Solving for the advance instead holds the kicker at a
    stated proportion of the wordmark whichever font is actually installed, which
    is the property the lockup needs and the literal-space trick cannot give.
    """
    natural = draw.textlength(text, font=font)
    tracking = (width - natural) / max(1, len(text) - 1)
    x, y = xy
    for character in text:
        draw.text((x, y), character, font=font, fill=fill)
        x += draw.textlength(character, font=font) + tracking


def build() -> Image.Image:
    """The finished banner: field, light, grain, then the logo on top."""
    image = Image.fromarray(np.clip(grain(light(field())), 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image, "RGBA")

    # The mark and the wordmark are measured as one block and centred together,
    # so the composition does not shift when the wordmark's width changes.
    cell = 30
    pitch = round(cell * 25 / 18)
    mark_width = pitch * 2 + cell
    gap = 44
    word_font, kicker_font = _font(76, "SemiBold"), _font(14)
    word_width = draw.textlength("xevals", font=word_font)
    kicker_width = word_width * KICKER_RATIO

    # The mark and the text are measured as one block and centred together, so
    # the composition does not shift when either changes width.
    total = mark_width + gap + max(word_width, kicker_width)
    left = (WIDTH - total) / 2
    mark_top = (HEIGHT - mark_width) / 2

    draw_mark(draw, int(left), int(mark_top), cell)
    text_x = left + mark_width + gap
    draw.text((text_x, HEIGHT / 2 - 50), "xevals", font=word_font, fill=PAPER)
    _tracked(
        draw,
        (text_x + 3, HEIGHT / 2 + 28),
        KICKER,
        kicker_font,
        width=kicker_width,
        fill=(*PAPER, 200),
    )
    return image


def main() -> int:
    import tools.make_logos as logos  # noqa: PLC0415 - a check, not a dependency

    assert FILLED == logos.FILLED, "the banner's mark has drifted from the SVGs"

    target = Path("assets/banner.webp")
    target.parent.mkdir(parents=True, exist_ok=True)
    build().save(target, "WEBP", quality=92, method=6)
    print(f"wrote {target} ({target.stat().st_size // 1024} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
