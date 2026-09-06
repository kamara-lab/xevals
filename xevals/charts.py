"""Charts drawn as inline SVG, for the HTML report.

The report used to embed matplotlib PNGs as base64. That works and it costs three
things worth having:

**A PNG cannot follow the page's theme.** A figure baked on paper sits on a dark
report as a lit rectangle. These charts take their colours from the same CSS
custom properties the rest of the page uses, so light and dark are one drawing
read from either end of the ramp, exactly as the diagram and the stylesheet are.

**A PNG is one size.** An SVG with a ``viewBox`` fills its column at any width and
stays sharp at any zoom, which matters for a matrix of forty rows that a reader
will zoom into.

**A PNG needs matplotlib.** The core of this library depends on numpy and the
standard library, and until now the report's figures quietly did not. These
charts have the same dependencies as everything else, so a bare install produces
a report with charts in it rather than a report with gaps.

The matplotlib figures are still written to ``figures/`` by
:mod:`xevals.plots`, because a paper wants a 200 dpi PNG and an SVG in a
stylesheet is not that. The split is deliberate: **plots are for publishing,
charts are for reading.**

Everything here is static. No script tag, no chart library, no fetch: the report
is a file you can email, and a chart that needs the network is not a chart in a
file you can email. Hover text uses SVG's own ``<title>``, which every browser
renders and no JavaScript is needed for.
"""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "bars",
    "budget",
    "ladders",
    "matrix",
    "radar",
    "scatter",
    "viridis",
]

#: Viridis, sampled at eleven stops. Copied rather than imported, because this
#: module must work without matplotlib and the ramp is eleven numbers.
_VIRIDIS = (
    "#440154", "#482878", "#3e4a89", "#31688e", "#26828e", "#1f9e89",
    "#35b779", "#6ece58", "#b5de2b", "#dfe318", "#fde725",
)


def viridis(value: float) -> str:
    """A hex colour from the viridis ramp, for ``value`` in ``[0, 1]``.

    Linear interpolation between the stops. Close enough for a matrix cell, and
    it keeps the module free of a plotting dependency.
    """
    x = min(1.0, max(0.0, float(value))) * (len(_VIRIDIS) - 1)
    low = int(math.floor(x))
    high = min(low + 1, len(_VIRIDIS) - 1)
    t = x - low
    a, b = _VIRIDIS[low].lstrip("#"), _VIRIDIS[high].lstrip("#")
    mixed = (
        round(int(a[i : i + 2], 16) * (1 - t) + int(b[i : i + 2], 16) * t)
        for i in (0, 2, 4)
    )
    return "#" + "".join(f"{c:02x}" for c in mixed)


def _e(value: Any) -> str:
    return html.escape(str(value))


def _svg(width: float, height: float, body: str, *, label: str) -> str:
    """One responsive chart. Fixed coordinate space, fluid on the page."""
    return (
        f'<svg class="chart" viewBox="0 0 {width:g} {height:g}" role="img" '
        f'aria-label="{_e(label)}" preserveAspectRatio="xMidYMid meet">{body}</svg>'
    )


def _text(
    x: float, y: float, content: str, *, size: float = 10, fill: str = "var(--ink)",
    anchor: str = "start", weight: int | None = None, opacity: float = 1.0,
) -> str:
    bits = [f'x="{x:g}"', f'y="{y:g}"', f'font-size="{size:g}"', f'fill="{fill}"']
    if anchor != "start":
        bits.append(f'text-anchor="{anchor}"')
    if weight:
        bits.append(f'font-weight="{weight}"')
    if opacity != 1.0:
        bits.append(f'opacity="{opacity:g}"')
    return f"<text {' '.join(bits)}>{_e(content)}</text>"


def _title(content: str) -> str:
    """An SVG tooltip. Free interactivity, and no script to deliver it."""
    return f"<title>{_e(content)}</title>"


# --------------------------------------------------------------------------
# Bars
# --------------------------------------------------------------------------


def bars(
    items: Sequence[tuple[str, float | None, str]],
    *,
    label: str = "Scores",
    width: float = 640,
    row: float = 30,
    gutter: float = 118,
) -> str:
    """Horizontal bars: label, bar, value. One row per item.

    The honest companion to the radar. A polygon's area is easy to over-read and
    a bar is not, so where the two disagree about which model looks better, the
    bars are right.
    """
    height = row * len(items) + 26
    track = width - gutter - 54
    body = [f'<rect width="{width}" height="{height}" fill="none"/>']

    # Gridlines behind everything, at the quarters, labelled once along the foot.
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = gutter + track * tick
        body.append(
            f'<line x1="{x:g}" y1="6" x2="{x:g}" y2="{row * len(items) + 4:g}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
        )
        body.append(
            _text(x, height - 6, f"{tick:g}", size=8.5, fill="var(--faint)",
                  anchor="middle")
        )

    for index, (name, value, colour) in enumerate(items):
        y = 6 + index * row
        mid = y + row / 2 - 4
        body.append(_text(gutter - 10, mid + 4, name, size=10.5, anchor="end",
                          fill="var(--muted)"))
        if value is None:
            body.append(
                f'<rect x="{gutter:g}" y="{y + 5:g}" width="{track:g}" height="{row - 14:g}"'
                f' rx="3" fill="var(--grid)" opacity="0.5"/>'
            )
            body.append(_text(gutter + 8, mid + 4, "not measured", size=9,
                              fill="var(--faint)"))
            continue
        length = max(2.0, track * float(value))
        body.append(
            f'<rect x="{gutter:g}" y="{y + 5:g}" width="{length:g}" height="{row - 14:g}" '
            f'rx="3" fill="{colour}">{_title(f"{name}: {value:.3f}")}</rect>'
        )
        body.append(_text(gutter + length + 8, mid + 4, f"{value:.2f}", size=10,
                          weight=600))
    return _svg(width, height, "".join(body), label=label)


# --------------------------------------------------------------------------
# Radar
# --------------------------------------------------------------------------


def radar(
    scores: Mapping[str, float | None],
    colours: Mapping[str, str],
    *,
    label: str = "Dimension scores",
    width: float = 560,
    height: float = 470,
) -> str:
    """The dimension radar, one axis per key, in the order given.

    Rings are labelled once on the bisector between the first two spokes, which
    is the only radius carrying neither a vertex nor a spoke label. Each vertex
    takes its dimension's own colour, and an axis nothing measured gets a hollow
    marker at the centre and a greyed label rather than a point at zero: drawing
    "unmeasured" and "measured as zero" the same way would be the most misleading
    thing this file could do.
    """
    names = list(scores)
    n = len(names)
    # Wider than tall, and the labels sit close: the longest of them
    # ("generalization") is horizontal, so the room it needs is horizontal too.
    cx, cy = width / 2, height / 2 + 4
    radius = min(width, height) * 0.30
    step = 2 * math.pi / n

    def point(index: int, value: float) -> tuple[float, float]:
        angle = -math.pi / 2 + index * step
        return cx + radius * value * math.cos(angle), cy + radius * value * math.sin(angle)

    body = []
    for ring in (0.25, 0.5, 0.75, 1.0):
        corners = " ".join(f"{x:g},{y:g}" for x, y in (point(i, ring) for i in range(n)))
        body.append(
            f'<polygon points="{corners}" fill="none" stroke="var(--grid)" '
            f'stroke-width="1"/>'
        )
    for index in range(n):
        x, y = point(index, 1.0)
        body.append(
            f'<line x1="{cx:g}" y1="{cy:g}" x2="{x:g}" y2="{y:g}" stroke="var(--grid)" '
            f'stroke-width="1"/>'
        )

    # Ring labels on the bisector, where nothing else is drawn.
    for ring in (0.25, 0.5, 0.75, 1.0):
        angle = -math.pi / 2 + step / 2
        x = cx + radius * ring * math.cos(angle)
        y = cy + radius * ring * math.sin(angle)
        body.append(_text(x, y - 3, f"{ring:g}", size=8, fill="var(--faint)",
                          anchor="middle"))

    filled = [(scores[name] or 0.0) for name in names]
    corners = " ".join(f"{x:g},{y:g}" for x, y in (point(i, v) for i, v in enumerate(filled)))
    body.append(
        f'<polygon points="{corners}" fill="var(--accent)" fill-opacity="0.13" '
        f'stroke="var(--accent)" stroke-width="2" stroke-linejoin="round"/>'
    )

    for index, name in enumerate(names):
        value = scores[name]
        angle = -math.pi / 2 + index * step
        lx = cx + (radius + 26) * math.cos(angle)
        ly = cy + (radius + 26) * math.sin(angle)
        anchor = "middle"
        if math.cos(angle) > 0.35:
            anchor = "start"
        elif math.cos(angle) < -0.35:
            anchor = "end"
        if value is None:
            body.append(
                f'<circle cx="{cx:g}" cy="{cy:g}" r="4.5" fill="var(--paper)" '
                f'stroke="var(--rule)" stroke-width="1.6">'
                f"{_title(f'{name}: not measured')}</circle>"
            )
            body.append(_text(lx, ly + 4, name, size=10.5, fill="var(--faint)",
                              anchor=anchor))
            body.append(_text(lx, ly + 16, "not measured", size=8,
                              fill="var(--faint)", anchor=anchor))
            continue
        x, y = point(index, value)
        body.append(
            f'<circle cx="{x:g}" cy="{y:g}" r="4.6" fill="{colours.get(name, "var(--accent)")}" '
            f'stroke="var(--paper)" stroke-width="1.4">'
            f"{_title(f'{name}: {value:.3f}')}</circle>"
        )
        body.append(_text(lx, ly + 4, name, size=10.5, anchor=anchor, weight=500))
        # The value sits along the spoke, *inside* the polygon for a high score
        # and outside for a low one. Always outward would put a 1.00 on top of
        # its own axis label; always inward would push a 0.11 through the centre.
        offset = -16 if value >= 0.35 else 16
        vx = cx + (radius * value + offset) * math.cos(angle)
        vy = cy + (radius * value + offset) * math.sin(angle)
        body.append(_text(vx, vy + 3.5, f"{value:.2f}", size=9.5, anchor="middle",
                          weight=600))
    return _svg(width, height, "".join(body), label=label)


# --------------------------------------------------------------------------
# Severity ladders
# --------------------------------------------------------------------------


def ladders(
    curves: Mapping[str, Mapping[float, float]],
    *,
    baseline: float | None = None,
    label: str = "Success against severity",
    width: float = 660,
    columns: int = 3,
) -> str:
    """Small multiples: one panel per perturbation family.

    Small multiples rather than one axis with seven lines, because past five
    categorical series no palette separates them and a legend does not rescue a
    chart the eye cannot decode. The dashed rule across every panel is the model's
    *own* clean success rate, so a curve is read against the model rather than
    against an assumed 1.00.
    """
    names = sorted(curves)
    if not names:
        return ""
    columns = min(columns, len(names))
    rows = math.ceil(len(names) / columns)
    pad_x, pad_y = 34, 30
    panel_w = (width - 8) / columns
    panel_h = 108
    height = rows * panel_h + 16

    body = []
    for index, name in enumerate(names):
        ox = (index % columns) * panel_w + 4
        oy = (index // columns) * panel_h + 4
        plot_w = panel_w - pad_x - 12
        plot_h = panel_h - pad_y - 22

        body.append(_text(ox + pad_x, oy + 12, name, size=9, weight=500))
        for tick in (0.0, 0.5, 1.0):
            y = oy + pad_y + plot_h * (1 - tick)
            body.append(
                f'<line x1="{ox + pad_x:g}" y1="{y:g}" x2="{ox + pad_x + plot_w:g}" '
                f'y2="{y:g}" stroke="var(--grid)" stroke-width="1"/>'
            )
            body.append(_text(ox + pad_x - 6, y + 3, f"{tick:g}", size=7.5,
                              fill="var(--faint)", anchor="end"))
        if baseline is not None:
            y = oy + pad_y + plot_h * (1 - baseline)
            body.append(
                f'<line x1="{ox + pad_x:g}" y1="{y:g}" x2="{ox + pad_x + plot_w:g}" '
                f'y2="{y:g}" stroke="var(--rule)" stroke-width="1" '
                f'stroke-dasharray="4 3"/>'
            )

        points = sorted(curves[name].items())
        coords = [
            (ox + pad_x + plot_w * severity, oy + pad_y + plot_h * (1 - value))
            for severity, value in points
        ]
        area = (
            f"M {coords[0][0]:g},{oy + pad_y + plot_h:g} "
            + " ".join(f"L {x:g},{y:g}" for x, y in coords)
            + f" L {coords[-1][0]:g},{oy + pad_y + plot_h:g} Z"
        )
        body.append(f'<path d="{area}" fill="var(--accent)" fill-opacity="0.09"/>')
        line = " ".join(f"{x:g},{y:g}" for x, y in coords)
        body.append(
            f'<polyline points="{line}" fill="none" stroke="var(--accent)" '
            f'stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for (severity, value), (x, y) in zip(points, coords, strict=True):
            body.append(
                f'<circle cx="{x:g}" cy="{y:g}" r="2.6" fill="var(--accent)">'
                f"{_title(f'severity {severity:g}: {value:.2f}')}</circle>"
            )
        last_severity, last_value = points[-1]
        body.append(_text(coords[-1][0] + 4, coords[-1][1] - 4, f"{last_value:.2f}",
                          size=8.5, weight=600))
        body.append(_text(ox + pad_x, oy + pad_y + plot_h + 14, "severity", size=7.5,
                          fill="var(--faint)"))
    return _svg(width, height, "".join(body), label=label)


# --------------------------------------------------------------------------
# The condition matrix
# --------------------------------------------------------------------------


def matrix(
    rows: Sequence[str],
    columns: Sequence[str],
    scores: Sequence[Sequence[float | None]],
    raw: Sequence[Sequence[float | None]],
    *,
    label: str = "Every condition, every metric",
    width: float = 660,
) -> str:
    """Cells against the metrics that scored them.

    **Colour is the score, the number is the measurement.** Every column is
    normalised through the same normaliser the dimension scores use, so one
    sequential ramp can run across columns that point in opposite directions: a
    violation rate of 0.78 and a success rate of 0.22 are both dark, because both
    are bad. Colouring the raw values would put "78 % of episodes left the
    workspace" at the bright end.

    A cell with nothing measured takes the page colour and shows a dash, so a
    blank can never be read as a low score.
    """
    gutter = 172
    cell_w = (width - gutter - 46) / max(1, len(columns))
    cell_h = 20
    top = 34
    height = top + cell_h * len(rows) + 26

    body = []
    for index, column in enumerate(columns):
        x = gutter + cell_w * (index + 0.5)
        body.append(_text(x, top - 9, column, size=8.8, fill="var(--muted)",
                          anchor="middle"))

    for r, name in enumerate(rows):
        y = top + cell_h * r
        body.append(_text(gutter - 9, y + cell_h / 2 + 3.4, name, size=8.6,
                          fill="var(--muted)", anchor="end"))
        for c in range(len(columns)):
            x = gutter + cell_w * c
            score, value = scores[r][c], raw[r][c]
            if score is None or value is None:
                body.append(
                    f'<rect x="{x + 1:g}" y="{y + 1:g}" width="{cell_w - 2:g}" '
                    f'height="{cell_h - 2:g}" rx="2" fill="var(--card)" '
                    f'stroke="var(--grid)" stroke-width="1"/>'
                )
                body.append(_text(x + cell_w / 2, y + cell_h / 2 + 3, "--", size=8,
                                  fill="var(--faint)", anchor="middle"))
                continue
            fill = viridis(score)
            body.append(
                f'<rect x="{x + 1:g}" y="{y + 1:g}" width="{cell_w - 2:g}" '
                f'height="{cell_h - 2:g}" rx="2" fill="{fill}">'
                f"{_title(f'{name} · {columns[c]}: {value:.3f} (score {score:.2f})')}"
                f"</rect>"
            )
            body.append(_text(
                x + cell_w / 2, y + cell_h / 2 + 3, f"{value:.2f}", size=8.2,
                fill="#FAFAF8" if score < 0.55 else "#121412", anchor="middle",
            ))

    # The ramp, as a legend, built from the same function the cells used.
    legend_w, legend_x = 120, width - 132
    for step in range(60):
        body.append(
            f'<rect x="{legend_x + legend_w * step / 60:g}" y="{height - 18:g}" '
            f'width="{legend_w / 60 + 0.6:g}" height="7" fill="{viridis(step / 59)}"/>'
        )
    body.append(_text(legend_x - 6, height - 11, "score", size=8,
                      fill="var(--faint)", anchor="end"))
    body.append(_text(legend_x, height - 4, "0", size=7.5, fill="var(--faint)"))
    body.append(_text(legend_x + legend_w, height - 4, "1", size=7.5,
                      fill="var(--faint)", anchor="end"))
    return _svg(width, height, "".join(body), label=label)


# --------------------------------------------------------------------------
# Latency against the control budget
# --------------------------------------------------------------------------


def budget(
    p50: float | None,
    p95: float | None,
    budget_ms: float,
    control_hz: float,
    *,
    width: float = 560,
) -> str:
    """Step latency against the period it has to fit inside.

    A latency in milliseconds means nothing without the rate the robot runs at,
    so the budget is a line on the same axis rather than a number in the prose. A
    fast model against a slow loop spans three orders of magnitude, so the axis
    goes logarithmic past a twentyfold span and says so.
    """
    values = [(name, v) for name, v in (("p50", p50), ("p95", p95)) if v is not None]
    if not values:
        return ""
    height = 150
    left, right = 52, width - 16
    span = right - left
    biggest = max(max(v for _n, v in values), budget_ms)
    smallest = min(v for _n, v in values)
    log = smallest > 0 and biggest / smallest > 20

    def position(value: float) -> float:
        if log:
            lo, hi = math.log10(smallest / 3), math.log10(biggest * 1.6)
            return left + span * (math.log10(max(value, smallest / 3)) - lo) / (hi - lo)
        return left + span * value / (biggest * 1.2)

    body = []
    row = 34
    for index, (name, value) in enumerate(values):
        y = 22 + index * row
        body.append(_text(left - 10, y + 14, name, size=10, anchor="end",
                          fill="var(--muted)"))
        length = max(2.0, position(value) - left)
        colour = "var(--accent)" if name == "p50" else "var(--warn)"
        body.append(
            f'<rect x="{left:g}" y="{y:g}" width="{length:g}" height="18" rx="3" '
            f'fill="{colour}">{_title(f"{name}: {value:.4g} ms")}</rect>'
        )
        body.append(_text(left + length + 8, y + 13, f"{value:.3g} ms", size=9.5,
                          weight=600))

    x = position(budget_ms)
    body.append(
        f'<line x1="{x:g}" y1="12" x2="{x:g}" y2="{22 + row * len(values):g}" '
        f'stroke="var(--bad)" stroke-width="1.4" stroke-dasharray="4 3"/>'
    )
    body.append(_text(x, 8, f"{control_hz:g} Hz budget ({budget_ms:.0f} ms)", size=8.5,
                      fill="var(--bad)", anchor="middle"))
    worst = max(v for _n, v in values)
    headroom = 1 - worst / budget_ms
    body.append(_text(
        left, height - 8,
        f"{headroom:.1%} of the control period unused"
        + ("  ·  log scale" if log else ""),
        size=9, fill="var(--muted)",
    ))
    return _svg(width, height, "".join(body), label="Latency against the control budget")


# --------------------------------------------------------------------------
# Trade-off scatter
# --------------------------------------------------------------------------


def scatter(
    points: Sequence[tuple[str, float, float]],
    *,
    frontier: Sequence[str] = (),
    knee: str | None = None,
    x_label: str = "x",
    y_label: str = "y",
    named: bool = False,
    width: float = 480,
) -> str:
    """A trade-off, with the frontier drawn only when there is one.

    The chart obeys the rule the analysis does: when the tension test came back
    undetermined or absent, ``frontier`` is empty and no line is drawn, because a
    hull through uncorrelated points reads as an exchange and there is none.
    """
    if not points:
        return ""
    height = width * 0.86
    left, bottom = 46, height - 40
    top, right = 16, width - 16
    span_x, span_y = right - left, bottom - top

    def place(x: float, y: float) -> tuple[float, float]:
        return left + span_x * x, bottom - span_y * y

    body = []
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx, _ = place(tick, 0)
        _, gy = place(0, tick)
        body.append(f'<line x1="{gx:g}" y1="{top:g}" x2="{gx:g}" y2="{bottom:g}" '
                    f'stroke="var(--grid)" stroke-width="1"/>')
        body.append(f'<line x1="{left:g}" y1="{gy:g}" x2="{right:g}" y2="{gy:g}" '
                    f'stroke="var(--grid)" stroke-width="1"/>')
        body.append(_text(gx, bottom + 14, f"{tick:g}", size=8, fill="var(--faint)",
                          anchor="middle"))
        body.append(_text(left - 7, gy + 3, f"{tick:g}", size=8, fill="var(--faint)",
                          anchor="end"))

    on = set(frontier)
    if on:
        line = sorted((p for p in points if p[0] in on), key=lambda p: p[1])
        path = " ".join(f"{x:g},{y:g}" for x, y in (place(p[1], p[2]) for p in line))
        body.append(
            f'<polyline points="{path}" fill="none" stroke="var(--accent)" '
            f'stroke-width="1.6" stroke-opacity="0.7"/>'
        )
    for name, x, y in points:
        px, py = place(x, y)
        winner = name in on
        body.append(
            f'<circle cx="{px:g}" cy="{py:g}" r="{4.6 if winner else 3.4:g}" '
            f'fill="{"var(--accent)" if winner else "var(--rule)"}" '
            f'stroke="var(--paper)" stroke-width="1">'
            f"{_title(f'{name}: {x:.3f}, {y:.3f}')}</circle>"
        )
        if name == knee:
            body.append(
                f'<circle cx="{px:g}" cy="{py:g}" r="9" fill="none" '
                f'stroke="var(--warn)" stroke-width="1.6"/>'
            )
        if named:
            body.append(_text(px + 7, py - 5, name, size=8.5, fill="var(--muted)"))
    body.append(_text((left + right) / 2, height - 6, x_label, size=9.5,
                      fill="var(--muted)", anchor="middle"))
    body.append(
        f'<text x="12" y="{(top + bottom) / 2:g}" font-size="9.5" fill="var(--muted)" '
        f'text-anchor="middle" transform="rotate(-90 12 {(top + bottom) / 2:g})">'
        f"{_e(y_label)}</text>"
    )
    return _svg(width, height, "".join(body), label=f"{y_label} against {x_label}")
