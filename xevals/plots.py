"""Figures, in the kamara palette, with each palette's categorical limit enforced.

Ported from xwm's ``plots.style`` and extended with one thing this library needs
and that one does not: **a fixed colour per dimension**. Accuracy is the same
purple in every radar, every bar chart and every docs page xevals has ever
produced, so a reader who has learnt the palette once can read any figure
without the legend.

That is only safe because a dimension never has to be told apart from another
*within* one chart by colour alone -- a radar gives each its own axis, a bar
chart its own bar. Seven categories exceed what viridis separates legibly (the
adjacent-pair ΔE table is in xwm's ``plots/style.py``), so :func:`palette_colors`
still refuses more than :data:`MAX_CATEGORICAL` genuinely categorical series and
tells you to use small multiples.

Chrome -- background, text, gridlines, spines -- comes from the kamara design
tokens, so a figure dropped into the README or the docs sits on the same ground
as the page around it rather than announcing itself as a white rectangle.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .dimensions import DIMENSION_ORDER
from .dimensions import color as dimension_color
from .errors import MissingExtra

__all__ = [
    "ACCENT",
    "AMBER",
    "BRAND",
    "FAINT",
    "GRID",
    "INK",
    "MAX_CATEGORICAL",
    "MUTED",
    "PAPER",
    "RULE",
    "dimension_bars",
    "dimension_color",
    "heatmap",
    "latency",
    "palette_colors",
    "plot_style",
    "radar",
    "DPI",
    "SIZES",
    "save_figure",
    "severity_curves",
    "tradeoff",
    "write_all",
]

#: Background. kamara's warm off-white, not pure white.
PAPER = "#FAFAF8"
#: Text: titles, axis labels, annotations.
INK = "#121412"
#: Rules at full strength -- the tint source for :data:`GRID` and :data:`RULE`.
BLUEPRINT = "#747974"
#: Secondary text: tick labels.
MUTED = "#5C615D"

#: A step below muted, for ring labels and for anything greyed out because
#: it was not measured. Deliberately quiet: it must not be mistaken for data.
FAINT = "#8B908A"
#: Blueprint at 22 % over paper. Gridlines, which sit under the data.
GRID = "#DCDEDB"
#: Blueprint at 35 % over paper. Spines and ticks, a step up from the grid so the
#: frame reads before the ruling does.
RULE = "#CBCDCA"
#: *The* accent. 3.99:1 on paper -- fine for a mark or a rule, not for body text.
ACCENT = "#367FC9"
#: The accent's channels reversed, so the pair sits at one lightness.
AMBER = "#C97F36"

#: Wong's colourblind-safe family, for curves carrying two to four series.
BLUE_ORANGE = ("#0072B2", "#E69F00", "#56B4E9", "#D55E00")
#: The accent pair, for the very common two-way comparison.
BRAND_PAIR = (ACCENT, AMBER)
#: Fraction of the viridis ramp to sample, trimmed where it is near-black/white.
VIRIDIS_SPAN = (0.06, 0.94)
#: Maximum categorical series viridis separates acceptably.
MAX_CATEGORICAL = 5

#: Brand token -> hex, for callers styling a figure by hand.
BRAND = {
    "paper": PAPER,
    "ink": INK,
    "blueprint": BLUEPRINT,
    "muted": MUTED,
    "grid": GRID,
    "rule": RULE,
    "accent": ACCENT,
    "amber": AMBER,
}

#: Axis labels for the radar, where the full names do not fit. Only the two
#: longest are shortened; abbreviating all seven would cost more legibility than
#: it buys.
_SHORT = {"generalization": "general.", "robustness": "robustness"}

MARKERS = ("o", "s", "^", "D", "v")
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))


def _matplotlib():
    """Import matplotlib with the Agg backend, or say which extra provides it."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise MissingExtra("matplotlib", "plots", "to draw figures") from exc
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def viridis_colors(n: int, span: tuple[float, float] = VIRIDIS_SPAN) -> list[str]:
    """``n`` hex colours sampled evenly from viridis, in a fixed order.

    Raises:
        ValueError: past :data:`MAX_CATEGORICAL`, where an adjacent pair falls
            below the normal-vision legibility floor. Use small multiples.
    """
    import matplotlib
    from matplotlib.colors import to_hex

    if n < 1:
        raise ValueError(f"need at least one colour, got {n}")
    if n > MAX_CATEGORICAL:
        raise ValueError(
            f"{n} categorical series exceeds the {MAX_CATEGORICAL} that viridis can "
            "separate legibly. Use small multiples, or group the tail into 'other'."
        )
    cmap = matplotlib.colormaps["viridis"]
    lo, hi = span
    if n == 1:
        return [to_hex(cmap(0.5 * (lo + hi)))]
    step = (hi - lo) / (n - 1)
    return [to_hex(cmap(lo + i * step)) for i in range(n)]


def palette_colors(n: int, palette: str = "viridis") -> list[str]:
    """``n`` colours from a named palette: ``viridis``, ``blue-orange``, ``brand``."""
    limits = {"viridis": MAX_CATEGORICAL, "blue-orange": 4, "brand": 2}
    if palette not in limits:
        raise ValueError(f"unknown palette {palette!r}; choose from {sorted(limits)}")
    if n > limits[palette]:
        raise ValueError(
            f"{n} series exceeds the {limits[palette]} the {palette!r} palette separates "
            "legibly; use small multiples"
        )
    if palette == "blue-orange":
        return list(BLUE_ORANGE[:n])
    if palette == "brand":
        return list(BRAND_PAIR[:n])
    return viridis_colors(n)


#: Figure sizes, in inches. Named rather than repeated, so every figure in a run
#: shares a scale and a set of them can sit on one page without one shouting.
SIZES = {
    "radar": (6.2, 5.9),
    "bars": (6.6, 3.9),
    "panel": (2.9, 2.35),
    "wide": (6.6, 3.2),
    "matrix": (6.8, 4.6),
}

#: Saved at 200 dpi. Enough for a two-column figure at print size, and small
#: enough that a report with six of them base64-inlined stays under a megabyte.
DPI = 200


def rc_params() -> dict[str, Any]:
    """Matplotlib settings for the xevals look.

    Recessive chrome, thin marks, and one type scale. The sizes are absolute
    rather than relative so that a small panel and a wide bar chart in the same
    report have labels of the same physical size, which is the thing that makes a
    set of figures look like a set rather than a collection.
    """
    return {
        # -- type: one scale, used everywhere -------------------------------
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "axes.titleweight": "medium",
        "axes.titlepad": 7,
        "axes.labelpad": 4,
        # -- chrome: the grid sits under the data, the frame reads first ----
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": RULE,
        "ytick.color": RULE,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        # -- data ------------------------------------------------------------
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.7,
        # -- output ----------------------------------------------------------
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }


def _caption_below(fig: Any, text: str, *, y: float = 0.985) -> None:
    """A subtitle under a figure-level title, at an explicit height."""
    if text:
        fig.text(0.5, y, text, ha="center", va="top", fontsize=7.6, color=MUTED)


def _caption(fig: Any, text: str) -> None:
    """A subtitle under the figure title: the conditions the numbers came from.

    A figure without its ``n`` and its conditions is not publication ready, and
    the caption in the surrounding document is not always carried with the image.
    """
    if text:
        fig.text(0.5, 0.985, text, ha="center", va="top", fontsize=7.6, color=MUTED)


@contextmanager
def plot_style():
    """Apply the xevals style for one block, leaving global rcParams untouched."""
    plt = _matplotlib()
    with plt.rc_context(rc_params()):
        yield


def save_figure(fig: Any, path: str | Path, *, dpi: int = DPI) -> Path:
    """Write a figure and close it, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=PAPER)
    fig.clf()
    _matplotlib().close(fig)
    return path


# --------------------------------------------------------------------------
# The figures
# --------------------------------------------------------------------------


def _run_caption(result: Any) -> str:
    """``suite · n episodes x m seeds · gate``, for under a figure title."""
    run = getattr(result, "run", {}) or {}
    parts = [str(getattr(result, "suite", None).name)] if getattr(result, "suite", None) else []
    if run.get("episodes"):
        parts.append(f"{run['episodes']} episodes x {len(run.get('seeds', []))} seeds")
    status = getattr(result, "gate_status", lambda: None)()
    if status:
        parts.append(f"gate {status}")
    return "  ·  ".join(parts)


def radar(result: Any, path: str | Path, *, others: Sequence[Any] = ()) -> Path:
    """The dimension radar: one axis per dimension, in the fixed order.

    Built to be read rather than admired. Rings are labelled once instead of on
    every spoke, each vertex carries its value and its dimension's own colour, and
    an axis nothing measured gets a hollow marker at the centre with a greyed
    label. A radar that drew "unmeasured" and "measured as zero" identically would
    be the most misleading chart this library could produce, so it does not.

    The polygon is a shape, and a shape invites area comparisons that a radar
    cannot support: two models with the same area can differ completely. The
    numbers at the vertices are there so the reader does not have to estimate.
    """
    plt = _matplotlib()
    labels = [d.value for d in DIMENSION_ORDER]
    angles = np.linspace(0, 2 * math.pi, len(labels), endpoint=False).tolist()
    closed = angles + angles[:1]
    runs = [result, *others][:MAX_CATEGORICAL]
    multi = len(runs) > 1

    with plot_style():
        fig, ax = plt.subplots(figsize=SIZES["radar"], subplot_kw={"projection": "polar"})
        # The labels sit outside the circle, so the axes are shrunk inside the
        # figure rather than trusted to bbox_inches: the polar projection draws
        # them past the boundary and tight bounds cannot recover them.
        fig.subplots_adjust(left=0.15, right=0.85, top=0.76, bottom=0.08)
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_facecolor(PAPER)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])
        ax.set_xticks(angles)
        ax.set_xticklabels([])
        ax.grid(color=GRID, linewidth=0.6)
        ax.spines["polar"].set_color(RULE)
        ax.spines["polar"].set_linewidth(0.7)
        ax.set_rlabel_position(0)

        # Rings labelled once, on the bisector between the first two spokes,
        # which is the only radius with neither a vertex nor a spoke label on it.
        bisector = math.pi / len(labels)
        for ring in (0.25, 0.5, 0.75, 1.0):
            ax.text(bisector, ring, f"{ring:g}", ha="center", va="bottom",
                    fontsize=6.6, color=FAINT, zorder=1)

        colours = (
            palette_colors(len(runs), "brand" if len(runs) <= 2 else "viridis")
            if multi
            else [ACCENT]
        )
        for index, run in enumerate(runs):
            scores = run.scores
            values = [scores.get(label) for label in labels]
            filled = [0.0 if v is None else float(v) for v in values]
            colour = colours[index % len(colours)]
            ax.plot(closed, filled + filled[:1], color=colour, linewidth=1.9,
                    zorder=3, label=_run_label(run) if multi else None)
            ax.fill(closed, filled + filled[:1], color=colour, alpha=0.10 if multi else 0.13,
                    zorder=2)
            for angle, value, label in zip(angles, values, labels, strict=True):
                if value is None:
                    ax.plot([angle], [0.045], marker="o", markersize=6.5, color=PAPER,
                            markeredgecolor=RULE, markeredgewidth=1.2, zorder=6)
                    continue
                # A single model gets each vertex in its dimension's own colour,
                # which is the palette the rest of the report uses. Several models
                # need the vertices to say *which model*, so they take the series
                # colour instead.
                ax.plot([angle], [value], marker="o", markersize=5.4,
                        color=dimension_color(label) if not multi else colour,
                        markeredgecolor=PAPER, markeredgewidth=0.9, zorder=5)
                if not multi:
                    ax.annotate(
                        f"{value:.2f}", (angle, value), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7.6, color=INK, zorder=7,
                    )

        # Spoke labels drawn by hand, so an unmeasured axis can be greyed and so
        # the longest of them does not have to be abbreviated.
        for angle, label in zip(angles, labels, strict=True):
            measured = result.scores.get(label) is not None
            radial = 1.19
            ax.text(
                angle, radial, label,
                ha="center", va="center", fontsize=8.6,
                color=INK if measured else FAINT,
                fontweight=500 if measured else 400,
            )
            if not measured:
                ax.text(angle, 1.33, "not measured", ha="center", va="center",
                        fontsize=6.4, color=FAINT, style="italic")

        # Title and caption go on the figure, not the axes: an axes title on a
        # polar plot sits inside the label ring and collides with the top spoke.
        fig.text(0.5, 0.965, "Dimension scores", ha="center", va="top",
                 fontsize=12, fontweight=600, color=INK)
        _caption_below(fig, _run_caption(result), y=0.925)
        if multi:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04),
                      ncols=min(len(runs), 3))
        return save_figure(fig, path)


def dimension_bars(result: Any, path: str | Path) -> Path:
    """One bar per dimension, each in its own fixed colour.

    The radar's companion, and the more honest of the two for reading off a
    value: a polygon's area is easy to over-read, and a bar is not.
    """
    plt = _matplotlib()
    scores = result.scores
    labels = [d.value for d in DIMENSION_ORDER if d.value in scores]
    values = [scores[label] for label in labels]
    with plot_style():
        fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(labels) + 1.2))
        positions = np.arange(len(labels))
        ax.barh(
            positions,
            [0.0 if v is None else v for v in values],
            color=[dimension_color(label) for label in labels],
            height=0.62,
        )
        for y, value in zip(positions, values, strict=True):
            if value is None:
                ax.text(0.02, y, "not measured", va="center", fontsize=8, color=MUTED)
            else:
                ax.text(value + 0.015, y, f"{value:.2f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("score")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        return save_figure(fig, path)


def severity_curves(result: Any, path: str | Path, *, family: str | None = None) -> Path:
    """Success against severity, one small multiple per perturbation family.

    Small multiples rather than one axis with seven lines: past five categorical
    series no palette separates them, and a legend does not rescue a chart the eye
    cannot decode. The panels share a y-axis and show its ticks once, on the left
    column, because seven copies of the same 0.0-1.0 scale is ink spent saying
    nothing.

    Each curve is anchored at severity 0 by the clean cell's own success rate, and
    that anchor is drawn as a rule across every panel so the reader can see how far
    the model has fallen from *its own* baseline rather than from an assumed 1.0.
    """
    plt = _matplotlib()
    curves = _curves_from_result(result)
    if family:
        curves = {k: v for k, v in curves.items() if k == family}
    if not curves:
        raise ValueError("no severity ladder in this run; run the robustness suite")

    names = sorted(curves)
    columns = 3 if len(names) > 4 else min(len(names), 2)
    rows = math.ceil(len(names) / columns)
    # Every family's curve is anchored at severity 0 by the same clean cell, so
    # any of them gives the baseline to rule across the panels.
    anchor = next((c[0.0] for c in curves.values() if 0.0 in c), None)

    with plot_style():
        fig, axes = plt.subplots(
            rows, columns,
            figsize=(SIZES["panel"][0] * columns, SIZES["panel"][1] * rows + 0.5),
            squeeze=False, sharey=True, sharex=True,
        )
        for index, name in enumerate(names):
            ax = axes[index // columns][index % columns]
            points = sorted(curves[name].items())
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            if anchor is not None:
                ax.axhline(anchor, color=RULE, linewidth=0.8, linestyle=(0, (4, 3)),
                           zorder=1)
            ax.plot(xs, ys, color=ACCENT, marker="o", markeredgecolor=PAPER,
                    markeredgewidth=0.7, zorder=3, clip_on=False)
            ax.fill_between(xs, 0, ys, color=ACCENT, alpha=0.08, zorder=2)
            # The endpoint is the number a reader takes away, so it is written
            # rather than estimated off the axis.
            ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(3, 5), fontsize=7.4, color=INK, zorder=4)
            ax.set_title(name, fontsize=8.6, color=INK)
            ax.set_ylim(0, 1.06)
            ax.set_xlim(-0.02, 1.04)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.grid(axis="x", visible=False)
        for index in range(len(names), rows * columns):
            axes[index // columns][index % columns].axis("off")
        for row in range(rows):
            axes[row][0].set_ylabel("success rate")
        for column in range(columns):
            axes[rows - 1][column].set_xlabel("severity")

        fig.suptitle("Success against perturbation severity", color=INK, fontsize=11.5,
                     fontweight=600, y=1.0)
        note = _run_caption(result)
        if anchor is not None:
            note += f"  ·  dashed rule is the clean cell at {anchor:.2f}"
        _caption_below(fig, note, y=0.965)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        return save_figure(fig, path)


def latency(result: Any, path: str | Path) -> Path:
    """Step latency against the control budget, so the number is interpretable.

    A latency in milliseconds means nothing without the rate the robot runs at,
    so the budget line is drawn on the same axes as the p50 and p95.
    """
    plt = _matplotlib()
    p50 = result.metric("efficiency/latency_p50")
    p95 = result.metric("efficiency/latency_p95")
    if p50 is None or not p50.measured:
        raise ValueError("no latency was recorded in this run")
    control_hz = float(result.run.get("control_hz", 10.0))
    budget = 1000.0 / control_hz
    values = [p50.value, p95.value if p95 and p95.measured else None]
    labels = ["p50", "p95"]
    shown = [float(v) for v in values if v is not None]

    # A fast model against a slow control loop spans three or four orders of
    # magnitude -- 0.1 ms under a 100 ms budget -- and on a linear axis the bars
    # are invisible and the chart says nothing. Past a 20x span the axis goes
    # logarithmic, and the values are printed on the bars either way so the
    # numbers are readable whichever scale was chosen.
    log = max(shown) * 20 < budget
    with plot_style():
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        bars = ax.bar(labels[: len(shown)], shown, color=[ACCENT, AMBER][: len(shown)],
                      width=0.5)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(min(shown) / 4, budget * 3)
        else:
            ax.set_ylim(0, max([*shown, budget]) * 1.25)
        ax.axhline(budget, color=RULE, linewidth=1.0, linestyle="--")
        ax.text(
            len(shown) - 0.45,
            budget * (1.15 if log else 1.02),
            f"{control_hz:g} Hz budget ({budget:.0f} ms)",
            fontsize=8,
            color=MUTED,
            ha="right",
        )
        for bar, value in zip(bars, shown, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * (1.25 if log else 1.0) if log else value + max(shown) * 0.03,
                f"{value:.3g} ms",
                ha="center",
                fontsize=9,
                color=INK,
            )
        headroom = 1.0 - max(shown) / budget
        ax.set_ylabel("ms per act() call" + (" (log)" if log else ""))
        ax.set_title(f"{headroom:.1%} of the control period unused", fontsize=10)
        return save_figure(fig, path)


#: Columns of the condition matrix: the metric, its short label, and whether the
#: raw number reads better as a percentage. Deliberately few: a matrix wide enough
#: to need scrolling is a table.
_MATRIX_COLUMNS = (
    ("accuracy/success_rate", "success"),
    ("robustness/retention", "retention"),
    ("security/attack_retention", "attack ret."),
    ("safety/violation_rate", "violations"),
    ("safety/violation_steps", "viol. steps"),
    ("consistency/paraphrase_agreement", "paraphrase"),
)


def heatmap(result: Any, path: str | Path) -> Path:
    """The condition matrix: every cell against the metrics that scored it.

    **Colour is the score, the number is the measurement.** Every column is
    normalised through the same normaliser the dimension scores use, so one
    sequential ramp can run across columns that point in opposite directions: a
    violation rate of 0.78 and a success rate of 0.22 are both dark, because both
    are bad. Colouring the raw values instead would put "78 % of episodes left the
    workspace" at the bright end of the scale.

    A single-column version of this chart is a bar chart, so columns with nothing
    in them are dropped and the figure is skipped entirely when fewer than two
    survive.
    """
    plt = _matplotlib()
    from matplotlib.colors import Normalize

    from .dimensions import normalise
    from .metrics import METRICS

    cells = [c.name for c in result.suite.cells if c.name in result.cells]
    columns = []
    for name, label in _MATRIX_COLUMNS:
        values = [result.cells.get(c, {}).get(name) for c in cells]
        if not any(v is not None and v.measured for v in values):
            continue
        higher = (
            bool(METRICS.entry(name).meta.get("higher_is_better", True))
            if name in METRICS else True
        )
        columns.append((name, label, higher, values))
    if len(columns) < 2 or not cells:
        raise ValueError("the condition matrix needs at least two measured metrics")

    raw = np.full((len(cells), len(columns)), np.nan)
    score = np.full((len(cells), len(columns)), np.nan)
    for j, (name, _label, higher, values) in enumerate(columns):
        for i, value in enumerate(values):
            if value is not None and value.measured:
                raw[i, j] = float(value.value)
                score[i, j] = normalise(name, float(value.value), higher_is_better=higher)

    height = max(2.6, 0.30 * len(cells) + 1.5)
    with plot_style():
        fig, ax = plt.subplots(figsize=(SIZES["matrix"][0], height))
        masked = np.ma.masked_invalid(score)
        # Cells with nothing measured take the page colour rather than a colour
        # from the ramp, so a blank cannot be read as a low score.
        cmap = plt.get_cmap("viridis").with_extremes(bad=PAPER)
        image = ax.imshow(masked, cmap=cmap, norm=Normalize(0, 1), aspect="auto")

        ax.set_xticks(range(len(columns)), [c[1] for c in columns], fontsize=8)
        ax.set_yticks(range(len(cells)), cells, fontsize=7.6)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(cells), 1), minor=True)
        ax.grid(which="minor", color=PAPER, linewidth=1.4)
        ax.grid(which="major", visible=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in range(len(cells)):
            for j in range(len(columns)):
                if np.isnan(raw[i, j]):
                    ax.text(j, i, "--", ha="center", va="center", fontsize=7,
                            color=FAINT)
                    continue
                # White on the dark end of viridis, ink on the light end.
                colour = PAPER if score[i, j] < 0.55 else INK
                ax.text(j, i, f"{raw[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.2, color=colour)

        bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        bar.set_label("score (higher is better)", color=MUTED, fontsize=8)
        bar.ax.tick_params(labelsize=7, length=2, color=RULE, labelcolor=MUTED)
        bar.outline.set_edgecolor(RULE)
        bar.outline.set_linewidth(0.7)

        ax.set_title(
            "Every condition, every metric", fontsize=11.5, fontweight=600, color=INK,
            pad=26, loc="left",
        )
        ax.text(
            0, -0.6, _run_caption(result) + "  ·  colour is the normalised score, "
            "the number is the measurement",
            transform=ax.get_xaxis_transform(), fontsize=7.6, color=MUTED,
            ha="left", va="bottom", clip_on=False,
        )
        return save_figure(fig, path)


def leaderboard_radar(results: Sequence[Any], path: str | Path) -> Path:
    """Several runs on one radar. Capped at :data:`MAX_CATEGORICAL` for legibility."""
    return radar(results[0], path, others=results[1:MAX_CATEGORICAL])


def write_all(result: Any, directory: str | Path) -> dict[str, Path]:
    """Write every figure this run supports, skipping the ones it does not.

    A figure that cannot be drawn -- no severity ladder, no latency -- is absent
    rather than empty, and its absence is not an error: the run directory should
    contain what was measured and nothing pretending to be.
    """
    directory = Path(directory)
    written: dict[str, Path] = {}
    for name, draw in (
        ("radar", radar),
        ("dimension-bars", dimension_bars),
        ("severity", severity_curves),
        ("latency", latency),
        ("heatmap", heatmap),
    ):
        try:
            written[name] = draw(result, directory / f"{name}.png")
        except (ValueError, KeyError, AttributeError):
            continue
    return written


def _curves_from_result(result: Any) -> dict[str, dict[float, float]]:
    """Rebuild the severity curves from a saved result's cells."""
    clean = result.cells.get("clean", {}).get("accuracy/success_rate")
    base = clean.value if clean and clean.measured else None
    curves: dict[str, dict[float, float]] = {}
    for cell in result.suite.cells:
        if cell.perturbation in ("", "none"):
            continue
        value = result.cells.get(cell.name, {}).get("accuracy/success_rate")
        if value is None or not value.measured:
            continue
        curve = curves.setdefault(cell.perturbation, {})
        if base is not None:
            curve[0.0] = float(base)
        curve[float(cell.severity)] = float(value.value)
    return {k: v for k, v in curves.items() if len(v) >= 2}


def _run_label(run: Any) -> str:
    """A short name for a run on a shared chart."""
    model = getattr(run, "model", {}) or {}
    return str(
        model.get("module") or model.get("model_id") or model.get("type", "model")
    )


def tradeoff(analysis: Any, path: str | Path) -> Path:
    """A trade-off as a scatter, with the frontier drawn only when there is one.

    The chart obeys the same rule the analysis does. When the tension test came
    back ``undetermined`` or ``absent``, no frontier line and no knee marker are
    drawn, because a hull through uncorrelated points reads as an exchange and
    there is none. The verdict is written on the chart instead, so the picture
    cannot be quoted apart from its caveat.
    """
    plt = _matplotlib()
    x_axis, y_axis = analysis.tradeoff.axes(analysis.scope)
    if x_axis is None or y_axis is None or not analysis.points:
        raise ValueError("this trade-off has no scatter at this scope")

    points = analysis.points
    on = set(analysis.frontier)
    with plot_style():
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        off_points = [p for p in points if p.label not in on]
        ax.scatter(
            [p.x for p in off_points], [p.y for p in off_points],
            s=34, color=RULE, edgecolor=PAPER, linewidth=0.6, zorder=2,
            label="dominated" if on else None,
        )
        if on:
            front = sorted((p for p in points if p.label in on), key=lambda p: p.x)
            ax.plot([p.x for p in front], [p.y for p in front], color=ACCENT,
                    linewidth=1.4, zorder=3, alpha=0.7)
            ax.scatter([p.x for p in front], [p.y for p in front], s=52, color=ACCENT,
                       edgecolor=PAPER, linewidth=0.8, zorder=4, label="frontier")
            if analysis.knee:
                knee = next(p for p in points if p.label == analysis.knee)
                ax.scatter([knee.x], [knee.y], s=150, facecolor="none",
                           edgecolor=AMBER, linewidth=1.6, zorder=5, label="knee")
        # Model-scope points are few and each is a decision, so they are named.
        if analysis.scope == "models":
            for p in points:
                ax.annotate(p.label, (p.x, p.y), textcoords="offset points",
                            xytext=(6, 4), fontsize=8, color=MUTED)
        ax.set_xlabel(x_axis.label)
        ax.set_ylabel(y_axis.label)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="both", color=GRID, linewidth=0.6)
        ax.set_title(analysis.tradeoff.title, fontsize=11)
        verdict = f"tension: {analysis.tension}"
        if analysis.correlation is not None:
            verdict += f"   ρ = {analysis.correlation:+.2f}"
            if analysis.correlation_ci:
                lo, hi = analysis.correlation_ci
                verdict += f" [{lo:+.2f}, {hi:+.2f}]"
        ax.text(0.5, -0.19, verdict, transform=ax.transAxes, ha="center", fontsize=8.5,
                color=MUTED)
        if on:
            ax.legend(loc="lower left", fontsize=8)
        return save_figure(fig, path)
