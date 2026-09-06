# Figures

Every figure uses the kamara design tokens for its chrome (background, text,
gridlines, spines), so a figure dropped into the docs or a README sits
on the same ground as the page around it rather than announcing itself as a white
rectangle.

```python
result.radar()                    # runs/<...>/figures/radar.png
from xevals import plots
plots.write_all(result, "figures/")
```

Needs `xevals[plots]`. Everything runs headless under `Agg`.

## What gets drawn

| Figure | Shows |
|---|---|
| `radar.png` | one axis per dimension, in the fixed order |
| `dimension-bars.png` | the same scores as bars, each in its dimension's colour |
| `severity.png` | success against severity, one small multiple per family |
| `latency.png` | p50 and p95 against the control-rate budget |
| `heatmap.png` | success rate per cell, viridis as a sequential ramp |

A figure that cannot be drawn (no severity ladder, no timings) is **absent**
rather than empty, and its absence is not an error. The run directory
should contain what was measured and nothing pretending to be.

## One colour per dimension

Accuracy is the same purple in every radar, every bar chart and every page `xevals`
has ever produced. A reader who has learnt the palette once can read any figure
without the legend.

That is only safe because a dimension never has to be told apart from another
*within* one chart by colour alone: a radar gives each its own axis, a bar chart
its own bar. Seven categories exceed what viridis separates legibly as a
categorical palette, and `palette_colors` still refuses more than five genuinely
categorical series:

```python
plots.palette_colors(5)          # fine
plots.palette_colors(6)          # ValueError: use small multiples
plots.palette_colors(3, "brand") # ValueError: the accent pair stops at two
```

The limit is enforced rather than documented, because past it an adjacent pair
falls below the normal-vision legibility floor and no legend rescues a chart the
eye cannot decode. That is why `severity.png` is small multiples.

## The radar draws holes, not zeros

An unmeasured dimension gets a hollow marker on its axis and a gap in the polygon.
Drawing "unmeasured" and "measured as zero" identically is the single most
misleading chart this library could produce.

## The palette

| Token | Hex | Used for |
|---|---|---|
| paper | `#FAFAF8` | figure and axes background |
| ink | `#121412` | titles, labels, annotations |
| muted | `#5C615D` | tick labels |
| rule | `#CBCDCA` | spines and ticks |
| grid | `#DCDEDB` | gridlines, under the data |
| accent | `#367FC9` | the single-series colour, and rules |
| amber | `#C97F36` | the accent's channels reversed; its pair |

The accent measures 3.99:1 on paper: fine for a line, a marker or a rule, not for
text. For an annotation meant to be *read*, darken it a step.

Series palettes are deliberately off-brand. Lightness alone cannot separate four
overlaid curves, and taking hue out would give up the &Delta;E guarantees, so
`BLUE_ORANGE` (Wong's colourblind-safe family) and viridis stay exactly as they
are.

::: xevals.plots
    options:
      heading_level: 2
      members: false
