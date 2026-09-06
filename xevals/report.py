"""One self-contained HTML page: the whole evaluation, openable offline.

A run directory holds JSON, tables, figures and videos, and reading them together
means opening six files in the right order. The report is the version a person
reads, and it is built for two of them: someone deciding whether the model is
good enough, and someone later checking whether the claim was justified.

That second reader is why the page looks the way it does. An auditor needs the
caveats before the numbers, the conditions beside every score, and enough
provenance at the bottom to re-run the thing. So:

**The caveats come first.** A failed replay gate, a truncated run and every
dimension that could not be measured appear as status chips in the header, before
a single score. The whole point of the library is that one confident number hides
the conditions it was measured under; a report that buried them would reproduce
the problem in a nicer font.

**Every state is named as well as coloured.** Pass, incomplete and failed each
carry an icon and a word, so the page survives being printed in greyscale or read
by someone who does not separate red from green.

**Clips are shown, not linked.** A gallery of links is a gallery nobody opens.
GIFs animate in an ``<img>`` and mp4 plays in a ``<video>``, both inline, both
still relative rather than base64: inlining a megabyte of video per cell would
defeat the point of a file you can email.

**It prints.** An audit trail is a PDF more often than a URL, so there is a print
stylesheet: backgrounds forced on, nothing sticky, nothing clipped.

Three constraints from the original design still hold. The page is
self-contained (inline CSS, base64 figures), there is no template engine
(f-strings and ``html.escape``: a dependency for one page is a dependency in
every install), and the icons are inline SVG rather than a font, for the same
reason.
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dimensions import DIMENSION_ORDER
from .dimensions import color as dimension_color

__all__ = [
    "BENCHMARK_PAGES",
    "PAGES",
    "render",
    "render_benchmark",
    "render_benchmark_pages",
    "render_pages",
    "write",
    "write_benchmark",
]

#: Stroke icons, 16px, drawn in ``currentColor`` so a chip's colour reaches them.
#: Inline rather than a font: a font is a second file, and this page is one file.
ICONS = {
    "check": "M3.5 8.5l3 3 6-7",
    "cross": "M4 4l8 8M12 4l-8 8",
    "alert": "M8 5.5v4M8 12h.01M6.7 2.6L1.5 12a1.2 1.2 0 001 1.8h11a1.2 1.2 0 001-1.8"
             "L9.3 2.6a1.2 1.2 0 00-2.6 0z",
    "question": "M6.2 6a1.9 1.9 0 113.1 1.5c-.7.5-1.3.9-1.3 1.9M8 12.5h.01",
    "gauge": "M2.5 12a5.5 5.5 0 1111 0M8 12l3-3.5",
    "grid": "M2.5 2.5h4.5v4.5H2.5zM9 2.5h4.5v4.5H9zM2.5 9h4.5v4.5H2.5zM9 9h4.5v4.5H9z",
    "layers": "M8 1.8L1.8 5 8 8.2 14.2 5zM1.8 8.5L8 11.7l6.2-3.2M1.8 11.5L8 14.7l6.2-3.2",
    "clock": "M8 3.5v4.5l3 1.6M14 8A6 6 0 112 8a6 6 0 0112 0z",
    "film": "M2 3.2h12v9.6H2zM5.2 3.2v9.6M10.8 3.2v9.6M2 8h12",
    "scale": "M8 2.2v11.6M3 5.5h10M5 5.5L3 10h4zM11 5.5L9 10h4zM5.5 13.8h5",
    "table": "M2.2 3h11.6v10H2.2zM2.2 6.6h11.6M6.4 6.6v6.4",
    "shield": "M8 1.9l5 1.9v3.4c0 3.2-2.1 5.6-5 6.9-2.9-1.3-5-3.7-5-6.9V3.8z",
    "fingerprint": "M5 13.4c-.7-1.3-1-2.7-1-4.2a4 4 0 018 0c0 .8-.1 1.6-.3 2.3"
                   "M8 9.2v1.4c0 1.1.2 2.1.6 3.1M2.4 5.2A6.6 6.6 0 018 2.2a6.6 6.6 0 015.6 3",
    "target": "M8 2.4v2M8 11.6v2M2.4 8h2M11.6 8h2M10.6 8a2.6 2.6 0 11-5.2 0 2.6 2.6 0 015.2 0z",
}


def icon(name: str, size: int = 15) -> str:
    """One inline SVG glyph, inheriting the surrounding colour."""
    path = ICONS.get(name, ICONS["question"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 16 16" fill="none" '
        f'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true"><path d="{path}"/></svg>'
    )


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return '<span class="null">--</span>'
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return _escape(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _table(headers: list[str], rows: list[list[Any]], *, mono_first: bool = True) -> str:
    """A table with numeric columns right-aligned and tabular figures.

    Alignment is decided per column from the data rather than per cell, so a
    column with one missing value does not jump between left and right.
    """
    numeric = [
        any(_is_number(row[i]) for row in rows) for i in range(len(headers))
    ]
    head = "".join(
        f'<th class="{"num" if numeric[i] else ""}">{_escape(h)}</th>'
        for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = []
        for i, value in enumerate(row):
            classes = []
            if numeric[i]:
                classes.append("num")
            if i == 0 and mono_first:
                classes.append("name")
            if value is None:
                classes.append("null")
                cells.append(f'<td class="{" ".join(classes)}">--</td>')
            elif isinstance(value, bool):
                cells.append(f'<td class="{" ".join(classes)}">{"yes" if value else "no"}</td>')
            elif isinstance(value, float):
                cells.append(f'<td class="{" ".join(classes)}">{value:.4f}</td>')
            else:
                cells.append(f'<td class="{" ".join(classes)}">{_escape(value)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _dimension_css() -> str:
    """``--dim-<name>`` for all seven, in both themes.

    Generated rather than written into the stylesheet literal, so the report's
    charts take the same palette as the figures and the diagram. The dark
    variants are lifted toward paper: viridis runs from near-black, and accuracy
    is the bar most likely to be looked at.
    """
    light = "".join(
        f"  --dim-{d.value}:{dimension_color(d)};\n" for d in DIMENSION_ORDER
    )
    dark = "".join(
        f"    --dim-{d.value}:{dimension_color(d, dark=True)};\n"
        for d in DIMENSION_ORDER
    )
    return (
        f"\n:root {{\n{light}}}\n"
        f"@media (prefers-color-scheme: dark) {{\n  :root {{\n{dark}  }}\n}}\n"
    )


def _chart(svg: str, caption: str) -> str:
    """One inline chart with its caption. Empty in, empty out."""
    if not svg:
        return ""
    return (
        f'<figure class="chartbox">{svg}'
        f"<figcaption>{_escape(caption)}</figcaption></figure>"
    )


def _dimension_colours() -> dict[str, str]:
    """dimension -> the CSS variable the stylesheet defines for it.

    A variable rather than a hex value, so the chart follows the page into dark
    mode instead of carrying a light palette onto an ink ground.
    """
    return {d.value: f"var(--dim-{d.value})" for d in DIMENSION_ORDER}


def _chip(state: str, glyph: str, text: str) -> str:
    return f'<span class="chip {state}">{icon(glyph, 13)}{_escape(text)}</span>'


def _banner(state: str, glyph: str, title: str, body: str) -> str:
    return (
        f'<div class="banner {state}">{icon(glyph)}<div><strong>{_escape(title)}</strong>'
        f'<div class="body">{body}</div></div></div>'
    )


def _section(anchor: str, glyph: str, title: str, body: str) -> str:
    return (
        f'<section id="{anchor}"><h2><span class="section-icon">{icon(glyph, 17)}</span>'
        f'{_escape(title)}</h2><div class="body">{body}</div></section>'
    )


#: The stylesheet. Inline, because the page is one file.
_CSS_BASE = r"""
/* ---------------------------------------------------------------- tokens --
   kamara's palette, plus three semantic states the neutral ramp cannot carry.
   A report is read by someone deciding whether to trust a number, so "this
   passed", "this is incomplete" and "this failed" have to be distinguishable
   at a glance and without relying on hue alone: every state also carries an
   icon and a word. */
:root {
  --paper:#FAFAF8; --card:#FFFFFF; --ink:#121412; --muted:#5C615D; --faint:#8B908A;
  --grid:#DCDEDB; --rule:#CBCDCA;
  --accent:#367FC9; --accent-text:#2769AD; --accent-soft:#EAF2FB;
  --ok:#1F7A5C; --ok-soft:#E6F2ED;
  --warn:#9A5B18; --warn-soft:#FAF0E4;
  --bad:#A32E22; --bad-soft:#FBEAE8;
  --radius:8px; --radius-sm:5px;
  --shadow:0 1px 2px rgba(18,20,18,.04), 0 1px 8px rgba(18,20,18,.03);
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper:#0B0D0B; --card:#161917; --ink:#F2F3F0; --muted:#A8ADA7; --faint:#767C76;
    --grid:#242825; --rule:#3A3F3B;
    --accent:#5DA3E9; --accent-text:#8BC5FF; --accent-soft:#16222E;
    --ok:#5CBAA7; --ok-soft:#132420;
    --warn:#E9A35D; --warn-soft:#2A2118;
    --bad:#E2796C; --bad-soft:#2B1815;
    --shadow:none;
  }
}
*,*::before,*::after { box-sizing:border-box; }
html { -webkit-text-size-adjust:100%; }
body {
  margin:0; background:var(--paper); color:var(--ink);
  font:15px/1.62 'Hanken Grotesk', ui-sans-serif, system-ui, -apple-system,
       'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
  font-variant-numeric:tabular-nums;
}
.wrap { max-width:1160px; margin:0 auto; padding:0 1.6rem 6rem; }

/* ------------------------------------------------------------------ type -- */
h1 { font-size:2.05rem; font-weight:650; letter-spacing:-.022em; margin:.15rem 0 .3rem; }
h2 { font-size:1.16rem; font-weight:600; letter-spacing:-.012em;
     margin:0 0 1rem; display:flex; align-items:center; gap:.55rem; }
h3 { font-size:.95rem; font-weight:600; margin:1.7rem 0 .5rem; }
p  { margin:.55rem 0; }
.kicker { font-size:.62rem; font-weight:600; letter-spacing:.22em;
          text-transform:uppercase; color:var(--accent-text); }
.lede { color:var(--muted); max-width:70ch; }
.small { font-size:.82rem; color:var(--muted); }
a { color:var(--accent-text); text-underline-offset:2px; }
code, pre, .num { font-family:'Geist Mono', ui-monospace, SFMono-Regular, Menlo,
                  monospace; font-size:.86em; }
code { background:var(--card); border:1px solid var(--grid); border-radius:4px;
       padding:.08em .34em; }
pre { background:var(--card); border:1px solid var(--grid); border-radius:var(--radius);
      padding:.9rem 1.05rem; overflow-x:auto; margin:.7rem 0; }
pre code { background:none; border:0; padding:0; }

/* ---------------------------------------------------------------- header -- */
header.top { padding:3rem 0 1.4rem; border-bottom:1px solid var(--grid); margin-bottom:1.6rem; }
.meta-strip { display:flex; flex-wrap:wrap; gap:.35rem 1.6rem; margin-top:1rem;
              font-size:.8rem; color:var(--muted); }
.meta-strip b { color:var(--ink); font-weight:600; }

/* ----------------------------------------------------------------- chips -- */
.chips { display:flex; flex-wrap:wrap; gap:.5rem; margin:1.1rem 0 0; }
.chip { display:inline-flex; align-items:center; gap:.42rem; border-radius:999px;
        padding:.3rem .78rem .3rem .6rem; font-size:.79rem; font-weight:500;
        border:1px solid var(--grid); background:var(--card); color:var(--ink); }
.chip svg { flex:none; }
.chip.ok   { color:var(--ok);   border-color:var(--ok);   background:var(--ok-soft); }
.chip.warn { color:var(--warn); border-color:var(--warn); background:var(--warn-soft); }
.chip.bad  { color:var(--bad);  border-color:var(--bad);  background:var(--bad-soft); }

/* --------------------------------------------------------------- banners -- */
.banner { display:flex; gap:.7rem; border:1px solid var(--grid); border-left-width:3px;
          border-radius:var(--radius); padding:.85rem 1.05rem; margin:1rem 0;
          background:var(--card); }
.banner svg { flex:none; margin-top:.15rem; }
.banner strong { display:block; margin-bottom:.15rem; }
.banner .body { font-size:.88rem; color:var(--muted); }
.banner.bad  { border-left-color:var(--bad);  background:var(--bad-soft); }
.banner.bad strong { color:var(--bad); }
.banner.warn { border-left-color:var(--warn); background:var(--warn-soft); }
.banner.warn strong { color:var(--warn); }
.banner.ok   { border-left-color:var(--ok);   background:var(--ok-soft); }
.banner.ok strong { color:var(--ok); }
.banner.info { border-left-color:var(--accent); background:var(--accent-soft); }

/* -------------------------------------------------------------- sections -- */
section { margin:2.6rem 0 0; scroll-margin-top:1rem; }
section > .body { border-top:1px solid var(--grid); padding-top:1.1rem; }
.section-icon { color:var(--accent); display:inline-flex; }

/* ----------------------------------------------------------------- cards -- */
.scores { display:grid; gap:.75rem; margin:1.2rem 0 1.6rem;
          grid-template-columns:repeat(auto-fit,minmax(158px,1fr)); }
/* Seven dimensions on one row where there is room: six and a stray reads as a
   grid that ran out, not as a set of seven. */
@media (min-width:1080px) { .scores { grid-template-columns:repeat(7,minmax(0,1fr)); } }
/* A benchmark's cards carry a name and a bar strip, so they need more width and
   there are only ever a handful of them. */
@media (min-width:1080px) { .models .scores, .scores:has(.model-name) {
  grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); } }
.score { position:relative; background:var(--card); border:1px solid var(--grid);
         border-radius:var(--radius); padding:.8rem .9rem .85rem; box-shadow:var(--shadow);
         overflow:hidden; }
.score::before { content:""; position:absolute; inset:0 auto 0 0; width:3px;
                 background:var(--dim, var(--rule)); }
.score .name { font-size:.63rem; font-weight:600; letter-spacing:.09em;
               text-transform:uppercase; color:var(--muted); }
.score .value { font-size:1.85rem; font-weight:650; letter-spacing:-.035em;
                line-height:1.1; margin-top:.12rem; }
.score .bar { height:4px; border-radius:2px; background:var(--grid); margin-top:.55rem;
              overflow:hidden; }
.score .bar span { display:block; height:100%; background:var(--dim); border-radius:2px; }
.score .q { font-size:.72rem; color:var(--faint); margin-top:.5rem; line-height:1.4; }
.score.unmeasured .value { font-size:.92rem; font-weight:500; letter-spacing:0;
                           color:var(--muted); padding:.42rem 0 .3rem; }
.score.unmeasured::before { background:repeating-linear-gradient(
    180deg, var(--rule) 0 4px, transparent 4px 8px); }

/* A benchmark card carries a model's whole profile as seven bars, so that two
   models with the same mean cannot look the same. */
.score .model-name { font-size:1.02rem; font-weight:600; letter-spacing:-.012em;
                     margin:.1rem 0 .05rem; }
.score.lead { border-color:var(--accent); }
.score .strip { display:flex; align-items:flex-end; gap:2px; height:26px;
                margin:.6rem 0 .5rem; }
.score .strip i { flex:1; min-height:2px; border-radius:1.5px; display:block; }
.score a { text-decoration:none; }

/* ---------------------------------------------------------------- tables -- */
.scroll { overflow-x:auto; margin:.8rem 0 1.2rem; border:1px solid var(--grid);
          border-radius:var(--radius); background:var(--card); }
table { border-collapse:collapse; width:100%; font-size:.855rem; }
th { text-align:left; font-size:.63rem; font-weight:600; letter-spacing:.085em;
     text-transform:uppercase; color:var(--muted); white-space:nowrap;
     padding:.6rem .85rem; border-bottom:1px solid var(--rule); background:var(--card);
     position:sticky; top:0; }
td { padding:.48rem .85rem; border-bottom:1px solid var(--grid); white-space:nowrap; }
tbody tr:last-child td { border-bottom:0; }
tbody tr:hover td { background:var(--accent-soft); }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
td.null { color:var(--faint); }
td.name { font-family:'Geist Mono', ui-monospace, monospace; font-size:.8rem; }

/* --------------------------------------------------------------- figures -- */
/* Charts are inline SVG rather than an image, so they take the page's type and
   its colours and stay sharp at any zoom. The plate keeps them legible in dark
   mode without baking a background into the drawing. */
figure.chartbox { background:var(--card); border:1px solid var(--grid);
                  border-radius:var(--radius); padding:.9rem 1rem .7rem; }
svg.chart { display:block; width:100%; height:auto; }
svg.chart text { font-family:'Hanken Grotesk', ui-sans-serif, system-ui, -apple-system,
                 'Segoe UI', sans-serif; font-variant-numeric:tabular-nums; }
figure { margin:1.1rem 0; }
figure img { width:100%; display:block; border:1px solid var(--grid);
             border-radius:var(--radius); background:var(--paper); }
figcaption { color:var(--muted); font-size:.79rem; margin-top:.5rem; line-height:1.5; }
.split { display:grid; gap:1.4rem; grid-template-columns:1fr; align-items:start; }
@media (min-width:900px) { .split { grid-template-columns:minmax(0,1fr) minmax(0,1fr); } }

/* ---------------------------------------------------------------- clips -- */
.gallery { display:grid; gap:1rem; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); }
.clip { margin:0; background:var(--card); border:1px solid var(--grid);
        border-radius:var(--radius); overflow:hidden; box-shadow:var(--shadow); }
.clip img, .clip video { width:100%; display:block; background:var(--paper);
                         border:0; border-radius:0; image-rendering:pixelated; }
.clip figcaption { margin:0; padding:.55rem .7rem .6rem; border-top:1px solid var(--grid); }
.clip .cell { font-family:'Geist Mono', ui-monospace, monospace; font-size:.74rem;
              color:var(--ink); display:block; overflow:hidden; text-overflow:ellipsis; }
.clip .outcome { font-size:.7rem; font-weight:600; letter-spacing:.06em;
                 text-transform:uppercase; }
.clip .outcome.ok { color:var(--ok); } .clip .outcome.bad { color:var(--bad); }

/* ------------------------------------------------------------------- nav --
   The report is a handful of pages in one directory rather than one very long
   one. The nav is the same on every page, so a reader who has found the episode
   list once can find it again without going back to the overview. */
nav.pages { display:flex; flex-wrap:wrap; gap:.25rem; margin:0 0 1.8rem;
            border-bottom:1px solid var(--grid); padding-bottom:0;
            position:sticky; top:0; z-index:20; background:var(--paper);
            backdrop-filter:saturate(1.6) blur(8px); }
/* The nav is sticky, so an anchored section has to clear it rather than land
   underneath it. */
section, .cellblock { scroll-margin-top:4rem; }
nav.pages a { display:inline-flex; align-items:center; gap:.42rem; padding:.55rem .85rem;
              font-size:.86rem; color:var(--muted); text-decoration:none;
              border-bottom:2px solid transparent; margin-bottom:-1px; }
nav.pages a:hover { color:var(--ink); }
nav.pages a[aria-current] { color:var(--accent-text); border-bottom-color:var(--accent);
                            font-weight:600; }
nav.pages a svg { opacity:.8; }

/* --------------------------------------------------------------- episodes --
   One block per condition: a header carrying the cell's own numbers, then its
   clips, then every episode it ran. */
.cellblock { border:1px solid var(--grid); border-radius:var(--radius);
             background:var(--card); margin:1.1rem 0; overflow:hidden; }
.cellblock > header { display:flex; flex-wrap:wrap; align-items:baseline; gap:.5rem 1.2rem;
                      padding:.75rem 1rem; border-bottom:1px solid var(--grid); }
.cellblock h3 { margin:0; font-family:'Geist Mono', ui-monospace, monospace;
                font-size:.92rem; }
.cellblock .facts { display:flex; gap:1.1rem; font-size:.8rem; color:var(--muted);
                    margin-left:auto; }
.cellblock .facts b { color:var(--ink); font-weight:600; }
.cellblock .inner { padding:1rem; }
.cellblock .gallery { margin-bottom:.9rem; }
.dot { display:inline-block; width:.62rem; height:.62rem; border-radius:2px;
       margin-right:2px; vertical-align:-1px; }
.dot.ok { background:var(--ok); } .dot.bad { background:var(--bad); }
.dot.none { background:var(--rule); }
.legend { font-size:.78rem; color:var(--muted); display:flex; gap:1rem; flex-wrap:wrap;
          margin:.4rem 0 0; }

/* ------------------------------------------------------------------- toc -- */
nav.toc { border:1px solid var(--grid); border-radius:var(--radius); background:var(--card);
          padding:.8rem 1rem; margin:1.6rem 0; }
nav.toc ol { list-style:none; margin:0; padding:0; display:flex; flex-wrap:wrap;
             gap:.3rem 1.3rem; font-size:.82rem; counter-reset:s; }
nav.toc a { text-decoration:none; color:var(--muted); }
nav.toc a:hover { color:var(--accent-text); }
nav.toc li::before { counter-increment:s; content:counter(s) ". "; color:var(--faint); }

/* ------------------------------------------------------------ provenance -- */
dl.meta { display:grid; grid-template-columns:max-content minmax(0,1fr);
          gap:.3rem 1.3rem; font-size:.84rem; margin:.6rem 0; }
dl.meta dt { color:var(--muted); }
dl.meta dd { margin:0; font-family:'Geist Mono', ui-monospace, monospace;
             font-size:.8rem; overflow-wrap:anywhere; }

footer { margin-top:4rem; padding-top:1.1rem; border-top:1px solid var(--grid);
         color:var(--faint); font-size:.79rem; display:flex; justify-content:space-between;
         flex-wrap:wrap; gap:.5rem; }

/* A report is an audit artefact, so it has to survive being printed to PDF:
   backgrounds forced on, nothing sticky, no clipped tables, links spelled out. */
@media print {
  :root { --paper:#fff; --card:#fff; --shadow:none; }
  body { font-size:10.5pt; }
  .wrap { max-width:none; padding:0; }
  nav.toc, nav.pages { display:none; }
  th { position:static; }
  .scroll { overflow:visible; break-inside:avoid; }
  section, figure, .clip, .banner, .score { break-inside:avoid; }
  * { -webkit-print-color-adjust:exact; print-color-adjust:exact; }
}
"""

#: The stylesheet, with the dimension palette appended.
CSS = _CSS_BASE + _dimension_css()



# --------------------------------------------------------------------------
# Status: the caveats, before anything else
# --------------------------------------------------------------------------

#: gate status -> (chip state, icon, wording). Named as well as coloured, so the
#: page reads correctly in greyscale and to anyone who does not separate hues.
_GATE = {
    "passed": ("ok", "shield", "Gate passed"),
    "failed": ("bad", "cross", "Gate FAILED"),
    "unmeasured": ("warn", "question", "Gate unmeasured"),
}


def _status_chips(result: Any) -> str:
    """The header strip: gate, completeness, coverage. Facts, not decoration."""
    state, glyph, text = _GATE[result.gate_status()]
    chips = [_chip(state, glyph, text)]

    measured = [d for d in result.dimensions.values() if d.score is not None]
    total = len(result.dimensions)
    chips.append(
        _chip(
            "ok" if len(measured) == total else "warn",
            "target",
            f"{len(measured)} of {total} dimensions measured",
        )
    )
    chips.append(
        _chip("bad", "alert", "Partial run") if result.partial
        else _chip("ok", "check", "Complete")
    )
    return f'<div class="chips">{"".join(chips)}</div>'


def _caveats(result: Any) -> str:
    """The banners. Above the numbers, always, and never below them."""
    blocks = []
    if result.gate_failed:
        rate = (result.baselines.get("replay") or {}).get("success_rate")
        blocks.append(_banner(
            "bad", "cross", "Replay gate failed",
            "Replaying the demonstrator's own actions succeeded on only "
            f"{_number(rate, 2)} of episodes in this environment. The environment does "
            "not match the data, so <strong>nothing below measures the model</strong>.",
        ))
    elif result.gate_status() == "unmeasured":
        blocks.append(_banner(
            "warn", "question", "Replay gate unmeasured",
            "This environment supplies no demonstrator actions, so its faithfulness to "
            "the data was not checked. That is a third state, and it is not a pass.",
        ))
    if result.partial:
        budget = result.run.get("budget") or {}
        blocks.append(_banner(
            "warn", "clock", "Partial run",
            f"A budget truncated the evaluation after "
            f"{_escape(budget.get('episodes_run', '?'))} episodes. Cells are complete or "
            "absent, never half-measured.",
        ))
    unmeasured = [
        (name, next(iter(score.skipped.values()), "not measured"))
        for name, score in result.dimensions.items()
        if score.score is None
    ]
    if unmeasured:
        items = "; ".join(f"<b>{_escape(n)}</b>: {_escape(r)}" for n, r in unmeasured)
        blocks.append(_banner(
            "info", "question", "Dimensions not measured",
            items + ". These are blank, not zero.",
        ))
    return "".join(blocks)


def _score_cards(result: Any) -> str:
    """One card per dimension, in the fixed order, each in its own colour."""
    cards = []
    for dimension in DIMENSION_ORDER:
        score = result.dimensions.get(dimension.value)
        if score is None:
            continue
        colour = dimension_color(dimension)
        if score.score is None:
            reason = next(iter(score.skipped.values()), "not measured")
            cards.append(
                f'<div class="score unmeasured"><div class="name">{dimension.value}</div>'
                f'<div class="value">not measured</div>'
                f'<div class="q">{_escape(reason)}</div></div>'
            )
            continue
        width = max(2, int(round(score.score * 100)))
        cards.append(
            f'<div class="score" style="--dim:{colour}">'
            f'<div class="name">{dimension.value}</div>'
            f'<div class="value">{score.score:.2f}</div>'
            f'<div class="bar"><span style="width:{width}%"></span></div>'
            f'<div class="q">{_escape(dimension.question)}</div></div>'
        )
    return f'<div class="scores">{"".join(cards)}</div>'


# --------------------------------------------------------------------------
# Clips: shown, not linked
# --------------------------------------------------------------------------

#: What a browser can play inline, and the element that plays it.
_PLAYABLE = {".gif": "img", ".webp": "img", ".png": "img", ".mp4": "video", ".webm": "video"}


def _gallery(result: Any, directory: Path) -> str:
    """Every recorded clip, embedded, with the condition and outcome under it.

    Relative rather than base64, deliberately: a megabyte of video per cell would
    defeat the point of a file you can attach to an email. The trade is that the
    clips travel with the directory, which is what the run directory is for.
    """
    videos = Path(directory) / "videos"
    if not videos.exists():
        return ""
    cells = {c.name: c for c in result.suite.cells}
    figures = []
    for cell_dir in sorted(p for p in videos.iterdir() if p.is_dir()):
        for clip in sorted(cell_dir.iterdir()):
            element = _PLAYABLE.get(clip.suffix)
            if element is None:
                continue
            src = _escape(f"videos/{cell_dir.name}/{clip.name}")
            cell = cells.get(cell_dir.name)
            condition = (
                f"{cell.perturbation} @ {cell.severity:g}"
                if cell and cell.perturbation not in ("", "none")
                else (cell.split if cell and cell.split != "in" else "clean")
            )
            success = _episode_outcome(result, cell_dir.name, clip.stem)
            outcome = (
                f'<span class="outcome {"ok" if success else "bad"}">'
                f'{"success" if success else "failed"}</span>'
                if success is not None
                else ""
            )
            media = (
                f'<img loading="lazy" alt="{_escape(cell_dir.name)}" src="{src}">'
                if element == "img"
                else f'<video src="{src}" controls loop muted playsinline></video>'
            )
            figures.append(
                f'<figure class="clip">{media}<figcaption>'
                f'<span class="cell">{_escape(cell_dir.name)}</span>'
                f'<span class="small">{_escape(condition)}</span> {outcome}'
                f"</figcaption></figure>"
            )
    return f'<div class="gallery">{"".join(figures)}</div>' if figures else ""


def _cell_gallery(result: Any, directory: Path, cell: str) -> str:
    """The clips recorded for one condition, embedded."""
    folder = Path(directory) / "videos" / cell
    if not folder.exists():
        return ""
    figures = []
    for clip in sorted(folder.iterdir()):
        element = _PLAYABLE.get(clip.suffix)
        if element is None:
            continue
        src = _escape(f"videos/{cell}/{clip.name}")
        success = _episode_outcome(result, cell, clip.stem)
        outcome = (
            f'<span class="outcome {"ok" if success else "bad"}">'
            f'{"success" if success else "failed"}</span>'
            if success is not None
            else ""
        )
        media = (
            f'<img loading="lazy" alt="{_escape(cell)} episode {_escape(clip.stem)}" '
            f'src="{src}">'
            if element == "img"
            else f'<video src="{src}" controls loop muted playsinline></video>'
        )
        figures.append(
            f'<figure class="clip">{media}<figcaption>'
            f'<span class="cell">episode {_escape(clip.stem)}</span>{outcome}'
            f"</figcaption></figure>"
        )
    return f'<div class="gallery">{"".join(figures)}</div>' if figures else ""


def _episode_outcome(result: Any, cell: str, stem: str) -> bool | None:
    """Whether the episode a clip shows succeeded, when the run kept it."""
    episodes = (result.trajectories or {}).get(cell)
    if not episodes:
        return None
    try:
        return bool(episodes[int(stem)].success)
    except (ValueError, IndexError, AttributeError):
        return None


# --------------------------------------------------------------------------
# Trade-offs
# --------------------------------------------------------------------------

#: verdict -> (banner state, icon, wording). The wording matters: a reader who
#: takes "undetermined" for "no trade-off" has learnt the opposite of the run.
_VERDICTS = {
    "present": ("warn", "scale", "A trade-off is present"),
    "absent": ("ok", "check", "No trade-off here: the two move together"),
    "undetermined": ("info", "question", "Undetermined"),
    "insufficient": ("info", "question", "Not enough points"),
}


def _tradeoffs(analyses: Any, directory: Path) -> str:
    """Verdict first, then the evidence for it."""
    if not analyses:
        return ""
    blocks = [
        '<p class="lede">Whether one dimension was bought with another, tested rather '
        "than assumed. <em>Undetermined</em> means the run supports no claim either way; "
        "no frontier or exchange rate is quoted unless the tension was demonstrated.</p>"
    ]
    for analysis in analyses:
        state, glyph, headline = _VERDICTS.get(
            analysis.tension, ("info", "question", analysis.tension)
        )
        detail = _escape(analysis.reason)
        if analysis.correlation is not None and analysis.correlation_ci:
            lo, hi = analysis.correlation_ci
            detail += (
                f" Spearman &rho; = {analysis.correlation:+.2f}, 95% "
                f"[{lo:+.2f}, {hi:+.2f}], over {len(analysis.points)} "
                f"{_escape(analysis.scope)}."
            )
        blocks.append(f"<h3>{_escape(analysis.tradeoff.title)}</h3>")
        blocks.append(f'<p class="lede">{_escape(analysis.tradeoff.question)}</p>')
        blocks.append(_banner(state, glyph, headline, detail))
        if analysis.exchange_rate is not None:
            x_axis, y_axis = analysis.tradeoff.axes(analysis.scope)
            turn = (
                f", turning at <code>{_escape(analysis.knee)}</code>"
                if analysis.knee else ""
            )
            blocks.append(
                f'<p class="lede">Along the frontier, roughly '
                f"<strong>{abs(analysis.exchange_rate):.2f}</strong> of "
                f"{_escape(y_axis.label)} is given up per unit of "
                f"{_escape(x_axis.label)}{turn}.</p>"
            )
        from . import charts

        x_axis, y_axis = analysis.tradeoff.axes(analysis.scope)
        if x_axis is not None and y_axis is not None and analysis.points:
            blocks.append(_chart(
                charts.scatter(
                    [(p.label, p.x, p.y) for p in analysis.points],
                    frontier=analysis.frontier,
                    knee=analysis.knee,
                    x_label=x_axis.label,
                    y_label=y_axis.label,
                    named=analysis.scope == "models",
                ),
                f"Each point is one {analysis.scope[:-1]}. "
                + (
                    "The line is the frontier and the ring is its turning point."
                    if analysis.frontier
                    else "No frontier is drawn: the tension was not demonstrated."
                ),
            ))
        headers, rows = analysis.group_rows()
        if rows:
            blocks.append(_table(headers, rows))
            if analysis.gap is not None:
                blocks.append(
                    f'<p class="lede">Gap between the groups: '
                    f"<strong>{analysis.gap:+.3f}</strong>.</p>"
                )
        blocks.append(
            f'<p class="small"><em>Why these might trade:</em> '
            f"{_escape(analysis.tradeoff.mechanism)}</p>"
        )
    return "".join(blocks)


# --------------------------------------------------------------------------
# Provenance: what an auditor needs to check the claim
# --------------------------------------------------------------------------


def _reproduce(result: Any) -> str:
    """The exact command that produces this run again."""
    run = result.run
    seeds = ",".join(str(s) for s in run.get("seeds", [0]))
    command = (
        f"xevals run config.toml suite.name={result.suite.name} "
        f"run.episodes={run.get('episodes', 20)} run.seeds=[{seeds}] "
        f"run.root_seed={run.get('root_seed', 0)}"
    )
    return f"<pre><code>{_escape(command)}</code></pre>"


def _provenance(result: Any) -> str:
    """Model, environment, library versions and the seeds, as a definition list."""
    pairs: list[tuple[str, Any]] = [("schema", result.to_dict().get("schema"))]
    pairs.append(("config hash", result.config_hash()))
    for key, value in (result.model or {}).items():
        pairs.append((f"model.{key}", value))
    for key, value in (result.env or {}).items():
        pairs.append((f"env.{key}", value))
    for key in ("seeds", "episodes", "horizon", "root_seed", "control_hz"):
        if key in result.run:
            pairs.append((f"run.{key}", result.run[key]))
    for key, value in (result.run.get("environment") or {}).items():
        pairs.append((key, value))
    items = "".join(
        f"<dt>{_escape(k)}</dt><dd>"
        f"{_escape(json.dumps(v) if isinstance(v, (list, dict)) else v)}</dd>"
        for k, v in pairs
    )
    return (
        '<p class="lede">Everything needed to re-run this evaluation and get the same '
        "numbers, wall-clock timings excepted.</p>"
        + _reproduce(result)
        + f'<dl class="meta">{items}</dl>'
    )


# --------------------------------------------------------------------------
# The pages
# --------------------------------------------------------------------------

#: The stylesheet lives in its own file once a report is several pages, so the
#: 10 kB of CSS is downloaded once rather than inlined six times.
STYLESHEET_NAME = "report.css"


def _chrome(
    *,
    title: str,
    kicker: str,
    subtitle: str,
    nav: str,
    body: str,
    css_href: str | None,
) -> str:
    """The shell every page shares: head, nav, body, footer."""
    from . import __version__

    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    style = (
        f'<link rel="stylesheet" href="{css_href}">'
        if css_href
        else f"<style>{CSS}</style>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_escape(kicker)} &middot; {_escape(title)}</title>"
        f"{style}</head><body><div class='wrap'>{nav}{body}"
        f"<footer><span>Generated by xevals {_escape(__version__)} on {stamp}</span>"
        f"<span>{_escape(subtitle)}</span></footer>"
        "</div></body></html>"
    )


def _nav(pages: list[tuple[str, str, str]], current: str) -> str:
    """The tab bar. Same on every page, so a reader can move without going back."""
    # Built with concatenation rather than a nested f-string: quoting an
    # attribute inside an f-string expression needs Python 3.12, and this package
    # supports 3.11.
    links = []
    for name, label, glyph in pages:
        current_attr = ' aria-current="page"' if name == current else ""
        links.append(
            f'<a href="{_escape(name)}"{current_attr}>{icon(glyph, 14)}'
            f"{_escape(label)}</a>"
        )
    links = "".join(links)
    return f'<nav class="pages" aria-label="Report sections">{links}</nav>'


def _page_header(result: Any, *, title: str, lede: str) -> str:
    run = result.run
    return (
        '<header class="top">'
        f'<p class="kicker">xevals evaluation &middot; {_escape(result.suite.name)}</p>'
        f"<h1>{_escape(title)}</h1>"
        f'<p class="lede">{_escape(lede)}</p>'
        f'<div class="meta-strip">'
        f'<span><b>{_escape(run.get("episodes", "?"))}</b> episodes '
        f'&times; <b>{len(run.get("seeds", []))}</b> seeds</span>'
        f'<span><b>{len(result.suite.cells)}</b> conditions</span>'
        f'<span>root seed <b>{_escape(run.get("root_seed", 0))}</b></span>'
        f'<span>config <b>{_escape(result.config_hash())}</b></span>'
        "</div>"
        f"{_status_chips(result)}"
        "</header>"
    )


def _model_title(result: Any) -> str:
    model = result.model or {}
    return str(model.get("model_id") or model.get("module") or model.get("type", "model"))


def _page_overview(result: Any, directory: Path) -> str:
    from . import charts

    colours = _dimension_colours()
    scores = {d.value: result.scores.get(d.value) for d in DIMENSION_ORDER
              if d.value in result.dimensions}
    body = [
        _page_header(result, title=_model_title(result),
                     lede=result.suite.description or ""),
        _caveats(result),
        _section(
            "summary", "target", "Summary",
            _score_cards(result)
            + '<div class="split">'
            + _chart(
                charts.radar(scores, colours),
                "Dimension scores. A hollow marker at the centre is an axis nothing "
                "measured, which is not the same as a score of zero.",
            )
            + _chart(
                charts.bars([(name, value, colours[name]) for name, value in scores.items()]),
                "The same scores as bars. A polygon's area is easy to over-read; a bar "
                "is not.",
            )
            + "</div>",
        ),
    ]
    headers, rows = result.dimension_rows()
    body.append(_section(
        "dimensions", "layers", "Dimensions",
        '<p class="lede">One row per dimension, in the fixed order. <em>skipped</em> '
        "counts metrics that could not be measured and are therefore blank rather than "
        "zero.</p>" + _table(headers, rows),
    ))
    return "".join(body)


#: Columns of the condition matrix. Few on purpose: a matrix wide enough to need
#: scrolling is a table, and there is a table directly above it.
_MATRIX_COLUMNS = (
    ("accuracy/success_rate", "success"),
    ("robustness/retention", "retention"),
    ("security/attack_retention", "attack ret."),
    ("safety/violation_rate", "violations"),
    ("safety/violation_steps", "viol. steps"),
    ("consistency/paraphrase_agreement", "paraphrase"),
)


def _condition_matrix(result: Any) -> str:
    """The matrix, as inline SVG: colour is the score, the number is the value."""
    from . import charts
    from .dimensions import normalise
    from .metrics import METRICS

    cells = [c.name for c in result.suite.cells if c.name in result.cells]
    columns, scores, raw = [], [], []
    kept = []
    for name, label in _MATRIX_COLUMNS:
        values = [result.cells.get(c, {}).get(name) for c in cells]
        if any(v is not None and v.measured for v in values):
            higher = (
                bool(METRICS.entry(name).meta.get("higher_is_better", True))
                if name in METRICS else True
            )
            kept.append((name, label, higher, values))
            columns.append(label)
    if len(columns) < 2 or not cells:
        return ""
    for index in range(len(cells)):
        score_row, raw_row = [], []
        for name, _label, higher, values in kept:
            value = values[index]
            if value is not None and value.measured:
                raw_row.append(float(value.value))
                score_row.append(
                    normalise(name, float(value.value), higher_is_better=higher)
                )
            else:
                raw_row.append(None)
                score_row.append(None)
        scores.append(score_row)
        raw.append(raw_row)
    return charts.matrix(cells, columns, scores, raw)


def _page_conditions(result: Any, directory: Path) -> str:
    from . import charts
    from .plots import _curves_from_result

    headers, rows = result.cell_rows()
    curves = _curves_from_result(result)
    anchor = next((c[0.0] for c in curves.values() if 0.0 in c), None)
    ladders = charts.ladders(curves, baseline=anchor)
    return (
        _page_header(result, title="Conditions",
                     lede="Every cell the suite ran, and what the model scored under it.")
        + _section(
            "cells", "grid", "Every condition",
            _table(headers, rows)
            + _chart(
                _condition_matrix(result),
                "Colour is the normalised score, so one ramp runs across columns that "
                "point in opposite directions; the number is the measurement.",
            ),
        )
        + _section(
            "severity", "target", "Severity ladders",
            '<p class="lede">One panel per perturbation family. The dashed rule is this '
            "model's own clean success rate, so a curve is read against the model rather "
            "than against an assumed 1.00.</p>"
            + (_chart(ladders, "Success against perturbation severity.")
               or '<p class="lede">No severity ladder was run.</p>'),
        )
    )


def _page_episodes(result: Any, directory: Path) -> str:
    """Every episode, grouped by the condition it ran under.

    The page a reader reaches for when a number looks wrong. Each condition gets
    its own block: the cell's own numbers, the clips that were recorded for it,
    and then a strip showing the outcome of *every* episode it ran, not only the
    ones with video. Per-episode outcomes survive in ``run.json`` even when the
    trajectories do not, so this page still works on a reloaded run.
    """
    blocks = []
    for cell in result.suite.cells:
        values = result.cells.get(cell.name, {})
        success = values.get("accuracy/success_rate")
        clips = _cell_gallery(result, directory, cell.name)
        outcomes = _episode_strip(success)
        if not clips and not outcomes:
            continue
        condition = (
            f"{cell.perturbation} @ {cell.severity:g}"
            if cell.perturbation not in ("", "none")
            else (cell.split if cell.split != "in" else "clean")
        )
        facts = [f"<span>{_escape(condition)}</span>"]
        if success is not None and success.measured:
            facts.append(f"<span>success <b>{success.value:.2f}</b></span>")
            facts.append(f"<span>n <b>{success.n}</b></span>")
        violations = values.get("safety/violation_rate")
        if violations is not None and violations.measured:
            facts.append(f"<span>violations <b>{violations.value:.2f}</b></span>")
        blocks.append(
            f'<div class="cellblock"><header><h3>{_escape(cell.name)}</h3>'
            f'<div class="facts">{"".join(facts)}</div></header>'
            f'<div class="inner">{clips}{outcomes}</div></div>'
        )
    if not blocks:
        blocks.append('<p class="lede">No episodes were recorded in this run.</p>')
    legend = (
        '<div class="legend">'
        '<span><span class="dot ok"></span> success</span>'
        '<span><span class="dot bad"></span> failure</span>'
        '<span><span class="dot none"></span> no verdict</span>'
        "</div>"
    )
    return (
        _page_header(result, title="Episodes",
                     lede="Every episode, grouped by the condition it ran under.")
        + _section(
            "episodes", "film", "Episodes by condition",
            '<p class="lede">Clips carry the step, the condition and the instruction '
            "<em>as the model received it</em>. Below each set, one mark per episode the "
            "cell ran, in seed order.</p>" + legend + "".join(blocks),
        )
    )


def _episode_strip(value: Any) -> str:
    """One mark per episode, in seed order, from the metric's per-episode values."""
    if value is None or not value.per_episode:
        return ""
    marks = "".join(
        f'<span class="dot {"ok" if outcome >= 0.5 else "bad"}" '
        f'title="episode {index}: {"success" if outcome >= 0.5 else "failure"}"></span>'
        for index, outcome in enumerate(value.per_episode)
    )
    return f'<div class="legend"><span>{marks}</span><span>{value.n} episodes</span></div>'


def _page_tradeoffs(result: Any, directory: Path) -> str:
    analyses = result._safe_tradeoffs()
    if not analyses:
        return ""
    return (
        _page_header(result, title="Trade-offs",
                     lede="Whether one dimension was bought with another.")
        + _section("tradeoffs", "scale", "Trade-offs", _tradeoffs(analyses, directory))
    )


def _page_metrics(result: Any, directory: Path) -> str:
    baseline_rows = [
        [name, info.get("success_rate"), info.get("episodes"),
         "gate" if info.get("gate") else "", info.get("reason", "")]
        for name, info in result.baselines.items()
    ]
    from . import charts

    p50 = result.metric("efficiency/latency_p50")
    p95 = result.metric("efficiency/latency_p95")
    control_hz = float(result.run.get("control_hz", 10.0))
    efficiency = _chart(
        charts.budget(
            p50.value if p50 and p50.measured else None,
            p95.value if p95 and p95.measured else None,
            1000.0 / control_hz,
            control_hz,
        ),
        "Milliseconds per act() call against the period the control loop allows.",
    )
    return (
        _page_header(result, title="Metrics",
                     lede="Every measurement, with its interval and its n.")
        + _section(
            "baselines", "shield", "Baselines and gate",
            '<p class="lede"><code>random</code> and <code>noop</code> bound the task from '
            "below. <code>replay</code> is a gate: if the demonstrator's own actions do not "
            "solve the task here, no number anywhere in this report measures the model.</p>"
            + (_table(["baseline", "success", "episodes", "role", "note"], baseline_rows)
               if baseline_rows else '<p class="lede">No baselines were run.</p>'),
        )
        + (_section("efficiency", "clock", "Efficiency", efficiency) if efficiency else "")
        + _section(
            "all", "table", "All metrics",
            '<p class="lede">Every metric, with its 95% bootstrap interval and the number '
            "of episodes behind it. A value without its <code>n</code> is a rumour, and a "
            "blank is a measurement that could not be made rather than a zero.</p>"
            + _table(*result.metric_rows()),
        )
    )


def _page_provenance(result: Any, directory: Path) -> str:
    return (
        _page_header(result, title="Provenance",
                     lede="Everything needed to re-run this and get the same numbers.")
        + _section("provenance", "fingerprint", "Provenance", _provenance(result))
    )


#: filename, label, icon, builder. Order is the nav order. A builder returning
#: an empty string drops its page *and* its tab, so a run with no trade-offs does
#: not advertise an empty one.
PAGES: tuple[tuple[str, str, str, Any], ...] = (
    ("report.html", "Overview", "target", _page_overview),
    ("conditions.html", "Conditions", "grid", _page_conditions),
    ("episodes.html", "Episodes", "film", _page_episodes),
    ("tradeoffs.html", "Trade-offs", "scale", _page_tradeoffs),
    ("metrics.html", "Metrics", "table", _page_metrics),
    ("provenance.html", "Provenance", "fingerprint", _page_provenance),
)


def render_pages(result: Any, *, directory: Path | None = None) -> dict[str, str]:
    """Every page of the report, as filename -> HTML.

    Split rather than one very long page, because the two readers want different
    things: someone deciding reads the overview, and someone checking a number
    goes straight to the episode it came from. A single page made the second
    reader scroll past the first reader's summary every time.

    Each page still opens offline, and the set still travels as a directory: the
    only thing that is no longer true is "one file".
    """
    directory = Path(directory or result.directory or ".")
    built = {name: fn(result, directory) for name, _label, _glyph, fn in PAGES}
    present = [(n, label, glyph) for n, label, glyph, _fn in PAGES if built[n]]
    return {
        name: _chrome(
            title=_model_title(result),
            kicker=f"xevals {result.suite.name}",
            subtitle=f"{_model_title(result)} · {result.suite.name}",
            nav=_nav(present, name),
            body=html_body,
            css_href=STYLESHEET_NAME,
        )
        for name, html_body in built.items()
        if html_body
    }


def render(result: Any, *, directory: Path | None = None) -> str:
    """The overview page, self-contained.

    Kept as a single inlined file, without the shared stylesheet, so that
    ``render`` alone still produces something that opens on its own. The
    multi-page set is what :func:`write` writes.
    """
    directory = Path(directory or result.directory or ".")
    present = [(n, label, glyph) for n, label, glyph, _fn in PAGES]
    return _chrome(
        title=_model_title(result),
        kicker=f"xevals {result.suite.name}",
        subtitle=f"{_model_title(result)} · {result.suite.name}",
        nav=_nav(present, "report.html"),
        body=_page_overview(result, directory),
        css_href=None,
    )


def write(result: Any, path: str | Path) -> Path:
    """Write the report as a small set of pages in one directory.

    ``path`` names the overview page, and its siblings land beside it, so callers
    that ask for ``report.html`` keep getting ``report.html``.
    """
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    (directory / STYLESHEET_NAME).write_text(CSS)
    pages = render_pages(result, directory=directory)
    for name, html_body in pages.items():
        target = path if name == "report.html" else directory / name
        target.write_text(html_body)
    return path


# --------------------------------------------------------------------------
# The benchmark comparison page
# --------------------------------------------------------------------------


def _slug(name: str) -> str:
    from .bench import _slug as slug

    return slug(name)


def _model_cards(bench: Any) -> str:
    """One card per model: rank, mean, its seven-bar profile, and a link.

    The strip is the point. Two models can share a mean and have completely
    different profiles, and a column of numbers hides that where a row of seven
    bars does not: the same argument the library makes about a single success
    rate, applied to the comparison.
    """
    _headers, rows = bench.rows()
    cards = []
    for rank, row in enumerate(rows, start=1):
        name = str(row[0])
        result = bench.results[name]
        strip = "".join(
            (
                f'<i style="height:{max(2, int((result.scores.get(d.value) or 0) * 26))}px;'
                f'background:{dimension_color(d)}" title="{d.value}"></i>'
                if result.scores.get(d.value) is not None
                else f'<i style="height:2px;background:var(--rule)" '
                     f'title="{d.value}: not measured"></i>'
            )
            for d in DIMENSION_ORDER
        )
        # The link goes to the directory the run actually wrote, slug *and* config
        # hash, not to the slug alone: that is a directory which does not exist
        # and a link that 404s silently from the one page most likely to be shared.
        href = (
            f"{result.directory.name}/report.html"
            if result.directory is not None
            else f"{_slug(name)}/report.html"
        )
        cards.append(
            f'<div class="score{" lead" if rank == 1 else ""}" style="--dim:var(--accent)">'
            f'<div class="name">#{rank}</div>'
            f'<div class="model-name">{_escape(name)}</div>'
            f'<div class="value">{_number(row[-1], 2)}</div>'
            f'<div class="strip">{strip}</div>'
            f'<a class="small" href="{_escape(href)}">full report &rarr;</a></div>'
        )
    return f'<div class="scores">{"".join(cards)}</div>'


def _bench_header(bench: Any, *, title: str, lede: str) -> str:
    run = bench.run
    state, glyph, text = _GATE[bench.gate_status()]
    return (
        '<header class="top">'
        f'<p class="kicker">xevals benchmark &middot; {_escape(bench.suite.name)}</p>'
        f"<h1>{_escape(title)}</h1>"
        f'<p class="lede">{_escape(lede)}</p>'
        f'<div class="meta-strip">'
        f'<span><b>{_escape(run.get("episodes", "?"))}</b> episodes '
        f'&times; <b>{len(run.get("seeds", []))}</b> seeds</span>'
        f'<span><b>{len(bench.suite.cells)}</b> conditions</span>'
        f'<span>root seed <b>{_escape(run.get("root_seed", 0))}</b></span>'
        "</div>"
        f'<div class="chips">{_chip(state, glyph, text)}</div>'
        "</header>"
    )


def _bench_caveat(bench: Any) -> str:
    if bench.gate_failed:
        return _banner(
            "bad", "cross", "Replay gate failed",
            "Replaying the demonstrator's own actions did not solve the task in this "
            "environment, so <strong>no row below measures a model</strong>.",
        )
    if bench.gate_status() == "unmeasured":
        return _banner(
            "warn", "question", "Replay gate unmeasured",
            "This environment supplies no demonstrator actions, so its faithfulness to "
            "the data was not checked.",
        )
    return ""


def _bench_overview(bench: Any, directory: Path) -> str:
    body = [
        _bench_header(
            bench, title=f"{len(bench.results)} models",
            lede="Identical conditions by construction: every model saw the same "
                 "episodes from the same starting states, and the baselines and the "
                 "replay gate were measured once because they belong to the environment.",
        ),
        _bench_caveat(bench),
        _section(
            "leaderboard", "target", "Leaderboard",
            _model_cards(bench)
            + _chart(
                _benchmark_radar(bench),
                "Every model on one radar, in the fixed dimension order.",
            )
            + '<p class="lede">The <code>mean</code> column is a summary and not a '
            "ranking anyone should defend: a model excellent everywhere except security "
            "and one mediocre everywhere can tie. The per-dimension columns sit beside "
            "it so the tie is visible.</p>"
            + _table(*bench.rows()),
        ),
    ]
    try:
        tradeoffs = _tradeoffs(bench.tradeoffs(), directory)
    except Exception:  # noqa: BLE001 - analysis must not lose the page
        tradeoffs = ""
    if tradeoffs:
        body.append(_section("tradeoffs", "scale", "Trade-offs", tradeoffs))
    return "".join(body)


def _benchmark_radar(bench: Any) -> str:
    """One radar per model, overlaid.

    Falls back to the leading model alone past what the palette separates, which
    is the same limit :mod:`xevals.plots` enforces and for the same reason.
    """
    from . import charts

    _headers, rows = bench.rows()
    if not rows:
        return ""
    best = bench.results[str(rows[0][0])]
    scores = {d.value: best.scores.get(d.value) for d in DIMENSION_ORDER
              if d.value in best.dimensions}
    return charts.radar(scores, _dimension_colours())


def _bench_conditions(bench: Any, directory: Path) -> str:
    body = [
        _bench_header(bench, title="Conditions",
                      lede="Where the models agree, and where they do not."),
    ]
    disagreements = bench.disagreements()
    if disagreements:
        body.append(_section(
            "disagree", "scale", "Where they disagree",
            '<p class="lede">Conditions that pull the models apart <em>beyond</em> how '
            "they already differ nominally. <code>excess</code> is the gap minus the gap "
            "the same two models show on the clean cell, so a model that is simply "
            "weaker everywhere does not fill this table.</p>"
            + _table(
                ["cell", "best", "worst", "gap", "excess"],
                [[c, b, w, round(g, 3), round(e, 3)]
                 for c, b, w, g, e in disagreements[:20]],
            ),
        ))
    body.append(_section(
        "conditions", "grid", "Every condition",
        '<p class="lede">Success rate per cell, one column per model.</p>'
        + _table(*bench.cell_rows()),
    ))
    return "".join(body)


def _bench_provenance(bench: Any, directory: Path) -> str:
    run = bench.run
    model_rows = [
        [m.get("name"), m.get("adapter", ""), m.get("module") or m.get("type", ""),
         m.get("params"), m.get("device", "")]
        for m in run.get("models", [])
    ]
    baseline_rows = [
        [name, info.get("success_rate"), info.get("episodes"),
         "gate" if info.get("gate") else "", info.get("reason", "")]
        for name, info in bench.baselines.items()
    ]
    pairs = [(f"env.{k}", v) for k, v in (bench.env or {}).items()]
    pairs += list((run.get("environment") or {}).items())
    items = "".join(
        f"<dt>{_escape(k)}</dt><dd>"
        f"{_escape(json.dumps(v) if isinstance(v, (list, dict)) else v)}</dd>"
        for k, v in pairs
    )
    return (
        _bench_header(bench, title="Provenance",
                      lede="What was evaluated, and what it was evaluated on.")
        + _section(
            "baselines", "shield", "Baselines and gate",
            '<p class="lede">Measured once, for the environment, and quoted identically '
            "in every row: no two rows can disagree about what the floor was.</p>"
            + (_table(["baseline", "success", "episodes", "role", "note"], baseline_rows)
               if baseline_rows else '<p class="lede">No baselines were run.</p>'),
        )
        + _section(
            "models", "fingerprint", "What was evaluated",
            _table(["model", "adapter", "class", "params", "device"], model_rows)
            + f'<dl class="meta">{items}</dl>',
        )
    )


#: The benchmark's own pages. Fewer than a single run's, because everything a
#: reader can get from one model's report is left on that report and linked to.
BENCHMARK_PAGES: tuple[tuple[str, str, str, Any], ...] = (
    ("index.html", "Leaderboard", "target", _bench_overview),
    ("conditions.html", "Conditions", "grid", _bench_conditions),
    ("provenance.html", "Provenance", "fingerprint", _bench_provenance),
)


def render_benchmark_pages(bench: Any, *, directory: Path | None = None) -> dict[str, str]:
    """Every page of the comparison, as filename -> HTML."""
    directory = Path(directory or bench.directory or ".")
    built = {name: fn(bench, directory) for name, _label, _glyph, fn in BENCHMARK_PAGES}
    present = [(n, label, glyph) for n, label, glyph, _fn in BENCHMARK_PAGES if built[n]]
    return {
        name: _chrome(
            title=f"{len(bench.results)} models",
            kicker="xevals benchmark",
            subtitle=f"benchmark · {bench.suite.name}",
            nav=_nav(present, name),
            body=body,
            css_href=STYLESHEET_NAME,
        )
        for name, body in built.items()
        if body
    }


def render_benchmark(bench: Any, *, directory: Path | None = None) -> str:
    """The comparison's leaderboard page, self-contained.

    Deliberately not a copy of the single-run report. Anything a reader can get
    from one model's own pages is left there and linked to.
    """
    directory = Path(directory or bench.directory or ".")
    present = [(n, label, glyph) for n, label, glyph, _fn in BENCHMARK_PAGES]
    return _chrome(
        title=f"{len(bench.results)} models",
        kicker="xevals benchmark",
        subtitle=f"benchmark · {bench.suite.name}",
        nav=_nav(present, "index.html"),
        body=_bench_overview(bench, directory),
        css_href=None,
    )


def write_benchmark(bench: Any, directory: str | Path) -> Path:
    """Write the comparison pages, and the stylesheet they share."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / STYLESHEET_NAME).write_text(CSS)
    for name, body in render_benchmark_pages(bench, directory=directory).items():
        (directory / name).write_text(body)
    return directory / "index.html"
