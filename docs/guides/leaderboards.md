# Leaderboards

One run is a fact about one model. The comparison is what anyone actually wants.

!!! tip "See a real one"
    [What a run looks like](example-output.md) has a leaderboard over three
    policies, and the comparison page that goes with it.

This page is about comparing runs that already exist: different days,
different machines, different suites. If you are comparing models you can run
*now*, use a [benchmark](benchmarks.md) instead: it pairs them on seeds, which
these after-the-fact comparisons cannot do.

```bash
xevals compare runs/*/ --out leaderboard/     # table + overlaid radar
xevals compare runs/a runs/b --diff           # metric by metric
```

```python
from xevals.results import leaderboard, compare, markdown_table

print(markdown_table(*leaderboard(["runs/a", "runs/b", "runs/c"])))
print(markdown_table(*compare("runs/a", "runs/b")))
```

## Read the mean column with suspicion

It is a summary, not a ranking anyone should defend. A model that is excellent
everywhere except security and one that is mediocre everywhere can tie. The
per-dimension columns sit beside it exactly so that the tie is visible, and the
`gate` column sits in front of both so that a leaderboard row from an unfaithful
environment is not read as a result.

## Comparability

Two runs are comparable when they share a suite, an environment and an episode
count. `xevals` does not enforce that (comparing a 5-episode probe with a
200-episode evaluation is sometimes exactly what you want), but every row carries
its `n`, and the diff will not call a difference significant that the error bars
do not support.

Because episode seeds depend only on `(root_seed, seed, index)` and never on which
cells a suite contains, **two runs with different suites remain comparable on the
cells they share.** Adding a cell does not renumber anything.

## Diffs

A difference is flagged significant when the two 95 % bootstrap intervals do not
overlap. Blunt, and the right blunt test here: it never calls a difference
significant that a reader could not see in the error bars, which is the failure
mode a comparison table has to avoid.

Rows are sorted significant-first, then by the change in the metric's
**normalised** value, not its raw one. Raw deltas are not comparable
across metrics: an undiscounted return moving by 40 in whatever units the task
chose would otherwise outrank a success rate collapsing from 0.9 to 0.0 and take
every row at the top of the table. Both are shown, so the ordering is legible.

## An overlaid radar

```python
from xevals.plots import leaderboard_radar
from xevals import Result

leaderboard_radar([Result.load(p) for p in paths], "radar.png")
```

Capped at five runs, and two is where it reads best: the brand accent pair
separates two polygons better than any five-way palette separates five.
