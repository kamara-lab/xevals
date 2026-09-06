# Results schema

A results file that cannot be re-run from its own contents is a screenshot.

## The run directory

```
runs/<name>-<config hash>/
  run.json          SCHEMA 1 -- everything, full precision
  config.json       the resolved config
  config.toml       the source file, verbatim, when there was one
  table.{md,tex,csv,json}      dimension scores
  metrics.{md,tex,csv,json}    every metric, with intervals and n
  cells.{md,tex,csv,json}      one row per condition
  figures/          radar.png dimension-bars.png severity.png latency.png heatmap.png
  videos/<cell>/    a few episodes, with a HUD
  trajectories/     optional: --save-trajectories
  report.html       self-contained; opens offline
```

The directory name's hash covers the suite, the fingerprints, the seeds and the
episode count, and deliberately **not** the output path: two runs that differ only
in where they were written are the same experiment and should not look like two.

## What `run.json` carries

| Key | Contents |
|---|---|
| `schema` | version, checked on load |
| `suite` | the **resolved** suite, every cell with its exact metric list |
| `model` | adapter, module, parameter count, device, capabilities |
| `environment` | type, action dim, whether it declares limits, whether `reset(state=...)` round-trips |
| `run` | seeds, episodes, horizon, root seed, budget, partial flag, library versions |
| `gate` | `passed` / `failed` / `unmeasured`, and the replay success rate |
| `baselines` | each baseline's success rate on the same seeds |
| `dimensions` | score, raw metrics, normalised contributions, and what was skipped with why |
| `normalisation` | the full normaliser table, as data |
| `cells` | every metric: value, 95 % interval, `n`, unit, reason, per-episode values |

Two of those are unusual and both are deliberate.

**The metric list is resolved, not a wildcard.** A suite file that says "every
robustness metric" means something different next release, and the saved run would
not say which it meant. So `SuiteSpec.resolved()` expands it before the run and
the expansion is what gets written.

**The normalisation is in the file.** Dimension scores can therefore be
recomputed (or argued with) from `run.json` alone, without knowing which
version of `xevals` produced it.

## Reading a run back

```python
from xevals import Result

result = Result.load("runs/synthetic-abc123def456")
result.scores                          # dimension -> score
result.metric("robustness/retention", "visual/camera_shift@0.6")
result.table(kind="cells", format="csv")
result.report()                        # rebuild report.html
```

Loading a file whose `schema` is newer than this `xevals` understands raises,
rather than reading it with today's assumptions and producing numbers that look
fine and mean something else.

A run directory can be re-scored, re-plotted and re-reported **without the model
present**, which is the property that makes an archive worth keeping.

## Comparing

```bash
xevals compare runs/*/ --out leaderboard/     # a table plus an overlaid radar
xevals compare runs/a runs/b --diff           # metric by metric
```

The leaderboard's `mean` column is a summary and not a ranking anyone should
defend: a model excellent everywhere except security and one mediocre everywhere
can tie. The per-dimension columns sit beside it so the tie is visible.

The diff flags a difference as significant when the two 95 % bootstrap intervals
do not overlap. That is a blunt test, and it is the right blunt test here: it
never calls a difference significant that a reader could not see in the error bars.

Rows are ordered by the change in the metric's **normalised** value, because raw
deltas are not comparable across metrics measured in different units.

::: xevals.results
    options:
      heading_level: 2
      members: false
