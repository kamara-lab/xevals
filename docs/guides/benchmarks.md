# Benchmarks

One evaluation is a fact about one model. Almost every question anyone actually
asks is a comparison. Is the new checkpoint better? Does the bigger model buy
robustness or only accuracy? Which of these three is safe enough to deploy?

```python
import xevals

bench = xevals.benchmark(
    {"baseline": old_policy, "candidate": new_policy, "ablation": stripped},
    "synthetic/reach",
    suite="robustness", episodes=20, seeds=range(3),
    out="runs/", name="checkpoint-42",
)
print(bench.table())
bench.disagreements()
```

or from a config, where the only difference from a single run is `[[models]]`
instead of `[model]`:

```toml
[[models]]
name = "baseline"
target = "mypkg.policies:baseline"

[[models]]
name = "candidate"
target = "mypkg.policies:candidate"

[suite]
name = "robustness"
```

```bash
xevals run configs/benchmark.toml
```

!!! tip "See the output"
    [What a run looks like](example-output.md) embeds the comparison page a
    benchmark writes, over three policies.

## Why not just run `evaluate` three times

Three runs and a spreadsheet look like a benchmark. Three things are different
here, and each of them changes what the numbers mean.

**Every model sees the same episodes.** Seeds derive from
`(root_seed, seed, index)` and never from the model, so model A's episode 7 and
model B's episode 7 start from the same state under the same perturbation. That
turns a comparison of two averages into a *paired* comparison, which resolves a
difference with roughly a tenth of the episodes.

**The baselines and the gate run once.** `random`, `noop` and the replay gate are
properties of the environment, not of any model. Running them per model wastes
the time and, worse, lets three rows of one leaderboard disagree about what the
floor was. Here they are measured once and quoted identically in
every row, and a failed gate invalidates the whole benchmark rather than one row
of it.

**The comparison is written down.** What gets shared is the comparison, not three
files and a claim about them:

```
runs/checkpoint-42/
  baseline-<hash>/     one full run directory per model
  candidate-<hash>/
  ablation-<hash>/
  leaderboard.{md,tex,csv,json}
  cells.{md,tex,csv,json}      one row per condition, one column per model
  radar.png                    every model on one radar
  index.html                   the comparison, linking to each model's report
```

Models are **not** run concurrently, and that is deliberate: two models sharing a
GPU report each other's contention as their own latency, and the efficiency
dimension would become a measurement of the benchmark harness.

## Reading the output

### The leaderboard is the least useful table

```
| model     | accuracy | robustness | mean   |
| --------- | -------- | ---------- | ------ |
| state     | 0.8590   | 0.9014     | 0.8802 |
| image     | 0.8539   | 0.4250     | 0.6394 |
```

The `mean` column is a summary and not a ranking anyone should defend: a model
excellent everywhere except security and one mediocre everywhere can tie. The
per-dimension columns sit beside it so the tie is visible, and the gate status
sits in front of both so a row from an unfaithful environment is not read as a
result.

### `disagreements()` is the useful one

```python
for cell, best, worst, gap, excess in bench.disagreements():
    print(cell, best, "over", worst, gap, excess)
```

```
visual/gaussian_noise@0.8   state over image   gap 1.00   excess 1.00
visual/brightness@0.2       state over image   gap 0.90   excess 0.90
```

`excess` is the gap **minus** the gap the same two models already show on the
clean cell, and ranking by it is the whole point. A model that is simply weaker
is weaker in every cell, so ranking by the raw gap returns the entire suite in
arbitrary order and buries the one condition that matters. The excess answers the
question a benchmark is run to answer: *given* that these models perform as they
do nominally, which condition pulls them apart further?

A uniformly weaker model produces an excess near zero everywhere and correctly
drops out of this list. It is already visible in the leaderboard, which is where
a uniform difference belongs.

### The per-cell table

```python
print(bench.table(kind="cells"))
```

One row per condition, one column per model. Two models with the same robustness
score routinely fail under *different* perturbations, and this is the only view
that shows it.

## Budgets are shared

```python
xevals.benchmark(models, env, budget=600)   # ten minutes, across all models
```

The cap counts episodes and seconds across the whole benchmark, not per model. A
benchmark that spends its entire allowance on the first model has not produced a
comparison, so `Budget.start()` is idempotent and the second model inherits a
clock that is already running.

## Reading a benchmark back

```python
from xevals import Benchmark

bench = Benchmark.load("runs/checkpoint-42")
bench.best("safety")
bench.table(kind="cells", format="csv")
bench.report()          # rebuild index.html
```

As with a single run, this needs no model and no environment, which is what
makes an archived benchmark worth keeping.

::: xevals.bench
    options:
      heading_level: 2
      members: false
