# Suites

A **cell** is one measurement condition: a perturbation at a severity, on a split,
scored by a set of metrics. A **suite** is a list of cells plus the baselines to
run beside them.

## Built-ins

| Suite | Cells | Dimensions | For |
|---|---|---|---|
| `smoke` | 2 | 3 | a wiring check, not an evaluation |
| `core` | 1 | accuracy, safety, efficiency | the cheap default |
| `robustness` | 31 | accuracy, robustness | severity ladders over six families |
| `safety` | 5 | safety, accuracy | limits, clean and under disturbance |
| `security` | 7 | security, accuracy | patches, pixel noise, injection |
| `generalization` | 4 | generalization, accuracy | unseen objects, layouts, wordings |
| `consistency` | 4 | consistency, accuracy | seed spread, paraphrase, determinism |
| `full` | 43 | all seven | the complete evaluation |

```bash
xevals suites list
xevals suites describe robustness
```

The `full` suite is *assembled* from the specialised ones rather than written out,
so a cell added to `robustness` reaches `full` automatically and the two cannot
drift apart.

## Three rules the built-ins follow

**Every suite starts with a clean cell**: not for its own sake, but because
every paired metric divides by it. A robustness suite without a clean baseline
reports ratios against nothing, and `SuiteSpec` refuses to be constructed without
one.

**A family is run as a ladder, not at one severity.** One severity gives one number
and no shape, and the shape is where models differ: a policy flat to 0.6 and then
falling off is a different proposition from one decaying from the start, and both
can share a mean.

**The safety suite includes perturbed cells on purpose.** A policy that stays inside
the workspace when everything is nominal and leaves it the moment the camera moves
is not a safe policy, and a clean-only safety suite would call it one.

## Writing your own

In Python:

```python
from xevals.suites import Cell, SuiteSpec
from xevals import Dimension

suite = SuiteSpec(
    name="camera-only",
    cells=(
        Cell("clean"),
        Cell("shift-mild",   "visual/camera_shift", 0.3),
        Cell("shift-strong", "visual/camera_shift", 0.9),
    ),
    dimensions=(Dimension.ACCURACY, Dimension.ROBUSTNESS),
    baselines=("random", "noop", "replay"),
)
```

Or inline in a config, which is the right shape when the conditions *are* the
contribution: the file that produced the numbers then sits beside them in
the run directory:

```toml
[suite]
name = "camera-only"
dimensions = ["accuracy", "robustness"]

[[suite.cells]]
name = "clean"

[[suite.cells]]
name = "shift-strong"
perturbation = "visual/camera_shift"
severity = 0.9
metrics = ["accuracy/success_rate", "robustness/retention"]
```

An unknown key, or an unknown metric name, is an **error** naming the valid ones.
A silently ignored typo in a suite file skips a whole dimension and reports it as
a confident blank.

## Resolution

`SuiteSpec.resolved()` runs before every evaluation and does two things:

**It expands the default metric list**, so `run.json` records the metrics actually
used rather than a wildcard that will mean something else next release.

**It filters that list to the cells each metric belongs in.** Without this, a suite
asking for "every metric of every dimension I claim" computes paraphrase agreement
on an occlusion cell and injection compliance on a blur cell, and those
numbers are not merely uninformative, they are averaged into the dimension score,
so a model's language consistency ends up measured mostly by how much occlusion
changed its actions.

An **explicit** `metrics` list on a cell is the author's choice and is kept
verbatim.

::: xevals.suites
    options:
      heading_level: 2
      members: false
