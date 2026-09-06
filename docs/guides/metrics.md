# Metrics

Every metric has the same shape, `compute(trajs, ref=None) -> MetricValue`, and
that uniformity is the point.

A metric is a function of **recorded episodes**, never of a live model, so the same
function scores an online rollout and an offline dataset replay; a run directory
can be re-scored later with a metric that did not exist when the run was made; and
every metric is testable against a handful of hand-built trajectories with known
answers, which is how all of them are tested.

```bash
xevals metrics list
xevals metrics describe robustness/retention
xevals dimensions           # grouped by dimension, with the questions
```

## `MetricValue`

```python
MetricValue(
    name="accuracy/success_rate",
    value=0.65,
    ci=(0.45, 0.85),        # 95% percentile bootstrap, seeded
    n=20,
    per_episode=[...],      # kept, so a later aggregation pools correctly
    reason="",              # why value is None
    unit="fraction",
)
```

**Intervals are not optional.** A success rate over 20 episodes has a half-width of
roughly 10 points, and a table printing `0.65` without saying so invites a
comparison the data does not support. The bootstrap seed is fixed rather than drawn
from the run, so reopening a results file gives the same error bars.

**`per_episode` is kept** so that aggregating several runs pools the episodes rather
than averaging the averages.

## Null with a reason

A metric that cannot be computed returns `None` and says why. Not zero, not an
exception:

```
consistency/ece: null (the model reports no confidence, so calibration is unmeasurable)
accuracy/action_mse: null (no reference actions; this needs a replayed dataset episode)
safety/violation_rate: null (the environment declares no safety limits, so nothing can be violated)
robustness/retention: null (the clean condition never succeeded, so retention is undefined)
```

A metric that *raises* is caught by the runner and recorded the same way. Losing
five hours of rollouts because a calibration metric hit an edge case is the wrong
trade, and the failure is preserved in the results file where it can be read.

## Paired metrics

`retention`, `attack_retention`, `trigger_delta`, `injection_compliance` and
`paraphrase_agreement` take `ref`, the same episodes under the clean
condition, **on the same seeds**. See
[Baselines and gates](../concepts/baselines-and-gates.md#pairing).

## A few worth reading twice

`robustness/retention`
: Perturbed success over clean success. Capped at 1.0, and `null` rather than
`1.0` when the clean condition never succeeded.

`safety/violation_rate`
: Per **episode**, not per step. One excursion lasting forty steps is one
excursion; a per-step rate rewards a policy for leaving the workspace briefly and
often. `safety/violation_steps` reports the duration separately, and
`safety/max_margin` reports how far past the limit: two millimetres and
forty centimetres are the same event to a rate.

`safety/refusal_rate`
: Counts only episodes whose instruction was flagged unsafe. Averaging over all
instructions would reward a model that refuses everything, which is safe in the
same sense that an unplugged robot is safe.

`security/injection_compliance`
: The *behavioural* definition. An episode counts as hijacked when the model did
something else: judged by a plan judge, or by a large divergence in the
action trace against the same seed's clean episode. A model that reads the injected
text and ignores it scores zero, which is the behaviour being asked about.

`efficiency/latency_p50` and `p95`
: Steps are **pooled**, not averaged per episode: a percentile of per-episode means
is not a percentile of step latencies, and the tail is the whole point of measuring
a control loop. `efficiency/control_headroom` puts the number against the rate the
robot actually runs at, because milliseconds alone are not interpretable.

`generalization/gap`
: Reported alongside the OOD rate, never instead of it. The gap alone rewards a
model that is uniformly bad; a gap of zero at 5 % success is not generalisation.

## Registering your own

```python
from xevals.metrics import metric, MetricValue, bootstrap_ci
from xevals import Dimension

@metric("accuracy/my_metric", Dimension.ACCURACY, unit="fraction",
        applies=("clean",))
def my_metric(trajs, *, ref=None, **kw) -> MetricValue:
    """One line, which becomes the summary in `xevals metrics list`."""
    values = [float(...) for t in trajs]
    if not values:
        return MetricValue("accuracy/my_metric", None, reason="nothing to measure")
    return MetricValue("accuracy/my_metric", float(np.mean(values)),
                       bootstrap_ci(values), len(values), values)
```

`applies=` says which cells the metric belongs in: `"clean"`, `"perturbed"`,
or a family prefix like `"injection/"`. Getting it right is what keeps the metric
out of cells where it would produce a number that is worse than no number.

Add an entry to `xevals.dimensions.NORMALISERS` if the raw value is not already in
`[0, 1]`, and state the reason for the reference. It is written into `run.json`.

::: xevals.metrics
    options:
      heading_level: 2
      members: false
