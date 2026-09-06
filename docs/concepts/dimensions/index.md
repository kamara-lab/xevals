# Dimensions

Seven, in a fixed order, each answering one question.

| # | Dimension | Question | Metrics |
|---|---|---|---|
| 1 | [`accuracy`](accuracy.md) | Does it do the task? | 9 |
| 2 | [`robustness`](robustness.md) | Does it keep doing it under nuisance change? | 3 |
| 3 | [`safety`](safety.md) | Does it stay inside physical and behavioural limits? | 6 |
| 4 | [`security`](security.md) | Can it be hijacked, or made to fail on purpose? | 4 |
| 5 | [`efficiency`](efficiency.md) | What does it cost to run? | 5 |
| 6 | [`generalization`](generalization.md) | Does it transfer to unseen objects, scenes, instructions? | 3 |
| 7 | [`consistency`](consistency.md) | Are its answers stable? | 5 |

Each page states, for every metric it owns: what it means, the equation, the
value to aim for and how it is normalised, and a worked example small enough to
check by hand.

The order is fixed and load-bearing. It fixes the column order of every table and
the colour of every dimension in every figure, so a reader who has learnt the
palette once can read any chart without the legend.

## Robustness and security are not the same dimension

They look alike. Both are "success under a changed input, relative to clean",
and keeping them apart is the most consequential split in the library.

**Robustness** asks what happens under change that *occurs*: the camera gets
knocked, the light changes, the network is slow. The right summary is an average
over a severity ladder, because a deployment meets these conditions at their
typical strength.

**Security** asks what happens under change that is *chosen to hurt*: a patch, a
sticker, a sentence. The right summary is the **worst** case, because an attacker
picks the worst case and averaging in the attacks that failed describes nobody's
threat model.

A model can be robust to average noise and be redirected by one sentence painted
on a box. Averaging those into one number hides exactly the failure a deployment
cares about.

## How a dimension is scored

A `DimensionScore` is the mean of its metrics' **normalised** values in `[0, 1]`.
Normalisation is the part that usually goes unstated and therefore
unreproducible, so in `xevals` it is data rather than code: `NORMALISERS` maps a
metric name to one of three shapes, and the resolved table is written into
`run.json` so any reader can recompute the aggregate, or disagree with it.

| Shape | For | Mapping |
|---|---|---|
| `unit` | already in `[0, 1]`, higher better (success rate) | identity |
| `inverse_unit` | in `[0, 1]`, lower better (violation rate) | `1 - x` |
| `scale` | open-ended physical quantity (latency, action MSE) | `exp(-x / reference)` |

The `scale` reference is the value scoring `1/e`. Every one of them is stated
with the reason it was chosen: 100 ms because that is 10 Hz control, the
floor for most manipulators; 0.05 action MSE because actions are normalised to
`[-1, 1]` and that is 22 % of full scale. They are the library's only magic
numbers and they are all in one place.

```bash
xevals dimensions --json     # the questions, the metrics, and every normaliser
```

## Aggregating across cells

A metric measured in twelve cells has to become one number, and the rule differs
by dimension because the choice changes what the dimension *means*. Averaging
everything (the obvious default) is wrong for four of the seven.

| Dimension | Rule | Why |
|---|---|---|
| `accuracy` | the clean cell | "Does it do the task?" is about the nominal condition; averaging in the perturbed cells makes accuracy a second, worse robustness score |
| `robustness` | mean | a claim about behaviour across conditions |
| `safety` | **worst** | a policy that stays inside the workspace on average and leaves it whenever the camera moves is not safe |
| `security` | **worst** | an attack that works is not averaged away by ones that do not |
| `efficiency` | mean | it barely varies between conditions |
| `generalization` | the clean cell | it is computed across cells and attached there |
| `consistency` | the clean cell | same |

## Unmeasured is not zero

A dimension with no measurable metric scores `None`. On the radar it is a hollow
marker on the axis rather than a point at the origin; in the table it is `--`; in
the report it is a card saying *not measured* with the reason.

A radar that draws "unmeasured" and "measured as zero" identically is the single
most misleading chart this library could produce, so it does not.

::: xevals.dimensions
    options:
      heading_level: 2
      members: false
