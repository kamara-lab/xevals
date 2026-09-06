# Generalisation

> **Does it transfer to unseen objects, scenes, instructions?**

Computed **across cells** rather than inside one, and attached to the clean cell,
because a gap needs both sides of it. The runner pools every *unperturbed* cell
for this: measuring a generalisation gap against a blurred out-of-distribution
cell would confound two dimensions, and the resulting number would be attributed
to whichever heading it was filed under.

The splits each change exactly one thing, so a drop is attributable:

| Split | What changes |
|---|---|
| `in` | nothing |
| `ood/object` | an object the training distribution never contained |
| `ood/layout` | spawn geometry the model has not seen |
| `ood/instruction` | a wording template never used, same task |

| Metric | Direction | Unit |
|---|---|---|
| [`ood_success`](#generalizationood_success) | higher | fraction |
| [`gap`](#generalizationgap) | lower | fraction |
| [`per_split`](#generalizationper_split) | higher | fraction |

---

## `generalization/ood_success`

**Definition.** Success on the out-of-distribution splits only.

$$
u_{\mathrm{ood}} \;=\; \frac{1}{\lvert \mathcal{D}_{\mathrm{ood}} \rvert}
  \sum_{i \,\in\, \mathcal{D}_{\mathrm{ood}}} \mathbf{1}\!\left[\,\mathrm{succ}_i\,\right],
\qquad \mathcal{D}_{\mathrm{ood}} = \{\, i : \mathrm{split}_i \neq \texttt{in} \,\}
$$

**Target.** $1.0$, used as-is.

**Example.** Thirty out-of-distribution episodes, twenty-two successful:

```
ood_success = 22 / 30 = 0.73
```

---

## `generalization/gap`

**Definition.** In-distribution success minus out-of-distribution success.

$$
\mathrm{gap} \;=\; u_{\mathrm{in}} \;-\; u_{\mathrm{ood}}
$$

**Target.** $0$, normalised as $1 - \mathrm{gap}$.

**Example.** Two models:

| model | in | ood | gap |
|---|---|---|---|
| A | 0.95 | 0.60 | **0.35** |
| B | 0.05 | 0.05 | **0.00** |

!!! warning "The gap alone rewards a model that is uniformly bad"
    Model B has a perfect gap and does nothing. That is why `xevals` reports the
    gap **alongside** `ood_success` and never instead of it: a gap of zero at 5 %
    success is not generalisation.

---

## `generalization/per_split`

**Definition.** The mean success across splits, weighting each split **equally**
rather than each episode.

$$
u_{\mathrm{split}} \;=\; \frac{1}{\lvert \mathcal{S} \rvert} \sum_{s \,\in\, \mathcal{S}}
  \left( \frac{1}{\lvert \mathcal{D}_s \rvert} \sum_{i \,\in\, \mathcal{D}_s}
    \mathbf{1}\!\left[\,\mathrm{succ}_i\,\right] \right)
$$

**Target.** $1.0$, used as-is. Requires at least two splits; with fewer it
reports `null` with that reason.

**Example.** A suite that runs 60 in-distribution episodes and 10 of each
out-of-distribution split:

| split | episodes | success |
|---|---|---|
| `in` | 60 | 1.00 |
| `ood/object` | 10 | 0.40 |
| `ood/layout` | 10 | 0.30 |
| `ood/instruction` | 10 | 0.20 |

```
pooled over episodes = (60×1.00 + 10×0.40 + 10×0.30 + 10×0.20) / 90 = 0.77
per_split            = (1.00 + 0.40 + 0.30 + 0.20) / 4              = 0.48
```

The pooled figure is mostly a measurement of the in-distribution split, because
that is where two-thirds of the episodes are. Equal weight per split is what stops
a suite's episode budget from deciding what the generalisation number means.
