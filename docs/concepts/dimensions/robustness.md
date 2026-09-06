# Robustness

> **Does it keep doing it under nuisance change?**

Change that *occurs*: the camera gets knocked, the light changes, the network is
slow, someone leaves a mug on the table. The right summary is an **average over a
severity ladder**, because a deployment meets these conditions at their typical
strength rather than at their worst.

Change that is *chosen to hurt* belongs to [security](security.md) instead, and
the split is the most consequential one in the library. A model can be robust to
average noise and be redirected by one sentence painted on a box.

Every robustness metric is a **ratio or a curve**, never an absolute. That is
what makes them comparable between models with different clean success rates, and
it is why every suite starts with a clean cell.

| Metric | Direction | Unit | Measured on |
|---|---|---|---|
| [`retention`](#robustnessretention) | higher | ratio | each perturbed cell, paired with clean |
| [`severity_auc`](#robustnessseverity_auc) | higher | ratio | the ladder, attached to the clean cell |
| [`worst_case`](#robustnessworst_case) | higher | fraction | every cell, attached to the clean cell |

---

## `robustness/retention`

**Definition.** Perturbed success as a fraction of clean success, **paired on
seeds**. The headline robustness number.

$$
\rho_p \;=\; \min\!\left(1,\;
  \frac{\mathbb{E}_{s \in S}\,\mathbf{1}\!\left[\,\mathrm{succ}(p, s)\,\right]}
       {\mathbb{E}_{s \in S}\,\mathbf{1}\!\left[\,\mathrm{succ}(\varnothing, s)\,\right]}
\right)
$$

where $S$ is the set of seeds run in **both** the perturbed cell $p$ and the
clean cell $\varnothing$. Since episode seeds derive from
`(root_seed, seed, index)` and never from the cell, $S$ is every seed in the run.

**Target.** $1.0$, meaning the perturbation cost nothing. Used as-is in the
dimension score.

**Example.** Ten seeds. Clean succeeds on all ten; under `visual/blur@0.6` it
succeeds on six:

```
retention = min(1, 0.6 / 1.0) = 0.60
```

Now the same model at a clean rate of $0.5$, dropping to $0.3$:

```
retention = min(1, 0.3 / 0.5) = 0.60
```

Both lost the same *proportion*, which is the question robustness asks. A raw
difference would have called them $0.4$ and $0.2$ apart and made the weaker model
look more robust.

!!! warning "Undefined, not perfect, when nothing succeeds"
    If the clean condition never succeeds, retention is $0/0$. `xevals` reports
    `null` with that reason. Returning $1.0$ is how a completely broken policy
    ends up at the top of a robustness table, and it happens more often than it
    should.

!!! note "Capped at 1.0"
    A perturbation that *helps* is interesting and is visible in the raw success
    rates. Letting retention exceed $1$ would let one lucky cell hide a failure
    elsewhere in the mean.

The confidence interval bootstraps the **paired** ratio, resampling seeds rather
than the two rates independently. Pairing is what makes the comparison sensitive:
it resolves a six-point difference with twenty episodes where pooled averages need
roughly two hundred.

---

## `robustness/severity_auc`

**Definition.** Area under the success-against-severity curve, normalised to
$[0, 1]$ by the width of the ladder. One number for a whole family.

$$
\mathrm{AUC}_f \;=\; \frac{1}{\sigma_{\max} - \sigma_{\min}}
  \int_{\sigma_{\min}}^{\sigma_{\max}} \! u_f(\sigma)\, \mathrm{d}\sigma
\;\approx\; \frac{\sum_{j} \tfrac{1}{2}\bigl(u_j + u_{j+1}\bigr)\bigl(\sigma_{j+1} - \sigma_j\bigr)}
                 {\sigma_{\max} - \sigma_{\min}}
$$

with $u_f(\sigma)$ the success rate of family $f$ at severity $\sigma$,
trapezoidally integrated over the ladder $(0, 0.2, 0.4, 0.6, 0.8, 1.0)$. The
curve is **anchored at severity 0 with the clean cell's own success rate**, so it
starts from the model's baseline rather than from an assumed $1.0$.

**Target.** $1.0$. A per-family value is recorded as
`robustness/severity_auc[visual/blur]` for the table, and the pooled mean across
families is what enters the dimension score. The bracketed per-family entries are
deliberately excluded from the score: counting both would weight the dimension by
how many families a suite happened to run.

**Example.** Two models, same worst case, different shapes:

| severity | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 | AUC |
|---|---|---|---|---|---|---|---|
| cliff | 1.00 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | **0.15** |
| decay | 1.00 | 1.00 | 0.80 | 0.45 | 0.15 | 0.05 | **0.58** |

Both bottom out at $0.05$. The first has no usable operating range at all; the
second works until severity $0.4$. A single-severity evaluation at $1.0$ would
have called them identical, and one at $0.2$ would have called the first broken
and the second perfect.

---

## `robustness/worst_case`

**Definition.** The lowest success rate over every cell in the suite.

$$
\mathrm{worst\_case} \;=\; \min_{c \,\in\, \mathcal{C}} \; u_c
$$

**Target.** $1.0$. Used as-is.

**Example.** A suite whose cells score
$\{1.00, 1.00, 0.93, 0.67, 0.20, 0.13, 0.03\}$:

```
mean over cells = 0.57
worst_case      = 0.03
```

The mean describes a model's average day. A robot meets its worst one, and the
gap between $0.57$ and $0.03$ is the whole reason both are reported.
