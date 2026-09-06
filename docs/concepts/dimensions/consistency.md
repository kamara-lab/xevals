# Consistency

> **Are its answers stable?**

Taken from the clean cell, and pooled only over cells that are the *same
condition*: the clean cell and its deliberate repeat. Pooling the generalisation
splits in would compare two episodes that share a seed but not a world, and would
report a perfectly deterministic model as non-deterministic because the layout
differed.

This is the dimension to read **first** when comparing two runs. A model that is
not deterministic at a fixed seed cannot be evaluated reproducibly by anyone,
including its own authors, so `xevals` measures that rather than assuming it away.

| Metric | Direction | Unit | Needs |
|---|---|---|---|
| [`determinism`](#consistencydeterminism) | higher | fraction | a repeated seed |
| [`seed_std`](#consistencyseed_std) | lower | fraction | 3+ seeds |
| [`paraphrase_agreement`](#consistencyparaphrase_agreement) | higher | fraction | `instruction/` cells |
| [`ece`](#consistencyece) | lower | fraction | `confidence()` |
| [`brier`](#consistencybrier) | lower | score | `confidence()` |

---

## `consistency/determinism`

**Definition.** The fraction of repeated seeds that reproduced their episode
exactly, comparing action traces.

$$
\mathrm{determinism} \;=\; \frac{1}{\lvert R \rvert} \sum_{s \,\in\, R}
  \mathbf{1}\!\left[\; \forall\, k :\; \bigl\lVert A^{(s,k)} - A^{(s,1)} \bigr\rVert_\infty < 10^{-6} \;\right]
$$

where $R$ is the set of seeds run more than once and $A^{(s,k)}$ is the $k$-th
run's action trace at seed $s$. Shapes must match too: an episode of a different
length is not a reproduction.

**Target.** $1.0$, used as-is. Reports `null` when no seed was run twice, which is
what the built-in `consistency` suite's `repeat` cell exists to prevent.

**Example.** Four seeds run twice each. Three reproduce; one diverges at step 12:

```
determinism = 3 / 4 = 0.75
```

A score below $1.0$ invalidates any comparison of two runs of this model before
it invalidates anything else, which is why it is worth reading before the
leaderboard.

---

## `consistency/seed_std`

**Definition.** The standard deviation of the per-seed success rate.

$$
\sigma \;=\; \sqrt{\frac{1}{\lvert S \rvert}\sum_{s \in S}\bigl(u_s - \bar u\bigr)^2},
\qquad u_s = \frac{1}{n_s}\sum_{i \,\in\, \mathcal{D}_s} \mathbf{1}\!\left[\,\mathrm{succ}_i\,\right]
$$

**Target.** $0$. Normaliser $\exp(-\sigma/0.1)$: ten points of success-rate spread
across seeds is a lot.

**Example.** Three seeds scoring $0.90$, $0.85$ and $0.65$:

```
mean     = 0.80
seed_std = 0.108   →   score exp(-0.108/0.1) = 0.339
```

!!! note "Three seeds minimum, enforced"
    With fewer, this reports `null` with the reason "2 seeds; a spread needs at
    least three" rather than presenting the spread of two numbers as a
    distribution.

---

## `consistency/paraphrase_agreement`

**Definition.** The fraction of steps whose action agrees between an instruction
and its paraphrase, on shared seeds.

$$
\mathrm{agreement} \;=\; \frac{1}{\lvert S \rvert}\sum_{s \in S}
  \frac{1}{T_s}\sum_{t=0}^{T_s-1}
    \mathbf{1}\!\left[\; \bigl\lVert a^{(\varnothing,s)}_t - a^{(p,s)}_t \bigr\rVert_2 < 0.1 \;\right]
$$

with the tolerance $0.1$ being a tenth of the $[-1, 1]$ action range.

**Target.** $1.0$, used as-is.

**Why agreement and not success.** Two runs can both succeed while the model has
understood the paraphrase differently, and two runs can both fail identically,
which is consistent even though it is not good. Success rate cannot separate
those; action agreement can.

**Example.** "push the red cube" against "shove the red cube, thanks". Over 40
steps on 3 seeds, 34, 40 and 31 steps agree:

```
paraphrase_agreement = (34/40 + 40/40 + 31/40) / 3 = 0.875
```

!!! note "The object noun is preserved by construction"
    The bundled paraphrases rewrite the verb phrase by regex and never touch the
    object. A drop here is therefore a failure of *language robustness* rather
    than of grounding, and a free-form rewrite would have merged those two very
    different findings.

A policy that ignores the instruction entirely scores $1.00$ here. That is a true
statement about it, not a compliment, and it is why this metric is read beside the
other six dimensions rather than alone.

---

## `consistency/ece`

**Definition.** Expected calibration error: the gap between how confident the
model is and how often it is right, binned over confidence.

$$
\mathrm{ECE} \;=\; \sum_{b=1}^{B} \frac{\lvert \mathcal{B}_b \rvert}{n}
  \Bigl\lvert\, \mathrm{acc}(\mathcal{B}_b) - \mathrm{conf}(\mathcal{B}_b) \,\Bigr\rvert
$$

with $B = 10$ equal-width bins over $[0, 1]$, $\mathrm{conf}$ the mean reported
confidence in a bin and $\mathrm{acc}$ the observed success rate in it. One
confidence per episode, the mean over its steps.

**Target.** $0$, normalised as $1 - \mathrm{ECE}$.

**Example.** Ten episodes. The model says $0.9$ on five and succeeds on all five;
it says $0.3$ on five and succeeds on two:

```
bin (0.8, 0.9]:  n=5   conf 0.90   acc 1.00   |diff| 0.10
bin (0.2, 0.3]:  n=5   conf 0.30   acc 0.40   |diff| 0.10
ECE = 0.5×0.10 + 0.5×0.10 = 0.10   →   score 0.90
```

Slightly under-confident in both bins, and consistently so.

!!! note "Unmeasurable is not zero"
    A model exposing no `confidence()` reports `null` with the reason "the model
    reports no confidence, so calibration is unmeasurable". Its calibration is
    *unknown*, and scoring a zero would punish it for the library's ignorance.

---

## `consistency/brier`

**Definition.** The Brier score of confidence against outcome. Mean squared error
of a probabilistic prediction.

$$
\mathrm{BS} \;=\; \frac{1}{n}\sum_{i=1}^{n}\bigl(c_i - y_i\bigr)^2,
\qquad y_i = \mathbf{1}\!\left[\,\mathrm{succ}_i\,\right]
$$

**Target.** $0$, normalised as $1 - \mathrm{BS}$. Always in $[0, 1]$ for
confidences in $[0, 1]$.

**Example.** The same ten episodes as above:

```
five at (0.9 - 1)² = 0.01
five at (0.3 - 0)² = 0.09  ×3  and  (0.3 - 1)² = 0.49  ×2
BS = (5×0.01 + 3×0.09 + 2×0.49) / 10 = 0.130   →   score 0.870
```

**Why both ECE and Brier.** ECE measures calibration alone and a model that
always answers the base rate can score perfectly on it. Brier is a *proper*
scoring rule: it rewards being calibrated **and** being decisive. Reported
together, they separate "honest but useless" from "useful".
