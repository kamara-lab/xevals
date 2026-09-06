# Efficiency

> **What does it cost to run?**

Averaged across cells, because it barely varies between them. This is the one
dimension a seed cannot make reproducible: wall-clock latency moves between runs,
and the test suite excludes it from the bit-for-bit determinism check rather than
pretending otherwise.

Timing is measured around the model's `act` call **alone**, not around the whole
step. Including the simulator would make a fast model on a slow simulator look
slow, and the efficiency dimension is a claim about the model.

| Metric | Direction | Unit | Reference |
|---|---|---|---|
| [`latency_p50`](#efficiencylatency_p50) | lower | ms | 100 ms |
| [`latency_p95`](#efficiencylatency_p95) | lower | ms | 200 ms |
| [`throughput`](#efficiencythroughput) | higher | steps/s | 10 steps/s |
| [`control_headroom`](#efficiencycontrol_headroom) | higher | ratio | the control rate |
| [`params`](#efficiencyparams) | lower | count | $10^9$ |

---

## `efficiency/latency_p50`

**Definition.** The median time for one `act` call, over **pooled steps**.

$$
\mathrm{p50} \;=\; \operatorname{median}\bigl(\, \{\, \ell_{i,t} \;:\; i \in [n],\; t \in [T_i] \,\} \,\bigr)
$$

**Target.** As low as possible. Normaliser $\exp(-\ell/100)$: 100 ms is 10 Hz
control, the floor for most manipulators, and it scores $0.368$.

**Example.** Two episodes, one with 90 steps at 1 ms and one with 10 steps at
100 ms:

```
pooled p50            = 1.0 ms
mean of episode means = (1.0 + 100.0) / 2 = 50.5 ms
```

!!! note "Pooled, not averaged per episode"
    A percentile of per-episode means is not a percentile of step latencies, and
    the tail is the whole point of measuring a control loop. The two numbers above
    differ by a factor of fifty on the same data.

---

## `efficiency/latency_p95`

**Definition.** The 95th percentile of pooled step latency. What a control loop
must budget for.

$$
\mathrm{p95} \;=\; Q_{0.95}\bigl(\, \{\, \ell_{i,t} \,\} \,\bigr)
$$

**Target.** As low as possible. Normaliser $\exp(-\ell/200)$, twice the p50
reference: the tail gets a tail budget.

**Example.** On the same 100 pooled steps as above:

```
p50 = 1.0 ms      p95 = 100.0 ms
```

A hundredfold spread between the median and the tail. A model quoted at "1 ms per
step" misses its deadline one step in twenty.

---

## `efficiency/throughput`

**Definition.** Steps per second from the model alone.

$$
\mathrm{throughput} \;=\; \frac{1000}{\bar{\ell}}, \qquad
\bar{\ell} \;=\; \frac{1}{N}\sum_{i,t} \ell_{i,t} \;\text{(ms)}
$$

**Target.** As high as possible. Normaliser $1 - \exp(-r/10)$, so 10 steps/s
scores $0.632$.

**Example.** A mean latency of $0.094$ ms gives $10\,638$ steps/s. The absolute
number is uninteresting for a numpy policy; what it demonstrates is the reporting
shape, and the same figure for a 7B VLA would be about $3$.

---

## `efficiency/control_headroom`

**Definition.** How much of the control period the model leaves unused, at the
declared control rate.

$$
h \;=\; 1 \;-\; \frac{\mathrm{p95}}{B}, \qquad B \;=\; \frac{1000}{f_{\mathrm{Hz}}} \;\text{ms}
$$

**Target.** As close to $1$ as possible. **Below zero means the model cannot keep
up with the loop it is meant to close.** Already a fraction, so it is used as-is,
clamped at $0$.

**Example.** At the default 10 Hz, the budget $B$ is 100 ms:

| p95 | headroom | reading |
|---|---|---|
| 0.10 ms | 0.999 | effectively free |
| 34 ms | 0.66 | comfortable |
| 96 ms | 0.04 | no margin for a slow step |
| 180 ms | **-0.80** | misses the deadline |

!!! note "Milliseconds alone are not interpretable"
    A latency means nothing without the rate the robot runs at, which is why this
    metric takes `control_hz` rather than assuming one, and why the latency figure
    draws the budget as a line rather than leaving the reader to divide.

---

## `efficiency/params`

**Definition.** The model's parameter count, when the adapter can report one.
Taken from the model fingerprint, not from the trajectories.

$$
\mathrm{params} \;=\; \sum_{\theta \,\in\, \mathrm{model}} \lvert \theta \rvert
$$

**Target.** Lower, all else equal. Normaliser $\exp(-p/10^9)$: a billion
parameters is one big model and scores $0.368$.

**Example.** A `torch.nn.Sequential(Linear(4, 2))` reports 10 parameters and
scores $\approx 1.0$. A 7B VLA scores $\exp(-7) \approx 0.001$.

!!! note "A weak signal on its own"
    Parameter count is a proxy for cost that ignores quantisation, sparsity,
    batching and the accelerator, and a model exposing no count reports `null`
    rather than zero. Latency is the number that decides whether a model can run
    on the robot; this one is context.
