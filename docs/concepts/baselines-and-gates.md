# Baselines and gates

Read this before quoting any number from an evaluation, including one of your own.

## The three baselines

Every suite runs reference policies on **the model's own seeds**, so the
comparison is not left to luck.

`random`
: Uniform actions in `[-1, 1]`. The floor. A model that does not clear it has not
been shown to do anything, and on a short-horizon task the floor is often
surprisingly high, which is itself worth knowing before anyone celebrates a 40 %
success rate.

`noop`
: Zero actions. Separates "the task rewards motion" from "the task rewards the
right motion". On some benchmarks doing nothing scores well, and that is a fact
about the benchmark that belongs beside the model's number rather than under it.

`replay`
: The demonstrator's own actions. Not a baseline: a **gate**.

## The replay gate

If replaying the actions that produced the data does not succeed in this
environment, the environment does not match the data. Every number measured in it
is then uninterpretable: a low success rate might be a bad model, or it might be
a simulator whose friction coefficient changed three releases ago.

This is not hypothetical. It is the most common silent failure in robot-learning
evaluation, because the environment and the checkpoint are usually maintained by
different people on different schedules, and nothing in a normal evaluation
script notices.

So `xevals` checks it, and reports one of three states, never two:

| Status | Meaning |
|---|---|
| `passed` | replaying the demonstrator solved the task; the environment is faithful |
| `failed` | it did not; **the numbers below do not measure the model** |
| `unmeasured` | this environment supplies no demonstrator actions, so nothing was checked |

`unmeasured` is a real third state and is not a pass. Most Gymnasium environments
land there, because Gym has no generic way to ask for a demonstration.

A failed gate is loud by design. It is a banner above the Markdown table, a
banner above everything in `report.html`, and **exit code 1** from `xevals run`, so
a pipeline stops rather than publishing.

```python
result.gate_status()   # "passed" | "failed" | "unmeasured"
result.gate_failed     # True only for "failed"
```

The threshold is deliberately low (half the episodes). The gate is not asking
whether the demonstrator is good; it is asking whether the environment is the one
the demonstrator acted in at all.

## Pairing

Every paired metric (retention, injection compliance, trigger delta,
paraphrase agreement) compares a perturbed episode with **the clean episode
that started from the same state**, not with a pooled clean average.

That is why episode seeds are derived from `(root_seed, seed, index)` and not
from the cell name: every cell runs the same episodes. Seeding cells
independently would leave the comparison to pooled averages, which need roughly
ten times the episodes to resolve the same difference, and would report
every paired metric as unmeasurable, because no two cells would share a seed.

## Retention is undefined, not perfect, when nothing succeeds

If the clean condition never succeeds, retention is `0 / 0`. `xevals` reports
`null` with that reason. Returning `1.0` is how a completely broken policy ends up
at the top of a robustness table, and it happens more often than it should.

Retention is also capped at `1.0`. A perturbation that *helps* is interesting and
is visible in the raw success rates; letting retention exceed 1 would let one
lucky cell hide a failure elsewhere in the mean.
