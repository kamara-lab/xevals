# Quickstart

Nothing here needs a download, a simulator, or a GPU. The built-in world is a
2-D point mass that renders its own frames in numpy, and it has everything the
library needs to measure: real workspace limits, a settable physics parameter,
named objects for out-of-distribution splits, and the ability to paint text into
the scene for injection tests.

## One model, one line

```python
import numpy as np
import xevals

def policy(obs):
    delta = obs["goal"] - obs["state"][:2]
    norm = np.linalg.norm(delta)
    return (delta / norm if norm > 1e-6 else delta).astype(np.float32)

result = xevals.evaluate(xevals.wrap(policy), "synthetic/reach", suite="core")
print(result.table())
```

```
| dimension  | score  | metrics | skipped | question                                             |
| ---------- | ------ | ------- | ------- | ---------------------------------------------------- |
| accuracy   | 0.9998 | 3       | 5       | Does it do the task?                                 |
| safety     | 1.0000 | 4       | 2       | Does it stay inside physical and behavioural limits? |
| efficiency | 1.0000 | 4       | 1       | What does it cost to run?                            |
```

Note the `skipped` column. Five accuracy metrics could not be measured (action
error needs a dataset, prediction error needs a world model), and each records
**why** in `run.json`. Nothing was silently scored zero.

## All seven dimensions

```python
result = xevals.evaluate(
    xevals.wrap(policy), "synthetic/reach",
    suite="full",
    episodes=10, seeds=range(3),
    out="runs/quickstart",
)
```

That writes a run directory:

```
runs/quickstart/full-<hash>/
  run.json          everything, full precision, schema-versioned
  table.{md,tex,csv,json}
  metrics.{md,tex,csv,json}
  cells.{md,tex,csv,json}
  figures/          radar, dimension-bars, severity, latency, heatmap
  videos/           a few episodes per cell, with a HUD
  report.html       self-contained; opens offline
```

Open `report.html`. The caveats come first, a failed gate, a truncated
run, every dimension that could not be measured, and then the numbers.

## From the command line

```bash
xevals run configs/synthetic.toml
xevals run configs/synthetic.toml run.episodes=50 suite.name=robustness
xevals compare runs/*/ --out leaderboard/
xevals report runs/synthetic-abc123def456/
```

An override the schema does not declare is an **error** naming the valid keys.
A silently ignored `run.episode=50` would produce a twenty-episode run that looks
exactly like the fifty-episode run you asked for.

## On a robot

The synthetic world is the zero-install first step, not the destination. With
`xevals[mujoco]` the same call runs against a Franka Emika Panda picking a block
off a table, and against four other arms beside it:

```python
xevals.evaluate(model, "mujoco/panda-pick", suite="core")
xevals.evaluate(model, "mujoco/ur5e-push", suite="robustness")
```

Every arm takes the same four numbers, `[dx, dy, dz, grip]`, so one policy is
comparable across all of them. See [Robots](../guides/robots.md).

## Wrapping a real model

```python
xevals.wrap(my_torch_module)                     # eval(), no_grad, device round trip
xevals.wrap(my_jax_fn)                           # device_get at the boundary
xevals.wrap(lerobot_policy)                      # its own select_action loop
xevals.wrap("https://my-endpoint/act")           # latency measured client-side
xevals.wrap(anthropic.Anthropic(), kind="planner")
xevals.wrap(my_fn, kind="world_model")
```

An object that already satisfies one of the [protocols](../reference/types.md) is
returned untouched: the model's own implementation is more faithful than anything
inferred about it.

## What to read next

- [Conventions](conventions.md): arrays, seeds, run directories, null-with-reason.
- [Dimensions](../concepts/dimensions/index.md): what each one asks and how it is scored.
- [Baselines and gates](../concepts/baselines-and-gates.md): read this before
  quoting a number.
