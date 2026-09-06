# CLI

Standard-library `argparse`, and `main(argv) -> int`, so the CLI is testable
without a subprocess. Every handler imports what it needs *inside* the function,
so `xevals --help` and `xevals doctor` are fast on a bare install, which is
exactly the install on which someone is most likely to be running `doctor`.

```
xevals run       config.toml [key=value ...] [--quiet]     # one model, or several
xevals compare   runs/* [--out DIR] [--diff]
xevals report    runs/<run>/
xevals doctor
xevals {suites,metrics,perturbations,adapters,envs,datasets,judges,dimensions} [list|describe] [name]
```

## `run`

```bash
xevals run configs/synthetic.toml
xevals run configs/synthetic.toml run.episodes=50 suite.name=robustness
xevals run configs/synthetic.toml model.kwargs.device=cuda
xevals run configs/robots.toml                      # a Panda, with xevals[mujoco]
xevals run configs/robots.toml target.env=mujoco/ur5e-push
```

Overrides are coerced by the field's declared type, and a path the schema does not
declare is an **error naming the valid keys**. Below an untyped dict such as
`model.kwargs`, any key is legitimate and the value is parsed as a literal.

A config with `[[models]]` instead of `[model]` runs a
[benchmark](benchmarks.md): same command, same options, one leaderboard
instead of one table. Setting both is an error rather than a precedence rule: a
config where one of the two silently wins runs something other than what it
appears to say.

**Exit code 1 means the replay gate failed.** A pipeline that publishes these
numbers should stop rather than publish them; every other failure is exit code 2.

## `compare`

```bash
xevals compare runs/*/ --out leaderboard/
xevals compare runs/before runs/after --diff
```

`--out` writes the table in four formats plus an overlaid radar. A missing
matplotlib skips the radar with a note rather than failing the comparison.

## `report`

```bash
xevals report runs/synthetic-abc123def456/
```

Rebuilds the figures and `report.html` from `run.json`. The model is not needed,
which is the property that makes an archive worth keeping.

## The registries

```bash
xevals perturbations list
xevals perturbations list --family visual
xevals metrics describe robustness/retention
xevals suites describe full
xevals dimensions --json
```

`list` prints a name, a one-line summary, and which extra an entry needs. `describe`
prints JSON. Neither builds anything, so listing what exists never triggers an
optional import.

## `doctor`

```bash
xevals doctor
```

```
xevals 0.1.0  python 3.13.13
  torch        ok       2.14.0       wrap torch models; gradient attacks
  jax          missing  --           wrap JAX and Equinox models
  sim          ok       1.0.0        Gymnasium environments, and LIBERO / ManiSkill / SimplerEnv
  ...
The core needs numpy only; everything above fails when used, not at import.
```

It never raises. That is the whole job: the machine where something is wrong is
the machine where `doctor` has to work.

::: xevals.cli
    options:
      heading_level: 2
      members: false
