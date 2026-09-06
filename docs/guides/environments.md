# Environments

`xevals` needs an environment for the online half of an evaluation, but it must not
*require* a simulator to be useful.

## Robots

```python
env = xevals.envs.create("mujoco/panda-pick")    # needs xevals[mujoco]
```

Five MuJoCo Menagerie arms on a table with a block and a marker, four tasks, one
four-number action space across all of them, and a second physics backend in
Newton. That is where the episodes in this documentation come from, and it has
a guide of its own: [Robots](robots.md).

## The built-in world

```python
import xevals
env = xevals.envs.create("synthetic/reach")     # or "synthetic/push"
```

A 2-D point mass that renders its own frames in numpy. It has no dependencies and
it is not a toy in the ways that matter here:

- **the workspace limit is real**: the safe region is smaller than the
  arena, so a policy that cuts a corner genuinely violates a limit and the safety
  dimension has something to measure;
- **physics is settable**: `set_physics(mass=, friction=, gain=)`, which is
  what `dynamics/*` drives;
- **objects are named and coloured**, so `ood/object` is a real split rather than
  a relabelling;
- **it renders text into the scene**, which is what `injection/scene_text` needs.

That is why the whole library, including the security dimension, is exercisable
in a second on a bare install, with no simulator, no assets and no download. It
is the environment the test suite runs on, and the one to reach for when the
question is about `xevals` rather than about a robot.

## Gymnasium

```python
env = xevals.envs.create("gymnasium/PushT-v0")           # needs xevals[sim]
env = xevals.envs.gymnasium_env("LIBERO-Spatial-v0", max_episode_steps=300)
```

One adapter reaches most third-party simulators, because LIBERO, ManiSkill and
SimplerEnv all expose Gym APIs. It handles two impedance mismatches for you: Gym's
`terminated` and `truncated` are kept distinct (conflating them turns a timeout
into a failure), and a bare array observation is wrapped into `{"state": ...}` so
it reaches models under the same key an image-and-state observation would.

Gym exposes no generic state setter, so `reset(state=...)` raises rather than
silently ignoring the argument, and the replay gate reads `unmeasured`.

!!! note "Dedicated adapters"
    Version 0.1 ships the Gymnasium adapter only. Recipes for LIBERO and
    SimplerEnv live here; first-class adapters, and ManiSkill3, are 0.2.

## Replay: the offline path

```python
from xevals import datasets

episodes = datasets.synthetic_episodes("reach", episodes=8)
env = episodes[0].to_env()
```

A `ReplayEnv` returns the next **recorded** observation whatever the model does.
That sounds useless until you notice what it buys: the same runner, the same
wrappers and the same metrics work on a dataset with no simulator attached. Action
error against the demonstrator, prediction error for a world model, and every
visual and language perturbation are all measurable offline, by exactly the code
that measures them online.

What it cannot measure is anything counterfactual (success under the model's
*own* actions) because the world does not respond. `xevals` marks
those trajectories offline and `accuracy/success_rate` reports `null` with that
reason, rather than publishing the demonstrator's success as the model's.

## The protocol

```python
class MyEnv:
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray

    def reset(self, *, seed=None, state=None) -> dict: ...
    def step(self, action) -> tuple[dict, float, bool, dict]: ...
    def state(self) -> np.ndarray: ...
    def render(self) -> np.ndarray | None: ...

    # optional, and each unlocks something
    def success(self, info) -> bool: ...          # the accuracy dimension
    def limits(self) -> SafetyLimits: ...         # the safety dimension
    def set_physics(self, **kwargs) -> None: ...  # dynamics/* perturbations
    def set_scene_text(self, text) -> None: ...   # injection/scene_text
```

It mirrors xwm's environment protocol closely enough that an xwm simulator
satisfies it unchanged, which was the design intent on both sides.

### The reset invariant

```python
env.reset(state=env.state())   # must reproduce env.state()
```

The whole reset-from-recorded-state protocol rests on this, and it is not free: a
simulator carries velocities, contacts and solver state a state vector may not
name. `xevals.envs.check_reset_invariant(env)` returns the maximum absolute
difference, and every run records it in `run.json`, which is better as a
number than as a surprise later.

## Splits

The generalisation suite asks the same task under four distributions, each
changing exactly one thing so a drop is attributable:

| Split | What changes |
|---|---|
| `in` | nothing |
| `ood/object` | an object colour the training distribution never contained |
| `ood/layout` | spawn geometry near the arena edges |
| `ood/instruction` | a wording template never seen, same task |

An environment that cannot vary by split simply ignores the argument, and the
generalisation metrics then report it as unmeasured, which is true.

## Safety limits

```python
def limits(self):
    return SafetyLimits(
        workspace_low=..., workspace_high=...,
        max_velocity=0.35, min_clearance=0.02,
    )
```

Every field is optional, and a limit the environment cannot observe is `None`.
The metrics depending on it then report `null` with a reason rather than assuming
a safe default: assuming one is how a safety score becomes flattering.

::: xevals.envs
    options:
      heading_level: 2
      members: false
