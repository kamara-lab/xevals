# Datasets

Most robot data is a corpus, not a world. `xevals` reads three formats and turns
each episode into a [replay environment](environments.md#replay-the-offline-path),
which the ordinary runner then drives.

| Reader | Format | Needs |
|---|---|---|
| `npz` | one `.npz` per episode: what `xevals` itself writes | nothing |
| `hdf5` | RoboMimic and LIBERO | `xevals[data]` |
| `lerobot` | parquet shards plus mp4 | `xevals[data]` |
| `synthetic` | the built-in scripted policy, recorded | nothing |

```python
from xevals import datasets

episodes = datasets.read_hdf5("libero_spatial.hdf5", episodes=20)
episodes = datasets.read_lerobot("~/.cache/lerobot/pusht", episodes=20)
episodes = datasets.synthetic_episodes("reach", episodes=8)   # no download
```

Each reader imports its format library lazily and asks for it by name at the call
that needs it, so a bare install can still list what readers exist.

## Running an offline evaluation

```python
import xevals
from xevals import datasets

episodes = datasets.read_lerobot("~/.cache/lerobot/pusht", episodes=20)
cycle = iter(episodes)

def make_env(task=None, split="in"):
    return next(cycle).to_env()

result = xevals.evaluate(my_policy, make_env, suite="robustness", baselines=False)
```

or from a config:

```toml
[target]
dataset = "lerobot:~/.cache/lerobot/pusht"
```

## What offline measures, and what it does not

**Measurable:** action error against the demonstrator (`accuracy/action_mse`,
`accuracy/action_mae`), world-model prediction error and horizon degradation, and
every visual, sensor and language perturbation, because none of those needs
the world to respond.

**Not measurable:** success under the model's own actions. The world replays
regardless, so `accuracy/success_rate` reports `null` with that reason. This is the
one place where a plausible-looking number would be maximally misleading: every
recorded episode succeeded, so publishing the recording's outcome as the model's
would give a policy that does nothing a perfect score.

## Action error is not success

A policy can have low action error and still fail the task, because errors compound
along a trajectory the demonstrator never visited. `xevals` reports action error
beside success rate, never instead of it, and offline, where success is
unmeasurable, the null carries the reason so nobody reads the one as the other.

## The replay gate, offline

Here the gate is meaningful and cheap: replaying the dataset's own actions through
`ReplayEnv` reproduces the recording by construction, so the gate checks the
*reader* rather than the simulator. A failure means the parquet columns or the
HDF5 keys were read wrongly, which is worth catching before anything else.

::: xevals.datasets
    options:
      heading_level: 2
      members: false
