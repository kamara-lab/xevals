# Wrapping models

`xevals.wrap(obj)` is the whole entry point. It looks at the object, picks an
adapter, and hands back something with an `act`, `predict` or `plan` method.
Nothing here requires the model's author to have heard of `xevals`.

```python
import xevals

xevals.wrap(my_fn)                             # a plain callable
xevals.wrap(my_torch_module)                   # torch.nn.Module
xevals.wrap(my_jax_fn)                         # JAX / Equinox
xevals.wrap(lerobot_policy)                    # LeRobot
xevals.wrap("https://endpoint/act")            # HTTP
xevals.wrap(anthropic_client, kind="planner")  # LLM planner
xevals.wrap(xwm_model)                         # the sibling library
```

An object that **already satisfies a protocol** is returned untouched, because the
model's own implementation is by definition more faithful than anything inferred
about it.

## The four model kinds

They exist because they answer four different questions, and a metric that makes
sense for one is meaningless for another.

| Kind | Contract | Measured by |
|---|---|---|
| `Policy` | `act(obs, *, instruction) -> action` | rollout success, latency, violations |
| `WorldModel` | `predict(obs, actions, *, horizon) -> prediction` | prediction error, horizon degradation |
| `Planner` | `plan(instruction, *, obs) -> plan` | plan correctness, refusal, injection resistance |
| `Scorer` | `score(obs, *, instruction) -> float` | agreement with the success signal |

A bare callable does not say which it is, so pass `kind=`:

```python
xevals.wrap(fn, kind="world_model")
xevals.wrap(fn, kind="planner")
```

## Optional capabilities

Sniffed with `isinstance`, never required. A model that exposes one gets the
metrics that depend on it; a model that does not gets `null` **with a reason**.

| Protocol | Method | Unlocks |
|---|---|---|
| `HasConfidence` | `confidence(obs) -> float` | `consistency/ece`, `consistency/brier` |
| `HasLatents` | `encode(obs) -> ndarray` | representation metrics |
| `HasCost` | `cost() -> dict` | `efficiency/params` |

A `reset()` method is called between episodes when present. For a chunked policy
that is not optional in practice: skipping it leaks the previous episode's
action queue into the next one.

## What each adapter handles for you

**`torch`**: `eval()` mode, `no_grad`, device placement, the batch dimension
added on the way in and removed on the way out, and the parameter count read into
the efficiency dimension. A forgotten `eval()` leaves dropout on and costs a few
points of success rate that then get attributed to the method.

**`lerobot`**: calls the policy's own `select_action` and `reset`, so its
normalisation statistics and action chunking stay the policy's business.
Reimplementing normalisation is the single most common way a wrapped VLA quietly
loses most of its performance.

**`hf_vla`**: the prompt template is a constructor argument, not a constant,
because the template is part of the checkpoint; a model trained on
`"In: What action should the robot take to {instruction}?\nOut:"` and prompted
with a bare instruction is out of distribution. Supply `unnorm_key` to use the
checkpoint's own action un-normalisation; without it the raw output is returned
and `run.json` records that it was.

**`remote`**: latency is measured client-side and includes the network,
which is the honest number for a deployment that would also call it over the
network.

**`chat`**: takes a bare `str -> str` callable, an Anthropic client, or an
OpenAI-compatible client. Temperature is 0: a consistency metric on a sampled
model measures the sampler.

## Writing your own

There is no base class. Satisfy the protocol:

```python
class MyPolicy:
    def act(self, obs, *, instruction=None):
        return my_action(obs)          # (A,) float32 numpy

    def reset(self):                   # optional
        self.history.clear()

    def confidence(self, obs):         # optional -> calibration metrics
        return float(self.last_logit)

    def describe(self):                # optional -> the run.json fingerprint
        return {"module": "MyPolicy", "params": 12_000_000}
```

`describe()` lands verbatim in `run.json` as the model fingerprint, so two runs
can be compared on *what was evaluated* and not only on the numbers.

::: xevals.adapters
    options:
      heading_level: 2
      members: false
