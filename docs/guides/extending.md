# Extending

Everything in `xevals` goes through one registry type, so adding to any namespace
looks the same.

## A metric

```python
from xevals import Dimension
from xevals.metrics import MetricValue, bootstrap_ci, metric

@metric("safety/tipping_rate", Dimension.SAFETY,
        higher_is_better=False, unit="fraction")
def tipping_rate(trajs, *, ref=None, **kw) -> MetricValue:
    """Fraction of episodes in which the object was knocked over."""
    values = [float(any(i.get("tipped") for i in t.infos)) for t in trajs]
    if not values:
        return MetricValue("safety/tipping_rate", None,
                           reason="the environment does not report tipping")
    return MetricValue("safety/tipping_rate", float(np.mean(values)),
                       bootstrap_ci(values), len(values), values)
```

Then add a normaliser, with the reason:

```python
from xevals.dimensions import NORMALISERS, Normaliser
NORMALISERS["safety/tipping_rate"] = Normaliser("inverse_unit")
```

Three things to get right, in order of how much damage they do when wrong:

1. **`applies=`**: which cells the metric belongs in. A metric computed
   where it is meaningless is averaged into a dimension score, so it is worse
   than no metric.
2. **Null with a reason**: never zero, never an exception.
3. **An interval**: `bootstrap_ci` on the per-episode values.

## A perturbation

```python
from xevals.dimensions import Dimension
from xevals.perturbations import PERTURBATIONS, _Base

class Vignette(_Base):
    """Lens vignetting. Severity 1.0 darkens the corners by 60%."""

    def apply_obs(self, obs, *, seed=0):
        ...   # deterministic in (seed, severity); same shape and dtype out

PERTURBATIONS.register(
    "visual/vignette",
    lambda *, severity=1.0, **kw: Vignette(
        severity=severity, name="visual/vignette", dimension=Dimension.ROBUSTNESS
    ),
    summary="lens vignetting",
    dimension="robustness",
    severities=[0.2, 0.4, 0.6, 0.8, 1.0],
)
```

The [three rules](../concepts/perturbations.md) are not optional, and the test
suite checks them on every registered perturbation, including yours, if it is
registered before the tests run.

Implement exactly one of `apply_obs`, `apply_action`, `apply_instruction`,
`apply_env`. The runner sniffs for whichever is present; there is nothing to stub.

## An adapter

```python
from xevals.adapters import ADAPTERS

class MyFrameworkPolicy:
    def __init__(self, model, **kwargs):
        import my_framework          # inside, always
        ...

    def act(self, obs, *, instruction=None):
        return numpy_action

    def describe(self):
        return {"adapter": "my_framework", "params": ...}

ADAPTERS.register("my_framework", lambda model, **kw: MyFrameworkPolicy(model, **kw),
                  summary="a policy from my framework", requires=("my_framework",))
```

Import the framework **inside** the class, and convert to numpy at the boundary in
both directions. Those two rules are what keep `import xevals` free on every install.

## An environment

Satisfy the [protocol](environments.md#the-protocol). Add `limits()` to unlock the
safety dimension, `set_physics()` for dynamics perturbations, `success(info)` for
the accuracy dimension, and `set_scene_text()` for scene-text injection. Each is
optional, and each missing one is reported as an unmeasured metric with its reason
rather than a silent gap.

## A judge

```python
from xevals.judges import JUDGES, Verdict

class MyJudge:
    name = "my_judge"

    def judge(self, text, *, reference=None) -> Verdict:
        return Verdict(passed, confidence, reason, self.name)

JUDGES.register("rules/mine", MyJudge, summary="what it decides")
```

Prefer a rule to a model. A judge is part of the measuring instrument, and a
model-based one that drifts between provider versions makes last quarter's
numbers incomparable with this quarter's, and does it invisibly, in the results
file.

::: xevals.registry
    options:
      heading_level: 2
      members: false
