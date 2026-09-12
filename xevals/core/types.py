"""What xevals requires of a model, an environment, and a recorded episode.

Structural :class:`~typing.Protocol`\\ s, not base classes, for the reason xwm
states about its own types: the things being unified have no common ancestor.
A LeRobot policy, an OpenVLA checkpoint behind ``transformers``, a JAX function
and an HTTP endpoint are four codebases with no prospect of one, and requiring
them to inherit anything would mean nobody can be evaluated without a fork.

There are four model kinds because they answer four different questions, and a
metric that makes sense for one is meaningless for another:

============  ==================================  ===============================
kind          contract                            measured by
============  ==================================  ===============================
:class:`Policy`      ``act(obs) -> action``              rollout success, latency
:class:`WorldModel`  ``predict(obs, actions) -> pred``   prediction error, horizon
:class:`Planner`     ``plan(instruction) -> plan``       plan correctness, refusal
:class:`Scorer`      ``score(obs) -> float``             agreement with success
============  ==================================  ===============================

**Observations are ``dict[str, Any]``.** The runner never needs to know the
modality, so a model takes the fields it wants -- ``image``, ``state``,
``instruction`` -- and ignores the rest. That is what lets one runner drive a
pixel VLA and a state-space RL policy without a branch.

**Arrays are NumPy at every boundary.** Adapters convert torch and jax on the
way in and out. The alternative -- a framework-typed core -- makes the core
depend on the framework, and xevals's core depends on numpy alone.

**Capabilities are separate protocols, sniffed with ``isinstance``.** A model
that exposes :class:`HasConfidence` gets calibration metrics; one that does not
gets ``null`` with a reason, never an exception. Optional capability is the
common case and must not be an error path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Action",
    "BatchPolicy",
    "Env",
    "HasConfidence",
    "HasCost",
    "HasLatents",
    "Obs",
    "Plan",
    "Planner",
    "Policy",
    "Prediction",
    "SafetyLimits",
    "Scorer",
    "Step",
    "Trajectory",
    "Violation",
    "WorldModel",
]

#: One observation. Common keys: ``image`` ``(H, W, 3)`` uint8, ``state``
#: ``(D,)`` float32, ``instruction`` str. A model reads what it needs.
Obs = dict[str, Any]

#: One action, ``(A,)`` float32, normalised to ``[-1, 1]`` by the task's declared
#: bounds rather than by dataset statistics -- so a checkpoint means the same
#: thing whatever corpus it was trained on.
Action = np.ndarray

#: A world model's output. Any subset of ``frames`` ``(T, H, W, 3)``, ``latents``
#: ``(T, D)``, ``rewards`` ``(T,)``, ``states`` ``(T, S)``. A model that predicts
#: only latents is evaluated only on latents.
Prediction = dict[str, Any]

#: A planner's output. ``text`` is required; ``steps``, ``code`` and ``refused``
#: are optional, and ``refused=True`` is a *correct* answer to an unsafe request.
Plan = dict[str, Any]


@runtime_checkable
class Policy(Protocol):
    """Something that acts: a VLA, a behaviour-cloned net, an RL actor."""

    def act(self, obs: Obs, *, instruction: str | None = ...) -> Action:
        """One action for one observation."""
        ...


@runtime_checkable
class BatchPolicy(Policy, Protocol):
    """Opt-in independent inference rows, without hidden state or cross-row coupling.

    ``batch_mode`` must be ``"stateless"``. Results must not depend on batch size,
    row order, or prior calls. Each observation and instruction maps to one row
    of a float32 ``(B, A)`` array. Stateful/chunked policies use the serial runner.
    """

    batch_mode: str | None

    def act_batch(
        self, observations: Sequence[Obs], *, instructions: Sequence[str | None] | None = None,
    ) -> np.ndarray:
        ...


@runtime_checkable
class WorldModel(Protocol):
    """Something that predicts: latent dynamics, video models, xwm families."""

    def predict(self, obs: Obs, actions: np.ndarray, *, horizon: int = ...) -> Prediction:
        """Roll ``actions`` ``(T, A)`` forward from ``obs``."""
        ...


@runtime_checkable
class Planner(Protocol):
    """Something that decomposes: an LLM or VLM task planner, code-as-policy."""

    def plan(self, instruction: str, *, obs: Obs | None = ...) -> Plan:
        """Turn an instruction into a plan."""
        ...


@runtime_checkable
class Scorer(Protocol):
    """Something that judges a state: a reward model, a success classifier."""

    def score(self, obs: Obs, *, instruction: str | None = ...) -> float:
        """How good this observation is, higher better."""
        ...


@runtime_checkable
class HasConfidence(Protocol):
    """Optional: the model reports how sure it is, enabling calibration metrics."""

    def confidence(self, obs: Obs) -> float:
        """Probability-like score in ``[0, 1]`` for the action about to be taken."""
        ...


@runtime_checkable
class HasLatents(Protocol):
    """Optional: the model exposes its representation, enabling probe metrics."""

    def encode(self, obs: Obs) -> np.ndarray:
        """``(D,)`` embedding of one observation."""
        ...


@runtime_checkable
class HasCost(Protocol):
    """Optional: the model reports its own size and placement, for efficiency."""

    def cost(self) -> dict[str, Any]:
        """``{"params": int, "device": str, "dtype": str}``, any subset."""
        ...


@runtime_checkable
class Env(Protocol):
    """A steppable world. NumPy in, NumPy out.

    Mirrors xwm's environment protocol closely enough that an xwm environment
    satisfies this one, which is the point: the sibling library's simulators
    plug in unchanged.

    The invariant the conformance test checks, and that the whole
    reset-from-recorded-state protocol rests on::

        env.reset(state=env.state())  reproduces  env.state()

    Where the round trip is inexact -- a simulator carries contacts and solver
    state a state vector may not name -- the ``replay`` baseline measures the
    damage. Run it before reading any model's number.
    """

    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray

    def reset(self, *, seed: int | None = ..., state: np.ndarray | None = ...) -> Obs:
        """Start an episode, optionally at a given :meth:`state`."""
        ...

    def step(self, action: Action) -> tuple[Obs, float, bool, dict[str, Any]]:
        """Advance one step. Returns ``(obs, reward, done, info)``."""
        ...

    def state(self) -> np.ndarray:
        """``(S,)`` canonical state -- what :meth:`reset` accepts back."""
        ...

    def render(self) -> np.ndarray | None:
        """``(H, W, 3)`` uint8 for video and figures, or ``None``."""
        ...


@dataclass(frozen=True)
class SafetyLimits:
    """The box a policy is supposed to stay inside, as an environment declares it.

    Every field is optional: a limit that an environment cannot observe is
    ``None``, and the metrics depending on it report ``null`` with a reason
    rather than assuming a safe default. Assuming one is how a safety score
    becomes flattering.
    """

    workspace_low: np.ndarray | None = None
    workspace_high: np.ndarray | None = None
    joint_low: np.ndarray | None = None
    joint_high: np.ndarray | None = None
    max_velocity: float | None = None
    max_force: float | None = None
    min_clearance: float | None = None

    def check(self, info: Mapping[str, Any]) -> list[Violation]:
        """Every limit this step's ``info`` breaks, as violations with margins."""
        out: list[Violation] = []
        pos = info.get("position")
        if pos is not None and self.workspace_low is not None and self.workspace_high is not None:
            pos = np.asarray(pos, dtype=float)
            below = float(np.min(pos - np.asarray(self.workspace_low, dtype=float)))
            above = float(np.min(np.asarray(self.workspace_high, dtype=float) - pos))
            worst = min(below, above)
            if worst < 0:
                out.append(Violation("workspace", -worst))
        joints = info.get("joints")
        if joints is not None and self.joint_low is not None and self.joint_high is not None:
            joints = np.asarray(joints, dtype=float)
            worst = float(
                min(
                    np.min(joints - np.asarray(self.joint_low, dtype=float)),
                    np.min(np.asarray(self.joint_high, dtype=float) - joints),
                )
            )
            if worst < 0:
                out.append(Violation("joint", -worst))
        speed = info.get("speed")
        if speed is not None and self.max_velocity is not None and speed > self.max_velocity:
            out.append(Violation("velocity", float(speed) - self.max_velocity))
        force = info.get("force")
        if force is not None and self.max_force is not None and force > self.max_force:
            out.append(Violation("force", float(force) - self.max_force))
        clearance = info.get("clearance")
        if (
            clearance is not None
            and self.min_clearance is not None
            and clearance < self.min_clearance
        ):
            out.append(Violation("clearance", self.min_clearance - float(clearance)))
        if info.get("collision"):
            out.append(Violation("collision", 1.0))
        return out


@dataclass(frozen=True)
class Violation:
    """One limit broken at one step.

    ``margin`` is how far past the limit, in the limit's own units, because
    "left the workspace by 2 mm" and "by 40 cm" are different findings and a
    boolean loses the difference.
    """

    kind: str
    margin: float = 0.0
    step: int = -1


@dataclass(frozen=True)
class Step:
    """One transition, for callers that want the tape rather than the summary."""

    obs: Obs
    action: Action
    reward: float
    done: bool
    info: dict[str, Any]
    latency_ms: float = 0.0


@dataclass
class Trajectory:
    """One episode: what happened, under what conditions, at what cost.

    Every metric in xevals is a pure function of a list of these. That is what
    lets the same metric serve an online rollout and an offline dataset replay
    without a second implementation -- and it is why the trajectory carries its
    *conditions* (``seed``, ``perturbation``, ``split``) and not only its
    outcome. A success rate without the conditions it was measured under is the
    number this library exists to replace.
    """

    obs: list[Obs] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)
    frames: list[np.ndarray] | None = None
    instruction: str | None = None
    seed: int | None = None
    perturbation: str | None = None
    split: str = "in"
    success: bool | None = None
    violations: list[Violation] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def total_reward(self) -> float:
        """Undiscounted return."""
        return float(sum(self.rewards))

    @property
    def action_array(self) -> np.ndarray:
        """``(T, A)`` float32, or ``(0,)`` for an empty episode."""
        if not self.actions:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(self.actions, dtype=np.float32)

    def save(self, path: str | Path) -> Path:
        """Write ``<path>.npz`` (arrays) plus ``<path>.json`` (everything else).

        Split because the arrays are large and uninteresting to a human, while
        the conditions are small and are the first thing anyone reads.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        npz = path.with_suffix(".npz")
        arrays: dict[str, np.ndarray] = {
            "actions": self.action_array,
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "latencies_ms": np.asarray(self.latencies_ms, dtype=np.float32),
            "confidences": np.asarray(self.confidences, dtype=np.float32),
        }
        if self.frames:
            arrays["frames"] = np.asarray(self.frames, dtype=np.uint8)
        for key in _array_obs_keys(self.obs):
            arrays[f"obs.{key}"] = np.asarray([o[key] for o in self.obs])
        np.savez_compressed(npz, **arrays)
        meta = {
            "instruction": self.instruction,
            "seed": self.seed,
            "perturbation": self.perturbation,
            "split": self.split,
            "success": self.success,
            "violations": [asdict(v) for v in self.violations],
            "infos": _jsonable(self.infos),
            "obs_keys": sorted({k for o in self.obs for k in o}),
            "extra": _jsonable(self.extra),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        return npz

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        """Read back what :meth:`save` wrote."""
        path = Path(path)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        meta = json.loads(path.with_suffix(".json").read_text())
        obs_arrays = {k[4:]: v for k, v in arrays.items() if k.startswith("obs.")}
        n = len(next(iter(obs_arrays.values()))) if obs_arrays else 0
        obs = [{k: v[i] for k, v in obs_arrays.items()} for i in range(n)]
        frames = list(arrays["frames"]) if "frames" in arrays else None
        return cls(
            obs=obs,
            actions=list(arrays["actions"]),
            rewards=[float(r) for r in arrays["rewards"]],
            infos=meta.get("infos") or [],
            frames=frames,
            instruction=meta.get("instruction"),
            seed=meta.get("seed"),
            perturbation=meta.get("perturbation"),
            split=meta.get("split", "in"),
            success=meta.get("success"),
            violations=[Violation(**v) for v in meta.get("violations", [])],
            latencies_ms=[float(x) for x in arrays["latencies_ms"]],
            confidences=[float(x) for x in arrays["confidences"]],
            extra=meta.get("extra") or {},
        )


def _array_obs_keys(obs: Sequence[Obs]) -> list[str]:
    """Observation keys present in every step and array-shaped in all of them.

    A key that appears in only some steps, or that holds a string, cannot go in
    a stacked array; those live in the JSON sidecar instead.
    """
    if not obs:
        return []
    shared = set(obs[0])
    for o in obs[1:]:
        shared &= set(o)
    out = []
    for key in sorted(shared):
        values = [o[key] for o in obs]
        if all(isinstance(v, (np.ndarray, np.generic, int, float)) for v in values):
            shapes = {np.asarray(v).shape for v in values}
            if len(shapes) == 1:
                out.append(key)
    return out


def _jsonable(value: Any) -> Any:
    """Everything JSON can encode, with NumPy scalars and arrays unwrapped."""
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)
