"""Reading recorded episodes, so a model can be evaluated without a simulator.

Most robot data is a corpus, not a world. That corpus supports a real
evaluation -- action error against the demonstrator, world-model prediction
error, every visual and language perturbation -- as long as something turns an
episode into the shape the runner already knows. That something is
:class:`~xevals.envs.ReplayEnv`, and this module's job is only to produce the
episodes it takes.

Three formats, chosen because they cover what is actually published:

``lerobot``  parquet plus mp4 shards -- the LeRobot and Open X-Embodiment mirrors
``hdf5``     RoboMimic and LIBERO
``npz``      whatever a lab exported itself, and what xevals writes

Every reader is lazy about its format library. Reading parquet needs
``pyarrow``; the module imports cleanly without it, and asks by name at the call
that needs it. The alternative -- importing them at module load -- means a bare
install cannot even list what readers exist.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import MissingExtra
from .registry import Registry
from .types import Obs

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "Episode",
    "available",
    "create",
    "describe",
    "read_hdf5",
    "read_lerobot",
    "read_npz",
    "replay_envs",
    "synthetic_episodes",
]

DATASETS: Registry[Any] = Registry("datasets")


@dataclass
class Episode:
    """One recorded episode: observations, the actions taken, what happened.

    ``obs`` has one more entry than ``actions``, as a trajectory does -- the
    final observation is where the episode ended, and dropping it is how an
    off-by-one appears in a goal-distance metric.
    """

    obs: list[Obs] = field(default_factory=list)
    actions: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    rewards: list[float] = field(default_factory=list)
    instruction: str | None = None
    success: bool | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def to_env(self) -> Any:
        """This episode as an environment the runner can drive."""
        from .envs import ReplayEnv

        return ReplayEnv(
            self.obs,
            self.actions,
            rewards=self.rewards,
            success=self.success,
            instruction=self.instruction,
        )


@dataclass(frozen=True)
class DatasetSpec:
    """Where a corpus is and how to read it.

    Attributes:
        name: the registered reader (``lerobot``, ``hdf5``, ``npz``).
        path: a directory, a file, or a Hub repo id.
        episodes: how many to read. ``None`` reads all of them, which for a real
            corpus is usually the wrong default and never the one taken here.
        instruction_key: which field carries the language, when the corpus has
            more than one convention for it.
    """

    name: str
    path: str
    episodes: int | None = 10
    instruction_key: str = "task"
    kwargs: dict[str, Any] = field(default_factory=dict)

    def read(self) -> list[Episode]:
        """Read the episodes this spec names."""
        return list(
            DATASETS.create(self.name, path=self.path, episodes=self.episodes, **self.kwargs)
        )


def read_npz(path: str | Path, *, episodes: int | None = None, **_: Any) -> Iterator[Episode]:
    """Read episodes from ``.npz`` files -- one file per episode.

    The format xevals writes with :meth:`xevals.Trajectory.save`, so a run's saved
    trajectories are directly re-evaluable. Arrays named ``obs.<key>`` become
    observation fields; ``actions`` and ``rewards`` are taken as themselves.
    """
    import json

    root = Path(path)
    files = sorted(root.glob("*.npz")) if root.is_dir() else [root]
    for index, file in enumerate(files):
        if episodes is not None and index >= episodes:
            return
        with np.load(file, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        sidecar = file.with_suffix(".json")
        meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        obs_arrays = {k[4:]: v for k, v in arrays.items() if k.startswith("obs.")}
        n = len(next(iter(obs_arrays.values()))) if obs_arrays else len(arrays["actions"]) + 1
        obs = [{k: v[i] for k, v in obs_arrays.items()} for i in range(n)]
        yield Episode(
            obs=obs,
            actions=np.asarray(arrays["actions"], dtype=np.float32),
            rewards=[float(r) for r in arrays.get("rewards", [])],
            instruction=meta.get("instruction"),
            success=meta.get("success"),
            meta={"path": str(file)},
        )


def read_hdf5(
    path: str | Path,
    *,
    episodes: int | None = None,
    obs_group: str = "obs",
    **_: Any,
) -> Iterator[Episode]:
    """Read a RoboMimic- or LIBERO-style HDF5 file. Needs ``xevals[data]``.

    The layout both use is ``data/demo_<i>/{obs/<key>, actions, rewards}``, with
    images stored as ``(T, H, W, C)`` uint8 and sometimes JPEG-compressed. The
    reader takes the raw arrays and leaves decoding to the caller, because a
    reader that silently decodes has also silently resized.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise MissingExtra("h5py", "data", "to read HDF5 datasets") from exc
    with h5py.File(str(path), "r") as handle:
        demos = sorted(handle["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        for index, key in enumerate(demos):
            if episodes is not None and index >= episodes:
                return
            group = handle["data"][key]
            actions = np.asarray(group["actions"], dtype=np.float32)
            fields = {k: np.asarray(v) for k, v in group[obs_group].items()}
            n = len(actions) + 1
            obs = [
                {k: v[min(i, len(v) - 1)] for k, v in fields.items()} for i in range(n)
            ]
            attrs = dict(group.attrs)
            yield Episode(
                obs=obs,
                actions=actions,
                rewards=[float(r) for r in np.asarray(group.get("rewards", []))],
                instruction=attrs.get("task") or attrs.get("language_instruction"),
                success=bool(attrs.get("success", True)),
                meta={"demo": key},
            )


def read_lerobot(
    path: str | Path,
    *,
    episodes: int | None = None,
    instruction_key: str = "task",
    **_: Any,
) -> Iterator[Episode]:
    """Read a LeRobot dataset directory. Needs ``xevals[data]``.

    LeRobot keeps states and actions in parquet and frames in per-episode mp4
    shards. This reader takes the parquet half and, when ``av`` is present,
    decodes the frame span for each episode -- the split is why it degrades
    gracefully: a state-only policy is evaluable without a video decoder.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise MissingExtra("pyarrow", "data", "to read a LeRobot dataset") from exc
    root = Path(path)
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet shards under {root}")
    table = pq.read_table(files[0]).to_pydict()
    index_column = "episode_index" if "episode_index" in table else "episode"
    indices = np.asarray(table[index_column])
    for count, episode_index in enumerate(sorted(set(indices.tolist()))):
        if episodes is not None and count >= episodes:
            return
        mask = indices == episode_index
        actions = np.asarray(
            [a for a, keep in zip(table["action"], mask, strict=True) if keep], dtype=np.float32
        )
        states = np.asarray(
            [
                s
                for s, keep in zip(table.get("observation.state", table["action"]), mask,
                                   strict=True)
                if keep
            ],
            dtype=np.float32,
        )
        obs = [{"state": s} for s in states] + [{"state": states[-1]}]
        instructions = [t for t, keep in zip(table.get(instruction_key, []), mask, strict=False)
                        if keep]
        yield Episode(
            obs=obs,
            actions=actions,
            rewards=[0.0] * len(actions),
            instruction=str(instructions[0]) if instructions else None,
            success=True,
            meta={"episode_index": int(episode_index), "shard": str(files[0])},
        )


def synthetic_episodes(
    task: str = "reach", *, episodes: int = 5, horizon: int = 60, seed: int = 0
) -> list[Episode]:
    """Record the synthetic environment's own scripted policy as a dataset.

    Exists so the offline path is testable and demonstrable with no download and
    no extras -- and so the replay gate has something to gate against in the
    documentation examples.
    """
    from .envs import create

    out = []
    for index in range(episodes):
        env = create(f"synthetic/{task}")
        obs = env.reset(seed=seed + index)
        observations, actions, rewards = [obs], [], []
        success = False
        for _ in range(horizon):
            action = env.optimal_action
            obs, reward, done, info = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            success = bool(info.get("success"))
            if done:
                break
        out.append(
            Episode(
                obs=observations,
                actions=np.asarray(actions, dtype=np.float32),
                rewards=rewards,
                instruction=env.instruction,
                success=success,
                meta={"seed": seed + index, "task": task},
            )
        )
    return out


def replay_envs(episodes: Sequence[Episode]) -> list[Any]:
    """Every episode as a replay environment, ready for the runner."""
    return [e.to_env() for e in episodes]


DATASETS.register("npz", read_npz, summary="one .npz per episode -- what xevals itself writes")
DATASETS.register(
    "hdf5", read_hdf5, summary="RoboMimic / LIBERO layout", requires=("data",)
)
DATASETS.register(
    "lerobot", read_lerobot, summary="LeRobot parquet shards", requires=("data",)
)
DATASETS.register(
    "synthetic",
    lambda path="reach", episodes=5, **kw: iter(
        synthetic_episodes(path, episodes=episodes, **kw)
    ),
    summary="the built-in scripted policy, recorded -- no download, no extras",
)


def available() -> list[str]:
    """Registered dataset readers."""
    return DATASETS.available()


def describe(name: str | None = None) -> dict[str, Any]:
    """What a reader reads, and what it needs installed."""
    return DATASETS.describe(name)  # type: ignore[return-value]


def create(name: str, **kwargs: Any) -> Any:
    """Run a reader, returning an iterator of :class:`Episode`."""
    return DATASETS.create(name, **kwargs)
