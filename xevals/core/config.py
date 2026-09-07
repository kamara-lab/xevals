"""Reading a config: TOML in, dataclass tree out, overrides applied loudly.

Ported from xwm's loader, and for the same three reasons, each of which is a way
configs usually fail:

**An unknown key is an error, naming the valid ones.** A silently ignored
``run.episode=50`` (the field is ``episodes``) produces a twenty-episode run that
looks exactly like the fifty-episode run you asked for, and the difference shows
up as noise in a comparison a month later.

**Overrides are coerced by the field's declared type**, not guessed from the
string, so ``run.episodes=4`` is an ``int`` and ``model.kwargs.size=small`` --
inside an untyped dict -- stays a string.

**The resolved config is written back as JSON.** ``tomllib`` reads and does not
write, and rather than take a dependency to write TOML the run directory keeps
the source file verbatim *and* the fully resolved tree that ``xevals report``
reads back.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
import types
import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "NamedModelConfig",
    "OutputConfig",
    "RunConfig",
    "SuiteConfig",
    "TargetConfig",
    "apply_overrides",
    "config_hash",
    "from_dict",
    "load",
    "save",
    "to_dict",
]


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """What to evaluate.

    Attributes:
        adapter: a registered adapter name, or ``""`` to let :func:`xevals.wrap`
            detect one.
        target: how to obtain the model -- ``"module:attribute"``, a checkpoint
            path, a Hub id, or an HTTP URL. Deliberately a string rather than a
            live object, because a config must survive being written to a file.
        kwargs: passed to the adapter's constructor.
    """

    adapter: str = ""
    target: str = ""
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class NamedModelConfig(ModelConfig):
    """A model in a benchmark, with the name its leaderboard row carries.

    Separate from :class:`ModelConfig` only by that name, and the name is
    required: a benchmark row labelled ``model-2`` is a row nobody can act on.
    """

    name: str = ""


@dataclasses.dataclass(frozen=True)
class TargetConfig:
    """What to evaluate it *on*: an environment, or a dataset to replay."""

    env: str = "synthetic/reach"
    dataset: str = ""
    task: str = ""
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class SuiteConfig:
    """Which suite, or an inline one.

    ``cells`` present means an inline suite and ``name`` becomes its label; that
    is how a paper-specific evaluation lives in one file beside its results.
    """

    name: str = "core"
    cells: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    dimensions: list[str] = dataclasses.field(default_factory=list)
    baselines: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """How much to run, and how reproducibly."""

    episodes: int = 20
    seeds: list[int] = dataclasses.field(default_factory=lambda: [0])
    horizon: int = 200
    root_seed: int = 0
    record: int = 2
    save_trajectories: bool = False
    budget_seconds: float | None = None
    control_hz: float = 10.0
    baselines: bool = True


@dataclasses.dataclass(frozen=True)
class OutputConfig:
    """Where the run directory goes. Excluded from the config hash."""

    dir: str = "runs"
    name: str = ""


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """One evaluation, or one benchmark, fully specified.

    ``model`` and ``models`` are the single- and multi-model forms of the same
    field. Giving both is an error rather than a precedence rule: a config where
    one of the two silently wins is a config that runs something other than what
    it appears to say.
    """

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    models: list[NamedModelConfig] = dataclasses.field(default_factory=list)
    target: TargetConfig = dataclasses.field(default_factory=TargetConfig)
    suite: SuiteConfig = dataclasses.field(default_factory=SuiteConfig)
    run: RunConfig = dataclasses.field(default_factory=RunConfig)
    output: OutputConfig = dataclasses.field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        if self.models and self.model.target:
            raise ValueError(
                "a config sets both [model] and [[models]]; use one or the other -- "
                "[model] for a single evaluation, [[models]] for a benchmark"
            )
        names = [m.name for m in self.models]
        if any(not n for n in names):
            raise ValueError("every entry in [[models]] needs a name for its leaderboard row")
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate model name(s) in [[models]]: {duplicates}")

    @property
    def is_benchmark(self) -> bool:
        """Whether this config describes several models rather than one."""
        return bool(self.models)


def _is_dataclass_type(annotation: Any) -> bool:
    return dataclasses.is_dataclass(annotation) and isinstance(annotation, type)


def _is_dataclass_list(annotation: Any) -> bool:
    """Whether a field is ``list[SomeDataclass]`` -- what ``[[models]]`` becomes."""
    if typing.get_origin(annotation) is not list:
        return False
    args = typing.get_args(annotation)
    return bool(args) and _is_dataclass_type(args[0])


def _resolve_hints(cls: type) -> dict[str, Any]:
    """Field annotations, with ``from __future__ import annotations`` undone."""
    import xevals.core.config as module

    return typing.get_type_hints(cls, vars(module))


def _unwrap(annotation: Any) -> Any:
    """The interesting half of ``X | None``."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def coerce(annotation: Any, raw: Any) -> Any:
    """Turn a TOML or command-line value into what the field declares it holds."""
    annotation = _unwrap(annotation)
    origin = typing.get_origin(annotation)
    if annotation is Any or annotation is None or raw is None:
        return raw
    if origin in (tuple, list) or annotation in (tuple, list):
        args = typing.get_args(annotation)
        items = raw if isinstance(raw, (list, tuple)) else [raw]
        if args:
            items = [coerce(args[0], v) for v in items]
        return tuple(items) if (origin is tuple or annotation is tuple) else list(items)
    if origin is dict or annotation is dict:
        return dict(raw)
    if annotation is bool:
        if isinstance(raw, str):
            if raw.lower() not in ("true", "false", "1", "0", "yes", "no"):
                raise ValueError(f"expected a boolean, got {raw!r}")
            return raw.lower() in ("true", "1", "yes")
        return bool(raw)
    if annotation in (int, float, str):
        return annotation(raw)
    return raw


def from_dict(data: Mapping[str, Any], cls: type = ExperimentConfig) -> Any:
    """Build a config tree, rejecting anything the schema does not declare."""
    hints = _resolve_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"{cls.__name__} has no field(s) {', '.join(repr(u) for u in unknown)}; "
            f"valid keys are {sorted(known)}"
        )
    built: dict[str, Any] = {}
    for name, value in data.items():
        annotation = _unwrap(hints[name])
        if _is_dataclass_type(annotation) and isinstance(value, Mapping):
            built[name] = from_dict(value, annotation)
        elif _is_dataclass_list(annotation) and isinstance(value, (list, tuple)):
            # A TOML array-of-tables such as [[models]]: each entry is a nested
            # config in its own right, and gets the same unknown-key checking.
            item_type = typing.get_args(annotation)[0]
            built[name] = [from_dict(item, item_type) for item in value]
        else:
            built[name] = coerce(hints[name], value)
    return cls(**built)


def to_dict(config: Any) -> dict[str, Any]:
    """A JSON-safe mapping, round-trippable through :func:`from_dict`."""
    return dataclasses.asdict(config)


def _parse_scalar(text: str) -> Any:
    """Best-effort literal, for a value landing in an untyped dict."""
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_parse_scalar(p.strip()) for p in inner.split(",") if p.strip()] if inner else []
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` strings onto a nested mapping.

    A path the schema does not declare raises, listing what it does declare --
    except below an untyped ``dict`` field such as ``model.kwargs``, where any
    key is legitimate and the value is parsed as a literal.
    """
    data = json.loads(json.dumps(data))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not `key=value`; e.g. run.episodes=50")
        path, _, raw = item.partition("=")
        parts = path.strip().split(".")
        node, cls = data, ExperimentConfig
        for depth, part in enumerate(parts[:-1]):
            hints = _resolve_hints(cls) if cls is not None else {}
            if cls is not None and part not in hints:
                raise ValueError(
                    f"override {item!r}: {cls.__name__} has no field {part!r}; "
                    f"valid keys are {sorted(hints)}"
                )
            annotation = _unwrap(hints.get(part)) if cls is not None else None
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(
                    f"override {item!r}: {'.'.join(parts[: depth + 1])} is a value, not a section"
                )
            cls = annotation if _is_dataclass_type(annotation) else None
        leaf = parts[-1]
        if cls is not None:
            hints = _resolve_hints(cls)
            if leaf not in hints:
                raise ValueError(
                    f"override {item!r}: {cls.__name__} has no field {leaf!r}; "
                    f"valid keys are {sorted(hints)}"
                )
            node[leaf] = coerce(hints[leaf], _parse_scalar(raw.strip()))
        else:
            node[leaf] = _parse_scalar(raw.strip())
    return data


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Depth-first merge: a section in the child updates, it does not replace."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load(path: str | Path, *, overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Read a TOML config, follow one level of ``extends``, and apply overrides."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no config at {path}")
    data = tomllib.loads(path.read_text())
    parent = data.pop("extends", None)
    if parent is not None:
        base = tomllib.loads((path.parent / parent).resolve().read_text())
        base.pop("extends", None)
        data = _merge(base, data)
    return from_dict(apply_overrides(data, overrides))


def config_hash(config: Any) -> str:
    """A short digest of everything that defines the run.

    Excludes ``output``: two runs that differ only in where they are written are
    the same experiment, and should not look like different ones in a table --
    nor land in differently-named directories.
    """
    payload = to_dict(config)
    payload.pop("output", None)
    digest = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(digest).hexdigest()[:12]


def save(config: ExperimentConfig, directory: str | Path, *, source: Path | None = None) -> Path:
    """Write the resolved config as ``config.json``, keeping the source beside it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "config.json"
    target.write_text(json.dumps(to_dict(config), indent=2, default=str) + "\n")
    if source is not None and Path(source).exists():
        (directory / "config.toml").write_text(Path(source).read_text())
    return target
