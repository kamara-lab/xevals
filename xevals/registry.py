"""One registry type, used by every namespace in the library.

Perturbations, metrics, adapters, environments and suites all want the same
four things: register by name, list what exists, describe an entry without
building it, and build one. Writing that five times is how the five copies
drift, so it is written once here and instantiated per namespace.

Two conventions the whole library relies on:

**Names are slash-namespaced** -- ``visual/blur``, ``accuracy/success_rate``,
``adapters/torch``. The prefix is the family, and :meth:`Registry.available`
can filter on it, which is what the CLI's ``list --family`` does.

**Entries build lazily.** The value stored is a zero-argument thunk returning
the constructor, so ``xevals.perturbations`` can name a PIL-backed perturbation
without importing PIL, and importing the library never costs an optional
dependency. The description is stored separately, so ``describe`` answers
without triggering the import at all.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

__all__ = ["Entry", "Registry"]

T = TypeVar("T")


@dataclass(frozen=True)
class Entry(Generic[T]):
    """What a registry stores for one name.

    Attributes:
        name: the slash-namespaced key.
        summary: one line, shown by ``xevals <namespace> list``.
        factory: called with the registration kwargs to build the object. May
            import an optional dependency; it is not called by ``describe``.
        requires: extras this entry needs, for ``xevals doctor`` and for the
            "skipped, because" reason when one is absent.
        meta: free-form facts a caller may want without building -- a metric's
            dimension, a perturbation's severity ladder.
    """

    name: str
    summary: str
    factory: Callable[..., T]
    requires: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def family(self) -> str:
        """The part before the first slash, or ``""`` for an unnamespaced entry."""
        return self.name.split("/")[0] if "/" in self.name else ""


class Registry(Generic[T]):
    """A named collection of lazily-built things.

    Args:
        namespace: what this registry holds, used in error messages and by the
            CLI (``xevals metrics list`` passes ``"metrics"``).
    """

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self._entries: dict[str, Entry[T]] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., T],
        *,
        summary: str = "",
        requires: tuple[str, ...] = (),
        **meta: Any,
    ) -> Callable[..., T]:
        """Add an entry. Returns ``factory``, so this also works as a decorator.

        Raises:
            ValueError: if ``name`` is already registered. Silent replacement is
                how a typo in a plugin quietly shadows a built-in metric.
        """
        if name in self._entries:
            raise ValueError(f"{self.namespace}: {name!r} is already registered")
        summary = summary or (factory.__doc__ or "").strip().split("\n")[0]
        self._entries[name] = Entry(name, summary, factory, tuple(requires), dict(meta))
        return factory

    def available(self, family: str | None = None) -> list[str]:
        """Registered names, sorted; optionally only those in one family."""
        names = sorted(self._entries)
        if family is None:
            return names
        return [n for n in names if n.split("/")[0] == family]

    def families(self) -> list[str]:
        """The distinct namespace prefixes present, sorted."""
        return sorted({e.family for e in self._entries.values() if e.family})

    def entry(self, name: str) -> Entry[T]:
        """The raw entry, without building it.

        Raises:
            KeyError: naming the closest alternatives, because a registry error
                should end the search rather than start one.
        """
        if name not in self._entries:
            raise KeyError(f"unknown {self.namespace} {name!r}; {self._suggest(name)}")
        return self._entries[name]

    def describe(self, name: str | None = None) -> dict[str, Any] | dict[str, dict[str, Any]]:
        """What an entry is, without building it. Omit ``name`` for all of them."""
        if name is None:
            return {n: self.describe(n) for n in self.available()}  # type: ignore[misc]
        e = self.entry(name)
        return {
            "name": e.name,
            "family": e.family,
            "summary": e.summary,
            "requires": list(e.requires),
            **e.meta,
        }

    def create(self, name: str, **kwargs: Any) -> T:
        """Build the entry named ``name``. Keyword arguments reach its factory."""
        return self.entry(name).factory(**kwargs)

    def _suggest(self, name: str) -> str:
        """A short, useful tail for a KeyError: near misses, else the families."""
        import difflib

        close = difflib.get_close_matches(name, self._entries, n=3, cutoff=0.5)
        if close:
            return "did you mean " + " or ".join(repr(c) for c in close) + "?"
        family = name.split("/")[0]
        siblings = self.available(family)
        if siblings:
            return f"{family!r} has {', '.join(repr(s) for s in siblings[:8])}"
        return f"known {self.namespace}: {', '.join(self.available()[:12])}"

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.available())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry({self.namespace!r}, {len(self)} entries)"
