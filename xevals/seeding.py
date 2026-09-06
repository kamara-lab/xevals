"""Seeds, and the determinism contract that rests on them.

An evaluation that cannot be re-run is a screenshot. Every source of randomness
xevals owns -- episode resets, perturbation samples, bootstrap resamples,
baseline actions -- is drawn from a :class:`numpy.random.SeedSequence` spawned
from the run's single root seed, so the whole run is a pure function of that
integer plus the config.

The contract, stated so it can be tested:

* ``evaluate(...)`` twice with the same config and seeds produces identical
  metric values, bit for bit, excluding wall-clock timings.
* A perturbation is a pure function of ``(seed, severity, input)``. Applying
  ``visual/blur`` at severity 0.4 to the same frame always gives the same frame.
* A cell's ``k``-th episode uses ``derive(root, cell_name, seed, k)``, so adding
  a cell to a suite does not renumber the episodes of the cells beside it. That
  is what makes two runs with different suites still comparable on the cells
  they share.

What xevals cannot promise is the *model's* determinism. A torch policy with
non-deterministic kernels, or a remote endpoint, will move; :func:`set_seed`
does what it can (it seeds torch and jax when they are already imported) and
the consistency dimension measures the rest rather than pretending it away.
"""

from __future__ import annotations

import hashlib
import random
import sys

import numpy as np

__all__ = ["derive", "seed_sequence", "seeds_for", "set_seed", "spawn"]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and any of torch/jax that is *already* imported.

    Deliberately does not import torch or jax to seed them: making the seeding
    call pull in a two-second import is worse than the problem it solves. If the
    caller has imported them, they are here in ``sys.modules`` and get seeded.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch = sys.modules.get("torch")
    if torch is not None:  # pragma: no cover - depends on the caller's imports
        torch.manual_seed(seed)
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def derive(*parts: object) -> int:
    """A stable 63-bit seed from any hashable description of a position in a run.

    Uses BLAKE2b rather than :func:`hash`, whose string hashing is randomised
    per process -- the exact failure this function exists to prevent, and one
    that only shows up as "the numbers moved between machines".
    """
    payload = "\x1f".join(repr(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") >> 1


def seed_sequence(seed: int, *parts: object) -> np.random.SeedSequence:
    """A :class:`~numpy.random.SeedSequence` for one named position in a run."""
    return np.random.SeedSequence([seed, derive(*parts)] if parts else [seed])


def spawn(seed: int, *parts: object) -> np.random.Generator:
    """A fresh generator for one named position. The workhorse of the library."""
    return np.random.default_rng(seed_sequence(seed, *parts))


def seeds_for(root: int, n: int, *parts: object) -> list[int]:
    """``n`` independent integer seeds below one root, for episodes in a cell."""
    ss = seed_sequence(root, *parts)
    return [int(s.generate_state(1, dtype=np.uint32)[0]) for s in ss.spawn(n)]
