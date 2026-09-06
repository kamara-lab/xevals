"""Settings and output paths shared by the examples, as xwm's ``_common`` does.

Every example reads its size from ``XEVALS_*`` environment variables so the same
script serves a five-second smoke test and a real evaluation without editing:

===================  =======  ==================================================
``XEVALS_EPISODES``   10       episodes per seed
``XEVALS_SEEDS``      3        how many seeds
``XEVALS_RECORD``     2        episodes per cell to keep frames for
``XEVALS_OUT``        ``examples/outputs``   where run directories go
===================  =======  ==================================================

CI runs them at ``XEVALS_EPISODES=2 XEVALS_SEEDS=1``; the figures published in the
documentation come from the defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["outputs", "setting", "setup"]


def setting(name: str, default: int) -> int:
    """One ``XEVALS_<NAME>`` integer setting, or its default."""
    raw = os.environ.get(f"XEVALS_{name.upper()}")
    return default if raw is None else int(raw)


def outputs(example: str) -> Path:
    """Where this example writes. Created if it does not exist."""
    root = Path(os.environ.get("XEVALS_OUT", "examples/outputs"))
    path = root / example
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup(example: str) -> dict:
    """The keyword arguments every example passes to :func:`xevals.evaluate`."""
    return {
        "episodes": setting("episodes", 10),
        "seeds": tuple(range(setting("seeds", 3))),
        "record": setting("record", 2),
        "out": outputs(example),
    }
