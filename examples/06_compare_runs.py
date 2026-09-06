"""Several runs into a leaderboard, an overlaid radar, and a diff.

Aggregation is where an evaluation library earns its keep: one run is a fact
about one model, and the comparison is what anyone actually wants. The mean
column is a summary and not a ranking anyone should defend -- which is why the
per-dimension columns sit beside it, and why the diff flags only differences the
error bars support.

    python examples/06_compare_runs.py
"""

from __future__ import annotations

from _common import outputs, setup
from policies import ImagePolicy, state_policy

import xevals
from xevals.results import Result, compare, leaderboard, markdown_table


class BlindPolicy:
    """Half the image policy: it finds the goal and ignores where it is."""

    def act(self, obs, *, instruction=None):
        import numpy as np

        return np.array([0.4, 0.4], dtype=np.float32)

    def describe(self):
        return {"adapter": "example", "module": "BlindPolicy"}


def main() -> int:
    settings = setup("06_compare_runs")
    directories = []
    for name, model in (
        ("state", xevals.wrap(state_policy)),
        ("image", ImagePolicy()),
        ("blind", BlindPolicy()),
    ):
        result = xevals.evaluate(model, "synthetic/reach", suite="robustness", name=name,
                                verbose=False, **settings)
        directories.append(result.directory)
        print(f"{name:8s} {result.summary().splitlines()[0]}")

    print()
    print(markdown_table(*leaderboard(directories)))

    try:
        from xevals.plots import leaderboard_radar

        path = leaderboard_radar(
            [Result.load(d) for d in directories[:2]], outputs("06_compare_runs") / "radar.png"
        )
        print(f"\noverlaid radar: {path}")
    except Exception as exc:  # noqa: BLE001 - a missing extra must not fail the example
        print(f"\nradar skipped ({type(exc).__name__}: {exc})")

    print()
    print("Diff, most significant first:")
    headers, rows = compare(directories[0], directories[1])
    print(markdown_table(headers, rows[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
