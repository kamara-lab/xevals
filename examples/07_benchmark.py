"""Several models in one run, under conditions that are identical by construction.

Three runs of ``evaluate`` and a spreadsheet look like a benchmark and are not.
This is: every model sees the same episodes from the same starting states, the
baselines and the replay gate are measured once because they belong to the
environment rather than to any model, and the comparison itself is written to
disk beside the runs it compares.

The last thing it prints is the useful one. A leaderboard says *which* model is
better; ``disagreements()`` says *where* -- the conditions that pull the models
apart beyond how they already differ nominally, which is where the next
experiment goes.

    python examples/07_benchmark.py
"""

from __future__ import annotations

import numpy as np
from _common import setup
from policies import ImagePolicy, state_policy

import xevals


class HalfBlind:
    """The image policy with a smaller colour tolerance: right idea, less margin.

    Included because two obviously different models make a boring benchmark. The
    interesting case is two models that are close, where the question is not
    which is better on average but under which conditions they differ at all.
    """

    def __init__(self) -> None:
        self.inner = ImagePolicy()
        self.inner.goal_tolerance = 0.05

    def act(self, obs, *, instruction=None):
        return self.inner.act(obs, instruction=instruction)

    def describe(self):
        return {"adapter": "example", "module": "HalfBlind", "goal_tolerance": 0.05}


class Still:
    """Zero actions. Present so the leaderboard has a floor drawn on it."""

    def act(self, obs, *, instruction=None):
        return np.zeros(2, dtype=np.float32)

    def describe(self):
        return {"adapter": "example", "module": "Still"}


def main() -> int:
    settings = setup("07_benchmark")
    settings.pop("record", None)

    bench = xevals.benchmark(
        {
            "state": xevals.wrap(state_policy),
            "image": ImagePolicy(),
            "image-tight": HalfBlind(),
            "still": Still(),
        },
        "synthetic/reach",
        suite="full",
        record=1,
        name="policies",
        **settings,
    )

    print()
    print("Side by side, one row per condition:")
    print(bench.table(kind="cells"))

    print()
    print("Where they disagree, beyond how they already differ on the clean cell:")
    rows = bench.disagreements()
    if not rows:
        print("  nowhere: every condition separates them by about as much as the clean one")
    for cell, best, worst, gap, excess in rows[:8]:
        print(f"  {cell:28s} {best:12s} over {worst:12s} gap {gap:.2f}  excess {excess:.2f}")

    print()
    print("Trade-offs across the four models:")
    for analysis in bench.tradeoffs():
        print("  " + analysis.summary().replace("\n", "\n  "))
    print(
        "\n  Four models is four points, and the analysis says so rather than\n"
        "  fitting a frontier to them. The evidence a single run can carry is\n"
        "  per cell, where there is an order of magnitude more of it:"
    )

    print()
    print("Trade-offs within one model, over its cells:")
    for analysis in bench["image"].tradeoffs():
        print("  " + analysis.summary().replace("\n", "\n  "))

    print()
    print(f"best overall     : {bench.best()}")
    print(f"best robustness  : {bench.best('robustness')}")
    print(f"comparison page  : {bench.directory / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
