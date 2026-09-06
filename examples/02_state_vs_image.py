"""Two policies, one suite: what a second dimension buys you.

Both solve the task. One reads the state vector and one reads pixels, and on task
success alone they are indistinguishable. Under perturbation they are not: the
state policy is invisible to every visual family by construction, and the image
policy is not.

That is the whole argument for this library, run rather than asserted. The
leaderboard at the end is what ``xevals compare`` prints.

    python examples/02_state_vs_image.py
"""

from __future__ import annotations

from _common import setup
from policies import ImagePolicy, state_policy

import xevals
from xevals.results import leaderboard, markdown_table


def main() -> int:
    settings = setup("02_state_vs_image")
    runs = []
    for name, model in (("state", xevals.wrap(state_policy)), ("image", ImagePolicy())):
        print(f"\n=== {name} policy ===")
        result = xevals.evaluate(model, "synthetic/reach", suite="robustness",
                                name=name, **settings)
        runs.append(result.directory)

    print()
    print(markdown_table(*leaderboard(runs)))
    print()
    print(
        "Same accuracy, different robustness. The state policy cannot see the\n"
        "perturbations, which is a true statement about it and not a compliment."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
