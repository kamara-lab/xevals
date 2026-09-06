"""Evaluating against a recorded corpus, with no simulator.

Each recorded episode becomes a replay environment, so the same runner, the same
perturbations and the same metrics work on data. Action error against the
demonstrator is measurable; success under the model's own actions is not, because
the world does not respond -- and those metrics say so rather than guessing.

Uses the built-in recorder so it runs with no download. Point ``read_lerobot`` or
``read_hdf5`` at a real corpus to do the same thing with one.

    python examples/04_offline_dataset.py
"""

from __future__ import annotations

from _common import setup
from policies import state_policy

import xevals
from xevals import datasets


def main() -> int:
    episodes = datasets.synthetic_episodes("reach", episodes=8)
    print(f"read {len(episodes)} episodes, {sum(len(e) for e in episodes)} transitions")

    cycle = {"index": 0}

    def make_env(task=None, split="in"):
        episode = episodes[cycle["index"] % len(episodes)]
        cycle["index"] += 1
        return episode.to_env()

    settings = setup("04_offline_dataset")
    result = xevals.evaluate(
        xevals.wrap(state_policy), make_env, suite="core", baselines=False, **settings
    )

    print()
    print(result.table(kind="metrics"))
    print()
    for name in ("accuracy/success_rate", "accuracy/action_mse"):
        print(f"  {result.metric(name)}")
    print()
    print(
        "Action error is a real measurement offline. Success is not -- the\n"
        "replayed world does not respond -- and the metric says so instead of\n"
        "publishing the demonstrator's success as the model's.\n"
        "\n"
        "The zero action error is not a good score either: this policy uses the\n"
        "same rule the recorder did, so it reproduces the demonstrator exactly.\n"
        "Point the same run at a real corpus and the number becomes interesting."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
