"""Figures and numbers for the robots guide.

Writes two stills per arm, a filmstrip of a Panda pick, and the table of what the
scripted demonstrator actually achieves. The last one is why this is a script
and not a hand-written table: the gate's credibility rests on those numbers
being measured rather than remembered.

The two stills are the two cameras, and the pairing is the point: ``scene`` is
the arm as a reader wants to see it, ``pov`` is the arm as a policy sees it.
Everything that is scored goes through the second one.

    python tools/make_robot_docs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from xevals import robots
from xevals.sim import ManipulationEnv

OUT = Path("docs/assets/robots")
STILL = 420
SEEDS = 12
ORDER = ("panda", "fr3", "ur5e", "yam", "so101")


def stills() -> None:
    for name in ORDER:
        task = "pick" if "pick" in robots.ROBOTS[name].tasks else "reach"
        for camera, suffix in (("scene", ""), ("pov", "-pov")):
            env = ManipulationEnv(name, task, image_size=STILL, camera=camera)
            env.reset(seed=1)
            for _ in range(8):
                env.step(env.optimal_action)
            path = OUT / f"{name}{suffix}.png"
            Image.fromarray(env.render()).save(path, optimize=True)
            print(path.name, path.stat().st_size // 1024, "kB")


def filmstrip() -> None:
    """The hero image: what the evaluation camera sees during a pick."""
    env = ManipulationEnv("panda", "pick", image_size=300)
    env.reset(seed=0)
    frames, keep = [], {0, 8, 14, 20, 26}
    for step in range(28):
        if step in keep:
            frames.append(env.render())
        env.step(env.optimal_action)
    Image.fromarray(np.concatenate(frames, axis=1)).save(OUT / "panda-pick.png", optimize=True)
    print("panda-pick.png", (OUT / "panda-pick.png").stat().st_size // 1024, "kB")


def wrist_strip() -> None:
    """What the camera on the hand sees, for the arms that have one."""
    shots = []
    for name in ORDER:
        if not robots.ROBOTS[name].has_gripper:
            continue
        task = "pick" if "pick" in robots.ROBOTS[name].tasks else "reach"
        env = ManipulationEnv(name, task, image_size=200, camera="wrist")
        env.reset(seed=1)
        for _ in range(10):
            env.step(env.optimal_action)
        shots.append(np.asarray(env.render()))
    Image.fromarray(np.concatenate(shots, axis=1)).save(OUT / "wrist.png", optimize=True)
    print("wrist.png", (OUT / "wrist.png").stat().st_size // 1024, "kB")


def demonstrator() -> None:
    tasks = ("reach", "push", "pick", "place")
    scores: dict[tuple[str, str], tuple[int, int]] = {}
    for robot in ORDER:
        for task in robots.ROBOTS[robot].tasks:
            env = ManipulationEnv(robot, task, image_size=32)
            wins, steps = 0, []
            for seed in range(SEEDS):
                env.reset(seed=seed)
                done, info = False, {}
                while not done:
                    _obs, _reward, done, info = env.step(env.optimal_action)
                if info["success"]:
                    wins += 1
                    steps.append(env._t)
            scores[robot, task] = (wins, int(np.median(steps)) if steps else 0)
            print(f"{robot:6s} {task:6s} {wins}/{SEEDS}", flush=True)

    lines = [
        f"Success rate over {SEEDS} seeds, with the median number of steps a",
        "solved episode took.",
        "",
        "| Robot | " + " | ".join(t for t in tasks) + " |",
        "|---|" + "---|" * len(tasks),
    ]
    for robot in ORDER:
        cells = []
        for task in tasks:
            if (robot, task) not in scores:
                cells.append("not offered")
                continue
            wins, steps = scores[robot, task]
            cells.append(f"{wins / SEEDS:.0%} ({steps} steps)")
        lines.append(f"| `{robot}` | " + " | ".join(cells) + " |")
    total = sum(w for w, _ in scores.values())
    lines += ["", f"Overall: {total} of {len(scores) * SEEDS} episodes solved."]
    (OUT / "demonstrator.md").write_text("\n".join(lines) + "\n")
    print("demonstrator.md written")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    stills()
    filmstrip()
    wrist_strip()
    demonstrator()
