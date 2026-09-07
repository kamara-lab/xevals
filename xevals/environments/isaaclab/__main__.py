"""Run or validate the native port inside the pinned Isaac Lab environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from xevals.environments.manipulation import TASKS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="reach")
    parser.add_argument("--policy", choices=("replay", "random"), default="replay")
    parser.add_argument("--renderer", choices=("none", "newton", "rtx"), default="newton")
    parser.add_argument(
        "--camera", choices=("pov", "wrist", "scene", "scene-wrist"), default="scene-wrist"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--out", type=Path, default=Path("runs/isaaclab-newton"))
    parser.add_argument(
        "--validate",
        action="store_true",
        help="check reset, free-object dynamics, images and demonstrator success",
    )
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.validate and args.policy != "replay":
        parser.error("--validate checks demonstrator success and requires --policy replay")

    app = None
    if args.renderer == "rtx":
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, enable_cameras=True).app
    try:
        from xevals import Dimension, evaluate
        from xevals.environments.isaaclab import IsaacLabEnv
        from xevals.evaluation.runner import random_policy, replay_policy
        from xevals.evaluation.suites import Cell, SuiteSpec

        with IsaacLabEnv(
            args.task,
            renderer=args.renderer,
            camera=args.camera,
            device=args.device,
            image_size=args.image_size,
        ) as env:
            if args.validate:
                validate_scene(env, args.seed)
            policy = (
                replay_policy(env)
                if args.policy == "replay"
                else random_policy(4, seed=args.seed)
            )
            # A clean-cell gate verifies the task before claiming robustness.
            suite = SuiteSpec(
                name=f"isaaclab-panda-{args.task}",
                cells=[
                    Cell(name="clean", metrics=["accuracy/success_rate", "accuracy/goal_distance"])
                ],
                dimensions=(Dimension.ACCURACY,),
                baselines=[],
            )
            result = evaluate(
                policy,
                env,
                suite=suite,
                episodes=args.episodes,
                seeds=(args.seed,),
                horizon=env.horizon,
                record=args.episodes if args.renderer != "none" else 0,
                baselines=False,
                out=args.out,
                control_hz=env.frame_rate,
                name=f"isaaclab-panda-{args.task}-{args.policy}",
            )
            if args.validate:
                value = result.cells["clean"]["accuracy/success_rate"].value
                if value is None or value < 1.0:
                    raise RuntimeError(f"demonstrator gate failed for {args.task}: success={value}")
                print(f"PASS: {args.task} demonstrator and scene checks")
    finally:
        if app is not None:
            app.close()
    return 0


def validate_scene(env, seed):
    """Integration checks against actual physics, not test doubles."""
    obs = env.reset(seed=seed)
    initial = env.state()
    env.reset(seed=seed)
    np.testing.assert_allclose(env.state(), initial, atol=1e-6, rtol=0)
    env.reset(state=initial)
    np.testing.assert_allclose(env.state(), initial, atol=1e-6, rtol=0)
    if env.config.renderer != "none":
        frame = obs["image"]
        assert frame.shape == (env.config.image_size, env.config.image_size, 3)
        assert frame.dtype == np.uint8 and frame.std() > 1, "blank camera output"
    # A lifted object must fall under gravity. This catches the pinned-object
    # shortcut in the old Newton adapter before any grasp scores are trusted.
    lifted = initial.copy()
    lifted[20] += 0.10  # object-pose z, after nine q and nine qd values
    env.reset(state=lifted)
    env._runtime.advance(env._tcp_cmd, 1.0)
    env._refresh()
    assert env.state()[20] < lifted[20] - 0.001, "the object did not fall under gravity"
    env.reset(seed=seed)
    print("PASS: repeatable reset, snapshot restore and free-object gravity")


if __name__ == "__main__":
    raise SystemExit(main())
