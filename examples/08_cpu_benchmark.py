"""Compare serial and CPU batching with state and image observations, using existing dependencies.

Run from the repository root:
    python examples/08_cpu_benchmark.py --episodes 32 --horizon 100 --repeats 3

Each configuration runs in a fresh process with one BLAS/Torch inference thread.
The model/scene is warmed before timing. Output is JSON (one record per sample),
including end-to-end wall time, peak process RSS, and component work durations.
Baselines, video and disk output are disabled equally in both modes. The scripted
controller and untrained small network exercise throughput, not model quality.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

# Permit direct invocation from a checkout without an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def measure(args):
    import resource
    import time

    import numpy as np

    import xevals
    from xevals.environments import robots

    if not robots.assets_available("panda"):
        raise RuntimeError("Panda assets must be cached; this benchmark never downloads them")
    if args.policy == "torch":
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        # Panda state: 7 positions, 7 velocities, TCP, grip, object, goal.
        model = torch.nn.Sequential(torch.nn.Linear(24, 64), torch.nn.Tanh(),
                                    torch.nn.Linear(64, 4), torch.nn.Tanh())
        if args.observation_mode == "image":
            class VisionPolicy(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.network = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 8, 5, stride=4), torch.nn.ReLU(),
                        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                        torch.nn.Linear(8, 4), torch.nn.Tanh())

                def forward(self, obs):
                    return self.network(obs["image"].float().permute(0, 3, 1, 2) / 255)
            model = VisionPolicy()
        input_key = "image" if args.observation_mode == "image" else "state"
        policy = xevals.wrap(model, device="cpu",
                            dict_input=args.observation_mode == "image",
                            feature_keys=(input_key,),
                            action_dim=4, batch_mode="stateless")
    else:
        def act(obs):
            if args.observation_mode == "image":
                return np.r_[obs["image"].mean(axis=(0, 1)) / 255 - 0.5, 1].astype(np.float32)
            delta = np.asarray(obs["goal"]) - np.asarray(obs["tcp"])
            return np.r_[np.clip(delta * 10, -1, 1), 1.0].astype(np.float32)

        policy = xevals.wrap(act, action_dim=4, batch_fn=lambda obs, instructions=None:
                            np.stack([act(o) for o in obs]))
    options = dict(
        model=policy, env=f"mujoco/panda-{args.task}", observation_mode=args.observation_mode,
        suite={"name": "cpu-throughput", "cells": [{"name": "clean"}],
               "dimensions": ["accuracy", "safety", "efficiency"]},
        execution=args.mode, batch_size=args.batch_size, workers=args.workers,
        record=0, out=None, baselines=False, verbose=False,
    )
    xevals.evaluate(**options, episodes=1, horizon=2)
    started = time.perf_counter()
    result = xevals.evaluate(**options, episodes=args.episodes, horizon=args.horizon)
    wall = time.perf_counter() - started
    execution = result.run["execution"]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024**2 if platform.system() == "Darwin" else 1024)
    print(json.dumps({
        "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "environment": result.run["environment"],
        "policy": args.policy, "task": args.task, "mode": args.mode,
        "observation_mode": args.observation_mode,
        "mujoco_gl": os.environ.get("MUJOCO_GL", "default"),
        "batch_size": args.batch_size, "workers": execution["workers"],
        "episodes": args.episodes, "horizon": args.horizon, "wall_s": wall,
        "steps_per_s": execution["control_steps"] / wall, "peak_rss_mb": peak_mb,
        "control_steps": execution["control_steps"], "inference_s": execution["inference_s"],
        "environment_work_s": execution["environment_work_s"],
        "scoring_s": execution["scoring_s"], "run_hash": result.config_hash(),
        "errors": sum("error" in t.extra for t in result.trajectories.get("clean", [])),
    }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tasks", default="reach,pick")
    parser.add_argument("--policies", default="scripted,torch")
    parser.add_argument("--observation-modes", default="state,image")
    parser.add_argument("--observation-mode", choices=("state", "image"), default="state")
    parser.add_argument("--mode", choices=("serial", "cpu_batch"))
    parser.add_argument("--task", default="reach")
    parser.add_argument("--policy", default="scripted")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if min(args.episodes, args.horizon, args.repeats) < 1:
        parser.error("episodes, horizon and repeats must be positive")
    if args.mode:
        measure(args)
        return
    configs = [("serial", 1, None)] + [
        ("cpu_batch", batch, workers)
        for batch in (1, 8, 32) for workers in (1, 2, 4, 8)
        if workers <= batch and workers <= (os.cpu_count() or 1)
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    for task in args.tasks.split(","):
        for policy in args.policies.split(","):
            for mode, batch, workers in configs:
                for _ in range(args.repeats):
                    command = [sys.executable, str(Path(__file__).resolve()),
                               "--mode", mode, "--task", task, "--policy", policy,
                               "--episodes", str(args.episodes), "--horizon", str(args.horizon),
                               "--batch-size", str(batch)]
                    if workers is not None:
                        command += ["--workers", str(workers)]
                    for observation_mode in args.observation_modes.split(","):
                        subprocess.run(command + ["--observation-mode", observation_mode],
                                       env=env, check=True)


if __name__ == "__main__":
    main()
