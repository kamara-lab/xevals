"""Bounded CPU episode groups using MuJoCo's existing GIL-releasing calls.

Only independent, explicitly stateless policy rows are admitted. Inference stays
on the coordinator; each worker owns a separate environment, model and mjData.
The scalar measurement path records the same contacts, violations and outcomes.
Rendering and observation perturbations stay on the coordinator. No batch timing
is disguised as single-action latency.
"""
from __future__ import annotations

import copy
import time
from contextlib import ExitStack
from typing import Any

from xevals.core.errors import CapabilityMissing
from xevals.core.types import Trajectory
from xevals.environments import envs
from xevals.evaluation.runner import (
    CellResult,
    _build_env,
    _close,
    _episode_seeds,
    _record_transition,
)
from xevals.integrations.adapters import validate_action, validate_batch


def check_policy(model: Any) -> None:
    if getattr(model, "batch_mode", None) != "stateless" or not callable(
        getattr(model, "act_batch", None)
    ):
        raise ValueError("cpu_batch requires a stateless BatchPolicy.act_batch implementation")
    device = getattr(model, "device", "cpu")
    if str(device).split(":")[0] != "cpu":
        raise ValueError("cpu_batch requires CPU inference; wrap the model with device='cpu'")


def check_environment(env: Any) -> None:
    from xevals.environments.sim import ManipulationEnv

    if not isinstance(env, ManipulationEnv) or env.backend != "mujoco":
        raise ValueError("cpu_batch supports built-in MuJoCo manipulation environments only")


def _infer(model, lanes, observations, instructions, stats):
    """Bisect raised batch calls so one bad observation does not fail other rows."""
    if not lanes:
        return {}, {}
    start = time.perf_counter()
    try:
        actions = validate_batch(
            model.act_batch([observations[i] for i in lanes],
                            instructions=[instructions[i] for i in lanes]), len(lanes),
        )
    except Exception as exc:
        stats.append({"size": len(lanes), "duration_ms": (time.perf_counter() - start) * 1e3,
                      "failed": True})
        if len(lanes) == 1:
            return {}, {lanes[0]: exc}
        middle = len(lanes) // 2
        left, le = _infer(model, lanes[:middle], observations, instructions, stats)
        right, re = _infer(model, lanes[middle:], observations, instructions, stats)
        return {**left, **right}, {**le, **re}
    stats.append({"size": len(lanes), "duration_ms": (time.perf_counter() - start) * 1e3,
                  "failed": False})
    return dict(zip(lanes, actions, strict=True)), {}


def run_batch_cell(model, make_env, cell, *, pool, batch_size, root_seed=0, seeds=(0,),
                   episodes=20, horizon=200, record=0, budget=None, on_step=None) -> CellResult:
    result = CellResult(cell)
    episode_seeds = _episode_seeds(root_seed, seeds, episodes)
    position = 0
    while position < len(episode_seeds):
        if budget is not None and budget.exhausted:
            result.skipped = "budget exhausted"
            break
        count = min(batch_size, len(episode_seeds) - position)
        if budget is not None and budget.episodes is not None:
            count = min(count, max(0, budget.episodes - budget._spent))
        group_seeds = episode_seeds[position:position + count]
        with ExitStack() as closing:
            environments, trajectories, observations, limits = [], [], [], []
            # Construct before resetting: reject factories that share mutable objects.
            owners = set()
            for seed in group_seeds:
                env, reason = _build_env(make_env, cell, seed=seed)
                closing.callback(_close, env)
                inner = env.env if isinstance(env, envs.PerturbedEnv) else env
                check_environment(inner)
                identities = {id(inner), id(inner.model), id(inner.data), id(inner._ik_data)}
                if owners & identities:
                    raise ValueError("cpu_batch factory reused mutable MuJoCo state")
                owners.update(identities)
                if reason:
                    result.skipped = reason
                    return result
                environments.append(env)
                limits.append(envs.limits_of(env))
                trajectories.append(Trajectory(
                    seed=seed, split=cell.split,
                    perturbation=None if cell.perturbation in ("", "none") else cell.perturbation,
                    frames=[] if position + len(trajectories) < record else None,
                    extra={"limits_declared": limits[-1] is not None, "execution": "cpu_batch"},
                ))
                observations.append(None)
            active = []

            def fail(i, exc, trajectories=trajectories):
                traj = trajectories[i]
                message = f"{type(exc).__name__}: {exc}"
                # Match run_cell's failed-episode semantics. An incomplete
                # episode cannot certify safety merely because it stopped early.
                trajectories[i] = Trajectory(
                    seed=traj.seed, split=traj.split, perturbation=traj.perturbation,
                    success=False, extra={"error": message, "limits_declared": False,
                                          "execution": "cpu_batch"},
                )
                result.errors.append(message)

            for i, env in enumerate(environments):
                try:
                    observations[i] = env.reset(seed=group_seeds[i])
                    if env.observation_mode == "image" and "image" not in observations[i]:
                        raise RuntimeError("image observations require a working MuJoCo renderer")
                    trajectories[i].instruction = observations[i].get("instruction")
                    active.append(i)
                except Exception as exc:
                    fail(i, exc)
            instructions = [t.instruction for t in trajectories]
            batch_calls = []
            for step in range(horizon):
                if not active:
                    break
                for i in active:
                    trajectories[i].obs.append(copy.deepcopy(observations[i]))
                    if trajectories[i].frames is not None:
                        frame = observations[i].get("image")
                        if frame is None:
                            frame = environments[i].render()
                        if frame is not None:
                            trajectories[i].frames.append(frame.copy())
                actions, errors = _infer(model, active, observations, instructions, batch_calls)
                futures = {}
                for i in active:
                    try:
                        if i in errors:
                            raise errors[i]
                        actions[i] = validate_action(actions[i], environments[i].action_dim)
                        confidence = getattr(model, "confidence", None)
                        if callable(confidence):
                            try:
                                trajectories[i].confidences.append(float(confidence(observations[i])))
                            except CapabilityMissing:
                                pass
                        env = environments[i]
                        # State stepping preserves custom step overrides. Vision
                        # stepping must never enter a GL context on a worker.
                        advance = env.advance if env.observation_mode == "image" else env.step
                        futures[i] = pool.submit(advance, actions[i].copy())
                    except Exception as exc:
                        fail(i, exc)
                following = []
                # Consume in episode order, never completion order.
                for i, future in futures.items():
                    try:
                        transition = future.result()
                        if environments[i].observation_mode == "image":
                            observation = environments[i].observe()
                            if "image" not in observation:
                                raise RuntimeError(
                                    "image observations require a working MuJoCo renderer"
                                )
                            transition = (observation, *transition)
                        obs, done, success = _record_transition(
                            trajectories[i], environments[i], actions[i], transition,
                            step, limits[i], on_step,
                        )
                        observations[i] = obs
                        trajectories[i].success = success
                        if not done:
                            following.append(i)
                    except Exception as exc:
                        fail(i, exc)
                active = following
            for i, traj in enumerate(trajectories):
                if observations[i] is not None and "error" not in traj.extra:
                    traj.obs.append(copy.deepcopy(observations[i]))
                    if traj.frames is not None and traj.actions:
                        frame = observations[i].get("image")
                        if frame is None:
                            frame = environments[i].render()
                        if frame is not None:
                            traj.frames.append(frame.copy())
                traj.extra["environment_timings"] = dict(environments[i].timings)
            # Store each batch call once, not once per row. It measures a group,
            # and consumers must not mistake repeated rows for additional samples.
            if trajectories:
                trajectories[0].extra["batch_calls"] = batch_calls
            result.trajectories.extend(trajectories)
            if budget is not None:
                budget.spend(len(trajectories))
        position += count
    return result
