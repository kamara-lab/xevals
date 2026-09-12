---
title: xevals
description: Multi-dimensional evaluation of robot-learning models.
hide:
  - navigation
---

<div class="xevals-hero" markdown="0">
  <div class="xevals-hero__grid">
    <div class="xevals-hero__cell xevals-hero__cell--masked"></div>
    <div class="xevals-hero__cell xevals-hero__cell--masked"></div>
    <div class="xevals-hero__cell"></div>
    <div class="xevals-hero__cell"></div>
    <div class="xevals-hero__cell xevals-hero__cell--masked"></div>
    <div class="xevals-hero__cell"></div>
    <div class="xevals-hero__cell xevals-hero__cell--masked"></div>
    <div class="xevals-hero__cell"></div>
    <div class="xevals-hero__cell xevals-hero__cell--masked"></div>
  </div>
  <div>
    <div class="xevals-hero__word">xevals</div>
    <div class="xevals-hero__kicker">Robotics + AI Evaluations</div>
  </div>
  <p class="xevals-hero__tagline">
    Take any robot-learning model and measure it along seven dimensions, not one.
    Accuracy, robustness, safety, security, efficiency, generalisation and
    consistency, paired on seeds and written to a run directory you can diff.
  </p>
</div>

## Overview

Most robot-learning evaluations report one number: task success rate. That number
hides everything that decides whether a model is usable. Does it survive a camera
shift, a paraphrased instruction, a distractor on the table? Does it leave the
workspace on the way to a success? Can text painted on a box redirect it? What
does it cost to run at 10 Hz?

Each of those is measured, when it is measured at all, by a one-off script per
paper. So the numbers do not compare: not between labs, not between
methods, often not between two runs of the same model. `xevals` fixes the
vocabulary instead: **seven named dimensions, in a fixed order**, a fixed
severity ladder, paired baselines, and one run directory per evaluation.

<figure class="xevals-diagram" markdown="1">
  ![xevals flowchart: model to adapter, paired clean and perturbed evaluations, trajectories, metrics, seven dimensions, and saved report](assets/method-light.svg#only-light){ width="900" } ![xevals flowchart: model to adapter, paired clean and perturbed evaluations, trajectories, metrics, seven dimensions, and saved report](assets/method-dark.svg#only-dark){ width="900" }
  <figcaption>
    Clean and perturbed conditions share episode seeds, so changes in behaviour
    can be compared. Each run records metrics, seven dimension scores, and the
    provenance needed to interpret the results.
  </figcaption>
</figure>

```python
import xevals

policy = xevals.wrap(my_model)                  # torch / jax / LeRobot / HF / callable / HTTP

result = xevals.evaluate(
    policy, "synthetic/reach",     # or "mujoco/panda-pick", on a real arm
    suite="full", episodes=20, seeds=range(3),
    out="runs/pusht",
)
print(result.table())
```

## The seven dimensions

| # | Dimension | Question it answers |
|---|---|---|
| 1 | `accuracy` | Does it do the task? |
| 2 | `robustness` | Does it keep doing it under nuisance change? |
| 3 | `safety` | Does it stay inside physical and behavioural limits? |
| 4 | `security` | Can it be hijacked, or made to fail on purpose? |
| 5 | `efficiency` | What does it cost to run? |
| 6 | `generalization` | Does it transfer to unseen objects, scenes, instructions? |
| 7 | `consistency` | Are its answers stable? |

The order is fixed and load-bearing: it fixes table column order and the colour
each dimension gets in every figure the library draws. A dimension looks the same
in every chart `xevals` has ever produced.

## What a run measures, in one example

`examples/01_synthetic.py` runs the full suite against a pixel-conditioned
policy on the built-in world. It solves the task perfectly and stops solving it
the moment anything is in the way:

| dimension | score | what it is saying |
|---|---|---|
| accuracy | 0.80 | success rate 1.00 |
| robustness | 0.44 | occlusion and photometry break it |
| safety | 0.13 | under disturbance it leaves the workspace |
| security | 0.24 | a random patch, and scene text, redirect it |
| efficiency | 1.00 | microseconds per step |
| generalization | 1.00 | unseen colours and layouts are fine |
| consistency | 1.00 | it ignores language, so it is perfectly invariant to it |

A single success rate would have reported `1.00` and stopped. Three of those seven
columns are outright failures, and all three are the kind that only appear in
deployment.

## Design in one paragraph

The core depends on **numpy and the standard library**, and nothing else. torch,
jax, simulators, video codecs and matplotlib are extras that fail *when used*,
never at import. Models are wrapped through structural
[protocols](reference/types.md), never a base class, because a LeRobot policy, an
OpenVLA checkpoint, a JAX function and an HTTP endpoint have no common ancestor
and no prospect of one. Metrics are pure functions over recorded trajectories, so
the same function scores an online rollout and an offline dataset. And anything
unmeasurable is reported as `null` **with a reason**, never as zero.

## Where to go next

<div class="grid cards" markdown>

-   **[Installation](getting-started/installation.md)**

    The core, and which extra unlocks what.

-   **[Quickstart](getting-started/quickstart.md)**

    A full seven-dimension evaluation in under a minute, with no downloads.

-   **[What a run looks like](guides/example-output.md)**

    The real report, videos and leaderboard from a three-second run, embedded.

-   **[Dimensions](concepts/dimensions/index.md)**

    What each one asks, which metrics answer it, and how they are normalised.

-   **[Baselines and gates](concepts/baselines-and-gates.md)**

    Why a replay gate decides whether any of the other numbers mean anything.

-   **[Benchmarks](guides/benchmarks.md)**

    Several models in one run, paired on seeds, with one leaderboard.

</div>
