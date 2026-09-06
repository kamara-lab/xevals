# Randomness

An evaluation that cannot be re-run is a screenshot. Here is exactly what is
reproducible, what is not, and which of those `xevals` measures rather than
assumes away.

## The contract

Every source of randomness `xevals` owns (episode resets, perturbation samples,
bootstrap resamples, baseline actions) is drawn from a
`numpy.random.SeedSequence` spawned from the run's single root seed. So:

- `evaluate(...)` twice with the same config and seeds gives identical metric
  values, bit for bit, **excluding wall-clock timings**;
- a perturbation is a pure function of `(seed, severity, input)`;
- a cell's *k*-th episode uses `derive(root, seed, k)`, the same in every
  cell, which is what makes pairing work;
- bootstrap intervals use a fixed seed, so reopening a results file gives the same
  error bars rather than new ones.

Seeds are derived with BLAKE2b rather than `hash()`, whose string hashing is
randomised per process. That is the exact failure this design prevents, and it
shows up only as "the numbers moved between machines".

## What is not reproducible

**Wall-clock latency.** The efficiency dimension is the one thing a seed cannot
fix, and the test suite excludes it from the bit-for-bit check rather than
pretending otherwise.

**The model.** A torch policy with non-deterministic kernels, or a remote
endpoint, will move. `set_seed` does what it can: it seeds Python, NumPy, and
torch or jax **if they are already imported**, deliberately not importing them
just to seed them. The rest is *measured* rather than assumed:

```
consistency/determinism    fraction of repeated seeds that reproduced the episode
consistency/seed_std       spread of success rate across seeds
```

A model that is not deterministic at a fixed seed cannot be evaluated
reproducibly by anyone, including its own authors. `xevals` reports that as a
number before anything is concluded from a comparison of two runs.

## Seeds and statistics

Three seeds is the practical floor, and it is enforced rather than advised:
`consistency/seed_std` reports `null` with the reason "2 seeds; a spread needs at
least three" rather than presenting the spread of two numbers as a distribution.

Every per-episode metric carries a 95 % percentile bootstrap interval and its
`n`. A success rate over 20 episodes has a half-width of roughly 10 points, and a
table that prints `0.65` without saying so invites a comparison the data does not
support.

::: xevals.seeding
    options:
      heading_level: 2
      members: false
