# API reference

The implementation is grouped into five packages:

| Package | Responsibility |
| --- | --- |
| `xevals.core` | Protocols, errors, registries, seeds, dimensions and configuration |
| `xevals.evaluation` | Suites, rollouts, benchmarks, metrics, perturbations and judges |
| `xevals.integrations` | Model adapters and dataset readers |
| `xevals.environments` | Synthetic and replay environments, robots and simulators |
| `xevals.reporting` | Results, trade-offs, charts, plots, media and HTML reports |

The CLI remains in `xevals.cli`. Common entry points are re-exported from
`xevals`: `evaluate`, `wrap`, `Result` and `Trajectory`.

Existing module imports remain supported: `xevals.runner` and
`xevals.evaluation.runner` refer to the same module. Use the grouped paths when
adding internal imports. The reference pages below retain their existing URLs.

## Contracts

<div class="grid cards" markdown>

-   **[`xevals.types`](types.md)**

    What xevals requires of a model, an environment, and a recorded episode.

-   **[`xevals.dimensions`](dimensions.md)**

    The seven dimensions, and how raw metrics become a score along one.

-   **[`xevals.registry`](registry.md)**

    One registry type, used by every namespace in the library.

-   **[`xevals.seeding`](seeding.md)**

    Seeds, and the determinism contract that rests on them.

</div>

## Measuring

<div class="grid cards" markdown>

-   **[`xevals.adapters`](adapters.md)**

    Wrapping any model so it satisfies one of the protocols.

-   **[`xevals.envs`](envs.md)**

    The built-in synthetic world, adapters, and wrappers.

-   **[`xevals.datasets`](datasets.md)**

    Reading recorded episodes, for evaluation without a simulator.

-   **[`xevals.perturbations`](perturbations.md)**

    The controlled changes a robustness or security claim rests on.

-   **[`xevals.metrics`](metrics.md)**

    Pure functions over trajectories, one registry, seven dimensions.

-   **[`xevals.judges`](judges.md)**

    Deciding whether a piece of text is right, refused, or hijacked.

</div>

## Running

<div class="grid cards" markdown>

-   **[`xevals.suites`](suites.md)**

    Which cells to run, on which splits, scored by which metrics.

-   **[`xevals.runner`](runner.md)**

    Running a suite, and the baselines that make it readable.

-   **[`xevals.bench`](bench.md)**

    Several models under identical conditions, with one leaderboard.

-   **[`xevals.config`](config.md)**

    TOML in, dataclass tree out, overrides applied loudly.

-   **[`xevals.cli`](cli.md)**

    ``xevals run|compare|report|list|doctor``.

</div>

## Reporting

<div class="grid cards" markdown>

-   **[`xevals.results`](results.md)**

    The run directory: what an evaluation leaves behind.

-   **[`xevals.tradeoffs`](tradeoffs.md)**

    Where one dimension is bought with another, and whether it is.

-   **[`xevals.media`](media.md)**

    Videos and GIFs of what the model saw, with the conditions on screen.

-   **[`xevals.plots`](plots.md)**

    Figures, in the kamara palette, with the categorical limits enforced.

-   **[`xevals.report`](report.md)**

    One self-contained HTML page, openable offline.

</div>
