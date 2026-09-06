# Concepts

Five ideas, and the rest of the library follows from them.

<div class="grid cards" markdown>

-   **[Dimensions](dimensions/index.md)**

    The seven questions, the metrics that answer each, and the normalisation
    that turns raw metrics into a score: written into `run.json` as data,
    so an aggregate can be recomputed or disagreed with.

-   **[Perturbations](perturbations.md)**

    The controlled changes a robustness or security claim rests on: deterministic,
    structure-preserving, and on a shared severity ladder.

-   **[Trade-offs](tradeoffs.md)**

    Where one dimension is bought with another, tested rather than assumed, and
    reported as undetermined when the run cannot say.

-   **[Baselines and gates](baselines-and-gates.md)**

    `random` and `noop` bound the task from below; `replay` decides whether any
    number from the environment is interpretable at all.

-   **[Results schema](results-schema.md)**

    What a run directory contains, and why it contains enough to be re-run.

-   **[Randomness](randomness.md)**

    What is reproducible, what is not, and which of those is measured rather than
    assumed away.

</div>
