# Conventions

Short list, and everything in the library follows it.

## Arrays are NumPy at every boundary

Observations are `dict[str, Any]`; actions are `(A,)` float32 in `[-1, 1]`;
frames are `(H, W, 3)` uint8. Adapters convert torch and jax on the way in and
out, so the core never holds a framework tensor, which is what lets the
core depend on numpy alone.

Actions are normalised by the task's **declared bounds**, not by dataset
statistics, so a checkpoint means the same thing whatever corpus it was trained
on.

## Observations are dictionaries, and a model takes what it wants

The runner never needs to know the modality. A pixel VLA reads `image`, a
state-space policy reads `state`, a language-conditioned model reads
`instruction`, and all three are driven by the same loop with no branch.

A perturbation that removes a field, `sensor/dropout`, genuinely
removes it, rather than zeroing it. A model that treats a zero image as a dark
room and one that notices the field is gone are different models, and zeroing
would conflate them.

## Anything unmeasurable is `null` with a reason

Never zero, never an exception. "This model exposes no confidence, so its
calibration is unknown" is a finding; a zero would be a lie, flattering for some
metrics and damning for others. Every null carries its reason into `run.json`,
into the table as `--`, and into the report as a hollow marker on the radar.

The same rule applies to a dimension. A suite that ran no security cell scores
`None` for security, not `0`.

## Seeds

One root seed per run. Every episode seed is derived from
`(root_seed, seed, index)`, **not** from the cell, so every cell in
a suite runs the same episodes from the same starting states. That is what makes
paired metrics work: retention, injection compliance and paraphrase agreement all
compare a perturbed episode with *the clean episode that started from the same
state*, which resolves a six-point difference with twenty episodes instead of two
hundred.

A perturbation is a pure function of `(seed, severity, input)`. Bootstrap
intervals are seeded at a fixed value, so reopening a results file gives the same
error bars.

The one thing a seed cannot fix is wall-clock latency, and the efficiency
dimension is therefore the one number that moves between runs.

## Run directories

`<out>/<name>-<config hash>/`. The hash covers the suite, the model and
environment fingerprints, the seeds and the episode count, and deliberately
**not** the output path, so two runs that differ only in where they were written
are recognisably the same experiment.

JSON keeps full precision. LaTeX and Markdown round. Rounding belongs to display
and not to storage, and a sweep reading `run.json` should not inherit the four
decimal places a table chose. Non-finite floats become `null`, because JSON has
no NaN and a bare `NaN` produces a file strict parsers reject.

## Registries

Slash-namespaced names: `visual/blur`, `accuracy/success_rate`, `injection/scene_text`.
Entries build lazily, so listing what exists never triggers an optional import.
Registering a name twice is an error: silent replacement is how a typo in a
plugin shadows a built-in metric.

## Logging

`print(..., flush=True)`. No `logging`, no wandb, no tensorboard. The `on_step`
hook is there for anyone who wants Rerun or a progress bar, and it is
`None`-tolerant and never required.

## Prose

British spelling, in the docs and the docstrings. Comments say *why*, not what.
Negative results are stated plainly: the library exists because flattering
numbers are the default failure mode of the field it measures.
