# Library review and enhancement proposals

This review accompanies the package reorganisation. Behavioural fixes below are
proposals, not changes included in the move.

## Findings, in priority order

1. **High: distinguish borrowed environments from owned environments.**
   `evaluation/runner.py:evaluate` closes its probe, while `_env_factory` returns
   the same object when the caller supplies a live environment. `run_cell` also
   closes that object after every episode. A `PointMass` subclass that rejects
   `reset()` after `close()` reproduces failed episodes before the policy has
   acted. Introduce explicit ownership: close factory-created environments in
   `finally`, and leave caller-owned instances open. Apply the same rule to
   benchmark probes and baselines. Verify with an environment that cannot be
   reset after closure, over multiple cells and episodes.

2. **High: restore the missing development tools.**
   `tests/test_brand.py:19` imports `tools.make_banner` and `tools.make_logos`, but
   no `tools` package is tracked. The full test suite fails during collection.
   `.github/workflows/ci.yml:117` also invokes `tools.make_example_output`.
   Restore the generators or replace those callers with maintained equivalents;
   do not silently skip the brand checks. Verify full test collection and the
   documentation artefact generation command from a fresh checkout.

3. **Medium: make baseline budget semantics explicit and enforce them.**
   `evaluation/runner.py:_run_baselines` receives no budget and calls `run_cell`
   without one. A smoke evaluation with `Budget(episodes=0)` still runs one
   random and one noop baseline episode when `episodes=1`. The benchmark path
   has the same omission. Either include baselines in the total cap or expose a
   separate baseline allowance and report both. Verify episode counts and
   elapsed-time behaviour for both `evaluate` and `benchmark`.

4. **Medium: publish run artefacts atomically.**
   `reporting/results.py:Result.save` creates an existing directory and writes
   `run.json` directly, followed by the other outputs. An interrupted write can
   leave invalid JSON, and a reader can observe a mixture of artefacts while a
   run is rewritten. Write to a temporary sibling directory or temporary files,
   then publish with a completion manifest. Define explicit overwrite behaviour.
   Verify interruption handling and repeated saves. This is a risk identified
   by inspection, not a reproduced data-loss incident.

## Organisation implemented

| Package | Modules |
| --- | --- |
| `core` | types, errors, registry, seeding, dimensions, config |
| `evaluation` | runner, bench, suites, metrics, perturbations, judges |
| `integrations` | adapters, datasets |
| `environments` | envs, robots, sim |
| `reporting` | results, report, charts, plots, media, tradeoffs |

`cli.py` remains the command entry point. The top-level API and old module
imports remain supported. Module aliases share class identities and registries;
small `robots.py` and `sim.py` bridges retain lazy simulator registration.
Implementation imports use the grouped paths. API reference pages keep their
URLs while documenting the actual implementation modules.

The groups express responsibilities, not an enforced dependency hierarchy.
There is still coupling between evaluation and reporting through result
construction and scoring. A next refactor could extract cell result records and
aggregation into separate evaluation modules, leaving presentation dependent on
those records. Split the large metrics and perturbations modules by measurement
or perturbation family, and isolate simulator backends, as follow-up changes
with focused behavioural tests.

## Further enhancements

- Validate direct Python entry-point arguments (episode counts, seeds, horizon,
  control frequency and budgets) consistently with configuration validation.
- Separate optional renderer failures from unexpected implementation failures,
  and retain structured output-generation diagnostics in saved runs.
- Replace the README's fixed test count and runtime promise with maintained CI
  information; simulator availability changes both coverage and runtime.

## Compatibility limits

Existing import paths and old pickle class references remain resolvable. New
objects identify their classes using the grouped implementation paths, so new
pickles are not promised to load in older releases. Code depending on physical
source locations or module `__name__` values must use the new layout.

## Validation of this change

- Ruff and the strict MkDocs build pass.
- All 25 import compatibility checks pass, including a smoke evaluation with
  optional dependency imports blocked.
- Excluding the existing brand collection error, 470 tests passed initially.
  Nine localhost/graphics tests blocked by the sandbox passed when rerun with
  the required access: 479 passing tests in total across those runs.
- Two documentation tests still fail because `tools.make_example_output` is
  missing; the full suite also cannot collect `tests/test_brand.py` for the
  same missing package. These require the development-tools proposal above.
- Both source distribution and wheel build successfully. Importing every
  packaged module and running a smoke evaluation directly from the built wheel
  pass.
