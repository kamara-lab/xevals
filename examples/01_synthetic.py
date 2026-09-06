"""The full seven-dimension evaluation, on the built-in world, with no extras.

Produces every artefact the library makes -- ``run.json``, four table formats,
five figures, videos with a HUD, and a self-contained HTML report -- in under a
minute on a laptop. It is the example to read first, and the one the docs quote.

The interesting part is not the score. It is that the *same* policy scores 1.00
on accuracy and 0.44 on robustness: it does the task perfectly and stops doing it
the moment something is in the way. A single success rate cannot say that.

    python examples/01_synthetic.py
"""

from __future__ import annotations

from _common import setup
from policies import ImagePolicy

import xevals


def main() -> int:
    result = xevals.evaluate(
        ImagePolicy(), "synthetic/reach", suite="full", **setup("01_synthetic")
    )

    print()
    print(result.table(kind="dimensions"))
    print()
    print("Where it breaks, worst cells first:")
    _, rows = result.cell_rows()
    ranked = sorted((r for r in rows if r[4] is not None), key=lambda r: r[4])
    for cell, perturbation, severity, _split, success, n in ranked[:6]:
        print(f"  {cell:28s} {success:.2f}  (n={n}, {perturbation} @ {severity:g})")
    print()
    print(f"Report: {result.directory / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
