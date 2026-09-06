"""Generate the example run the documentation shows, and prune it for the web.

The docs claim a run directory contains a self-contained report, videos and a
leaderboard. Screenshotting that would be a claim about a claim, so the pages
embed the real files: the same ``report.html`` a run writes, opened in an
``iframe``, with the same GIFs and the same tables.

The run is a Franka Emika Panda picking a block off a table, because that is what
the library is for. Three policies, chosen so the leaderboard has a top, a middle
and a floor rather than three of the same thing:

``scripted``  the environment's own demonstrator, reading the simulator state
``visual``    finds the block by colour and back-projects it onto the table
``random``    uniform actions, which is the floor drawn on the chart

The **visual** policy is the one the single-run report features, and deliberately.
A state-reading policy is invisible to every visual perturbation, so its report is
a wall of ``1.00`` and demonstrates nothing about how a failure is presented. The
visual policy actually breaks under brightness, occlusion and the patch, which is
what makes its report worth looking at.

The run takes a couple of minutes, most of it physics. What takes the *space* is
GIFs and figures, so this script prunes:

* video is kept for a few named cells only, at every other frame. A
  ``visual/gaussian_noise`` cell defeats GIF compression completely and produced
  a 1.5 MB file on its own, which is why the showcase suite uses brightness
  instead: it is also the more interesting failure, being a cliff rather than a
  decay;
* the ``.tex`` and ``.csv`` copies of every table are dropped, since the page
  inlines the Markdown;
* trajectories are never written.

A budget is enforced at the end, because a documentation asset that quietly
grows to ten megabytes is one nobody notices until the site is slow.

Run from the repository root::

    python -m tools.make_example_output
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

import xevals
from xevals import Dimension
from xevals.suites import Cell, SuiteSpec

#: Where the docs read it from.
TARGET = Path("docs/assets/example")

#: Total budget for everything committed. Generous enough for real artefacts,
#: tight enough that adding a noisy video is noticed.
BUDGET_KB = 2200

#: The report is a set of pages now, so the docs copy takes all of them plus the
#: stylesheet they share. Listing them explicitly rather than globbing keeps the
#: budget honest about what is being committed.
REPORT_PAGES = (
    "report.html", "conditions.html", "episodes.html", "tradeoffs.html",
    "metrics.html", "provenance.html", "report.css",
)

#: Which model's single-run report the docs embed. The one that actually fails
#: somewhere: a report with no failures shows none of the machinery for reporting
#: them, which is most of what a reader came to see.
FEATURED = "visual"

#: Cells to keep video for. **Their directory layout is preserved**, because the
#: report links to ``videos/<cell>/<episode>.gif`` and renaming them into a flat
#: set silently broke every link in the embedded copy. A curated artefact that no
#: longer matches the page it belongs to is worse than no artefact.
#:
#: Chosen to show the HUD doing its job on a clean episode, a perturbed one and an
#: attacked one, and to show *motion*: brightness would be the obvious perturbed
#: choice and is a nearly static clip, because a blinded colour-tracker returns
#: zero and sits still. Occlusion makes it wander, which is both the more
#: informative failure and the more watchable one.
KEEP_VIDEO = ("clean", "occlusion@0.75", "patch@1")

#: Every other frame, capped. Enough to read the motion, half the bytes.
FRAME_STRIDE = 2
MAX_FRAMES = 24


#: Bigger than the environment's default, because these frames end up as the
#: videos the documentation shows and a 5 cm block on a table needs the pixels.
IMAGE_SIZE = 112


def environment(task=None, split="in"):
    """The scene every model in the showcase is measured in."""
    return xevals.envs.create("mujoco/panda-pick", split=split, image_size=IMAGE_SIZE)


class Scripted:
    """The environment's own demonstrator, driven through the runner.

    It reads the simulator rather than the frame, which is exactly why it is the
    top of the leaderboard and the flat line in the robustness column: nothing
    done to the pixels can touch it. That is a true statement about this policy
    and not a flattering one.
    """

    def __init__(self) -> None:
        self.env = None

    def bind(self, env):
        self.env = env

    def act(self, obs, *, instruction=None):
        if self.env is None:
            return np.zeros(4, dtype=np.float32)
        return np.asarray(self.env.optimal_action, dtype=np.float32)

    def describe(self):
        return {"adapter": "example", "module": "Scripted", "reads": "state"}


def visual_policy():
    """The colour-servoing policy from ``examples/policies.py``.

    Imported rather than reimplemented: the examples already name it, and a
    second copy here would be a second thing to keep in step.
    """
    from examples.policies import ArmPolicy

    return ArmPolicy()


class Random:
    """Uniform actions. The floor, drawn on the leaderboard rather than implied."""

    def __init__(self) -> None:
        self.rng = np.random.default_rng(1)

    def reset(self) -> None:
        self.rng = np.random.default_rng(1)

    def act(self, obs, *, instruction=None):
        return self.rng.uniform(-1.0, 1.0, 4).astype(np.float32)

    def describe(self):
        return {"adapter": "example", "module": "Random"}


def suite() -> SuiteSpec:
    """Seven cells: a clean one, a ladder, an attack and an unseen object.

    Small enough to run in seconds and wide enough that the report has a severity
    curve, a security row and a generalisation split to show. It claims all seven
    dimensions, so the radar and the trade-off section have something to say.
    """
    return SuiteSpec(
        name="showcase",
        cells=(
            Cell("clean"),
            Cell("brightness@0.4", "visual/brightness", 0.4),
            Cell("brightness@0.8", "visual/brightness", 0.8),
            Cell("occlusion@0.75", "visual/occlusion", 0.75),
            Cell("action-noise@0.6", "action/noise", 0.6),
            Cell("patch@1", "adversarial/patch", 1.0),
            Cell("unseen-object", "none", 0.0, "ood/object"),
        ),
        dimensions=tuple(Dimension),
        baselines=("random", "noop", "replay"),
        episodes=4,
        description="a Panda picking a block: clean, a ladder, an attack, an unseen object",
    )


def shrink_gif(source: Path, target: Path) -> None:
    """Re-encode a GIF at every ``FRAME_STRIDE``-th frame, capped."""
    from PIL import Image, ImageSequence

    with Image.open(source) as image:
        frames = [f.copy() for f in ImageSequence.Iterator(image)]
        duration = image.info.get("duration", 100)
    kept = frames[::FRAME_STRIDE][:MAX_FRAMES]
    target.parent.mkdir(parents=True, exist_ok=True)
    kept[0].save(
        target,
        save_all=True,
        append_images=kept[1:],
        duration=duration * FRAME_STRIDE,
        loop=0,
        optimize=True,
    )


def curate(run: Path, bench: Path) -> None:
    """Copy the pieces the docs embed, and nothing else.

    Whatever is dropped must not leave a dangling reference behind it, so
    :func:`prune_gallery` rewrites the report's gallery to the clips that
    survived rather than leaving links to the ones that did not.
    """
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)

    # The pages, verbatim except for the gallery pruning below. They are the
    # artefact being demonstrated, so editing them further would defeat the point.
    for name in REPORT_PAGES:
        source = run / name
        if not source.exists():
            continue
        text = source.read_text()
        if name.endswith(".html"):
            text = prune_gallery(text, KEEP_VIDEO)
        (TARGET / name).write_text(text)

    # The benchmark's own pages go in a subdirectory, so its `report.css` and its
    # `conditions.html` cannot collide with the single run's.
    (TARGET / "benchmark").mkdir(exist_ok=True)
    for source in bench.glob("*.html"):
        shutil.copy(source, TARGET / "benchmark" / source.name)
    shutil.copy(bench / "report.css", TARGET / "benchmark" / "report.css")
    # One matplotlib radar is kept, because the docs page shows it as an example
    # of the publication figures a run also writes. The report's own charts are
    # inline SVG and reference no PNG at all, so a second copy beside the
    # benchmark pages would be a file nothing points at.
    shutil.copy(bench / "radar.png", TARGET / "radar.png")

    # Same names, same nesting, so the report's own links resolve.
    for cell in KEEP_VIDEO:
        for source in sorted((run / "videos" / cell).glob("*.gif")):
            shrink_gif(source, TARGET / "videos" / cell / source.name)

    # Markdown only: the page inlines these, and the other three formats would be
    # three more copies of numbers nobody reads from the docs.
    for source, name in (
        (bench / "leaderboard.md", "leaderboard.md"),
        (bench / "cells.md", "benchmark-cells.md"),
        (run / "table.md", "dimensions.md"),
        (run / "cells.md", "cells.md"),
    ):
        if source.exists():
            shutil.copy(source, TARGET / name)


def prune_gallery(html: str, keep: tuple[str, ...]) -> str:
    """Drop gallery entries whose clip was not copied.

    The report is written for a complete run directory. This copy is a subset, so
    the entries pointing at clips that are not here have to go: a gallery of dead
    links is exactly the "I cannot see the videos" that prompted this function.
    """
    import re

    def drop(match: re.Match[str]) -> str:
        entry = match.group(0)
        return entry if any(f"videos/{cell}/" in entry for cell in keep) else ""

    html = re.sub(r"<figure class=\"clip\">.*?</figure>", drop, html, flags=re.S)
    # A condition block whose clips all went is left with an empty gallery, which
    # renders as a stray gap. The per-episode marks below it still carry the cell.
    return html.replace('<div class="gallery"></div>', "")


def report_size() -> int:
    """Total kilobytes committed, and a per-file breakdown on the way."""
    total = 0
    for path in sorted(TARGET.rglob("*")):
        if path.is_file():
            kb = path.stat().st_size // 1024
            total += kb
            print(f"  {str(path.relative_to(TARGET)):28s} {kb:5d} kB")
    return total


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        bench = xevals.benchmark(
            {"scripted": Scripted(), "visual": visual_policy(), "random": Random()},
            environment,
            suite=suite(),
            episodes=4,
            seeds=(0, 1),
            record=1,
            out=tmp,
            name="showcase",
            verbose=False,
        )
        directory = bench.directory
        assert directory is not None
        featured = next(
            d for d in directory.iterdir() if d.is_dir() and d.name.startswith(FEATURED)
        )
        curate(featured, directory)
        print(bench.table())
        print()

    total = report_size()
    print(f"  {'total':28s} {total:5d} kB  (budget {BUDGET_KB} kB)")
    if total > BUDGET_KB:
        print("! over budget; prune a video or drop a figure")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
