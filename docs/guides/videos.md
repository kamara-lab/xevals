# Videos

A video is the fastest way to find out that a "robustness failure" is really a
rendering bug, that a policy is oscillating rather than stuck, or that the
occlusion happened to land on the goal.

!!! tip "See real ones"
    [What a run looks like](example-output.md) shows three clips from one run:
    clean, occluded and under an adversarial patch.

```python
result = xevals.evaluate(model, env, suite="full", record=2, out="runs/x")
# runs/x/<name>-<hash>/videos/<cell>/000.mp4
```

`record` is a **count**, not a boolean: keeping frames for every episode of a
43-cell suite is how a run directory reaches ten gigabytes, and the first two per
cell are what a report shows.

## The HUD

Each frame carries a strip with the step, the perturbation and severity, the
instruction, and the outcome on the final frame. The instruction shown is the
**perturbed** one: watching a paraphrase cell with the original instruction
printed over it hides the very thing being varied.

The strip is drawn with a 3&times;5 bitmap font built into the library, so it needs
no font file and no Pillow: a video is exactly the artefact someone wants
when their install is minimal and something is going wrong.

## Degradation

mp4 through `imageio-ffmpeg` when `xevals[video]` is installed, GIF through Pillow
when it is not, and nothing (with a printed note) when neither is. A completed
evaluation is never lost to a missing codec.

`save_video` returns **the path it actually wrote**, and the report links to that
return value, so a GIF fallback never produces a broken link.

## Small frames

The synthetic world renders 64&times;64. `xevals` upscales with nearest-neighbour to
at least 256 and forces even dimensions, because most encoders reject odd ones.
Nearest-neighbour rather than smooth interpolation: this is a record of what the
model saw, and interpolation would invent detail it did not.

## Contact sheets

```python
from xevals.media import tile
sheet = tile(trajectory.frames, columns=6)
```

Sometimes better than the video: a still grid reads at a glance and survives being
pasted into an issue.

::: xevals.media
    options:
      heading_level: 2
      members: false
