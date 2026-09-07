"""Videos and GIFs of what the model actually saw, with the conditions on screen.

A video is the fastest way to find out that a "robustness failure" is really a
rendering bug, that a policy is oscillating rather than stuck, or that the
occlusion landed on the goal. So xevals records a few episodes per cell and
writes them with a HUD strip carrying the step, the instruction *as the model
received it*, the perturbation and severity, and the outcome.

The instruction shown is the perturbed one on purpose. Watching a paraphrase
cell with the original instruction printed over it hides the very thing being
varied.

Degradation is deliberate and quiet: mp4 through ``imageio-ffmpeg`` when it is
installed, GIF through Pillow when it is not, and nothing -- with a printed note
-- when neither is. A completed evaluation must never be lost to a missing codec.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from xevals.core.errors import MissingExtra

__all__ = ["overlay", "save_gif", "save_video", "tile", "write_all"]

#: kamara paper and ink, so a HUD strip matches the figures beside it.
_PAPER = (250, 250, 248)
_INK = (18, 20, 18)
_ACCENT = (54, 127, 201)


def _imageio():
    try:
        import imageio.v3 as iio
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise MissingExtra("imageio", "video", "to write video") from exc
    return iio


def save_video(frames: Sequence[np.ndarray], path: str | Path, *, fps: int = 10) -> Path:
    """Write frames as mp4, falling back to GIF when no encoder is available.

    Returns the path actually written, which may have a different suffix from
    the one asked for -- the caller links to the return value rather than to
    what it requested, so a GIF fallback does not produce a broken link in the
    report.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = _stack(frames)
    try:
        iio = _imageio()
        iio.imwrite(path.with_suffix(".mp4"), stack, fps=fps, codec="libx264")
        return path.with_suffix(".mp4")
    except Exception:  # noqa: BLE001 - any encoder failure falls back to GIF
        return save_gif(stack, path.with_suffix(".gif"), fps=fps)


def save_gif(frames: Sequence[np.ndarray], path: str | Path, *, fps: int = 10) -> Path:
    """Write frames as an animated GIF through Pillow."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise MissingExtra("pillow", "video", "to write a GIF") from exc
    path = Path(path).with_suffix(".gif")
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / max(1, fps)),
        loop=0,
        optimize=True,
    )
    return path


def _stack(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Frames as one ``(T, H, W, 3)`` uint8 array, upscaled if they are tiny.

    A 64-pixel frame is unwatchable and most video encoders reject odd
    dimensions, so small frames are nearest-neighbour upscaled to at least 256
    and forced even. Nearest-neighbour rather than smooth: this is a record of
    what the model saw, and interpolation would invent detail it did not.
    """
    if not frames:
        raise ValueError("no frames to write")
    stack = np.asarray(frames, dtype=np.uint8)
    if stack.ndim == 3:
        stack = np.repeat(stack[..., None], 3, axis=-1)
    height, width = stack.shape[1:3]
    factor = max(1, int(np.ceil(256 / min(height, width))))
    if factor > 1:
        stack = np.repeat(np.repeat(stack, factor, axis=1), factor, axis=2)
    height, width = stack.shape[1:3]
    if height % 2 or width % 2:
        stack = stack[:, : height - height % 2, : width - width % 2]
    return stack


def overlay(
    frames: Sequence[np.ndarray],
    *,
    instruction: str | None = None,
    perturbation: str | None = None,
    severity: float | None = None,
    success: bool | None = None,
    strip: int = 26,
) -> list[np.ndarray]:
    """Add a HUD strip: step, instruction, condition, outcome.

    Drawn with the same 3x5 bitmap font the synthetic environment uses, so the
    HUD needs no font file and no Pillow -- a video is exactly the artefact
    someone wants when their install is minimal and something is going wrong.
    """
    from xevals.environments.envs import _draw_text

    stack = _stack(frames)
    height, width = stack.shape[1:3]
    out = []
    condition = perturbation or "clean"
    if severity is not None and perturbation and perturbation != "none":
        condition = f"{condition} {severity:g}"
    for index, frame in enumerate(stack):
        canvas = np.full((height + strip, width, 3), _PAPER, dtype=np.uint8)
        canvas[strip:] = frame
        canvas[strip - 1] = _ACCENT
        _draw_text(canvas, f"{index:03d} {condition}"[: width // 4], origin=(3, 3), rgb=_INK)
        second = (instruction or "")[: max(0, width // 4)]
        if success is not None and index == len(stack) - 1:
            second = ("SUCCESS " + second) if success else ("FAILED " + second)
        _draw_text(canvas, second, origin=(3, 12), rgb=_INK)
        out.append(canvas)
    return out


def tile(frames: Sequence[np.ndarray], *, columns: int = 6) -> np.ndarray:
    """A contact sheet of an episode, for a figure or a README.

    Sometimes better than the video: a still grid can be read at a glance and
    survives being pasted into an issue.
    """
    stack = _stack(frames)
    step = max(1, len(stack) // (columns * 2))
    chosen = stack[::step][: columns * 2]
    rows = int(np.ceil(len(chosen) / columns))
    height, width = chosen.shape[1:3]
    sheet = np.full((rows * height, columns * width, 3), _PAPER, dtype=np.uint8)
    for index, frame in enumerate(chosen):
        r, c = divmod(index, columns)
        sheet[r * height : (r + 1) * height, c * width : (c + 1) * width] = frame
    return sheet


def write_all(result: Any, directory: str | Path, *, per_cell: int | None = 2, fps: int = 10):
    """Write episodes per cell with a HUD; ``per_cell=None`` saves all recorded ones.

    Returns cell name -> paths written. Cells whose episodes carry no frames --
    because ``record`` was 0, or because the environment renders nothing -- are
    absent rather than empty.
    """
    directory = Path(directory)
    written: dict[str, list[Path]] = {}
    for cell_name, episodes in result.trajectories.items():
        cell = next((c for c in result.suite.cells if c.name == cell_name), None)
        paths = []
        for index, traj in enumerate(episodes[:per_cell]):
            if not traj.frames:
                continue
            frames = overlay(
                traj.frames,
                instruction=traj.instruction,
                perturbation=traj.perturbation,
                severity=cell.severity if cell else None,
                success=traj.success,
            )
            paths.append(save_video(frames, directory / cell_name / f"{index:03d}", fps=fps))
        if paths:
            written[cell_name] = paths
    return written
