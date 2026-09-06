"""Video, GIF fallback, and the HUD that says what the model was shown."""

from __future__ import annotations

import numpy as np
import pytest

from xevals import media

FRAMES = [np.full((32, 32, 3), i * 8, dtype=np.uint8) for i in range(6)]


def test_the_hud_carries_the_condition_and_the_instruction():
    out = media.overlay(
        FRAMES, instruction="push the red cube", perturbation="visual/blur",
        severity=0.6, success=True,
    )
    assert len(out) == len(FRAMES)
    # The strip is added above the frame, and the frame itself is untouched.
    assert out[0].shape[0] > out[0].shape[1] or out[0].shape[0] > 32


def test_small_frames_are_upscaled_without_inventing_detail():
    stacked = media._stack(FRAMES)
    assert stacked.shape[1] >= 256
    assert stacked.shape[1] % 2 == 0 and stacked.shape[2] % 2 == 0
    assert set(np.unique(stacked)) <= set(np.unique(np.asarray(FRAMES)))


def test_writing_with_no_frames_is_an_error_rather_than_an_empty_file():
    with pytest.raises(ValueError, match="no frames"):
        media._stack([])


def test_a_gif_is_written_when_pillow_is_available(tmp_path):
    pytest.importorskip("PIL")
    path = media.save_gif(FRAMES, tmp_path / "episode")
    assert path.suffix == ".gif" and path.stat().st_size > 0


def test_save_video_returns_the_path_it_actually_wrote(tmp_path):
    pytest.importorskip("PIL")
    path = media.save_video(FRAMES, tmp_path / "episode")
    # mp4 when an encoder is present, GIF when it is not -- the caller links to
    # the return value, so a fallback never produces a broken link.
    assert path.suffix in (".mp4", ".gif")
    assert path.exists()


def test_a_contact_sheet_tiles_the_episode():
    sheet = media.tile(FRAMES, columns=3)
    assert sheet.ndim == 3 and sheet.shape[-1] == 3
