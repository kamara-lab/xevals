"""The three rules every perturbation obeys, checked on every registered one."""

from __future__ import annotations

import numpy as np
import pytest

from xevals import perturbations
from xevals.dimensions import Dimension
from xevals.perturbations import SEVERITIES, Compose

FRAME = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
OBS = {"image": FRAME, "state": np.zeros(4, dtype=np.float32), "instruction": "push the red cube"}

#: Every perturbation except the ones needing an environment to attach to.
NAMES = [n for n in perturbations.available() if not n.startswith("dynamics/")]


@pytest.mark.parametrize("name", NAMES)
def test_a_perturbation_is_deterministic_in_seed_and_severity(name):
    first = perturbations.create(name, severity=0.6)
    second = perturbations.create(name, severity=0.6)
    for attribute in ("apply_obs", "apply_action", "apply_instruction"):
        left, right = getattr(first, attribute, None), getattr(second, attribute, None)
        if left is None:
            continue
        payload = {
            "apply_obs": OBS,
            "apply_action": np.ones(2, dtype=np.float32),
            "apply_instruction": "push the red cube",
        }[attribute]
        a, b = left(payload, seed=3), right(payload, seed=3)
        if isinstance(a, dict):
            assert sorted(a) == sorted(b)
            assert all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a
                       if not isinstance(a[k], str))
        elif isinstance(a, str):
            assert a == b
        else:
            assert np.array_equal(a, b)


@pytest.mark.parametrize("name", NAMES)
def test_a_perturbation_preserves_shape_and_dtype(name):
    perturbation = perturbations.create(name, severity=0.8)
    apply_obs = getattr(perturbation, "apply_obs", None)
    if apply_obs is None:
        return
    out = apply_obs(OBS, seed=1)
    if "image" not in out:  # sensor/dropout removes fields on purpose
        assert name == "sensor/dropout"
        return
    image = np.asarray(out["image"])
    assert image.shape == FRAME.shape
    assert image.dtype == FRAME.dtype


@pytest.mark.parametrize("name", NAMES)
def test_a_perturbation_declares_its_dimension_and_ladder(name):
    described = perturbations.describe(name)
    assert described["dimension"] in {d.value for d in Dimension}
    if name != "none":
        assert described["severities"] == list(SEVERITIES)


def test_severity_zero_is_the_clean_condition():
    identity = perturbations.identity()
    assert identity.severity == 0.0
    assert identity.apply_obs(OBS) is OBS


def test_severity_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError, match=r"severity must be in \[0, 1\]"):
        perturbations.create("visual/blur", severity=1.5)


def test_stronger_visual_noise_moves_the_frame_further():
    frame = FRAME.astype(np.float64)
    distances = [
        float(
            np.mean(
                np.abs(
                    perturbations.create("visual/gaussian_noise", severity=s)
                    .apply_obs(OBS, seed=0)["image"]
                    .astype(np.float64)
                    - frame
                )
            )
        )
        for s in SEVERITIES
    ]
    assert distances == sorted(distances)


def test_a_paraphrase_keeps_the_object_the_model_must_still_ground():
    out = perturbations.create("instruction/paraphrase", severity=1.0).apply_instruction(
        "push the red cube", seed=0
    )
    assert out != "push the red cube"
    assert "red cube" in out


def test_an_injection_is_filed_under_security_not_robustness():
    for name in perturbations.available("injection") + perturbations.available("adversarial"):
        assert perturbations.describe(name)["dimension"] == "security"


def test_composition_takes_the_strongest_severity_and_the_stricter_dimension():
    composed = Compose(
        [
            perturbations.create("visual/blur", severity=0.4),
            perturbations.create("injection/instruction", severity=0.8),
        ]
    )
    assert composed.severity == 0.8
    assert composed.dimension is Dimension.SECURITY
    assert "ignore" in composed.apply_instruction("push the red cube", seed=0)


def test_a_camera_shift_holds_the_edge_rather_than_wrapping():
    # A ramp, so a wrap is visible as a discontinuity that a hold does not have.
    ramp = np.tile(np.arange(32, dtype=np.uint8).reshape(1, 32, 1), (32, 1, 3))
    shifted = perturbations.create("visual/camera_shift", severity=1.0).apply_obs(
        {"image": ramp}, seed=2
    )["image"]

    assert shifted.shape == ramp.shape and shifted.dtype == ramp.dtype
    row = shifted[shifted.shape[0] // 2, :, 0].astype(int)
    # np.roll would put 31 next to 0 somewhere in the row. Holding the edge
    # cannot: the row stays monotonically non-decreasing.
    assert all(b - a >= 0 for a, b in zip(row, row[1:], strict=False)), row.tolist()


def test_a_camera_shift_moves_the_frame_at_all():
    ramp = np.tile(np.arange(32, dtype=np.uint8).reshape(1, 32, 1), (32, 1, 3))
    moved = [
        not np.array_equal(
            perturbations.create("visual/camera_shift", severity=1.0).apply_obs(
                {"image": ramp}, seed=seed
            )["image"],
            ramp,
        )
        for seed in range(8)
    ]
    assert sum(moved) >= 6, "an edge-holding shift must still be a shift"
