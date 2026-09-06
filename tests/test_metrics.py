"""Metrics, against hand-built trajectories with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from xevals import metrics
from xevals.metrics import bootstrap_ci
from xevals.types import Trajectory, Violation


def test_success_rate_is_the_fraction_of_successful_episodes(episodes):
    value = metrics.success_rate(episodes(lambda s: s % 2 == 0))
    assert value.value == pytest.approx(0.5)
    assert value.n == 10


def test_a_metric_with_no_verdict_reports_null_with_a_reason():
    value = metrics.success_rate([Trajectory(success=None)])
    assert value.value is None
    assert "success verdict" in value.reason


def test_a_confidence_interval_narrows_as_episodes_are_added():
    rng = np.random.default_rng(0)
    widths = []
    for n in (10, 100, 1000):
        values = rng.binomial(1, 0.5, n).astype(float).tolist()
        lo, hi = bootstrap_ci(values)
        widths.append(hi - lo)
    assert widths == sorted(widths, reverse=True)


def test_an_interval_needs_at_least_two_values():
    assert bootstrap_ci([1.0]) is None


def test_the_same_data_gives_the_same_interval_every_time():
    # A results file whose error bars move when you reopen it is not a results file.
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    assert bootstrap_ci(values) == bootstrap_ci(values)


def test_retention_is_paired_on_seeds(episodes):
    clean = episodes(lambda s: True)
    dirty = episodes(lambda s: s < 6)
    assert metrics.retention(dirty, ref=clean).value == pytest.approx(0.6)


def test_retention_is_undefined_rather_than_perfect_when_nothing_succeeds(episodes):
    # Dividing zero by zero to get 1.0 is how a broken policy tops a robustness table.
    value = metrics.retention(episodes(lambda s: False), ref=episodes(lambda s: False))
    assert value.value is None
    assert "undefined" in value.reason


def test_retention_is_capped_at_one_when_a_perturbation_helps(episodes):
    value = metrics.retention(episodes(lambda s: True), ref=episodes(lambda s: s < 5))
    assert value.value == pytest.approx(1.0)


def test_retention_needs_shared_seeds():
    left = [Trajectory(seed=s, success=True) for s in (0, 1)]
    right = [Trajectory(seed=s, success=True) for s in (8, 9)]
    assert "share no seeds" in metrics.retention(left, ref=right).reason


def test_area_under_the_severity_curve_is_normalised_to_the_ladder():
    flat = metrics.severity_auc([], curve={0.0: 1.0, 0.5: 1.0, 1.0: 1.0})
    decaying = metrics.severity_auc([], curve={0.0: 1.0, 0.5: 0.5, 1.0: 0.0})
    assert flat.value == pytest.approx(1.0)
    assert decaying.value == pytest.approx(0.5)


def test_violations_are_counted_per_episode_not_per_step():
    # One excursion lasting forty steps is one excursion.
    long = Trajectory(
        actions=[np.zeros(2)] * 40,
        violations=[Violation("workspace", 0.1, step) for step in range(40)],
        extra={"limits_declared": True},
    )
    assert metrics.violation_rate([long]).value == pytest.approx(1.0)
    assert metrics.violation_steps([long]).value == pytest.approx(1.0)


def test_safety_is_unmeasured_when_the_environment_declares_no_limits():
    value = metrics.violation_rate([Trajectory(success=True)])
    assert value.value is None
    assert "declares no safety limits" in value.reason


def test_refusal_is_only_scored_on_instructions_that_were_unsafe():
    mixed = [
        Trajectory(extra={"unsafe_instruction": True, "refused": True}),
        Trajectory(extra={"unsafe_instruction": False, "refused": False}),
    ]
    # A model that refuses everything is safe the way an unplugged robot is.
    assert metrics.refusal_rate(mixed).value == pytest.approx(1.0)
    assert metrics.refusal_rate(mixed).n == 1


def test_latency_percentiles_pool_steps_rather_than_averaging_episodes():
    trajs = [
        Trajectory(latencies_ms=[1.0] * 90, success=True),
        Trajectory(latencies_ms=[100.0] * 10, success=True),
    ]
    # Averaging the two episode means would give 50.5; the pooled median is 1.0,
    # and the tail -- which is what a control loop must budget for -- is visible
    # only because the steps were pooled.
    assert metrics.latency_p50(trajs).value == pytest.approx(1.0)
    assert metrics.latency_p95(trajs).value == pytest.approx(100.0)


def test_control_headroom_goes_negative_when_the_model_cannot_keep_up():
    slow = [Trajectory(latencies_ms=[500.0] * 10)]
    assert metrics.control_headroom(slow, control_hz=10.0).value < 0


def test_the_generalisation_gap_needs_both_sides():
    inside = [Trajectory(seed=s, success=True, split="in") for s in range(4)]
    assert metrics.generalization_gap(inside).value is None
    outside = [Trajectory(seed=s, success=False, split="ood/object") for s in range(4)]
    assert metrics.generalization_gap(inside + outside).value == pytest.approx(1.0)


def test_seed_spread_needs_at_least_three_seeds():
    two = [Trajectory(seed=s, success=True) for s in (0, 1)]
    assert "at least three" in metrics.seed_std(two).reason


def test_determinism_compares_repeats_of_one_seed():
    same = [Trajectory(seed=0, actions=[np.ones(2, dtype=np.float32)]) for _ in range(2)]
    different = [
        Trajectory(seed=1, actions=[np.ones(2, dtype=np.float32)]),
        Trajectory(seed=1, actions=[np.zeros(2, dtype=np.float32)]),
    ]
    assert metrics.determinism(same).value == pytest.approx(1.0)
    assert metrics.determinism(same + different).value == pytest.approx(0.5)


def test_calibration_is_unmeasurable_without_confidence(episodes):
    value = metrics.expected_calibration_error(episodes(lambda s: True))
    assert value.value is None
    assert "confidence" in value.reason


def test_a_perfectly_calibrated_model_has_no_calibration_error():
    trajs = [Trajectory(seed=s, confidences=[1.0], success=True) for s in range(5)]
    trajs += [Trajectory(seed=s, confidences=[0.0], success=False) for s in range(5, 10)]
    assert metrics.expected_calibration_error(trajs).value == pytest.approx(0.0)


def test_a_metric_that_raises_becomes_a_null_rather_than_ending_the_run():
    metrics.METRICS.register(
        "accuracy/explodes",
        lambda **kw: (lambda *a, **k: 1 / 0),
        dimension="accuracy",
        higher_is_better=True,
    )
    value = metrics.compute(["accuracy/explodes"], [])["accuracy/explodes"]
    assert value.value is None
    assert "ZeroDivisionError" in value.reason


def test_applicability_keeps_a_metric_out_of_cells_it_would_mislead_in():
    assert metrics.applies_to("consistency/paraphrase_agreement", "instruction/paraphrase", "in")
    assert not metrics.applies_to("consistency/paraphrase_agreement", "visual/occlusion", "in")
    assert not metrics.applies_to("security/injection_compliance", "visual/blur", "in")
    assert metrics.applies_to("accuracy/success_rate", "visual/blur", "in")


def test_every_metric_declares_a_dimension_and_a_direction():
    for name in metrics.available():
        described = metrics.describe(name)
        assert "dimension" in described
        assert isinstance(described["higher_is_better"], bool)
