"""Dimensions, normalisation, and the difference between zero and unmeasured."""

from __future__ import annotations

import pytest

from xevals.dimensions import (
    DIMENSION_ORDER,
    NORMALISERS,
    Dimension,
    Normaliser,
    color,
    normalise,
    score_dimension,
)


def test_the_order_is_fixed_because_it_fixes_the_colours():
    assert [d.value for d in DIMENSION_ORDER] == [
        "accuracy",
        "robustness",
        "safety",
        "security",
        "efficiency",
        "generalization",
        "consistency",
    ]
    assert len({color(d) for d in DIMENSION_ORDER}) == len(DIMENSION_ORDER)


def test_every_dimension_has_a_question():
    assert all(d.question.endswith("?") for d in DIMENSION_ORDER)


def test_a_unit_metric_is_used_as_is_and_an_inverse_one_is_flipped():
    assert normalise("accuracy/success_rate", 0.8) == pytest.approx(0.8)
    assert normalise("safety/violation_rate", 0.25) == pytest.approx(0.75)


def test_a_scale_metric_scores_1_over_e_at_its_reference():
    reference = NORMALISERS["efficiency/latency_p50"].reference
    assert normalise("efficiency/latency_p50", reference, higher_is_better=False) == pytest.approx(
        0.3679, abs=1e-3
    )


def test_a_scale_normaliser_needs_a_reference():
    with pytest.raises(ValueError, match="reference"):
        Normaliser("scale")


def test_an_unmeasured_metric_is_skipped_rather_than_counted_as_zero():
    score = score_dimension(
        Dimension.CONSISTENCY,
        {"consistency/determinism": 1.0, "consistency/ece": None},
    )
    # Averaging a zero in would punish the model for the library's ignorance.
    assert score.score == pytest.approx(1.0)
    assert "consistency/ece" in score.skipped


def test_a_dimension_with_nothing_measurable_scores_none_not_zero():
    score = score_dimension(Dimension.SECURITY, {"security/jailbreak_rate": None})
    assert score.score is None
    assert not score.measured


def test_every_normaliser_is_documented_or_self_evident():
    for name, normaliser in NORMALISERS.items():
        if normaliser.kind == "scale":
            assert normaliser.why, f"{name} has a magic reference with no stated reason"


def test_every_metric_has_a_normaliser_and_every_normaliser_has_a_metric():
    from xevals.metrics import METRICS

    # A metric without one silently falls back to the unit interval, which is
    # wrong for anything measured in physical units; a normaliser without one is
    # a rule nothing obeys.
    assert [n for n in METRICS.available() if n not in NORMALISERS] == []
    assert [n for n in NORMALISERS if n not in METRICS] == []


def test_a_negative_value_on_a_scale_metric_scores_the_bad_direction():
    # A clearance of -0.07 is a workspace violation, not a large clearance.
    assert normalise("safety/min_clearance", -0.07) == pytest.approx(0.0)
    assert normalise("safety/min_clearance", 0.10) > 0.9
