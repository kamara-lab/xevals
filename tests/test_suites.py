"""Suites: coverage, pairing, and the metric list a run actually used."""

from __future__ import annotations

import pytest

from xevals import suites
from xevals.dimensions import Dimension
from xevals.suites import Cell, SuiteSpec


def test_every_built_in_suite_has_a_clean_cell():
    # Every paired metric divides by it; a suite without one reports ratios
    # against nothing.
    for name in suites.available():
        assert suites.create(name).clean_cell.name


def test_a_suite_without_a_clean_cell_is_rejected():
    with pytest.raises(ValueError, match="no clean cell"):
        SuiteSpec("bad", (Cell("blur", "visual/blur", 1.0),), (Dimension.ROBUSTNESS,))


def test_duplicate_cells_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        SuiteSpec("bad", (Cell("clean"), Cell("clean")), (Dimension.ACCURACY,))


def test_resolving_records_the_metric_list_that_was_actually_used():
    resolved = suites.create("robustness").resolved()
    assert all(cell.metrics for cell in resolved.cells)
    # A wildcard would mean something different next release, and the saved run
    # would not say which it meant.
    assert "accuracy/success_rate" in resolved.clean_cell.metrics


def test_resolving_filters_metrics_to_the_cells_they_mean_something_in():
    resolved = suites.create("full").resolved()
    blur = next(c for c in resolved.cells if c.perturbation == "visual/camera_shift")
    injection = next(c for c in resolved.cells if c.perturbation.startswith("injection/"))

    assert "consistency/paraphrase_agreement" not in blur.metrics
    assert "security/injection_compliance" not in blur.metrics
    assert "security/injection_compliance" in injection.metrics
    assert "accuracy/success_rate" in blur.metrics


def test_every_perturbed_cell_is_paired_with_the_clean_one():
    for cell in suites.create("full").resolved().cells:
        assert cell.pair_with == ("clean" if not cell.is_clean else None)


def test_a_suite_claims_only_the_dimensions_it_runs_cells_for():
    security = suites.create("security")
    assert Dimension.SECURITY in security.dimensions
    assert any(c.perturbation.startswith(("adversarial/", "injection/")) for c in security.cells)

    core = suites.create("core")
    assert Dimension.SECURITY not in core.dimensions


def test_the_full_suite_is_assembled_from_the_specialised_ones():
    full = {c.name for c in suites.create("full").cells}
    for name in ("robustness", "safety", "security", "generalization", "consistency"):
        assert {c.name for c in suites.create(name).cells} <= full


def test_a_suite_can_be_built_from_a_mapping():
    spec = suites.from_dict(
        {
            "name": "custom",
            "dimensions": ["accuracy"],
            "cells": [{"name": "clean"}, {"name": "blur", "perturbation": "visual/blur",
                                          "severity": 0.5}],
        }
    )
    assert len(spec.cells) == 2
    assert spec.dimensions == (Dimension.ACCURACY,)


def test_a_typo_in_a_suite_file_is_an_error_not_a_skipped_dimension():
    with pytest.raises(ValueError, match="no field"):
        suites.from_dict({"name": "custom", "cell": []})
    with pytest.raises(ValueError, match="unknown metric"):
        suites.from_dict(
            {"name": "c", "cells": [{"name": "clean", "metrics": ["accuracy/sucess_rate"]}]}
        )


def test_the_smoke_suite_is_small_enough_to_be_a_wiring_check():
    smoke = suites.create("smoke")
    assert len(smoke.cells) == 2
    assert smoke.episodes <= 3
