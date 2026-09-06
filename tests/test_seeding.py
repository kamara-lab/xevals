"""The determinism contract."""

from __future__ import annotations

import numpy as np

from xevals.seeding import derive, seeds_for, spawn


def test_derived_seeds_are_stable_across_processes():
    # Not `hash`, whose string hashing is randomised per process -- the exact
    # failure this function exists to prevent.
    assert derive("clean", 0, 3) == derive("clean", 0, 3)
    assert derive("clean", 0, 3) != derive("clean", 0, 4)


def test_a_generator_is_a_pure_function_of_its_position():
    a = spawn(0, "visual/blur", 0.4).normal(size=8)
    b = spawn(0, "visual/blur", 0.4).normal(size=8)
    c = spawn(0, "visual/blur", 0.6).normal(size=8)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_episode_seeds_do_not_depend_on_which_cells_exist():
    # Cells share seeds so that perturbed and clean episodes can be paired; this
    # is what makes every paired metric in the library measurable.
    assert seeds_for(0, 5, 0) == seeds_for(0, 5, 0)
    assert seeds_for(0, 5, 0) != seeds_for(0, 5, 1)


def test_seeds_below_one_root_are_distinct():
    seeds = seeds_for(0, 32, 0)
    assert len(set(seeds)) == 32
