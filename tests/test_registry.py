"""The one registry type every namespace uses."""

from __future__ import annotations

import pytest

from xevals.registry import Registry


def test_an_entry_can_be_described_without_being_built():
    built = []
    registry: Registry[str] = Registry("things")
    registry.register("a/one", lambda: built.append(1) or "one", summary="the first")

    assert registry.describe("a/one")["summary"] == "the first"
    assert built == [], "describe must not trigger the factory, or its imports"
    assert registry.create("a/one") == "one"


def test_registering_a_name_twice_is_an_error():
    registry: Registry[int] = Registry("things")
    registry.register("x", lambda: 1)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("x", lambda: 2)


def test_an_unknown_name_suggests_the_near_miss():
    registry: Registry[int] = Registry("metrics")
    registry.register("accuracy/success_rate", lambda: 1)
    with pytest.raises(KeyError, match="success_rate"):
        registry.entry("accuracy/sucess_rate")


def test_families_are_the_slash_prefixes():
    registry: Registry[int] = Registry("things")
    registry.register("visual/blur", lambda: 1)
    registry.register("visual/noise", lambda: 2)
    registry.register("action/noise", lambda: 3)

    assert registry.families() == ["action", "visual"]
    assert registry.available("visual") == ["visual/blur", "visual/noise"]
