"""Configs: extends, overrides, and errors that name the valid keys."""

from __future__ import annotations

import json

import pytest

from xevals import config

BASE = """
[target]
env = "synthetic/reach"

[run]
episodes = 20
seeds = [0, 1]
"""

CHILD = """
extends = "base.toml"

[model]
target = "mypkg:policy"

[run]
episodes = 5
"""


def test_a_child_config_updates_a_section_rather_than_replacing_it(tmp_path):
    (tmp_path / "base.toml").write_text(BASE)
    (tmp_path / "child.toml").write_text(CHILD)

    cfg = config.load(tmp_path / "child.toml")
    assert cfg.run.episodes == 5
    assert cfg.run.seeds == [0, 1], "the parent's seeds survive the child's episode count"
    assert cfg.target.env == "synthetic/reach"
    assert cfg.model.target == "mypkg:policy"


def test_an_unknown_key_names_the_valid_ones(tmp_path):
    (tmp_path / "bad.toml").write_text("[run]\nepisode = 5\n")
    with pytest.raises(ValueError, match="valid keys are"):
        config.load(tmp_path / "bad.toml")


def test_an_unknown_section_is_rejected(tmp_path):
    (tmp_path / "bad.toml").write_text("[trian]\nsteps = 5\n")
    with pytest.raises(ValueError, match="no field"):
        config.load(tmp_path / "bad.toml")


def test_overrides_are_coerced_by_the_declared_type(tmp_path):
    (tmp_path / "c.toml").write_text(BASE)
    cfg = config.load(
        tmp_path / "c.toml",
        overrides=["run.episodes=7", "run.seeds=[3,4]", "run.save_trajectories=true"],
    )
    assert cfg.run.episodes == 7 and isinstance(cfg.run.episodes, int)
    assert cfg.run.seeds == [3, 4]
    assert cfg.run.save_trajectories is True


def test_an_override_below_an_untyped_dict_keeps_its_literal(tmp_path):
    (tmp_path / "c.toml").write_text(BASE)
    cfg = config.load(tmp_path / "c.toml", overrides=["model.kwargs.device=cuda"])
    assert cfg.model.kwargs == {"device": "cuda"}


def test_a_mistyped_override_path_is_an_error(tmp_path):
    (tmp_path / "c.toml").write_text(BASE)
    with pytest.raises(ValueError, match="has no field 'episode'"):
        config.load(tmp_path / "c.toml", overrides=["run.episode=7"])


def test_an_override_without_an_equals_sign_says_what_the_shape_is(tmp_path):
    (tmp_path / "c.toml").write_text(BASE)
    with pytest.raises(ValueError, match="key=value"):
        config.load(tmp_path / "c.toml", overrides=["run.episodes"])


def test_the_config_hash_ignores_the_output_section():
    a = config.from_dict({"output": {"dir": "runs/a"}, "run": {"episodes": 4}})
    b = config.from_dict({"output": {"dir": "runs/b"}, "run": {"episodes": 4}})
    c = config.from_dict({"output": {"dir": "runs/a"}, "run": {"episodes": 5}})

    assert config.config_hash(a) == config.config_hash(b)
    assert config.config_hash(a) != config.config_hash(c)


def test_the_resolved_config_is_written_beside_its_source(tmp_path):
    source = tmp_path / "c.toml"
    source.write_text(BASE)
    cfg = config.load(source)
    config.save(cfg, tmp_path / "run", source=source)

    assert json.loads((tmp_path / "run" / "config.json").read_text())["run"]["episodes"] == 20
    assert (tmp_path / "run" / "config.toml").read_text() == BASE


def test_a_missing_config_says_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="no config at"):
        config.load(tmp_path / "absent.toml")


BENCH = """
[[models]]
name = "baseline"
target = "pkg:old"

[[models]]
name = "candidate"
target = "pkg:new"

[target]
env = "synthetic/reach"
"""


def test_a_config_with_models_is_a_benchmark(tmp_path):
    (tmp_path / "b.toml").write_text(BENCH)
    cfg = config.load(tmp_path / "b.toml")

    assert cfg.is_benchmark
    assert [m.name for m in cfg.models] == ["baseline", "candidate"]
    assert cfg.models[0].target == "pkg:old"


def test_setting_both_model_and_models_is_an_error(tmp_path):
    (tmp_path / "b.toml").write_text(BENCH + '\n[model]\ntarget = "pkg:third"\n')
    # A config where one of the two silently wins runs something other than what
    # it appears to say.
    with pytest.raises(ValueError, match=r"both \[model\] and \[\[models\]\]"):
        config.load(tmp_path / "b.toml")


def test_every_benchmark_model_needs_a_name(tmp_path):
    (tmp_path / "b.toml").write_text('[[models]]\ntarget = "pkg:x"\n')
    with pytest.raises(ValueError, match="needs a name"):
        config.load(tmp_path / "b.toml")


def test_duplicate_benchmark_names_are_rejected(tmp_path):
    (tmp_path / "b.toml").write_text(
        '[[models]]\nname = "a"\ntarget = "p:x"\n[[models]]\nname = "a"\ntarget = "p:y"\n'
    )
    with pytest.raises(ValueError, match="duplicate model name"):
        config.load(tmp_path / "b.toml")


def test_a_typo_inside_a_models_entry_names_the_valid_keys(tmp_path):
    (tmp_path / "b.toml").write_text('[[models]]\nname = "a"\ntargt = "p:x"\n')
    with pytest.raises(ValueError, match="NamedModelConfig has no field"):
        config.load(tmp_path / "b.toml")
