"""The command line, driven through ``main(argv)`` rather than a subprocess."""

from __future__ import annotations

import json

import numpy as np

from xevals.cli import main


def straight_line(obs):
    """A module-level policy, so a config can name it as ``module:attribute``."""
    state, goal = obs.get("state"), obs.get("goal")
    if state is None or goal is None:
        return np.zeros(2, dtype=np.float32)
    delta = goal - state[:2]
    norm = float(np.linalg.norm(delta))
    return (delta / norm if norm > 1e-6 else delta).astype(np.float32)


CONFIG = """
[model]
target = "test_cli:straight_line"

[target]
env = "synthetic/reach"

[suite]
name = "smoke"

[run]
episodes = 3
seeds = [0]
record = 1

[output]
dir = "{out}"
"""


def test_help_and_version_do_not_need_the_optional_extras(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0"
    assert main([]) == 0


def test_doctor_reports_every_extra_and_never_raises(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "torch" in out and "plots" in out
    assert "fails when used, not at import" in out


def test_listing_a_registry_shows_names_and_what_they_need(capsys):
    assert main(["perturbations", "list"]) == 0
    out = capsys.readouterr().out
    assert "visual/camera_shift" in out
    assert "injection/scene_text" in out


def test_describing_one_entry_is_machine_readable(capsys):
    assert main(["metrics", "describe", "robustness/retention"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dimension"] == "robustness"
    assert payload["paired"] is True


def test_dimensions_lists_the_seven_with_their_questions(capsys):
    assert main(["dimensions"]) == 0
    out = capsys.readouterr().out
    for name in ("accuracy", "robustness", "safety", "security", "efficiency",
                 "generalization", "consistency"):
        assert name in out


def test_an_unknown_entry_exits_with_a_message_not_a_traceback(capsys):
    assert main(["metrics", "describe", "accuracy/sucess_rate"]) == 2
    assert "did you mean" in capsys.readouterr().err


def test_run_writes_a_directory_and_exits_zero(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(__import__("pathlib").Path(__file__).parent))
    config = tmp_path / "config.toml"
    config.write_text(CONFIG.format(out=tmp_path / "runs"))

    assert main(["run", str(config), "--quiet"]) == 0
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "run.json").exists()
    assert (runs[0] / "config.toml").exists(), "the source config is kept verbatim"


def test_an_override_reaches_the_run(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(__import__("pathlib").Path(__file__).parent))
    config = tmp_path / "config.toml"
    config.write_text(CONFIG.format(out=tmp_path / "runs"))

    assert main(["run", str(config), "run.episodes=2", "--quiet"]) == 0
    run = next((tmp_path / "runs").iterdir())
    assert json.loads((run / "run.json").read_text())["run"]["episodes"] == 2


def test_a_mistyped_override_names_the_valid_keys(tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text(CONFIG.format(out=tmp_path / "runs"))
    # A silently ignored override produces a run at the default that looks
    # exactly like the run you asked for.
    assert main(["run", str(config), "run.episode=2", "--quiet"]) == 2
    assert "has no field 'episode'" in capsys.readouterr().err


def test_report_rebuilds_from_a_saved_run(tmp_path, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(__import__("pathlib").Path(__file__).parent))
    config = tmp_path / "config.toml"
    config.write_text(CONFIG.format(out=tmp_path / "runs"))
    main(["run", str(config), "--quiet"])
    run = next((tmp_path / "runs").iterdir())
    (run / "report.html").unlink(missing_ok=True)

    assert main(["report", str(run)]) == 0
    assert (run / "report.html").exists(), "a report must build without the model present"


BENCH_CONFIG = """
[[models]]
name = "straight"
target = "test_cli:straight_line"

[[models]]
name = "still"
target = "test_cli:do_nothing"

[target]
env = "synthetic/reach"

[suite]
name = "smoke"

[run]
episodes = 3
seeds = [0]
record = 0

[output]
dir = "{out}"
name = "bench"
"""


def do_nothing(obs):
    """A second module-level policy, so a benchmark config has two to name."""
    return np.zeros(2, dtype=np.float32)


def test_a_config_with_models_runs_a_benchmark(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(__import__("pathlib").Path(__file__).parent))
    config = tmp_path / "bench.toml"
    config.write_text(BENCH_CONFIG.format(out=tmp_path / "runs"))

    assert main(["run", str(config), "--quiet"]) == 0

    directory = tmp_path / "runs" / "bench"
    assert (directory / "leaderboard.md").exists()
    assert (directory / "index.html").exists()
    assert (directory / "config.toml").exists(), "the source config is kept verbatim"
    runs = sorted(p.name for p in directory.iterdir() if p.is_dir())
    assert len(runs) == 2
    assert any(r.startswith("straight-") for r in runs)
