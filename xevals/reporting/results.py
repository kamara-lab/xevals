"""The run directory: what an evaluation leaves behind, and how to read it back.

A results file that cannot be re-run from its own contents is a screenshot. So a
run directory carries the full config, the model and environment fingerprints,
the exact metric list, the normalisation used for every dimension score, the
baselines and the gate verdict -- everything needed to recompute the aggregate
or to reproduce the run.

::

    runs/<name>-<config hash>/
      run.json        SCHEMA 1: everything, full precision
      config.json     the resolved config; config.toml beside it if there was one
      table.md        the same numbers, rounded, for a terminal or a README
      table.tex       for a paper
      table.csv       for a spreadsheet
      metrics.json    a flat name -> value map, for a sweep aggregator
      figures/        radar, severity curves, latency, dimension bars
      videos/         a few episodes per cell, with a HUD
      trajectories/   optional, --save-trajectories
      report.html     self-contained, opens offline

Two conventions worth stating because they are what make the files usable:

**JSON keeps full precision; LaTeX and Markdown round.** Rounding belongs to
display, not to storage. A sweep that reads ``run.json`` should not inherit the
four decimal places a table chose.

**Non-finite floats become ``null``.** JSON has no NaN, and emitting a bare
``NaN`` produces a file that strict parsers reject -- discovered, as usual, by a
downstream tool six months later.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from xevals.core.dimensions import DIMENSION_ORDER, NORMALISERS, Dimension, DimensionScore
from xevals.evaluation.metrics import MetricValue
from xevals.evaluation.suites import SuiteSpec

__all__ = [
    "SCHEMA",
    "Result",
    "build_result",
    "compare",
    "jsonable",
    "leaderboard",
    "markdown_table",
    "save_table",
]

#: Version of the ``run.json`` layout. Bumped when a field changes meaning, and
#: checked on load: a results file that silently means something different from
#: what the reader expects is worse than one that refuses to open.
SCHEMA = 1


# --------------------------------------------------------------------------
# Table rendering
# --------------------------------------------------------------------------

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: str) -> str:
    """Escape LaTeX specials. Metric names like ``success_rate`` need it."""
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in text)


def format_cell(value: Any, float_format: str = "{:.4f}") -> str:
    """One cell: floats formatted, NumPy scalars unwrapped, ``None`` as an em dash.

    ``None`` prints as ``--`` rather than ``0`` or an empty cell, so a reader can
    see at a glance that a number is missing rather than small.
    """
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    item = getattr(value, "item", None)
    if item is not None and getattr(value, "ndim", 0) == 0:
        value = item()
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return "--" if not math.isfinite(value) else float_format.format(value)
    if isinstance(value, int):
        return str(value)
    return str(value)


def markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    float_format: str = "{:.4f}",
) -> str:
    """A GitHub-flavoured table, column-aligned so it reads in a terminal too.

    Numeric columns are right-aligned through the delimiter row, which GitHub and
    MkDocs both honour, and padded so the raw Markdown lines up in a diff. The
    second is what makes a table useful in a pull request, where nobody renders it.
    """
    body = [[format_cell(v, float_format) for v in row] for row in rows]
    head = [str(h) for h in headers]
    align = _column_alignment(headers, rows)
    # At least three, so a one-character column's delimiter row is a valid
    # `--:` rather than a bare `:`, which no renderer accepts.
    widths = [
        max(3, max([len(head[i])] + [len(r[i]) for r in body]) if body else len(head[i]))
        for i in range(len(head))
    ]

    def pad(text: str, width: int, right: bool) -> str:
        return text.rjust(width) if right else text.ljust(width)

    right = [a == "r" for a in align]
    lines = [
        "| " + " | ".join(
            pad(h, w, r) for h, w, r in zip(head, widths, right, strict=True)
        ) + " |",
        "| " + " | ".join(
            ("-" * (w - 1) + ":") if r else ("-" * w)
            for w, r in zip(widths, right, strict=True)
        ) + " |",
    ]
    lines += [
        "| " + " | ".join(
            pad(c, w, r) for c, w, r in zip(row, widths, right, strict=True)
        ) + " |"
        for row in body
    ]
    return "\n".join(lines)


def _column_alignment(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Left for text, right for numbers, decided per column from the data.

    Per column rather than per cell: a column with one missing value that jumped
    between left and right would be worse than either choice.
    """
    align = []
    for index in range(len(headers)):
        numeric = any(
            isinstance(row[index], (int, float)) and not isinstance(row[index], bool)
            for row in rows
        )
        align.append("r" if numeric else "l")
    return "".join(align)


def latex_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    caption: str | None = None,
    label: str | None = None,
    float_format: str = "{:.4f}",
    note: str | None = None,
) -> str:
    r"""A booktabs ``tabular``, ready to drop into a paper.

    Publication-ready means a few specific things, none of which are automatic:

    * **booktabs rules and no vertical lines.** Requires ``\usepackage{booktabs}``.
    * **Column alignment from the data**, so numeric columns line up on the right
      and their decimal points line up with each other.
    * **Numbers in ``\num``-friendly form.** Missing values become ``\textendash``
      rather than ``--``, which in text mode is an en-dash by accident and a
      broken column by intent.
    * **A caption above and a note below.** ``\caption`` goes before the tabular,
      which is the convention for tables and the opposite of figures, and the
      note carries the ``n`` and the conditions so the table survives being
      lifted out of its section.

    Args:
        note: a ``\footnotesize`` line under the rule. The right home for "20
            episodes x 3 seeds, gate passed", which a caption should not carry
            and a reader should not have to hunt for.
    """
    head = [rf"\textsc{{\small {escape_latex(str(h))}}}" for h in headers]
    body = []
    for row in rows:
        cells = []
        for value in row:
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                cells.append(r"\textendash")
            else:
                cells.append(escape_latex(format_cell(value, float_format)))
        body.append(cells)

    align = _column_alignment(headers, rows)
    lines = [
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(head) + r" \\",
        r"\midrule",
        *[" & ".join(row) + r" \\" for row in body],
        r"\bottomrule",
    ]
    if note:
        lines.append(
            rf"\addlinespace[2pt] \multicolumn{{{len(head)}}}{{l}}"
            rf"{{\footnotesize {escape_latex(note)}}} \\"
        )
    lines.append(r"\end{tabular}")
    tabular = "\n".join(lines)

    if caption is None and label is None:
        return tabular
    wrapped = [r"\begin{table}[t]", r"\centering"]
    # Caption above the tabular: the convention for tables, and the opposite of
    # the one for figures.
    if caption:
        wrapped.append(f"\\caption{{{escape_latex(caption)}}}")
    if label:
        wrapped.append(f"\\label{{{label}}}")
    wrapped += [tabular, r"\end{table}"]
    return "\n".join(wrapped)


def csv_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """CSV at full precision -- a spreadsheet can round, a file cannot un-round."""
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(headers)
    for row in rows:
        writer.writerow(["" if v is None else jsonable(v) for v in row])
    return buffer.getvalue()


def jsonable(value: Any) -> Any:
    """Anything, as something :mod:`json` can encode, without rounding."""
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "describe") and callable(value.describe):
        return jsonable(value.describe())
    return str(value)


def save_table(
    stem: str | Path,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    caption: str | None = None,
    label: str | None = None,
    note: str | None = None,
) -> dict[str, Path]:
    """Write ``<stem>.{md,tex,csv,json}`` from one set of rows.

    Four formats from one call, and each is the right shape for its reader: JSON
    at full precision for a later script, LaTeX with booktabs for a paper, CSV for
    a spreadsheet, Markdown for a terminal or a pull request.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = {}
    payload = {
        "columns": [str(h) for h in headers],
        "rows": [
            {str(h): jsonable(v) for h, v in zip(headers, row, strict=True)} for row in rows
        ],
    }
    if caption:
        payload["caption"] = caption
    for suffix, text in (
        (".md", markdown_table(headers, rows) + "\n"),
        (".tex", latex_table(headers, rows, caption=caption, label=label, note=note)
         + "\n"),
        (".csv", csv_table(headers, rows)),
        (".json", json.dumps(payload, indent=2) + "\n"),
    ):
        path = stem.with_suffix(suffix)
        path.write_text(text)
        written[suffix.lstrip(".")] = path
    return written


# --------------------------------------------------------------------------
# The result object
# --------------------------------------------------------------------------


@dataclass
class Result:
    """One evaluation, in memory and on disk.

    Attributes:
        suite: the resolved suite that ran.
        dimensions: dimension name -> score.
        cells: cell name -> metric name -> value.
        baselines: baseline name -> success rate and gate verdict.
        model: the model fingerprint.
        env: the environment fingerprint.
        run: seeds, episodes, budget, environment versions.
        directory: where it was written, once it has been.
    """

    suite: SuiteSpec
    dimensions: dict[str, DimensionScore] = field(default_factory=dict)
    cells: dict[str, dict[str, MetricValue]] = field(default_factory=dict)
    baselines: dict[str, dict[str, Any]] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)
    trajectories: dict[str, list[Any]] = field(default_factory=dict, repr=False)
    directory: Path | None = None

    # -- properties -------------------------------------------------------

    @property
    def gate_failed(self) -> bool:
        """Whether the replay gate ran and did not pass.

        An *unmeasured* gate is not a failure -- most environments cannot supply
        demonstrator actions -- but it is not a pass either, and
        :meth:`gate_status` distinguishes the three.
        """
        gate = self.baselines.get("replay")
        return bool(gate and gate.get("passed") is False)

    def gate_status(self) -> str:
        """``"passed"``, ``"failed"`` or ``"unmeasured"``."""
        gate = self.baselines.get("replay")
        if not gate or gate.get("passed") is None:
            return "unmeasured"
        return "passed" if gate["passed"] else "failed"

    @property
    def partial(self) -> bool:
        """Whether a budget truncated the run."""
        return bool(self.run.get("partial"))

    @property
    def scores(self) -> dict[str, float | None]:
        """Dimension name -> score, in the fixed dimension order."""
        return {
            d.value: self.dimensions[d.value].score
            for d in DIMENSION_ORDER
            if d.value in self.dimensions
        }

    def metric(self, name: str, cell: str = "clean") -> MetricValue | None:
        """One metric from one cell, or ``None`` if it was not measured there."""
        return self.cells.get(cell, {}).get(name)

    # -- tables -----------------------------------------------------------

    def dimension_rows(self) -> tuple[list[str], list[list[Any]]]:
        """The headline table: one row per dimension, in the fixed order."""
        headers = ["dimension", "score", "metrics", "skipped", "question"]
        rows = []
        for dim in DIMENSION_ORDER:
            score = self.dimensions.get(dim.value)
            if score is None:
                continue
            rows.append(
                [
                    dim.value,
                    score.score,
                    len(score.contributions),
                    len(score.skipped),
                    dim.question,
                ]
            )
        return headers, rows

    def metric_rows(self) -> tuple[list[str], list[list[Any]]]:
        """Every measured metric, with its cell, interval and episode count."""
        headers = ["cell", "metric", "value", "ci_low", "ci_high", "n", "unit"]
        rows = []
        for cell_name in self.cells:
            for name, value in sorted(self.cells[cell_name].items()):
                lo, hi = value.ci if value.ci else (None, None)
                rows.append(
                    [cell_name, name, value.value, lo, hi, value.n, value.unit or ""]
                )
        return headers, rows

    def cell_rows(self) -> tuple[list[str], list[list[Any]]]:
        """One row per cell: the condition and what the model scored under it."""
        headers = ["cell", "perturbation", "severity", "split", "success", "n"]
        rows = []
        for cell in self.suite.cells:
            values = self.cells.get(cell.name, {})
            success = values.get("accuracy/success_rate")
            rows.append(
                [
                    cell.name,
                    cell.perturbation,
                    cell.severity,
                    cell.split,
                    None if success is None else success.value,
                    0 if success is None else success.n,
                ]
            )
        return headers, rows

    def table(self, format: str = "md", kind: str = "dimensions") -> str:
        """Render a table. ``kind`` is ``dimensions``, ``metrics`` or ``cells``.

        The dimension table is prefixed with a warning line when the replay gate
        failed. Burying that in a JSON field would let a reader quote a success
        rate from an environment the data does not match.
        """
        headers, rows = {
            "dimensions": self.dimension_rows,
            "metrics": self.metric_rows,
            "cells": self.cell_rows,
        }[kind]()
        text = {
            "md": lambda: markdown_table(headers, rows),
            "tex": lambda: latex_table(headers, rows),
            "csv": lambda: csv_table(headers, rows),
        }[format]()
        prefix = []
        if self.gate_failed:
            prefix.append(
                "> **Gate failed.** Replaying the demonstrator's own actions did not "
                "solve the task in this environment, so the numbers below do not "
                "measure the model."
            )
        if self.partial:
            prefix.append("> **Partial run.** A budget truncated the evaluation.")
        return ("\n".join(prefix) + "\n\n" + text) if prefix and format == "md" else text

    def summary(self) -> str:
        """A few lines for a terminal: gate, dimensions, baselines."""
        lines = [f"suite {self.suite.name!r}  gate {self.gate_status()}"]
        for name, score in self.scores.items():
            shown = "--" if score is None else f"{score:.3f}"
            lines.append(f"  {name:16s} {shown}")
        for name, info in self.baselines.items():
            rate = info.get("success_rate")
            shown = "--" if rate is None else f"{rate:.2f}"
            lines.append(f"  baseline {name:8s} {shown}")
        return "\n".join(lines)

    # -- serialisation ----------------------------------------------------

    def _safe_tradeoffs(self) -> list[Any]:
        """The analyses, or none. A trade-off failing must never lose a run."""
        try:
            return self.tradeoffs()
        except Exception:  # noqa: BLE001
            return []

    def config_hash(self) -> str:
        """A digest of what defines the run, excluding where it was written."""
        payload = {
            "suite": self.suite.describe(),
            "model": {k: v for k, v in self.model.items() if k != "describe_error"},
            "env": {k: v for k, v in self.env.items() if not k.startswith("reset_state")},
            "seeds": self.run.get("seeds"),
            "episodes": self.run.get("episodes"),
            "root_seed": self.run.get("root_seed"),
        }
        execution = self.run.get("execution", {})
        if execution.get("mode", "serial") != "serial":
            payload["execution"] = {k: execution.get(k) for k in ("mode", "batch_size", "workers")}
        digest = json.dumps(jsonable(payload), sort_keys=True).encode()
        return hashlib.sha256(digest).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        """The full ``run.json`` payload."""
        return {
            "schema": SCHEMA,
            "suite": self.suite.describe(),
            "model": jsonable(self.model),
            "environment": jsonable(self.env),
            "run": jsonable(self.run),
            "gate": {"status": self.gate_status(), **jsonable(self.baselines.get("replay", {}))},
            "baselines": jsonable(self.baselines),
            "dimensions": {
                name: {
                    "score": score.score,
                    "metrics": jsonable(score.metrics),
                    "contributions": jsonable(score.contributions),
                    "skipped": score.skipped,
                }
                for name, score in self.dimensions.items()
            },
            # The normalisation is data, not code, so a reader can recompute the
            # dimension scores -- or disagree with them -- from this file alone.
            "normalisation": {
                name: {"kind": n.kind, "reference": n.reference, "why": n.why}
                for name, n in NORMALISERS.items()
            },
            "tradeoffs": [a.describe() for a in self._safe_tradeoffs()],
            "cells": {
                cell: {
                    name: {
                        "value": jsonable(v.value),
                        "ci": list(v.ci) if v.ci else None,
                        "n": v.n,
                        "unit": v.unit,
                        "reason": v.reason,
                        "per_episode": jsonable(v.per_episode),
                    }
                    for name, v in values.items()
                }
                for cell, values in self.cells.items()
            },
        }

    def save(
        self,
        out: str | Path = "runs",
        *,
        name: str | None = None,
        trajectories: bool = False,
        figures: bool = True,
        videos: bool = True,
        report: bool = True,
    ) -> Path:
        """Write the run directory and return its path.

        Figures, videos and the report are attempted and skipped with a printed
        note if their optional dependency is missing. An evaluation that
        completed should never be lost because matplotlib is not installed.
        """
        directory = Path(out) / f"{name or self.suite.name}-{self.config_hash()}"
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        (directory / "run.json").write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        (directory / "config.json").write_text(
            json.dumps(jsonable({"suite": self.suite.describe(), **self.run}), indent=2) + "\n"
        )
        headers, rows = self.dimension_rows()
        save_table(directory / "table", headers, rows, caption=f"xevals: {self.suite.name}")
        save_table(directory / "metrics", *self.metric_rows())
        save_table(directory / "cells", *self.cell_rows())
        if trajectories and self.trajectories:
            for cell, episodes in self.trajectories.items():
                for index, traj in enumerate(episodes):
                    traj.save(directory / "trajectories" / cell / f"{index:03d}")
        self._write_tradeoffs(directory)
        if figures:
            self._write_figures(directory)
        if videos:
            self._write_videos(directory)
        if report:
            self._write_report(directory)
        return directory

    def _write_tradeoffs(self, directory: Path) -> None:
        """One table per trade-off, plus a scatter where there is one to draw."""
        try:
            analyses = self.tradeoffs()
        except Exception as exc:  # noqa: BLE001 - analysis must not lose a run
            print(f"xevals: trade-offs skipped ({type(exc).__name__}: {exc})", flush=True)
            return
        for analysis in analyses:
            stem = directory / "tradeoffs" / analysis.tradeoff.name
            headers, rows = analysis.rows()
            if rows:
                save_table(stem, headers, rows, caption=analysis.tradeoff.title)
            headers, rows = analysis.group_rows()
            if rows:
                save_table(stem.with_name(f"{analysis.tradeoff.name}-groups"), headers, rows)
            try:
                from xevals.reporting import plots

                plots.tradeoff(analysis, stem.with_suffix(".png"))
            except Exception:  # noqa: BLE001 - a missing extra, or nothing to draw
                pass

    def _write_figures(self, directory: Path) -> None:
        try:
            from xevals.reporting import plots

            plots.write_all(self, directory / "figures")
        except Exception as exc:  # noqa: BLE001 - a missing extra must not lose a run
            print(f"xevals: figures skipped ({type(exc).__name__}: {exc})", flush=True)

    def _write_videos(self, directory: Path) -> None:
        if not self.trajectories:
            return
        try:
            from xevals.reporting import media

            media.write_all(
                self, directory / "videos", per_cell=None, fps=self.run.get("control_hz", 10.0)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"xevals: videos skipped ({type(exc).__name__}: {exc})", flush=True)

    def _write_report(self, directory: Path) -> None:
        try:
            from xevals.reporting import report

            report.write(self, directory / "report.html")
        except Exception as exc:  # noqa: BLE001
            print(f"xevals: report skipped ({type(exc).__name__}: {exc})", flush=True)

    def report(self, path: str | Path | None = None) -> Path:
        """(Re)build the HTML report, from the saved run if there is one."""
        from xevals.reporting import report as report_module

        target = Path(path) if path else (self.directory or Path(".")) / "report.html"
        return report_module.write(self, target)

    def tradeoffs(self, name: str | None = None) -> list[Any]:
        """Test the registered trade-offs against this run's cells.

        Returns a list of :class:`~xevals.tradeoffs.TradeOffAnalysis`, whose
        ``tension`` may well be ``"undetermined"``. That is a result: the run
        does not support a claim either way, and saying so is the point.
        """
        from xevals.reporting import tradeoffs as module

        if name is not None:
            return [module.analyse(self, name)]
        return module.analyse_all(self)

    def radar(self, path: str | Path | None = None) -> Path:
        """Write the dimension radar. Needs ``xevals[plots]``."""
        from xevals.reporting import plots

        target = Path(path) if path else (self.directory or Path(".")) / "figures" / "radar.png"
        return plots.radar(self, target)

    @classmethod
    def load(cls, path: str | Path) -> Result:
        """Read a run directory back.

        Raises:
            ValueError: when the schema version is newer than this xevals knows.
                Reading a future file with today's assumptions produces numbers
                that look fine and mean something else.
        """
        directory = Path(path)
        if directory.is_file():
            directory = directory.parent
        data = json.loads((directory / "run.json").read_text())
        if int(data.get("schema", 0)) > SCHEMA:
            raise ValueError(
                f"{directory / 'run.json'} uses schema {data['schema']}, but this xevals "
                f"understands up to {SCHEMA}; upgrade xevals to read it"
            )
        from xevals.evaluation.suites import from_dict as suite_from_dict

        suite = suite_from_dict(
            {k: v for k, v in data["suite"].items() if k != "cells"}
            | {"cells": data["suite"]["cells"]}
        )
        result = cls(
            suite=suite,
            model=data.get("model", {}),
            env=data.get("environment", {}),
            run=data.get("run", {}),
            baselines=data.get("baselines", {}),
            directory=directory,
        )
        for name, payload in data.get("dimensions", {}).items():
            result.dimensions[name] = DimensionScore(
                Dimension(name),
                payload.get("score"),
                payload.get("metrics", {}),
                payload.get("contributions", {}),
                payload.get("skipped", {}),
            )
        for cell, values in data.get("cells", {}).items():
            result.cells[cell] = {
                name: MetricValue(
                    name,
                    v.get("value"),
                    tuple(v["ci"]) if v.get("ci") else None,
                    int(v.get("n", 0)),
                    v.get("per_episode"),
                    v.get("reason", ""),
                    v.get("unit", ""),
                )
                for name, v in values.items()
            }
        return result


def build_result(
    *,
    spec: SuiteSpec,
    cells: dict[str, Any],
    baselines: dict[str, dict[str, Any]],
    model_info: dict[str, Any],
    env_info: dict[str, Any],
    run_info: dict[str, Any],
    control_hz: float = 10.0,
) -> Result:
    """Score every cell, collapse to dimensions, and package the result."""
    from xevals.evaluation.runner import dimension_scores, score_cells

    score_cells(spec, cells, model_info=model_info, control_hz=control_hz)
    result = Result(
        suite=spec,
        dimensions=dimension_scores(spec, cells, control_hz=control_hz),
        cells={name: r.values for name, r in cells.items()},
        baselines=baselines,
        model=model_info,
        env=env_info,
        run=run_info,
    )
    result.trajectories = {
        name: r.trajectories for name, r in cells.items() if r.trajectories
    }
    return result


# --------------------------------------------------------------------------
# Comparing and aggregating runs
# --------------------------------------------------------------------------


def leaderboard(paths: Sequence[str | Path]) -> tuple[list[str], list[list[Any]]]:
    """A table over several run directories, one row per run.

    Sorted by the mean dimension score, which is a *summary* and not a ranking
    anyone should defend: a model that is excellent everywhere except security
    and one that is mediocre everywhere can tie. The per-dimension columns are
    beside it precisely so the tie is visible.
    """
    results = [Result.load(p) for p in paths]
    headers = ["run", "model", "gate"] + [d.value for d in DIMENSION_ORDER] + ["mean"]
    rows = []
    for result in results:
        scores = [result.scores.get(d.value) for d in DIMENSION_ORDER]
        measured = [s for s in scores if s is not None]
        rows.append(
            [
                result.directory.name if result.directory else result.suite.name,
                result.model.get("module")
                or result.model.get("model_id")
                or result.model.get("type", "?"),
                result.gate_status(),
                *scores,
                float(np.mean(measured)) if measured else None,
            ]
        )
    rows.sort(key=lambda r: (r[-1] is None, -(r[-1] or 0.0)))
    return headers, rows


def compare(a: str | Path | Result, b: str | Path | Result) -> tuple[list[str], list[list[Any]]]:
    """Diff two runs metric by metric, flagging non-overlapping intervals.

    "Significant" here means the two 95 % bootstrap intervals do not overlap. It
    is a blunt test and it is the right blunt test for this table: it never calls
    a difference significant that a reader could not see in the error bars, which
    is the failure mode a comparison table has to avoid.
    """
    from xevals.core.dimensions import normalise
    from xevals.evaluation.metrics import METRICS

    left = a if isinstance(a, Result) else Result.load(a)
    right = b if isinstance(b, Result) else Result.load(b)
    headers = ["cell", "metric", "a", "b", "delta", "score delta", "significant"]
    rows = []
    for cell in sorted(set(left.cells) & set(right.cells)):
        for name in sorted(set(left.cells[cell]) & set(right.cells[cell])):
            va, vb = left.cells[cell][name], right.cells[cell][name]
            if va.value is None or vb.value is None:
                continue
            significant = bool(va.ci and vb.ci and (va.ci[1] < vb.ci[0] or vb.ci[1] < va.ci[0]))
            higher = True
            if name in METRICS:
                higher = bool(METRICS.entry(name).meta.get("higher_is_better", True))
            score_delta = normalise(name, vb.value, higher_is_better=higher) - normalise(
                name, va.value, higher_is_better=higher
            )
            rows.append(
                [cell, name, va.value, vb.value, vb.value - va.value, score_delta, significant]
            )
    # Ordered by the *normalised* change, not the raw one. Raw deltas are not
    # comparable across metrics: an undiscounted return moving by 40 in whatever
    # units the task chose would otherwise outrank a success rate collapsing from
    # 0.9 to 0.0, and push every bounded metric off the top of the table.
    rows.sort(key=lambda r: (not r[-1], -abs(r[5])))
    return headers, rows
