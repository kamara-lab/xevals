# xeval — implementation plan

> Multi-dimensional evaluation of robot-learning models (VLA, world models, RL policies, LLM/VLM planners, …). Not a training library: it takes *any* model as input and measures it along several dimensions — accuracy, robustness, safety, security, efficiency, generalisation, consistency — producing organised, reproducible results (tables, videos, figures, JSON).

Sibling of `xwm` (`../../research/oss/xwm`). Same brand (kamara), same docs stack, same code conventions. Where this plan says "as xwm", the concrete reference file is named so the implementer can copy rather than reinvent.

---

## 0. Context

**Problem.** Robot-learning evaluations almost always report one number: task success rate. That hides everything that decides whether a model is usable: does it survive a camera shift, a paraphrased instruction, a distractor object? Does it violate workspace limits on the way to success? Can text in the scene hijack it? How much latency does it add? Each of these is measured, if at all, with a one-off script per paper, so numbers are not comparable across models or labs.

**Goal.** A lightweight Python library, `xeval`, that:

1. **accepts any model** through a minimal adapter surface (a `Protocol` per model kind; auto-wrapping for common frameworks);
2. **runs a suite** of tasks × perturbations × metrics against it, with paired baselines and fixed seeds;
3. **scores it along named dimensions**, not only task success;
4. **writes one run directory** — `run.json` (full precision, versioned schema), tables (Markdown/LaTeX/CSV), videos, figures, trajectories, a self-contained HTML report — that can be diffed, compared and aggregated into leaderboards;
5. is **lightweight**: the core depends on `numpy` and the stdlib only; torch/jax/simulators/video codecs are optional extras that fail *when used*, never at import.

**Non-goals.** Training, fine-tuning, dataset curation, hosting a leaderboard service. `xeval` reads models and writes results.

**Design mandate.** Logo, colours, typography, docs theme, README shape, CI — identical in system to xwm (details in §2). Only the mark's cell pattern, the wordmark text, the kicker and the `xwm-`/`--xwm-` prefixes change.

---

## 1. Decisions taken up front (so implementation does not re-litigate)

| Decision | Choice | Why (and the xwm precedent) |
|---|---|---|
| Package / import name | `xeval` | matches `xwm` naming |
| Python | `>=3.11`, `.python-version` = `3.13` | as xwm |
| Build / toolchain | `hatchling`, `uv` + `uv.lock` | as xwm |
| Core deps | `numpy>=1.26` only | "lightweight" is a hard requirement; xwm's `_backend.py` lazy-import discipline |
| Arrays at the boundary | `numpy.ndarray` | framework-agnostic; adapters convert torch/jax → numpy |
| Model contract | `typing.Protocol` (`@runtime_checkable`), one per model kind; no base class required | xwm `core/types.py`: "the things being unified have no common ancestor" |
| Env contract | `Protocol` mirroring xwm `envs/protocol.py` (`reset(seed, state)`, `step(a)`, `state()`, `render()`) | lets xwm envs plug in unchanged |
| Registries | plain dicts + `available()` / `describe()` / `create()`; slash-namespaced names; lazy imports inside entries | xwm `families/registry.py` |
| Config | frozen dataclasses + `tomllib` + dotted overrides + `extends` | xwm `config/{schema,loader}.py` |
| Results | run directory; `run.json` with `SCHEMA` version + environment fingerprint; JSON full precision, LaTeX/Markdown rounded | xwm `bench/report.py`, `plots/tables.py`; "a results file that cannot be re-run from its own contents is a screenshot" |
| Logging | `print(..., flush=True)`; no `logging`, no wandb/tensorboard; optional Rerun later | as xwm |
| CLI | stdlib `argparse`; `main(argv) -> int`; deps imported inside handlers; not imported by `xeval/__init__.py` | xwm `cli/__init__.py` |
| Docs | MkDocs Material + mkdocstrings, two-line reference stubs, GitHub Pages via Actions artifact | as xwm |
| Tests | flat `tests/test_<area>.py`, sentence-case names, tiny `conftest.py`, `pytest -n auto --dist worksteal` | as xwm |
| Licence | Apache-2.0 | as xwm |
| Version | `0.1.0`, duplicated in `pyproject.toml` and `xeval/__init__.py`, asserted equal by release workflow | as xwm |
| LLM-as-judge | optional (`xeval[judge]`), rule-based judges are the default | keeps the core light and deterministic |

---

## 2. Brand & design system (port from xwm)

Everything below is copied from xwm with `xwm → xeval` renames. Source files: `xwm/docs/stylesheets/extra.css`, `xwm/overrides/partials/header.html`, `xwm/assets/*.svg`, `xwm/docs/assets/*.svg`, `xwm/mkdocs.yml`, `xwm/xwm/plots/style.py`.

### 2.1 Palette (kamara's — verbatim)

| Token | Hex | Role |
|---|---|---|
| paper | `#FAFAF8` | page / figure ground |
| ink | `#121412` | text, logo mark |
| blueprint | `#747974` | rules at full strength |
| muted | `#5C615D` | secondary text |
| grid | `#DCDEDB` | gridlines (blueprint 22 % over paper) |
| rule | `#CBCDCA` | spines, ticks (blueprint 35 % over paper) |
| accent | `#367FC9` | *the* accent — marks, rules, underlines (3.99:1 on paper → not for text) |
| accent-600 | `#2769AD` | light-mode link text (5.42:1) |
| accent-400 | `#5DA3E9` | dark-mode link text (6.95:1) |
| amber | `#C97F36` | warm counterpart (accent channels reversed) |
| amber-400 | `#E9A35D` | dark-mode amber |

Neutral ramp `--xeval-50…950`: `#fafaf8 #f2f3f0 #e4e6e2 #cbcdca #a8ada7 #8b908a #747974 #5c615d #3d423e #262a27 #121412`; near-black `#0b0d0b`; dark code bg `#1b1f1c`. Accent ramp 300–700: `#8bc5ff #5da3e9 #367fc9 #2769ad #1a5591`.

Data palettes for figures (kept off-brand for ΔE, as xwm): `BLUE_ORANGE = ("#0072B2", "#E69F00", "#56B4E9", "#D55E00")`, sequential `viridis` span `(0.06, 0.94)`, `MAX_CATEGORICAL = 5`, `BRAND_PAIR = (accent, amber)`.

**New for xeval — a fixed *dimension* palette** so every radar/bar chart colours dimensions identically across runs and docs. Seven dimensions → exceeds `MAX_CATEGORICAL`; use viridis samples at fixed positions per dimension (order below in §4.3), so a dimension always has the same colour. Document this in `guides/figures.md`.

### 2.2 Typography & geometry

- Text **Hanken Grotesk**, code **Geist Mono** (Material Google-fonts loader).
- h1 `650 / -0.02em`; h2 `600 / -0.015em` + 1 px bottom rule; hero word `3rem / 650 / -0.055em`; kicker `0.62rem / 500 / 0.22em / uppercase`; table `th` `0.66rem uppercase 0.06em`.
- `--xeval-radius: 6px`; inline code `4px`; `.md-button` pill `999px`. Derived from the logo's 18 px cell / 4.5 px radius (1:4).

### 2.3 Logo

Same geometry as xwm: 96×96 viewBox, 3×3 grid of `18×18` rects, `rx="4.5"`, at x/y ∈ {14, 39, 64}; filled cells in ink, outlined cells `fill="none" stroke-width="2.5" opacity="0.3"` (inverse variants `0.45` wordmark / `0.38` mark). Tile variants add `<rect width="96" height="96" rx="20"/>`.

**xeval's cell pattern** (xwm traces an X; xeval traces a **check mark**):

```
row 0:  ○ ○ ●      (r,c) filled: (0,2) (1,0) (1,2) (2,1)
row 1:  ● ○ ●      outlined: the other five
row 2:  ○ ● ○
```

Filled cells = the measurement that passed; outlined cells = the dimensions still to be measured. `<desc>`: "Filled cells trace a check; outlined cells are the dimensions an evaluation has yet to measure."

Wordmark: `<text x="118" y="63" font-size="52" font-weight="600" letter-spacing="-1.5">xeval</text>` and kicker `<text x="119" y="84" font-size="13" letter-spacing="2.6" opacity="0.75">EVALUATIONS</text>`; viewBox `0 0 262 96` may need widening to ~`0 0 290 96` for the longer word — measure and adjust.

Files to produce:

| Path | Content |
|---|---|
| `assets/logo.svg` | mark + wordmark, ink |
| `assets/logo-inverse.svg` | same, paper |
| `assets/logo-mark.svg` | paper tile + ink mark (== `docs/assets/favicon.svg`) |
| `assets/logo-mark-inverse.svg` | ink tile + paper mark (== `docs/assets/logo-mark-inverse.svg`) |
| `assets/banner.webp` | 1600×300, for README header (generated from the SVG via a small script in `tools/` or by hand) |
| `docs/assets/logo-mark-ink.svg` | 96×96 transparent, ink — the Material `theme.logo`; dark mode uses `filter: invert(1)` |
| `docs/assets/texture-accent.png` | 720×323 **alpha-only** blob mask for the hero (copy xwm's; the colour comes from CSS) |

Hero markup in `docs/index.md` reuses the `.xeval-hero__grid` with the check pattern (4 `xeval-hero__cell`, 5 `xeval-hero__cell--masked`).

### 2.4 Docs theme files

- `mkdocs.yml`: copy xwm's; change `site_name: xeval`, `site_description: Multi-dimensional evaluation of robot-learning models.`, URLs to `kamara-lab/xeval`, `nav` (§9), mkdocstrings `paths: [.]` unchanged, same features list (no `navigation.tabs`), same `markdown_extensions`, same `validation`, `hooks: [docs/hooks/figures.py]`, `exclude_docs` (`*.tex`, `hooks/`).
- `docs/stylesheets/extra.css`: copy verbatim, `sed 's/xwm/xeval/g'`; header comment updated.
- `overrides/partials/header.html`: copy verbatim; rename `xwm-powered → xeval-powered`. Keep the "re-copy when upgrading Material" comment.
- `docs/hooks/figures.py` and `docs/hooks/rst_docstrings.py`: copy verbatim (they are generic).
- `docs/javascripts/mathjax.js`: copy verbatim.

### 2.5 README

Same skeleton as xwm's: centred banner → bold one-line tagline → `Powered by kamara` → six shields (`style=flat-square&color=367FC9&labelColor=FAFAF8`: PyPI, Python 3.11+, "framework-agnostic · numpy", tests, ruff, Apache-2.0) → `---` → two dense paragraphs → `<picture>` light/dark diagram → `##` sections: Install · Quick start · What gets measured (dimensions table) · Wrapping a model · Environments & datasets · Suites · Results & reports · Examples · Conventions · Tests · References · Contributors · Supported by · License. British spelling, declarative tone, "why" comments, negative results stated plainly.

---

## 3. Repository layout

```
xeval/
  .github/workflows/{ci,docs,release}.yml
  .plan/implementation.md            (this file)
  .gitignore  .python-version  pyproject.toml  uv.lock  mkdocs.yml  README.md  LICENSE
  assets/                            README brand assets (§2.3)
  overrides/partials/header.html
  configs/
    _base/policy.toml  _base/world_model.toml  _base/planner.toml
    suites/{core,robustness,safety,security,efficiency,generalization,consistency}.toml
    examples/*.toml
  docs/  (§9)
  examples/  (§10)
  tests/  (§11)
  xeval/
    __init__.py         re-exports + __version__ + `evaluate` + `wrap`
    core/               types.py (Protocols, Trajectory, Step, Batch), seeding.py, dimensions.py, errors.py
    adapters/           registry.py, callable.py, torch.py, jax.py, lerobot.py, hf.py, remote.py, xwm.py
    envs/               protocol.py, registry.py, gymnasium.py, replay.py, wrappers.py, xwm.py
    datasets/           spec.py, registry.py, lerobot.py, hdf5.py, npz.py, cache.py
    tasks/              spec.py, registry.py, scoring.py
    perturbations/      base.py, registry.py, visual.py, sensor.py, action.py, instruction.py, dynamics.py, adversarial.py, injection.py
    metrics/            base.py, registry.py, accuracy.py, robustness.py, safety.py, security.py, efficiency.py, generalization.py, consistency.py, prediction.py, text.py
    judges/             base.py, rules.py, llm.py (optional extra)
    suites/             spec.py, registry.py, builtin.py
    runner/             episode.py, rollout.py, offline.py, baselines.py, budget.py, run.py
    results/            schema.py, store.py, tables.py, aggregate.py, compare.py
    media/              video.py, gif.py, overlay.py, _backend.py
    plots/              style.py, radar.py, curves.py, heatmap.py, export.py, _backend.py
    report/             html.py, templates/report.html, assets/report.css
    config/             schema.py, loader.py
    cli/                __init__.py, run.py, compare.py, report.py, list.py, doctor.py
    tools/              fingerprint.py, hashing.py, timing.py
```

Every subpackage `__init__.py` is a curated façade with a case-insensitively sorted `__all__`, and its module docstring is the docs page (mkdocstrings renders it). `xeval.cli` is not imported at top level.

---

## 4. Core design

### 4.1 Model contracts — `xeval/core/types.py`

All `@runtime_checkable Protocol`s. Arrays are `numpy.ndarray`; observations are `dict[str, Any]` (`Obs = dict[str, Any]`) so models pick the fields they need without the runner knowing the modality — same reasoning as xwm's `Batch`.

```python
class Policy(Protocol):                       # VLA, BC, RL actors
    def reset(self) -> None: ...              # optional: clear history
    def act(self, obs: Obs, *, instruction: str | None = None) -> Action: ...

class WorldModel(Protocol):                   # predictive / latent dynamics
    def predict(self, obs: Obs, actions: np.ndarray, *, horizon: int) -> Prediction: ...
    # Prediction = dict: "frames" | "latents" | "rewards" | "states" (any subset)

class Planner(Protocol):                      # LLM / VLM task planners, code-as-policy
    def plan(self, instruction: str, *, obs: Obs | None = None) -> Plan: ...
    # Plan = dict: "text", optional "steps": list[str], "code": str, "refused": bool

class Scorer(Protocol):                       # reward / value / success classifiers
    def score(self, obs: Obs, *, instruction: str | None = None) -> float: ...
```

Optional capability protocols the runner sniffs with `isinstance`: `HasConfidence` (`confidence(obs) -> float`, for calibration), `HasLatents` (`encode(obs) -> ndarray`, for representation metrics), `HasCost` (`params`, `device`). Missing capability ⇒ the dependent metric is reported as `null` with a reason, never an error.

`Trajectory` (frozen dataclass): `obs: list[Obs]`, `actions`, `rewards`, `infos`, `frames: list[ndarray] | None`, `instruction`, `seed`, `perturbation: str | None`, `success: bool | None`, `timing: dict` (per-step latency), `violations: list[Violation]`. Serialisable to `npz` + JSON sidecar.

### 4.2 Adapters — `xeval/adapters/`

`xeval.wrap(obj, kind=None, **hints) -> Policy | WorldModel | Planner`:

1. already satisfies a Protocol → return as-is;
2. plain callable → `CallablePolicy` (or `kind="world_model"` etc.);
3. `torch.nn.Module` → `TorchPolicy` (`eval()`, `no_grad`, device placement, numpy↔tensor at the boundary, dict obs support);
4. jax/equinox callable → `JaxPolicy` (`jax.device_get` at the boundary);
5. `lerobot` policy → `LeRobotPolicy` (normalisation stats, action chunking, `select_action` loop);
6. HF `transformers` VLA (OpenVLA-style) → `HFVLAPolicy` (prompt template, action un-normalisation, tokenizer);
7. HTTP endpoint → `RemotePolicy` (JSON/msgpack, latency measured client-side);
8. xwm `WorldModel` (`Plannable`) → `XWMWorldModel` — zero-copy bridge to the sibling library;
9. LLM chat model (OpenAI/Anthropic-compatible client, or any `str -> str` callable) → `ChatPlanner`.

Each adapter lives in its own module and imports its framework *inside* the class/function, so `import xeval.adapters` is always free. `adapters/registry.py`: `ADAPTERS: dict[str, Callable]`, `available()`, `describe()`, `create(name, **kw)`, plus `detect(obj) -> str | None`.

### 4.3 Dimensions — `xeval/core/dimensions.py`

Fixed, ordered enum (order fixes colours and table columns):

| # | Dimension | Question it answers | Representative metrics |
|---|---|---|---|
| 1 | `accuracy` | Does it do the task? | success rate, return, goal distance, action MSE/MAE on held-out data, prediction PSNR/SSIM/latent error (WM), plan correctness / executability (planner) |
| 2 | `robustness` | Does it keep doing it under nuisance change? | success retention ratio under each perturbation family, area-under-severity curve, worst-case success |
| 3 | `safety` | Does it stay within physical/behavioural limits? | workspace/joint/velocity/force violation rate, collision count, unsafe-instruction refusal rate, min-distance-to-obstacle |
| 4 | `security` | Can it be hijacked or made to fail on purpose? | success under adversarial patch / pixel attack, instruction-injection compliance rate, backdoor-trigger delta, jailbreak rate (planner) |
| 5 | `efficiency` | What does it cost to run? | p50/p95 step latency, throughput, peak memory, params, control-frequency headroom |
| 6 | `generalization` | Does it transfer to unseen objects/scenes/instructions? | success on OOD splits (unseen object, layout, instruction template), in-vs-out gap |
| 7 | `consistency` | Are its answers stable? | seed variance, paraphrase invariance (action agreement), rollout determinism, calibration (ECE/Brier when `HasConfidence`) |

A `DimensionScore` = mean of the dimension's normalised metric scores in `[0, 1]` plus the raw metrics; `higher_is_better` metrics are used as-is, others inverted with a documented reference scale. The normalisation is written to `run.json` so the aggregate is reproducible.

### 4.4 Perturbations — `xeval/perturbations/`

`Perturbation` protocol: `name`, `dimension` (robustness or security), `severity: float`, and one of `apply_obs(obs)`, `apply_action(a)`, `apply_instruction(text)`, `apply_env(env)`. Composable via `Compose([...])`. Registry `PERTURBATIONS` with slash names:

- `visual/{gaussian_noise, blur, brightness, contrast, color_shift, jpeg, occlusion, cutout, camera_shift, camera_rotate, crop, distractor_overlay}`
- `sensor/{dropout, latency, frame_skip, resolution}`
- `action/{noise, delay, clip, dropout}`
- `instruction/{paraphrase, typo, reorder, translate?, distractor_clause}` (paraphrase from a bundled template set; optional LLM paraphraser under `xeval[judge]`)
- `dynamics/{mass, friction, gravity, actuator_gain}` (requires an env exposing `set_physics(**kw)`; otherwise skipped with reason)
- `adversarial/{patch, pgd_linf}` (gradient-based ones need `torch`/`jax` extra and a differentiable adapter; the random-search patch works on any model)
- `injection/{scene_text, instruction_suffix, system_override}` (text-in-image via PIL, appended instructions)

Every perturbation is deterministic given `(seed, severity)`, and states its severity ladder (`SEVERITIES = (0.2, 0.4, 0.6, 0.8, 1.0)` by default) so severity curves are comparable across models.

### 4.5 Metrics — `xeval/metrics/`

`Metric` protocol: `name`, `dimension`, `higher_is_better`, `unit`, `requires: frozenset[str]` (e.g. `{"frames"}`, `{"confidence"}`), `compute(trajs: list[Trajectory], *, ref: list[Trajectory] | None) -> MetricValue`. Metrics are **pure functions over trajectories** (as xwm's `metrics/`), so the same metric serves online rollouts and offline datasets. `MetricValue` carries `value`, `ci: (lo, hi) | None` (bootstrap, seeded), `n`, `per_episode: list[float] | None`.

Registry `METRICS`, names `accuracy/success_rate`, `robustness/retention`, `safety/violation_rate`, … Bootstrap CI helper in `metrics/base.py`.

### 4.6 Tasks, envs, datasets

- `Env` protocol as xwm: `reset(*, seed, state=None) -> Obs`, `step(a) -> (Obs, reward, done, info)`, `state()`, `render() -> ndarray | None`, plus optional `success(info) -> bool`, `limits() -> SafetyLimits`, `set_physics(**kw)`. Invariant kept from xwm: `reset(state=env.state())` reproduces the state (tested).
- Env adapters: `gymnasium` (incl. LIBERO/ManiSkill/SimplerEnv via their gym APIs — extras `sim`, `libero`), `xwm` envs, and `ReplayEnv` — an env built from a dataset episode that returns recorded observations regardless of action, enabling **offline evaluation** (action error, prediction error) with the same runner.
- `TaskSpec` (frozen): `name`, `env`, `env_kwargs`, `instruction(s)`, `horizon`, `success` metric name, `splits` (`in`, `ood/object`, `ood/scene`, `ood/instruction`) for generalisation.
- `DatasetSpec` + readers for LeRobot (`parquet` + mp4), HDF5 (robomimic-style), NPZ; a `cache.require()` helper as xwm.

### 4.7 Suites — `xeval/suites/`

`SuiteSpec` = `name`, `dimensions`, list of `(task, perturbations, metrics, episodes, seeds)` cells, and `baselines`. Built-ins: `core` (accuracy + efficiency + safety at severity 0), `robustness`, `safety`, `security`, `generalization`, `consistency`, and `full` (all). Suites are also expressible in TOML (`configs/suites/*.toml`) and the registry loads either.

### 4.8 Runner — `xeval/runner/`

`xeval.evaluate(model, env_or_dataset, *, suite="core", episodes=20, seeds=(0,), record=True, out="runs/...", budget=None, on_step=None) -> Result`

1. Fingerprint model (`adapters.describe`, param count if available, hash of adapter kwargs) and environment (platform, python, versions of numpy/torch/jax if present, GPU).
2. For each suite cell: build env, wrap with perturbation wrappers, run `episodes × seeds` rollouts (`runner/rollout.py`), recording frames if `record`, timing each `act`.
3. **Paired baselines** on identical seeds: `random`, `noop`, and — when a dataset is present — `replay`, which is a *gate* (as xwm): if replaying the demonstrator's actions does not succeed, the env is not faithful and the run is flagged `gate_failed`.
4. Compute metrics, dimension scores, CIs.
5. Write the run directory (§4.9). Progress via `print(..., flush=True)`; `on_step` hook is `None`-tolerant for Rerun or custom logging.
6. `budget`: a wall-clock/episode cap that truncates gracefully and marks the run `partial`.

Offline path (`runner/offline.py`): same loop over `ReplayEnv`, computing action-error and prediction metrics; used for WMs and for datasets without a simulator.

### 4.9 Results — `xeval/results/`

Run directory (`out/<name>-<config_hash>/`):

```
run.json              SCHEMA=1: model fingerprint, env fingerprint, suite spec, seeds, per-cell metric values + CIs,
                      dimension scores + normalisation used, baselines, gate status, timing, partial flag
config.toml / config.json     verbatim source + resolved config (as xwm)
table.md  table.tex  table.csv  metrics.json      (results/tables.py, as xwm plots/tables.save_table)
trajectories/<cell>/<seed>-<ep>.npz (+ .json)     optional, --save-trajectories
videos/<cell>/<seed>-<ep>.mp4 (or .gif)           first N episodes per cell; overlay: step, instruction, perturbation, success
figures/radar.png  severity-<family>.png  latency.png  dimension-bars.png
report.html                                       self-contained (inline CSS from report.css, base64 thumbnails, links to videos)
```

`Result` object: `.table(format="md")`, `.dimensions`, `.metrics`, `.radar()`, `.report()`, `.save()`, `.load(path)`. `results/aggregate.py` builds a **leaderboard** over many run dirs (`xeval compare runs/*`), grouping by model and averaging by dimension; `results/compare.py` diffs two runs and highlights significant changes (non-overlapping CIs).

### 4.10 Media & plots

- `media/video.py`: `save_video(frames, path, fps=10)` via `imageio-ffmpeg` (extra `video`), fallback to GIF via Pillow; `overlay.py` draws the HUD strip in paper/ink with the brand fonts falling back to PIL default.
- `plots/style.py`: port xwm's brand tokens and palettes; `radar.py` (dimension radar, multi-model overlay), `curves.py` (success vs severity per perturbation family with CI bands), `heatmap.py` (task × perturbation), `export.py` (`save_figure`, `tile_frames`, `save_gif` — port from xwm `plots/export.py`).

### 4.11 Config & CLI

- `config/schema.py`: `ModelConfig(adapter, kwargs)`, `TargetConfig(env | dataset, kwargs, task)`, `SuiteConfig(name | cells)`, `RunConfig(episodes, seeds, record, save_trajectories, budget)`, `OutputConfig(dir, name)`, `ExperimentConfig`. `loader.py`: `load(path, overrides)`, `extends`, unknown-key errors listing valid keys, type-coerced overrides, `config_hash` excluding output section — port xwm's loader.
- CLI: `xeval run config.toml [k=v ...]`, `xeval compare runs/* [--out leaderboard/]`, `xeval report runs/x/` (rebuild HTML/figures), `xeval {suites,metrics,perturbations,adapters,envs} {list,describe}`, `xeval doctor` (which extras are installed: torch/jax/gymnasium/ffmpeg/PIL/…).

---

## 5. Public API (what `README` Quick start shows)

```python
import xeval

policy = xeval.wrap(my_model)                          # torch / jax / lerobot / HF / callable / http
env = xeval.envs.create("gymnasium/PushT-v0")          # or xeval.envs.replay(dataset)

result = xeval.evaluate(
    policy, env,
    suite="full",                                      # accuracy · robustness · safety · security · efficiency · generalization · consistency
    episodes=20, seeds=range(3), record=True,
    out="runs/pusht",
)
print(result.table())                                  # dimension scores + key metrics, Markdown
result.report()                                        # runs/pusht-<hash>/report.html, videos/, figures/
```

Top-level names: `evaluate`, `wrap`, `Result`, `Trajectory`, `Dimension`, `set_seed`, and the subpackages.

---

## 6. Dependencies & extras (`pyproject.toml`)

```toml
dependencies = ["numpy>=1.26"]

[project.optional-dependencies]
torch  = ["torch>=2.2"]
jax    = ["jax>=0.4.38"]
sim    = ["gymnasium>=1.0"]
lerobot= ["lerobot>=0.1", "pyarrow>=17", "av>=12"]
data   = ["pyarrow>=17", "h5py>=3.11", "av>=12", "huggingface-hub>=0.26"]
video  = ["imageio>=2.34", "imageio-ffmpeg>=0.5", "pillow>=10.0"]
plots  = ["matplotlib>=3.8", "pillow>=10.0"]
judge  = ["anthropic>=0.40", "openai>=1.40"]          # LLM-as-judge / paraphraser; optional
xwm    = ["xwm>=0.1"]
dev    = ["pytest>=8.0", "pytest-xdist>=3.6", "ruff>=0.6", "matplotlib>=3.8", "pillow>=10.0",
          "imageio>=2.34", "gymnasium>=1.0", "torch>=2.2"]
docs   = ["mkdocs>=1.6", "mkdocs-material>=9.5", "mkdocstrings[python]>=0.26"]
```

Ruff `line-length = 100`, `select = ["E","F","I","UP","B"]`, `target-version = "py311"`. `[project.scripts] xeval = "xeval.cli:main"`. Hatch sdist `only-include = ["xeval","tests","examples","configs","README.md","pyproject.toml"]`, `exclude = ["examples/outputs"]`.

---

## 7. Phased implementation with checklists

Each phase ends with `ruff check`, `pytest -q -n auto`, and (from phase 2 on) `mkdocs build --strict` passing.

### Phase 0 — Repository bootstrap
- [ ] `git init`; `.gitignore` (xwm's + `runs/`, `examples/outputs/`, `*.mp4`, `*.gif` under outputs)
- [ ] `.python-version` = `3.13`; `pyproject.toml` per §6; `uv lock`; `uv pip install -e ".[dev,docs]"`
- [ ] `LICENSE` (Apache-2.0), `xeval/__init__.py` with `__version__ = "0.1.0"` and docstring map
- [ ] `.github/workflows/ci.yml` (lint / test 3.11–3.13 / docs / package), `docs.yml` (Pages via artifact), `release.yml` (tag == version, Trusted Publishing) — copy xwm's, rename, drop the JAX compilation-cache steps
- [ ] Bare-install smoke test in CI: `python -c "import xeval, xeval.adapters, xeval.plots, xeval.media"` in an env with only numpy

### Phase 1 — Brand & docs skeleton
- [ ] Logo SVGs per §2.3 (check-mark pattern; ink, inverse, mark, mark-inverse, mark-ink, favicon); verify wordmark fits the viewBox
- [ ] `assets/banner.webp` 1600×300 (paper ground, mark + wordmark + tagline; produce via a `tools/make_banner.py` using Pillow + cairosvg or export by hand — document which)
- [ ] `docs/assets/texture-accent.png` copied from xwm (verify alpha channel exists)
- [ ] `docs/stylesheets/extra.css` ported with `xeval` prefixes; `overrides/partials/header.html` ported (`xeval-powered`)
- [ ] `mkdocs.yml` per §2.4 with nav per §9; `docs/hooks/*.py`, `docs/javascripts/mathjax.js` copied
- [ ] `docs/index.md` hero with check-pattern grid, kicker "Evaluations", tagline
- [ ] `README.md` header block (banner, tagline, powered-by, six badges) and section headings (bodies filled in Phase 12)
- [ ] `mkdocs build --strict` passes on the skeleton

### Phase 2 — Core types & seeding
- [ ] `core/types.py`: `Obs`, `Action`, `Prediction`, `Plan`, `Policy`, `WorldModel`, `Planner`, `Scorer`, capability protocols, `Trajectory`, `Step`, `Violation`, `SafetyLimits`
- [ ] `core/dimensions.py`: `Dimension` enum (ordered), `DimensionScore`, normalisation table with references
- [ ] `core/seeding.py`: `set_seed`, `seed_sequence(seed, n)` (numpy `SeedSequence`), documented determinism contract
- [ ] `core/errors.py`: `MissingExtra`, `GateFailed`, `CapabilityMissing`
- [ ] Tests: protocols are runtime-checkable; `Trajectory` round-trips through npz+json; seeds reproduce

### Phase 3 — Adapters
- [ ] `adapters/registry.py` (`ADAPTERS`, `available`, `describe`, `create`, `detect`) and `xeval.wrap`
- [ ] `callable.py`, `torch.py`, `jax.py`, `lerobot.py`, `hf.py` (OpenVLA-style prompt/action decode), `remote.py`, `xwm.py`, `chat.py` (`ChatPlanner`)
- [ ] Each adapter: numpy at the boundary, `describe()` returns params/device/framework, optional `confidence`/`encode` passthrough
- [ ] Tests: callable + fake torch module (importorskip) + fake HTTP server; `detect` picks the right adapter; `import xeval.adapters` works without torch/jax

### Phase 4 — Envs, datasets, tasks
- [ ] `envs/protocol.py`, `envs/registry.py`, `envs/gymnasium.py`, `envs/replay.py` (`ReplayEnv`), `envs/wrappers.py` (perturbation, recording, timing, limit-tracking), `envs/xwm.py`
- [ ] `datasets/{spec,registry,lerobot,hdf5,npz,cache}.py`
- [ ] `tasks/spec.py` with splits; `tasks/registry.py`; `tasks/scoring.py` (`success`, `goal_distance`)
- [ ] A **synthetic built-in env** (`envs/synthetic.py`: 2-D point-mass reach/push with a rendered frame, workspace limits, a settable physics param, an object palette for OOD splits, and scene-text rendering for injection tests) so the whole library is testable and demo-able with zero extras
- [ ] Tests: `reset(state=env.state())` invariant; replay env returns recorded obs; synthetic env renders deterministic frames

### Phase 5 — Perturbations
- [ ] `perturbations/base.py` (`Perturbation`, `Compose`, `SEVERITIES`), `registry.py`
- [ ] `visual.py`, `sensor.py`, `action.py`, `instruction.py` (bundled paraphrase templates), `dynamics.py`, `adversarial.py` (random-search patch core; PGD behind torch/jax), `injection.py` (PIL scene text)
- [ ] Tests: determinism given seed/severity; severity monotonicity where meaningful; shapes/dtypes preserved; `describe()` lists requirements

### Phase 6 — Metrics & judges
- [ ] `metrics/base.py` (`Metric`, `MetricValue`, bootstrap CI), `registry.py`
- [ ] `accuracy.py` (success, return, goal distance, action MSE/MAE, plan correctness), `robustness.py` (retention, AUC over severity, worst-case), `safety.py` (violation rates from `SafetyLimits`, collisions, refusal), `security.py` (adversarial delta, injection compliance, trigger delta), `efficiency.py` (latency percentiles, throughput, memory, params), `generalization.py` (split gaps), `consistency.py` (seed variance, paraphrase agreement, ECE/Brier), `prediction.py` (PSNR/SSIM/latent error, horizon degradation for WMs), `text.py` (exact/fuzzy match, executability check)
- [ ] `judges/rules.py` (keyword/regex/schema judges for plans and refusals), `judges/llm.py` (optional; prompt templates, cached, seeded temperature 0)
- [ ] Tests: hand-built trajectories with known answers per metric; CI widths shrink with `n`; `requires` gating yields `null` + reason

### Phase 7 — Runner & baselines
- [ ] `runner/episode.py`, `rollout.py` (online), `offline.py` (ReplayEnv path), `baselines.py` (`random`, `noop`, `replay` gate), `budget.py`, `run.py` (`evaluate`)
- [ ] Fingerprinting (`tools/fingerprint.py`), config hashing (`tools/hashing.py`), timing (`tools/timing.py`)
- [ ] `on_step` hook, `None`-tolerant; `partial` flag on budget truncation
- [ ] Tests: full `evaluate` on synthetic env + callable policy under 5 s; gate failure surfaces; seeds reproduce metric values bit-for-bit

### Phase 8 — Results, tables, aggregation
- [ ] `results/schema.py` (`SCHEMA = 1`, dataclasses), `store.py` (write/load run dir), `tables.py` (port xwm `save_table`, `markdown_table`, `jsonable`, `format_cell`; add CSV), `aggregate.py` (leaderboard), `compare.py` (diff with CI overlap)
- [ ] `Result` object API (§4.9)
- [ ] Tests: run dir round-trip; table formats agree on values; leaderboard ordering; schema version check on load

### Phase 9 — Media, plots, HTML report
- [ ] `media/_backend.py` (lazy imageio/PIL), `video.py`, `gif.py`, `overlay.py` (HUD in brand colours)
- [ ] `plots/style.py` (ported tokens + dimension palette), `radar.py`, `curves.py`, `heatmap.py`, `export.py`
- [ ] `report/html.py` + `templates/report.html` + `assets/report.css` (same tokens as `extra.css`; light/dark via `prefers-color-scheme`): summary header, dimension radar, per-dimension sections with tables + severity curves, video gallery (thumbnails → mp4), baselines & gate panel, environment fingerprint, "how to reproduce" block with the exact CLI
- [ ] Tests: GIF fallback when ffmpeg missing; figures render headless (`Agg`); report builds from a saved run without the model present

### Phase 10 — Suites, configs, CLI
- [ ] `suites/spec.py`, `registry.py`, `builtin.py` (`core`, `robustness`, `safety`, `security`, `generalization`, `consistency`, `full`) + TOML equivalents in `configs/suites/`
- [ ] `config/schema.py`, `config/loader.py` (port); `configs/_base/*.toml`; `configs/examples/*.toml`
- [ ] `cli/`: `run`, `compare`, `report`, `list/describe` for every registry, `doctor`; `main(argv) -> int`
- [ ] Tests: `extends` merges; unknown key errors list valid keys; `xeval run configs/examples/synthetic.toml` end-to-end in tmp dir; `xeval doctor` exit 0

### Phase 11 — Examples
- [ ] `examples/_common.py` (`setting()` reading `XEVAL_*` env vars, `outputs()`, `setup()`), as xwm
- [ ] `01_synthetic_policy.py` — full suite on the built-in env, scripted policy; produces every artifact type
- [ ] `02_torch_policy.py` — wrap a tiny torch MLP; efficiency + robustness
- [ ] `03_lerobot_policy.py` — LeRobot policy on a LeRobot dataset via `ReplayEnv` (offline) — gated on extras
- [ ] `04_vla_openvla.py` — HF VLA on SimplerEnv/LIBERO with instruction perturbations and injection — gated
- [ ] `05_world_model.py` — xwm or callable WM: prediction accuracy, horizon degradation, robustness of predictions
- [ ] `06_llm_planner.py` — `ChatPlanner`: plan correctness, paraphrase consistency, refusal on unsafe instructions, injection resistance
- [ ] `07_compare_models.py` — several runs → leaderboard + overlaid radar
- [ ] Each example's outputs are what `docs/hooks/figures.py` publishes into the docs

### Phase 12 — Docs content & README
- [ ] Getting started: `installation.md` (extras matrix), `quickstart.md`, `conventions.md` (arrays are numpy; seeds; run dirs; null-with-reason; British spelling)
- [ ] Concepts: `dimensions.md` (the seven, with normalisation), `perturbations.md`, `baselines-and-gates.md`, `results-schema.md`, `randomness.md`
- [ ] Guides: `wrapping-models.md`, `environments.md`, `datasets.md`, `suites.md`, `metrics.md`, `videos.md`, `figures.md`, `reports.md`, `leaderboards.md`, `cli.md`, `examples.md`, `extending.md`
- [ ] Findings page (`findings.md`) reserved for measured observations from the examples
- [ ] `reference/index.md` grid cards (Contracts / Measuring / Running / Reporting) + one two-line stub per subpackage
- [ ] README bodies per §2.5; badge counts updated; `<picture>` diagram (`docs/assets/pipeline-{light,dark}.svg`: model → adapter → runner (env × perturbations) → metrics → dimensions → run dir)
- [ ] `mkdocs build --strict` clean; every public docstring Google-style

### Phase 13 — Release
- [ ] `uv build`, `twine check --strict`, wheel import smoke test
- [ ] Tag `v0.1.0`; release workflow to TestPyPI, then PyPI; GitHub Release notes
- [ ] Pages enabled (Settings → Pages → Source: GitHub Actions); site live at `kamara-lab.github.io/xeval/`

---

## 8. Verification (end-to-end)

1. **Cold install:** `uv venv && uv pip install .` → `python -c "import xeval; print(xeval.__version__)"` succeeds with numpy only; `xeval doctor` reports every extra as missing without raising.
2. **Synthetic end-to-end:** `python examples/01_synthetic_policy.py` produces `runs/…/{run.json, table.md, table.tex, table.csv, videos/, figures/radar.png, report.html}` in < 60 s on CPU; `report.html` opens offline and shows all seven dimensions.
3. **Determinism:** run twice with the same seeds → `metrics.json` byte-identical (excluding timing fields).
4. **Gate:** deliberately break `ReplayEnv` fidelity in a test → `gate_failed` is set and the table carries the warning.
5. **Any-model:** tests wrap a callable, a torch module, a fake HTTP server, and (importorskip) an xwm model through `xeval.wrap` and run the same suite cell.
6. **Compare:** two runs → `xeval compare` prints a leaderboard and writes an overlaid radar.
7. **Docs/CI:** `ruff check`, `pytest -n auto`, `mkdocs build --strict`, `uv build && twine check` all green on 3.11/3.12/3.13.
8. **Design parity:** open xwm docs and xeval docs side-by-side: same header, fonts, colours, radii; only mark pattern, wordmark and kicker differ.

---

## 9. Docs nav (mkdocs.yml)

```
Home
Getting started: Installation · Quickstart · Conventions
Concepts: index · Dimensions · Perturbations · Baselines & gates · Results schema · Randomness
Guides: Wrapping models · Environments · Datasets · Suites · Metrics · Videos · Figures · Reports · Leaderboards · CLI · Examples · Extending
Findings
API reference: index · core · adapters · envs · datasets · tasks · perturbations · metrics · judges · suites · runner · results · media · plots · report · config · tools
```

---

## 10. Open questions (do not block Phase 0–2; decide before Phase 3/4)

- **Third-party sims to support first-class in 0.1:** SimplerEnv and LIBERO are the natural VLA targets; ManiSkill3 next. Proposal: gymnasium adapter only in 0.1, with LIBERO/SimplerEnv recipes in `guides/environments.md`; dedicated adapters in 0.2.
- **LLM-as-judge default provider:** keep rule-based judges default; `xeval[judge]` supports Anthropic and OpenAI-compatible clients through one tiny interface.
- **Logo pattern:** check-mark proposed in §2.3; alternative is a "scorecard" (rows filled 3/2/1). Confirm before producing the banner.
- **Repo/org:** `kamara-lab/xeval` assumed for all URLs and badges.
