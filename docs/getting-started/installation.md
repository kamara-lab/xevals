# Installation

```bash
pip install xevals
```

That is the whole core: `xevals` and `numpy`. Everything else is an extra, and
every extra fails **when it is used**, never at import. A bare install can import
the library, list every registered adapter, metric and perturbation, and run the
full seven-dimension suite against the built-in world.

```bash
python -c "import xevals; print(xevals.__version__)"
xevals doctor
```

`xevals doctor` prints which extras are present, which are missing, and what each
one unlocks. It never raises, that is the whole job.

## Extras

| Extra | Installs | Unlocks |
|---|---|---|
| `torch` | `torch` | wrapping `torch.nn.Module`, HuggingFace VLAs |
| `jax` | `jax` | wrapping JAX and Equinox callables |
| `sim` | `gymnasium` | Gymnasium environments, and through them LIBERO and ManiSkill |
| `mujoco` | `mujoco`, `robot_descriptions` | the five [Menagerie manipulation robots](../guides/robots.md) |
| `newton` | `newton[sim]`, and the above | stepping the same robots with Newton |
| `data` | `pyarrow`, `h5py`, `av`, `huggingface-hub` | LeRobot, RoboMimic and LIBERO datasets |
| `video` | `imageio`, `imageio-ffmpeg`, `pillow` | mp4 recordings (GIF is the fallback) |
| `plots` | `matplotlib`, `pillow` | radars, severity curves, heatmaps |
| `judge` | `anthropic`, `openai` | the optional LLM judge and paraphraser |
| `xwm` | `xwm` | the sibling world-model library |

```bash
pip install 'xevals[mujoco]'               # the manipulation robots
pip install 'xevals[plots,video]'          # figures and videos in the run directory
pip install 'xevals[torch,sim]'            # a torch policy in a Gym environment
pip install 'xevals[dev]'                  # tests and lint
```

## Why the core is this small

An evaluation library gets installed into whatever environment the *model* needs,
and that environment is usually already a minefield of pinned CUDA versions. A
core that pulled in torch would make `xevals` uninstallable next to jax, and one
that pulled in both would be uninstallable next to either.

So the discipline is strict and it is tested: CI runs
`python -c "import xevals, xevals.adapters, xevals.plots, xevals.media"` in an
environment with numpy and nothing else. Every framework import in the library
sits **inside** the function or class that needs it.

## Development

```bash
git clone https://github.com/kamara-lab/xevals
cd xevals
uv venv && uv pip install -e '.[dev,docs]'
pytest -q -n auto
ruff check .
mkdocs serve
```
