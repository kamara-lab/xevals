# CPU evaluation

CPU batching supports both state and image policies, without adding Python
dependencies. State evaluations can avoid rendering entirely. Vision evaluations
still need a working OpenGL renderer. The serial path remains the default. The optional CPU batch path
uses a persistent Python thread pool around MuJoCo's native, GIL-releasing calls;
each physics call advances all substeps of one control interval.

## State observations

```python
import xevals

result = xevals.evaluate(
    policy,
    "mujoco/panda-reach",
    observation_mode="state",
    record=0,
    out=None,
)
```

`observation_mode="state"` removes image observations and avoids rendering during
reset and stepping. `record=0` separately disables video recording. Both are
needed for evaluation without rendering. Existing image observations remain the
default; disabling video alone does not remove a vision policy's camera input.

Visual perturbations, adversarial pixels/patches, and scene-text injections are
unmeasurable in state-only mode. Their metrics carry a reason rather than a
successful score for a perturbation that was never applied. State, action,
dynamics, and instruction perturbations continue to work.

## Batched episodes

```python
policy = xevals.wrap(
    my_torch_module,
    device="cpu",
    feature_keys=("state",),
    action_dim=4,
    batch_mode="stateless",
)
result = xevals.evaluate(
    policy,
    "mujoco/panda-reach",
    observation_mode="state",
    execution="cpu_batch",
    batch_size=8,
    workers=4,
    episodes=32,
    record=0,
    out="runs",
)
```

Batching requires a [stateless batch adapter](wrapping-models.md#independent-batched-inference),
CPU inference, and a built-in MuJoCo manipulation environment. Use an environment
name or a factory constructing a fresh `ManipulationEnv` in either observation mode.
A single live environment cannot supply independent episodes. Newton, arbitrary
Gym environments and stateful adapters use serial execution.
When passing `observation_mode=` alongside a factory, the factory must already
construct that mode; xevals does not silently change caller-owned environments.

Episodes are grouped within one condition. Completed episodes stop stepping;
the last group may be smaller. Physics models, controller state, perturbations,
and episode seeds are isolated. Results retain seed order even when workers
finish in another order. All environments are closed before their group ends.

A raised batch inference call is bisected to isolate invalid observations.
Individual invalid actions or environment failures fail only their episodes.
This recovery relies on the stateless contract. Episode budgets cap admission
exactly; time budgets stop admission of new groups, allowing already admitted
episodes to finish. Reference baselines retain the existing serial path and
separate budget behaviour.

## Image observations

Use `observation_mode="image"` with the same batch settings. For a Torch image
policy, use `dict_input=True, feature_keys=("image",), device="cpu",
batch_mode="stateless"`. The module receives a dictionary containing a batched
uint8 image tensor `(B, H, W, C)`; preprocessing (normalization and channel order)
belongs in its forward method. Include `"state"` in `feature_keys` for a multimodal
policy. Explicit batching requires independent rows and no episode history.
LeRobot action queues and other stateful adapters therefore remain serial.

Physics workers never render. The coordinator produces each image and applies
its observation perturbation exactly once, before inference. Visual robustness
conditions run normally. `record` is supported in either mode; for image policies,
recorded frames reuse the actual perturbed observations. Each episode owns its
physics model and data; the renderer is reused sequentially.

CPU inference does not imply CPU rendering. For GPU-free vision on Linux, set
`MUJOCO_GL=osmesa` **before importing MuJoCo**, with OSMesa already available on
the host. xevals does not install system graphics libraries. EGL and macOS's
default OpenGL path are not a guarantee of GPU-free operation. See
[MuJoCo rendering](https://mujoco.readthedocs.io/en/stable/programming/visualization.html).
Missing camera frames fail image batch episodes rather than silently evaluating
a state-only observation. Software rendering can dominate total runtime;
parallel physics cannot remove that cost.

## Interpreting performance

Batch throughput is not single-action control latency. Batch evaluation leaves
scalar latency, scalar throughput, and control-headroom metrics unmeasured, with
an explanation. Parameter counts remain available. Batch call durations are
stored once per episode group in trajectory metadata; save trajectories to
retain the individual samples on disk.

`result.run["execution"]` always records the mode and execution settings. It also
records evaluation wall time, aggregate inference time, completed control steps,
steps per second, scoring time, and summed environment work durations. Work
durations overlap across workers, so their sum is not elapsed time. Evaluation
wall time excludes baseline runs, scoring, report generation, and saving.

The current implementation keeps full MuJoCo state per active environment to
retain access to contacts and safety measurements. Larger batches consume more
memory. Python inverse kinematics and bookkeeping can dominate once physics
becomes faster; small policies and short episodes may run faster serially.
BLAS/Torch threads can also compete with simulation workers. xevals does not
change process-global thread settings. Benchmark explicit thread budgets on the
target machine; the worker default is at most four and never exceeds batch size.

From a checkout with cached Panda assets:

```bash
python examples/08_cpu_benchmark.py --episodes 32 --horizon 100 --repeats 3 > cpu-benchmark.jsonl
```

The benchmark compares serial execution with batch sizes 1, 8 and 32 and worker
counts 1, 2, 4 and 8 where available, on Panda reach/pick in both state and image
modes. State workloads use a scripted controller and a small MLP; image workloads
use an image-statistics policy and a small untrained CNN. Each configuration uses a fresh process,
one BLAS/Torch inference thread, and a warm-up. JSON records include complete
`evaluate()` wall time, peak process RSS, and component work durations. Baselines,
file output and checkpoint quality are outside this throughput test. Image-mode
measurements include rendering and record the selected GL environment setting.
Use `--observation-modes state` to run only the renderer-free workloads.
The script never downloads robot assets or adds dependencies.

## Local state-only measurements

An exploratory sweep on a 10-logical-core Apple Silicon Mac (macOS 26.6.2,
MuJoCo 3.12.0, Torch 2.14.0) used 32 episodes, a 100-step horizon, and three
repeats per configuration. These are median complete evaluation times, with the
best batch configuration selected after the sweep. Short reaching episodes can
terminate well before the horizon. No episode errors occurred in 120 samples.

| Policy / task | Serial | Best CPU batch | Batch / workers | Speedup | Batch peak RSS |
|---|---:|---:|---:|---:|---:|
| Scripted / reach | 0.243 s | 0.238 s | 8 / 2 | 1.02× | 677 MiB |
| Scripted / pick | 0.986 s | 0.732 s | 8 / 4 | 1.35× | 682 MiB |
| Small Torch / reach | 0.878 s | 0.803 s | 32 / 2 | 1.09× | 1696 MiB |
| Small Torch / pick | 1.696 s | 1.329 s | 32 / 2 | 1.28× | 1698 MiB |

The serial processes peaked at approximately 424–587 MiB. Larger batches trade
memory for throughput, and tiny gains such as 1.02× should not be treated as a
reliable improvement. These local measurements do not predict Linux server
performance or VLA inference speed. [All samples and version metadata](../assets/cpu-benchmark.json)
are available for inspection.

## State and image smoke comparison

A subsequent short comparison used Panda pick, 8 episodes, a 20-step horizon,
batch size 8, two workers, and three repeats on the same Mac. All 24 runs finished
without episode errors. These median complete evaluation times include image
rendering where applicable; macOS default GL does **not** validate GPU-free
rendering. The scripted image policy uses image statistics; the Torch image
policy is an untrained CNN. No policy-quality claim is implied.

| Observation / policy | Serial | CPU batch | Speedup |
|---|---:|---:|---:|
| state / scripted | 0.102 s | 0.127 s | 0.80× |
| state / torch | 0.106 s | 0.120 s | 0.89× |
| image / scripted | 1.226 s | 1.226 s | 1.00× |
| image / torch | 1.266 s | 1.294 s | 0.98× |

These short workloads show no consistent batch advantage. Benchmark realistic
horizons and checkpoints before selecting an execution mode. The older state
sweep above used longer episodes and selected its best configuration afterward;
it should not be compared directly to this fixed-configuration smoke test.
[All state/image samples](../assets/cpu-vision-smoke.json) include memory,
phase timings, environment metadata, and GL selection.
