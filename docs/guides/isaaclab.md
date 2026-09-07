# Isaac Lab with Newton physics

The native port provides Panda **reach, push, pick, and place** in an Isaac Lab
scene stepped by Newton's MuJoCo Warp solver. The object is a free rigid body;
no physics state is mirrored into MuJoCo. This is separate from the legacy
`newton/panda-reach` environment, which keeps its existing behaviour.

## Runtime

Use Ubuntu with an NVIDIA CUDA GPU and Python 3.12. Install Isaac Lab at the
revision used for this implementation, together with its locked dependencies:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 18086270eecac181e4aad1ae323bde3b9c863072
uv sync --frozen --no-dev
uv pip install --python .venv/bin/python -e "/path/to/xevals[video]"
```

Isaac Lab owns the Newton, Warp, CUDA Torch and USD dependency versions. Install
xevals into that environment, rather than adding arbitrary Isaac packages to a
MuJoCo development environment. The pinned checkout's `uv.lock` includes the
ARM64 CUDA packages needed on a DGX Spark.

Newton integration is an upstream beta. The port targets the pinned revision's
`ProxyArray` data API and **xyzw** quaternions. Older Isaac Lab 2.x APIs are not
interchangeable with this implementation.

## Run the demonstrator gate

Use the Isaac Lab environment's Python interpreter:

```bash
python -m xevals.environments.isaaclab \
  --task reach --renderer none --episodes 2 --validate

python -m xevals.environments.isaaclab \
  --task pick --renderer newton --camera pov --image-size 512 --episodes 3 --validate
```

Repeat for `push` and `place`. The command checks seeded resets, kinematic
snapshot restoration, a free object's response to gravity, camera output when
enabled, and the reference controller's success. A failed gate exits nonzero;
a scene's existence is not evidence that its grasp dynamics are correct.

To evaluate uniform random actions instead of the demonstrator, use
`--policy random --episodes 3` (omit `--validate`, whose success gate is specific
to the demonstrator). The policy draws all four action components uniformly
from `[-1, 1]` and records its seed in the run metadata. The baseline resets its
action generator each episode, while episode initial states vary by seed.

Rendered runs save an episode MP4 under
`runs/isaaclab-newton/<run-id>/videos/clean/000.mp4` and an HTML report beside it.
Videos include the terminal frame and play at the configured control rate
(20 fps by default). The launcher records every episode, including failures,
as `000.mp4`, `001.mp4`, and so on. Its default camera is `scene-wrist` at
768 pixels. `--renderer none` disables images and videos.

`--renderer newton` uses Newton's camera renderer without Isaac Sim. For RTX,
install the Isaac Sim version compatible with the pinned Isaac Lab checkout
and use `--renderer rtx`; the launcher creates the required application before
constructing the scene. Cameras may be `pov`, `wrist`, or `scene`. The Newton
renderer is not a claim of RTX photorealism.

For a moving gripper point of view, use `--camera wrist --image-size 512`.
The camera sits 75 mm forward of the hand frame, clear of the palm housing,
and follows the measured hand pose on every rendered frame.
Use `--camera scene-wrist --image-size 768` for a closer external view with a
synchronized gripper-camera inset in the top-right corner. Both views are
rendered at the same physics state.

## Evaluate a policy

```python
import xevals
from xevals.environments.isaaclab import IsaacLabEnv

with IsaacLabEnv("reach", renderer="newton", camera="pov") as env:
    result = xevals.evaluate(
        policy, env, suite="smoke", episodes=5,
        record=5,
        horizon=env.horizon, control_hz=env.frame_rate,
        out="runs/isaaclab",
    )
```

The caller owns a live scene and closes it after evaluation. For suites that
change generalisation splits, pass a factory instead:

```python
def environment(task=None, split="in"):
    return IsaacLabEnv(task or "pick", split=split, renderer="newton")

result = xevals.evaluate(policy, environment, suite="full", control_hz=20.0)
```

The Python `evaluate` API retains its explicit recording budget: `record` is
the number of episodes saved per cell. To record every episode, set it to
`episodes * len(seeds)`. This is independent of the simulator backend; existing
MuJoCo runs do not automatically become Isaac Lab runs.

Factories currently construct and close a scene per episode, which is slower
than reusing a live scene. Only one Isaac Lab simulation context may exist in a
process. The registry names `isaaclab-newton/panda-reach`,
`isaaclab-newton/panda-push`, `isaaclab-newton/panda-pick`, and
`isaaclab-newton/panda-place` are also available. Listing them needs no simulator.

Actions are `[dx, dy, dz, grip]` in `[-1, 1]`, with positive grip opening the
hand. Observations contain the original joint-state/tool/object/goal vector,
`tcp`, `goal`, `instruction`, and `image` when rendering is enabled. The same
four task specifications and scripted controller are shared with MuJoCo.

## Accuracy and coverage

DGX Spark validation (NVIDIA GB10, Ubuntu ARM64, pinned runtime above): reach,
push, pick and place each passed three clean episodes with seed 0 and the
Newton renderer at 512 pixels. Reset, snapshot, nonblank camera and free-object
gravity checks also passed. These are scripted-controller smoke tests, not a
robustness benchmark or evidence of calibrated real-world accuracy. RTX has
not been validated. The place controller establishes lift clearance before
transporting the object to the marker.

The initial solver profile uses `implicitfast`, an elliptic friction cone,
240 Hz physics with two solver substeps, and 20 Hz policy control. These are
explicit starting values, not experimentally calibrated robot parameters.
`IsaacLabConfig` exposes timestep, solver convergence/capacity, mass and friction.
Saved environment metadata records the profile, asset path, and package versions.

The Panda uses Isaac Lab's Menagerie-derived USD asset. Alternate USD paths must
preserve its articulation names and hierarchy. The other xevals robots are not
yet exposed under this backend.

Finger/object contact forces support the reference controller's grasp detector.
Whole-arm collision classification and maximum contact force are not measured,
so this port does not publish synthetic zeros for them. Runtime dynamics
perturbations and scene-text attacks are not implemented. The single-environment
NumPy interface is intended for correctness testing; a batched GPU evaluator is
future work.

Snapshots include joint positions/velocities, object pose/velocity, controller
targets, goal, episode counter and lift history. They exclude solver warm-start
and contact caches, so restoring a snapshot does not promise bitwise-identical
future trajectories. Cross-backend matching is a useful port check; comparison
with real recordings is required to establish physical accuracy.

See upstream [Newton installation](https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/physical-backends/newton/installation.html)
and [MJWarp tuning](https://isaac-sim.github.io/IsaacLab/develop/source/overview/core-concepts/physical-backends/newton/mjwarp-solver.html).
