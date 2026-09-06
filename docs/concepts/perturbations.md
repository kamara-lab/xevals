# Perturbations

A robustness number is a **ratio**, and a ratio only means something if the two
runs differ in exactly one thing. So every perturbation obeys three rules, and
the test suite checks all three on every registered one.

**Deterministic in `(seed, severity)`.** The same frame perturbed twice with the
same seed is the same frame, byte for byte. Otherwise two evaluations of one model
disagree and nobody can tell whether the model moved or the noise did.

**Structure-preserving.** Shape and dtype out match shape and dtype in. A
perturbation that quietly turns uint8 into float64 changes what the model sees in
a second, undeclared way, and the resulting drop gets attributed to the
perturbation that was named.

**A stated severity ladder.** Severity runs over `(0.2, 0.4, 0.6, 0.8, 1.0)` and
means the same thing across families: 0 is untouched, 1.0 is the strongest level
the family defines as still a *nuisance* rather than a destroyed observation.
That is what makes an area-under-severity number comparable between a blur and a
camera shift, and between two models measured months apart.

## The families

| Family | Changes | Members |
|---|---|---|
| `visual/` | the pixels | `gaussian_noise` `blur` `brightness` `contrast` `color_shift` `quantize` `occlusion` `cutout` `camera_shift` `camera_rotate` `crop` `distractor_overlay` |
| `sensor/` | the sensing process | `dropout` `latency` `frame_skip` `resolution` |
| `action/` | the command channel | `noise` `delay` `clip` `dropout` |
| `instruction/` | the language | `paraphrase` `typo` `reorder` `distractor_clause` |
| `dynamics/` | the world | `mass` `friction` `gain` |
| `adversarial/` | a chosen worst case | `patch` `pixel` |
| `injection/` | text placed to hijack | `scene_text` `instruction` `system_override` |

`adversarial/` and `injection/` belong to the **security** dimension; everything
else to **robustness**. See [Dimensions](dimensions/index.md#robustness-and-security-are-not-the-same-dimension)
for why that split is not cosmetic.

```bash
xevals perturbations list
xevals perturbations describe visual/camera_shift
```

## Details worth knowing

**Camera shift holds the edge rather than wrapping.** A wrapped frame contains
scene content in impossible places and measures something no real camera does.

**Paraphrase preserves the object noun.** The rewrites operate on the verb phrase
by regex, so a drop under paraphrase is a failure of language robustness and not
of grounding: two very different findings that a free-form rewrite would
merge.

**Sensor dropout removes the field.** It does not zero it. See
[Conventions](../getting-started/conventions.md#observations-are-dictionaries-and-a-model-takes-what-it-wants).

**The adversarial patch is a random search, not a gradient attack.** That is a
deliberate weakening. A gradient attack needs a differentiable adapter and
therefore torch or jax, which would make the security dimension unmeasurable on
exactly the models most likely to be evaluated as black boxes: remote
endpoints and closed VLAs. So read the result the right way round: **a model that
fails it fails easily; a model that survives it has not been shown to be secure.**

**Dynamics perturbations need `env.set_physics(**kwargs)`.** An environment
without it is not silently skipped; the cell is recorded as skipped *with the
reason*, because "we could not measure dynamics robustness on this simulator" and
"this model is robust to dynamics" must not look the same in a table.

## Composing

```python
from xevals.perturbations import Compose, create

dim_room_moved_camera = Compose([
    create("visual/brightness", severity=0.6),
    create("visual/camera_shift", severity=0.4),
])
```

A composition is its own cell, and worth running as one: the interaction is
usually where a model actually breaks.

::: xevals.perturbations
    options:
      heading_level: 2
      members: false
