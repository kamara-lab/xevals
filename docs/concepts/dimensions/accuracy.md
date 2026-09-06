# Accuracy

> **Does it do the task?**

The nominal-condition dimension. Its score is taken from the **clean cell only**,
not averaged over the perturbed ones: averaging them in would make accuracy a
second, worse robustness score, and the two questions would stop being separable.

Nine metrics, because "does it do the task" means something different for a
policy, a world model and a planner. A run measures the ones its model and its
environment can support and reports the rest as `null` with a reason.

| Metric | Direction | Unit | Scored |
|---|---|---|---|
| [`success_rate`](#accuracysuccess_rate) | higher | fraction | yes |
| [`goal_distance`](#accuracygoal_distance) | lower | workspace | yes |
| [`return`](#accuracyreturn) | higher | reward | **no** |
| [`action_mse`](#accuracyaction_mse) | lower | action² | yes |
| [`action_mae`](#accuracyaction_mae) | lower | action | yes |
| [`prediction_error`](#accuracyprediction_error) | lower | latent MSE | yes |
| [`horizon_degradation`](#accuracyhorizon_degradation) | lower | slope | yes |
| [`psnr`](#accuracypsnr) | higher | dB | yes |
| [`plan_correct`](#accuracyplan_correct) | higher | fraction | yes |

---

## `accuracy/success_rate`

**Definition.** The fraction of episodes the *environment* scored as successful.
The verdict is the environment's, never the runner's: `env.success(info)` if the
environment defines it, otherwise `info["success"]`. An environment that defines
neither produces `null`, not zero.

$$
\mathrm{success\_rate} \;=\; \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}\!\left[\,\mathrm{succ}_i\,\right]
$$

**Target.** $1.0$. Used as-is in the dimension score, so a success rate of $0.72$
contributes $0.72$. Read it against the `random` and `noop` baselines rather than
against $1.0$: on a short-horizon task the floor is often surprisingly high.

**Example.** Twenty episodes, thirteen successful:

```
success_rate = 13 / 20 = 0.65   ci [0.45, 0.85]   n = 20
```

The interval is roughly $\pm 10$ points at $n = 20$, which is why it is printed
beside the value and why a $0.65$ against a $0.60$ is not a result.

!!! warning "Unmeasurable offline"
    A replayed dataset does not respond to the model's actions, so whether *this*
    model would have succeeded is unknowable. `xevals` reports `null` with that
    reason rather than publishing the demonstrator's success as the model's.

---

## `accuracy/goal_distance`

**Definition.** The distance from the goal at the final step, averaged over
episodes. Measures *how close* a failure came, which a boolean success rate
discards entirely.

$$
\mathrm{goal\_distance} \;=\; \frac{1}{n}\sum_{i=1}^{n} \bigl\lVert\, x^{(i)}_{T_i} - g^{(i)} \,\bigr\rVert_2
$$

where $x_{T}$ is the final position and $g$ the goal, both in normalised
workspace units.

**Target.** $0$. Lower is better, and the normaliser is $\exp(-d/0.1)$: a
distance of $0.1$, a tenth of the reachable span, scores $0.368$.

**Example.** Three episodes ending $0.02$, $0.05$ and $0.31$ from the goal:

```
goal_distance = (0.02 + 0.05 + 0.31) / 3 = 0.127
score         = exp(-0.127 / 0.1)        = 0.281
```

Two near misses and one that went somewhere else. The success rate would say
$0/3$ for all three; this says which one is a different failure.

---

## `accuracy/return`

**Definition.** The mean undiscounted return over an episode.

$$
G \;=\; \frac{1}{n}\sum_{i=1}^{n}\; \sum_{t=0}^{T_i} r^{(i)}_t
$$

**Target.** Task-specific, and that is exactly the problem.

!!! note "Reported, but not scored"
    A reward scale is the task author's choice. One environment's returns live in
    $[0, 1]$ and another's are unbounded distance penalties, so no single
    normalisation reference is right for both. A metric that moves a dimension
    score by an amount depending on the task's units rather than on the model is
    worse than one that is simply printed, so `return` carries
    `contributes=False`: it appears in every table and in `run.json`, and it does
    not enter the mean.

**Example.** On the built-in environment, whose reward is a distance penalty, a
policy that solves the task in six steps returns about $-1.2$ and one that never
arrives returns about $-42$. Both numbers are informative; neither is comparable
with a return from a different task.

---

## `accuracy/action_mse`

**Definition.** Mean squared error against the demonstrator's action, on a
replayed dataset. The offline half of accuracy.

$$
\mathrm{action\_mse} \;=\; \frac{1}{n}\sum_{i=1}^{n} \frac{1}{T_i}\sum_{t=0}^{T_i-1}
  \bigl\lVert\, a^{(i)}_t - \hat a^{(i)}_t \,\bigr\rVert_2^2 \big/ A
$$

with $\hat a_t$ the recorded action, $a_t$ the model's, and $A$ the action
dimension.

**Target.** $0$. Normaliser $\exp(-e/0.05)$: actions are normalised to $[-1, 1]$,
so an MSE of $0.05$ is about 22 % of full scale and scores $0.368$.

**Example.** A two-dimensional action, demonstrator $(0.8, -0.2)$, model
$(0.6, -0.1)$:

```
squared error = ((0.8-0.6)² + (-0.2-(-0.1))²) / 2 = (0.04 + 0.01) / 2 = 0.025
score         = exp(-0.025 / 0.05)                = 0.607
```

!!! warning "Low action error is not success"
    A policy can match the demonstrator closely and still fail the task, because
    errors compound along a trajectory the demonstrator never visited. `xevals`
    reports action error *beside* success rate, never instead of it.

---

## `accuracy/action_mae`

**Definition.** Mean absolute error against the demonstrator. Less sensitive to a
single large deviation than the squared version, so the two disagree exactly when
one step went badly wrong.

$$
\mathrm{action\_mae} \;=\; \frac{1}{n}\sum_{i=1}^{n} \frac{1}{T_i}\sum_{t=0}^{T_i-1}
  \frac{1}{A}\sum_{k=1}^{A} \bigl\lvert\, a^{(i)}_{t,k} - \hat a^{(i)}_{t,k} \,\bigr\rvert
$$

**Target.** $0$. Normaliser $\exp(-e/0.15)$, which is $0.15$ of a $[-1, 1]$ range.

**Example.** Ten steps matching within $0.05$ and one off by $1.4$:

```
mae = (10 × 0.05 + 1.4) / 11 = 0.173
mse = (10 × 0.0025 + 1.96) / 11 = 0.180
```

The MAE barely moves; the MSE is dominated by the single bad step. Reading both
tells you whether the error is spread or concentrated.

---

## `accuracy/prediction_error`

**Definition.** A world model's error at the evaluated horizon, in the model's
own latent space. Requires a `WorldModel`, not a `Policy`.

$$
\mathrm{prediction\_error} \;=\; \frac{1}{n}\sum_{i=1}^{n}
  \frac{1}{H}\sum_{h=1}^{H} \bigl\lVert\, \hat z^{(i)}_{t+h} - z^{(i)}_{t+h} \,\bigr\rVert_2^2
$$

**Target.** $0$. Normaliser $\exp(-e/0.05)$, the latent MSE at which a rollout
stops being usable for planning.

**Example.** Predicting five steps ahead with per-step latent errors
$0.01, 0.02, 0.04, 0.07, 0.11$:

```
prediction_error = 0.05
score            = exp(-0.05 / 0.05) = 0.368
```

---

## `accuracy/horizon_degradation`

**Definition.** How fast a world model's error grows with horizon, as the slope
of a straight line fitted to error against step.

$$
\hat\beta \;=\; \arg\min_{\beta,\,\alpha} \sum_{h=1}^{H}\bigl(e_h - (\alpha + \beta h)\bigr)^2
$$

**Target.** $0$, meaning error that does not compound. Normaliser
$\exp(-\beta/0.01)$: a growth of $0.01$ per step doubles a hundred-step rollout's
error.

**Example.** The errors above, $0.01$ through $0.11$ over five steps, fit a slope
of about $0.025$ per step. Reported *separately* from the error itself because
they answer different questions: a model with high error and a flat slope is
uniformly imprecise and still usable for long-horizon planning, and a model with
low error and a steep slope is not. A single-horizon number cannot tell them
apart.

---

## `accuracy/psnr`

**Definition.** Peak signal-to-noise ratio of a world model's predicted frames,
for models that predict pixels.

$$
\mathrm{PSNR} \;=\; 10 \log_{10}\!\left( \frac{\mathrm{MAX}^2}{\mathrm{MSE}} \right)
$$

**Target.** Higher; $30$ dB is the usual "good" line and the normaliser's
reference.

**Example.** A predicted frame with pixel MSE $0.001$ on a $[0, 1]$ scale gives
$10\log_{10}(1/0.001) = 30$ dB.

!!! note "The weaker of the two prediction numbers"
    PSNR rewards blur. A model that predicts the scene mean scores respectably and
    has predicted nothing. It is reported beside the latent error rather than
    instead of it, and never shown alone.

---

## `accuracy/plan_correct`

**Definition.** The fraction of plans a [judge](../../guides/metrics.md) accepted.
Planner accuracy.

$$
\mathrm{plan\_correct} \;=\; \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}\!\left[\,J(p_i) \,\right]
$$

with $J$ the judge's verdict. The default judges are rule-based, deliberately: a
model-based judge that drifts between provider versions makes last quarter's
numbers incomparable with this quarter's, invisibly, in the results file.

**Target.** $1.0$, used as-is.

**Example.** A `StepJudge` requiring `approach`, `grasp`, `release` *in order*.
For four plans, three of which have all three steps in order:

```
plan_correct = 3 / 4 = 0.75
```

Order is checked, not just presence: "place the cube, then pick it up" contains
every required step and is not a plan.
