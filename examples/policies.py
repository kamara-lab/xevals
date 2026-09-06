"""The policies the examples and configs evaluate.

Two of them, and the difference between them is the point of the whole library.
:func:`state_policy` reads the state vector and is therefore invisible to every
visual perturbation; :func:`image_policy` reads pixels and is not. Run both
through the same suite and the robustness column separates them completely while
the accuracy column does not -- which is the argument for measuring more than one
number, made with two functions rather than a paragraph.

Neither is a serious policy. They are scripted controllers, present so that every
figure, table and video in the documentation comes from a real run rather than
from fabricated numbers.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ArmPolicy", "ImagePolicy", "arm_policy", "image_policy", "state_policy"]

#: RGB of the agent disc and the goal disc as the synthetic world renders them,
#: on a 0-1 scale. The policy finds them by colour, which is what makes it
#: sensitive to photometric perturbation as well as geometric.
_AGENT_RGB = np.array([0.071, 0.078, 0.071], dtype=np.float32)
_GOAL_RGB = np.array([0.863, 0.804, 0.745], dtype=np.float32)


def state_policy(obs: dict) -> np.ndarray:
    """Steer straight at the goal, reading the state vector.

    Solves the task, and is completely unaffected by anything done to the
    pixels -- which shows up as a robustness score of 1.0 against the visual
    families and is a true statement about this policy, not a flattering one.
    """
    state, goal = obs.get("state"), obs.get("goal")
    if state is None or goal is None:  # a sensor-dropout cell removed a field
        return np.zeros(2, dtype=np.float32)
    delta = np.asarray(goal, dtype=np.float32) - np.asarray(state, dtype=np.float32)[:2]
    norm = float(np.linalg.norm(delta))
    return (delta / norm if norm > 1e-6 else delta).astype(np.float32)


class ImagePolicy:
    """Locate the agent and the goal in the frame by colour, and steer between them.

    A stand-in for a pixel-conditioned policy, and it fails in the ways one does:
    occlusion hides the goal, a strong brightness shift moves both colours out of
    tolerance, and an adversarial patch pasted over either disc redirects it. The
    numbers in the documentation are these failures, measured.
    """

    #: How close a pixel must be to the reference colour, in RGB distance. Loose
    #: enough to survive mild noise, tight enough that the two discs never merge.
    agent_tolerance = 0.28
    goal_tolerance = 0.10

    def act(self, obs: dict, *, instruction: str | None = None) -> np.ndarray:
        """One action, or zero when the frame does not show what it needs."""
        frame = obs.get("image")
        if frame is None:
            return np.zeros(2, dtype=np.float32)
        pixels = np.asarray(frame, dtype=np.float32) / 255.0
        agent = self._centroid(pixels, _AGENT_RGB, self.agent_tolerance)
        goal = self._centroid(pixels, _GOAL_RGB, self.goal_tolerance)
        if agent is None or goal is None:
            return np.zeros(2, dtype=np.float32)
        delta = goal - agent
        norm = float(np.linalg.norm(delta))
        return (delta / norm if norm > 1e-6 else delta).astype(np.float32)

    def describe(self) -> dict:
        """What lands in ``run.json`` as the model fingerprint."""
        return {"adapter": "example", "module": "ImagePolicy", "reads": "image"}

    @staticmethod
    def _centroid(pixels: np.ndarray, colour: np.ndarray, tolerance: float):
        """Workspace coordinates of the matching pixels, or ``None`` if there are none."""
        mask = np.linalg.norm(pixels - colour, axis=-1) < tolerance
        if not mask.any():
            return None
        rows, columns = np.nonzero(mask)
        size = pixels.shape[0]
        u = columns.mean() / (size - 1) * 2 - 1
        v = 1 - rows.mean() / (size - 1) * 2
        return np.array([u, v], dtype=np.float32)


def image_policy() -> ImagePolicy:
    """Factory, so a config can name ``examples.policies:image_policy``."""
    return ImagePolicy()


class ArmPolicy:
    """A visual servo for the manipulation robots: find the block, go to it, grasp.

    Where :class:`ImagePolicy` works in the plane of a 2-D world, this one has to
    get from pixels to a place on a table. It masks the block by colour, takes the
    centroid, and back-projects that pixel through the scene camera onto the table
    plane; then it servos the tool there in the four-number action space every arm
    shares, and closes the hand.

    It is allowed to know the camera calibration, which a fixed-camera visual
    servo would, and it takes that from the environment through the ``bind`` hook
    the runner already calls. Everything about *where the block is* still comes
    from the image, which is the part that a visual perturbation breaks: occlusion
    hides the block, brightness moves its colour out of tolerance, and a patch
    pasted over the table gives it somewhere else to go.
    """

    #: How close a pixel's colour must be to the block's, on a 0-1 scale. Loose
    #: enough to survive the shading on a lit cube, tight enough that the table
    #: and the arm never match.
    tolerance = 0.30
    #: How many pixels have to match before the estimate is believed at all.
    minimum_pixels = 4
    #: Frames spent closing the hand before it is treated as holding something.
    close_steps = 8

    def __init__(self) -> None:
        self.env: object | None = None
        self._eye: np.ndarray | None = None
        self._rotation: np.ndarray | None = None
        self._colour = np.array([0.78, 0.24, 0.19], dtype=np.float32)
        self._plane = 0.03
        self._surface = 0.025
        self._step = 0.02
        self._last: np.ndarray | None = None
        self._closing = 0
        self._holding = False

    def reset(self) -> None:
        """Forget the block, and the grasp, between episodes."""
        self._last = None
        self._closing = 0
        self._holding = False

    def bind(self, env: object) -> None:
        """Take the camera pose and the block's colour from the live scene."""
        import mujoco

        from xevals.envs import OBJECT_COLORS

        self.env = env
        model = env.model  # type: ignore[attr-defined]
        camera = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, getattr(env, "camera", "pov")
        )
        # The model's own fields, not the data's: the scene camera is bolted to
        # the world, and ``bind`` runs before the first ``reset`` has put
        # anything in ``data``.
        self._eye = np.array(model.cam_pos[camera], dtype=np.float64, copy=True)
        self._rotation = np.array(
            model.cam_mat0[camera], dtype=np.float64, copy=True
        ).reshape(3, 3)
        self._fovy = float(np.radians(model.cam_fovy[camera]))
        colour = OBJECT_COLORS.get(env.object_name, (198, 62, 48))  # type: ignore[attr-defined]
        self._colour = np.array([c / 255.0 for c in colour], dtype=np.float32)
        # Back-project onto the block's top face, not its middle: most of what
        # the camera sees of a block on a table is its lid, and a centroid
        # unprojected onto the mid-height plane lands short, toward the lens.
        # On this scene that is the difference between a 19 mm error and a
        # 10 mm one, which is the difference between a grasp and a nudge.
        spec = env.spec  # type: ignore[attr-defined]
        self._surface = float(spec.platform_height + spec.object_half[2])
        self._plane = float(spec.platform_height + 2.0 * spec.object_half[2])
        self._step = float(env.spec.step_size)  # type: ignore[attr-defined]

    def _table_point(self, frame: np.ndarray) -> np.ndarray | None:
        """Where on the table the block's pixels say it is, or ``None``."""
        if self._eye is None or self._rotation is None:
            return None
        image = np.asarray(frame, dtype=np.float32)
        if image.ndim != 3 or image.shape[-1] < 3:
            return None
        if image.max() > 1.5:
            image = image / 255.0
        distance = np.linalg.norm(image[..., :3] - self._colour, axis=-1)
        rows, cols = np.nonzero(distance < self.tolerance)
        if rows.size < self.minimum_pixels:
            return None
        height, width = image.shape[0], image.shape[1]
        # MuJoCo's field of view is vertical, and its camera looks down its own
        # -z with +x right and +y up.
        scale = 2.0 * np.tan(self._fovy / 2.0) / height
        ray = np.array(
            [
                (float(cols.mean()) - width / 2.0) * scale,
                -(float(rows.mean()) - height / 2.0) * scale,
                -1.0,
            ]
        )
        direction = self._rotation @ ray
        if direction[2] > -1e-6:  # pointing at the sky: nothing to intersect
            return None
        travel = (self._plane - self._eye[2]) / direction[2]
        return self._eye + travel * direction

    def act(self, obs: dict, *, instruction: str | None = None) -> np.ndarray:
        """Approach the block, descend onto it, close the hand and lift."""
        tool = obs.get("tcp")
        frame = obs.get("image")
        if tool is None or frame is None:
            return np.zeros(4, dtype=np.float32)
        seen = self._table_point(frame)
        if seen is not None:
            # A running mean rather than the latest reading. One centroid off a
            # handful of pixels is noisy, and the difference between a grasp and
            # a nudge here is about a centimetre.
            self._last = seen if self._last is None else 0.6 * self._last + 0.4 * seen
        target = self._last
        if target is None:
            # Nothing recognisable and nothing remembered, so back the tool off
            # and look again: at the start of an episode the commonest reason to
            # see no block is that the hand is in front of it. Rising cannot
            # uncover a block behind an occluding square, which is why this
            # still fails the cells it should.
            return np.array([0.0, 0.0, 0.4, 1.0], dtype=np.float32)

        tool = np.asarray(tool, dtype=np.float64)
        if self._holding:
            # Once the hand has shut on something, keep it shut and go up.
            # Re-deciding every step is how a policy puts the block back down as
            # soon as it has lifted it out of the pose it was reaching for.
            return np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)

        planar = float(np.linalg.norm((target - tool)[:2]))
        hover = self._surface + 0.10
        # Two stops on the way down, and the tolerance tightens at the second.
        # A centimetre of lateral error is the whole margin a 5 cm block leaves
        # in an 8 cm jaw, so the last correction has to happen close in.
        near = tool[2] < self._surface + 0.045
        tolerance = 0.004 if near else 0.008
        if planar > tolerance and tool[2] < hover - 0.01 and not near:
            delta = np.array([0.0, 0.0, hover - tool[2]])
        elif planar > tolerance:
            delta = np.array([target[0] - tool[0], target[1] - tool[1], 0.0])
        elif not near:
            delta = np.array([0.0, 0.0, self._surface + 0.03 - tool[2]])
        elif tool[2] > self._surface + 0.008:
            delta = np.array([0.0, 0.0, self._surface - tool[2]])
        else:
            self._closing += 1
            if self._closing >= self.close_steps:
                self._holding = True
            return np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        move = np.clip(delta / (2.5 * self._step), -1.0, 1.0)
        return np.array([move[0], move[1], move[2], 1.0], dtype=np.float32)

    def describe(self) -> dict:
        return {"adapter": "example", "module": "ArmPolicy", "reads": "image"}


def arm_policy() -> ArmPolicy:
    """The visual servo, as a callable object the runner can drive."""
    return ArmPolicy()
