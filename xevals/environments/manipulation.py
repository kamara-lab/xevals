"""Shared manipulation task definitions and scripted reference controller.

This module has no simulator dependencies. Both MuJoCo and Isaac Lab use the
same thresholds, reward interpretation and demonstrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    """One thing to do with the object on the table."""

    name: str
    summary: str
    horizon: int
    tolerance: float
    #: Whether the task needs a hand that closes. ``reach`` and ``push`` do not.
    needs_gripper: bool = False
    #: How far above its start the object counts as picked up, in metres.
    lift: float = 0.0
    verb: str = "move to"
    paraphrase: str = "get yourself over to"


TASKS: dict[str, TaskSpec] = {
    "reach": TaskSpec(
        name="reach",
        summary="bring the tool to the object without touching it",
        horizon=60,
        tolerance=0.04,
        verb="move the gripper to",
        paraphrase="bring the tool over to whichever thing is",
    ),
    "push": TaskSpec(
        name="push",
        summary="push the object across the table to the marker",
        horizon=150,
        tolerance=0.06,
        verb="push",
        paraphrase="shove whichever thing is",
    ),
    "pick": TaskSpec(
        name="pick",
        summary="grasp the object and lift it off the table",
        horizon=110,
        tolerance=0.03,
        needs_gripper=True,
        lift=0.09,
        verb="pick up",
        paraphrase="lift up whichever thing is",
    ),
    "place": TaskSpec(
        name="place",
        summary="pick the object up and set it down on the marker",
        horizon=200,
        tolerance=0.06,
        needs_gripper=True,
        lift=0.07,
        verb="put",
        paraphrase="set down whichever thing is",
    ),
}


class ManipulationDemonstrator:
    """Reference actions using the implementing environment's measured geometry."""

    @property
    def optimal_action(self) -> np.ndarray:
        """A scripted demonstrator, in the same action space a policy uses.

        This exists so the replay gate means something: if these actions do not
        solve the task, the environment is broken and no model score from it is
        worth reading. It infers its phase from the measured tool/object state
        and the environment's lift history, which is included in snapshots.
        """
        tcp = self._tool_position()
        obj = self._object_position()
        grip_open = 1.0
        grip_close = -1.0
        hover = obj + np.array([0.0, 0.0, self._hover_height()])
        grasp = obj + np.array([0.0, 0.0, self.spec.grasp_height])

        if self.task == "reach":
            return self._toward(tcp, obj + np.array([0.0, 0.0, 0.02]), grip_open)
        if self.task == "push":
            return self._push_action(tcp, obj)

        holding = self._holding()
        if not holding:
            planar = float(np.linalg.norm((hover - tcp)[:2]))
            # Once the tool is down among the jaws' working height, going back
            # up to re-approach would knock the object over. Correct sideways at
            # this height instead, and only then close.
            low = tcp[2] < grasp[2] + 0.05
            if planar > (0.02 if low else 0.012):
                target = np.array([hover[0], hover[1], tcp[2] if low else max(tcp[2], hover[2])])
                return self._toward(tcp, target, grip_open)
            if abs(tcp[2] - grasp[2]) > 0.006:
                return self._toward(tcp, np.array([grasp[0], grasp[1], grasp[2]]), grip_open)
            return np.array([0.0, 0.0, 0.0, grip_close], dtype=np.float32)

        if self.task == "pick":
            up = self._object_start + np.array([0.0, 0.0, self.lift_height + 0.03])
            return self._toward(tcp, tcp + (up - obj), grip_close)
        # place: carry the object over the marker, set it down, then let go.
        over = np.array([self._goal[0], self._goal[1], self._object_start[2] + 0.12])
        # Establish table clearance before moving sideways. A nearby marker
        # can otherwise trigger descent before the required lift ever occurs.
        if not self._lifted:
            return self._toward(tcp, tcp + np.array([0.0, 0.0, over[2] - obj[2]]), grip_close)
        planar = float(np.linalg.norm((over - obj)[:2]))
        if planar > 0.02:
            return self._toward(tcp, tcp + (over - obj), grip_close)
        if obj[2] > self._rest_height() + 0.012:
            return self._toward(tcp, tcp - np.array([0.0, 0.0, 0.05]), grip_close)
        return np.array([0.0, 0.0, 0.3, grip_open], dtype=np.float32)

    # -- demonstrator helpers ---------------------------------------------

    def _toward(self, tcp: np.ndarray, target: np.ndarray, grip: float) -> np.ndarray:
        """A unit-ish move toward a tool target, saturating far from it."""
        delta = target - tcp
        # Divided by more than one step, so the command eases into the target
        # rather than commanding a full step at it: a servo that is still moving
        # would otherwise be told to reverse every frame, and the tool rings
        # around the object instead of arriving at it.
        move = np.clip(delta / (2.5 * max(self.spec.step_size, 1e-6)), -1.0, 1.0)
        move[np.abs(delta) < 0.003] = 0.0
        return np.array([move[0], move[1], move[2], grip], dtype=np.float32)

    def _push_action(self, tcp: np.ndarray, obj: np.ndarray) -> np.ndarray:
        """Get behind the object relative to the marker, then drive through it.

        The tool travels between those two places over the top of the object.
        Cutting the corner at working height is what turns a push into a swipe:
        the object leaves on a line nobody chose and the episode is over.
        """
        closed = -1.0
        goal = np.array([self._goal[0], self._goal[1], obj[2]])
        to_goal = goal - obj
        norm = float(np.linalg.norm(to_goal[:2]))
        direction = (to_goal / norm) if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        push_z = self._rest_height()
        stand_off = self.spec.object_half[0] + 0.035
        behind = obj - direction * stand_off
        behind[2] = push_z
        transit = push_z + self._hover_height()

        offset = tcp[:2] - obj[:2]
        along = float(offset @ direction[:2])
        lateral = float(np.linalg.norm(offset - along * direction[:2]))
        low = tcp[2] < push_z + 0.05
        in_place = along < -self.spec.object_half[0] * 0.5 and lateral < 0.03

        if low and in_place:
            return self._toward(tcp, goal + direction * (stand_off + 0.01), closed)
        planar = float(np.linalg.norm((behind - tcp)[:2]))
        # Once the tool is over the pre-push point, commit to descending. Asking
        # it to hold transit height first would trade the descent for a hover.
        if planar < 0.02 or (low and planar < 0.05):
            return self._toward(tcp, behind, closed)
        if tcp[2] < transit - 0.01:
            return self._toward(tcp, np.array([tcp[0], tcp[1], transit]), closed)
        return self._toward(tcp, np.array([behind[0], behind[1], transit]), closed)
