"""Optional native Isaac Lab + Newton Panda tasks.

Importing or listing these tasks does not import Isaac Lab, Newton or Torch.
"""

from functools import partial

from xevals.environments.manipulation import TASKS

from .config import ISAACLAB_REVISION, IsaacLabConfig
from .env import IsaacLabEnv

__all__ = ["ISAACLAB_REVISION", "IsaacLabConfig", "IsaacLabEnv", "register"]


def register():
    from xevals.environments.envs import ENVS

    for task, spec in TASKS.items():
        ENVS.register(
            f"isaaclab-newton/panda-{task}",
            partial(IsaacLabEnv, task),
            summary=f"Panda {task} in Isaac Lab, with Newton/MJWarp and a free object",
            requires=("isaaclab", "isaaclab_newton"),
            robot="panda",
            task=task,
            backend="isaaclab-newton",
            solver="mujoco_warp",
            horizon=spec.horizon,
        )
