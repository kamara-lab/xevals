"""Native Isaac Lab scene and Newton physics. Imports are deferred until use.

Targets the upstream revision recorded in config.py (ProxyArray data and xyzw
quaternions). No MuJoCo stepping, rendering, or state mirroring occurs here.
"""

from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from xevals.environments.envs import OBJECT_COLORS
from xevals.environments.robots import ROBOTS

from .config import IsaacLabConfig


def _numpy(value):
    value = getattr(value, "torch", value)
    return value.detach().cpu().numpy().copy()


def rotation_xyzw(q):
    """Rotation matrix for Isaac Lab 3's xyzw quaternion convention."""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid quaternion returned by Isaac Lab")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def ik_delta(jacobian, position_error, axis, damping=0.12):
    """Damped least squares holding the approach axis down, with free yaw."""
    jac = np.array(jacobian, dtype=np.float64, copy=True)
    jac[3:] *= 0.5
    error = np.concatenate((position_error, 0.5 * np.cross(axis, [0.0, 0.0, -1.0])))
    return 0.55 * jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), error)


def _scene_config(config, object_name, object_shape):
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg, ContactSensorCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_assets.robots.franka import FRANKA_PANDA_MENAGERIE_CFG

    spec = ROBOTS["panda"]
    robot = FRANKA_PANDA_MENAGERIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    if config.robot_usd:
        robot.spawn.usd_path = config.robot_usd
    robot.spawn.activate_contact_sensors = True
    robot.init_state.joint_pos = {
        **{f"panda_joint{i + 1}": q for i, q in enumerate(spec.home)},
        "panda_finger_joint.*": 0.04,
    }
    material = sim.RigidBodyMaterialCfg(
        static_friction=config.friction,
        dynamic_friction=config.friction,
        restitution=0.0,
    )
    shape_kwargs = dict(
        rigid_props=sim.RigidBodyPropertiesCfg(),
        mass_props=sim.MassPropertiesCfg(mass=config.object_mass),
        collision_props=sim.CollisionPropertiesCfg(),
        physics_material=material,
        visual_material=sim.PreviewSurfaceCfg(
            diffuse_color=tuple(c / 255.0 for c in OBJECT_COLORS[object_name]),
        ),
    )
    if object_shape == "cylinder":
        object_spawn = sim.CylinderCfg(radius=0.025, height=0.05, **shape_kwargs)
    else:
        object_spawn = sim.CuboidCfg(size=(0.05, 0.05, 0.05), **shape_kwargs)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        pass

    scene = SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    scene.robot = robot
    scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=object_spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.46, 0.0, 0.025)),
    )
    scene.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim.CuboidCfg(
            size=(1.36, 1.36, 0.05),
            collision_props=sim.CollisionPropertiesCfg(),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.78, 0.76, 0.72)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)),
    )
    scene.light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim.DomeLightCfg(intensity=2500.0),
    )
    # The converted Menagerie asset uses this nested hand path. An alternate USD
    # must retain its joint/body names and hierarchy, not just look like a Panda.
    hand_path = (
        "{ENV_REGEX_NS}/Robot/Geometry/panda_link0/panda_link1/panda_link2/"
        "panda_link3/panda_link4/panda_link5/panda_link6/panda_link7/panda_hand"
    )
    for side in ("left", "right"):
        setattr(
            scene,
            f"{side}_contact",
            ContactSensorCfg(
                prim_path=f"{hand_path}/panda_{side}finger",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                update_period=0.0,
            ),
        )
    if config.renderer != "none":
        if config.renderer == "newton":
            from isaaclab_newton.renderers import NewtonWarpRendererCfg

            renderer = NewtonWarpRendererCfg(enable_shadows=True)
        else:
            from isaaclab_physx.renderers import IsaacRtxRendererCfg

            renderer = IsaacRtxRendererCfg()
        scene.camera = CameraCfg(
            prim_path=(
                f"{hand_path}/Camera" if config.camera == "wrist" else "{ENV_REGEX_NS}/Camera"
            ),
            height=config.image_size,
            width=config.image_size,
            data_types=["rgb"],
            update_period=0.0,
            renderer_cfg=renderer,
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.075) if config.camera == "wrist" else (0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                convention="ros",
            ),
            spawn=sim.PinholeCameraCfg(
                focal_length=24.0,
                horizontal_aperture=48.0
                * math.tan(
                    math.radians(
                        70.0
                        if config.camera == "wrist"
                        else 48.0
                        if config.camera == "scene"
                        else 52.0
                    )
                    / 2.0
                ),
                clipping_range=(0.01, 10.0),
            ),
        )
    return scene


class IsaacLabRuntime:
    """Owns one native scene. The application, when needed for RTX, is caller-owned."""

    def __init__(self, config: IsaacLabConfig, object_name: str, object_shape: str):
        try:
            import torch
            from isaaclab.scene import InteractiveScene
            from isaaclab.sim import SimulationCfg, SimulationContext
            from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
        except ImportError as exc:
            raise RuntimeError(
                "Isaac Lab Newton is unavailable. Install the pinned Isaac Lab checkout "
                "on Ubuntu with its Newton dependencies; see docs/guides/isaaclab.md."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Isaac Lab Newton evaluation requires an available NVIDIA CUDA GPU")
        if SimulationContext.instance() is not None:
            raise RuntimeError(
                "An Isaac Lab simulation is already active; close it before creating one"
            )
        self.config = config
        self.torch = torch
        self._closed = False
        self.sim = SimulationContext(
            SimulationCfg(
                device=config.device,
                dt=config.physics_dt,
                render_interval=config.decimation,
                physics=NewtonCfg(
                    solver_cfg=MJWarpSolverCfg(
                        solver="newton",
                        integrator="implicitfast",
                        cone="elliptic",
                        impratio=10.0,
                        iterations=config.iterations,
                        ls_iterations=config.ls_iterations,
                        tolerance=config.tolerance,
                        nconmax=config.nconmax,
                        njmax=config.njmax,
                    ),
                    num_substeps=config.substeps,
                ),
            )
        )
        try:
            self.scene = InteractiveScene(_scene_config(config, object_name, object_shape))
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            self._goal_marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/World/GoalMarker",
                    markers={
                        "goal": sim_utils.CylinderCfg(
                            radius=0.045,
                            height=0.002,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.20, 0.55, 0.35)
                            ),
                        )
                    },
                )
            )
            self.robot = self.scene["robot"]
            self.object = self.scene["object"]
            self.sim.reset()
            self.scene.update(config.physics_dt)
            self.arm_ids, _ = self.robot.find_joints(
                [f"panda_joint{i}" for i in range(1, 8)],
                preserve_order=True,
            )
            self.finger_ids, _ = self.robot.find_joints(
                ["panda_finger_joint1", "panda_finger_joint2"],
                preserve_order=True,
            )
            hand_ids, _ = self.robot.find_bodies("panda_hand")
            if len(self.arm_ids) != 7 or len(self.finger_ids) != 2 or len(hand_ids) != 1:
                raise RuntimeError(
                    "The Panda USD must expose seven arm joints, two fingers and a hand"
                )
            self.hand_id = hand_ids[0]
            self.arm_limits = _numpy(self.robot.data.joint_pos_limits)[0, self.arm_ids]
            self.origin = np.asarray(self.scene.env_origins[0].cpu()).copy()
            self.camera = self.scene["camera"] if config.renderer != "none" else None
            if self.camera is not None and config.camera != "wrist":
                reach = 0.46
                if config.camera == "pov":
                    eye = np.array([reach, 0.0, 0.0]) + reach * np.array(ROBOTS["panda"].pov_offset)
                    target = np.array([reach, 0.0, 0.0]) + reach * np.array(ROBOTS["panda"].pov_aim)
                else:
                    target = np.array([reach * 0.55, 0.0, 0.18])
                    eye = target + 1.6 * np.array([0.56, -0.64, 0.40])
                self.camera.set_world_poses_from_view(
                    eyes=self._tensor([eye + self.origin]),
                    targets=self._tensor([target + self.origin]),
                )
            self.provenance = config.describe()
            for package in ("isaaclab", "newton", "mujoco-warp", "warp-lang"):
                try:
                    self.provenance[package] = version(package)
                except PackageNotFoundError:
                    self.provenance[package] = "unknown"
            self.provenance["robot_usd"] = self.robot.cfg.spawn.usd_path
        except BaseException:
            self.close()
            raise

    def _tensor(self, value):
        return self.torch.as_tensor(
            np.asarray(value), dtype=self.torch.float32, device=self.config.device
        )

    def read(self):
        pose = _numpy(self.robot.data.body_link_pose_w)[0, self.hand_id]
        rotation = rotation_xyzw(pose[3:7])
        offset = rotation @ np.array([0.0, 0.0, 0.1034])
        velocity = _numpy(self.robot.data.body_link_vel_w)[0, self.hand_id]
        object_pose = _numpy(self.object.data.root_link_pose_w)[0]
        object_pose[:3] -= self.origin
        forces = []
        for side in ("left", "right"):
            matrix = self.scene[f"{side}_contact"].data.force_matrix_w
            if matrix is None:
                raise RuntimeError(
                    "Newton did not provide the requested finger/object contact forces"
                )
            forces.append(float(np.linalg.norm(_numpy(matrix)[0].sum(axis=(0, 1)))))
        return {
            "joint_pos": _numpy(self.robot.data.joint_pos)[0],
            "joint_vel": _numpy(self.robot.data.joint_vel)[0],
            "object_pose": object_pose,
            "object_velocity": _numpy(self.object.data.root_link_vel_w)[0],
            "tcp": pose[:3] + offset - self.origin,
            "tcp_axis": rotation[:, 2],
            "tcp_offset": offset,
            "tcp_velocity": velocity[:3] + np.cross(velocity[3:], offset),
            "finger_forces": np.asarray(forces),
        }

    def set_goal(self, goal):
        position = np.array(goal, copy=True)
        position[2] = 0.002
        self._goal_marker.visualize(translations=np.asarray([position + self.origin]))

    def reset(self, object_pose):
        q = _numpy(self.robot.data.default_joint_pos)[0]
        q[self.arm_ids] = ROBOTS["panda"].home
        q[self.finger_ids] = 0.04
        self.restore(q, np.zeros_like(q), object_pose, np.zeros(6))

    def restore(self, q, qd, object_pose, object_velocity):
        pose = np.array(object_pose, copy=True)
        pose[:3] += self.origin
        self.scene.reset()
        self.robot.write_joint_position_to_sim_index(position=self._tensor([q]))
        self.robot.write_joint_velocity_to_sim_index(velocity=self._tensor([qd]))
        self.robot.set_joint_position_target_index(target=self._tensor([q]))
        self.object.write_root_pose_to_sim_index(root_pose=self._tensor([pose]))
        self.object.write_root_velocity_to_sim_index(root_velocity=self._tensor([object_velocity]))
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(self.config.physics_dt)

    def advance(self, target, opening):
        # One controller update per policy command. Re-solving around each
        # intermediate measured pose makes the target chase tracking error and
        # caused approach oscillations in the GPU grasp regression.
        state = self.read()
        # The Jacobian is about the link origin. Translate its linear part
        # to the tool centre before solving, keeping angular rows unchanged.
        jac_idx = self.hand_id - 1 if self.robot.is_fixed_base else self.hand_id
        columns = [i + self.robot.num_base_dofs for i in self.arm_ids]
        jac = _numpy(self.robot.data.body_link_jacobian_w)[0, jac_idx][:, columns]
        jac[:3] += np.cross(jac[3:].T, state["tcp_offset"]).T
        q = state["joint_pos"].copy()
        q[self.arm_ids] = np.clip(
            q[self.arm_ids] + ik_delta(jac, target - state["tcp"], state["tcp_axis"]),
            self.arm_limits[:, 0],
            self.arm_limits[:, 1],
        )
        q[self.finger_ids] = 0.04 * opening
        self.robot.set_joint_position_target_index(target=self._tensor([q]))
        for _ in range(self.config.decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.config.physics_dt)

    def render(self):
        if self.camera is None:
            return None
        if self.config.camera == "scene-wrist":
            # Render both views without advancing physics between them.
            self.camera.set_world_poses_from_view(
                eyes=self._tensor([self.origin + [0.86, -0.58, 0.58]]),
                targets=self._tensor([self.origin + [0.45, 0.0, 0.09]]),
            )
            main = self._camera_frame()
        if self.config.camera in ("wrist", "scene-wrist"):
            # Newton advances the articulation independently of the USD parent
            # transform. Drive the camera from the measured hand pose as well.
            hand = _numpy(self.robot.data.body_link_pose_w)[0, self.hand_id]
            position = hand[:3] + rotation_xyzw(hand[3:7]) @ np.array([0.0, 0.0, 0.075])
            self.camera.set_world_poses(
                positions=self._tensor([position]),
                orientations=self._tensor([hand[3:7]]),
                convention="ros",
            )
        frame = self._camera_frame()
        if self.config.camera == "scene-wrist":
            # A one-third-size inset keeps the manipulation workspace visible.
            size = frame.shape[0] // 3
            inset = frame[: size * 3, : size * 3].reshape(size, 3, size, 3, 3)
            inset = inset.mean(axis=(1, 3)).astype(np.uint8)
            margin = max(4, frame.shape[0] // 64)
            top, right = margin, main.shape[1] - margin
            main[top - 2 : top + size + 2, right - size - 2 : right + 2] = 255
            main[top : top + size, right - size : right] = inset
            return main
        return frame

    def _camera_frame(self):
        self.sim.render()
        self.camera.update(0.0, force_recompute=True)
        return _numpy(self.camera.data.output["rgb"])[0, ..., :3].astype(np.uint8)

    def close(self):
        if not self._closed:
            self._closed = True
            self.sim.clear_instance()
