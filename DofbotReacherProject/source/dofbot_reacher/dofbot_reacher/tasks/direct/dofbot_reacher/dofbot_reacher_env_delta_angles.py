from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from dofbot_reacher.robots import DOFBOT_DOF_LIMITS

from .dofbot_reacher_env_cfg import DofbotReacherEnvCfg


def define_markers() -> VisualizationMarkers:
    """Small spheres used to visualize the goal (red) and the tracked end-effector (cyan)."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/dofbotReacherMarkers",
        markers={
            "end_effector": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "goal": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


# ------------------------------------------------------------------------- #
# Randomization (kept isolated from the env's core loop, as requested)
# ------------------------------------------------------------------------- #
class DofbotRandomizer:
    """Owns every source of randomness for the task: reset noise on the arm
    pose/velocity, sampling of new goal positions, and observation noise.
    Keeping this separate means `_reset_idx` / `_get_observations` stay
    readable and the noise model can be tuned/disabled in one place.
    """

    def __init__(self, env: "DofbotReacherEnv"):
        self.env = env
        self.cfg = env.cfg

    def randomize_dof_on_reset(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a noisy but valid joint position/velocity for the given envs."""
        env = self.env
        num_resets = len(env_ids)

        rand_delta = math_utils.sample_uniform(
            -1.0, 1.0, (num_resets, env.num_arm_dofs), device=env.device
        )
        delta_max = env.arm_dof_upper_limits - env.arm_dof_default_pos
        delta_min = env.arm_dof_lower_limits - env.arm_dof_default_pos
        dof_pos = env.arm_dof_default_pos + self.cfg.reset_dof_pos_noise * (
            delta_min + (delta_max - delta_min) * (rand_delta + 1.0) * 0.5
        )
        dof_pos = dof_pos.clamp(env.arm_dof_lower_limits, env.arm_dof_upper_limits)

        dof_vel = self.cfg.reset_dof_vel_noise * math_utils.sample_uniform(
            -1.0, 1.0, (num_resets, env.num_arm_dofs), device=env.device
        )
        return dof_pos, dof_vel

    def randomize_goal_pose(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample a new goal position inside the arm's reachable workspace."""
        num_resets = len(env_ids)
        new_pos = math_utils.sample_uniform(-1.0, 1.0, (num_resets, 3), device=self.env.device)
        new_pos[:, 0] = new_pos[:, 0] * 0.05 + 0.15 * torch.sign(new_pos[:, 0])
        new_pos[:, 1] = new_pos[:, 1] * 0.05 + 0.15 * torch.sign(new_pos[:, 1])
        new_pos[:, 2] = torch.abs(new_pos[:, 2] * 0.2) + 0.15
        return new_pos

    def apply_observation_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Add (optional) gaussian noise to the observation vector."""
        noise_scale = getattr(self.cfg, "obs_noise_scale", 0.0)
        if noise_scale == 0.0:
            return obs
        return obs + noise_scale * torch.randn_like(obs)


class DofbotReacherEnv(DirectRLEnv):
    cfg: DofbotReacherEnvCfg

    def __init__(self, cfg: DofbotReacherEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        self.num_arm_dofs = len(self.dof_idx)
        self.end_effector_idx, _ = self.robot.find_bodies(self.cfg.end_effector_body_name)
        self.end_effector_idx = self.end_effector_idx[0]

        # Task-specific joint ranges (tighter than the raw hardware limits).
        task_limits = torch.tensor(DOFBOT_DOF_LIMITS, dtype=torch.float32, device=self.device)
        self.arm_dof_lower_limits = task_limits[:, 0]
        self.arm_dof_upper_limits = task_limits[:, 1]
        self.arm_dof_default_pos = torch.zeros(self.num_arm_dofs, device=self.device)

        self.prev_targets = torch.zeros((self.num_envs, self.num_arm_dofs), device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_arm_dofs), device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_arm_dofs), device=self.device)

        # goal bookkeeping
        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_dist = torch.zeros(self.num_envs, device=self.device)
        self._delta_pose = torch.zeros((self.num_envs, 3), device=self.device)
        self._current_dist = torch.zeros(self.num_envs, device=self.device)
        self._goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.randomizer = DofbotRandomizer(self)

    # ------------------------------------------------------------------
    # scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0).to(self.device)

        if self.cfg.use_delta_actions:
            # Phase 2: the policy outputs a *delta* applied to the previous
            # target, not an absolute angle. `delta_scale_rad` bounds how much
            # a single step can move the joint, which keeps the accumulated
            # target from drifting/oscillating. Credit assignment here is
            # harder since a reward at t may depend on deltas chosen several
            # steps earlier.
            delta = self.actions * self.cfg.delta_scale_rad
            self.cur_targets = self.prev_targets + delta
        else:
            # Phase 1: the policy outputs the absolute joint angle directly
            # (denormalized from [-1, 1]). No smoothing is applied so the
            # reward at t is attributable to the action at t with no lag,
            # matching the reference env's own sub-stepped interpolation.
            target = self._unit_to_range(self.actions, self.arm_dof_lower_limits, self.arm_dof_upper_limits)
            self.cur_targets = (
                self.cfg.actions_moving_average * target
                + (1.0 - self.cfg.actions_moving_average) * self.prev_targets
            )

        self.cur_targets = torch.clamp(self.cur_targets, self.arm_dof_lower_limits, self.arm_dof_upper_limits)
        self.prev_targets = self.cur_targets.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.cur_targets, joint_ids=self.dof_idx)

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        arm_dof_pos = self.robot.data.joint_pos[:, self.dof_idx]
        angles_scaled = self._range_to_unit(arm_dof_pos, self.arm_dof_lower_limits, self.arm_dof_upper_limits)

        end_effector_pos = self._get_end_effector_pos()
        delta_pose = self.goal_pos - end_effector_pos

        self._delta_pose = delta_pose
        self._current_dist = torch.norm(delta_pose, p=2, dim=-1)

        self._visualize_markers(end_effector_pos)

        obs = torch.cat((delta_pose, angles_scaled), dim=-1)
        obs = self.randomizer.apply_observation_noise(obs)
        return {"policy": obs}

    def _get_end_effector_pos(self) -> torch.Tensor:
        pos_w = self.robot.data.body_pos_w[:, self.end_effector_idx]
        return pos_w - self.scene.env_origins

    def _visualize_markers(self, end_effector_pos: torch.Tensor) -> None:
        ee_world = end_effector_pos + self.scene.env_origins
        goal_world = self.goal_pos + self.scene.env_origins

        loc = torch.vstack((ee_world, goal_world))
        identity_quat = torch.zeros((2 * self.num_envs, 4), device=self.device)
        identity_quat[:, 0] = 1.0
        all_envs = torch.arange(self.num_envs, device=self.device)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
        self.visualization_markers.visualize(loc, identity_quat, marker_indices=indices)

    # ------------------------------------------------------------------
    # rewards / dones
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        reward, self._goal_reached = compute_rewards(
            self._current_dist,
            self.prev_dist,
            self.actions,
            self.cfg.success_tolerance,
            self.cfg.progress_reward_scale,
            self.cfg.reach_goal_bonus,
            self.cfg.action_penalty_scale,
        )
        self.prev_dist = self._current_dist.clone()
        self.extras["goal_reached"] = self._goal_reached.float().mean()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._goal_reached
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # noisy arm pose/velocity, delegated to the randomizer
        dof_pos, dof_vel = self.randomizer.randomize_dof_on_reset(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:, self.dof_idx] = dof_pos
        joint_vel[:, self.dof_idx] = dof_vel

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(dof_pos, joint_ids=self.dof_idx, env_ids=env_ids)

        # new goal, delegated to the randomizer
        self.goal_pos[env_ids] = self.randomizer.randomize_goal_pose(env_ids)

        # re-initialize the progress-reward baseline distance
        end_effector_pos = self._get_end_effector_pos()
        self.prev_dist[env_ids] = torch.norm(self.goal_pos[env_ids] - end_effector_pos[env_ids], p=2, dim=-1)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _range_to_unit(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """Map values from [lower, upper] to [-1, 1]."""
        return (2.0 * x - upper - lower) / (upper - lower)

    @staticmethod
    def _unit_to_range(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """Map values from [-1, 1] to [lower, upper]."""
        return 0.5 * (x + 1.0) * (upper - lower) + lower


#####################################################################
# jit-friendly reward function
#####################################################################


@torch.jit.script
def compute_rewards(
    dist: torch.Tensor,
    prev_dist: torch.Tensor,
    actions: torch.Tensor,
    success_tolerance: float,
    progress_reward_scale: float,
    reach_goal_bonus: float,
    action_penalty_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """reward = -dist + progress_bonus - action_penalty (+ bonus if goal reached)."""
    reward = -dist
    progress_bonus = (prev_dist - dist) * progress_reward_scale
    reward = reward + progress_bonus

    if action_penalty_scale != 0.0:
        reward = reward - action_penalty_scale * torch.sum(actions**2, dim=-1)

    goal_reached = dist < success_tolerance
    reward = torch.where(goal_reached, reward + reach_goal_bonus, reward)

    return reward, goal_reached
