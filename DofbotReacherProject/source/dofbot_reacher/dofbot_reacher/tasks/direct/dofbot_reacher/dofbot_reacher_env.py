# Copyright (c) 2022-2023, Johnson Sun
# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct RL environment: teach a Yahboom Dofbot (5-DoF) arm to reach random
3D target positions with its end effector.

This is an Isaac Lab port of the ``ReachDofbotEnvGym`` gymnasium environment
used for sim2real experiments on the physical Dofbot. The action space,
observation space and reward function are kept as close as possible to that
reference implementation:

- **Action space**: ``Box(-1, 1, shape=(4,))``, one normalized value per
  actuated joint (``joint1`` .. ``joint4`` = ``theta``/``alpha``/``beta``/``gamma``
  in the reference env). Denormalized to each joint's own angular range
  before being sent to the robot as a position target (the reference env
  denormalizes to a fixed +-90deg range and linearly interpolates the
  motion in software; here Isaac Lab's PD joint drive does the equivalent
  job of moving smoothly toward the target every physics substep). The
  5th joint (wrist roll, ``joint5``) is *not* actuated: it only changes the
  end effector's orientation, not its position, so it is held fixed
  instead of wasting an action dimension on it -- exactly like the
  reference environment, whose ``forward_k()`` only takes 4 angles.
- **Observation space**: ``Box(-inf, inf, shape=(7,))`` = ``delta_pose``
  (goal - end-effector position, 3 values) concatenated with the normalized
  joint ``angles`` of the 4 actuated joints -- the flattened equivalent of
  the reference env's ``{"delta_pose": ..., "angles": ...}`` observation dict.
- **Reward**: ``reward_function()`` below is a direct, vectorized port of
  ``ReachDofbotEnvGym.reward_function``: ``-distance`` plus a "progress"
  bonus of ``(prev_dist - dist) * 5``, plus a ``+10`` bonus (and episode
  termination) once the end effector enters the goal's tolerance sphere.

Differences from the reference environment (required to run thousands of
robots in parallel inside Isaac Lab, instead of a single physical/simulated
one commanded through serial servos):

- The goal is a purely visual marker (:class:`~isaaclab.markers.VisualizationMarkers`)
  instead of a value read back from ``detect_position()``.
- The Dofbot is spawned directly from its URDF description (bundled with
  this extension) and driven through Isaac Lab's implicit PD actuators,
  instead of ``Arm_serial_servo_write6()``.
- Joint ranges are taken from each joint's own configured limits
  (:data:`dofbot_reacher.robots.DOFBOT_DOF_LIMITS`) instead of a single
  fixed +-90deg range, since the Dofbot's joints do not all share the same
  travel.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab import cloner 
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from dofbot_reacher.robots import DOFBOT_DOF_LIMITS

from .dofbot_reacher_env_cfg import DofbotReacherEnvCfg

from .randomization import DofbotRandomizer

from datetime import datetime

logs_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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


class DofbotReacherEnv(DirectRLEnv):
    cfg: DofbotReacherEnvCfg

    def __init__(self, cfg: DofbotReacherEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        #self.LOGS_TRANSITION = f"scripts\skrl\logs_folder\logs_{logs_date}_deltaAngles" if cfg.use_delta_actions == True else f"scripts\skrl\logs_folder\logs_{logs_date}_absoluteAngles"


        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        self.num_arm_dofs = len(self.dof_idx)
        self.end_effector_idx, _ = self.robot.find_bodies(self.cfg.end_effector_body_name)
        self.end_effector_idx = self.end_effector_idx[0]


        # Other joints present in the articulation (e.g. gripper open/close
        # joints bundled inside the "link5" sub-assembly) that are not part of
        # the action space, but still need an explicit target every step so
        # they don't drift/fall under gravity during simulation.
        if len(self.cfg.gripper_joint_names) > 0:
            self.gripper_dof_idx, _ = self.robot.find_joints(self.cfg.gripper_joint_names)
            gripper_pos = self.cfg.gripper_joint_pos
            if isinstance(gripper_pos, (int, float)):
                gripper_pos = [float(gripper_pos)] * len(self.gripper_dof_idx)
            self.gripper_dof_target = (
                torch.tensor(gripper_pos, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .repeat(self.num_envs, 1)
            )
        else:
            self.gripper_dof_idx = []
            self.gripper_dof_target = None

        


        # Task-specific joint ranges (tighter than the raw hardware limits),
        # ported from ReacherTask._dof_limits / the reference env's implicit
        # +-90deg range, generalized per-joint. Only keep the 4 actuated joints.
        task_limits = torch.tensor(DOFBOT_DOF_LIMITS, dtype=torch.float32, device=self.device)
        self.arm_dof_lower_limits = task_limits[: self.num_arm_dofs, 0]
        self.arm_dof_upper_limits = task_limits[: self.num_arm_dofs, 1]

        self.arm_dof_default_pos = torch.zeros(self.num_arm_dofs, device=self.device)

        self.prev_targets = torch.zeros((self.num_envs, self.num_arm_dofs), device=self.device)

        self.cur_targets = torch.zeros((self.num_envs, self.num_arm_dofs), device=self.device)

        self.actions = torch.zeros((self.num_envs, self.num_arm_dofs), device=self.device)

        # goal bookkeeping
        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # distance to goal on the previous step, used for the "progress" reward bonus
        self.prev_dist = torch.zeros(self.num_envs, device=self.device)
        # filled in by `_get_observations`, consumed by `_get_rewards`/`_get_dones`
        self._delta_pose = torch.zeros((self.num_envs, 3), device=self.device)
        self._current_dist = torch.zeros(self.num_envs, device=self.device)
        self._goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.randomizer = DofbotRandomizer(self,cfg)


    # ------------------------------------------------------------------
    # scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        src, dest = "/World/envs/env_0","/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device= self.scene.device)[0]
        plan = cloner.ClonePlan.from_env_0(src, dest, self.scene.num_envs, self.scene.device, pos)
        cloner.replicate(plan, stage=self.scene.stage)
        
        
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # ligne manquante : à l'origine du non mouvement du robot
        self.actions = actions.clone().clamp(-1.0,1.0).to(self.device)
        if self.cfg.use_delta_actions:
            # Phase 2: the policy outputs a *delta* applied to the previous
            # target, not an absolute angle. `delta_scale_rad` bounds how much
            # a single step can move the joint, which keeps the accumulated
            # target from drifting/oscillating. Credit assignment here is
            # harder since a reward at t may depend on deltas chosen several
            # steps earlier.
            delta = self.actions * self.cfg.delta_scale_rad
            #current_joint_pos = self.robot.data.joint_pos[:, self.dof_idx]
            self.cur_targets = self.prev_targets + delta

        else :
            target = self._unit_to_range(self.actions, self.arm_dof_lower_limits, self.arm_dof_upper_limits)
            self.cur_targets = (
            self.cfg.actions_moving_average * target + (1.0 - self.cfg.actions_moving_average) * self.prev_targets
        )

        #self.cur_targets[:, 0] = 1.0  #torch.rand_like(actions)*2 - 1.0

        self.cur_targets = torch.clamp(self.cur_targets, self.arm_dof_lower_limits, self.arm_dof_upper_limits)
        #self.logging_transition(self.LOGS_TRANSITION, self.cur_targets,"ACTIONS PROPOSEES")
        self.prev_targets = self.cur_targets.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.cur_targets, joint_ids=self.dof_idx)
        #self.logging_transition(self.LOGS_TRANSITION, self.robot.data.joint_pos[:, self.dof_idx], "ACTIONS A APPLIQUER")
        # Keep any non-actuated gripper joints locked at their fixed target
        # every physics substep (they are not driven by the policy).
        if len(self.gripper_dof_idx) > 0:
            self.robot.set_joint_position_target(self.gripper_dof_target, joint_ids=self.gripper_dof_idx)

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        arm_dof_pos = self.robot.data.joint_pos[:, self.dof_idx]
        angles_scaled = self._range_to_unit(arm_dof_pos, self.arm_dof_lower_limits, self.arm_dof_upper_limits)

        end_effector_pos = self._get_end_effector_pos()
        delta_pose = self.goal_pos - end_effector_pos  # (x_tgt - x_cur, y_tgt - y_cur, z_tgt - z_cur)

        self._delta_pose = delta_pose
        self._current_dist = torch.norm(delta_pose, p=2, dim=-1)

        self._visualize_markers(end_effector_pos)

        obs = torch.cat((delta_pose, angles_scaled), dim=-1)
        obs_non_scaled = torch.cat((delta_pose, arm_dof_pos), dim=-1)

        #self.logging_transition(self.LOGS_TRANSITION, obs_non_scaled, "OBSERVATION EN RAD NON NOISEES")

        obs = self.randomizer.apply_observation_noise(obs)

        obs_non_scaled[:, 3:] = self._unit_to_range(obs[: , 3:], self.arm_dof_lower_limits, self.arm_dof_upper_limits)
        #self.logging_transition(self.LOGS_TRANSITION, obs_non_scaled, "OBSERVATION EN RAD NOISEES")


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
        # recording for logs
        #self.logging_transition(self.LOGS_TRANSITION,reward, "REWARDS")
        self.prev_dist = self._current_dist.clone()
        
        #custom metrics to simplify understanding agent's training progression
        
        self.extras["episode"] = {
            "goal_reached" : self._goal_reached.float().mean(),
            "mean_distance_to_goal" : self._current_dist.mean()}
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # "zone de prise atteinte" -> episode terminates successfully, exactly
        # like `reward_function()` returning `terminated=True` in the reference env.
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

        dof_pos, dof_vel = self.randomizer.randomize_dof_on_reset(env_ids)


        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:, self.dof_idx] = dof_pos
        joint_vel[:, self.dof_idx] = dof_vel

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(dof_pos, joint_ids=self.dof_idx, env_ids=env_ids)
        if len(self.gripper_dof_idx) > 0:
            self.robot.set_joint_position_target(
                self.gripper_dof_target[env_ids], joint_ids=self.gripper_dof_idx, env_ids=env_ids
            )

        # --- reset target ("pos_tgt") to a new reachable position ---
        self._reset_target_pose(env_ids)

        # --- (re)initialize the progress-reward baseline distance ---
        end_effector_pos = self._get_end_effector_pos()
        self.prev_dist[env_ids] = torch.norm(self.goal_pos[env_ids] - end_effector_pos[env_ids], p=2, dim=-1)

    


    def _reset_target_pose(self, env_ids: torch.Tensor) -> None:
        """Sample a new goal position inside the arm's reachable workspace."""
        num_resets = len(env_ids)
        new_pos = math_utils.sample_uniform(-1.0, 1.0, (num_resets, 3), device=self.device)
        new_pos[:, 0] = new_pos[:, 0] * 0.05 + 0.15 * torch.sign(new_pos[:, 0])
        new_pos[:, 1] = torch.abs(new_pos[:, 1] * 0.1) + 0.1
        new_pos[:, 2] = torch.abs(new_pos[:, 2] * 0.2) + 0.1     #torch.abs(new_pos[:, 2] * 0.20) + 0.10 
        self.goal_pos[env_ids] = new_pos

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _range_to_unit(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """Map values from [lower, upper] to [-1, 1] (i.e. ``normaliser``)."""
        return (2.0 * x - upper - lower) / (upper - lower)

    @staticmethod
    def _unit_to_range(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """Map values from [-1, 1] to [lower, upper] (i.e. ``denormaliser``)."""
        return 0.5 * (x + 1.0) * (upper - lower) + lower
    @staticmethod
    def logging_transition(file_name, infos: torch.Tensor,infos_title ):
        infos_for_debug = infos.clone()
        with open(file_name, "a") as logs:
            dim_infos = infos_for_debug.dim()
            #if dim_infos != 1 :     
            #    if len(infos[0,:]) == 7 : # observations or actions
            #        infos_for_debug[: , 3:] = torch.rad2deg(infos_for_debug[: , 3:])
            #    if len(infos[0,:]) == 4:
            #       infos_for_debug = torch.rad2deg(infos_for_debug)
            logs.write(str(infos_title) + "\n")
            logs.write(str(infos_for_debug) + "\n")

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
    """Direct, vectorized port of ``ReachDofbotEnvGym.reward_function``.

    Reference implementation::

        reward = -dist_tgt_cur
        bonus = (prev_dist - dist_tgt_cur) * 5
        reward += bonus
        if dist_tgt_cur < eps:
            reward += 10
            return reward, True
        else:
            return reward, False
    """
    reward = -dist
    progress_bonus = (prev_dist - dist) * progress_reward_scale
    reward = reward + progress_bonus

    if action_penalty_scale != 0.0:
        reward = reward - action_penalty_scale * torch.sum(actions**2, dim=-1)

    goal_reached = dist < success_tolerance
    reward = torch.where(goal_reached, reward + reach_goal_bonus, reward)

    return reward, goal_reached


