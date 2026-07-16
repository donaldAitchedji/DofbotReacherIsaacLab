# Copyright (c) 2022-2023, Johnson Sun
# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Dofbot Reacher Direct RL environment.

The action/observation/reward design of this task is a direct port of the
lightweight ``ReachDofbotEnvGym`` gymnasium environment used for sim2real
experiments on the physical Yahboom Dofbot:

- Action space: one normalized value in ``[-1, 1]`` per actuated joint,
  denormalized to the joint's own angular range (equivalent to the
  ``denormaliser(action, max_val=90)`` step in the reference implementation,
  generalized to each joint's actual limits instead of a fixed +-90deg range).
  Only the first 4 joints (``theta``/``joint1``, ``alpha``/``joint2``,
  ``beta``/``joint3``, ``gamma``/``joint4``) are actuated by the policy,
  exactly like the reference environment: the wrist-roll joint
  (``joint5``) does not move the end effector's *position* (only its
  orientation), so it is held fixed at a constant angle instead of being
  part of the action space.
- Observation space: the 3D position error between the goal and the
  end effector (``delta_pose``) concatenated with the normalized joint
  angles (``angles``) of the 4 actuated joints, exactly like the reference
  ``state`` dictionary (here returned as a single flat vector, as required
  by Isaac Lab's Direct workflow).
- Reward: ``-distance`` plus a "progress" bonus proportional to how much
  the distance to the goal decreased this step, plus a one-off bonus when
  the end effector enters the goal's tolerance sphere (see
  ``dofbot_reacher_env.py::compute_rewards`` for the exact port of
  ``ReachDofbotEnvGym.reward_function``).
"""

from dofbot_reacher.robots import DOFBOT_CONFIG, DOFBOT_GRIPPER_JOINT_NAMES

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class DofbotReacherEnvCfg(DirectRLEnvCfg):
    # ------------------------------------------------------------------
    # env
    # ------------------------------------------------------------------
    decimation = 2  # 60 Hz control, matching controlFrequencyInv: 2 in the original OmniIsaacGymEnvs cfg
    episode_length_s = 10.0  # ~ the "max_steps" episode budget of the reference ReachDofbotEnvGym

    # spaces definition
    # Only 4 of the 5 arm joints are actuated by the policy (theta/alpha/beta/gamma
    # in the reference ReachDofbotEnvGym), the wrist-roll joint is fixed (see below).
    action_space = 4
    # delta_pose (3) + normalized joint angles (4), ported from ReachDofbotEnvGym's observation dict
    observation_space = 7
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = DOFBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    # actuated joints (the action space): base yaw, shoulder, elbow, wrist pitch
    dof_names = ["joint1", "joint2", "joint3", "joint4"]
    
    gripper_joint_names: list = DOFBOT_GRIPPER_JOINT_NAMES
    gripper_joint_pos = 0.0
    end_effector_body_name = "gripper_link"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=1.0, replicate_physics=True)

    # ------------------------------------------------------------------
    # control
    # ------------------------------------------------------------------
    # Exponential moving average applied to the (denormalized) joint position
    # targets. 1.0 disables smoothing (the target is applied immediately),
    # matching the reference environment which interpolates the motion itself
    # via `n_steps` sub-steps inside `step()`.
    actions_moving_average = 1.0

    use_delta_actions: bool = True # False = angles are absolute targets, True = angles are deltas from the current position
    delta_scale_rad: float = 0.05236  # amplitude of the delta actions in radians, used only if use_delta_actions is True

    # ------------------------------------------------------------------
    # reward shaping - ported from ReachDofbotEnvGym.reward_function()
    # ------------------------------------------------------------------
    success_tolerance = 0.02 #0.005       # `eps` in the reference implementation
    progress_reward_scale = 20  #5.0    # bonus = (prev_dist - dist) * 5
    reach_goal_bonus = 100.0        # bonus once the goal tolerance sphere is reached
    action_penalty_scale = 0.0     # disabled by default (not present in the reference env)

    # ------------------------------------------------------------------
    # reset randomization
    # ------------------------------------------------------------------
    reset_dof_pos_noise = 0.1 #0.2 
    reset_dof_vel_noise = 0.0

    # ------------------------------------------------------------------
    # goal randomization (reachable workspace in front of the arm)
    # ------------------------------------------------------------------
    goal_marker_scale = 0.05
