# randomization.py

from __future__ import annotations
import isaaclab.utils.math as math_utils


import torch


class DofbotRandomizer:
    def __init__(self, env, env_cfg):
        self.env = env
        self.device = env.device
        self.cfg = env_cfg

        # paramètres (tu peux ajuster)
        self.mass_scale_range = (0.8, 1.2)
        self.friction_range = (0.5, 1.5)
        self.joint_noise_std = 0.03 #0.04

    # ------------------------------------------------------------------ #
    # RESET RANDOMIZATION
    # ------------------------------------------------------------------ #

    def randomize_on_reset(self, env_ids: torch.Tensor):
        self._randomize_joint_positions(env_ids)
        self._randomize_masses(env_ids)
        self._randomize_friction(env_ids)


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


    # ------------------------------------------------------------------ #
    # JOINTS
    # ------------------------------------------------------------------ #

    def _randomize_joint_positions(self, env_ids):
        # bruit autour de 0
        noise = torch.randn((len(env_ids), 4), device=self.device) * 0.2

        # clamp dans [-pi/2, pi/2]
        noise = torch.clamp(noise, -torch.pi / 2, torch.pi / 2)

        self.env.robot.set_joint_position_target(
            noise,
            joint_ids=self.env.joint_ids,
            env_ids=env_ids,
        )

    # ------------------------------------------------------------------ #
    # MASSES
    # ------------------------------------------------------------------ #

    def _randomize_masses(self, env_ids):
        robot = self.env.robot

        # masses actuelles
        masses = robot.data.body_mass.clone()

        scale = torch.rand((len(env_ids), 1), device=self.device)
        scale = scale * (self.mass_scale_range[1] - self.mass_scale_range[0]) + self.mass_scale_range[0]

        masses[env_ids] *= scale

        robot.set_body_mass(masses, env_ids=env_ids)

    # ------------------------------------------------------------------ #
    # FRICTION
    # ------------------------------------------------------------------ #

    def _randomize_friction(self, env_ids):
        robot = self.env.robot

        friction = torch.rand((len(env_ids), 1), device=self.device)
        friction = friction * (self.friction_range[1] - self.friction_range[0]) + self.friction_range[0]

        robot.set_material_properties(
            static_friction=friction,
            dynamic_friction=friction,
            env_ids=env_ids,
        )

    # ------------------------------------------------------------------ #
    # OBS NOISE
    # ------------------------------------------------------------------ #

    def apply_observation_noise(self, obs: torch.Tensor):
        noise = torch.randn_like(obs) * self.joint_noise_std
        return obs + noise