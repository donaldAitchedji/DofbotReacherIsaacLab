# Directly inspects the DofbotReacherEnv's internal goal_pos, end-effector
# position, and computed distance right after reset - to find exactly where
# a ~1m discrepancy is coming from, without relying on the viewport or
# training curves.
#
# Usage:
#   python inspect_reward_pipeline.py --num_envs 4

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import dofbot_reacher.tasks  # noqa: F401

env_cfg = parse_env_cfg("Isaac-Dofbot-Reacher-Direct-v0", device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make("Isaac-Dofbot-Reacher-Direct-v0", cfg=env_cfg)
obs, info = env.reset()

base_env = env.unwrapped

print("=" * 70)
print(f"num_envs = {base_env.num_envs}")
print(f"scene.env_origins:\n{base_env.scene.env_origins}")
print("-" * 70)

end_effector_pos = base_env._get_end_effector_pos()  # local (already minus env_origins)
ee_world = base_env.robot.data.body_pos_w[:, base_env.end_effector_idx]

print(f"goal_pos (local, stored by _reset_target_pose):\n{base_env.goal_pos}")
print(f"end_effector_pos (local, env_origins subtracted):\n{end_effector_pos}")
print(f"end_effector_pos (RAW world, body_pos_w, no subtraction):\n{ee_world}")
print("-" * 70)

delta_local = base_env.goal_pos - end_effector_pos
dist_local = torch.norm(delta_local, dim=-1)
print(f"delta_pose (goal_pos - end_effector_pos, LOCAL frame):\n{delta_local}")
print(f"distance (norm of the above, this is what the reward/training curve uses):\n{dist_local}")
print("-" * 70)

# Sanity check: what if env_origins were NOT subtracted at all (a common bug)?
delta_if_no_subtraction = base_env.goal_pos - ee_world
dist_if_no_subtraction = torch.norm(delta_if_no_subtraction, dim=-1)
print(f"[SANITY CHECK] distance if env_origins were mistakenly NOT subtracted:\n{dist_if_no_subtraction}")
print("=" * 70)

# also print what _get_observations() itself actually returns/stores,
# to make sure there's no discrepancy between this script's manual
# recomputation and the real training-time value
obs_dict = base_env._get_observations()
print(f"obs['policy'] (delta_pose[0:3] | angles[3:]):\n{obs_dict['policy']}")
print(f"self._current_dist (the exact tensor used by _get_rewards):\n{base_env._current_dist}")
print("=" * 70)

env.close()
simulation_app.close()
