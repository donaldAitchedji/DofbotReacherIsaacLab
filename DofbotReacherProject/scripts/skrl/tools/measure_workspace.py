# Measures the ACTUAL reachable workspace of gripper_link by sweeping random
# joint configurations within DOFBOT_DOF_LIMITS, and reports min/max/mean
# end-effector position per axis (relative to base_link). Use this to
# recalibrate the goal-sampling bounds in dofbot_reacher_env.py's
# `_reset_target_pose()` instead of guessing.
#
# Usage:
#   python measure_workspace.py --num_samples 500

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from dofbot_reacher.robots import DOFBOT_CONFIG, DOFBOT_DOF_LIMITS, DOFBOT_JOINT_NAMES, DOFBOT_END_EFFECTOR_LINK

sim_cfg = sim_utils.SimulationCfg(dt=1 / 120)
sim = SimulationContext(sim_cfg)
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

robot = Articulation(DOFBOT_CONFIG.replace(prim_path="/World/Robot"))
sim.reset()

device = sim.device
dof_idx, _ = robot.find_joints(DOFBOT_JOINT_NAMES)
ee_idx, _ = robot.find_bodies(DOFBOT_END_EFFECTOR_LINK)
ee_idx = ee_idx[0]

limits = torch.tensor(DOFBOT_DOF_LIMITS, dtype=torch.float32, device=device)
lower, upper = limits[:, 0], limits[:, 1]

positions = []
n = args_cli.num_samples
for i in range(n):
    rand = torch.rand(1, len(dof_idx), device=device)
    dof_pos = lower + rand * (upper - lower)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, dof_idx] = dof_pos
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(dof_pos, joint_ids=dof_idx)

    # let the arm actually settle into this pose before reading the pose back
    for _ in range(5):
        sim.step()
        robot.update(sim.get_physics_dt())

    ee_pos = robot.data.body_pos_w[0, ee_idx] - robot.data.root_pos_w[0]
    positions.append(ee_pos.cpu().clone())

positions = torch.stack(positions)
print("=" * 60)
print(f"Sampled {n} random joint configurations within DOFBOT_DOF_LIMITS.")
print(f"gripper_link position relative to base_link (meters):")
for i, axis in enumerate(["x", "y", "z"]):
    col = positions[:, i]
    print(f"  {axis}: min={col.min().item():+.4f}  max={col.max().item():+.4f}  mean={col.mean().item():+.4f}")
print("=" * 60)
print("Use these ranges to set the goal-sampling bounds in")
print("dofbot_reacher_env.py::_reset_target_pose(), instead of the hardcoded")
print("0.15 / 0.05 / 0.2 values (which assume the ORIGINAL Dofbot's dimensions).")

simulation_app.close()
