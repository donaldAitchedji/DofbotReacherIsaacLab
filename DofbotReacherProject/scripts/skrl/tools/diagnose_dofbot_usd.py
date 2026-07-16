# Quick diagnostic: run this to see EXACTLY what Isaac Lab's physics parser
# sees on your dofbot.usd (joints AND bodies), so you can compare against
# what you see in the Isaac Sim Stage tree.
#
# Usage:
#   python diagnose_dofbot_usd.py

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# --- everything below needs the simulation app to be started first ---

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from dofbot_reacher.robots import DOFBOT_CONFIG

# A SimulationContext (the physics scene) MUST exist before any Articulation
# can be instantiated -- this is what was missing before and caused
# "AttributeError: 'NoneType' object has no attribute 'physics_manager'".
sim_cfg = sim_utils.SimulationCfg(dt=1 / 120)
sim = SimulationContext(sim_cfg)

# ground plane (not strictly required, but avoids an "empty stage" warning)
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

# spawn the robot
robot = Articulation(DOFBOT_CONFIG.replace(prim_path="/World/Robot"))

# IMPORTANT: reset the simulation once so the articulation is actually
# parsed/played by PhysX -- joint_names / body_names are only populated
# correctly after this.
sim.reset()

print("=" * 60)
print("Joints seen by the physics articulation:")
print(robot.joint_names)
print("Bodies (rigid links) seen by the physics articulation:")
print(robot.data.body_names)
print("=" * 60)
print("If 'gripper_link' is missing from the bodies list above but you can")
print("see it in the Isaac Sim Stage tree, it means the prim either lacks a")
print("RigidBodyAPI, or its connecting Fixed Joint's body0/body1 targets are")
print("not correctly set.")

simulation_app.close()
