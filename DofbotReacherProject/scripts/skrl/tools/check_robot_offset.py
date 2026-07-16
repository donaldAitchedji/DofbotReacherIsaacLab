# Checks whether the articulation root (= scene.env_origins reference point)
# is actually co-located with base_link, or offset from it.
#
# Usage:
#   python check_robot_offset.py

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from dofbot_reacher.robots import DOFBOT_CONFIG, DOFBOT_END_EFFECTOR_LINK

sim_cfg = sim_utils.SimulationCfg(dt=1 / 120)
sim = SimulationContext(sim_cfg)
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

robot = Articulation(DOFBOT_CONFIG.replace(prim_path="/World/Robot"))
sim.reset()

base_idx, _ = robot.find_bodies("base_link")
base_idx = base_idx[0]
ee_idx, _ = robot.find_bodies(DOFBOT_END_EFFECTOR_LINK)
ee_idx = ee_idx[0]

root_pos = robot.data.root_pos_w[0]
base_pos = robot.data.body_pos_w[0, base_idx]
ee_pos = robot.data.body_pos_w[0, ee_idx]

print("=" * 60)
print(f"Articulation root world position (== scene.env_origins for this env): {root_pos.tolist()}")
print(f"base_link world position:                                            {base_pos.tolist()}")
print(f"gripper_link world position:                                         {ee_pos.tolist()}")
print("-" * 60)
offset = base_pos - root_pos
print(f"root -> base_link offset: {offset.tolist()}  (norm={offset.norm().item():.4f} m)")
print("=" * 60)
if offset.norm().item() > 0.01:
    print("[FOUND IT] base_link is NOT at the articulation root's local origin.")
    print("This offset is silently added to every 'delta_pose' and every goal")
    print("marker position computed as (local_value + scene.env_origins), which")
    print("explains goals/markers appearing shifted away from the visible robot.")
    print("Fix: either re-parent/zero-out the transform above base_link in your")
    print("USD so the articulation root and base_link coincide, or subtract this")
    print("offset explicitly in dofbot_reacher_env.py wherever env_origins is used.")
else:
    print("root and base_link coincide -> this is NOT the source of the offset.")

simulation_app.close()
