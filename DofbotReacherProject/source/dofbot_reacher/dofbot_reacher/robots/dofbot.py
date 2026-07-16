# Copyright (c) 2022-2023, Johnson Sun
# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Yahboom Dofbot manipulator.

The robot is spawned from a **USD** asset (``DOFBOT_USD_PATH``), authored
directly in Isaac Sim (not converted from the bundled URDF). In this asset:

- Only 4 revolute joints drive the arm itself: ``joint1``..``joint4`` (base
  yaw, shoulder, elbow, wrist pitch). There is no separate wrist-roll joint.
- ``link5`` is not a single link but an entire gripper sub-assembly (its own
  links/joints, e.g. to open/close the fingers). Those extra joints are not
  actuated by the Reacher task -- list their exact names in
  ``DOFBOT_GRIPPER_JOINT_NAMES`` below so the environment can hold them at a
  fixed position every step (see ``dofbot_reacher_env_cfg.py::gripper_joint_names``).
- A virtual, geometry-less link named ``gripper_link`` is rigidly attached
  somewhere inside that gripper assembly through a fixed joint, and used as
  the end-effector ("tool tip") frame tracked by the task.

If you generate the USD yourself from the bundled URDF instead (see
``scripts/tools/convert_dofbot_urdf.py``), make sure the joint/link names in
the result match ``DOFBOT_JOINT_NAMES`` / ``DOFBOT_GRIPPER_JOINT_NAMES`` /
``DOFBOT_END_EFFECTOR_LINK`` below (update them here, and in
``dofbot_reacher_env_cfg.py``, if your names differ).

Reference URDF: https://github.com/J3soon/OmniIsaacGymEnvs-DofbotReacher
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_DOFBOT_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "dofbot")

# Bundled URDF (arm only, no gripper assembly). Not used at runtime; kept only
# as an optional source for scripts/tools/convert_dofbot_urdf.py.
DOFBOT_URDF_PATH = os.path.join(_DOFBOT_ASSETS_DIR, "urdf", "dofbot.urdf")

# USD asset actually spawned by DOFBOT_CONFIG. Point this at your own
# dofbot.usd (the one containing the gripper assembly + the gripper_link
# end-effector frame), local file path or Nucleus URL.
DOFBOT_USD_PATH = os.path.join(_DOFBOT_ASSETS_DIR, "usd", "dofbot.usd")

# The 4 actuated arm joints (base yaw -> wrist pitch). This is the RL action space.
DOFBOT_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]

# Any other joint present in the USD's articulation (e.g. the gripper's own
# open/close joints, bundled inside the "link5" sub-assembly). These are NOT
# actuated by the policy but still need to be driven to a fixed target every
# step so they don't drift/fall under gravity and destabilize the sim.
#
# TODO: fill this in with the exact joint names from your USD. You can list
# every joint of the loaded articulation with:
#   python - <<'PY'
#   from isaaclab.app import AppLauncher
#   app = AppLauncher(headless=True).app
#   from isaaclab.assets import Articulation
#   from dofbot_reacher.robots import DOFBOT_CONFIG
#   robot = Articulation(DOFBOT_CONFIG.replace(prim_path="/Robot"))
#   print(robot.joint_names)
#   PY
DOFBOT_GRIPPER_JOINT_NAMES: list = ["Wrist_Twist_RevoluteJoint", "Finger_Left_01_RevoluteJoint", "Finger_Right_01_RevoluteJoint", "Finger_Left_02_RevoluteJoint", "Finger_Right_02_RevoluteJoint", "Finger_Left_03_RevoluteJoint", "Finger_Right_03_RevoluteJoint"]

# Name of the body used as the "end-effector" frame for the reaching task.
# This is a *virtual* link with no geometry/collision, rigidly attached
# somewhere inside the link5/gripper assembly through a fixed joint (added
# directly in the USD asset) purely so its world pose can be queried as a
# clean "tool tip" frame.
DOFBOT_END_EFFECTOR_LINK = "gripper_link"

_ACTUATORS = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["joint[1-4]"],
        effort_limit= 60, #30.0,
        velocity_limit=3.1416,#1.0,
        stiffness=4000,
        damping=200,
    ),
}
if DOFBOT_GRIPPER_JOINT_NAMES:
    # Stiff position-hold actuator for the gripper's own joints, so they stay
    # wherever `gripper_joint_pos` puts them every step instead of
    # sagging/oscillating under gravity. Only added when
    # DOFBOT_GRIPPER_JOINT_NAMES is non-empty: an ImplicitActuatorCfg whose
    # regex matches zero joints in the USD would fail at parse time.
    _ACTUATORS["gripper"] = ImplicitActuatorCfg(
        joint_names_expr=DOFBOT_GRIPPER_JOINT_NAMES,
        effort_limit=5.0,
        velocity_limit= 1.0,
        stiffness=200.0,
        damping=10.0,
    )

DOFBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DOFBOT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # NOTE: if your gripper joints (DOFBOT_GRIPPER_JOINT_NAMES) need a
        # non-zero rest position, add them here too, e.g.
        # {**{n: 0.0 for n in DOFBOT_JOINT_NAMES}, "gripper_left_joint": 0.02}
        joint_pos={joint_name: 0.0 for joint_name in DOFBOT_JOINT_NAMES},
    ),
    actuators=_ACTUATORS,
)
"""Configuration for the Yahboom Dofbot arm, spawned from a hand-authored USD asset."""

# Per-joint action range used by the Reacher task. This intentionally clamps
# joint2/3/4 tighter than their hardware limits (+-pi/2) to keep the arm within
# a comfortably reachable, self-collision-free workspace, matching the
# original OmniIsaacGymEnvs DofbotReacher task design.
DOFBOT_DOF_LIMITS = [
    (-math.pi / 2, math.pi / 2),  # joint1: base yaw
    (-math.pi / 4, math.pi / 4),  # joint2: shoulder
    (-math.pi / 4, math.pi / 4),  # joint3: elbow
    (-math.pi / 4, math.pi / 4),  # joint4: wrist pitch
]
