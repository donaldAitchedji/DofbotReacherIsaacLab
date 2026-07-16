# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Isaac Lab extension: train a Yahboom Dofbot (5-DoF) manipulator to reach
random 3D target positions with reinforcement learning.

This is a port of the OmniIsaacGymEnvs ``DofbotReacher`` task
(https://github.com/J3soon/OmniIsaacGymEnvs-DofbotReacher) to the Isaac Lab
Direct RL workflow.
"""

# Register Gym environments.
from .tasks import *  # noqa: F401, F403
