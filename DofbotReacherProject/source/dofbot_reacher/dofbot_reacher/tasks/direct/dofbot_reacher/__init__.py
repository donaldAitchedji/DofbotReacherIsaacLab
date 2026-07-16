# Copyright (c) 2022-2023, Johnson Sun
# Copyright (c) 2025, Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym


##
# Register Gym environments.
##

gym.register(
    id="Isaac-Dofbot-Reacher-Direct-v0",
    entry_point=f"{__name__}.dofbot_reacher_env:DofbotReacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dofbot_reacher_env_cfg:DofbotReacherEnvCfg",
    },
)
