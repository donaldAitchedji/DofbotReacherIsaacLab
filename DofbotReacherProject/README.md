# Dofbot Reacher (Isaac Lab)

An [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) **Direct RL** extension that trains a
Yahboom **Dofbot** (5-DoF arm) to reach random 3D target positions with its end effector.

This project is a port of the [`OmniIsaacGymEnvs-DofbotReacher`](https://github.com/J3soon/OmniIsaacGymEnvs-DofbotReacher)
task (built for the now-deprecated OmniIsaacGymEnvs framework) to the modern Isaac Lab
Direct workflow, following the structure of
[`IsaacLabTutorial`](https://github.com/isaac-sim/IsaacLabTutorial). **Only the Dofbot
robot/task is included** -- every other robot, task, or asset from the original
OmniIsaacGymEnvs repository has been dropped.

The action space, observation space, and reward function follow a lightweight
`ReachDofbotEnvGym` gymnasium reference implementation used for sim2real experiments on
the physical robot (see [Task design](#task-design) below), rather than the more elaborate
quaternion-based reward used by the original OmniIsaacGymEnvs task.

## Task design

- **Robot**: Dofbot 5-DoF arm, spawned directly from its bundled URDF description
  (`source/dofbot_reacher/dofbot_reacher/assets/dofbot/`) -- no Nucleus-hosted USD asset
  required.
- **Action space** -- `Box(-1, 1, shape=(4,))`: one normalized value per actuated joint
  (`joint1`..`joint4`, i.e. `theta`/`alpha`/`beta`/`gamma`: base yaw, shoulder, elbow,
  wrist pitch), denormalized to that joint's own angular range and sent as a position
  target. The 5th joint (`joint5`, wrist roll) only changes the end effector's
  *orientation*, not its position, so it is **not** part of the action space -- it is held
  fixed, exactly like the reference environment's `forward_k()`, which only takes 4 angles.
- **Observation space** -- `Box(-inf, inf, shape=(7,))`: `delta_pose` (goal position minus
  end-effector position, 3 values) concatenated with the 4 actuated joints' angles
  normalized to `[-1, 1]`.
- **Reward** -- a direct, vectorized port of `ReachDofbotEnvGym.reward_function()`:
  `reward = -distance + (prev_distance - distance) * 5`, plus a `+10` bonus (and episode
  termination) once the end effector enters the goal's tolerance sphere
  (`success_tolerance`, default `0.02` m).

See `dofbot_reacher_env.py` and `dofbot_reacher_env_cfg.py` for the full implementation
and inline comments mapping each piece back to the reference Gym environment.

## Installation

Requires a working [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
installation (Isaac Sim 4.5+).

```bash
# from the root of this repository, with Isaac Lab's Python environment active
python -m pip install -e source/dofbot_reacher
```

## Usage

List the registered environment:

```bash
python scripts/list_envs.py
```

Run a random-action sanity check:

```bash
python scripts/random_agent.py --task Isaac-Dofbot-Reacher-Direct-v0 --num_envs 32
```

Train with [skrl](https://skrl.readthedocs.io) (PPO):

```bash
python scripts/skrl/train.py --task Isaac-Dofbot-Reacher-Direct-v0 --num_envs 1024 --headless
```

Play back a trained checkpoint:

```bash
python scripts/skrl/play.py --task Isaac-Dofbot-Reacher-Direct-v0 --num_envs 32 \
    --checkpoint logs/skrl/dofbot_reacher_direct/<run>/checkpoints/<checkpoint>.pt
```

## Project structure

```
DofbotReacher/
├── scripts/
│   ├── list_envs.py            # list registered Gym environments
│   ├── random_agent.py         # random-action sanity check
│   ├── zero_agent.py           # zero-action sanity check
│   └── skrl/
│       ├── train.py            # skrl PPO training entry point
│       └── play.py             # play back a trained checkpoint
└── source/dofbot_reacher/
    ├── config/extension.toml   # Isaac Sim extension metadata
    ├── setup.py / pyproject.toml
    └── dofbot_reacher/
        ├── assets/dofbot/      # bundled URDF + STL meshes
        ├── robots/dofbot.py    # ArticulationCfg + joint limits
        └── tasks/direct/dofbot_reacher/
            ├── dofbot_reacher_env.py       # environment logic (MDP)
            ├── dofbot_reacher_env_cfg.py   # environment configuration
            └── agents/skrl_ppo_cfg.yaml    # skrl PPO hyperparameters
```

## Credits

- Dofbot URDF/meshes and original task design: [J3soon/OmniIsaacGymEnvs-DofbotReacher](https://github.com/J3soon/OmniIsaacGymEnvs-DofbotReacher)
  (itself building on [NVIDIA-Omniverse/OmniIsaacGymEnvs](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs)).
- Extension/project structure: [isaac-sim/IsaacLabTutorial](https://github.com/isaac-sim/IsaacLabTutorial).
