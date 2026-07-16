"""Train the Dofbot reacher, phase 1: the policy predicts absolute joint
angles directly (no delta actions, no EMA smoothing on the targets), so the
reward at step t can be attributed to the action at step t with no lag.

Usage:
    ./isaaclab.sh -p train.py --algorithm a2c --num_envs 4096 --headless
    ./isaaclab.sh -p train.py --algorithm ppo --num_envs 4096 --headless
    ./isaaclab.sh -p train.py --algorithm sac --num_envs 4096 --headless

    # with periodic video recordings (saved under ./videos/train_<algorithm>/)
    ./isaaclab.sh -p train.py --algorithm ppo --num_envs 4096 --headless --video
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train the Dofbot reacher task with skrl.")
parser.add_argument("--algorithm", type=str, default="a2c", choices=["a2c", "sac", "ppo"], help="RL algorithm.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments (overrides cfg).")
parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides cfg).")
parser.add_argument("--max_iterations", type=int, default=None, help="Total timesteps (overrides cfg).")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint (.pt) to resume training from.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video (steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Steps between recordings.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# off-screen rendering is required to capture frames for the video, even headless
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------
# Everything below needs the simulation app running.
# --------------------------------------------------------------------------
import os
from datetime import datetime
import torch

import gymnasium as gym
import yaml

from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab_rl.skrl import SkrlVecEnvWrapper

# NOTE: adjust this import to match your project's task-registration module.
import dofbot_reacher.tasks  # noqa: F401  (registers the gym task on import)
from dofbot_reacher.tasks.direct.dofbot_reacher.dofbot_reacher_env_cfg import DofbotReacherEnvCfg

from skrl_models import build_a2c_models, build_ppo_models, build_sac_models

# NOTE: adjust to the task id you actually registered with gym.register(...).
TASK_NAME = "Isaac-Dofbot-Reacher-Direct-v0"
AGENT_CFG_PATH = os.path.join(os.path.dirname(__file__), "agents", f"skrl_{args_cli.algorithm}_cfg.yaml")


def main():
    with open(AGENT_CFG_PATH) as f:
        agent_cfg = yaml.safe_load(f)

    seed = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 42)
    set_seed(seed)

    # --- env config: force phase-1 action parameterization ---
    env_cfg = DofbotReacherEnvCfg()
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.use_delta_actions = False
    env_cfg.actions_moving_average = 1.0

    env = gym.make(TASK_NAME, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args_cli.video:
        video_folder = os.path.join(os.path.dirname(__file__), "videos", f"train_{args_cli.algorithm}",experiment_name)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
        }
        print(f"[INFO] Recording videos to: {video_folder}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env)
    device = env.device

    

    if args_cli.algorithm == "a2c":
        from skrl.agents.torch.a2c import A2C, A2C_CFG
        from skrl.memories.torch import RandomMemory

        models = build_a2c_models(env.observation_space, env.action_space, device)
        memory = RandomMemory(memory_size=agent_cfg["agent"]["rollouts"], num_envs=env.num_envs, device=device)

        agent_kwargs = {k: v for k, v in agent_cfg["agent"].items() if k != "experiment"}
        cfg = A2C_CFG(**agent_kwargs)
        exp_cfg = agent_cfg["agent"]["experiment"]
        cfg.experiment.directory = exp_cfg["directory"]
        cfg.experiment.write_interval = exp_cfg.get("write_interval", 200)
        cfg.experiment.checkpoint_interval = exp_cfg.get("checkpoint_interval", 2000)
        cfg.experiment.experiment_name = experiment_name

        agent = A2C(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )

    elif args_cli.algorithm == "ppo":
        from skrl.agents.torch.ppo import PPO, PPO_CFG
        from skrl.memories.torch import RandomMemory

        models = build_ppo_models(env.observation_space, env.action_space, device)
        memory = RandomMemory(memory_size=agent_cfg["agent"]["rollouts"], num_envs=env.num_envs, device=device)

        agent_kwargs = {k: v for k, v in agent_cfg["agent"].items() if k != "experiment"}
        cfg = PPO_CFG(**agent_kwargs)
        exp_cfg = agent_cfg["agent"]["experiment"]
        cfg.experiment.directory = exp_cfg["directory"]
        cfg.experiment.write_interval = exp_cfg.get("write_interval", 200)
        cfg.experiment.checkpoint_interval = exp_cfg.get("checkpoint_interval", 2000)
        cfg.experiment.experiment_name = experiment_name

        agent = PPO(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )

    else:  # sac
        from skrl.agents.torch.sac import SAC, SAC_CFG
        from skrl.memories.torch import RandomMemory

        models = build_sac_models(env.observation_space, env.action_space, device)
        # off-policy replay buffer (RandomMemory doubles as one when sampled with replacement)
        memory = RandomMemory(
            memory_size=agent_cfg["memory"]["size"], num_envs=env.num_envs, device=device, replacement=True
        )

        agent_kwargs = {k: v for k, v in agent_cfg["agent"].items() if k != "experiment"}
        cfg = SAC_CFG(**agent_kwargs)
        exp_cfg = agent_cfg["agent"]["experiment"]
        cfg.experiment.directory = exp_cfg["directory"]
        cfg.experiment.write_interval = exp_cfg.get("write_interval", 200)
        cfg.experiment.checkpoint_interval = exp_cfg.get("checkpoint_interval", 5000)
        cfg.experiment.experiment_name = experiment_name

        agent = SAC(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )

    if args_cli.checkpoint is not None:
        agent.load(args_cli.checkpoint)

    timesteps = args_cli.max_iterations or agent_cfg["trainer"]["timesteps"]
    trainer = SequentialTrainer(cfg={"timesteps": timesteps, "headless": True}, env=env, agents=agent)
    trainer.train() 

    torch.save(agent.models["policy"].net.state_dict(),f"scripts/skrl/models/policy_best_{args_cli.algorithm}_{experiment_name}_train.pt")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
