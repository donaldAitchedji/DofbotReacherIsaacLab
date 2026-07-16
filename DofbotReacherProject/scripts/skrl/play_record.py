"""Play back a trained checkpoint on the Dofbot reacher task, phase 1
(absolute joint angles). Loads the same env config and model architecture
used during training so the checkpoint's weights match exactly.

Usage:
    ./isaaclab.sh -p play.py --algorithm a2c --checkpoint path/to/agent.pt
    ./isaaclab.sh -p play.py --algorithm sac --checkpoint path/to/agent.pt

    # with video recording (saved under ./videos/play_<algorithm>/), still viewable live
    ./isaaclab.sh -p play.py --algorithm ppo --checkpoint path/to/agent.pt --video
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained agent on the Dofbot reacher task.")
parser.add_argument("--algorithm", type=str, default="a2c", choices=["a2c", "sac", "ppo"], help="Algorithm used to train the checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained checkpoint (.pt).")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments to visualize.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the rollout.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (steps).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

from datetime import datetime

import gymnasium as gym
import yaml

from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import dofbot_reacher.tasks  # noqa: F401
from dofbot_reacher.tasks.direct.dofbot_reacher.dofbot_reacher_env_cfg import DofbotReacherEnvCfg

from skrl_models import build_a2c_models, build_ppo_models, build_sac_models

from skrl.agents.torch.a2c import A2C, A2C_CFG
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.agents.torch.sac import SAC, SAC_CFG

import torch 
TASK_NAME = "Isaac-Dofbot-Reacher-Direct-v0"
AGENT_CFG_PATH = os.path.join(os.path.dirname(__file__), "agents", f"skrl_{args_cli.algorithm}_cfg.yaml")


def main():
    with open(AGENT_CFG_PATH) as f:
        agent_cfg = yaml.safe_load(f)
    set_seed(agent_cfg.get("seed", 42))

    env_cfg = DofbotReacherEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.use_delta_actions = False
    env_cfg.actions_moving_average = 1.0

    env = gym.make(TASK_NAME, cfg=env_cfg, render_mode="rgb_array")
    
    experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    if args_cli.video:
        video_folder = os.path.join(os.path.dirname(__file__), "videos", f"play_{args_cli.algorithm}",experiment_name)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,  # one recording covering the whole rollout
            "video_length": args_cli.video_length,
        }
        print(f"[INFO] Recording video to: {video_folder}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env)
    device = env.device

    if args_cli.algorithm == "a2c":

        models = build_a2c_models(env.observation_space, env.action_space, device)
        cfg = A2C_CFG()
        agent = A2C(
            models=models,
            memory=None,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
    elif args_cli.algorithm == "ppo":

        models = build_ppo_models(env.observation_space, env.action_space, device)
        cfg = PPO_CFG()
        agent = PPO(
            models=models,
            memory=None,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
    else:

        models = build_sac_models(env.observation_space, env.action_space, device)
        cfg = SAC_CFG()
        agent = SAC(
            models=models,
            memory=None,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )

    agent.load(args_cli.checkpoint)
    agent.enable_models_training_mode(False)   #set_running_mode("eval")

    # deterministic rollout, rendered, no training updates
    trainer = SequentialTrainer(cfg={"timesteps": 5000, "headless": False}, env=env, agents=agent)
    trainer.eval()
    torch.save(agent.models["policy"].net.state_dict(),f"scripts/skrl/models/policy_best_{args_cli.algorithm}_{experiment_name}_train.pt")


    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
